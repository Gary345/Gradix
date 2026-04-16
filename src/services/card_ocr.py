from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import unicodedata
from typing import List

import cv2
import numpy as np

try:
    import pytesseract
    from pytesseract import Output, TesseractNotFoundError
except ImportError:  # pragma: no cover - depende del entorno
    pytesseract = None
    Output = None

    class TesseractNotFoundError(RuntimeError):
        pass


@dataclass(frozen=True)
class OCRNameResult:
    raw_text: str
    cleaned_text: str
    engine: str
    roi_bgr: np.ndarray | None = None
    variants: List[np.ndarray] | None = None


def _crop_name_regions(warped_card: np.ndarray) -> List[np.ndarray]:
    height, width = warped_card.shape[:2]
    candidate_boxes = [
        # Banda superior centrada en el nombre; evitamos la linea "Evoluciona de"
        # y reducimos el area de HP/etapa para bajar contaminacion del OCR.
        (0.16, 0.03, 0.66, 0.12),
        (0.12, 0.025, 0.74, 0.125),
        (0.10, 0.02, 0.80, 0.13),
    ]
    regions: List[np.ndarray] = []

    for x0_ratio, y0_ratio, x1_ratio, y1_ratio in candidate_boxes:
        x0 = int(width * x0_ratio)
        y0 = int(height * y0_ratio)
        x1 = int(width * x1_ratio)
        y1 = int(height * y1_ratio)
        region = warped_card[y0:y1, x0:x1]
        if region.size:
            regions.append(region)

    return regions


def _build_ocr_variants(name_roi_bgr: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(name_roi_bgr, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)

    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    inverted_otsu = cv2.bitwise_not(otsu)

    return [upscaled, otsu, adaptive, inverted_otsu]


def _resolve_tesseract_command() -> str | None:
    system_path = shutil.which("tesseract")
    if system_path:
        return system_path

    candidate_paths = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)

    return None


def _normalize_pokemon_name(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"[^A-Za-z0-9'\-\.\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .-_")

    if not normalized:
        return ""

    tokens = normalized.split()
    ignored_prefix_tokens = {
        "basico",
        "fase1",
        "fase2",
        "etapa1",
        "etapa2",
        "stage1",
        "stage2",
    }

    while tokens and tokens[0].lower().replace(" ", "") in ignored_prefix_tokens:
        tokens.pop(0)

    cleaned_tokens: List[str] = []
    for token in tokens:
        if token.lower() in {"ps", "hp"}:
            break
        if len(token) <= 1 and not token.isdigit():
            continue
        if token.isdigit():
            continue
        cleaned_tokens.append(token)

    normalized = " ".join(cleaned_tokens).strip()
    if not normalized:
        return ""

    return normalized.title()


def _extract_text_with_confidence(image: np.ndarray, config: str) -> tuple[str, float]:
    if pytesseract is None or Output is None:
        return "", -1.0

    data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)
    tokens: List[str] = []
    confidences: List[float] = []

    for raw_text, raw_confidence in zip(data.get("text", []), data.get("conf", [])):
        token = str(raw_text).strip()
        if not token:
            continue

        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = -1.0

        tokens.append(token)
        if confidence >= 0:
            confidences.append(confidence)

    joined_text = " ".join(tokens).strip()
    average_confidence = sum(confidences) / len(confidences) if confidences else -1.0
    return joined_text, average_confidence


def _score_candidate(cleaned_text: str, confidence: float) -> tuple[float, float, int, int]:
    letters = sum(char.isalpha() for char in cleaned_text)
    digits = sum(char.isdigit() for char in cleaned_text)
    compact_length = len(cleaned_text.replace(" ", ""))
    alpha_ratio = letters / max(compact_length, 1)
    return confidence, alpha_ratio, -digits, -compact_length


def extract_pokemon_name_from_warped_card(warped_card: np.ndarray) -> OCRNameResult:
    if warped_card is None or warped_card.size == 0:
        return OCRNameResult(raw_text="", cleaned_text="", engine="unavailable")

    name_rois = _crop_name_regions(warped_card)
    if not name_rois:
        return OCRNameResult(raw_text="", cleaned_text="", engine="invalid-roi")

    primary_roi = name_rois[0]
    primary_variants = _build_ocr_variants(primary_roi)

    if pytesseract is None:
        return OCRNameResult(
            raw_text="",
            cleaned_text="",
            engine="missing-pytesseract",
            roi_bgr=primary_roi,
            variants=primary_variants,
        )

    tesseract_cmd = _resolve_tesseract_command()
    if tesseract_cmd is None:
        return OCRNameResult(
            raw_text="",
            cleaned_text="",
            engine="missing-tesseract",
            roi_bgr=primary_roi,
            variants=primary_variants,
        )

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    configs = [
        "--psm 7 --oem 3 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-'",
        "--psm 8 --oem 3 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-'",
    ]
    candidates: List[tuple[str, str, float, np.ndarray, List[np.ndarray]]] = []

    try:
        for name_roi in name_rois:
            variants = _build_ocr_variants(name_roi)
            for config in configs:
                for variant in variants:
                    raw_text, confidence = _extract_text_with_confidence(variant, config)
                    cleaned_text = _normalize_pokemon_name(raw_text)
                    if cleaned_text:
                        candidates.append(
                            (raw_text.strip(), cleaned_text, confidence, name_roi, variants)
                        )
    except TesseractNotFoundError:
        return OCRNameResult(
            raw_text="",
            cleaned_text="",
            engine="missing-tesseract",
            roi_bgr=primary_roi,
            variants=primary_variants,
        )

    if not candidates:
        return OCRNameResult(
            raw_text="",
            cleaned_text="",
            engine="pytesseract",
            roi_bgr=primary_roi,
            variants=primary_variants,
        )

    best_raw, best_cleaned, best_confidence, best_roi, best_variants = max(
        candidates,
        key=lambda item: _score_candidate(item[1], item[2]),
    )
    return OCRNameResult(
        raw_text=best_raw,
        cleaned_text=best_cleaned,
        engine="pytesseract",
        roi_bgr=best_roi,
        variants=best_variants,
    )
