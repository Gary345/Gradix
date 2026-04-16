# src/features/whitening_surface_features.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import cv2
import numpy as np

from src.utils.image_utils import ensure_bgr_uint8


@dataclass
class WhiteningSurfaceResult:
    whitening_score: float
    surface_score: float
    whitening_surface_score: float

    edge_whitening_ratio: float
    corner_whitening_ratio: float
    whitening_confidence: float

    glare_ratio: float
    texture_anomaly_ratio: float
    dark_spot_ratio: float
    surface_confidence: float

    edge_band_width_px: int
    corner_patch_size_px: int
    inner_roi_shape: Tuple[int, int]

    debug: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _clip10(x: float) -> float:
    return float(max(0.0, min(10.0, x)))


def _safe_resize_min_side(img: np.ndarray, min_side: int = 600) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) >= min_side:
        return img

    scale = min_side / float(min(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _extract_inner_roi(image: np.ndarray, margin_ratio: float = 0.12) -> np.ndarray:
    h, w = image.shape[:2]
    my = int(round(h * margin_ratio))
    mx = int(round(w * margin_ratio))

    y1 = max(0, my)
    y2 = min(h, h - my)
    x1 = max(0, mx)
    x2 = min(w, w - mx)

    if y2 <= y1 or x2 <= x1:
        return image.copy()

    return image[y1:y2, x1:x2].copy()


def _edge_pair_masks(h: int, w: int, band_px: int):
    masks = {}

    outer = np.zeros((h, w), dtype=np.uint8)
    outer[:band_px, :] = 255
    inner = np.zeros((h, w), dtype=np.uint8)
    inner[band_px:2 * band_px, :] = 255
    masks["top"] = (outer, inner)

    outer = np.zeros((h, w), dtype=np.uint8)
    outer[h - band_px:, :] = 255
    inner = np.zeros((h, w), dtype=np.uint8)
    inner[h - 2 * band_px:h - band_px, :] = 255
    masks["bottom"] = (outer, inner)

    outer = np.zeros((h, w), dtype=np.uint8)
    outer[:, :band_px] = 255
    inner = np.zeros((h, w), dtype=np.uint8)
    inner[:, band_px:2 * band_px] = 255
    masks["left"] = (outer, inner)

    outer = np.zeros((h, w), dtype=np.uint8)
    outer[:, w - band_px:] = 255
    inner = np.zeros((h, w), dtype=np.uint8)
    inner[:, w - 2 * band_px:w - band_px] = 255
    masks["right"] = (outer, inner)

    return masks


def _corner_pair_masks(h: int, w: int, patch_px: int):
    masks = {}

    outer = np.zeros((h, w), dtype=np.uint8)
    outer[:patch_px, :patch_px] = 255
    inner = np.zeros((h, w), dtype=np.uint8)
    inner[patch_px:2 * patch_px, patch_px:2 * patch_px] = 255
    masks["top_left"] = (outer, inner)

    outer = np.zeros((h, w), dtype=np.uint8)
    outer[:patch_px, w - patch_px:] = 255
    inner = np.zeros((h, w), dtype=np.uint8)
    inner[patch_px:2 * patch_px, w - 2 * patch_px:w - patch_px] = 255
    masks["top_right"] = (outer, inner)

    outer = np.zeros((h, w), dtype=np.uint8)
    outer[h - patch_px:, :patch_px] = 255
    inner = np.zeros((h, w), dtype=np.uint8)
    inner[h - 2 * patch_px:h - patch_px, patch_px:2 * patch_px] = 255
    masks["bottom_left"] = (outer, inner)

    outer = np.zeros((h, w), dtype=np.uint8)
    outer[h - patch_px:, w - patch_px:] = 255
    inner = np.zeros((h, w), dtype=np.uint8)
    inner[h - 2 * patch_px:h - patch_px, w - 2 * patch_px:w - patch_px] = 255
    masks["bottom_right"] = (outer, inner)

    return masks


def _masked_values(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return channel[mask > 0]


def _ratio_of_mask(mask: np.ndarray) -> float:
    total = mask.size
    active = int(np.count_nonzero(mask))
    if total == 0:
        return 0.0
    return float(active / total)


def _build_relative_whitening_mask(
    bgr: np.ndarray,
    outer_mask: np.ndarray,
    inner_mask: np.ndarray,
) -> np.ndarray:
    """
    Whitening relativo:
    buscamos píxeles en la banda externa que sean:
    - más brillantes que la banda interior vecina
    - menos saturados que la banda interior vecina
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    l = lab[:, :, 0].astype(np.float32)

    outer_s = _masked_values(s, outer_mask)
    outer_v = _masked_values(v, outer_mask)
    outer_l = _masked_values(l, outer_mask)

    inner_s = _masked_values(s, inner_mask)
    inner_v = _masked_values(v, inner_mask)
    inner_l = _masked_values(l, inner_mask)

    if inner_s.size == 0 or outer_s.size == 0:
        return np.zeros_like(outer_mask)

    inner_s_mean = float(np.mean(inner_s))
    inner_v_mean = float(np.mean(inner_v))
    inner_l_mean = float(np.mean(inner_l))
    inner_s_std = float(np.std(inner_s))
    inner_v_std = float(np.std(inner_v))
    inner_l_std = float(np.std(inner_l))

    # Condiciones relativas
    sat_condition = s <= max(20.0, inner_s_mean - 0.35 * inner_s_std)
    val_condition = v >= min(250.0, inner_v_mean + 0.45 * inner_v_std + 8.0)
    lum_condition = l >= min(250.0, inner_l_mean + 0.45 * inner_l_std + 6.0)

    candidate = (sat_condition & val_condition & lum_condition).astype(np.uint8) * 255
    candidate = cv2.bitwise_and(candidate, outer_mask)

    kernel = np.ones((3, 3), np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)

    return candidate


def _compute_whitening_ratios(bgr: np.ndarray) -> Dict[str, float]:
    h, w = bgr.shape[:2]

    band_px = max(6, int(round(min(h, w) * 0.025)))
    patch_px = max(12, int(round(min(h, w) * 0.07)))

    edge_pairs = _edge_pair_masks(h, w, band_px)
    corner_pairs = _corner_pair_masks(h, w, patch_px)

    edge_ratios = []
    for _, (outer, inner) in edge_pairs.items():
        candidate = _build_relative_whitening_mask(bgr, outer, inner)
        denom = max(1, int(np.count_nonzero(outer)))
        edge_ratios.append(float(np.count_nonzero(candidate)) / denom)

    corner_ratios = []
    for _, (outer, inner) in corner_pairs.items():
        candidate = _build_relative_whitening_mask(bgr, outer, inner)
        denom = max(1, int(np.count_nonzero(outer)))
        corner_ratios.append(float(np.count_nonzero(candidate)) / denom)

    edge_ratio = float(np.mean(edge_ratios)) if edge_ratios else 0.0
    corner_ratio = float(np.mean(corner_ratios)) if corner_ratios else 0.0

    confidence = _clip01(
        0.55
        + 0.20 * min(1.0, band_px / 18.0)
        + 0.25 * min(1.0, patch_px / 36.0)
    )

    return {
        "edge_whitening_ratio": edge_ratio,
        "corner_whitening_ratio": corner_ratio,
        "whitening_confidence": confidence,
        "edge_band_width_px": band_px,
        "corner_patch_size_px": patch_px,
    }


def _compute_glare_ratio(inner_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    glare_mask = ((v >= 245) & (s <= 35)).astype(np.uint8) * 255
    glare_mask = cv2.medianBlur(glare_mask, 3)

    return _ratio_of_mask(glare_mask)


def _compute_texture_anomaly_ratio(inner_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    high_freq = cv2.absdiff(gray, blur)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)
    lap_abs = np.clip(lap_abs, 0, 255).astype(np.uint8)

    hf_thr = max(18.0, float(np.mean(high_freq) + 1.2 * np.std(high_freq)))
    lap_thr = max(20.0, float(np.mean(lap_abs) + 1.2 * np.std(lap_abs)))

    hf_mask = (high_freq >= hf_thr).astype(np.uint8) * 255
    lap_mask = (lap_abs >= lap_thr).astype(np.uint8) * 255

    anomaly = cv2.bitwise_and(hf_mask, lap_mask)

    kernel = np.ones((3, 3), np.uint8)
    anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN, kernel)

    return _ratio_of_mask(anomaly)


def _compute_dark_spot_ratio(inner_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(inner_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    diff = cv2.subtract(blur, gray)

    thr = max(12.0, float(np.mean(diff) + 1.5 * np.std(diff)))
    dark_spots = (diff >= thr).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    dark_spots = cv2.morphologyEx(dark_spots, cv2.MORPH_OPEN, kernel)

    return _ratio_of_mask(dark_spots)


def _score_whitening(
    edge_ratio: float,
    corner_ratio: float,
    blur_score: float = 300.0,
    contrast_score: float = 50.0,
    capture_score: float = 8.0,
    surface_score: float = 8.0,
    ) -> float:
    """
    Calcula score de whitening (0-10) ajustado por calidad real de captura.
    
    Los ratios de whitening son penalizados más si la captura es mediocre.
    """
    # Penalidades base por whitening detectado
    edge_penalty = min(1.0, edge_ratio / 0.20)  # ↑ reducido de 0.25 (más sensible)
    corner_penalty = min(1.0, corner_ratio / 0.25)  # ↑ reducido de 0.30 (más sensible)

    # Factor de confianza: si la captura es pobre, desconfiamos del whitening
    lighting_factor = 1.0

    if capture_score < 7.0:  # Captura mediocre
        lighting_factor *= 0.50  # ↑ de 0.65 (más severo)

    if blur_score < 150:  # ↑ de 220 (umbral más alto)
        lighting_factor *= 0.80

    if blur_score < 100:  # Nuevo: muy borroso
        lighting_factor *= 0.60

    if contrast_score < 30:  # ↑ de 40 (más exigente)
        lighting_factor *= 0.85

    if contrast_score < 20:  # Nuevo: muy bajo contraste
        lighting_factor *= 0.60

    # Penalidad compuesta
    penalty = lighting_factor * (
        5.0 * edge_penalty +      # ↑ de 4.0 (más peso a bordes)
        5.0 * corner_penalty      # ↑ de 4.0 (más peso a esquinas)
    )

    score = _clip10(10.0 - penalty)

    # Failsafe relajado: si la superficie está excelente, no bajar tanto
    if surface_score > 8.5 and score < 3.5:  # ↑ de 4.5 a 3.5 (menos laxo)
        score = 3.5

    return score



def _score_surface(glare_ratio: float, texture_ratio: float, dark_spot_ratio: float) -> float:
    penalty = (
        4.0 * min(1.0, glare_ratio / 0.03) +
        3.5 * min(1.0, texture_ratio / 0.08) +
        2.5 * min(1.0, dark_spot_ratio / 0.04)
    )
    return _clip10(10.0 - penalty)


def compute_whitening_surface_features(
    rectified_card: np.ndarray,
    resize_min_side: int = 600,
    blur_score: float = None,
    contrast_score: float = None,
    capture_score: float = None,
) -> Dict:
    bgr = ensure_bgr_uint8(rectified_card)
    bgr = _safe_resize_min_side(bgr, min_side=resize_min_side)

    # Gray real para métricas de captura locales
    bgr_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Si no vienen del pipeline, calcular aquí
    if blur_score is None:
        blur_score = float(cv2.Laplacian(bgr_gray, cv2.CV_32F).var())

    if contrast_score is None:
        contrast_score = float(np.std(bgr_gray))

    if capture_score is None:
        capture_score = 8.0  # fallback seguro mientras no llegue del pipeline

    h, w = bgr.shape[:2]
    if h < 120 or w < 80:
        raise ValueError(
            f"rectified_card too small for whitening/surface analysis: shape={bgr.shape}"
        )

    whitening_data = _compute_whitening_ratios(bgr)

    inner_roi = _extract_inner_roi(bgr, margin_ratio=0.12)
    glare_ratio = _compute_glare_ratio(inner_roi)
    texture_anomaly_ratio = _compute_texture_anomaly_ratio(inner_roi)
    dark_spot_ratio = _compute_dark_spot_ratio(inner_roi)

    surface_confidence = _clip01(
        0.60
        + 0.20 * min(1.0, inner_roi.shape[0] / 400.0)
        + 0.20 * min(1.0, inner_roi.shape[1] / 280.0)
    )

    # Primero surface_score real
    surface_score = _score_surface(
        glare_ratio=glare_ratio,
        texture_ratio=texture_anomaly_ratio,
        dark_spot_ratio=dark_spot_ratio,
    )

    # Luego whitening_score con parámetros reales
    whitening_score = _score_whitening(
        edge_ratio=whitening_data["edge_whitening_ratio"],
        corner_ratio=whitening_data["corner_whitening_ratio"],
        blur_score=blur_score,
        contrast_score=contrast_score,
        capture_score=capture_score,
        surface_score=surface_score,
    )

    whitening_surface_score = _clip10(
        0.55 * whitening_score + 0.45 * surface_score
    )

    result = WhiteningSurfaceResult(
        whitening_score=whitening_score,
        surface_score=surface_score,
        whitening_surface_score=whitening_surface_score,

        edge_whitening_ratio=whitening_data["edge_whitening_ratio"],
        corner_whitening_ratio=whitening_data["corner_whitening_ratio"],
        whitening_confidence=whitening_data["whitening_confidence"],

        glare_ratio=glare_ratio,
        texture_anomaly_ratio=texture_anomaly_ratio,
        dark_spot_ratio=dark_spot_ratio,
        surface_confidence=surface_confidence,

        edge_band_width_px=whitening_data["edge_band_width_px"],
        corner_patch_size_px=whitening_data["corner_patch_size_px"],
        inner_roi_shape=(int(inner_roi.shape[0]), int(inner_roi.shape[1])),

        debug={
            "image_height": float(h),
            "image_width": float(w),
            "blur_score_used": float(blur_score),
            "contrast_score_used": float(contrast_score),
            "capture_score_used": float(capture_score),
        },
    )

    return result.to_dict()
