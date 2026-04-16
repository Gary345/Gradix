# src/features/corner_features.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
from src.utils.helpers import clamp, safe_float, normalize_to_0_1


@dataclass
class CornerPatchResult:
    corner: str
    score: float
    highlight_ratio: float
    roughness: float
    mean_intensity: float
    std_intensity: float




def _extract_corner_patches(
    image_bgr: np.ndarray,
    patch_ratio: float = 0.12,
    inset_ratio: float = 0.01,
) -> Dict[str, np.ndarray]:
    """
    Extrae 4 parches internos correspondientes a las esquinas de la carta rectificada.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr is empty or None.")

    h, w = image_bgr.shape[:2]
    min_dim = min(h, w)

    patch = max(16, int(min_dim * patch_ratio))
    inset = max(1, int(min_dim * inset_ratio))

    patch = min(patch, h // 3, w // 3)
    inset = min(inset, h // 10, w // 10)

    top_left = image_bgr[inset:inset + patch, inset:inset + patch]
    top_right = image_bgr[inset:inset + patch, w - inset - patch:w - inset]
    bottom_left = image_bgr[h - inset - patch:h - inset, inset:inset + patch]
    bottom_right = image_bgr[h - inset - patch:h - inset, w - inset - patch:w - inset]

    return {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
    }


def _compute_corner_metrics(gray_patch: np.ndarray, corner_name: str) -> CornerPatchResult:
    if gray_patch is None or gray_patch.size == 0:
        return CornerPatchResult(
            corner=corner_name,
            score=0.0,
            highlight_ratio=1.0,
            roughness=1.0,
            mean_intensity=0.0,
            std_intensity=0.0,
        )

    gray_patch = cv2.GaussianBlur(gray_patch, (3, 3), 0)

    mean_intensity = float(np.mean(gray_patch))
    std_intensity = float(np.std(gray_patch))

    lap_var = float(cv2.Laplacian(gray_patch, cv2.CV_64F).var())

    highlight_threshold = mean_intensity + max(10.0, std_intensity * 1.2)
    highlight_ratio = float(np.mean(gray_patch > highlight_threshold))

    # Roughness como proxy de irregularidad local
    roughness = lap_var

    highlight_penalty = normalize_to_0_1(highlight_ratio, 0.01, 0.20)
    roughness_penalty = normalize_to_0_1(roughness, 35.0, 550.0)

    # Penalización suave si la esquina está demasiado lavada o demasiado plana
    low_texture_penalty = 0.0
    if std_intensity < 8.0:
        low_texture_penalty = normalize_to_0_1(8.0 - std_intensity, 0.0, 8.0) * 0.20

    total_penalty = (
        0.35 * highlight_penalty
        + 0.45 * roughness_penalty
        + 0.20 * low_texture_penalty
    )

    score = 100.0 * (1.0 - clamp(total_penalty, 0.0, 1.0))

    return CornerPatchResult(
        corner=corner_name,
        score=safe_float(score),
        highlight_ratio=safe_float(highlight_ratio),
        roughness=safe_float(roughness),
        mean_intensity=safe_float(mean_intensity),
        std_intensity=safe_float(std_intensity)
    )


def _estimate_corner_confidence(
    image_bgr: np.ndarray,
    corner_results: Dict[str, CornerPatchResult],
) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    global_sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    global_contrast = float(np.std(gray))
    avg_patch_std = float(np.mean([r.std_intensity for r in corner_results.values()]))

    sharpness_score = normalize_to_0_1(global_sharpness, 40.0, 450.0)
    contrast_score = normalize_to_0_1(global_contrast, 20.0, 70.0)
    patch_texture_score = normalize_to_0_1(avg_patch_std, 8.0, 35.0)

    confidence = (
        0.45 * sharpness_score
        + 0.30 * contrast_score
        + 0.25 * patch_texture_score
    )

    return clamp(confidence, 0.0, 1.0)


def _corner_label_from_score(score: float) -> str:
    if score >= 85:
        return "good"
    if score >= 70:
        return "acceptable"
    if score >= 50:
        return "weak"
    return "poor"


def compute_corner_features(
    rectified_card_bgr: np.ndarray,
    patch_ratio: float = 0.12,
    inset_ratio: float = 0.01,
) -> Dict[str, float]:
    """
    Calcula features MVP de esquinas sobre la carta rectificada.
    Devuelve score global 0-100, confianza y métricas por esquina.
    """
    if rectified_card_bgr is None or rectified_card_bgr.size == 0:
        raise ValueError("rectified_card_bgr is empty or None.")

    if len(rectified_card_bgr.shape) != 3 or rectified_card_bgr.shape[2] != 3:
        raise ValueError("rectified_card_bgr must be a BGR image with 3 channels.")

    patches = _extract_corner_patches(
        image_bgr=rectified_card_bgr,
        patch_ratio=patch_ratio,
        inset_ratio=inset_ratio,
    )

    corner_results: Dict[str, CornerPatchResult] = {}

    for corner_name, patch in patches.items():
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        corner_results[corner_name] = _compute_corner_metrics(gray_patch, corner_name)

    corner_score_raw = float(np.mean([r.score for r in corner_results.values()]))
    corner_confidence = _estimate_corner_confidence(rectified_card_bgr, corner_results)

    if corner_confidence < 0.20:
        corner_score_raw *= 0.85
    elif corner_confidence < 0.35:
        corner_score_raw *= 0.92


    corner_score_raw = clamp(corner_score_raw, 0.0, 100.0)
    corner_label = _corner_label_from_score(corner_score_raw)

    features: Dict[str, float] = {
        "corner_score_raw": safe_float(corner_score_raw),
        "corner_confidence": safe_float(corner_confidence),
        "corner_label": corner_label,
        "corner_patch_ratio": safe_float(patch_ratio),
        "corner_inset_ratio": safe_float(inset_ratio)
    }

    for corner_name, result in corner_results.items():
        features[f"{corner_name}_corner_score"] = safe_float(result.score)
        features[f"{corner_name}_highlight_ratio"] = safe_float(result.highlight_ratio)
        features[f"{corner_name}_roughness"] = safe_float(result.roughness)
        features[f"{corner_name}_mean_intensity"] = safe_float(result.mean_intensity)
        features[f"{corner_name}_std_intensity"] = safe_float(result.std_intensity)


    return features


def draw_corner_patch_overlay(
    rectified_card_bgr: np.ndarray,
    patch_ratio: float = 0.12,
    inset_ratio: float = 0.01,
) -> np.ndarray:
    """
    Dibuja los 4 parches de esquina usados en el análisis.
    """
    if rectified_card_bgr is None or rectified_card_bgr.size == 0:
        raise ValueError("rectified_card_bgr is empty or None.")

    overlay = rectified_card_bgr.copy()
    h, w = overlay.shape[:2]
    min_dim = min(h, w)

    patch = max(16, int(min_dim * patch_ratio))
    inset = max(1, int(min_dim * inset_ratio))

    patch = min(patch, h // 3, w // 3)
    inset = min(inset, h // 10, w // 10)

    color = (255, 255, 255)
    thickness = 2

    cv2.rectangle(overlay, (inset, inset), (inset + patch, inset + patch), color, thickness)
    cv2.rectangle(overlay, (w - inset - patch, inset), (w - inset, inset + patch), color, thickness)
    cv2.rectangle(overlay, (inset, h - inset - patch), (inset + patch, h - inset), color, thickness)
    cv2.rectangle(
        overlay,
        (w - inset - patch, h - inset - patch),
        (w - inset, h - inset),
        color,
        thickness,
    )

    return overlay