from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
from src.utils.helpers import clamp, safe_float, normalize_to_0_1


@dataclass
class EdgeBandResult:
    side: str
    score: float
    anomaly_ratio: float
    highlight_ratio: float
    roughness: float
    mean_intensity: float
    std_intensity: float
    profile_length: int


def _moving_average_1d(signal: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    if kernel_size < 1:
        kernel_size = 1

    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    if signal.size == 0:
        return signal.copy()

    if signal.size < kernel_size:
        kernel_size = max(1, signal.size if signal.size % 2 == 1 else signal.size - 1)

    if kernel_size <= 1:
        return signal.copy()

    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(signal, kernel, mode="same")


def _extract_edge_bands(
    image_bgr: np.ndarray,
    band_ratio: float = 0.03,
    inset_ratio: float = 0.01,
) -> Dict[str, np.ndarray]:
    """
    Extrae 4 bandas internas de la carta rectificada.
    band_ratio: grosor relativo de la banda respecto a la dimensión menor.
    inset_ratio: pequeño margen interno para evitar contaminar con el borde externo exacto.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr is empty or None.")

    h, w = image_bgr.shape[:2]
    min_dim = min(h, w)

    band = max(4, int(min_dim * band_ratio))
    inset = max(1, int(min_dim * inset_ratio))

    # Asegurar que no rompamos límites
    band = min(band, h // 4, w // 4)
    inset = min(inset, h // 10, w // 10)

    top = image_bgr[inset:inset + band, inset:w - inset]
    bottom = image_bgr[h - inset - band:h - inset, inset:w - inset]
    left = image_bgr[inset:h - inset, inset:inset + band]
    right = image_bgr[inset:h - inset, w - inset - band:w - inset]

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }


def _compute_longitudinal_profile(gray_band: np.ndarray, side: str) -> np.ndarray:
    """
    Devuelve un perfil 1D a lo largo del borde:
    - top/bottom: promedio por columna
    - left/right: promedio por fila
    """
    if side in ("top", "bottom"):
        profile = gray_band.mean(axis=0)
    else:
        profile = gray_band.mean(axis=1)

    return profile.astype(np.float32)


def _compute_band_metrics(gray_band: np.ndarray, side: str) -> EdgeBandResult:
    """
    Heurística MVP:
    - anomaly_ratio: fracción del perfil con residuales altos respecto a su versión suavizada
    - highlight_ratio: fracción de pixeles excesivamente claros dentro del strip
    - roughness: variabilidad de la derivada del perfil
    """
    if gray_band is None or gray_band.size == 0:
        return EdgeBandResult(
            side=side,
            score=0.0,
            anomaly_ratio=1.0,
            highlight_ratio=1.0,
            roughness=1.0,
            mean_intensity=0.0,
            std_intensity=0.0,
            profile_length=0,
        )

    gray_band = cv2.GaussianBlur(gray_band, (3, 3), 0)

    profile = _compute_longitudinal_profile(gray_band, side)
    smooth_profile = _moving_average_1d(profile, kernel_size=11)

    residual = np.abs(profile - smooth_profile)
    residual_std = float(np.std(residual))
    anomaly_threshold = max(6.0, residual_std * 1.5)
    anomaly_ratio = float(np.mean(residual > anomaly_threshold))

    derivative = np.diff(profile)
    roughness = float(np.std(derivative)) if derivative.size > 0 else 0.0

    band_mean = float(np.mean(gray_band))
    band_std = float(np.std(gray_band))

    # Highlights anómalos: píxeles claramente más claros que el promedio local
    highlight_threshold = band_mean + max(10.0, band_std * 1.2)
    highlight_ratio = float(np.mean(gray_band > highlight_threshold))

    # Normalizaciones heurísticas MVP
    anomaly_penalty = normalize_to_0_1(anomaly_ratio, 0.02, 0.22)
    highlight_penalty = normalize_to_0_1(highlight_ratio, 0.01, 0.18)
    roughness_penalty = normalize_to_0_1(roughness, 2.5, 18.0)

    total_penalty = (
        0.45 * anomaly_penalty
        + 0.30 * roughness_penalty
        + 0.25 * highlight_penalty
    )

    score = 100.0 * (1.0 - clamp(total_penalty, 0.0, 1.0))

    return EdgeBandResult(
        side=side,
        score=safe_float(score),
        anomaly_ratio=safe_float(anomaly_ratio),
        highlight_ratio=safe_float(highlight_ratio),
        roughness=safe_float(roughness),
        mean_intensity=safe_float(band_mean),
        std_intensity=safe_float(band_std),
        profile_length=int(profile.size),
    )


def _edge_label_from_score(score: float) -> str:
    if score >= 85:
        return "good"
    if score >= 70:
        return "acceptable"
    if score >= 50:
        return "weak"
    return "poor"


def _estimate_edge_confidence(
    image_bgr: np.ndarray,
    band_results: Dict[str, EdgeBandResult],
) -> float:
    """
    Confianza heurística de lectura:
    - mejor si hay suficiente detalle/contraste global en bandas
    - baja si todo está demasiado plano o demasiado ruidoso
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    global_contrast = float(np.std(gray))

    avg_band_std = float(np.mean([r.std_intensity for r in band_results.values()]))

    sharpness_score = normalize_to_0_1(lap_var, 40.0, 300.0)
    contrast_score = normalize_to_0_1(global_contrast, 20.0, 70.0)
    band_texture_score = normalize_to_0_1(avg_band_std, 8.0, 35.0)

    confidence = (
        0.45 * sharpness_score
        + 0.30 * contrast_score
        + 0.25 * band_texture_score
    )

    return clamp(confidence, 0.0, 1.0)


def compute_edge_features(
    rectified_card_bgr: np.ndarray,
    band_ratio: float = 0.03,
    inset_ratio: float = 0.01,
) -> Dict[str, float]:
    """
    Calcula features y score preliminar de bordes sobre una carta rectificada.

    Parameters
    ----------
    rectified_card_bgr : np.ndarray
        Imagen BGR de la carta rectificada.
    band_ratio : float
        Grosor relativo de la banda usada por borde.
    inset_ratio : float
        Margen interno para evitar capturar exactamente el borde exterior.

    Returns
    -------
    Dict[str, float]
        Diccionario plano con métricas por lado y score global.
    """
    if rectified_card_bgr is None or rectified_card_bgr.size == 0:
        raise ValueError("rectified_card_bgr is empty or None.")

    if len(rectified_card_bgr.shape) != 3 or rectified_card_bgr.shape[2] != 3:
        raise ValueError("rectified_card_bgr must be a BGR image with 3 channels.")

    bands = _extract_edge_bands(
        image_bgr=rectified_card_bgr,
        band_ratio=band_ratio,
        inset_ratio=inset_ratio,
    )

    band_results: Dict[str, EdgeBandResult] = {}

    for side, band in bands.items():
        gray_band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        band_results[side] = _compute_band_metrics(gray_band, side)

    edge_score = float(np.mean([result.score for result in band_results.values()]))
    edge_label = _edge_label_from_score(edge_score)
    edge_confidence = _estimate_edge_confidence(rectified_card_bgr, band_results)

    # Penalización leve si la confianza de lectura es baja

    if edge_confidence < 0.20:
        edge_score *= 0.85
    elif edge_confidence < 0.35:
        edge_score *= 0.92

    edge_score = clamp(edge_score, 0.0, 100.0)
    edge_label = _edge_label_from_score(edge_score)

    features: Dict[str, float] = {
        "edge_score": safe_float(edge_score),
        "edge_confidence": safe_float(edge_confidence),
        "edge_label": edge_label,
        "edge_band_ratio": safe_float(band_ratio),
        "edge_inset_ratio": safe_float(inset_ratio),
    }

    for side, result in band_results.items():
        features[f"{side}_edge_score"] = safe_float(result.score)
        features[f"{side}_anomaly_ratio"] = safe_float(result.anomaly_ratio)
        features[f"{side}_highlight_ratio"] = safe_float(result.highlight_ratio)
        features[f"{side}_roughness"] = safe_float(result.roughness)
        features[f"{side}_mean_intensity"] = safe_float(result.mean_intensity)
        features[f"{side}_std_intensity"] = safe_float(result.std_intensity)
        features[f"{side}_profile_length"] = float(result.profile_length)

    return features


def draw_edge_band_overlay(
    rectified_card_bgr: np.ndarray,
    band_ratio: float = 0.03,
    inset_ratio: float = 0.01,
) -> np.ndarray:
    """
    Dibuja overlay visual de las 4 bandas analizadas.
    Útil para depuración/UI.
    """
    if rectified_card_bgr is None or rectified_card_bgr.size == 0:
        raise ValueError("rectified_card_bgr is empty or None.")

    overlay = rectified_card_bgr.copy()
    h, w = overlay.shape[:2]
    min_dim = min(h, w)

    band = max(4, int(min_dim * band_ratio))
    inset = max(1, int(min_dim * inset_ratio))

    band = min(band, h // 4, w // 4)
    inset = min(inset, h // 10, w // 10)

    color = (255, 255, 255)
    thickness = 2

    # top
    cv2.rectangle(overlay, (inset, inset), (w - inset, inset + band), color, thickness)
    # bottom
    cv2.rectangle(
        overlay,
        (inset, h - inset - band),
        (w - inset, h - inset),
        color,
        thickness,
    )
    # left
    cv2.rectangle(overlay, (inset, inset), (inset + band, h - inset), color, thickness)
    # right
    cv2.rectangle(
        overlay,
        (w - inset - band, inset),
        (w - inset, h - inset),
        color,
        thickness,
    )

    return overlay