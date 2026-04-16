from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from src.config.settings import (
    CARD_POSTWARP_MAX_CROP_RISK,
    CARD_POSTWARP_MIN_BORDER_SCORE,
    CARD_POSTWARP_MIN_VALID_SCORE,
    CARD_POSTWARP_RETRY_SCORE_THRESHOLD,
)
from src.utils.image_utils import ensure_bgr_uint8

CARD_TARGET_ASPECT_RATIO = 0.714


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-6:
        return 0.0
    return float(numerator / denominator)


def _aspect_ratio_score(
    image_shape: tuple[int, int],
    target_aspect_ratio: float = CARD_TARGET_ASPECT_RATIO,
) -> Dict[str, float]:
    height, width = image_shape[:2]
    aspect_ratio = _safe_div(float(width), float(height))
    deviation = abs(aspect_ratio - target_aspect_ratio)

    if deviation <= 0.025:
        score = 1.0
    elif deviation <= 0.055:
        score = 0.82
    elif deviation <= 0.090:
        score = 0.55
    else:
        score = max(0.0, 1.0 - deviation / 0.18)

    return {
        "rectified_aspect_ratio": float(aspect_ratio),
        "rectified_aspect_ratio_deviation": float(deviation),
        "rectified_aspect_ratio_score": float(score),
    }


def _transition_continuity_score(
    gray: np.ndarray,
    side: str,
    peak_index: int,
) -> float:
    h, w = gray.shape[:2]
    if side == "top":
        row_a = int(np.clip(peak_index - 1, 0, h - 1))
        row_b = int(np.clip(peak_index, 0, h - 1))
        transition = np.abs(
            gray[row_b, :].astype(np.float32) - gray[row_a, :].astype(np.float32)
        )
    elif side == "bottom":
        row_a = int(np.clip(h - peak_index, 0, h - 1))
        row_b = int(np.clip(h - peak_index - 1, 0, h - 1))
        transition = np.abs(
            gray[row_b, :].astype(np.float32) - gray[row_a, :].astype(np.float32)
        )
    elif side == "left":
        col_a = int(np.clip(peak_index - 1, 0, w - 1))
        col_b = int(np.clip(peak_index, 0, w - 1))
        transition = np.abs(
            gray[:, col_b].astype(np.float32) - gray[:, col_a].astype(np.float32)
        )
    else:
        col_a = int(np.clip(w - peak_index, 0, w - 1))
        col_b = int(np.clip(w - peak_index - 1, 0, w - 1))
        transition = np.abs(
            gray[:, col_b].astype(np.float32) - gray[:, col_a].astype(np.float32)
        )

    if transition.size == 0:
        return 0.0

    threshold = float(np.mean(transition) + 0.35 * np.std(transition))
    threshold = max(6.0, threshold)
    return _clip01(float(np.mean(transition >= threshold)))


def _sample_outer_inner_texture(
    gray: np.ndarray,
    side: str,
    border_depth: int,
    search_depth: int,
) -> Dict[str, float]:
    h, w = gray.shape[:2]
    inner_end = min(max(border_depth + 1, 2 * border_depth), search_depth)

    if side == "top":
        outer_band = gray[:border_depth, :]
        inner_band = gray[border_depth:inner_end, :]
    elif side == "bottom":
        outer_band = gray[max(0, h - border_depth):, :]
        inner_band = gray[max(0, h - inner_end):max(0, h - border_depth), :]
    elif side == "left":
        outer_band = gray[:, :border_depth]
        inner_band = gray[:, border_depth:inner_end]
    else:
        outer_band = gray[:, max(0, w - border_depth):]
        inner_band = gray[:, max(0, w - inner_end):max(0, w - border_depth)]

    if outer_band.size == 0 or inner_band.size == 0:
        return {
            "outer_band_std": 0.0,
            "inner_band_std": 0.0,
            "outer_inner_texture_ratio": 1.0,
        }

    outer_std = float(np.std(outer_band))
    inner_std = float(np.std(inner_band))
    return {
        "outer_band_std": outer_std,
        "inner_band_std": inner_std,
        "outer_inner_texture_ratio": _safe_div(outer_std + 1e-6, inner_std + 1e-6),
    }


def _side_border_metrics(gray: np.ndarray, side: str) -> Dict[str, float]:
    h, w = gray.shape[:2]
    min_dim = max(1, min(h, w))
    search_depth = int(np.clip(round(min_dim * 0.14), 10, max(10, min_dim // 4)))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if side == "top":
        gradient_map = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        profile = np.mean(np.abs(gradient_map[:search_depth, :]), axis=1)
    elif side == "bottom":
        gradient_map = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        profile = np.mean(np.abs(gradient_map[h - search_depth :, :]), axis=1)[::-1]
    elif side == "left":
        gradient_map = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        profile = np.mean(np.abs(gradient_map[:, :search_depth]), axis=0)
    else:
        gradient_map = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        profile = np.mean(np.abs(gradient_map[:, w - search_depth :]), axis=0)[::-1]

    if profile.size == 0:
        return {
            f"{side}_border_score": 0.0,
            f"{side}_border_peak_index": 0.0,
            f"{side}_border_peak_ratio": 1.0,
            f"{side}_border_strength": 0.0,
            f"{side}_border_continuity": 0.0,
            f"{side}_crop_risk": 1.0,
            f"{side}_outer_inner_texture_ratio": 1.0,
        }

    peak_index = int(np.argmax(profile)) + 1
    peak_strength = float(profile[peak_index - 1])
    profile_reference = float(np.percentile(profile, 90)) if profile.size > 3 else peak_strength
    strength_score = _clip01(_safe_div(peak_strength, profile_reference + 1e-6))

    peak_ratio = _safe_div(float(peak_index), float(search_depth))
    if 0.08 <= peak_ratio <= 0.62:
        position_score = 1.0
    elif 0.03 <= peak_ratio <= 0.78:
        position_score = 0.70
    else:
        position_score = 0.25

    continuity_score = _transition_continuity_score(gray, side=side, peak_index=peak_index)
    texture_metrics = _sample_outer_inner_texture(
        gray,
        side=side,
        border_depth=max(2, peak_index),
        search_depth=search_depth,
    )
    texture_ratio = float(texture_metrics["outer_inner_texture_ratio"])

    if texture_ratio <= 1.20:
        texture_score = 1.0
    elif texture_ratio <= 1.55:
        texture_score = 0.72
    elif texture_ratio <= 2.10:
        texture_score = 0.42
    else:
        texture_score = 0.18

    border_score = float(
        np.clip(
            0.34 * strength_score
            + 0.28 * continuity_score
            + 0.23 * position_score
            + 0.15 * texture_score,
            0.0,
            1.0,
        )
    )

    crop_risk = float(
        np.clip(
            0.40 * (1.0 - position_score)
            + 0.35 * (1.0 - continuity_score)
            + 0.25 * (1.0 - texture_score),
            0.0,
            1.0,
        )
    )

    return {
        f"{side}_border_score": border_score,
        f"{side}_border_peak_index": float(peak_index),
        f"{side}_border_peak_ratio": float(peak_ratio),
        f"{side}_border_strength": float(strength_score),
        f"{side}_border_continuity": float(continuity_score),
        f"{side}_crop_risk": float(crop_risk),
        f"{side}_outer_band_std": float(texture_metrics["outer_band_std"]),
        f"{side}_inner_band_std": float(texture_metrics["inner_band_std"]),
        f"{side}_outer_inner_texture_ratio": float(texture_ratio),
    }


def _border_consistency_metrics(gray: np.ndarray) -> Dict[str, float]:
    side_metrics = {
        side: _side_border_metrics(gray, side)
        for side in ("top", "bottom", "left", "right")
    }

    top_score = side_metrics["top"]["top_border_score"]
    bottom_score = side_metrics["bottom"]["bottom_border_score"]
    left_score = side_metrics["left"]["left_border_score"]
    right_score = side_metrics["right"]["right_border_score"]

    top_peak = side_metrics["top"]["top_border_peak_ratio"]
    bottom_peak = side_metrics["bottom"]["bottom_border_peak_ratio"]
    left_peak = side_metrics["left"]["left_border_peak_ratio"]
    right_peak = side_metrics["right"]["right_border_peak_ratio"]

    mean_border_score = float(np.mean([top_score, bottom_score, left_score, right_score]))
    mean_continuity = float(
        np.mean(
            [
                side_metrics["top"]["top_border_continuity"],
                side_metrics["bottom"]["bottom_border_continuity"],
                side_metrics["left"]["left_border_continuity"],
                side_metrics["right"]["right_border_continuity"],
            ]
        )
    )
    mean_crop_risk = float(
        np.mean(
            [
                side_metrics["top"]["top_crop_risk"],
                side_metrics["bottom"]["bottom_crop_risk"],
                side_metrics["left"]["left_crop_risk"],
                side_metrics["right"]["right_crop_risk"],
            ]
        )
    )

    vertical_margin_consistency = _clip01(1.0 - abs(top_peak - bottom_peak) / 0.35)
    horizontal_margin_consistency = _clip01(1.0 - abs(left_peak - right_peak) / 0.35)
    margin_consistency_score = float(
        0.5 * vertical_margin_consistency + 0.5 * horizontal_margin_consistency
    )

    frame_coherence_score = float(
        np.clip(
            0.40 * mean_border_score
            + 0.30 * mean_continuity
            + 0.30 * margin_consistency_score,
            0.0,
            1.0,
        )
    )

    metrics: Dict[str, float] = {
        "outer_border_score": mean_border_score,
        "outer_border_continuity_score": mean_continuity,
        "margin_consistency_score": margin_consistency_score,
        "vertical_margin_consistency_score": vertical_margin_consistency,
        "horizontal_margin_consistency_score": horizontal_margin_consistency,
        "crop_risk_score": mean_crop_risk,
        "frame_coherence_score": frame_coherence_score,
    }
    for side in ("top", "bottom", "left", "right"):
        metrics.update(side_metrics[side])
    return metrics


def _apply_detection_context(
    postwarp_score: float,
    detection_metrics: Optional[Dict[str, object]],
) -> Dict[str, float]:
    if not detection_metrics:
        return {
            "postwarp_score_adjusted": float(postwarp_score),
            "postwarp_context_factor": 1.0,
        }

    weak_detection = float(detection_metrics.get("weak_detection", 0.0))
    detection_confidence = float(detection_metrics.get("detection_confidence", 0.0))

    context_factor = 1.0
    if weak_detection >= 0.5:
        context_factor *= 0.95
    if 0.0 < detection_confidence < 0.40:
        context_factor *= 0.96

    return {
        "postwarp_score_adjusted": float(np.clip(postwarp_score * context_factor, 0.0, 1.0)),
        "postwarp_context_factor": float(context_factor),
    }


def validate_rectified_card(
    warped_card_bgr: np.ndarray,
    detection_metrics: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    bgr = ensure_bgr_uint8(warped_card_bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    aspect_metrics = _aspect_ratio_score(bgr.shape[:2])
    border_metrics = _border_consistency_metrics(gray)

    postwarp_score_raw = float(
        np.clip(
            0.24 * aspect_metrics["rectified_aspect_ratio_score"]
            + 0.26 * border_metrics["outer_border_score"]
            + 0.18 * border_metrics["outer_border_continuity_score"]
            + 0.16 * border_metrics["margin_consistency_score"]
            + 0.16 * (1.0 - border_metrics["crop_risk_score"]),
            0.0,
            1.0,
        )
    )

    context_metrics = _apply_detection_context(
        postwarp_score=postwarp_score_raw,
        detection_metrics=detection_metrics,
    )
    postwarp_score = float(context_metrics["postwarp_score_adjusted"])

    invalid_reasons: List[str] = []
    if aspect_metrics["rectified_aspect_ratio_score"] < 0.35:
        invalid_reasons.append("implausible_rectified_aspect_ratio")
    if border_metrics["outer_border_score"] < CARD_POSTWARP_MIN_BORDER_SCORE:
        invalid_reasons.append("weak_outer_border_visibility")
    if border_metrics["outer_border_continuity_score"] < 0.28:
        invalid_reasons.append("weak_outer_border_continuity")
    if border_metrics["margin_consistency_score"] < 0.24:
        invalid_reasons.append("inconsistent_outer_margins")
    if border_metrics["crop_risk_score"] > CARD_POSTWARP_MAX_CROP_RISK:
        invalid_reasons.append("high_crop_risk")

    postwarp_valid = bool(
        postwarp_score >= CARD_POSTWARP_MIN_VALID_SCORE
        and aspect_metrics["rectified_aspect_ratio_score"] >= 0.35
        and border_metrics["crop_risk_score"] <= CARD_POSTWARP_MAX_CROP_RISK
        and border_metrics["outer_border_score"] >= CARD_POSTWARP_MIN_BORDER_SCORE
    )

    retry_recommended = bool(
        (not postwarp_valid)
        or postwarp_score < CARD_POSTWARP_RETRY_SCORE_THRESHOLD
        or border_metrics["crop_risk_score"] > 0.55
        or border_metrics["outer_border_continuity_score"] < 0.42
    )

    result: Dict[str, object] = {
        "postwarp_valid": postwarp_valid,
        "postwarp_score": postwarp_score,
        "postwarp_score_raw": postwarp_score_raw,
        "retry_recommended": retry_recommended,
        "invalid_reasons": invalid_reasons,
        "image_width": int(bgr.shape[1]),
        "image_height": int(bgr.shape[0]),
        **aspect_metrics,
        **border_metrics,
        **context_metrics,
    }
    return result
