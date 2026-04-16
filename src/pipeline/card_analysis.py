from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.features.capture_assessment import classify_capture_quality
from src.features.centering_features import draw_content_bbox, extract_centering_features
from src.features.corner_features import compute_corner_features, draw_corner_patch_overlay
from src.features.edge_features import compute_edge_features, draw_edge_band_overlay
from src.features.geometry_features import extract_geometry_features
from src.features.visual_features import extract_visual_features
from src.features.whitening_surface_features import compute_whitening_surface_features
from src.scoring.condition_score import (
    compute_capture_quality_score,
    compute_centering_score,
    compute_corner_score,
    compute_edge_score,
    compute_gradix_condition_stub,
    compute_gradix_condition_stub_v2,
    compute_gradix_condition_stub_v3,
    compute_gradix_condition_stub_v4,
    compute_preliminary_gradix_score,
    compute_whitening_surface_score,
)
from src.utils.image_utils import ensure_bgr_uint8
from src.vision.card_detector import detect_card_contour
from src.vision.perspective import warp_card_perspective
from src.vision.postwarp_validation import validate_rectified_card


def analyze_card_image(image_bgr: np.ndarray) -> Dict[str, Any]:
    image_bgr = ensure_bgr_uint8(image_bgr)

    detection = detect_card_contour(image_bgr)
    corners = detection.get("corners")
    contour = detection.get("contour")
    used_fallback = detection.get("used_fallback", True)
    detection_metrics = detection.get("metrics", {})
    debug_images: Dict[str, np.ndarray] = dict(detection.get("debug_images", {}))

    warp: Dict[str, Any] = {
        "computed": False,
        "reason": "missing_valid_corners",
        "data": None,
    }
    postwarp_validation: Dict[str, Any] = {
        "computed": False,
        "reason": "warp_not_available",
        "data": None,
    }
    features: Dict[str, Any] = {
        "computed": False,
        "reason": "warp_not_available",
        "visual": None,
        "geometry": None,
        "centering": None,
        "edge": None,
        "corner": None,
        "whitening_surface": None,
    }
    scores: Dict[str, Any] = {
        "computed": False,
        "reason": "features_not_available",
        "capture_quality": None,
        "preliminary": None,
        "centering": None,
        "edge": None,
        "corner": None,
        "whitening_surface": None,
        "condition_stub_v1": None,
        "condition_stub_v2": None,
        "condition_stub_v3": None,
        "condition_stub_v4": None,
    }
    assessment: Dict[str, Any] = {
        "computed": False,
        "reason": "scores_not_available",
        "capture_quality": None,
        "analysis_ready": False,
        "analysis_recommended": False,
    }

    warped_card: Optional[np.ndarray] = None

    if corners is not None and len(corners) == 4:
        warp_result = warp_card_perspective(image_bgr, corners)
        warped_card = warp_result.get("warped_image")
        warp = {
            "computed": True,
            "reason": "",
            "data": warp_result,
        }

    if warped_card is not None:
        postwarp_result = validate_rectified_card(
            warped_card,
            detection_metrics=detection_metrics,
        )
        postwarp_validation = {
            "computed": True,
            "reason": "",
            "data": postwarp_result,
        }

        visual_features = extract_visual_features(warped_card)
        capture_quality_score = compute_capture_quality_score(visual_features)
        geometry_features = extract_geometry_features(
            contour=contour,
            image_shape=image_bgr.shape,
            warped_aspect_ratio=visual_features["aspect_ratio"],
            used_fallback=used_fallback,
        )
        centering_features = extract_centering_features(warped_card)
        edge_features = compute_edge_features(warped_card)
        corner_features = compute_corner_features(warped_card)
        whitening_surface_features = compute_whitening_surface_features(
            warped_card,
            blur_score=visual_features["blur_score"],
            contrast_score=visual_features["contrast_score"],
            capture_score=capture_quality_score["capture_quality_score"],
        )

        features = {
            "computed": True,
            "reason": "",
            "visual": visual_features,
            "geometry": geometry_features,
            "centering": centering_features,
            "edge": edge_features,
            "corner": corner_features,
            "whitening_surface": whitening_surface_features,
        }

        preliminary_score = compute_preliminary_gradix_score(
            capture_quality_score=capture_quality_score["capture_quality_score"],
            geometry_features=geometry_features,
        )
        centering_score = compute_centering_score(centering_features)
        edge_score = compute_edge_score(edge_features)
        corner_score = compute_corner_score(corner_features)
        whitening_surface_score = compute_whitening_surface_score(
            whitening_surface_features
        )
        condition_stub_v1 = compute_gradix_condition_stub(
            preliminary_gradix_score=preliminary_score["gradix_preliminary_score"],
            centering_score=centering_score["centering_score"],
        )
        condition_stub_v2 = compute_gradix_condition_stub_v2(
            preliminary_gradix_score=preliminary_score["gradix_preliminary_score"],
            centering_score=centering_score["centering_score"],
            gradix_edge_score=edge_score["gradix_edge_score"],
            edge_confidence=edge_features["edge_confidence"],
        )
        condition_stub_v3 = compute_gradix_condition_stub_v3(
            preliminary_gradix_score=preliminary_score["gradix_preliminary_score"],
            centering_score=centering_score["centering_score"],
            gradix_edge_score=edge_score["gradix_edge_score"],
            gradix_corner_score=corner_score["gradix_corner_score"],
            edge_confidence=edge_features["edge_confidence"],
            corner_confidence=corner_features["corner_confidence"],
        )
        condition_stub_v4 = compute_gradix_condition_stub_v4(
            preliminary_gradix_score=preliminary_score["gradix_preliminary_score"],
            centering_score=centering_score["centering_score"],
            gradix_edge_score=edge_score["gradix_edge_score"],
            gradix_corner_score=corner_score["gradix_corner_score"],
            gradix_whitening_surface_score=whitening_surface_score[
                "gradix_whitening_surface_score"
            ],
            edge_confidence=edge_features["edge_confidence"],
            corner_confidence=corner_features["corner_confidence"],
            whitening_confidence=whitening_surface_features["whitening_confidence"],
            surface_confidence=whitening_surface_features["surface_confidence"],
        )

        scores = {
            "computed": True,
            "reason": "",
            "capture_quality": capture_quality_score,
            "preliminary": preliminary_score,
            "centering": centering_score,
            "edge": edge_score,
            "corner": corner_score,
            "whitening_surface": whitening_surface_score,
            "condition_stub_v1": condition_stub_v1,
            "condition_stub_v2": condition_stub_v2,
            "condition_stub_v3": condition_stub_v3,
            "condition_stub_v4": condition_stub_v4,
        }

        capture_quality_assessment = classify_capture_quality(
            contour=contour,
            image_shape=image_bgr.shape,
            aspect_ratio=visual_features["aspect_ratio"],
            capture_score=capture_quality_score["capture_quality_score"],
            used_fallback=used_fallback,
        )

        assessment = {
            "computed": True,
            "reason": "",
            "capture_quality": capture_quality_assessment,
            "analysis_ready": bool(postwarp_result["postwarp_valid"]),
            "analysis_recommended": bool(not postwarp_result["retry_recommended"]),
        }

        debug_images["warped_card"] = warped_card
        debug_images["warped_with_bbox"] = draw_content_bbox(
            warped_card,
            centering_features["content_bbox"],
        )
        debug_images["edge_overlay"] = draw_edge_band_overlay(warped_card)
        debug_images["corner_overlay"] = draw_corner_patch_overlay(warped_card)

    return {
        "detection": detection,
        "warp": warp,
        "postwarp_validation": postwarp_validation,
        "features": features,
        "scores": scores,
        "assessment": assessment,
        "debug_images": debug_images,
    }
