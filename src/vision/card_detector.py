from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config.settings import (
    CARD_BORDER_SUPPORT_BAND_WIDTH,
    CARD_BORDER_SUPPORT_ENABLED,
    CARD_BORDER_SUPPORT_VERTICAL_MIN,
    CARD_COLOR_SAMPLE_OFFSET,
    CARD_COLOR_SEPARATION_ENABLED,
    CARD_GEOMETRY_STRICTNESS,
    CARD_MIN_AREA_RATIO,
    CARD_MIN_HEIGHT_RATIO,
    CARD_MIN_WIDTH_RATIO,
    CARD_OUTER_ENABLE_YOLO_PRIOR,
    CARD_OUTER_MASK_LONG_KERNEL_RATIO,
    CARD_OUTER_MASK_SHORT_KERNEL_RATIO,
    CARD_OUTER_MASK_SQUARE_KERNEL_RATIO,
    CARD_OUTER_MIN_CANDIDATE_SCORE,
    CARD_OUTER_MIN_COMPONENT_AREA,
    YOLO_CARD_BBOX_MARGIN,
)
from src.services.yolo_card_detector import detect_card_bbox, get_yolo_status
from src.utils.helpers import order_points, polygon_area
from src.utils.image_utils import ensure_bgr_uint8

CARD_TARGET_ASPECT_RATIO = 0.715
CARD_SIDE_SUPPORT_MIN_BOTTLENECK = 0.16
CARD_SIDE_SUPPORT_HARD_MIN = 0.10
CARD_SIDE_SUPPORT_STD_SOFT_MAX = 0.18
CARD_SIDE_SUPPORT_STD_HARD_MAX = 0.26
CARD_TOP_BOTTOM_DIFF_SOFT_MAX = 0.14
CARD_TOP_BOTTOM_DIFF_HARD_MAX = 0.24
CARD_LEFT_RIGHT_DIFF_SOFT_MAX = 0.16
CARD_LEFT_RIGHT_DIFF_HARD_MAX = 0.26
CARD_ASPECT_DEVIATION_SOFT_MAX = 0.065
CARD_ASPECT_DEVIATION_HARD_MAX = 0.14
CARD_EVIDENCE_COVERAGE_SOFT_MIN = 0.78
CARD_EVIDENCE_COVERAGE_HARD_MIN = 0.60
CARD_REFINEMENT_GEOMETRY_DELTA = 0.04
CARD_REFINEMENT_SIDE_SUPPORT_DELTA = 0.03
CARD_ROI_SECOND_PASS_MARGIN_RATIO = 0.18


@dataclass
class CardDetectionResult:
    success: bool
    contour: Optional[np.ndarray]
    corners: Optional[np.ndarray]
    used_fallback: bool
    debug_images: Dict[str, np.ndarray]
    metrics: Dict[str, object]

    def to_dict(self) -> Dict:
        return asdict(self)


def _resize_for_detection(
    image: np.ndarray, target_long_side: int = 1200
) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= target_long_side:
        return image.copy(), 1.0

    scale = target_long_side / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _build_grayscale_view(image: np.ndarray) -> np.ndarray:
    return _to_gray(image)


def _build_edge_view(gray: np.ndarray) -> np.ndarray:
    med = float(np.median(gray))
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    return cv2.Canny(gray, lower, upper)


def _build_binary_view(gray: np.ndarray) -> np.ndarray:
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        6,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)


def _build_hsv_view(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def _build_lab_view(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def _preprocess_gray(gray: np.ndarray, blur_size: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(gray, (blur_size, blur_size), 0)


def _make_candidate_maps(gray: np.ndarray, kernel_scale: float = 1.0) -> Dict[str, np.ndarray]:
    med = float(np.median(gray))
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    canny = cv2.Canny(gray, lower, upper)

    grad_kernel = max(3, int(round(5 * kernel_scale)))
    if grad_kernel % 2 == 0:
        grad_kernel += 1
    close_kernel = max(5, int(round(9 * kernel_scale)))
    if close_kernel % 2 == 0:
        close_kernel += 1

    kernel_g = cv2.getStructuringElement(cv2.MORPH_RECT, (grad_kernel, grad_kernel))
    morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_g)
    _, morph_bin = cv2.threshold(
        morph_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adap = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        6,
    )
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_kernel, close_kernel)
    )
    canny_closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    morph_closed = cv2.morphologyEx(
        morph_bin, cv2.MORPH_CLOSE, kernel_close, iterations=2
    )
    adap_closed = cv2.morphologyEx(adap, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    otsu_closed = cv2.morphologyEx(
        otsu_inv, cv2.MORPH_CLOSE, kernel_close, iterations=2
    )

    return {
        "canny": canny,
        "canny_closed": canny_closed,
        "morph_grad": morph_grad,
        "morph_closed": morph_closed,
        "adaptive_closed": adap_closed,
        "otsu_closed": otsu_closed,
    }


def _build_multilayer_views(
    image: np.ndarray,
    blur_size: int = 5,
    kernel_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    gray = _build_grayscale_view(image)
    gray_blurred = _preprocess_gray(gray, blur_size=blur_size)
    candidate_maps = _make_candidate_maps(gray_blurred, kernel_scale=kernel_scale)
    binary_view = _build_binary_view(gray_blurred)
    hsv_view = _build_hsv_view(image)
    lab_view = _build_lab_view(image)
    structural_mask = _build_structural_mask(candidate_maps)

    multilayer_overlay = cv2.cvtColor(gray_blurred, cv2.COLOR_GRAY2BGR)
    multilayer_overlay[:, :, 1] = cv2.max(multilayer_overlay[:, :, 1], structural_mask)
    multilayer_overlay[:, :, 2] = cv2.max(
        multilayer_overlay[:, :, 2],
        candidate_maps.get("canny_closed", candidate_maps.get("canny", structural_mask)),
    )

    return {
        "gray": gray_blurred,
        "edge_view": _build_edge_view(gray_blurred),
        "binary_view": binary_view,
        "hsv_view": hsv_view,
        "lab_view": lab_view,
        "structural_mask": structural_mask,
        "multilayer_overlay": multilayer_overlay,
        **candidate_maps,
    }
def _rect_from_min_area(contour: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.asarray(box, dtype=np.float32)


def _quad_side_lengths(quad: np.ndarray) -> Tuple[float, float, float, float]:
    quad = order_points(quad)
    top = float(np.linalg.norm(quad[1] - quad[0]))
    right = float(np.linalg.norm(quad[2] - quad[1]))
    bottom = float(np.linalg.norm(quad[2] - quad[3]))
    left = float(np.linalg.norm(quad[3] - quad[0]))
    return top, right, bottom, left


def _aspect_ratio_of_quad(quad: np.ndarray) -> float:
    top, right, bottom, left = _quad_side_lengths(quad)
    width = max(1.0, (top + bottom) / 2.0)
    height = max(1.0, (right + left) / 2.0)
    return width / height


def _center_distance_score(quad: np.ndarray, image_shape: Tuple[int, int]) -> float:
    h, w = image_shape[:2]
    img_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    quad_center = np.mean(order_points(quad), axis=0)
    dist = np.linalg.norm(quad_center - img_center)
    max_dist = np.linalg.norm(np.array([w / 2.0, h / 2.0], dtype=np.float32))
    score = 1.0 - (dist / max_dist if max_dist > 0 else 0.0)
    return float(max(0.0, min(1.0, score)))


def _candidate_margin_ratios(
    quad: np.ndarray, image_shape: Tuple[int, int]
) -> np.ndarray:
    h, w = image_shape[:2]
    ordered = order_points(quad)
    return np.array(
        [
            ordered[:, 0].min(),
            w - ordered[:, 0].max(),
            ordered[:, 1].min(),
            h - ordered[:, 1].max(),
        ],
        dtype=np.float32,
    ) / float(max(1, min(h, w)))


def _candidate_touch_count(
    quad: np.ndarray,
    image_shape: Tuple[int, int],
    touch_threshold: float = 0.0035,
) -> int:
    margin_ratios = _candidate_margin_ratios(quad, image_shape)
    return int(np.sum(margin_ratios <= touch_threshold))


def _outerness_score(
    quad: np.ndarray,
    image_shape: Tuple[int, int],
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> float:
    if roi_bbox is not None:
        x1, y1, x2, y2 = roi_bbox
        local_h = max(1, y2 - y1)
        local_w = max(1, x2 - x1)
        shifted = order_points(quad) - np.array([x1, y1], dtype=np.float32)
        return _outerness_score(shifted, (local_h, local_w))

    margin_ratios = _candidate_margin_ratios(quad, image_shape)
    touch_count = _candidate_touch_count(quad, image_shape)
    if touch_count >= 3:
        return 0.0

    sorted_margins = np.sort(margin_ratios)
    second_smallest_margin = float(sorted_margins[1]) if sorted_margins.size >= 2 else 0.0
    img_h, img_w = image_shape[:2]
    poly_area = polygon_area(order_points(quad))
    area_ratio = poly_area / float(max(1, img_w * img_h))

    area_term = min(1.0, area_ratio / 0.82)
    margin_term = min(1.0, second_smallest_margin / 0.025)

    touch_penalty = 1.0
    if touch_count == 2:
        touch_penalty = 0.70
    elif touch_count == 1:
        touch_penalty = 0.92

    return float((0.85 * area_term + 0.15 * margin_term) * touch_penalty)

def _aspect_validity_score(aspect_ratio: float) -> float:
    target = CARD_TARGET_ASPECT_RATIO
    diff = abs(aspect_ratio - target)
    if diff <= 0.03:
        return 1.0
    if diff <= 0.08:
        return 0.75
    if diff <= 0.14:
        return 0.45
    return 0.0


def _edge_support_score(quad: np.ndarray, edge_map: Optional[np.ndarray]) -> float:
    if edge_map is None or edge_map.size == 0:
        return 0.0

    h, w = edge_map.shape[:2]
    min_dim = max(1, min(h, w))
    thickness = max(2, int(round(min_dim * 0.012)))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(mask, [order_points(quad).astype(np.int32)], True, 255, thickness)
    active = edge_map[mask > 0]
    if active.size == 0:
        return 0.0

    support_ratio = float(np.mean(active > 0))
    return float(min(1.0, support_ratio / 0.30))


def _build_structural_mask(candidate_maps: Dict[str, np.ndarray]) -> np.ndarray:
    base = None
    for key in ("canny_closed", "adaptive_closed", "otsu_closed", "morph_closed"):
        candidate_map = candidate_maps.get(key)
        if candidate_map is None:
            continue
        binary = candidate_map
        if binary.dtype != np.uint8:
            binary = np.clip(binary, 0, 255).astype(np.uint8)
        if np.unique(binary).size > 2:
            _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        base = binary.copy() if base is None else cv2.bitwise_or(base, binary)

    if base is None:
        return np.zeros((1, 1), dtype=np.uint8)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    mask = cv2.dilate(mask, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_line, iterations=1)
    return mask


def _candidate_bbox_xyxy(quad: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    quad = order_points(quad)
    x1 = int(np.clip(np.floor(np.min(quad[:, 0])), 0, max(0, w - 1)))
    y1 = int(np.clip(np.floor(np.min(quad[:, 1])), 0, max(0, h - 1)))
    x2 = int(np.clip(np.ceil(np.max(quad[:, 0])), x1 + 1, max(x1 + 1, w)))
    y2 = int(np.clip(np.ceil(np.max(quad[:, 1])), y1 + 1, max(y1 + 1, h)))
    return x1, y1, x2, y2


def _bbox_area(bbox_xyxy: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    return float(max(1, x2 - x1) * max(1, y2 - y1))


def _clip_bbox_xyxy(
    bbox_xyxy: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(np.clip(x1, 0, max(0, w - 1)))
    y1 = int(np.clip(y1, 0, max(0, h - 1)))
    x2 = int(np.clip(x2, x1 + 1, max(x1 + 1, w)))
    y2 = int(np.clip(y2, y1 + 1, max(y1 + 1, h)))
    return x1, y1, x2, y2


def _bbox_iou(
    bbox_a: Tuple[int, int, int, int],
    bbox_b: Tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    union = _bbox_area(bbox_a) + _bbox_area(bbox_b) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _bbox_center(bbox_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def _bbox_shape_metrics(
    bbox_xyxy: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> Dict[str, float]:
    img_h, img_w = image_shape[:2]
    bbox_w = float(max(1, bbox_xyxy[2] - bbox_xyxy[0]))
    bbox_h = float(max(1, bbox_xyxy[3] - bbox_xyxy[1]))
    area_ratio = float((bbox_w * bbox_h) / max(1.0, img_h * img_w))
    width_ratio = float(bbox_w / max(1.0, img_w))
    height_ratio = float(bbox_h / max(1.0, img_h))
    aspect_ratio = float(bbox_w / max(1.0, bbox_h))
    aspect_deviation = float(abs(aspect_ratio - CARD_TARGET_ASPECT_RATIO))
    return {
        "area_ratio": area_ratio,
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "aspect_ratio": aspect_ratio,
        "aspect_deviation": aspect_deviation,
    }


def _window_peak(profile: np.ndarray, start_ratio: float, end_ratio: float) -> float:
    if profile.size == 0:
        return 0.0
    start = int(np.clip(round(profile.shape[0] * start_ratio), 0, profile.shape[0] - 1))
    end = int(np.clip(round(profile.shape[0] * end_ratio), start + 1, profile.shape[0]))
    if end <= start:
        return 0.0
    return float(np.max(profile[start:end]))


def _inner_parallel_contour_score(
    binary_view: Optional[np.ndarray],
    bbox_xyxy: Optional[Tuple[int, int, int, int]],
) -> Tuple[float, Dict[str, float]]:
    metrics = {
        "inner_parallel_border_score": 0.0,
        "inner_parallel_left_peak": 0.0,
        "inner_parallel_right_peak": 0.0,
        "inner_parallel_top_peak": 0.0,
        "inner_parallel_bottom_peak": 0.0,
        "inner_parallel_center_penalty": 0.0,
    }
    if binary_view is None or bbox_xyxy is None:
        return 0.0, metrics

    x1, y1, x2, y2 = bbox_xyxy
    roi = binary_view[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 40 or roi.shape[1] < 30:
        return 0.0, metrics

    binary = roi
    if binary.dtype != np.uint8:
        binary = np.clip(binary, 0, 255).astype(np.uint8)
    if np.unique(binary).size > 2:
        _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    col_profile = binary.mean(axis=0).astype(np.float32) / 255.0
    row_profile = binary.mean(axis=1).astype(np.float32) / 255.0
    col_profile = cv2.GaussianBlur(col_profile.reshape(1, -1), (1, 31), 0).ravel()
    row_profile = cv2.GaussianBlur(row_profile.reshape(-1, 1), (31, 1), 0).ravel()

    left_peak = _window_peak(col_profile, 0.03, 0.20)
    right_peak = _window_peak(col_profile, 0.80, 0.97)
    top_peak = _window_peak(row_profile, 0.03, 0.18)
    bottom_peak = _window_peak(row_profile, 0.72, 0.97)
    center_col_peak = _window_peak(col_profile, 0.30, 0.70)
    center_row_peak = _window_peak(row_profile, 0.25, 0.75)

    max_col = float(max(1e-6, np.max(col_profile)))
    max_row = float(max(1e-6, np.max(row_profile)))
    left_norm = float(np.clip(left_peak / max_col, 0.0, 1.0))
    right_norm = float(np.clip(right_peak / max_col, 0.0, 1.0))
    top_norm = float(np.clip(top_peak / max_row, 0.0, 1.0))
    bottom_norm = float(np.clip(bottom_peak / max_row, 0.0, 1.0))

    side_score = 0.25 * (left_norm + right_norm + top_norm + bottom_norm)
    center_col_norm = float(np.clip(center_col_peak / max_col, 0.0, 1.0))
    center_row_norm = float(np.clip(center_row_peak / max_row, 0.0, 1.0))
    center_penalty = float(
        np.clip(
            0.5 * max(0.0, center_col_norm - 0.5 * (left_norm + right_norm))
            + 0.5 * max(0.0, center_row_norm - 0.5 * (top_norm + bottom_norm)),
            0.0,
            1.0,
        )
    )
    score = float(np.clip(side_score - 0.35 * center_penalty, 0.0, 1.0))

    metrics.update(
        {
            "inner_parallel_border_score": score,
            "inner_parallel_left_peak": left_norm,
            "inner_parallel_right_peak": right_norm,
            "inner_parallel_top_peak": top_norm,
            "inner_parallel_bottom_peak": bottom_norm,
            "inner_parallel_center_penalty": center_penalty,
        }
    )
    return score, metrics


def _select_consistent_card_bbox(
    projection_bbox: Optional[Tuple[int, int, int, int]],
    component_bbox: Optional[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
    binary_view: Optional[np.ndarray] = None,
) -> Tuple[Optional[Tuple[int, int, int, int]], Dict[str, float]]:
    metrics: Dict[str, float] = {
        "bbox_selection_used_projection": 0.0,
        "bbox_selection_used_component": 0.0,
        "bbox_selection_replaced_projection": 0.0,
        "bbox_selection_iou": 0.0,
        "bbox_selection_center_distance_ratio": 1.0,
        "projection_bbox_area_ratio": 0.0,
        "component_bbox_area_ratio": 0.0,
        "projection_inner_parallel_border_score": 0.0,
        "component_inner_parallel_border_score": 0.0,
    }

    if projection_bbox is None and component_bbox is None:
        metrics["bbox_selection_mode"] = 0.0
        return None, metrics
    if projection_bbox is None:
        metrics["bbox_selection_mode"] = 1.0
        metrics["bbox_selection_used_component"] = 1.0
        if component_bbox is not None:
            metrics["component_bbox_area_ratio"] = _bbox_shape_metrics(component_bbox, image_shape)["area_ratio"]
        return component_bbox, metrics
    if component_bbox is None:
        metrics["bbox_selection_mode"] = 2.0
        metrics["bbox_selection_used_projection"] = 1.0
        metrics["projection_bbox_area_ratio"] = _bbox_shape_metrics(projection_bbox, image_shape)["area_ratio"]
        return projection_bbox, metrics

    proj_metrics = _bbox_shape_metrics(projection_bbox, image_shape)
    comp_metrics = _bbox_shape_metrics(component_bbox, image_shape)
    metrics["projection_bbox_area_ratio"] = proj_metrics["area_ratio"]
    metrics["component_bbox_area_ratio"] = comp_metrics["area_ratio"]
    projection_inner_score, projection_inner_metrics = _inner_parallel_contour_score(binary_view, projection_bbox)
    component_inner_score, component_inner_metrics = _inner_parallel_contour_score(binary_view, component_bbox)
    metrics["projection_inner_parallel_border_score"] = projection_inner_score
    metrics["component_inner_parallel_border_score"] = component_inner_score
    metrics["projection_inner_parallel_center_penalty"] = projection_inner_metrics["inner_parallel_center_penalty"]
    metrics["component_inner_parallel_center_penalty"] = component_inner_metrics["inner_parallel_center_penalty"]

    iou = _bbox_iou(projection_bbox, component_bbox)
    metrics["bbox_selection_iou"] = float(iou)
    center_dist = float(np.linalg.norm(_bbox_center(projection_bbox) - _bbox_center(component_bbox)))
    min_dim = float(max(1, min(image_shape[:2])))
    center_dist_ratio = center_dist / min_dim
    metrics["bbox_selection_center_distance_ratio"] = float(center_dist_ratio)

    proj_plausible = _is_plausible_outer_bbox(projection_bbox, image_shape)
    comp_plausible = _is_plausible_outer_bbox(component_bbox, image_shape)
    proj_area = proj_metrics["area_ratio"]
    comp_area = comp_metrics["area_ratio"]
    area_ratio = max(proj_area, comp_area) / max(1e-6, min(proj_area, comp_area))

    choose_component = False
    if comp_plausible and not proj_plausible:
        choose_component = True
    elif proj_plausible and not comp_plausible:
        choose_component = False
    elif center_dist_ratio <= 0.16 and area_ratio >= 1.45:
        choose_component = comp_area < proj_area
    elif iou >= 0.45 and area_ratio >= 1.30:
        choose_component = comp_area < proj_area
    elif proj_area > 0.52 and comp_plausible:
        choose_component = True
    elif proj_metrics["width_ratio"] > 0.78 and comp_plausible:
        choose_component = True
    elif proj_metrics["height_ratio"] > 0.94 and comp_plausible:
        choose_component = True
    elif proj_metrics["aspect_deviation"] > comp_metrics["aspect_deviation"] + 0.03 and comp_plausible:
        choose_component = True
    elif component_inner_score > projection_inner_score + 0.18:
        choose_component = True
    elif projection_inner_score > component_inner_score + 0.22 and proj_plausible:
        choose_component = False

    if choose_component:
        metrics["bbox_selection_mode"] = 3.0
        metrics["bbox_selection_used_component"] = 1.0
        metrics["bbox_selection_replaced_projection"] = 1.0
        return component_bbox, metrics

    metrics["bbox_selection_mode"] = 4.0
    metrics["bbox_selection_used_projection"] = 1.0
    return projection_bbox, metrics

def _candidate_outer_bbox_metrics(
    quad: np.ndarray,
    outer_bbox: Optional[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
    metric_prefix: str,
) -> Dict[str, object]:
    if outer_bbox is None:
        return {
            f"{metric_prefix}_is_internal": 0.0,
            f"{metric_prefix}_distance_to_outer_border": 1.0,
            f"{metric_prefix}_relative_size_to_card_bbox": 0.0,
            f"{metric_prefix}_touches_outer_region": 0.0,
        }

    quad_bbox = _candidate_bbox_xyxy(quad, image_shape)
    qx1, qy1, qx2, qy2 = quad_bbox
    ox1, oy1, ox2, oy2 = outer_bbox
    outer_w = float(max(1, ox2 - ox1))
    outer_h = float(max(1, oy2 - oy1))
    norm = float(max(1.0, min(outer_w, outer_h)))

    margins = np.array(
        [qx1 - ox1, ox2 - qx2, qy1 - oy1, oy2 - qy2],
        dtype=np.float32,
    )
    normalized_margins = margins / norm
    contained = bool(np.all(margins >= -0.03 * norm))
    close_threshold = 0.12
    touch_count = int(np.sum(normalized_margins <= close_threshold))
    touches_outer_region = bool(
        touch_count >= 2
        or min(normalized_margins[0], normalized_margins[1]) <= close_threshold
        or min(normalized_margins[2], normalized_margins[3]) <= close_threshold
    )

    quad_center = np.mean(order_points(quad), axis=0)
    outer_center = np.array(
        [(ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0],
        dtype=np.float32,
    )
    center_distance = float(np.linalg.norm(quad_center - outer_center) / norm)
    relative_size = float(polygon_area(order_points(quad)) / max(1.0, _bbox_area(outer_bbox)))
    border_distance = float(np.mean(np.clip(normalized_margins, 0.0, None)))

    is_internal = bool(
        contained
        and relative_size < 0.72
        and not touches_outer_region
        and border_distance > 0.10
        and center_distance < 0.28
    )

    rejection_reason = ""
    if is_internal:
        if relative_size < 0.42:
            rejection_reason = "too_small_inside_outer_bbox"
        elif border_distance > 0.18:
            rejection_reason = "too_far_from_outer_border"
        else:
            rejection_reason = "central_internal_shape"

    return {
        f"{metric_prefix}_is_internal": float(1.0 if is_internal else 0.0),
        f"{metric_prefix}_distance_to_outer_border": float(border_distance),
        f"{metric_prefix}_relative_size_to_card_bbox": float(np.clip(relative_size, 0.0, 2.0)),
        f"{metric_prefix}_touches_outer_region": float(1.0 if touches_outer_region else 0.0),
        f"{metric_prefix}_outer_touch_count": float(touch_count),
        f"{metric_prefix}_center_distance_to_card_bbox": float(center_distance),
        f"{metric_prefix}_rejection_reason": rejection_reason,
    }


def _compute_evidence_coverage_metrics(
    quad: np.ndarray,
    image_shape: Tuple[int, int],
    structural_mask: Optional[np.ndarray],
    binary_view: Optional[np.ndarray],
) -> Dict[str, float]:
    combined: Optional[np.ndarray] = None
    for source in (structural_mask, binary_view):
        if source is None or source.size == 0:
            continue
        binary = source
        if binary.dtype != np.uint8:
            binary = np.clip(binary, 0, 255).astype(np.uint8)
        if binary.ndim != 2:
            continue
        if np.unique(binary).size > 2:
            _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        combined = binary.copy() if combined is None else cv2.bitwise_or(combined, binary)

    if combined is None or combined.size == 0:
        return {
            "evidence_bbox_width_ratio": 0.0,
            "evidence_bbox_height_ratio": 0.0,
            "evidence_bbox_area_ratio": 0.0,
            "evidence_coverage_score": 0.0,
        }

    x1, y1, x2, y2 = _candidate_bbox_xyxy(quad, image_shape)
    roi = combined[y1:y2, x1:x2]
    if roi.size == 0:
        return {
            "evidence_bbox_width_ratio": 0.0,
            "evidence_bbox_height_ratio": 0.0,
            "evidence_bbox_area_ratio": 0.0,
            "evidence_coverage_score": 0.0,
        }

    ys, xs = np.where(roi > 0)
    if xs.size == 0 or ys.size == 0:
        return {
            "evidence_bbox_width_ratio": 0.0,
            "evidence_bbox_height_ratio": 0.0,
            "evidence_bbox_area_ratio": 0.0,
            "evidence_coverage_score": 0.0,
        }

    evidence_w = float(xs.max() - xs.min() + 1)
    evidence_h = float(ys.max() - ys.min() + 1)
    bbox_w = float(max(1, x2 - x1))
    bbox_h = float(max(1, y2 - y1))
    width_ratio = float(np.clip(evidence_w / bbox_w, 0.0, 1.0))
    height_ratio = float(np.clip(evidence_h / bbox_h, 0.0, 1.0))
    evidence_area = evidence_w * evidence_h
    bbox_area = bbox_w * bbox_h
    area_ratio = float(np.clip(evidence_area / max(1.0, bbox_area), 0.0, 1.0))
    coverage_score = float(
        np.clip(
            0.42 * width_ratio + 0.42 * height_ratio + 0.16 * np.sqrt(area_ratio),
            0.0,
            1.0,
        )
    )
    return {
        "evidence_bbox_width_ratio": width_ratio,
        "evidence_bbox_height_ratio": height_ratio,
        "evidence_bbox_area_ratio": area_ratio,
        "evidence_coverage_score": coverage_score,
    }


def _collect_side_support_stats(metrics: Dict[str, float]) -> Dict[str, float]:
    side_values = np.asarray(
        [
            float(metrics.get("left_border_support", 0.0)),
            float(metrics.get("right_border_support", 0.0)),
            float(metrics.get("top_border_support", 0.0)),
            float(metrics.get("bottom_border_support", 0.0)),
        ],
        dtype=np.float32,
    )
    return {
        "min_side_support": float(np.min(side_values)),
        "max_side_support": float(np.max(side_values)),
        "side_support_std": float(np.std(side_values)),
    }


def validate_candidate_geometry(
    quad: np.ndarray,
    image_shape: Tuple[int, int],
    metrics: Dict[str, float],
) -> Dict[str, object]:
    quad = order_points(quad)
    img_h, img_w = image_shape[:2]
    image_area = float(max(1, img_h * img_w))

    top, right, bottom, left = _quad_side_lengths(quad)
    top_bottom_diff = abs(top - bottom) / max(1.0, max(top, bottom))
    left_right_diff = abs(left - right) / max(1.0, max(left, right))
    top_bottom_ratio = min(top, bottom) / max(1.0, max(top, bottom))
    left_right_ratio = min(left, right) / max(1.0, max(left, right))

    aspect_ratio = _aspect_ratio_of_quad(quad)
    aspect_ratio_deviation = abs(aspect_ratio - CARD_TARGET_ASPECT_RATIO)

    quad_area = polygon_area(quad)
    area_ratio = quad_area / image_area

    bbox_width_ratio = float(metrics.get("bbox_width_ratio", 0.0))
    bbox_height_ratio = float(metrics.get("bbox_height_ratio", 0.0))
    evidence_width_ratio = float(metrics.get("evidence_bbox_width_ratio", 0.0))
    evidence_height_ratio = float(metrics.get("evidence_bbox_height_ratio", 0.0))
    evidence_coverage_score = float(metrics.get("evidence_coverage_score", 0.0))
    min_side_support = float(metrics.get("min_side_support", 0.0))
    side_support_std = float(metrics.get("side_support_std", 0.0))

    if area_ratio < CARD_MIN_AREA_RATIO:
        area_plausibility_score = 0.0
    elif area_ratio < 0.18:
        area_plausibility_score = 0.35
    elif area_ratio < 0.28:
        area_plausibility_score = 0.75
    elif area_ratio < 0.58:
        area_plausibility_score = 1.0
    elif area_ratio < 0.72:
        area_plausibility_score = 0.55
    elif area_ratio < 0.84:
        area_plausibility_score = 0.20
    else:
        area_plausibility_score = 0.0

    side_length_consistency_score = float(
        np.clip(
            0.55 * max(0.0, 1.0 - (top_bottom_diff / CARD_TOP_BOTTOM_DIFF_HARD_MAX))
            + 0.45 * max(0.0, 1.0 - (left_right_diff / CARD_LEFT_RIGHT_DIFF_HARD_MAX)),
            0.0,
            1.0,
        )
    )

    reasons: List[str] = []
    penalty = 0.0
    geometry_valid = True

    if top_bottom_diff > CARD_TOP_BOTTOM_DIFF_SOFT_MAX:
        penalty += 0.18
        reasons.append("top_bottom_length_mismatch")
    if top_bottom_diff > CARD_TOP_BOTTOM_DIFF_HARD_MAX:
        geometry_valid = False
        penalty += 0.24

    if left_right_diff > CARD_LEFT_RIGHT_DIFF_SOFT_MAX:
        penalty += 0.16
        reasons.append("left_right_length_mismatch")
    if left_right_diff > CARD_LEFT_RIGHT_DIFF_HARD_MAX:
        geometry_valid = False
        penalty += 0.22

    if aspect_ratio_deviation > CARD_ASPECT_DEVIATION_SOFT_MAX:
        penalty += 0.18
        reasons.append("aspect_ratio_off")
    if aspect_ratio_deviation > CARD_ASPECT_DEVIATION_HARD_MAX:
        geometry_valid = False
        penalty += 0.24

    if area_plausibility_score < 0.40:
        penalty += 0.15
        reasons.append("implausible_area")
    if area_plausibility_score <= 0.05:
        geometry_valid = False
        penalty += 0.20

    if max(bbox_width_ratio, bbox_height_ratio) > 0.96 and area_ratio > 0.78:
        penalty += 0.16
        reasons.append("bbox_too_large")
    if max(bbox_width_ratio, bbox_height_ratio) > 0.985 and area_ratio > 0.84:
        geometry_valid = False
        penalty += 0.18

    if min(evidence_width_ratio, evidence_height_ratio) < CARD_EVIDENCE_COVERAGE_SOFT_MIN:
        penalty += 0.22
        reasons.append("weak_evidence_coverage")
    if min(evidence_width_ratio, evidence_height_ratio) < CARD_EVIDENCE_COVERAGE_HARD_MIN:
        geometry_valid = False
        penalty += 0.26

    if evidence_coverage_score < 0.72:
        penalty += 0.12
        reasons.append("oversized_vs_evidence")
    if evidence_coverage_score < 0.56:
        geometry_valid = False
        penalty += 0.18

    if min_side_support < CARD_SIDE_SUPPORT_MIN_BOTTLENECK:
        penalty += 0.28
        reasons.append("low_min_side_support")
    if min_side_support < CARD_SIDE_SUPPORT_HARD_MIN:
        geometry_valid = False
        penalty += 0.24

    if side_support_std > CARD_SIDE_SUPPORT_STD_SOFT_MAX:
        penalty += 0.14
        reasons.append("unbalanced_side_support")
    if side_support_std > CARD_SIDE_SUPPORT_STD_HARD_MAX and min_side_support < 0.18:
        geometry_valid = False
        penalty += 0.18

    penalty = float(np.clip(penalty, 0.0, 1.0))
    return {
        "geometry_valid": bool(geometry_valid),
        "geometry_penalty": penalty,
        "geometry_reasons": reasons,
        "side_length_consistency_score": side_length_consistency_score,
        "aspect_ratio_deviation": float(aspect_ratio_deviation),
        "area_plausibility_score": float(area_plausibility_score),
        "top_length": float(top),
        "right_length": float(right),
        "bottom_length": float(bottom),
        "left_length": float(left),
        "top_bottom_diff_ratio": float(top_bottom_diff),
        "left_right_diff_ratio": float(left_right_diff),
        "top_bottom_length_ratio": float(top_bottom_ratio),
        "left_right_length_ratio": float(left_right_ratio),
        "aspect_ratio": float(aspect_ratio),
        "area_ratio": float(area_ratio),
    }


def _side_normal(side_start: np.ndarray, side_end: np.ndarray, center: np.ndarray) -> np.ndarray:
    direction = side_end - side_start
    norm = np.linalg.norm(direction)
    if norm <= 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)
    direction = direction / norm
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    midpoint = (side_start + side_end) / 2.0
    if np.dot(center - midpoint, normal) < 0:
        normal = -normal
    return normal


def _sample_side_support(
    structural_mask: np.ndarray,
    side_start: np.ndarray,
    side_end: np.ndarray,
    center: np.ndarray,
) -> Tuple[float, np.ndarray]:
    if structural_mask is None or structural_mask.size == 0:
        return 0.0, np.empty((0, 2), dtype=np.float32)

    h, w = structural_mask.shape[:2]
    length = float(np.linalg.norm(side_end - side_start))
    samples = max(28, int(round(length / 4.0)))
    band_half_width = max(3, int(round(min(h, w) * CARD_BORDER_SUPPORT_BAND_WIDTH)))
    inward_normal = _side_normal(side_start, side_end, center)
    near_outer = max(2, int(round(band_half_width * 0.45)))
    near_inner = max(3, int(round(band_half_width * 0.65)))
    deep_inner_start = max(near_inner + 1, int(round(band_half_width * 0.55)))
    near_offsets = np.arange(-near_outer, near_inner + 1, dtype=np.float32)
    deep_offsets = np.arange(deep_inner_start, band_half_width + 1, dtype=np.float32)
    side_scores: List[float] = []
    band_points: List[np.ndarray] = []

    for t in np.linspace(0.03, 0.97, num=samples):
        point = side_start + (side_end - side_start) * float(t)
        near_best = 0.0
        near_point: Optional[np.ndarray] = None
        for offset in near_offsets:
            sample_point = point + inward_normal * offset
            x = int(np.clip(round(float(sample_point[0])), 0, w - 1))
            y = int(np.clip(round(float(sample_point[1])), 0, h - 1))
            if structural_mask[y, x] > 0:
                if offset < 0:
                    distance_score = 1.0 - min(1.0, abs(float(offset)) / max(1.0, near_outer))
                    candidate_score = 0.75 * distance_score
                else:
                    distance_score = 1.0 - min(1.0, float(offset) / max(1.0, near_inner))
                    candidate_score = distance_score
                if candidate_score > near_best:
                    near_best = float(candidate_score)
                    near_point = np.array([x, y], dtype=np.float32)

        deep_hits = 0.0
        deep_total = 0.0
        for offset in deep_offsets:
            sample_point = point + inward_normal * offset
            x = int(np.clip(round(float(sample_point[0])), 0, w - 1))
            y = int(np.clip(round(float(sample_point[1])), 0, h - 1))
            deep_total += 1.0
            if structural_mask[y, x] > 0:
                deep_hits += 1.0

        deep_penalty = (deep_hits / deep_total) if deep_total > 0 else 0.0
        side_score = float(np.clip(near_best * (1.0 - 0.72 * deep_penalty), 0.0, 1.0))
        side_scores.append(side_score)
        if near_point is not None and side_score >= 0.25:
            band_points.append(near_point)

    if not side_scores:
        return 0.0, np.empty((0, 2), dtype=np.float32)

    hit_array = np.asarray(side_scores, dtype=np.float32)
    support_ratio = float(np.mean(hit_array))
    longest_run = 0
    current_run = 0
    for hit in hit_array:
        if hit >= 0.45:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    continuity = longest_run / float(len(hit_array))
    support_score = float(np.clip(0.68 * support_ratio + 0.32 * continuity, 0.0, 1.0))
    return support_score, np.asarray(band_points, dtype=np.float32)


def _silhouette_support_scores(
    quad: np.ndarray,
    structural_mask: Optional[np.ndarray],
    metric_prefix: str,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], float]:
    if (not CARD_BORDER_SUPPORT_ENABLED) or structural_mask is None or structural_mask.size == 0:
        zero_metrics = {
            f"{metric_prefix}_left_border_support": 0.0,
            f"{metric_prefix}_right_border_support": 0.0,
            f"{metric_prefix}_top_border_support": 0.0,
            f"{metric_prefix}_bottom_border_support": 0.0,
            f"{metric_prefix}_silhouette_support_score": 0.0,
        }
        return zero_metrics, {}, 0.0

    quad = order_points(quad)
    center = np.mean(quad, axis=0)
    sides = {
        "top": (quad[0], quad[1]),
        "right": (quad[1], quad[2]),
        "bottom": (quad[3], quad[2]),
        "left": (quad[0], quad[3]),
    }

    side_scores: Dict[str, float] = {}
    overlay_points: Dict[str, np.ndarray] = {}
    for side_name, (side_start, side_end) in sides.items():
        score, points = _sample_side_support(structural_mask, side_start, side_end, center)
        side_scores[side_name] = score
        overlay_points[side_name] = points

    vertical_avg = 0.5 * (side_scores["left"] + side_scores["right"])
    horizontal_avg = 0.5 * (side_scores["top"] + side_scores["bottom"])
    silhouette_score = 0.40 * side_scores["left"] + 0.40 * side_scores["right"]
    silhouette_score += 0.10 * side_scores["top"] + 0.10 * side_scores["bottom"]
    if vertical_avg < CARD_BORDER_SUPPORT_VERTICAL_MIN:
        silhouette_score *= 0.55
    if max(side_scores["top"], side_scores["bottom"]) < 0.08 and vertical_avg < 0.30:
        silhouette_score *= 0.75
    silhouette_score = float(np.clip(silhouette_score, 0.0, 1.0))

    metrics = {
        f"{metric_prefix}_left_border_support": float(side_scores["left"]),
        f"{metric_prefix}_right_border_support": float(side_scores["right"]),
        f"{metric_prefix}_top_border_support": float(side_scores["top"]),
        f"{metric_prefix}_bottom_border_support": float(side_scores["bottom"]),
        f"{metric_prefix}_silhouette_support_score": silhouette_score,
        f"{metric_prefix}_vertical_border_support": float(vertical_avg),
        f"{metric_prefix}_horizontal_border_support": float(horizontal_avg),
    }
    return metrics, overlay_points, silhouette_score


def _draw_border_support_overlay(
    image: np.ndarray,
    overlay_points: Dict[str, np.ndarray],
    quad: Optional[np.ndarray] = None,
) -> np.ndarray:
    overlay = image.copy()
    color_map = {
        "left": (0, 255, 255),
        "right": (255, 255, 0),
        "top": (255, 0, 255),
        "bottom": (0, 200, 255),
    }
    for side_name, points in overlay_points.items():
        color = color_map.get(side_name, (0, 255, 0))
        if points.size == 0:
            continue
        for point in points.astype(np.int32):
            cv2.circle(overlay, tuple(point), 1, color, -1)
    if quad is not None:
        cv2.polylines(overlay, [order_points(quad).astype(np.int32)], True, (0, 255, 0), 2)
    return overlay


def _sample_perimeter_color_transition(
    quad: np.ndarray,
    image_shape: Tuple[int, int],
    hsv_view: Optional[np.ndarray],
    lab_view: Optional[np.ndarray],
) -> Tuple[float, Dict[str, float]]:
    if (
        (not CARD_COLOR_SEPARATION_ENABLED)
        or hsv_view is None
        or lab_view is None
        or hsv_view.size == 0
        or lab_view.size == 0
    ):
        return 0.0, {
            "color_separation_score": 0.0,
            "color_luminance_diff": 0.0,
            "color_chroma_diff": 0.0,
            "color_saturation_diff": 0.0,
        }

    h, w = image_shape[:2]
    quad = order_points(quad)
    center = np.mean(quad, axis=0)
    min_dim = max(1.0, float(min(h, w)))
    sample_offset = max(2.0, CARD_COLOR_SAMPLE_OFFSET * min_dim)
    interior_samples: List[np.ndarray] = []
    exterior_samples: List[np.ndarray] = []

    sides = (
        (quad[0], quad[1]),
        (quad[1], quad[2]),
        (quad[3], quad[2]),
        (quad[0], quad[3]),
    )
    for side_start, side_end in sides:
        length = float(np.linalg.norm(side_end - side_start))
        samples = max(18, int(round(length / 6.0)))
        inward_normal = _side_normal(side_start, side_end, center)
        for t in np.linspace(0.08, 0.92, num=samples):
            point = side_start + (side_end - side_start) * float(t)
            inner = point + inward_normal * sample_offset
            outer = point - inward_normal * sample_offset

            ix = int(np.clip(round(float(inner[0])), 0, w - 1))
            iy = int(np.clip(round(float(inner[1])), 0, h - 1))
            ox = int(np.clip(round(float(outer[0])), 0, w - 1))
            oy = int(np.clip(round(float(outer[1])), 0, h - 1))

            interior_samples.append(
                np.array(
                    [
                        float(lab_view[iy, ix, 0]),
                        float(lab_view[iy, ix, 1]),
                        float(lab_view[iy, ix, 2]),
                        float(hsv_view[iy, ix, 1]),
                        float(hsv_view[iy, ix, 2]),
                    ],
                    dtype=np.float32,
                )
            )
            exterior_samples.append(
                np.array(
                    [
                        float(lab_view[oy, ox, 0]),
                        float(lab_view[oy, ox, 1]),
                        float(lab_view[oy, ox, 2]),
                        float(hsv_view[oy, ox, 1]),
                        float(hsv_view[oy, ox, 2]),
                    ],
                    dtype=np.float32,
                )
            )

    if not interior_samples or not exterior_samples:
        return 0.0, {
            "color_separation_score": 0.0,
            "color_luminance_diff": 0.0,
            "color_chroma_diff": 0.0,
            "color_saturation_diff": 0.0,
        }

    interior_mean = np.mean(np.vstack(interior_samples), axis=0)
    exterior_mean = np.mean(np.vstack(exterior_samples), axis=0)
    luminance_diff = abs(float(interior_mean[0] - exterior_mean[0])) / 255.0
    chroma_diff = float(
        np.linalg.norm(interior_mean[1:3] - exterior_mean[1:3]) / 181.0
    )
    saturation_diff = abs(float(interior_mean[3] - exterior_mean[3])) / 255.0
    value_diff = abs(float(interior_mean[4] - exterior_mean[4])) / 255.0
    color_score = float(
        np.clip(
            0.42 * luminance_diff
            + 0.28 * chroma_diff
            + 0.15 * saturation_diff
            + 0.15 * value_diff,
            0.0,
            1.0,
        )
    )
    return color_score, {
        "color_separation_score": color_score,
        "color_luminance_diff": float(luminance_diff),
        "color_chroma_diff": float(chroma_diff),
        "color_saturation_diff": float(saturation_diff),
        "color_value_diff": float(value_diff),
    }


def _size_plausibility_score(
    quad: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[float, Dict[str, float]]:
    img_h, img_w = image_shape[:2]
    quad = order_points(quad)
    area_ratio = polygon_area(quad) / float(max(1, img_h * img_w))
    width_ratio = (quad[:, 0].max() - quad[:, 0].min()) / float(max(1, img_w))
    height_ratio = (quad[:, 1].max() - quad[:, 1].min()) / float(max(1, img_h))

    area_component = min(1.0, area_ratio / max(CARD_MIN_AREA_RATIO, 1e-6))
    width_component = min(1.0, width_ratio / max(CARD_MIN_WIDTH_RATIO, 1e-6))
    height_component = min(1.0, height_ratio / max(CARD_MIN_HEIGHT_RATIO, 1e-6))
    size_score = float(
        np.clip(
            0.40 * area_component + 0.25 * width_component + 0.35 * height_component,
            0.0,
            1.0,
        )
    )
    return size_score, {
        "size_plausibility_score": size_score,
        "bbox_width_ratio": float(width_ratio),
        "bbox_height_ratio": float(height_ratio),
    }


def _quad_angles_deg(quad: np.ndarray) -> np.ndarray:
    quad = order_points(quad)
    angles: List[float] = []
    for idx in range(4):
        prev_pt = quad[(idx - 1) % 4] - quad[idx]
        next_pt = quad[(idx + 1) % 4] - quad[idx]
        denom = np.linalg.norm(prev_pt) * np.linalg.norm(next_pt)
        if denom <= 1e-6:
            angles.append(0.0)
            continue
        cosine = float(np.clip(np.dot(prev_pt, next_pt) / denom, -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cosine))))
    return np.asarray(angles, dtype=np.float32)


def _line_angle_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    delta = p2 - p1
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def _angle_score(quad: np.ndarray) -> Tuple[float, float]:
    angles = _quad_angles_deg(quad)
    mean_deviation = float(np.mean(np.abs(angles - 90.0)))
    score = max(0.0, 1.0 - (mean_deviation / 25.0))
    return float(score), mean_deviation


def _parallelism_score(quad: np.ndarray) -> Tuple[float, float]:
    quad = order_points(quad)
    top_angle = _line_angle_deg(quad[0], quad[1])
    bottom_angle = _line_angle_deg(quad[3], quad[2])
    left_angle = _line_angle_deg(quad[0], quad[3])
    right_angle = _line_angle_deg(quad[1], quad[2])

    horiz_diff = abs(((top_angle - bottom_angle + 90.0) % 180.0) - 90.0)
    vert_diff = abs(((left_angle - right_angle + 90.0) % 180.0) - 90.0)
    mean_diff = float((horiz_diff + vert_diff) / 2.0)
    score = max(0.0, 1.0 - (mean_diff / 18.0))
    return float(score), mean_diff


def _opposite_side_similarity_score(quad: np.ndarray) -> Tuple[float, float, float]:
    top, right, bottom, left = _quad_side_lengths(quad)
    width_diff = abs(top - bottom) / max(1.0, max(top, bottom))
    height_diff = abs(left - right) / max(1.0, max(left, right))
    mean_diff = float((width_diff + height_diff) / 2.0)
    score = max(0.0, 1.0 - (mean_diff / 0.28))
    return float(score), float(width_diff), float(height_diff)


def _trapezoid_penalty(quad: np.ndarray) -> float:
    _, width_diff, height_diff = _opposite_side_similarity_score(quad)
    penalty = min(1.0, (0.7 * width_diff + 0.3 * height_diff) / 0.35)
    return float(penalty)


def _rectangularity_score(quad: np.ndarray) -> Tuple[float, Dict[str, float]]:
    angle_score, angle_deviation = _angle_score(quad)
    parallel_score, parallel_diff = _parallelism_score(quad)
    side_similarity, width_diff, height_diff = _opposite_side_similarity_score(quad)
    trapezoid_penalty = _trapezoid_penalty(quad)
    rectangularity = max(
        0.0,
        min(
            1.0,
            0.42 * angle_score
            + 0.33 * parallel_score
            + 0.25 * side_similarity
            - 0.18 * trapezoid_penalty,
        ),
    )
    return float(rectangularity), {
        "angle_score": float(angle_score),
        "angle_deviation": float(angle_deviation),
        "parallelism_score": float(parallel_score),
        "parallelism_diff": float(parallel_diff),
        "opposite_side_similarity_score": float(side_similarity),
        "top_bottom_diff_ratio": float(width_diff),
        "left_right_diff_ratio": float(height_diff),
        "trapezoid_penalty": float(trapezoid_penalty),
    }


def _bbox_fit_score(
    quad: np.ndarray,
    bbox_xyxy: Optional[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
) -> float:
    if bbox_xyxy is None:
        return 1.0

    x1, y1, x2, y2 = bbox_xyxy
    quad = order_points(quad)
    center = np.mean(quad, axis=0)
    inside = 1.0 if (x1 <= center[0] <= x2 and y1 <= center[1] <= y2) else 0.0

    bbox_area = float(max(1, (x2 - x1) * (y2 - y1)))
    quad_area = polygon_area(quad)
    area_ratio = min(quad_area, bbox_area) / max(quad_area, bbox_area)

    h, w = image_shape[:2]
    bbox_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    max_dist = np.linalg.norm(np.array([w / 2.0, h / 2.0], dtype=np.float32))
    center_score = 1.0 - (
        np.linalg.norm(center - bbox_center) / max(max_dist, 1.0)
    )
    center_score = float(np.clip(center_score, 0.0, 1.0))
    return float(
        np.clip(0.45 * inside + 0.35 * area_ratio + 0.20 * center_score, 0.0, 1.0)
    )


def _rank_candidate(
    quad: np.ndarray,
    image_shape: Tuple[int, int],
    edge_map: Optional[np.ndarray] = None,
    structural_mask: Optional[np.ndarray] = None,
    binary_view: Optional[np.ndarray] = None,
    hsv_view: Optional[np.ndarray] = None,
    lab_view: Optional[np.ndarray] = None,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    expected_outer_bbox: Optional[Tuple[int, int, int, int]] = None,
    strictness: float = CARD_GEOMETRY_STRICTNESS,
    metric_prefix: str = "candidate",
) -> Tuple[float, Dict[str, float]]:
    img_h, img_w = image_shape[:2]
    quad = order_points(quad)

    poly_area = polygon_area(quad)
    image_area = float(max(1, img_h * img_w))
    area_ratio = poly_area / image_area

    touch_count = _candidate_touch_count(quad, image_shape)
    aspect_ratio = _aspect_ratio_of_quad(quad)
    aspect_score = _aspect_validity_score(aspect_ratio)
    center_score = _center_distance_score(quad, image_shape)
    outer_score = _outerness_score(quad, image_shape, roi_bbox=roi_bbox)
    edge_support = _edge_support_score(quad, edge_map)
    convex_score = 1.0 if cv2.isContourConvex(quad.astype(np.int32)) else 0.0

    rectangularity_score, rect_metrics = _rectangularity_score(quad)
    bbox_fit = _bbox_fit_score(quad, roi_bbox, image_shape)

    size_plausibility_score, size_metrics = _size_plausibility_score(
        quad,
        image_shape,
    )

    color_separation_score, color_metrics = _sample_perimeter_color_transition(
        quad,
        image_shape=image_shape,
        hsv_view=hsv_view,
        lab_view=lab_view,
    )

    silhouette_metrics, _, silhouette_score = _silhouette_support_scores(
        quad,
        structural_mask=structural_mask,
        metric_prefix=metric_prefix,
    )

    left_support = silhouette_metrics.get(f"{metric_prefix}_left_border_support", 0.0)
    right_support = silhouette_metrics.get(f"{metric_prefix}_right_border_support", 0.0)
    top_support = silhouette_metrics.get(f"{metric_prefix}_top_border_support", 0.0)
    bottom_support = silhouette_metrics.get(f"{metric_prefix}_bottom_border_support", 0.0)

    width_ratio = float(size_metrics.get("bbox_width_ratio", 0.0))
    height_ratio = float(size_metrics.get("bbox_height_ratio", 0.0))
    side_support_stats = _collect_side_support_stats(
        {
            "left_border_support": left_support,
            "right_border_support": right_support,
            "top_border_support": top_support,
            "bottom_border_support": bottom_support,
        }
    )
    min_side_support = float(side_support_stats["min_side_support"])
    max_side_support = float(side_support_stats["max_side_support"])
    side_support_std = float(side_support_stats["side_support_std"])
    evidence_metrics = _compute_evidence_coverage_metrics(
        quad,
        image_shape=image_shape,
        structural_mask=structural_mask,
        binary_view=binary_view,
    )
    geometry_metrics = validate_candidate_geometry(
        quad,
        image_shape=image_shape,
        metrics={
            "bbox_width_ratio": width_ratio,
            "bbox_height_ratio": height_ratio,
            "left_border_support": left_support,
            "right_border_support": right_support,
            "top_border_support": top_support,
            "bottom_border_support": bottom_support,
            **side_support_stats,
            **evidence_metrics,
        },
    )
    outer_bbox_metrics = _candidate_outer_bbox_metrics(
        quad,
        outer_bbox=expected_outer_bbox,
        image_shape=image_shape,
        metric_prefix=metric_prefix,
    )
    candidate_is_internal = float(outer_bbox_metrics.get(f"{metric_prefix}_is_internal", 0.0))
    candidate_border_distance = float(
        outer_bbox_metrics.get(f"{metric_prefix}_distance_to_outer_border", 1.0)
    )
    candidate_relative_size = float(
        outer_bbox_metrics.get(f"{metric_prefix}_relative_size_to_card_bbox", 0.0)
    )
    candidate_touches_outer_region = float(
        outer_bbox_metrics.get(f"{metric_prefix}_touches_outer_region", 0.0)
    )

    # =========================================================
    # HARD RULES: si falla aqui, el candidato NO compite
    # =========================================================
    hard_rules_ok = True
    hard_fail_reasons = []

    if area_ratio < CARD_MIN_AREA_RATIO:
        hard_rules_ok = False
        hard_fail_reasons.append("area_too_small")

    if width_ratio < CARD_MIN_WIDTH_RATIO:
        hard_rules_ok = False
        hard_fail_reasons.append("width_too_small")

    if height_ratio < CARD_MIN_HEIGHT_RATIO:
        hard_rules_ok = False
        hard_fail_reasons.append("height_too_small")

    if max(left_support, right_support) < CARD_BORDER_SUPPORT_VERTICAL_MIN:
        hard_rules_ok = False
        hard_fail_reasons.append("low_vertical_support")

    if silhouette_score < 0.55:  # Sube el umbral manualmente aquí para ser estricto
        hard_rules_ok = False
        hard_fail_reasons.append("low_silhouette_support")

    if min(left_support + right_support, top_support + bottom_support) < 0.40:
        hard_rules_ok = False
        hard_fail_reasons.append("asymmetric_border_support")

    if min_side_support < CARD_SIDE_SUPPORT_HARD_MIN:
        hard_rules_ok = False
        hard_fail_reasons.append("weak_side_bottleneck")

    if not bool(geometry_metrics["geometry_valid"]):
        hard_rules_ok = False
        hard_fail_reasons.append("invalid_geometry")

    if rect_metrics["trapezoid_penalty"] > 0.50:
        hard_rules_ok = False
        hard_fail_reasons.append("strong_trapezoid")

    if rect_metrics["angle_deviation"] > (15.0 * strictness):
        hard_rules_ok = False
        hard_fail_reasons.append("bad_angles")

    if touch_count >= 3:
        hard_rules_ok = False
        hard_fail_reasons.append("touches_too_many_borders")

    if candidate_is_internal >= 0.5:
        hard_rules_ok = False
        hard_fail_reasons.append("internal_candidate")

    if expected_outer_bbox is not None and candidate_relative_size < 0.35:
        hard_rules_ok = False
        hard_fail_reasons.append("too_small_vs_outer_bbox")

    # =========================================================
    # AREA SCORE MAS ESTRICTO
    # Evita que gane un quad enorme del piso o uno diminuto
    # =========================================================
    if area_ratio < CARD_MIN_AREA_RATIO:
        area_score = 0.0
    elif area_ratio < 0.20:
        area_score = 0.35
    elif area_ratio < 0.28:
        area_score = 0.65
    elif area_ratio < 0.55:
        area_score = 1.0
    elif area_ratio < 0.75:
        area_score = 0.55
    else:
        area_score = 0.10

    # =========================================================
    # SI FALLA HARD RULES, devolver score 0
    # =========================================================
    if not hard_rules_ok:
        metrics = {
            f"{metric_prefix}_area_ratio": float(area_ratio),
            f"{metric_prefix}_aspect_ratio": float(aspect_ratio),
            f"{metric_prefix}_area_score": float(area_score),
            f"{metric_prefix}_outer_score": float(outer_score),
            f"{metric_prefix}_touch_count": float(touch_count),
            f"{metric_prefix}_edge_support": float(edge_support),
            f"{metric_prefix}_aspect_score": float(aspect_score),
            f"{metric_prefix}_center_score": float(center_score),
            f"{metric_prefix}_convex_score": float(convex_score),
            f"{metric_prefix}_bbox_fit_score": float(bbox_fit),
            f"{metric_prefix}_rectangularity_score": float(rectangularity_score),
            f"{metric_prefix}_angle_score": float(rect_metrics["angle_score"]),
            f"{metric_prefix}_angle_deviation": float(rect_metrics["angle_deviation"]),
            f"{metric_prefix}_parallelism_score": float(rect_metrics["parallelism_score"]),
            f"{metric_prefix}_parallelism_diff": float(rect_metrics["parallelism_diff"]),
            f"{metric_prefix}_opposite_side_similarity_score": float(
                rect_metrics["opposite_side_similarity_score"]
            ),
            f"{metric_prefix}_top_bottom_diff_ratio": float(
                rect_metrics["top_bottom_diff_ratio"]
            ),
            f"{metric_prefix}_left_right_diff_ratio": float(
                rect_metrics["left_right_diff_ratio"]
            ),
            f"{metric_prefix}_trapezoid_penalty": float(rect_metrics["trapezoid_penalty"]),
            f"{metric_prefix}_color_separation_score": float(color_separation_score),
            f"{metric_prefix}_color_luminance_diff": float(
                color_metrics["color_luminance_diff"]
            ),
            f"{metric_prefix}_color_chroma_diff": float(color_metrics["color_chroma_diff"]),
            f"{metric_prefix}_color_saturation_diff": float(
                color_metrics["color_saturation_diff"]
            ),
            f"{metric_prefix}_color_value_diff": float(color_metrics["color_value_diff"]),
            f"{metric_prefix}_size_plausibility_score": float(size_plausibility_score),
            f"{metric_prefix}_bbox_width_ratio": float(width_ratio),
            f"{metric_prefix}_bbox_height_ratio": float(height_ratio),
            f"{metric_prefix}_geometry_penalty": float(geometry_metrics["geometry_penalty"]),
            f"{metric_prefix}_geometry_valid": float(1.0 if geometry_metrics["geometry_valid"] else 0.0),
            f"{metric_prefix}_side_length_consistency_score": float(
                geometry_metrics["side_length_consistency_score"]
            ),
            f"{metric_prefix}_aspect_ratio_deviation": float(
                geometry_metrics["aspect_ratio_deviation"]
            ),
            f"{metric_prefix}_area_plausibility_score": float(
                geometry_metrics["area_plausibility_score"]
            ),
            f"{metric_prefix}_top_bottom_length_ratio": float(
                geometry_metrics["top_bottom_length_ratio"]
            ),
            f"{metric_prefix}_left_right_length_ratio": float(
                geometry_metrics["left_right_length_ratio"]
            ),
            f"{metric_prefix}_min_side_support": float(min_side_support),
            f"{metric_prefix}_max_side_support": float(max_side_support),
            f"{metric_prefix}_side_support_std": float(side_support_std),
            f"{metric_prefix}_evidence_bbox_width_ratio": float(
                evidence_metrics["evidence_bbox_width_ratio"]
            ),
            f"{metric_prefix}_evidence_bbox_height_ratio": float(
                evidence_metrics["evidence_bbox_height_ratio"]
            ),
            f"{metric_prefix}_evidence_bbox_area_ratio": float(
                evidence_metrics["evidence_bbox_area_ratio"]
            ),
            f"{metric_prefix}_evidence_coverage_score": float(
                evidence_metrics["evidence_coverage_score"]
            ),
            f"{metric_prefix}_hard_rules_ok": 0.0,
            f"{metric_prefix}_hard_fail_count": float(len(hard_fail_reasons)),
            f"{metric_prefix}_rejection_reason": hard_fail_reasons[0] if hard_fail_reasons else "",
            f"{metric_prefix}_total_score": 0.0,
        }
        metrics.update(outer_bbox_metrics)
        metrics.update(silhouette_metrics)
        return 0.0, metrics

    # =========================================================
    # PENALIZACIONES SUAVES
    # =========================================================
    penalty = 1.0

    if area_ratio > 0.98:
        penalty *= 0.05

    if rect_metrics["trapezoid_penalty"] > 0.55:
        penalty *= 0.72

    if rect_metrics["angle_deviation"] > (18.0 * strictness):
        penalty *= 0.85

    if size_plausibility_score < 0.92:
        penalty *= 0.88

    if top_support < 0.08 and bottom_support < 0.08 and max(left_support, right_support) < 0.35:
        penalty *= 0.78

    geometry_penalty = float(geometry_metrics["geometry_penalty"])
    if geometry_penalty > 0.0:
        penalty *= max(0.02, 1.0 - geometry_penalty)

    bottleneck_factor = float(np.clip(min_side_support / CARD_SIDE_SUPPORT_MIN_BOTTLENECK, 0.0, 1.0))
    if min_side_support >= CARD_SIDE_SUPPORT_MIN_BOTTLENECK:
        bottleneck_factor = float(
            np.clip(0.78 + 0.22 * min(1.0, min_side_support / 0.35), 0.0, 1.0)
        )
    penalty *= bottleneck_factor

    if side_support_std > CARD_SIDE_SUPPORT_STD_SOFT_MAX:
        penalty *= 0.88

    if float(evidence_metrics["evidence_coverage_score"]) < 0.72:
        penalty *= 0.82

    if expected_outer_bbox is not None:
        if candidate_touches_outer_region < 0.5:
            penalty *= 0.70
        penalty *= float(np.clip(1.05 - candidate_border_distance, 0.10, 1.0))
        penalty *= float(np.clip(candidate_relative_size / 0.82, 0.12, 1.0))

    # =========================================================
    # SCORE FINAL
    # Más peso a tamano plausible, menos a outerness
    # =========================================================
    total_score = (
        0.06 * area_score
        + 0.05 * outer_score
        + 0.08 * aspect_score
        + 0.03 * center_score
        + 0.04 * convex_score
        + 0.22 * edge_support
        + 0.18 * rectangularity_score
        + 0.10 * bbox_fit
        + 0.14 * size_plausibility_score
        + 0.22 * silhouette_score
        + 0.04 * color_separation_score
        + 0.12 * candidate_touches_outer_region
        + 0.10 * min(1.0, candidate_relative_size)
    ) * penalty

    total_score = float(np.clip(total_score, 0.0, 1.0))

    metrics = {
        f"{metric_prefix}_area_ratio": float(area_ratio),
        f"{metric_prefix}_aspect_ratio": float(aspect_ratio),
        f"{metric_prefix}_area_score": float(area_score),
        f"{metric_prefix}_outer_score": float(outer_score),
        f"{metric_prefix}_touch_count": float(touch_count),
        f"{metric_prefix}_edge_support": float(edge_support),
        f"{metric_prefix}_aspect_score": float(aspect_score),
        f"{metric_prefix}_center_score": float(center_score),
        f"{metric_prefix}_convex_score": float(convex_score),
        f"{metric_prefix}_bbox_fit_score": float(bbox_fit),
        f"{metric_prefix}_rectangularity_score": float(rectangularity_score),
        f"{metric_prefix}_angle_score": float(rect_metrics["angle_score"]),
        f"{metric_prefix}_angle_deviation": float(rect_metrics["angle_deviation"]),
        f"{metric_prefix}_parallelism_score": float(rect_metrics["parallelism_score"]),
        f"{metric_prefix}_parallelism_diff": float(rect_metrics["parallelism_diff"]),
        f"{metric_prefix}_opposite_side_similarity_score": float(
            rect_metrics["opposite_side_similarity_score"]
        ),
        f"{metric_prefix}_top_bottom_diff_ratio": float(rect_metrics["top_bottom_diff_ratio"]),
        f"{metric_prefix}_left_right_diff_ratio": float(rect_metrics["left_right_diff_ratio"]),
        f"{metric_prefix}_trapezoid_penalty": float(rect_metrics["trapezoid_penalty"]),
        f"{metric_prefix}_color_separation_score": float(color_separation_score),
        f"{metric_prefix}_color_luminance_diff": float(color_metrics["color_luminance_diff"]),
        f"{metric_prefix}_color_chroma_diff": float(color_metrics["color_chroma_diff"]),
        f"{metric_prefix}_color_saturation_diff": float(color_metrics["color_saturation_diff"]),
        f"{metric_prefix}_color_value_diff": float(color_metrics["color_value_diff"]),
        f"{metric_prefix}_size_plausibility_score": float(size_plausibility_score),
        f"{metric_prefix}_bbox_width_ratio": float(width_ratio),
        f"{metric_prefix}_bbox_height_ratio": float(height_ratio),
        f"{metric_prefix}_geometry_penalty": float(geometry_penalty),
        f"{metric_prefix}_geometry_valid": float(1.0 if geometry_metrics["geometry_valid"] else 0.0),
        f"{metric_prefix}_side_length_consistency_score": float(
            geometry_metrics["side_length_consistency_score"]
        ),
        f"{metric_prefix}_aspect_ratio_deviation": float(
            geometry_metrics["aspect_ratio_deviation"]
        ),
        f"{metric_prefix}_area_plausibility_score": float(
            geometry_metrics["area_plausibility_score"]
        ),
        f"{metric_prefix}_top_bottom_length_ratio": float(
            geometry_metrics["top_bottom_length_ratio"]
        ),
        f"{metric_prefix}_left_right_length_ratio": float(
            geometry_metrics["left_right_length_ratio"]
        ),
        f"{metric_prefix}_min_side_support": float(min_side_support),
        f"{metric_prefix}_max_side_support": float(max_side_support),
        f"{metric_prefix}_side_support_std": float(side_support_std),
        f"{metric_prefix}_evidence_bbox_width_ratio": float(
            evidence_metrics["evidence_bbox_width_ratio"]
        ),
        f"{metric_prefix}_evidence_bbox_height_ratio": float(
            evidence_metrics["evidence_bbox_height_ratio"]
        ),
        f"{metric_prefix}_evidence_bbox_area_ratio": float(
            evidence_metrics["evidence_bbox_area_ratio"]
        ),
        f"{metric_prefix}_evidence_coverage_score": float(
            evidence_metrics["evidence_coverage_score"]
        ),
        f"{metric_prefix}_hard_rules_ok": 1.0,
        f"{metric_prefix}_hard_fail_count": 0.0,
        f"{metric_prefix}_rejection_reason": "",
        f"{metric_prefix}_total_score": float(total_score),
    }
    metrics.update(outer_bbox_metrics)
    metrics.update(silhouette_metrics)

    return total_score, metrics

def _build_conservative_fallback_quad(
    image_shape: Tuple[int, int],
    estimated_outer_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, str, float]:
    image_area = float(max(1, image_shape[0] * image_shape[1]))
    if estimated_outer_bbox is not None:
        bbox_area_ratio = _bbox_area(estimated_outer_bbox) / image_area
        if 0.10 <= bbox_area_ratio <= 0.78:
            return _quad_from_bbox_xyxy(estimated_outer_bbox), "estimated_outer_bbox", 0.0

    h, w = image_shape[:2]
    target_h = int(round(h * 0.68))
    target_w = int(round(target_h * CARD_TARGET_ASPECT_RATIO))
    if target_w > int(round(w * 0.78)):
        target_w = int(round(w * 0.78))
        target_h = int(round(target_w / CARD_TARGET_ASPECT_RATIO))

    cx = w // 2
    cy = h // 2
    x1 = max(0, int(round(cx - target_w / 2.0)))
    y1 = max(0, int(round(cy - target_h / 2.0)))
    x2 = min(w, int(round(cx + target_w / 2.0)))
    y2 = min(h, int(round(cy + target_h / 2.0)))
    quad = _quad_from_bbox_xyxy((x1, y1, x2, y2))
    return quad, "centered_conservative", 1.0


def _is_plausible_outer_bbox(
    bbox_xyxy: Optional[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
) -> bool:
    if bbox_xyxy is None:
        return False
    x1, y1, x2, y2 = bbox_xyxy
    img_h, img_w = image_shape[:2]
    bbox_w = float(max(1, x2 - x1))
    bbox_h = float(max(1, y2 - y1))
    area_ratio = float((bbox_w * bbox_h) / max(1.0, img_h * img_w))
    width_ratio = bbox_w / max(1.0, img_w)
    height_ratio = bbox_h / max(1.0, img_h)
    aspect_ratio = bbox_w / max(1.0, bbox_h)
    aspect_deviation = abs(aspect_ratio - CARD_TARGET_ASPECT_RATIO)
    return bool(
        0.12 <= area_ratio <= 0.82
        and width_ratio >= max(0.38, CARD_MIN_WIDTH_RATIO)
        and height_ratio >= max(0.50, CARD_MIN_HEIGHT_RATIO)
        and aspect_deviation <= 0.12
    )


def _expand_bbox_xyxy(
    bbox_xyxy: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    margin_ratio: float = CARD_ROI_SECOND_PASS_MARGIN_RATIO,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    bbox_w = float(max(1, x2 - x1))
    bbox_h = float(max(1, y2 - y1))
    mx = bbox_w * margin_ratio
    my = bbox_h * margin_ratio
    return _clip_bbox_xyxy(
        (
            int(round(x1 - mx)),
            int(round(y1 - my)),
            int(round(x2 + mx)),
            int(round(y2 + my)),
        ),
        image_shape,
    )


def _translate_quad(quad: np.ndarray, offset_xy: Tuple[int, int]) -> np.ndarray:
    offset = np.array(offset_xy, dtype=np.float32)
    return order_points(np.asarray(quad, dtype=np.float32) + offset)


def _shift_bbox_to_local(
    bbox_xyxy: Tuple[int, int, int, int],
    roi_bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    rx1, ry1, _, _ = roi_bbox
    return _clip_bbox_xyxy((x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1), image_shape)


def _coarse_seed_bbox(
    detected_quad: Optional[np.ndarray],
    estimated_outer_bbox: Optional[Tuple[int, int, int, int]],
    projection_bbox: Optional[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    if detected_quad is not None:
        return _candidate_bbox_xyxy(detected_quad, image_shape), "detected_quad"
    if estimated_outer_bbox is not None:
        return estimated_outer_bbox, "estimated_outer_bbox"
    if projection_bbox is not None:
        return projection_bbox, "projection_bbox"
    return None, ""


def _seed_mask_from_bbox(
    bbox_xyxy: Optional[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if bbox_xyxy is None:
        return mask
    x1, y1, x2, y2 = _clip_bbox_xyxy(bbox_xyxy, image_shape)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def _strengthen_outer_mask_with_seed(
    outer_mask: np.ndarray,
    seed_bbox_xyxy: Optional[Tuple[int, int, int, int]],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    h, w = outer_mask.shape[:2]
    if outer_mask.size == 0:
        zero_mask = np.zeros((max(1, h), max(1, w)), dtype=np.uint8)
        return zero_mask, {}, {}

    seed_mask = _seed_mask_from_bbox(seed_bbox_xyxy, outer_mask.shape[:2])
    min_dim = max(1, min(h, w))
    border_band = max(3, int(round(min_dim * 0.05)))
    border_touch_mask = np.zeros_like(outer_mask)
    border_touch_mask[:border_band, :] = 255
    border_touch_mask[-border_band:, :] = 255
    border_touch_mask[:, :border_band] = 255
    border_touch_mask[:, -border_band:] = 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        outer_mask, connectivity=8
    )
    background_suppressed = np.zeros_like(outer_mask)
    component_selection_overlay = cv2.cvtColor(outer_mask, cv2.COLOR_GRAY2BGR)
    main_component_mask = np.zeros_like(outer_mask)
    kept_labels: List[int] = []
    best_label = -1
    best_score = -1.0
    seed_area = float(max(1, np.count_nonzero(seed_mask)))
    border_touch_rejected = 0.0
    best_seed_overlap = 0.0
    best_component_area_ratio = 0.0

    for label in range(1, num_labels):
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        bbox_xyxy = _clip_bbox_xyxy((x, y, x + bw, y + bh), outer_mask.shape[:2])
        component_border_area = float(
            np.count_nonzero(cv2.bitwise_and(component_mask, border_touch_mask))
        )
        border_touch_ratio = component_border_area / max(1.0, area)
        seed_intersection = float(
            np.count_nonzero(cv2.bitwise_and(component_mask, seed_mask))
        )
        seed_overlap = seed_intersection / max(1.0, seed_area)
        component_seed_ratio = seed_intersection / max(1.0, area)
        bbox_center = _bbox_center(bbox_xyxy)
        if seed_bbox_xyxy is not None:
            seed_center = _bbox_center(seed_bbox_xyxy)
            center_dist = float(np.linalg.norm(bbox_center - seed_center))
            center_score = 1.0 - (
                center_dist / max(1.0, float(np.linalg.norm(np.array([w, h], dtype=np.float32))))
            )
        else:
            center_score = 0.5
        center_score = float(np.clip(center_score, 0.0, 1.0))

        reject_border_component = bool(
            border_touch_ratio >= 0.12 and seed_overlap < 0.02 and component_seed_ratio < 0.08
        )
        color = (0, 0, 255) if reject_border_component else (0, 255, 255)
        cv2.rectangle(
            component_selection_overlay,
            (bbox_xyxy[0], bbox_xyxy[1]),
            (bbox_xyxy[2], bbox_xyxy[3]),
            color,
            2,
        )
        if reject_border_component:
            border_touch_rejected += 1.0
            continue

        cv2.bitwise_or(background_suppressed, component_mask, dst=background_suppressed)
        kept_labels.append(label)
        selection_score = (
            0.60 * seed_overlap
            + 0.25 * component_seed_ratio
            + 0.15 * center_score
        )
        if selection_score > best_score:
            best_score = float(selection_score)
            best_label = label
            best_seed_overlap = float(seed_overlap)
            best_component_area_ratio = float(area / max(1.0, h * w))

    if best_label > 0:
        main_component_mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    elif kept_labels:
        main_component_mask = np.where(labels == kept_labels[0], 255, 0).astype(np.uint8)
        best_component_area_ratio = float(
            np.count_nonzero(main_component_mask) / max(1.0, h * w)
        )
    else:
        main_component_mask = background_suppressed.copy()

    final_mask = main_component_mask.copy()
    if np.count_nonzero(final_mask) == 0:
        final_mask = background_suppressed.copy()
    if np.count_nonzero(final_mask) == 0:
        final_mask = outer_mask.copy()

    seed_overlap_overlay = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    if seed_bbox_xyxy is not None:
        cv2.rectangle(
            seed_overlap_overlay,
            (seed_bbox_xyxy[0], seed_bbox_xyxy[1]),
            (seed_bbox_xyxy[2], seed_bbox_xyxy[3]),
            (255, 180, 0),
            2,
        )
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        best_contour = max(contours, key=cv2.contourArea)
        quad = _rect_from_min_area(best_contour)
        cv2.polylines(seed_overlap_overlay, [order_points(quad).astype(np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(component_selection_overlay, [order_points(quad).astype(np.int32)], True, (0, 255, 0), 2)

    debug_images = {
        "outer_candidate_background_suppressed": background_suppressed,
        "outer_candidate_border_touch_mask": border_touch_mask,
        "outer_candidate_seed_overlap_overlay": seed_overlap_overlay,
        "outer_candidate_main_component_mask": final_mask,
        "outer_candidate_component_selection_overlay": component_selection_overlay,
    }
    metrics = {
        "outer_candidate_border_touch_rejected": float(border_touch_rejected),
        "outer_candidate_seed_overlap_score": float(best_seed_overlap),
        "outer_candidate_main_component_area_ratio": float(best_component_area_ratio),
    }
    return final_mask, debug_images, metrics


def _cleanup_coarse_outer_mask(
    outer_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    h, w = outer_mask.shape[:2]
    if outer_mask.size == 0:
        zero_mask = np.zeros((max(1, h), max(1, w)), dtype=np.uint8)
        return zero_mask, {}, {}

    min_dim = max(1, min(h, w))
    border_band = max(3, int(round(min_dim * 0.05)))
    border_touch_mask = np.zeros_like(outer_mask)
    border_touch_mask[:border_band, :] = 255
    border_touch_mask[-border_band:, :] = 255
    border_touch_mask[:, :border_band] = 255
    border_touch_mask[:, -border_band:] = 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        outer_mask, connectivity=8
    )
    background_suppressed = np.zeros_like(outer_mask)
    component_selection_overlay = cv2.cvtColor(outer_mask, cv2.COLOR_GRAY2BGR)
    best_label = -1
    best_score = -1.0
    border_touch_rejected = 0.0
    kept_count = 0.0
    best_component_area_ratio = 0.0

    img_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    max_center_dist = float(max(1.0, np.linalg.norm(img_center)))

    for label in range(1, num_labels):
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        bbox_xyxy = _clip_bbox_xyxy((x, y, x + bw, y + bh), outer_mask.shape[:2])
        bbox_area_ratio = _bbox_area(bbox_xyxy) / float(max(1, h * w))
        border_touch_area = float(
            np.count_nonzero(cv2.bitwise_and(component_mask, border_touch_mask))
        )
        border_touch_ratio = border_touch_area / max(1.0, area)
        component_center = _bbox_center(bbox_xyxy)
        center_dist = float(np.linalg.norm(component_center - img_center))
        center_score = float(np.clip(1.0 - (center_dist / max_center_dist), 0.0, 1.0))
        plausible_bbox = _is_plausible_outer_bbox(bbox_xyxy, outer_mask.shape[:2])
        area_score = float(np.clip(1.0 - abs(bbox_area_ratio - 0.28) / 0.28, 0.0, 1.0))
        fill_ratio = area / max(1.0, _bbox_area(bbox_xyxy))
        fill_score = float(np.clip(fill_ratio / 0.85, 0.0, 1.0))

        reject_border_component = bool(
            border_touch_ratio >= 0.18
            and bbox_area_ratio >= 0.55
            and not plausible_bbox
        )
        color = (0, 0, 255) if reject_border_component else (0, 255, 255)
        cv2.rectangle(
            component_selection_overlay,
            (bbox_xyxy[0], bbox_xyxy[1]),
            (bbox_xyxy[2], bbox_xyxy[3]),
            color,
            2,
        )
        if reject_border_component:
            border_touch_rejected += 1.0
            continue

        kept_count += 1.0
        cv2.bitwise_or(background_suppressed, component_mask, dst=background_suppressed)
        plausible_bonus = 0.30 if plausible_bbox else 0.0
        selection_score = 0.34 * area_score + 0.26 * center_score + 0.20 * fill_score
        selection_score += plausible_bonus
        selection_score -= 0.18 * min(1.0, border_touch_ratio)
        if selection_score > best_score:
            best_score = float(selection_score)
            best_label = label
            best_component_area_ratio = float(area / max(1.0, h * w))

    main_component_mask = np.zeros_like(outer_mask)
    if best_label > 0:
        main_component_mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    elif np.count_nonzero(background_suppressed) > 0:
        main_component_mask = background_suppressed.copy()
        best_component_area_ratio = float(
            np.count_nonzero(main_component_mask) / max(1.0, h * w)
        )
    else:
        main_component_mask = outer_mask.copy()
        best_component_area_ratio = float(
            np.count_nonzero(main_component_mask) / max(1.0, h * w)
        )

    seed_candidate_overlay = cv2.cvtColor(main_component_mask, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(main_component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        best_contour = max(contours, key=cv2.contourArea)
        quad = _rect_from_min_area(best_contour)
        cv2.polylines(seed_candidate_overlay, [order_points(quad).astype(np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(component_selection_overlay, [order_points(quad).astype(np.int32)], True, (0, 255, 0), 2)

    debug_images = {
        "outer_candidate_background_suppressed": background_suppressed,
        "outer_candidate_border_touch_mask": border_touch_mask,
        "outer_candidate_main_component_mask": main_component_mask,
        "outer_candidate_component_selection_overlay": component_selection_overlay,
        "outer_candidate_seed_candidate_overlay": seed_candidate_overlay,
    }
    metrics = {
        "outer_candidate_border_touch_rejected": float(border_touch_rejected),
        "outer_candidate_main_component_area_ratio": float(best_component_area_ratio),
        "outer_candidate_component_count_after_cleanup": float(kept_count),
    }
    return main_component_mask, debug_images, metrics


def _attempt_binary_rescue_quad(
    binary_view: np.ndarray,
    image_shape: Tuple[int, int],
    candidate_mask: Optional[np.ndarray] = None,
    structural_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    attempted = True
    debug_images: Dict[str, np.ndarray] = {}
    metrics: Dict[str, float] = {
        "binary_rescue_attempted": 1.0,
        "binary_rescue_found": 0.0,
    }

    combined: Optional[np.ndarray] = None
    for source in (candidate_mask, structural_mask, binary_view):
        if source is None or source.size == 0:
            continue
        binary = source
        if binary.dtype != np.uint8:
            binary = np.clip(binary, 0, 255).astype(np.uint8)
        if binary.ndim != 2:
            continue
        if np.unique(binary).size > 2:
            _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        combined = binary.copy() if combined is None else cv2.bitwise_or(combined, binary)

    if combined is None or combined.size == 0:
        zero = np.zeros(image_shape[:2], dtype=np.uint8)
        debug_images["binary_rescue_component_mask"] = zero
        debug_images["binary_rescue_selection_overlay"] = cv2.cvtColor(zero, cv2.COLOR_GRAY2BGR)
        return {
            "quad": None,
            "bbox": None,
            "metrics": metrics,
            "debug_images": debug_images,
        }

    component_mask = _build_card_candidate_mask(combined)
    if structural_mask is not None and structural_mask.size > 0:
        structural = structural_mask
        if structural.dtype != np.uint8:
            structural = np.clip(structural, 0, 255).astype(np.uint8)
        if np.unique(structural).size > 2:
            _, structural = cv2.threshold(structural, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        component_mask = cv2.bitwise_or(component_mask, structural)
    component_mask = cv2.morphologyEx(
        component_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    selection_overlay = cv2.cvtColor(component_mask, cv2.COLOR_GRAY2BGR)
    best_quad: Optional[np.ndarray] = None
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_score = -1.0

    img_h, img_w = image_shape[:2]
    img_center = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float32)
    max_center_dist = float(max(1.0, np.linalg.norm(img_center)))

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        area = float(cv2.contourArea(contour))
        if area < CARD_OUTER_MIN_COMPONENT_AREA:
            continue
        quad = order_points(_rect_from_min_area(contour))
        bbox_xyxy = _candidate_bbox_xyxy(quad, image_shape)
        bbox_area_ratio = _bbox_area(bbox_xyxy) / float(max(1, img_h * img_w))
        plausible = _is_plausible_outer_bbox(bbox_xyxy, image_shape)
        center = _bbox_center(bbox_xyxy)
        center_score = float(
            np.clip(1.0 - (np.linalg.norm(center - img_center) / max_center_dist), 0.0, 1.0)
        )
        fill_ratio = area / max(1.0, _bbox_area(bbox_xyxy))
        fill_score = float(np.clip(fill_ratio / 0.90, 0.0, 1.0))
        area_score = float(np.clip(1.0 - abs(bbox_area_ratio - 0.26) / 0.26, 0.0, 1.0))
        score = 0.45 * area_score + 0.30 * fill_score + 0.25 * center_score
        if plausible:
            score += 0.25

        color = (0, 255, 255) if plausible else (0, 140, 255)
        cv2.rectangle(
            selection_overlay,
            (bbox_xyxy[0], bbox_xyxy[1]),
            (bbox_xyxy[2], bbox_xyxy[3]),
            color,
            2,
        )

        if score > best_score:
            best_score = float(score)
            best_quad = quad
            best_bbox = bbox_xyxy

    debug_images["binary_rescue_component_mask"] = component_mask
    debug_images["binary_rescue_selection_overlay"] = selection_overlay

    if best_quad is not None and best_bbox is not None:
        quad_overlay = _draw_quad_overlay(
            cv2.cvtColor(component_mask, cv2.COLOR_GRAY2BGR),
            best_quad,
            (0, 255, 0),
        )
        debug_images["binary_rescue_quad_overlay"] = quad_overlay
        metrics.update(
            {
                "binary_rescue_found": 1.0,
                "binary_rescue_bbox_x1": float(best_bbox[0]),
                "binary_rescue_bbox_y1": float(best_bbox[1]),
                "binary_rescue_bbox_x2": float(best_bbox[2]),
                "binary_rescue_bbox_y2": float(best_bbox[3]),
            }
        )
    else:
        debug_images["binary_rescue_quad_overlay"] = cv2.cvtColor(
            component_mask, cv2.COLOR_GRAY2BGR
        )

    return {
        "quad": best_quad,
        "bbox": best_bbox,
        "metrics": metrics,
        "debug_images": debug_images,
    }


def _build_card_candidate_mask(binary_view: np.ndarray) -> np.ndarray:
    binary = binary_view
    if binary.dtype != np.uint8:
        binary = np.clip(binary, 0, 255).astype(np.uint8)
    if np.unique(binary).size > 2:
        _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    min_dim = max(1, min(binary.shape[:2]))
    k_long = max(15, int(round(min_dim * CARD_OUTER_MASK_LONG_KERNEL_RATIO)))
    k_short = max(3, int(round(min_dim * CARD_OUTER_MASK_SHORT_KERNEL_RATIO)))
    if k_long % 2 == 0:
        k_long += 1
    if k_short % 2 == 0:
        k_short += 1

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k_long, k_short))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (k_short, k_long))
    kernel_sq = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(5, int(round(min_dim * CARD_OUTER_MASK_SQUARE_KERNEL_RATIO))),
            max(5, int(round(min_dim * CARD_OUTER_MASK_SQUARE_KERNEL_RATIO))),
        ),
    )

    mask_h = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h, iterations=2)
    mask_v = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_v, iterations=2)
    combined = cv2.bitwise_or(mask_h, mask_v)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_sq, iterations=2)
    combined = cv2.dilate(combined, kernel_sq, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_sq, iterations=1)
    return combined


def _build_binary_direct_candidate_mask(binary_view: np.ndarray) -> np.ndarray:
    binary = binary_view
    if binary.dtype != np.uint8:
        binary = np.clip(binary, 0, 255).astype(np.uint8)
    if np.unique(binary).size > 2:
        _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    min_dim = max(1, min(binary.shape[:2]))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_mid = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(5, int(round(min_dim * 0.018))),
            max(5, int(round(min_dim * 0.018))),
        ),
    )
    kernel_h = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(9, int(round(min_dim * 0.08))),
            max(3, int(round(min_dim * 0.012))),
        ),
    )
    kernel_v = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(3, int(round(min_dim * 0.012))),
            max(9, int(round(min_dim * 0.08))),
        ),
    )

    direct = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    direct_h = cv2.morphologyEx(direct, cv2.MORPH_CLOSE, kernel_h, iterations=1)
    direct_v = cv2.morphologyEx(direct, cv2.MORPH_CLOSE, kernel_v, iterations=1)
    direct = cv2.bitwise_or(direct_h, direct_v)
    direct = cv2.morphologyEx(direct, cv2.MORPH_CLOSE, kernel_mid, iterations=2)
    direct = cv2.morphologyEx(direct, cv2.MORPH_OPEN, kernel_small, iterations=1)
    return direct


def _extract_binary_direct_candidates(
    binary_view: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[List[np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    direct_mask = _build_binary_direct_candidate_mask(binary_view)
    contours, _ = cv2.findContours(
        direct_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    overlay = cv2.cvtColor(direct_mask, cv2.COLOR_GRAY2BGR)
    kept_quads: List[np.ndarray] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        area = float(cv2.contourArea(contour))
        if area < CARD_OUTER_MIN_COMPONENT_AREA:
            continue
        quad = order_points(_rect_from_min_area(contour))
        bbox_xyxy = _candidate_bbox_xyxy(quad, image_shape)
        if not _is_plausible_outer_bbox(bbox_xyxy, image_shape):
            continue
        kept_quads.append(quad)
        cv2.polylines(overlay, [quad.astype(np.int32)], True, (0, 255, 255), 2)

    best_overlay = cv2.cvtColor(direct_mask, cv2.COLOR_GRAY2BGR)
    if kept_quads:
        cv2.polylines(
            best_overlay,
            [kept_quads[0].astype(np.int32)],
            True,
            (0, 255, 0),
            2,
        )

    debug_images = {
        "binary_direct_candidate_mask": direct_mask,
        "binary_direct_component_overlay": overlay,
        "binary_direct_quad_overlay": best_overlay,
    }
    metrics = {
        "binary_direct_candidate_found": float(1.0 if kept_quads else 0.0),
        "binary_direct_candidate_count": float(len(kept_quads)),
    }
    return kept_quads, debug_images, metrics


def _extract_binary_thin_contour_candidates(
    binary_view: np.ndarray,
    image_shape: Tuple[int, int],
    edge_map: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    binary = binary_view
    if binary.dtype != np.uint8:
        binary = np.clip(binary, 0, 255).astype(np.uint8)
    if np.unique(binary).size > 2:
        _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    combined = binary.copy()
    if edge_map is not None and edge_map.size > 0:
        edges = edge_map
        if edges.dtype != np.uint8:
            edges = np.clip(edges, 0, 255).astype(np.uint8)
        if np.unique(edges).size > 2:
            _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        combined = cv2.bitwise_or(combined, edges)

    min_dim = max(1, min(image_shape[:2]))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_h = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(5, int(round(min_dim * 0.035))), 3),
    )
    kernel_v = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, max(5, int(round(min_dim * 0.035)))),
    )

    thin = cv2.morphologyEx(combined, cv2.MORPH_GRADIENT, kernel_small)
    thin_h = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, kernel_h, iterations=1)
    thin_v = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, kernel_v, iterations=1)
    thin_mask = cv2.bitwise_or(thin_h, thin_v)
    thin_mask = cv2.morphologyEx(thin_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    contours, _ = cv2.findContours(
        thin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_overlay = cv2.cvtColor(thin_mask, cv2.COLOR_GRAY2BGR)
    raw_quad_overlay = cv2.cvtColor(thin_mask, cv2.COLOR_GRAY2BGR)
    local_accept_overlay = cv2.cvtColor(thin_mask, cv2.COLOR_GRAY2BGR)
    kept_quads: List[np.ndarray] = []
    raw_quad: Optional[np.ndarray] = None
    accepted_quad: Optional[np.ndarray] = None
    accepted_area_ratio = 0.0
    accepted_aspect_ratio = 0.0
    accepted_is_convex = 0.0
    accepted_rectangularity = 0.0
    accepted_side_support_mean = 0.0
    local_rejection_reason = "no_contour"
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        area = float(cv2.contourArea(contour))
        if area < CARD_OUTER_MIN_COMPONENT_AREA:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        epsilon = max(2.0, 0.02 * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            quad = order_points(approx.reshape(4, 2).astype(np.float32))
        else:
            quad = order_points(_rect_from_min_area(contour))
        if raw_quad is None:
            raw_quad = quad.copy()
            cv2.polylines(
                raw_quad_overlay,
                [raw_quad.astype(np.int32)],
                True,
                (255, 255, 0),
                2,
            )
        bbox_xyxy = _candidate_bbox_xyxy(quad, image_shape)
        if not _is_plausible_outer_bbox(bbox_xyxy, image_shape):
            if local_rejection_reason == "no_contour":
                local_rejection_reason = "implausible_bbox"
            continue
        kept_quads.append(quad)
        if accepted_quad is None:
            accepted_quad = quad.copy()
            img_h, img_w = image_shape[:2]
            accepted_area_ratio = float(
                polygon_area(accepted_quad) / max(1.0, img_h * img_w)
            )
            accepted_aspect_ratio = float(_aspect_ratio_of_quad(accepted_quad))
            accepted_is_convex = float(
                1.0 if cv2.isContourConvex(accepted_quad.astype(np.int32)) else 0.0
            )
            rect_score, _ = _rectangularity_score(accepted_quad)
            accepted_rectangularity = float(rect_score)
            thin_support_metrics, _, _ = _silhouette_support_scores(
                accepted_quad,
                structural_mask=thin_mask,
                metric_prefix="binary_thin_local",
            )
            accepted_side_support_mean = float(
                np.mean(
                    [
                        thin_support_metrics.get("binary_thin_local_left_border_support", 0.0),
                        thin_support_metrics.get("binary_thin_local_right_border_support", 0.0),
                        thin_support_metrics.get("binary_thin_local_top_border_support", 0.0),
                        thin_support_metrics.get("binary_thin_local_bottom_border_support", 0.0),
                    ]
                )
            )
            cv2.polylines(
                local_accept_overlay,
                [accepted_quad.astype(np.int32)],
                True,
                (0, 255, 0),
                2,
            )
            local_rejection_reason = ""
        cv2.polylines(contour_overlay, [quad.astype(np.int32)], True, (0, 255, 255), 2)

    debug_images = {
        "binary_thin_candidate_mask": thin_mask,
        "binary_thin_contour_overlay": contour_overlay,
        "binary_thin_raw_quad_overlay": raw_quad_overlay,
        "binary_thin_local_accept_overlay": local_accept_overlay,
    }
    metrics = {
        "binary_thin_contour_count": float(len(contours)),
        "binary_thin_quad_generated": float(1.0 if raw_quad is not None else 0.0),
        "binary_thin_passed_local_filter": float(1.0 if kept_quads else 0.0),
        "binary_thin_added_to_global_candidates": 0.0,
        "binary_thin_reached_ranking": 0.0,
        "binary_thin_hard_rules_passed": 0.0,
        "binary_thin_selected_as_detected_quad": 0.0,
        "binary_thin_local_rejection_reason": local_rejection_reason,
        "binary_thin_global_rejection_reason": "",
        "binary_thin_area_ratio": float(accepted_area_ratio),
        "binary_thin_aspect_ratio": float(accepted_aspect_ratio),
        "binary_thin_rectangularity": float(accepted_rectangularity),
        "binary_thin_side_support_mean": float(accepted_side_support_mean),
        "binary_thin_is_convex": float(accepted_is_convex),
        "binary_thin_rank_score": 0.0,
    }
    return kept_quads, debug_images, metrics


def _build_outer_candidate_mask(
    binary_view: np.ndarray,
    structural_mask: Optional[np.ndarray] = None,
    edge_map: Optional[np.ndarray] = None,
    yolo_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
    preferred_seed_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    binary = binary_view
    if binary.dtype != np.uint8:
        binary = np.clip(binary, 0, 255).astype(np.uint8)
    if np.unique(binary).size > 2:
        _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    base_mask = _build_card_candidate_mask(binary)
    combined = base_mask.copy()

    if structural_mask is not None and structural_mask.size > 0:
        structural = structural_mask
        if structural.dtype != np.uint8:
            structural = np.clip(structural, 0, 255).astype(np.uint8)
        if np.unique(structural).size > 2:
            _, structural = cv2.threshold(
                structural, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        combined = cv2.bitwise_or(combined, structural)

    if edge_map is not None and edge_map.size > 0:
        edges = edge_map
        if edges.dtype != np.uint8:
            edges = np.clip(edges, 0, 255).astype(np.uint8)
        if np.unique(edges).size > 2:
            _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        min_dim = max(1, min(edges.shape[:2]))
        edge_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(3, int(round(min_dim * 0.010))), max(3, int(round(min_dim * 0.010)))),
        )
        edge_shell = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_kernel, iterations=2)
        combined = cv2.bitwise_or(combined, edge_shell)
    else:
        edge_shell = np.zeros_like(combined)

    min_dim = max(1, min(combined.shape[:2]))
    k_long = max(21, int(round(min_dim * CARD_OUTER_MASK_LONG_KERNEL_RATIO)))
    k_short = max(3, int(round(min_dim * CARD_OUTER_MASK_SHORT_KERNEL_RATIO)))
    if k_long % 2 == 0:
        k_long += 1
    if k_short % 2 == 0:
        k_short += 1

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k_long, k_short))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (k_short, k_long))
    kernel_sq = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(5, int(round(min_dim * CARD_OUTER_MASK_SQUARE_KERNEL_RATIO))),
            max(5, int(round(min_dim * CARD_OUTER_MASK_SQUARE_KERNEL_RATIO))),
        ),
    )

    oriented_h = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_h, iterations=2)
    oriented_v = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_v, iterations=2)
    outer_mask = cv2.bitwise_or(oriented_h, oriented_v)
    outer_mask = cv2.morphologyEx(outer_mask, cv2.MORPH_CLOSE, kernel_sq, iterations=2)
    outer_mask = cv2.dilate(outer_mask, kernel_sq, iterations=1)
    outer_mask = cv2.morphologyEx(outer_mask, cv2.MORPH_OPEN, kernel_sq, iterations=1)

    filtered_mask = np.zeros_like(outer_mask)
    contours, _ = cv2.findContours(outer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < CARD_OUTER_MIN_COMPONENT_AREA:
            continue
        quad = _rect_from_min_area(contour)
        bbox_xyxy = _candidate_bbox_xyxy(quad, outer_mask.shape[:2])
        if not _is_plausible_outer_bbox(bbox_xyxy, outer_mask.shape[:2]):
            continue
        cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=-1)

    if np.count_nonzero(filtered_mask) == 0:
        filtered_mask = outer_mask

    if CARD_OUTER_ENABLE_YOLO_PRIOR and yolo_bbox_xyxy is not None:
        prior_mask = np.zeros_like(filtered_mask)
        x1, y1, x2, y2 = _clip_bbox_xyxy(yolo_bbox_xyxy, filtered_mask.shape[:2])
        cv2.rectangle(prior_mask, (x1, y1), (x2, y2), 255, thickness=-1)
        filtered_mask = cv2.bitwise_and(filtered_mask, prior_mask)
        if np.count_nonzero(filtered_mask) == 0:
            filtered_mask = outer_mask

    if preferred_seed_bbox is not None:
        strengthened_mask, strengthened_debug, strengthened_metrics = (
            _strengthen_outer_mask_with_seed(
                filtered_mask,
                preferred_seed_bbox,
            )
        )
        if np.count_nonzero(strengthened_mask) > 0:
            filtered_mask = strengthened_mask
        debug_images = {
            "outer_candidate_base_mask": base_mask,
            "outer_candidate_combined_mask": combined,
            "outer_candidate_edge_shell": edge_shell,
            "outer_candidate_mask_raw": outer_mask,
            "outer_candidate_mask_filtered": filtered_mask,
            **strengthened_debug,
        }
        return filtered_mask, debug_images, strengthened_metrics

    cleaned_mask, cleaned_debug, cleaned_metrics = _cleanup_coarse_outer_mask(filtered_mask)
    if np.count_nonzero(cleaned_mask) > 0:
        filtered_mask = cleaned_mask

    debug_images = {
        "outer_candidate_base_mask": base_mask,
        "outer_candidate_combined_mask": combined,
        "outer_candidate_edge_shell": edge_shell,
        "outer_candidate_mask_raw": outer_mask,
        "outer_candidate_mask_filtered": filtered_mask,
        **cleaned_debug,
    }
    return filtered_mask, debug_images, cleaned_metrics


def _greedy_profile_peaks(
    profile: np.ndarray,
    start: int,
    end: int,
    count: int,
    min_distance: int,
) -> List[int]:
    start = max(0, int(start))
    end = min(len(profile), int(end))
    if end <= start:
        return []
    work = profile[start:end].copy()
    peaks: List[int] = []
    for _ in range(max(1, count)):
        local_idx = int(np.argmax(work))
        value = float(work[local_idx])
        if value <= 0:
            break
        idx = start + local_idx
        peaks.append(idx)
        left = max(0, local_idx - min_distance)
        right = min(work.shape[0], local_idx + min_distance + 1)
        work[left:right] = 0.0
    return peaks


def _estimate_card_bbox_from_line_projections(
    binary_view: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[
    Optional[Tuple[int, int, int, int]],
    Dict[str, float],
    Dict[str, np.ndarray],
]:
    binary = binary_view
    if binary.dtype != np.uint8:
        binary = np.clip(binary, 0, 255).astype(np.uint8)
    if np.unique(binary).size > 2:
        _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = image_shape[:2]
    kx = max(15, int(round(w * 0.10)))
    ky = max(15, int(round(h * 0.10)))
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, ky))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h, iterations=1)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v, iterations=1)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    col_profile = vertical_lines.mean(axis=0).astype(np.float32) / 255.0
    row_profile = horizontal_lines.mean(axis=1).astype(np.float32) / 255.0
    col_profile = cv2.GaussianBlur(col_profile.reshape(1, -1), (1, 51), 0).ravel()
    row_profile = cv2.GaussianBlur(row_profile.reshape(-1, 1), (51, 1), 0).ravel()

    x_peaks_left = _greedy_profile_peaks(col_profile, 0.03 * w, 0.48 * w, 8, max(8, w // 30))
    x_peaks_right = _greedy_profile_peaks(col_profile, 0.52 * w, 0.97 * w, 8, max(8, w // 30))
    y_peaks_top = _greedy_profile_peaks(row_profile, 0.03 * h, 0.42 * h, 8, max(8, h // 30))
    y_peaks_bottom = _greedy_profile_peaks(row_profile, 0.58 * h, 0.97 * h, 8, max(8, h // 30))

    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_score = -1.0
    for x1 in x_peaks_left:
        for x2 in x_peaks_right:
            if x2 <= x1:
                continue
            for y1 in y_peaks_top:
                for y2 in y_peaks_bottom:
                    if y2 <= y1:
                        continue
                    bbox = _clip_bbox_xyxy((x1, y1, x2, y2), image_shape)
                    bw = float(max(1, bbox[2] - bbox[0]))
                    bh = float(max(1, bbox[3] - bbox[1]))
                    width_ratio = bw / max(1.0, w)
                    height_ratio = bh / max(1.0, h)
                    if width_ratio < 0.12 or height_ratio < 0.20:
                        continue
                    if width_ratio > 0.75 or height_ratio > 0.92:
                        continue
                    aspect_ratio = bw / max(1.0, bh)
                    aspect_score = max(0.0, 1.0 - abs(aspect_ratio - CARD_TARGET_ASPECT_RATIO) / 0.18)
                    if aspect_score <= 0.0:
                        continue
                    area_ratio = (bw * bh) / max(1.0, w * h)
                    area_score = max(0.0, 1.0 - abs(area_ratio - 0.22) / 0.20)
                    center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32)
                    img_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
                    center_dist = np.linalg.norm(center - img_center)
                    center_score = 1.0 - (center_dist / max(1.0, np.linalg.norm(img_center)))
                    line_score = float(
                        np.clip(
                            0.25 * col_profile[bbox[0]]
                            + 0.25 * col_profile[min(w - 1, bbox[2] - 1)]
                            + 0.25 * row_profile[bbox[1]]
                            + 0.25 * row_profile[min(h - 1, bbox[3] - 1)],
                            0.0,
                            1.0,
                        )
                    )
                    score = (
                        0.42 * line_score
                        + 0.26 * aspect_score
                        + 0.18 * area_score
                        + 0.14 * center_score
                    )
                    if score > best_score:
                        best_score = float(score)
                        best_bbox = bbox

    metrics: Dict[str, float] = {
        "projection_bbox_found": float(1.0 if best_bbox is not None else 0.0),
        "projection_bbox_score": float(max(0.0, best_score)),
    }
    if best_bbox is not None:
        metrics.update(
            {
                "projection_bbox_x1": float(best_bbox[0]),
                "projection_bbox_y1": float(best_bbox[1]),
                "projection_bbox_x2": float(best_bbox[2]),
                "projection_bbox_y2": float(best_bbox[3]),
            }
        )

    debug_images = {
        "projection_horizontal_lines": horizontal_lines,
        "projection_vertical_lines": vertical_lines,
    }
    return best_bbox, metrics, debug_images


def _component_quad_metrics(
    contour: np.ndarray,
    image_shape: Tuple[int, int],
    binary_view: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Dict[str, float], float]:
    contour_area = float(cv2.contourArea(contour))
    rect_quad = _rect_from_min_area(contour)
    ordered_quad = order_points(rect_quad)
    bbox_xyxy = _candidate_bbox_xyxy(ordered_quad, image_shape)
    bbox_area = _bbox_area(bbox_xyxy)
    img_h, img_w = image_shape[:2]
    area_ratio = float(bbox_area / max(1.0, img_h * img_w))
    fill_ratio = float(contour_area / max(1.0, bbox_area))
    aspect_ratio = _aspect_ratio_of_quad(ordered_quad)
    aspect_deviation = abs(aspect_ratio - CARD_TARGET_ASPECT_RATIO)
    center_score = _center_distance_score(ordered_quad, image_shape)
    aspect_score = max(0.0, 1.0 - (aspect_deviation / 0.18))
    area_score = max(0.0, 1.0 - abs(area_ratio - 0.30) / 0.24)
    fill_score = float(np.clip(fill_ratio / 0.80, 0.0, 1.0))
    inner_parallel_score, inner_parallel_metrics = _inner_parallel_contour_score(
        binary_view=binary_view,
        bbox_xyxy=bbox_xyxy,
    )
    score = float(
        np.clip(
            0.28 * aspect_score
            + 0.24 * area_score
            + 0.18 * fill_score
            + 0.12 * center_score
            + 0.18 * inner_parallel_score,
            0.0,
            1.0,
        )
    )
    metrics = {
        "component_area_ratio": area_ratio,
        "component_fill_ratio": fill_ratio,
        "component_aspect_ratio": float(aspect_ratio),
        "component_aspect_deviation": float(aspect_deviation),
        "component_center_score": float(center_score),
        "component_inner_parallel_border_score": float(inner_parallel_score),
        "component_score": score,
    }
    metrics.update(
        {
            f"component_{key}": float(value)
            for key, value in inner_parallel_metrics.items()
        }
    )
    return ordered_quad, bbox_xyxy, metrics, score


def _detect_card_from_binary(
    binary_view: np.ndarray,
    image_shape: Tuple[int, int],
    edge_map: Optional[np.ndarray] = None,
    structural_mask: Optional[np.ndarray] = None,
    hsv_view: Optional[np.ndarray] = None,
    lab_view: Optional[np.ndarray] = None,
    expected_outer_bbox: Optional[Tuple[int, int, int, int]] = None,
    yolo_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
    preferred_seed_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[
    Optional[np.ndarray],
    Optional[Tuple[int, int, int, int]],
    Dict[str, object],
    Dict[str, np.ndarray],
]:
    candidate_mask, candidate_mask_debug, candidate_mask_metrics = _build_outer_candidate_mask(
        binary_view=binary_view,
        structural_mask=structural_mask,
        edge_map=edge_map,
        yolo_bbox_xyxy=yolo_bbox_xyxy,
        preferred_seed_bbox=preferred_seed_bbox,
    )
    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    binary_direct_quads, binary_direct_debug, binary_direct_metrics = (
        _extract_binary_direct_candidates(
            binary_view=binary_view,
            image_shape=image_shape,
        )
    )
    binary_thin_quads, binary_thin_debug, binary_thin_metrics = (
        _extract_binary_thin_contour_candidates(
            binary_view=binary_view,
            image_shape=image_shape,
            edge_map=edge_map,
        )
    )
    best_quad: Optional[np.ndarray] = None
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_score = -1.0
    best_source_name = ""
    best_metrics: Dict[str, object] = {
        "detector_mode": "binary_bbox",
        "candidate_mask_components": float(len(contours)),
        "candidate_mask_mode": "outer_prewarp",
    }

    rejected_quads: List[np.ndarray] = []
    ranked_candidates = 0
    candidate_quads: List[Tuple[np.ndarray, str]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < 2000:
            continue
        candidate_quads.append((order_points(_rect_from_min_area(contour)), "outer_mask"))
    candidate_quads.extend((quad, "binary_direct") for quad in binary_direct_quads)
    candidate_quads.extend((quad, "binary_thin") for quad in binary_thin_quads)

    binary_thin_global_candidates: List[np.ndarray] = []
    binary_thin_rank_rejected_quads: List[np.ndarray] = []
    binary_thin_selected_quads: List[np.ndarray] = []
    seen_quads = set()
    for quad, source_name in candidate_quads:
        quad = order_points(quad)
        quad_key = tuple(np.round(quad.reshape(-1), 1).tolist())
        if source_name == "binary_thin":
            binary_thin_metrics["binary_thin_added_to_global_candidates"] = 1.0
        if quad_key in seen_quads:
            if (
                source_name == "binary_thin"
                and not binary_thin_metrics.get("binary_thin_global_rejection_reason")
            ):
                binary_thin_metrics["binary_thin_global_rejection_reason"] = "dedupe"
            continue
        seen_quads.add(quad_key)
        if source_name == "binary_thin":
            binary_thin_global_candidates.append(quad.copy())
            binary_thin_metrics["binary_thin_reached_ranking"] = 1.0
        bbox_xyxy = _candidate_bbox_xyxy(quad, image_shape)
        score, metrics = _rank_candidate(
            quad=quad,
            image_shape=image_shape,
            edge_map=edge_map,
            structural_mask=structural_mask,
            binary_view=binary_view,
            hsv_view=hsv_view,
            lab_view=lab_view,
            roi_bbox=yolo_bbox_xyxy,
            expected_outer_bbox=expected_outer_bbox,
            metric_prefix="candidate",
        )
        if not _is_plausible_outer_bbox(bbox_xyxy, image_shape):
            rejected_quads.append(quad)
            if source_name == "binary_thin":
                binary_thin_rank_rejected_quads.append(quad.copy())
                if not binary_thin_metrics.get("binary_thin_global_rejection_reason"):
                    binary_thin_metrics["binary_thin_global_rejection_reason"] = "implausible_bbox"
            continue
        if score <= 0.0:
            rejected_quads.append(quad)
            if source_name == "binary_thin":
                binary_thin_rank_rejected_quads.append(quad.copy())
                binary_thin_metrics["binary_thin_rank_score"] = float(
                    max(binary_thin_metrics.get("binary_thin_rank_score", 0.0), score)
                )
                binary_thin_metrics["binary_thin_hard_rules_passed"] = 0.0
                binary_thin_metrics["binary_thin_global_rejection_reason"] = str(
                    metrics.get("candidate_rejection_reason", "score_le_zero")
                )
            continue
        ranked_candidates += 1
        if source_name == "binary_thin":
            binary_thin_metrics["binary_thin_rank_score"] = float(
                max(binary_thin_metrics.get("binary_thin_rank_score", 0.0), score)
            )
            binary_thin_metrics["binary_thin_hard_rules_passed"] = 1.0
            binary_thin_metrics["binary_thin_global_rejection_reason"] = ""
        if score > best_score:
            best_quad = quad
            best_bbox = bbox_xyxy
            best_score = score
            best_source_name = source_name
            best_metrics.update(metrics)

    if best_source_name == "binary_thin" and best_quad is not None:
        binary_thin_selected_quads = [best_quad.copy()]

    debug_images: Dict[str, np.ndarray] = {
        "candidate_mask": candidate_mask,
        "candidate_mask_overlay": _draw_multiple_quads_overlay(
            cv2.cvtColor(candidate_mask, cv2.COLOR_GRAY2BGR),
            rejected_quads,
            (0, 0, 255),
        ),
    }
    debug_images.update(candidate_mask_debug)
    debug_images.update(binary_direct_debug)
    debug_images.update(binary_thin_debug)
    debug_images["binary_thin_global_candidate_overlay"] = _draw_multiple_quads_overlay(
        cv2.cvtColor(candidate_mask, cv2.COLOR_GRAY2BGR),
        binary_thin_global_candidates,
        (255, 200, 0),
    )
    debug_images["binary_thin_rank_rejected_overlay"] = _draw_multiple_quads_overlay(
        cv2.cvtColor(candidate_mask, cv2.COLOR_GRAY2BGR),
        binary_thin_rank_rejected_quads,
        (0, 0, 255),
    )
    debug_images["binary_thin_selected_overlay"] = _draw_multiple_quads_overlay(
        cv2.cvtColor(candidate_mask, cv2.COLOR_GRAY2BGR),
        binary_thin_selected_quads,
        (0, 255, 0),
    )
    if candidate_mask_metrics:
        best_metrics.update(candidate_mask_metrics)
    if binary_direct_metrics:
        best_metrics.update(binary_direct_metrics)
    if binary_thin_metrics:
        binary_thin_metrics["binary_thin_selected_as_detected_quad"] = float(
            1.0 if best_source_name == "binary_thin" else 0.0
        )
        if (
            binary_thin_metrics.get("binary_thin_passed_local_filter", 0.0) >= 0.5
            and binary_thin_metrics.get("binary_thin_added_to_global_candidates", 0.0) < 0.5
            and not binary_thin_metrics.get("binary_thin_global_rejection_reason")
        ):
            binary_thin_metrics["binary_thin_global_rejection_reason"] = "not_added_to_global_candidates"
        elif (
            binary_thin_metrics.get("binary_thin_added_to_global_candidates", 0.0) >= 0.5
            and binary_thin_metrics.get("binary_thin_reached_ranking", 0.0) < 0.5
            and not binary_thin_metrics.get("binary_thin_global_rejection_reason")
        ):
            binary_thin_metrics["binary_thin_global_rejection_reason"] = "lost_before_ranking"
        elif (
            binary_thin_metrics.get("binary_thin_reached_ranking", 0.0) >= 0.5
            and binary_thin_metrics.get("binary_thin_rank_score", 0.0) <= 0.0
            and not binary_thin_metrics.get("binary_thin_global_rejection_reason")
        ):
            binary_thin_metrics["binary_thin_global_rejection_reason"] = "score_le_zero"
        best_metrics.update(binary_thin_metrics)
    if best_quad is not None and best_bbox is not None:
        best_metrics.update(
            {
                "estimated_outer_card_bbox_found": 1.0,
                "estimated_outer_card_bbox_score": float(best_score),
                "estimated_outer_card_bbox_x1": float(best_bbox[0]),
                "estimated_outer_card_bbox_y1": float(best_bbox[1]),
                "estimated_outer_card_bbox_x2": float(best_bbox[2]),
                "estimated_outer_card_bbox_y2": float(best_bbox[3]),
                "candidate_ranked_count": float(ranked_candidates),
                "candidate_mask_components": float(
                    len(contours) + len(binary_direct_quads) + len(binary_thin_quads)
                ),
            }
        )
    else:
        best_metrics.update(
            {
                "estimated_outer_card_bbox_found": 0.0,
                "estimated_outer_card_bbox_score": 0.0,
                "candidate_ranked_count": float(ranked_candidates),
                "candidate_mask_components": float(
                    len(contours) + len(binary_direct_quads) + len(binary_thin_quads)
                ),
            }
        )

    return best_quad, best_bbox, best_metrics, debug_images


def _expand_xyxy_bbox(
    bbox: Dict[str, float],
    image_shape: Tuple[int, int],
    margin_ratio: float = YOLO_CARD_BBOX_MARGIN,
) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    x1 = float(bbox["x1"])
    y1 = float(bbox["y1"])
    x2 = float(bbox["x2"])
    y2 = float(bbox["y2"])
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    mx = bw * margin_ratio
    my = bh * margin_ratio
    return (
        max(0, int(round(x1 - mx))),
        max(0, int(round(y1 - my))),
        min(w, int(round(x2 + mx))),
        min(h, int(round(y2 + my))),
    )


def _quad_from_bbox_xyxy(bbox_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def _draw_quad_overlay(
    image: np.ndarray,
    quad: Optional[np.ndarray],
    color: Tuple[int, int, int],
    thickness: int = 3,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    overlay = image.copy()
    if roi_bbox is not None:
        x1, y1, x2, y2 = roi_bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), 2)
    if quad is not None:
        cv2.polylines(overlay, [order_points(quad).astype(np.int32)], True, color, thickness)
    return overlay


def _draw_bbox_overlay(
    image: np.ndarray,
    bbox_xyxy: Optional[Tuple[int, int, int, int]],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    overlay = image.copy()
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
    return overlay


def _draw_multiple_quads_overlay(
    image: np.ndarray,
    quads: List[np.ndarray],
    color: Tuple[int, int, int],
    thickness: int = 2,
    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    overlay = image.copy()
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), 2)
    for quad in quads:
        cv2.polylines(overlay, [order_points(quad).astype(np.int32)], True, color, thickness)
    return overlay


def _normalize_detection_input(
    image: np.ndarray,
    target_long_side: int = 1200,
) -> Dict[str, object]:
    bgr = ensure_bgr_uint8(image)
    resized, scale = _resize_for_detection(bgr, target_long_side=target_long_side)
    yolo_status = get_yolo_status()
    yolo_bbox = detect_card_bbox(resized) if yolo_status.get("available") else None
    yolo_bbox_xyxy = (
        _expand_xyxy_bbox(yolo_bbox, resized.shape[:2])
        if yolo_bbox is not None
        else None
    )
    return {
        "bgr": bgr,
        "resized": resized,
        "scale": scale,
        "yolo_status": yolo_status,
        "yolo_bbox": yolo_bbox,
        "yolo_bbox_xyxy": yolo_bbox_xyxy,
    }


def _build_detection_views(image_bgr: np.ndarray) -> Dict[str, object]:
    multilayer_views = _build_multilayer_views(image_bgr)
    return {
        "multilayer_views": multilayer_views,
        "gray": multilayer_views["gray"],
        "binary_view": multilayer_views["binary_view"],
        "structural_mask": multilayer_views["structural_mask"],
        "hsv_view": multilayer_views["hsv_view"],
        "lab_view": multilayer_views["lab_view"],
        "edge_map": multilayer_views.get("canny_closed", multilayer_views.get("canny")),
    }


def _generate_outer_candidates(
    binary_view: np.ndarray,
    image_shape: Tuple[int, int],
    edge_map: Optional[np.ndarray] = None,
    structural_mask: Optional[np.ndarray] = None,
    hsv_view: Optional[np.ndarray] = None,
    lab_view: Optional[np.ndarray] = None,
    yolo_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
    preferred_seed_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, object]:
    projection_bbox, projection_metrics, projection_debug = (
        _estimate_card_bbox_from_line_projections(
            binary_view=binary_view,
            image_shape=image_shape,
        )
    )
    component_quad, component_bbox, detection_metrics, component_debug = (
        _detect_card_from_binary(
            binary_view=binary_view,
            image_shape=image_shape,
            edge_map=edge_map,
            structural_mask=structural_mask,
            hsv_view=hsv_view,
            lab_view=lab_view,
            expected_outer_bbox=projection_bbox,
            yolo_bbox_xyxy=yolo_bbox_xyxy,
            preferred_seed_bbox=preferred_seed_bbox,
        )
    )
    return {
        "projection_bbox": projection_bbox,
        "projection_metrics": projection_metrics,
        "projection_debug": projection_debug,
        "component_quad": component_quad,
        "component_bbox": component_bbox,
        "detection_metrics": detection_metrics,
        "component_debug": component_debug,
    }


def _select_ranked_outer_candidate(
    candidate_stage: Dict[str, object],
    image_shape: Tuple[int, int],
    binary_view: np.ndarray,
) -> Dict[str, object]:
    projection_bbox = candidate_stage.get("projection_bbox")
    component_bbox = candidate_stage.get("component_bbox")
    component_quad = candidate_stage.get("component_quad")
    projection_metrics = candidate_stage.get("projection_metrics", {})
    detection_metrics = dict(candidate_stage.get("detection_metrics", {}))

    estimated_outer_bbox, bbox_selection_metrics = _select_consistent_card_bbox(
        projection_bbox=projection_bbox,
        component_bbox=component_bbox,
        image_shape=image_shape,
        binary_view=binary_view,
    )

    detected_quad: Optional[np.ndarray] = None
    if estimated_outer_bbox is not None:
        if (
            component_bbox is not None
            and estimated_outer_bbox == component_bbox
            and component_quad is not None
        ):
            detected_quad = component_quad
            detection_metrics["detector_mode"] = "binary_bbox"
        else:
            detected_quad = _quad_from_bbox_xyxy(estimated_outer_bbox)
            detection_metrics["detector_mode"] = "projection_bbox"

        selected_score = float(
            detection_metrics.get("estimated_outer_card_bbox_score", 0.0)
        )
        if projection_bbox is not None and estimated_outer_bbox == projection_bbox:
            selected_score = float(
                projection_metrics.get("projection_bbox_score", selected_score)
            )

        detection_metrics.update(
            {
                "estimated_outer_card_bbox_found": 1.0,
                "estimated_outer_card_bbox_score": selected_score,
                "estimated_outer_card_bbox_x1": float(estimated_outer_bbox[0]),
                "estimated_outer_card_bbox_y1": float(estimated_outer_bbox[1]),
                "estimated_outer_card_bbox_x2": float(estimated_outer_bbox[2]),
                "estimated_outer_card_bbox_y2": float(estimated_outer_bbox[3]),
            }
        )

    return {
        "estimated_outer_bbox": estimated_outer_bbox,
        "detected_quad": detected_quad,
        "detection_metrics": detection_metrics,
        "bbox_selection_metrics": bbox_selection_metrics,
    }


def _attempt_roi_second_pass(
    resized: np.ndarray,
    views_stage: Dict[str, object],
    candidate_stage: Dict[str, object],
    selection_stage: Dict[str, object],
    yolo_bbox_xyxy: Optional[Tuple[int, int, int, int]],
) -> Dict[str, object]:
    coarse_quad = selection_stage.get("detected_quad")
    estimated_outer_bbox = selection_stage.get("estimated_outer_bbox")
    projection_bbox = candidate_stage.get("projection_bbox")
    image_shape = resized.shape[:2]

    seed_bbox, seed_source = _coarse_seed_bbox(
        detected_quad=coarse_quad,
        estimated_outer_bbox=estimated_outer_bbox,
        projection_bbox=projection_bbox,
        image_shape=image_shape,
    )

    roi_metrics: Dict[str, object] = {
        "roi_second_pass_used": 0.0,
        "roi_second_pass_candidate_found": 0.0,
        "roi_second_pass_improved": 0.0,
        "roi_second_pass_source": seed_source,
    }
    debug_images: Dict[str, np.ndarray] = {
        "coarse_outer_bbox_overlay": _draw_bbox_overlay(
            resized,
            seed_bbox,
            (255, 180, 0),
        ),
    }

    if seed_bbox is None:
        return {
            "detected_quad": coarse_quad,
            "selected_candidate_quad": coarse_quad,
            "roi_quad": None,
            "roi_quad_found": 0.0,
            "roi_bbox": None,
            "roi_metrics": roi_metrics,
            "debug_images": debug_images,
        }

    roi_bbox = _expand_bbox_xyxy(seed_bbox, image_shape)
    x1, y1, x2, y2 = roi_bbox
    roi_bgr = resized[y1:y2, x1:x2]
    debug_images["coarse_outer_bbox_overlay"] = _draw_quad_overlay(
        debug_images["coarse_outer_bbox_overlay"],
        coarse_quad,
        (0, 255, 255),
        roi_bbox=roi_bbox,
    )

    if roi_bgr.size == 0 or roi_bgr.shape[0] < 40 or roi_bgr.shape[1] < 30:
        return {
            "detected_quad": coarse_quad,
            "selected_candidate_quad": coarse_quad,
            "roi_quad": None,
            "roi_quad_found": 0.0,
            "roi_bbox": roi_bbox,
            "roi_metrics": roi_metrics,
            "debug_images": debug_images,
        }

    local_seed_bbox = _shift_bbox_to_local(seed_bbox, roi_bbox, roi_bgr.shape[:2])
    roi_views_stage = _build_detection_views(roi_bgr)
    roi_candidate_stage = _generate_outer_candidates(
        binary_view=roi_views_stage["binary_view"],
        image_shape=roi_bgr.shape[:2],
        edge_map=roi_views_stage["edge_map"],
        structural_mask=roi_views_stage["structural_mask"],
        hsv_view=roi_views_stage["hsv_view"],
        lab_view=roi_views_stage["lab_view"],
        yolo_bbox_xyxy=None,
        preferred_seed_bbox=local_seed_bbox,
    )
    roi_selection_stage = _select_ranked_outer_candidate(
        candidate_stage=roi_candidate_stage,
        image_shape=roi_bgr.shape[:2],
        binary_view=roi_views_stage["binary_view"],
    )

    roi_metrics["roi_second_pass_used"] = 1.0
    local_quad = roi_selection_stage.get("detected_quad")
    debug_images.update(
        {
            "roi_crop_view": roi_bgr,
            "roi_binary_view": roi_views_stage["binary_view"],
            "roi_structural_mask": roi_views_stage["structural_mask"],
            "roi_edge_map": roi_views_stage["edge_map"],
            "roi_background_suppressed": roi_candidate_stage["component_debug"].get(
                "outer_candidate_background_suppressed",
                np.zeros(roi_bgr.shape[:2], dtype=np.uint8),
            ),
            "roi_border_touch_mask": roi_candidate_stage["component_debug"].get(
                "outer_candidate_border_touch_mask",
                np.zeros(roi_bgr.shape[:2], dtype=np.uint8),
            ),
            "roi_seed_overlap_overlay": roi_candidate_stage["component_debug"].get(
                "outer_candidate_seed_overlap_overlay",
                roi_bgr.copy(),
            ),
            "roi_main_component_mask": roi_candidate_stage["component_debug"].get(
                "outer_candidate_main_component_mask",
                np.zeros(roi_bgr.shape[:2], dtype=np.uint8),
            ),
            "roi_component_selection_overlay": roi_candidate_stage["component_debug"].get(
                "outer_candidate_component_selection_overlay",
                roi_bgr.copy(),
            ),
            "roi_detected_quad_overlay": _draw_quad_overlay(
                roi_bgr,
                local_quad,
                (0, 255, 0),
            ),
            "roi_selected_candidate_overlay": _draw_bbox_overlay(
                roi_bgr,
                roi_selection_stage.get("estimated_outer_bbox"),
                (255, 180, 0),
            ),
        }
    )

    if local_quad is None:
        return {
            "detected_quad": coarse_quad,
            "selected_candidate_quad": coarse_quad,
            "roi_quad": None,
            "roi_quad_found": 0.0,
            "roi_bbox": roi_bbox,
            "roi_metrics": roi_metrics,
            "debug_images": debug_images,
        }

    roi_metrics["roi_second_pass_candidate_found"] = 1.0
    translated_quad = _translate_quad(local_quad, (x1, y1))
    translated_bbox = _candidate_bbox_xyxy(translated_quad, image_shape)
    if not _is_plausible_outer_bbox(translated_bbox, image_shape):
        return {
            "detected_quad": coarse_quad,
            "selected_candidate_quad": coarse_quad,
            "roi_quad": translated_quad,
            "roi_quad_found": 1.0,
            "roi_bbox": roi_bbox,
            "roi_metrics": roi_metrics,
            "debug_images": debug_images,
        }

    edge_map = views_stage["edge_map"]
    structural_mask = views_stage["structural_mask"]
    binary_view = views_stage["binary_view"]
    hsv_view = views_stage["hsv_view"]
    lab_view = views_stage["lab_view"]

    coarse_score = -1.0
    if coarse_quad is not None:
        coarse_score, _ = _rank_candidate(
            quad=coarse_quad,
            image_shape=image_shape,
            edge_map=edge_map,
            structural_mask=structural_mask,
            binary_view=binary_view,
            hsv_view=hsv_view,
            lab_view=lab_view,
            roi_bbox=yolo_bbox_xyxy,
            expected_outer_bbox=estimated_outer_bbox,
            metric_prefix="candidate",
        )

    translated_score, _ = _rank_candidate(
        quad=translated_quad,
        image_shape=image_shape,
        edge_map=edge_map,
        structural_mask=structural_mask,
        binary_view=binary_view,
        hsv_view=hsv_view,
        lab_view=lab_view,
        roi_bbox=yolo_bbox_xyxy,
        expected_outer_bbox=estimated_outer_bbox,
        metric_prefix="candidate",
    )
    roi_metrics["roi_second_pass_candidate_score"] = float(translated_score)
    roi_metrics["roi_second_pass_previous_score"] = float(max(0.0, coarse_score))
    roi_metrics["roi_border_touch_rejected"] = float(
        roi_candidate_stage["detection_metrics"].get("outer_candidate_border_touch_rejected", 0.0)
    )
    roi_metrics["roi_seed_overlap_score"] = float(
        roi_candidate_stage["detection_metrics"].get("outer_candidate_seed_overlap_score", 0.0)
    )
    roi_metrics["roi_main_component_area_ratio"] = float(
        roi_candidate_stage["detection_metrics"].get("outer_candidate_main_component_area_ratio", 0.0)
    )

    selected_quad = coarse_quad
    if translated_score > 0.0 and (
        coarse_quad is None or translated_score >= coarse_score
    ):
        selected_quad = translated_quad
        roi_metrics["roi_second_pass_improved"] = float(
            1.0 if coarse_quad is None or translated_score > coarse_score else 0.0
        )

    debug_images["roi_selected_candidate_overlay"] = _draw_quad_overlay(
        debug_images["roi_selected_candidate_overlay"],
        order_points(selected_quad) if selected_quad is not None else None,
        (0, 255, 255),
    )

    return {
        "detected_quad": selected_quad,
        "selected_candidate_quad": selected_quad,
        "roi_quad": translated_quad,
        "roi_quad_found": 1.0,
        "roi_bbox": roi_bbox,
        "roi_metrics": roi_metrics,
        "debug_images": debug_images,
    }


def _resolve_final_candidate_quad(
    detected_quad: Optional[np.ndarray],
    estimated_outer_bbox: Optional[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
) -> Dict[str, object]:
    used_fallback = False
    fallback_trigger_reason = ""
    fallback_source = ""
    fallback_is_weak = 0.0
    final_quad = detected_quad

    if final_quad is None:
        final_quad, fallback_source, fallback_is_weak = _build_conservative_fallback_quad(
            image_shape,
            estimated_outer_bbox=estimated_outer_bbox,
        )
        used_fallback = True
        fallback_trigger_reason = "no_binary_bbox_detected"

    return {
        "final_quad": final_quad,
        "used_fallback": used_fallback,
        "fallback_trigger_reason": fallback_trigger_reason,
        "fallback_source": fallback_source,
        "fallback_is_weak": fallback_is_weak,
    }


def _attempt_pre_fallback_binary_rescue(
    views_stage: Dict[str, object],
    candidate_stage: Dict[str, object],
    selection_stage: Dict[str, object],
    roi_stage: Dict[str, object],
) -> Dict[str, object]:
    # Binary rescue is intentionally disabled to avoid promoting oversized quads.
    return {
        "detected_quad": None,
        "metrics": {
            "binary_rescue_attempted": 0.0,
            "binary_rescue_found": 0.0,
            "binary_rescue_status": "disabled",
        },
        "debug_images": {},
    }


def _project_detection_to_original_scale(
    final_quad: np.ndarray,
    scale: float,
) -> Dict[str, np.ndarray]:
    corners_resized = order_points(final_quad)
    corners_original = corners_resized / scale if scale != 1.0 else corners_resized.copy()
    contour_original = corners_original.astype(np.int32).reshape(-1, 1, 2)
    return {
        "corners_resized": corners_resized,
        "corners_original": corners_original.astype(np.float32),
        "contour_original": contour_original,
    }


def _build_detection_debug_images(
    resized: np.ndarray,
    views_stage: Dict[str, object],
    candidate_stage: Dict[str, object],
    selection_stage: Dict[str, object],
    roi_stage: Dict[str, object],
    binary_rescue_stage: Dict[str, object],
    projection_stage: Dict[str, np.ndarray],
    support_stage: Dict[str, object],
    yolo_bbox_xyxy: Optional[Tuple[int, int, int, int]],
) -> Dict[str, np.ndarray]:
    binary_view = views_stage["binary_view"]
    hsv_view = views_stage["hsv_view"]
    lab_view = views_stage["lab_view"]
    structural_mask = views_stage["structural_mask"]
    edge_map = views_stage["edge_map"]
    multilayer_views = views_stage["multilayer_views"]

    estimated_outer_bbox = selection_stage["estimated_outer_bbox"]
    coarse_detected_quad = selection_stage["detected_quad"]
    detected_quad = roi_stage["detected_quad"]
    roi_quad = roi_stage.get("roi_quad")
    roi_bbox = roi_stage.get("roi_bbox")
    corners_resized = projection_stage["corners_resized"]
    final_overlay_points = support_stage["final_overlay_points"]

    debug_maps: Dict[str, np.ndarray] = {
        "gray": views_stage["gray"],
        "binary_view": binary_view,
        "hsv_view": cv2.cvtColor(hsv_view, cv2.COLOR_HSV2BGR),
        "lab_view": cv2.cvtColor(lab_view, cv2.COLOR_LAB2BGR),
        "structural_mask": structural_mask,
        "canny": multilayer_views.get("canny", np.zeros_like(binary_view)),
        "canny_closed": multilayer_views.get("canny_closed", np.zeros_like(binary_view)),
        "multilayer_overlay": multilayer_views["multilayer_overlay"],
        "estimated_outer_card_bbox_overlay": _draw_bbox_overlay(
            resized,
            estimated_outer_bbox,
            (255, 180, 0),
        ),
        "detected_contour": _draw_quad_overlay(
            resized,
            corners_resized,
            (0, 255, 0),
        ),
        "initial_quad_overlay": _draw_quad_overlay(
            resized,
            coarse_detected_quad if coarse_detected_quad is not None else corners_resized,
            (0, 255, 255),
        ),
        "candidate_border_support_overlay": _draw_border_support_overlay(
            resized,
            final_overlay_points,
            quad=corners_resized,
        ),
        "rejected_internal_candidate_overlay": candidate_stage["component_debug"].get(
            "candidate_mask_overlay",
            resized.copy(),
        ),
        "refinement_roi_overlay": _draw_bbox_overlay(
            resized,
            roi_bbox if roi_bbox is not None else estimated_outer_bbox,
            (255, 180, 0),
        ),
        "coarse_background_suppressed": candidate_stage["component_debug"].get(
            "outer_candidate_background_suppressed",
            np.zeros_like(binary_view),
        ),
        "coarse_border_touch_mask": candidate_stage["component_debug"].get(
            "outer_candidate_border_touch_mask",
            np.zeros_like(binary_view),
        ),
        "coarse_main_component_mask": candidate_stage["component_debug"].get(
            "outer_candidate_main_component_mask",
            np.zeros_like(binary_view),
        ),
        "coarse_component_selection_overlay": candidate_stage["component_debug"].get(
            "outer_candidate_component_selection_overlay",
            resized.copy(),
        ),
        "coarse_seed_candidate_overlay": candidate_stage["component_debug"].get(
            "outer_candidate_seed_candidate_overlay",
            resized.copy(),
        ),
    }
    if coarse_detected_quad is not None:
        debug_maps["coarse_quad_overlay"] = _draw_quad_overlay(
            resized,
            coarse_detected_quad,
            (0, 255, 255),
        )
    if roi_stage.get("roi_quad_found", 0.0) >= 0.5 and roi_quad is not None:
        debug_maps["roi_quad_overlay"] = _draw_quad_overlay(
            resized,
            roi_quad,
            (255, 255, 0),
        )
    if detected_quad is not None:
        debug_maps["roi_selected_quad_overlay"] = _draw_quad_overlay(
            resized,
            detected_quad,
            (0, 200, 255),
        )
    if corners_resized is not None:
        debug_maps["final_quad_overlay"] = _draw_quad_overlay(
            resized,
            corners_resized,
            (0, 255, 0),
        )
        debug_maps["final_quad_used_for_warp_overlay"] = _draw_quad_overlay(
            resized,
            corners_resized,
            (0, 255, 0),
        )
    if edge_map is not None:
        debug_maps["edge_view"] = edge_map

    debug_maps.update(candidate_stage["component_debug"])
    debug_maps.update(candidate_stage["projection_debug"])
    debug_maps.update(roi_stage.get("debug_images", {}))
    debug_maps.update(binary_rescue_stage.get("debug_images", {}))

    if yolo_bbox_xyxy is not None:
        debug_maps["yolo_bbox"] = _draw_quad_overlay(
            resized,
            None,
            (255, 200, 0),
            roi_bbox=yolo_bbox_xyxy,
        )

    return debug_maps


def _build_detection_metrics(
    resized: np.ndarray,
    scale: float,
    yolo_status: Dict[str, object],
    yolo_bbox: Optional[Dict[str, float]],
    selection_stage: Dict[str, object],
    roi_stage: Dict[str, object],
    binary_rescue_stage: Dict[str, object],
    fallback_stage: Dict[str, object],
    projection_stage: Dict[str, np.ndarray],
    support_stage: Dict[str, object],
    candidate_stage: Dict[str, object],
    min_candidate_score: float,
) -> Dict[str, object]:
    detection_metrics = selection_stage["detection_metrics"]
    projection_metrics = candidate_stage["projection_metrics"]
    bbox_selection_metrics = selection_stage["bbox_selection_metrics"]
    roi_metrics = roi_stage.get("roi_metrics", {})
    binary_rescue_metrics = binary_rescue_stage.get("metrics", {})
    used_fallback = fallback_stage["used_fallback"]
    estimated_outer_bbox = selection_stage["estimated_outer_bbox"]
    fallback_is_weak = fallback_stage["fallback_is_weak"]
    fallback_source = fallback_stage["fallback_source"]
    fallback_trigger_reason = fallback_stage["fallback_trigger_reason"]
    corners_resized = projection_stage["corners_resized"]
    final_support_metrics = support_stage["final_support_metrics"]
    final_silhouette_score = support_stage["final_silhouette_score"]
    coarse_quad = selection_stage.get("detected_quad")
    roi_quad = roi_stage.get("roi_quad")
    final_quad = fallback_stage.get("final_quad")

    coarse_quad_found = float(1.0 if coarse_quad is not None else 0.0)
    roi_quad_found = float(roi_stage.get("roi_quad_found", 0.0))
    final_quad_found = float(1.0 if final_quad is not None else 0.0)

    final_quad_source = "none"
    final_quad_replaced_previous = 0.0
    final_quad_selection_reason = "no_quad"
    if used_fallback:
        final_quad_source = "fallback"
        final_quad_selection_reason = fallback_trigger_reason or "fallback_applied"
        final_quad_replaced_previous = 0.0
    elif roi_quad is not None and roi_quad_found >= 0.5:
        final_quad_source = "roi"
        final_quad_replaced_previous = float(
            1.0 if coarse_quad is not None and not np.allclose(order_points(roi_quad), order_points(coarse_quad)) else 0.0
        )
        if coarse_quad is None:
            final_quad_selection_reason = "roi_only_valid_quad"
        elif roi_metrics.get("roi_second_pass_improved", 0.0) >= 0.5:
            final_quad_selection_reason = "roi_improved_candidate"
        else:
            final_quad_selection_reason = "roi_kept_candidate"
    elif coarse_quad is not None:
        final_quad_source = "coarse"
        final_quad_selection_reason = "coarse_kept"
        final_quad_replaced_previous = 0.0
    elif binary_rescue_metrics.get("binary_rescue_found", 0.0) >= 0.5:
        final_quad_source = "binary_rescue"
        final_quad_selection_reason = "binary_rescue_quad"
        final_quad_replaced_previous = 1.0
    elif estimated_outer_bbox is not None:
        final_quad_source = "projection"
        final_quad_selection_reason = "projection_bbox_only"
        final_quad_replaced_previous = 0.0

    best_score = float(detection_metrics.get("estimated_outer_card_bbox_score", 0.0))
    if not used_fallback:
        best_score = max(best_score, min_candidate_score)

    quad_bbox = _candidate_bbox_xyxy(corners_resized, resized.shape[:2])
    bbox_w = float(max(1, quad_bbox[2] - quad_bbox[0]))
    bbox_h = float(max(1, quad_bbox[3] - quad_bbox[1]))
    image_area = float(max(1, resized.shape[0] * resized.shape[1]))
    candidate_area_ratio = float(polygon_area(corners_resized) / image_area)
    candidate_width_ratio = float(bbox_w / max(1.0, resized.shape[1]))
    candidate_height_ratio = float(bbox_h / max(1.0, resized.shape[0]))
    candidate_aspect_ratio = float(_aspect_ratio_of_quad(corners_resized))
    candidate_aspect_deviation = float(
        abs(candidate_aspect_ratio - CARD_TARGET_ASPECT_RATIO)
    )
    candidate_touch_metrics = _candidate_outer_bbox_metrics(
        corners_resized,
        outer_bbox=estimated_outer_bbox,
        image_shape=resized.shape[:2],
        metric_prefix="candidate",
    )

    metrics: Dict[str, object] = {
        "scale": float(scale),
        "detector_mode": detection_metrics.get("detector_mode", "binary_bbox"),
        "best_score": float(best_score),
        "initial_candidate_score": float(best_score),
        "refined_candidate_score": float(best_score),
        "used_fallback": float(1.0 if used_fallback else 0.0),
        "hard_rules_ok": 1.0,
        "yolo_available": float(1.0 if yolo_status.get("available") else 0.0),
        "yolo_used": 0.0,
        "fallback_is_weak": float(fallback_is_weak),
        "fallback_trigger_reason": fallback_trigger_reason,
        "fallback_source": fallback_source,
        "accepted_estimated_outer_bbox": float(
            1.0 if (estimated_outer_bbox is not None and not used_fallback) else 0.0
        ),
        "accepted_estimated_outer_bbox_score": float(
            detection_metrics.get("estimated_outer_card_bbox_score", 0.0)
        ),
        "num_candidates": float(detection_metrics.get("candidate_mask_components", 0.0)),
        "num_ranked_candidates": float(
            detection_metrics.get("candidate_mask_components", 0.0)
        ),
        "candidate_area_ratio": candidate_area_ratio,
        "candidate_bbox_width_ratio": candidate_width_ratio,
        "candidate_bbox_height_ratio": candidate_height_ratio,
        "candidate_aspect_ratio": candidate_aspect_ratio,
        "candidate_aspect_ratio_deviation": candidate_aspect_deviation,
        "candidate_geometry_valid": float(
            1.0 if candidate_aspect_deviation <= 0.12 else 0.0
        ),
        "candidate_min_side_support": float(
            min(
                final_support_metrics.get("candidate_left_border_support", 0.0),
                final_support_metrics.get("candidate_right_border_support", 0.0),
                final_support_metrics.get("candidate_top_border_support", 0.0),
                final_support_metrics.get("candidate_bottom_border_support", 0.0),
            )
        ),
        "candidate_size_plausibility_score": float(
            np.clip(
                0.40
                * min(1.0, candidate_area_ratio / max(CARD_MIN_AREA_RATIO, 1e-6))
                + 0.30
                * min(1.0, candidate_width_ratio / max(CARD_MIN_WIDTH_RATIO, 1e-6))
                + 0.30
                * min(1.0, candidate_height_ratio / max(CARD_MIN_HEIGHT_RATIO, 1e-6)),
                0.0,
                1.0,
            )
        ),
        "candidate_evidence_coverage_score": float(
            detection_metrics.get("component_fill_ratio", 0.0)
        ),
        "candidate_rejection_reason": "" if not used_fallback else fallback_trigger_reason,
        "candidate_final_silhouette_score": float(final_silhouette_score),
        "detection_confidence": float(
            np.clip(
                0.45 * best_score
                + 0.30 * final_silhouette_score
                + 0.15 * float(
                    detection_metrics.get("candidate_rectangularity_score", 0.0)
                )
                + 0.10 * float(
                    detection_metrics.get("candidate_size_plausibility_score", 0.0)
                ),
                0.0,
                1.0,
            )
        ),
        "weak_detection": float(
            1.0
            if (
                used_fallback
                or best_score < 0.62
                or final_silhouette_score < 0.58
                or float(detection_metrics.get("candidate_hard_rules_ok", 0.0)) < 0.5
            )
            else 0.0
        ),
        "prewarp_stage_input_normalized": 1.0,
        "prewarp_stage_views_built": 1.0,
        "prewarp_stage_candidates_generated": 1.0,
        "prewarp_stage_candidate_ranked": float(
            1.0 if estimated_outer_bbox is not None else 0.0
        ),
        "prewarp_stage_fallback_applied": float(1.0 if used_fallback else 0.0),
        "refinement_applied": 0.0,
        "refinement_improved": 0.0,
        "refined_candidate_rejection_reason": "simplified_pipeline_disabled",
        "coarse_quad_found": coarse_quad_found,
        "roi_quad_found": roi_quad_found,
        "final_quad_found": final_quad_found,
        "final_quad_source": final_quad_source,
        "final_quad_replaced_previous": float(final_quad_replaced_previous),
        "final_quad_selection_reason": final_quad_selection_reason,
        "coarse_border_touch_rejected": float(
            detection_metrics.get("outer_candidate_border_touch_rejected", 0.0)
        ),
        "coarse_main_component_area_ratio": float(
            detection_metrics.get("outer_candidate_main_component_area_ratio", 0.0)
        ),
        "coarse_component_count_after_cleanup": float(
            detection_metrics.get("outer_candidate_component_count_after_cleanup", 0.0)
        ),
    }
    metrics.update(detection_metrics)
    metrics.update(projection_metrics)
    metrics.update(bbox_selection_metrics)
    metrics.update(roi_metrics)
    metrics.update(binary_rescue_metrics)
    metrics.update(candidate_touch_metrics)
    metrics.update(final_support_metrics)

    if yolo_bbox is not None:
        metrics.update(
            {
                "yolo_confidence": float(yolo_bbox["confidence"]),
                "yolo_bbox_x1": float(yolo_bbox["x1"]),
                "yolo_bbox_y1": float(yolo_bbox["y1"]),
                "yolo_bbox_x2": float(yolo_bbox["x2"]),
                "yolo_bbox_y2": float(yolo_bbox["y2"]),
            }
        )

    return metrics


def detect_card_contour(
    image: np.ndarray,
    min_candidate_score: float = CARD_OUTER_MIN_CANDIDATE_SCORE,
) -> Dict:
    input_stage = _normalize_detection_input(image)
    resized = input_stage["resized"]
    scale = input_stage["scale"]
    yolo_status = input_stage["yolo_status"]
    yolo_bbox = input_stage["yolo_bbox"]
    yolo_bbox_xyxy = input_stage["yolo_bbox_xyxy"]

    views_stage = _build_detection_views(resized)
    candidate_stage = _generate_outer_candidates(
        binary_view=views_stage["binary_view"],
        image_shape=resized.shape[:2],
        edge_map=views_stage["edge_map"],
        structural_mask=views_stage["structural_mask"],
        hsv_view=views_stage["hsv_view"],
        lab_view=views_stage["lab_view"],
        yolo_bbox_xyxy=yolo_bbox_xyxy,
    )
    selection_stage = _select_ranked_outer_candidate(
        candidate_stage=candidate_stage,
        image_shape=resized.shape[:2],
        binary_view=views_stage["binary_view"],
    )
    roi_stage = _attempt_roi_second_pass(
        resized=resized,
        views_stage=views_stage,
        candidate_stage=candidate_stage,
        selection_stage=selection_stage,
        yolo_bbox_xyxy=yolo_bbox_xyxy,
    )
    binary_rescue_stage = _attempt_pre_fallback_binary_rescue(
        views_stage=views_stage,
        candidate_stage=candidate_stage,
        selection_stage=selection_stage,
        roi_stage=roi_stage,
    )
    pre_fallback_quad = roi_stage["detected_quad"]
    if pre_fallback_quad is None and binary_rescue_stage.get("detected_quad") is not None:
        pre_fallback_quad = binary_rescue_stage["detected_quad"]
    fallback_stage = _resolve_final_candidate_quad(
        detected_quad=pre_fallback_quad,
        estimated_outer_bbox=selection_stage["estimated_outer_bbox"],
        image_shape=resized.shape[:2],
    )
    projection_stage = _project_detection_to_original_scale(
        final_quad=fallback_stage["final_quad"],
        scale=scale,
    )

    final_support_metrics, final_overlay_points, final_silhouette_score = (
        _silhouette_support_scores(
            projection_stage["corners_resized"],
            structural_mask=views_stage["structural_mask"],
            metric_prefix="candidate",
        )
    )
    support_stage = {
        "final_support_metrics": final_support_metrics,
        "final_overlay_points": final_overlay_points,
        "final_silhouette_score": final_silhouette_score,
    }

    debug_maps = _build_detection_debug_images(
        resized=resized,
        views_stage=views_stage,
        candidate_stage=candidate_stage,
        selection_stage=selection_stage,
        roi_stage=roi_stage,
        binary_rescue_stage=binary_rescue_stage,
        projection_stage=projection_stage,
        support_stage=support_stage,
        yolo_bbox_xyxy=yolo_bbox_xyxy,
    )
    metrics = _build_detection_metrics(
        resized=resized,
        scale=scale,
        yolo_status=yolo_status,
        yolo_bbox=yolo_bbox,
        selection_stage=selection_stage,
        roi_stage=roi_stage,
        binary_rescue_stage=binary_rescue_stage,
        fallback_stage=fallback_stage,
        projection_stage=projection_stage,
        support_stage=support_stage,
        candidate_stage=candidate_stage,
        min_candidate_score=min_candidate_score,
    )

    result = CardDetectionResult(
        success=True,
        contour=projection_stage["contour_original"],
        corners=projection_stage["corners_original"],
        used_fallback=fallback_stage["used_fallback"],
        debug_images=debug_maps,
        metrics=metrics,
    )
    return result.to_dict()

def find_card_contour(image: np.ndarray):
    result = detect_card_contour(image)
    contour = result.get("contour", None)
    used_fallback = result.get("used_fallback", True)
    debug_images = result.get("debug_images", {})
    edges = debug_images.get("canny_closed")
    if edges is None:
        edges = debug_images.get("canny")
    if edges is None:
        edges = np.zeros(image.shape[:2], dtype=np.uint8)
    return contour, edges, used_fallback


def draw_contour(
    image: np.ndarray,
    contour: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    output = image.copy()
    if contour is None:
        return output
    contour_to_draw = np.asarray(contour, dtype=np.int32)
    if contour_to_draw.ndim == 2:
        contour_to_draw = contour_to_draw.reshape(-1, 1, 2)
    cv2.polylines(output, [contour_to_draw], True, color, thickness)
    return output
