from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from src.utils.helpers import order_points
from src.utils.image_utils import ensure_bgr_uint8


@dataclass
class PerspectiveResult:
    success: bool
    warped_image: Optional[np.ndarray]
    ordered_corners: Optional[np.ndarray]
    transform_matrix: Optional[np.ndarray]
    output_size: Tuple[int, int]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def _compute_side_lengths(ordered_corners: np.ndarray) -> Dict[str, float]:
    tl, tr, br, bl = ordered_corners

    width_top = _distance(tl, tr)
    width_bottom = _distance(bl, br)
    height_left = _distance(tl, bl)
    height_right = _distance(tr, br)

    return {
        "width_top": width_top,
        "width_bottom": width_bottom,
        "height_left": height_left,
        "height_right": height_right,
    }


def _estimate_raw_size(ordered_corners: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
    lengths = _compute_side_lengths(ordered_corners)

    raw_width = max(1.0, (lengths["width_top"] + lengths["width_bottom"]) / 2.0)
    raw_height = max(1.0, (lengths["height_left"] + lengths["height_right"]) / 2.0)

    metrics = {
        **lengths,
        "raw_width": float(raw_width),
        "raw_height": float(raw_height),
        "raw_aspect_ratio": float(raw_width / raw_height),
    }
    return raw_width, raw_height, metrics


def _stabilize_output_size(
    raw_width: float,
    raw_height: float,
    target_aspect_ratio: float = 0.714,
    min_output_height: int = 700,
    max_output_height: int = 1200,
) -> Tuple[int, int, Dict[str, float]]:
    """
    target_aspect_ratio = width / height
    """
    raw_ratio = raw_width / max(1.0, raw_height)

    # Tomamos la altura como referencia más estable para cartas verticales.
    out_height = int(round(raw_height))
    out_height = max(min_output_height, min(max_output_height, out_height))

    # Si el ratio bruto está razonablemente cerca del esperado, usamos el target.
    # Si está demasiado lejos, igual usamos el target para estabilizar la rectificación MVP.
    out_width = int(round(out_height * target_aspect_ratio))

    out_width = max(300, out_width)
    out_height = max(420, out_height)

    metrics = {
        "target_aspect_ratio": float(target_aspect_ratio),
        "raw_aspect_ratio_before_stabilization": float(raw_ratio),
        "stabilized_width": float(out_width),
        "stabilized_height": float(out_height),
        "stabilized_aspect_ratio": float(out_width / out_height),
    }
    return out_width, out_height, metrics


def _expand_quad_about_center(ordered_corners: np.ndarray, expand_ratio: float = 0.012) -> np.ndarray:
    """
    Expande ligeramente el cuadrilátero para no recortar bordes en el warp.
    """
    quad = ordered_corners.astype(np.float32)
    center = np.mean(quad, axis=0, keepdims=True)
    expanded = center + (quad - center) * (1.0 + expand_ratio)
    return expanded.astype(np.float32)


def warp_card_perspective(
    image: np.ndarray,
    corners: np.ndarray,
    target_aspect_ratio: float = 0.714,
    expand_ratio: float = 0.012,
    min_output_height: int = 700,
    max_output_height: int = 1200,
) -> Dict:
    """
    Rectifica una carta usando 4 esquinas detectadas.

    Parámetros
    ----------
    image : np.ndarray
        Imagen original BGR.
    corners : np.ndarray
        Array de 4 puntos.
    target_aspect_ratio : float
        Ratio objetivo width/height para estabilizar salida.
    expand_ratio : float
        Expansión ligera del cuadrilátero para no recortar bordes.
    """
    bgr = ensure_bgr_uint8(image)

    if corners is None:
        raise ValueError("corners is None")

    ordered = order_points(corners)
    expanded = _expand_quad_about_center(ordered, expand_ratio=expand_ratio)

    raw_width, raw_height, raw_metrics = _estimate_raw_size(expanded)

    out_w, out_h, size_metrics = _stabilize_output_size(
        raw_width=raw_width,
        raw_height=raw_height,
        target_aspect_ratio=target_aspect_ratio,
        min_output_height=min_output_height,
        max_output_height=max_output_height,
    )

    dst = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(expanded.astype(np.float32), dst)
    warped = cv2.warpPerspective(
        bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    metrics = {
        "expand_ratio": float(expand_ratio),
        **raw_metrics,
        **size_metrics,
        "output_width": float(out_w),
        "output_height": float(out_h),
    }

    result = PerspectiveResult(
        success=True,
        warped_image=warped,
        ordered_corners=expanded.astype(np.float32),
        transform_matrix=M.astype(np.float32),
        output_size=(int(out_w), int(out_h)),
        metrics=metrics,
    )
    return result.to_dict()

def four_point_transform(
    image: np.ndarray,
    corners: np.ndarray,
    target_aspect_ratio: float = 0.714,
    expand_ratio: float = 0.012,
):
    """
    Wrapper de compatibilidad con la versión previa del proyecto.

    Retorna solo la imagen rectificada, como esperaba la app original.
    """
    result = warp_card_perspective(
        image=image,
        corners=corners,
        target_aspect_ratio=target_aspect_ratio,
        expand_ratio=expand_ratio,
    )
    return result["warped_image"]
