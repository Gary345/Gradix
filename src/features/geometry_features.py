import numpy as np
from src.utils.helpers import clamp, polygon_area




def compute_card_coverage(contour, image_shape) -> float:
    if contour is None:
        return 0.0

    image_height, image_width = image_shape[:2]
    image_area = float(image_height * image_width)

    points = contour.reshape(-1, 2)
    contour_area = polygon_area(points)

    if image_area <= 0:
        return 0.0

    return float(contour_area / image_area)


def compute_aspect_ratio_quality(aspect_ratio: float) -> float:
    """
    Carta vertical esperada aproximadamente entre 0.65 y 0.78.
    Mejor puntuación cerca de 0.715.
    """
    ideal_ratio = 0.715
    max_distance = 0.10

    distance = abs(aspect_ratio - ideal_ratio)
    score = 1.0 - (distance / max_distance)

    return clamp(score, 0.0, 1.0)


def extract_geometry_features(
    contour,
    image_shape,
    warped_aspect_ratio: float,
    used_fallback: bool,
) -> dict:
    coverage_ratio = compute_card_coverage(contour, image_shape)
    aspect_ratio_quality = compute_aspect_ratio_quality(warped_aspect_ratio)

    return {
        "coverage_ratio": round(coverage_ratio, 3),
        "aspect_ratio_quality": round(aspect_ratio_quality, 3),
        "used_fallback": used_fallback,
    }