import numpy as np
from src.utils.helpers import polygon_area

def compute_card_coverage(contour, image_shape) -> float:
    if contour is None:
        return 0.0

    image_height, image_width = image_shape[:2]
    image_area = float(image_height * image_width)

    contour_area = float(abs(polygon_area(contour.reshape(-1, 2))))
    if image_area <= 0:
        return 0.0

    return contour_area / image_area


def classify_capture_quality(
    contour,
    image_shape,
    aspect_ratio: float,
    capture_score: float,
    used_fallback: bool,
) -> dict:
    coverage = compute_card_coverage(contour, image_shape)

    aspect_ok = 0.65 <= aspect_ratio <= 0.78
    coverage_good = coverage >= 0.35
    coverage_medium = coverage >= 0.22

    if used_fallback:
        level = "deficiente"
        message = "La detección usó un contorno de respaldo. Mejora el contraste contra el fondo."
    elif coverage > 0.95:
        # ¡NUEVO! Detectar si el usuario acercó demasiado la cámara
        level = "mejorable" if capture_score >= 5.0 else "deficiente"
        message = (
            "La carta no tiene márgenes visibles. Aleja un poco la cámara "
            "para que el borde real sea visible contra un fondo oscuro."
        )
    elif aspect_ok and coverage_good and capture_score >= 7.0:
        level = "buena"
        message = (
            "La captura es adecuada para análisis preliminar. "
            "La carta ocupa suficiente espacio y la calidad visual es razonable."
        )
    elif aspect_ok and coverage_medium and capture_score >= 5.0:
        level = "mejorable"
        message = (
            "La captura puede procesarse, pero conviene acercar más la carta, "
            "reducir reflejos o mejorar la iluminación."
        )
    else:
        level = "deficiente"
        message = (
            "La captura no es ideal para análisis confiable. "
            "Acerca más la carta, usa mejor iluminación y un fondo más limpio."
        )

    return {
        "coverage_ratio": round(coverage, 3),
        "aspect_ok": aspect_ok,
        "used_fallback": used_fallback,
        "capture_assessment": level,
        "capture_message": message,
    }