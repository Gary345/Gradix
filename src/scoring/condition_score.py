from src.utils.helpers import clamp

def normalize_blur_score(blur_score: float) -> float:
    normalized = blur_score / 400.0
    return clamp(normalized, 0.0, 1.0)


def normalize_brightness_score(brightness_score: float) -> float:
    distance = abs(brightness_score - 150.0)
    normalized = 1.0 - (distance / 150.0)
    return clamp(normalized, 0.0, 1.0)


def normalize_contrast_score(contrast_score: float) -> float:
    normalized = contrast_score / 80.0
    return clamp(normalized, 0.0, 1.0)


def compute_capture_quality_score(features: dict) -> dict:
    blur_component = normalize_blur_score(features["blur_score"])
    brightness_component = normalize_brightness_score(features["brightness_score"])
    contrast_component = normalize_contrast_score(features["contrast_score"])

    weighted_score = (
        0.50 * blur_component
        + 0.25 * brightness_component
        + 0.25 * contrast_component
    )

    final_score = 1.0 + (weighted_score * 9.0)

    return {
        "capture_quality_score": round(final_score, 2),
        "capture_quality_components": {
            "blur_component": round(blur_component, 3),
            "brightness_component": round(brightness_component, 3),
            "contrast_component": round(contrast_component, 3),
        },
    }


def normalize_capture_score(capture_score: float) -> float:
    normalized = (capture_score - 1.0) / 9.0
    return clamp(normalized, 0.0, 1.0)


def normalize_coverage_ratio(coverage_ratio: float) -> float:
    """
    Queremos que la carta ocupe una porción importante del frame.
    """
    normalized = coverage_ratio / 0.60
    return clamp(normalized, 0.0, 1.0)


def compute_preliminary_gradix_score(
    capture_quality_score: float,
    geometry_features: dict,
) -> dict:
    capture_component = normalize_capture_score(capture_quality_score)
    coverage_component = normalize_coverage_ratio(geometry_features["coverage_ratio"])
    aspect_component = geometry_features["aspect_ratio_quality"]

    fallback_penalty = 0.20 if geometry_features["used_fallback"] else 0.0

    weighted_score = (
        0.55 * capture_component
        + 0.30 * coverage_component
        + 0.15 * aspect_component
    )

    weighted_score = weighted_score - fallback_penalty
    weighted_score = clamp(weighted_score, 0.0, 1.0)

    final_score = 1.0 + (weighted_score * 9.0)

    return {
        "gradix_preliminary_score": round(final_score, 2),
        "gradix_preliminary_components": {
            "capture_component": round(capture_component, 3),
            "coverage_component": round(coverage_component, 3),
            "aspect_component": round(aspect_component, 3),
            "fallback_penalty": round(fallback_penalty, 3),
        },
    }


def compute_centering_score(centering_features: dict) -> dict:
    horizontal_component = centering_features["horizontal_balance"]
    vertical_component = centering_features["vertical_balance"]

    weighted_score = 0.5 * horizontal_component + 0.5 * vertical_component
    final_score = 1.0 + (weighted_score * 9.0)

    return {
        "centering_score": round(final_score, 2),
        "centering_components": {
            "horizontal_component": round(horizontal_component, 3),
            "vertical_component": round(vertical_component, 3),
        },
    }


def compute_gradix_condition_stub(
    preliminary_gradix_score: float,
    centering_score: float,
) -> dict:
    """
    Score stub inicial de condición:
    mezcla score preliminar del pipeline con centrado aproximado.
    """
    prelim_norm = clamp((preliminary_gradix_score - 1.0) / 9.0, 0.0, 1.0)
    centering_norm = clamp((centering_score - 1.0) / 9.0, 0.0, 1.0)

    weighted_score = 0.65 * prelim_norm + 0.35 * centering_norm
    final_score = 1.0 + (weighted_score * 9.0)

    return {
        "gradix_condition_stub": round(final_score, 2),
        "gradix_condition_components": {
            "preliminary_component": round(prelim_norm, 3),
            "centering_component": round(centering_norm, 3),
        },
    }


# ============================================================
# NUEVO BLOQUE MVP: EDGE SCORE
# ============================================================

def normalize_edge_score_0_1(edge_score: float) -> float:
    """
    Convierte edge_score de escala 0-100 a 0-1.
    """
    normalized = edge_score / 100.0
    return clamp(normalized, 0.0, 1.0)


def normalize_edge_confidence(edge_confidence: float) -> float:
    return clamp(edge_confidence, 0.0, 1.0)


def compute_edge_score(edge_features: dict) -> dict:
    """
    Toma el output de src/features/edge_features.py y lo convierte
    a una escala Gradix 1-10 para mantener consistencia con el resto
    del pipeline.

    Espera llaves como:
    - edge_score
    - edge_confidence
    - top_edge_score
    - bottom_edge_score
    - left_edge_score
    - right_edge_score
    """
    edge_score_raw = edge_features["edge_score"]
    edge_confidence = edge_features.get("edge_confidence", 1.0)

    edge_component = normalize_edge_score_0_1(edge_score_raw)
    confidence_component = normalize_edge_confidence(edge_confidence)

    # Si la confianza baja, moderamos un poco el score efectivo
    adjusted_component = edge_component * (0.75 + 0.25 * confidence_component)
    adjusted_component = clamp(adjusted_component, 0.0, 1.0)

    final_score = 1.0 + (adjusted_component * 9.0)

    return {
        "gradix_edge_score": round(final_score, 2),
        "gradix_edge_components": {
            "edge_component": round(edge_component, 3),
            "confidence_component": round(confidence_component, 3),
            "adjusted_edge_component": round(adjusted_component, 3),
            "top_edge_score": round(edge_features.get("top_edge_score", 0.0), 2),
            "bottom_edge_score": round(edge_features.get("bottom_edge_score", 0.0), 2),
            "left_edge_score": round(edge_features.get("left_edge_score", 0.0), 2),
            "right_edge_score": round(edge_features.get("right_edge_score", 0.0), 2),
        },
    }


def compute_gradix_condition_stub_v2(
    preliminary_gradix_score: float,
    centering_score: float,
    gradix_edge_score: float,
    edge_confidence: float = 1.0,
) -> dict:
    """
    Versión V2 del condition stub:
    mezcla score preliminar + centrado + bordes.

    Todas las entradas están en escala 1-10, excepto edge_confidence
    que va de 0 a 1.
    """
    prelim_norm = clamp((preliminary_gradix_score - 1.0) / 9.0, 0.0, 1.0)
    centering_norm = clamp((centering_score - 1.0) / 9.0, 0.0, 1.0)
    edge_norm = clamp((gradix_edge_score - 1.0) / 9.0, 0.0, 1.0)
    edge_conf_norm = clamp(edge_confidence, 0.0, 1.0)

    # Peso base MVP
    preliminary_weight = 0.50
    centering_weight = 0.20
    edge_weight = 0.30

    # Si edge_confidence baja, reducimos ligeramente el peso efectivo de bordes
    adjusted_edge_weight = edge_weight * (0.60 + 0.40 * edge_conf_norm)
    redistributed_weight = edge_weight - adjusted_edge_weight
    adjusted_preliminary_weight = preliminary_weight + redistributed_weight

    weighted_score = (
        adjusted_preliminary_weight * prelim_norm
        + centering_weight * centering_norm
        + adjusted_edge_weight * edge_norm
    )
    weighted_score = clamp(weighted_score, 0.0, 1.0)

    final_score = 1.0 + (weighted_score * 9.0)

    return {
        "gradix_condition_stub_v2": round(final_score, 2),
        "gradix_condition_v2_components": {
            "preliminary_component": round(prelim_norm, 3),
            "centering_component": round(centering_norm, 3),
            "edge_component": round(edge_norm, 3),
            "edge_confidence": round(edge_conf_norm, 3),
            "preliminary_weight_used": round(adjusted_preliminary_weight, 3),
            "centering_weight_used": round(centering_weight, 3),
            "edge_weight_used": round(adjusted_edge_weight, 3),
        },
    }
# ============================================================
# NUEVO BLOQUE MVP: CORNER SCORE
# ============================================================

def normalize_corner_score_0_1(corner_score_raw: float) -> float:
    """
    Convierte corner_score_raw de escala 0-100 a 0-1.
    """
    normalized = corner_score_raw / 100.0
    return clamp(normalized, 0.0, 1.0)


def normalize_corner_confidence(corner_confidence: float) -> float:
    return clamp(corner_confidence, 0.0, 1.0)


def compute_corner_score(corner_features: dict) -> dict:
    """
    Toma el output de src/features/corner_features.py y lo convierte
    a escala Gradix 1-10.

    Espera llaves como:
    - corner_score_raw
    - corner_confidence
    - top_left_corner_score
    - top_right_corner_score
    - bottom_left_corner_score
    - bottom_right_corner_score
    """
    corner_score_raw = corner_features["corner_score_raw"]
    corner_confidence = corner_features.get("corner_confidence", 1.0)

    corner_component = normalize_corner_score_0_1(corner_score_raw)
    confidence_component = normalize_corner_confidence(corner_confidence)

    adjusted_component = corner_component * (0.75 + 0.25 * confidence_component)
    adjusted_component = clamp(adjusted_component, 0.0, 1.0)

    final_score = 1.0 + (adjusted_component * 9.0)

    return {
        "gradix_corner_score": round(final_score, 2),
        "gradix_corner_components": {
            "corner_component": round(corner_component, 3),
            "confidence_component": round(confidence_component, 3),
            "adjusted_corner_component": round(adjusted_component, 3),
            "top_left_corner_score": round(corner_features.get("top_left_corner_score", 0.0), 2),
            "top_right_corner_score": round(corner_features.get("top_right_corner_score", 0.0), 2),
            "bottom_left_corner_score": round(corner_features.get("bottom_left_corner_score", 0.0), 2),
            "bottom_right_corner_score": round(corner_features.get("bottom_right_corner_score", 0.0), 2),
        },
    }


def compute_gradix_condition_stub_v3(
    preliminary_gradix_score: float,
    centering_score: float,
    gradix_edge_score: float,
    gradix_corner_score: float,
    edge_confidence: float = 1.0,
    corner_confidence: float = 1.0,
) -> dict:
    """
    Versión V3 del condition stub:
    mezcla score preliminar + centrado + bordes + esquinas.

    Entradas:
    - preliminary_gradix_score: escala 1-10
    - centering_score: escala 1-10
    - gradix_edge_score: escala 1-10
    - gradix_corner_score: escala 1-10
    - edge_confidence: 0-1
    - corner_confidence: 0-1
    """
    prelim_norm = clamp((preliminary_gradix_score - 1.0) / 9.0, 0.0, 1.0)
    centering_norm = clamp((centering_score - 1.0) / 9.0, 0.0, 1.0)
    edge_norm = clamp((gradix_edge_score - 1.0) / 9.0, 0.0, 1.0)
    corner_norm = clamp((gradix_corner_score - 1.0) / 9.0, 0.0, 1.0)

    edge_conf_norm = clamp(edge_confidence, 0.0, 1.0)
    corner_conf_norm = clamp(corner_confidence, 0.0, 1.0)

    # Pesos base MVP V3
    preliminary_weight = 0.40
    centering_weight = 0.20
    edge_weight = 0.20
    corner_weight = 0.20

    adjusted_edge_weight = edge_weight * (0.60 + 0.40 * edge_conf_norm)
    adjusted_corner_weight = corner_weight * (0.60 + 0.40 * corner_conf_norm)

    redistributed_weight = (
        (edge_weight - adjusted_edge_weight)
        + (corner_weight - adjusted_corner_weight)
    )
    adjusted_preliminary_weight = preliminary_weight + redistributed_weight

    weighted_score = (
        adjusted_preliminary_weight * prelim_norm
        + centering_weight * centering_norm
        + adjusted_edge_weight * edge_norm
        + adjusted_corner_weight * corner_norm
    )
    weighted_score = clamp(weighted_score, 0.0, 1.0)

    final_score = 1.0 + (weighted_score * 9.0)

    return {
        "gradix_condition_stub_v3": round(final_score, 2),
        "gradix_condition_v3_components": {
            "preliminary_component": round(prelim_norm, 3),
            "centering_component": round(centering_norm, 3),
            "edge_component": round(edge_norm, 3),
            "corner_component": round(corner_norm, 3),
            "edge_confidence": round(edge_conf_norm, 3),
            "corner_confidence": round(corner_conf_norm, 3),
            "preliminary_weight_used": round(adjusted_preliminary_weight, 3),
            "centering_weight_used": round(centering_weight, 3),
            "edge_weight_used": round(adjusted_edge_weight, 3),
            "corner_weight_used": round(adjusted_corner_weight, 3),
        },
    }

def normalize_whitening_score_0_1(whitening_score: float) -> float:
    """
    Convierte whitening_score de escala 0-10 a 0-1.
    """
    normalized = whitening_score / 10.0
    return clamp(normalized, 0.0, 1.0)


def normalize_surface_score_0_1(surface_score: float) -> float:
    """
    Convierte surface_score de escala 0-10 a 0-1.
    """
    normalized = surface_score / 10.0
    return clamp(normalized, 0.0, 1.0)


def normalize_whitening_confidence(whitening_confidence: float) -> float:
    return clamp(whitening_confidence, 0.0, 1.0)


def normalize_surface_confidence(surface_confidence: float) -> float:
    return clamp(surface_confidence, 0.0, 1.0)


def compute_whitening_surface_score(whitening_surface_features: dict) -> dict:
    """
    Toma el output de whitening_surface_features.py y lo convierte
    a escala Gradix 1-10 con componentes separados.

    Espera llaves como:
    - whitening_score
    - surface_score
    - whitening_surface_score
    - whitening_confidence
    - surface_confidence
    """
    whitening_score = whitening_surface_features["whitening_score"]
    surface_score = whitening_surface_features["surface_score"]
    combined_score = whitening_surface_features["whitening_surface_score"]

    whitening_confidence = whitening_surface_features.get("whitening_confidence", 1.0)
    surface_confidence = whitening_surface_features.get("surface_confidence", 1.0)

    whitening_component = normalize_whitening_score_0_1(whitening_score)
    surface_component = normalize_surface_score_0_1(surface_score)
    combined_component = clamp(combined_score / 10.0, 0.0, 1.0)

    whitening_conf_component = normalize_whitening_confidence(whitening_confidence)
    surface_conf_component = normalize_surface_confidence(surface_confidence)

    confidence_blend = 0.5 * whitening_conf_component + 0.5 * surface_conf_component

    adjusted_component = combined_component * (0.75 + 0.25 * confidence_blend)
    adjusted_component = clamp(adjusted_component, 0.0, 1.0)

    final_score = 1.0 + (adjusted_component * 9.0)

    return {
        "gradix_whitening_surface_score": round(final_score, 2),
        "gradix_whitening_surface_components": {
            "whitening_component": round(whitening_component, 3),
            "surface_component": round(surface_component, 3),
            "combined_component": round(combined_component, 3),
            "whitening_confidence": round(whitening_conf_component, 3),
            "surface_confidence": round(surface_conf_component, 3),
            "adjusted_component": round(adjusted_component, 3),
            "edge_whitening_ratio": round(
                whitening_surface_features.get("edge_whitening_ratio", 0.0), 4
            ),
            "corner_whitening_ratio": round(
                whitening_surface_features.get("corner_whitening_ratio", 0.0), 4
            ),
            "glare_ratio": round(
                whitening_surface_features.get("glare_ratio", 0.0), 4
            ),
            "texture_anomaly_ratio": round(
                whitening_surface_features.get("texture_anomaly_ratio", 0.0), 4
            ),
            "dark_spot_ratio": round(
                whitening_surface_features.get("dark_spot_ratio", 0.0), 4
            ),
        },
    }


def compute_gradix_condition_stub_v4(
    preliminary_gradix_score: float,
    centering_score: float,
    gradix_edge_score: float,
    gradix_corner_score: float,
    gradix_whitening_surface_score: float,
    edge_confidence: float = 1.0,
    corner_confidence: float = 1.0,
    whitening_confidence: float = 1.0,
    surface_confidence: float = 1.0,
) -> dict:
    """
    Versión V4 del condition stub:
    mezcla score preliminar + centrado + bordes + esquinas + whitening/surface.
    """
    prelim_norm = clamp((preliminary_gradix_score - 1.0) / 9.0, 0.0, 1.0)
    centering_norm = clamp((centering_score - 1.0) / 9.0, 0.0, 1.0)
    edge_norm = clamp((gradix_edge_score - 1.0) / 9.0, 0.0, 1.0)
    corner_norm = clamp((gradix_corner_score - 1.0) / 9.0, 0.0, 1.0)
    ws_norm = clamp((gradix_whitening_surface_score - 1.0) / 9.0, 0.0, 1.0)

    edge_conf_norm = clamp(edge_confidence, 0.0, 1.0)
    corner_conf_norm = clamp(corner_confidence, 0.0, 1.0)
    whitening_conf_norm = clamp(whitening_confidence, 0.0, 1.0)
    surface_conf_norm = clamp(surface_confidence, 0.0, 1.0)
    ws_conf_norm = 0.5 * whitening_conf_norm + 0.5 * surface_conf_norm

    # Pesos base MVP V4
    preliminary_weight = 0.25  
    centering_weight = 0.15   
    edge_weight = 0.25        
    corner_weight = 0.20      
    ws_weight = 0.15           

    adjusted_edge_weight = edge_weight * (0.60 + 0.40 * edge_conf_norm)
    adjusted_corner_weight = corner_weight * (0.60 + 0.40 * corner_conf_norm)
    adjusted_ws_weight = ws_weight * (0.60 + 0.40 * ws_conf_norm)

    redistributed_weight = (
        (edge_weight - adjusted_edge_weight)
        + (corner_weight - adjusted_corner_weight)
        + (ws_weight - adjusted_ws_weight)
    )

    adjusted_preliminary_weight = preliminary_weight + redistributed_weight

    weighted_score = (
        adjusted_preliminary_weight * prelim_norm
        + centering_weight * centering_norm
        + adjusted_edge_weight * edge_norm
        + adjusted_corner_weight * corner_norm
        + adjusted_ws_weight * ws_norm
    )
    weighted_score = clamp(weighted_score, 0.0, 1.0)

    final_score = 1.0 + (weighted_score * 9.0)

    return {
        "gradix_condition_stub_v4": round(final_score, 2),
        "gradix_condition_v4_components": {
            "preliminary_component": round(prelim_norm, 3),
            "centering_component": round(centering_norm, 3),
            "edge_component": round(edge_norm, 3),
            "corner_component": round(corner_norm, 3),
            "whitening_surface_component": round(ws_norm, 3),
            "edge_confidence": round(edge_conf_norm, 3),
            "corner_confidence": round(corner_conf_norm, 3),
            "whitening_confidence": round(whitening_conf_norm, 3),
            "surface_confidence": round(surface_conf_norm, 3),
            "preliminary_weight_used": round(adjusted_preliminary_weight, 3),
            "centering_weight_used": round(centering_weight, 3),
            "edge_weight_used": round(adjusted_edge_weight, 3),
            "corner_weight_used": round(adjusted_corner_weight, 3),
            "whitening_surface_weight_used": round(adjusted_ws_weight, 3),
        },
    }