# src/utils/helpers.py
"""Funciones auxiliares reutilizables en todo el proyecto."""

import numpy as np


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Limita un valor entre min_value y max_value.
    
    Args:
        value: Valor a limitar
        min_value: Mínimo permitido
        max_value: Máximo permitido
    
    Returns:
        Valor limitado en el rango [min_value, max_value]
    
    Examples:
        >>> clamp(5, 0, 10)
        5
        >>> clamp(15, 0, 10)
        10
    """
    return max(min_value, min(value, max_value))


def safe_float(value) -> float:
    """
    Convierte un valor a float de manera segura, manejando NaN e inf.
    
    Args:
        value: Valor a convertir (puede ser None, NaN, inf, etc.)
    
    Returns:
        Float válido o 0.0 si hay error
    """
    if value is None:
        return 0.0
    
    try:
        if isinstance(value, (np.floating, np.integer)):
            value = float(value)
        if np.isnan(value) or np.isinf(value):
            return 0.0
    except (TypeError, ValueError):
        return 0.0
    
    return float(value)


def normalize_to_0_1(
    value: float, 
    min_ref: float, 
    max_ref: float
) -> float:
    """
    Normaliza un valor al rango [0, 1] basado en referencias.
    
    Args:
        value: Valor a normalizar
        min_ref: Valor de referencia mínimo (mapea a 0)
        max_ref: Valor de referencia máximo (mapea a 1)
    
    Returns:
        Valor normalizado en [0, 1]
    
    Examples:
        >>> normalize_to_0_1(50, 0, 100)
        0.5
        >>> normalize_to_0_1(150, 0, 100)  # Clampea a 1
        1.0
    """
    if max_ref <= min_ref:
        return 0.0
    
    normalized = (value - min_ref) / (max_ref - min_ref)
    return clamp(normalized, 0.0, 1.0)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Ordena 4 puntos en: top-left, top-right, bottom-right, bottom-left.
    
    Usa sum() y diff() para identificar posiciones.
    
    Args:
        pts: Array de puntos (N, 2) o flatten
    
    Returns:
        Array ordenado de shape (4, 2)
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    
    return np.array(
        [top_left, top_right, bottom_right, bottom_left], 
        dtype=np.float32
    )


def polygon_area(points: np.ndarray) -> float:
    """
    Calcula el área de un polígono usando la fórmula de Shoelace.
    
    Args:
        points: Array de shape (N, 2)
    
    Returns:
        Área del polígono
    """
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
