import cv2
import numpy as np

from src.utils.image_utils import ensure_bgr_uint8


def compute_dimensions(image_bgr: np.ndarray) -> dict:
    image_bgr = ensure_bgr_uint8(image_bgr)
    height, width = image_bgr.shape[:2]
    aspect_ratio = width / float(height) if height > 0 else 0.0

    return {
        "width_px": int(width),
        "height_px": int(height),
        "aspect_ratio": float(aspect_ratio),
        "area_px": int(width * height),
    }


def compute_blur_score(image_bgr: np.ndarray) -> float:
    image_bgr = ensure_bgr_uint8(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(variance)


def compute_brightness_score(image_bgr: np.ndarray) -> float:
    image_bgr = ensure_bgr_uint8(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_contrast_score(image_bgr: np.ndarray) -> float:
    image_bgr = ensure_bgr_uint8(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def extract_visual_features(image_bgr: np.ndarray) -> dict:
    image_bgr = ensure_bgr_uint8(image_bgr)
    dimensions = compute_dimensions(image_bgr)
    blur_score = compute_blur_score(image_bgr)
    brightness_score = compute_brightness_score(image_bgr)
    contrast_score = compute_contrast_score(image_bgr)

    features = {
        **dimensions,
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
    }

    return features
