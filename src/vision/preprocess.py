import cv2
import numpy as np

from src.utils.image_utils import ensure_bgr_uint8


def resize_image(image: np.ndarray, max_width: int = 900) -> np.ndarray:
    image = ensure_bgr_uint8(image)
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    image = ensure_bgr_uint8(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def aplicar_clahe_color(image_bgr):
    """Mejora el contraste local sin saturar la imagen."""
    image_bgr = ensure_bgr_uint8(image_bgr)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
