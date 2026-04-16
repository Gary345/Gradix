from typing import Optional

import cv2
import numpy as np
from PIL import Image


def load_pil_image(uploaded_file) -> Optional[Image.Image]:
    if uploaded_file is None:
        return None
    return Image.open(uploaded_file).convert("RGB")


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    """Convierte RGB a BGR (para OpenCV)."""
    if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
        return image_rgb
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convierte BGR a RGB (para mostrar en Streamlit)."""
    if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
