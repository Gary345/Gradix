from __future__ import annotations

import cv2
import numpy as np


def ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be numpy.ndarray")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3) or be grayscale")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image
