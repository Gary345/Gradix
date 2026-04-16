import cv2
import numpy as np

from src.utils.image_utils import ensure_bgr_uint8


def _find_content_bbox(image_bgr: np.ndarray) -> tuple[int, int, int, int]:
    image_bgr = ensure_bgr_uint8(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    abs_x = np.abs(grad_x)
    abs_y = np.abs(grad_y)

    col_profile = abs_x.mean(axis=0)
    row_profile = abs_y.mean(axis=1)

    w = gray.shape[1]
    h = gray.shape[0]

    # suavizar perfiles
    col_profile = cv2.GaussianBlur(col_profile.reshape(1, -1), (1, 31), 0).ravel()
    row_profile = cv2.GaussianBlur(row_profile.reshape(-1, 1), (31, 1), 0).ravel()

    # buscar dentro de una región razonable, evitando pegarse al borde externo
    x_left_zone = col_profile[int(w * 0.05):int(w * 0.35)]
    x_right_zone = col_profile[int(w * 0.65):int(w * 0.95)]
    y_top_zone = row_profile[int(h * 0.05):int(h * 0.25)]
    y_bottom_zone = row_profile[int(h * 0.75):int(h * 0.95)]

    x_min = int(w * 0.05) + int(np.argmax(x_left_zone))
    x_max = int(w * 0.65) + int(np.argmax(x_right_zone))
    y_min = int(h * 0.05) + int(np.argmax(y_top_zone))
    y_max = int(h * 0.75) + int(np.argmax(y_bottom_zone))

    if x_max <= x_min or y_max <= y_min:
        return int(w * 0.12), int(h * 0.10), int(w * 0.88), int(h * 0.90)

    return x_min, y_min, x_max, y_max


def _balance_score(a: float, b: float) -> float:
    """
    Devuelve un score 0..1 donde 1 es perfecto balance.
    """
    total = a + b
    if total <= 0:
        return 0.0

    diff_ratio = abs(a - b) / total
    return max(0.0, 1.0 - diff_ratio)


def extract_centering_features(image_bgr: np.ndarray) -> dict:
    image_bgr = ensure_bgr_uint8(image_bgr)
    h, w = image_bgr.shape[:2]
    x_min, y_min, x_max, y_max = _find_content_bbox(image_bgr)

    left_margin = float(x_min)
    right_margin = float(w - x_max)
    top_margin = float(y_min)
    bottom_margin = float(h - y_max)

    horizontal_balance = _balance_score(left_margin, right_margin)
    vertical_balance = _balance_score(top_margin, bottom_margin)

    overall_centering = (horizontal_balance + vertical_balance) / 2.0

    return {
        "content_bbox": {
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max),
        },
        "left_margin": round(left_margin, 2),
        "right_margin": round(right_margin, 2),
        "top_margin": round(top_margin, 2),
        "bottom_margin": round(bottom_margin, 2),
        "horizontal_balance": round(horizontal_balance, 3),
        "vertical_balance": round(vertical_balance, 3),
        "overall_centering": round(overall_centering, 3),
    }


def draw_content_bbox(image_bgr: np.ndarray, bbox: dict) -> np.ndarray:
    image_bgr = ensure_bgr_uint8(image_bgr)
    output = image_bgr.copy()

    pt1 = (bbox["x_min"], bbox["y_min"])
    pt2 = (bbox["x_max"], bbox["y_max"])

    cv2.rectangle(output, pt1, pt2, (255, 255, 255), 3)

    return output
