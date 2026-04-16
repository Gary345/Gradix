from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.config.settings import (
    YOLO_CARD_CONFIDENCE,
    YOLO_CARD_ENABLED,
    YOLO_CARD_IOU,
    YOLO_CARD_MODEL_PATH,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_YOLO_MODEL = None
_YOLO_STATUS: Optional[Dict[str, object]] = None


def _load_model() -> Dict[str, object]:
    global _YOLO_MODEL, _YOLO_STATUS

    if _YOLO_STATUS is not None:
        return _YOLO_STATUS

    if not YOLO_CARD_ENABLED:
        _YOLO_STATUS = {"available": False, "reason": "disabled"}
        return _YOLO_STATUS

    model_path = Path(YOLO_CARD_MODEL_PATH)
    if not model_path.exists():
        _YOLO_STATUS = {
            "available": False,
            "reason": "missing_model_path",
            "model_path": str(model_path),
        }
        return _YOLO_STATUS

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        _YOLO_STATUS = {
            "available": False,
            "reason": "ultralytics_not_installed",
            "error": str(exc),
            "model_path": str(model_path),
        }
        return _YOLO_STATUS

    try:
        _YOLO_MODEL = YOLO(str(model_path))
        _YOLO_STATUS = {
            "available": True,
            "reason": "loaded",
            "model_path": str(model_path),
        }
        logger.info("YOLO detector loaded from %s", model_path)
        return _YOLO_STATUS
    except Exception as exc:
        _YOLO_STATUS = {
            "available": False,
            "reason": "model_load_error",
            "error": str(exc),
            "model_path": str(model_path),
        }
        logger.warning("Could not load YOLO model from %s: %s", model_path, exc)
        return _YOLO_STATUS


def get_yolo_status() -> Dict[str, object]:
    return _load_model().copy()


def detect_card_bbox(image_bgr: np.ndarray) -> Optional[Dict[str, float]]:
    status = _load_model()
    if not status.get("available"):
        return None

    if image_bgr is None or image_bgr.size == 0:
        return None

    try:
        results = _YOLO_MODEL.predict(
            source=image_bgr,
            conf=YOLO_CARD_CONFIDENCE,
            iou=YOLO_CARD_IOU,
            verbose=False,
            max_det=1,
        )
    except Exception as exc:
        logger.warning("YOLO inference failed: %s", exc)
        return None

    if not results:
        return None

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    best_box = boxes[0]
    xyxy = best_box.xyxy[0].detach().cpu().numpy().astype(float)
    conf = float(best_box.conf[0].detach().cpu().numpy())
    cls = float(best_box.cls[0].detach().cpu().numpy()) if best_box.cls is not None else 0.0

    x1, y1, x2, y2 = xyxy.tolist()
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "confidence": conf,
        "class_id": cls,
    }
