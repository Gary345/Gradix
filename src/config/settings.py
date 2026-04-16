import os
from pathlib import Path


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized == "":
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _get_env_float(
    name: str,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip()
    if normalized == "":
        return default

    try:
        parsed = float(normalized)
    except (TypeError, ValueError):
        return default

    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"

APP_TITLE = "Gradix"
APP_SUBTITLE = "Pre-grading visual para cartas Pokémon TCG"
SUPPORTED_IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".webp"]

YOLO_CARD_ENABLED = _get_env_bool("YOLO_CARD_ENABLED", False)
YOLO_CARD_MODEL_PATH = os.getenv(
    "YOLO_CARD_MODEL_PATH",
    str(PROJECT_ROOT / "assets" / "models" / "card_detector.pt"),
)
YOLO_CARD_CONFIDENCE = _get_env_float("YOLO_CARD_CONFIDENCE", 0.35, 0.0, 1.0)
YOLO_CARD_IOU = _get_env_float("YOLO_CARD_IOU", 0.45, 0.0, 1.0)
YOLO_CARD_BBOX_MARGIN = _get_env_float("YOLO_CARD_BBOX_MARGIN", 0.08, 0.0, 0.5)

CARD_GEOMETRY_STRICTNESS = _get_env_float("CARD_GEOMETRY_STRICTNESS", 1.0, 0.1, 3.0)
CARD_BORDER_SUPPORT_ENABLED = _get_env_bool("CARD_BORDER_SUPPORT_ENABLED", True)
CARD_BORDER_SUPPORT_BAND_WIDTH = _get_env_float(
    "CARD_BORDER_SUPPORT_BAND_WIDTH", 0.018, 0.001, 0.2
)
CARD_BORDER_SUPPORT_VERTICAL_MIN = _get_env_float(
    "CARD_BORDER_SUPPORT_VERTICAL_MIN", 0.26, 0.0, 1.0
)
CARD_MIN_AREA_RATIO = _get_env_float("CARD_MIN_AREA_RATIO", 0.16, 0.01, 1.0)
CARD_MIN_WIDTH_RATIO = _get_env_float("CARD_MIN_WIDTH_RATIO", 0.32, 0.01, 1.0)
CARD_MIN_HEIGHT_RATIO = _get_env_float("CARD_MIN_HEIGHT_RATIO", 0.42, 0.01, 1.0)
CARD_COLOR_SEPARATION_ENABLED = _get_env_bool("CARD_COLOR_SEPARATION_ENABLED", True)
CARD_COLOR_SAMPLE_OFFSET = _get_env_float("CARD_COLOR_SAMPLE_OFFSET", 0.020, 0.0, 0.2)

CARD_OUTER_MIN_CANDIDATE_SCORE = _get_env_float(
    "CARD_OUTER_MIN_CANDIDATE_SCORE", 0.42, 0.0, 1.0
)
CARD_OUTER_MIN_COMPONENT_AREA = _get_env_float(
    "CARD_OUTER_MIN_COMPONENT_AREA", 1500.0, 1.0, None
)
CARD_OUTER_MASK_LONG_KERNEL_RATIO = _get_env_float(
    "CARD_OUTER_MASK_LONG_KERNEL_RATIO", 0.12, 0.01, 0.5
)
CARD_OUTER_MASK_SHORT_KERNEL_RATIO = _get_env_float(
    "CARD_OUTER_MASK_SHORT_KERNEL_RATIO", 0.014, 0.001, 0.1
)
CARD_OUTER_MASK_SQUARE_KERNEL_RATIO = _get_env_float(
    "CARD_OUTER_MASK_SQUARE_KERNEL_RATIO", 0.03, 0.003, 0.2
)
CARD_OUTER_ENABLE_YOLO_PRIOR = _get_env_bool("CARD_OUTER_ENABLE_YOLO_PRIOR", True)

CARD_POSTWARP_MIN_VALID_SCORE = _get_env_float(
    "CARD_POSTWARP_MIN_VALID_SCORE", 0.56, 0.0, 1.0
)
CARD_POSTWARP_RETRY_SCORE_THRESHOLD = _get_env_float(
    "CARD_POSTWARP_RETRY_SCORE_THRESHOLD", 0.64, 0.0, 1.0
)
CARD_POSTWARP_MIN_BORDER_SCORE = _get_env_float(
    "CARD_POSTWARP_MIN_BORDER_SCORE", 0.30, 0.0, 1.0
)
CARD_POSTWARP_MAX_CROP_RISK = _get_env_float(
    "CARD_POSTWARP_MAX_CROP_RISK", 0.78, 0.0, 1.0
)
