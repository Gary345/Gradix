from __future__ import annotations

import json
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _default_model_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "model_v3"


def _model_dir() -> Path:
    override = os.getenv("GRADIX_CONDITION_MODEL_DIR")
    return Path(override) if override else _default_model_dir()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return value
    return None


def _flatten_nested(prefix: str, value: Any, output: Dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}_{key}" if prefix else str(key)
            _flatten_nested(nested_prefix, nested_value, output)
        return

    numeric_value = _to_float(value)
    if numeric_value is not None and prefix:
        output[prefix] = numeric_value


@lru_cache(maxsize=1)
def _load_feature_names() -> list[str]:
    feature_names_path = _model_dir() / "feature_names.json"
    if not feature_names_path.exists():
        raise FileNotFoundError(
            f"No se encontró feature_names.json en: {feature_names_path}"
        )
    feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    if not isinstance(feature_names, list):
        raise ValueError("feature_names.json no contiene una lista valida.")
    return feature_names


def _load_serialized_model(model_path: Path) -> Any:
    try:
        import joblib  # type: ignore

        return joblib.load(model_path)
    except ModuleNotFoundError:
        with model_path.open("rb") as file_obj:
            return pickle.load(file_obj)


@lru_cache(maxsize=1)
def load_condition_model() -> Dict[str, Any]:
    model_dir = _model_dir()
    model_path = model_dir / "condition_model_hgb_v3.pkl"
    feature_names_path = model_dir / "feature_names.json"

    if not model_dir.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio del modelo: {model_dir}"
        )
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"La ruta configurada para el modelo no es un directorio: {model_dir}"
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró condition_model_hgb_v3.pkl en: {model_path}"
        )
    if not feature_names_path.exists():
        raise FileNotFoundError(
            f"No se encontró feature_names.json en: {feature_names_path}"
        )

    model = _load_serialized_model(model_path)
    feature_names = _load_feature_names()

    return {
        "model": model,
        "feature_names": feature_names,
        "model_path": model_path,
        "feature_names_path": feature_names_path,
    }


def flatten_for_model(features_block: dict, scores_block: dict) -> Dict[str, float]:
    flat: Dict[str, float] = {}

    block_map = {
        "visual": (features_block or {}).get("visual"),
        "geometry": (features_block or {}).get("geometry"),
        "centrado": (features_block or {}).get("centering"),
        "borde": (features_block or {}).get("edge"),
        "esquina": (features_block or {}).get("corner"),
        "superficie": (features_block or {}).get("whitening_surface"),
        "score_capture": (scores_block or {}).get("capture_quality"),
        "score_prelim": (scores_block or {}).get("preliminary"),
        "score_centering": (scores_block or {}).get("centering"),
        "score_edge": (scores_block or {}).get("edge"),
        "score_corner": (scores_block or {}).get("corner"),
        "score_ws": (scores_block or {}).get("whitening_surface"),
    }

    for prefix, block in block_map.items():
        if isinstance(block, dict):
            _flatten_nested(prefix, block, flat)

    return flat


def build_model_input(features_block: dict, scores_block: dict) -> np.ndarray:
    feature_names = _load_feature_names()
    flat_features = flatten_for_model(features_block, scores_block)

    row = [float(flat_features.get(feature_name, 0.0)) for feature_name in feature_names]
    return np.asarray([row], dtype=np.float32)


def predict_condition(features_block: dict, scores_block: dict) -> Dict[str, Any]:
    try:
        loaded = load_condition_model()
        model = loaded["model"]
        model_input = build_model_input(features_block, scores_block)

        prediction = int(model.predict(model_input)[0])

        probability_damaged = None
        probability_undamaged = None
        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(model_input)[0]
            if len(probabilities) >= 2:
                classes = list(getattr(model, "classes_", [0, 1]))
                class_to_probability = {
                    int(class_value): float(probability)
                    for class_value, probability in zip(classes, probabilities)
                }
                probability_undamaged = class_to_probability.get(0)
                probability_damaged = class_to_probability.get(1)
                if probability_damaged is not None and probability_undamaged is not None:
                    confidence = max(probability_damaged, probability_undamaged)

        return {
            "available": True,
            "prediction": prediction,
            "label": "damaged" if prediction == 1 else "undamaged",
            "probability_damaged": probability_damaged,
            "probability_undamaged": probability_undamaged,
            "confidence": confidence,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
            "model_dir": str(_model_dir()),
        }
