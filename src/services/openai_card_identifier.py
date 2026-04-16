from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict

import cv2
import numpy as np
import requests


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = os.getenv("OPENAI_CARD_IDENTIFIER_MODEL", "gpt-4.1-mini")


def _encode_bgr_to_base64_jpeg(image_bgr: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not success:
        raise ValueError("No se pudo codificar la imagen para OpenAI.")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("La respuesta de OpenAI no devolvió un objeto JSON.")
    return parsed


def _safe_confidence(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        confidence = float(value)
        return max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        return None


def identify_card_from_warped_image(warped_card_bgr) -> dict:
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "available": False,
                "reason": "OPENAI_API_KEY no configurada.",
            }

        if warped_card_bgr is None or getattr(warped_card_bgr, "size", 0) == 0:
            return {
                "available": False,
                "reason": "Imagen rectificada no disponible para OpenAI.",
            }

        image_base64 = _encode_bgr_to_base64_jpeg(warped_card_bgr)
        prompt = (
            "Identifica con precision la carta Pokemon mostrada en la imagen. "
            "Prioriza el nombre exacto impreso en la parte superior, el set, el numero de carta, la rareza y el HP. "
            "No inventes datos: si un campo no es legible, devuelvelo como cadena vacia. "
            "Devuelve solo JSON con las claves: "
            "name, set_name, card_number, rarity, hp, confidence. "
            "confidence debe estar entre 0.0 y 1.0 y debe bajar si la lectura visual es dudosa."
        )

        payload = {
            "model": DEFAULT_MODEL,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        },
                    ],
                }
            ],
        }

        response = requests.post(
            OPENAI_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()

        message_content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = _extract_json_object(message_content)
        name = str(parsed.get("name", "") or "").strip()

        if not name:
            return {
                "available": False,
                "reason": "OpenAI no devolvió un nombre utilizable.",
            }

        return {
            "available": True,
            "name": name,
            "set_name": str(parsed.get("set_name", "") or "").strip(),
            "card_number": str(parsed.get("card_number", "") or "").strip(),
            "rarity": str(parsed.get("rarity", "") or "").strip(),
            "hp": str(parsed.get("hp", "") or "").strip(),
            "confidence": _safe_confidence(parsed.get("confidence")),
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
        }
