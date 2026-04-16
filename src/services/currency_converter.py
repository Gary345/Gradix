from __future__ import annotations

import logging
import time
from typing import Dict, Iterable, Optional

import requests


logger = logging.getLogger(__name__)

FRANKFURTER_URL = "https://api.frankfurter.app/latest"
DEFAULT_TARGET_CURRENCIES = ("USD", "MXN", "COP", "ARS")
_RATE_CACHE: dict[tuple[str, tuple[str, ...]], tuple[float, Dict[str, float]]] = {}
_CACHE_TTL_SECONDS = 60 * 60 * 6


def _normalize_targets(targets: Optional[Iterable[str]] = None) -> tuple[str, ...]:
    normalized = []
    for code in (targets or DEFAULT_TARGET_CURRENCIES):
        code = str(code).strip().upper()
        if code and code not in normalized:
            normalized.append(code)
    return tuple(normalized)


def get_exchange_rates(
    base_currency: str = "EUR",
    targets: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    base_currency = str(base_currency).strip().upper() or "EUR"
    normalized_targets = _normalize_targets(targets)
    cache_key = (base_currency, normalized_targets)
    now = time.time()

    cached = _RATE_CACHE.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return {
            "available": True,
            "base": base_currency,
            "rates": dict(cached[1]),
            "cached": True,
            "source": "frankfurter.app",
        }

    try:
        params = {
            "from": base_currency,
            "to": ",".join(normalized_targets),
        }
        response = requests.get(FRANKFURTER_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        rates = data.get("rates", {})
        if not isinstance(rates, dict) or not rates:
            raise ValueError("La API de tipo de cambio no devolvio tasas validas.")

        numeric_rates = {}
        for code, value in rates.items():
            try:
                numeric_rates[str(code).upper()] = float(value)
            except (TypeError, ValueError):
                continue

        if not numeric_rates:
            raise ValueError("No se pudieron interpretar las tasas de cambio.")

        _RATE_CACHE[cache_key] = (now, numeric_rates)
        return {
            "available": True,
            "base": base_currency,
            "rates": numeric_rates,
            "cached": False,
            "source": "frankfurter.app",
        }
    except Exception as exc:
        logger.warning("No se pudieron obtener tasas de cambio: %s", exc)
        return {
            "available": False,
            "base": base_currency,
            "rates": {},
            "reason": str(exc),
            "source": "frankfurter.app",
        }


def convert_eur_reference_amount(
    amount_eur: object,
    targets: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    try:
        amount = float(amount_eur)
    except (TypeError, ValueError):
        return {
            "available": False,
            "reason": "No hay un monto EUR valido para convertir.",
        }

    rate_result = get_exchange_rates(base_currency="EUR", targets=targets)
    if not rate_result.get("available"):
        return {
            "available": False,
            "reason": rate_result.get("reason", "No se pudieron cargar las tasas."),
            "source": rate_result.get("source"),
        }

    conversions = {
        currency: amount * rate
        for currency, rate in rate_result.get("rates", {}).items()
    }
    return {
        "available": True,
        "base_amount_eur": amount,
        "conversions": conversions,
        "source": rate_result.get("source"),
        "cached": rate_result.get("cached", False),
    }
