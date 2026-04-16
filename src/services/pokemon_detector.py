"""
Servicio para detectar Pokémon en cartas y obtener información.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from difflib import SequenceMatcher
import logging
import re
from typing import Dict, Optional, List, Tuple
import unicodedata

from src.services.tcgdex_api import TCGdexClient, CardInfo

logger = logging.getLogger(__name__)


class PokemonDetector:
    """Detector de Pokémon y información de cartas"""
    
    def __init__(self, language: str = "es"):
        self.tcgdex = TCGdexClient(language=language)
        self.cache: Dict[str, Tuple[CardInfo, datetime]] = {}
        self.cache_ttl = timedelta(hours=1)
    
    def _get_cached_card(self, card_id: str) -> Optional[CardInfo]:
        """Obtiene una carta del caché si está disponible y válida"""
        if card_id in self.cache:
            card_info, timestamp = self.cache[card_id]
            if datetime.now() - timestamp < self.cache_ttl:
                return card_info
            else:
                del self.cache[card_id]
        return None
    
    def _cache_card(self, card_id: str, card_info: CardInfo):
        """Guarda una carta en caché"""
        self.cache[card_id] = (card_info, datetime.now())

    @staticmethod
    def _normalize_name(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text)
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _build_fallback_queries(self, card_name: str) -> List[str]:
        normalized_name = self._normalize_name(card_name)
        if not normalized_name:
            return []

        queries: List[str] = []

        def add_query(value: str) -> None:
            candidate = value.strip()
            if len(candidate) < 4:
                return
            if candidate not in queries:
                queries.append(candidate)

        add_query(normalized_name)

        tokens = normalized_name.split()
        if tokens:
            first_token = tokens[0]
            add_query(first_token)

            compact_token = re.sub(r"[^a-z0-9]", "", first_token)
            max_prefix = min(len(compact_token), 14)
            for prefix_length in range(max_prefix, 4, -1):
                add_query(compact_token[:prefix_length])

        if len(tokens) >= 2:
            add_query(" ".join(tokens[:2]))

        return queries

    def _rank_card_matches(self, query: str, cards: List[Dict]) -> List[Dict]:
        normalized_query = self._normalize_name(query).replace(" ", "")
        if not normalized_query:
            return cards

        def score(card: Dict) -> Tuple[float, int, int]:
            card_name = self._normalize_name(card.get("name", "")).replace(" ", "")
            similarity = SequenceMatcher(None, normalized_query, card_name).ratio()
            shared_prefix = 0
            for left_char, right_char in zip(normalized_query, card_name):
                if left_char != right_char:
                    break
                shared_prefix += 1
            return similarity, shared_prefix, -abs(len(card_name) - len(normalized_query))

        return sorted(cards, key=score, reverse=True)
    
    def search_card_by_name(self, card_name: str) -> List[CardInfo]:
        """
        Busca una carta por nombre
        
        Args:
            card_name: Nombre de la carta
        
        Returns:
            Lista de CardInfo encontradas
        """
        try:
            primary_results = self.tcgdex.search_card(card_name)
            if primary_results:
                ranked_results = self._rank_card_matches(card_name, primary_results)
                return [CardInfo(card) for card in ranked_results]

            aggregated_results: Dict[str, Dict] = {}
            for fallback_query in self._build_fallback_queries(card_name):
                fallback_results = self.tcgdex.search_card(fallback_query)
                for card in fallback_results:
                    card_id = card.get("id")
                    if card_id and card_id not in aggregated_results:
                        aggregated_results[card_id] = card

                if len(aggregated_results) >= 10:
                    break

            ranked_fallbacks = self._rank_card_matches(
                card_name,
                list(aggregated_results.values()),
            )
            return [CardInfo(card) for card in ranked_fallbacks]
        except Exception as e:
            logger.error(f"Error buscando carta: {e}")
            return []
    
    def search_pokemon_cards(self, pokemon_name: str) -> List[CardInfo]:
        """
        Busca todas las cartas de un Pokémon específico
        
        Args:
            pokemon_name: Nombre del Pokémon
        
        Returns:
            Lista de todas las cartas del Pokémon
        """
        try:
            results = self.tcgdex.search_pokemon(pokemon_name)
            return [CardInfo(card) for card in results]
        except Exception as e:
            logger.error(f"Error buscando Pokémon: {e}")
            return []
    
    def get_card_by_id(self, card_id: str) -> Optional[CardInfo]:
        """
        Obtiene información detallada de una carta por ID
        
        Args:
            card_id: ID de la carta en TCGdex
        
        Returns:
            CardInfo con la información o None si no existe
        """
        # Intentar obtener del caché primero
        cached = self._get_cached_card(card_id)
        if cached:
            return cached
        
        try:
            card_data = self.tcgdex.get_card_by_id(card_id)
            if card_data:
                card_info = CardInfo(card_data)
                self._cache_card(card_id, card_info)
                return card_info
            return None
        except Exception as e:
            logger.error(f"Error obteniendo carta {card_id}: {e}")
            return None
    
    def get_pokemon_type_color(self, pokemon_type: str) -> Dict[str, str]:
        """
        Obtiene el color asociado al tipo de Pokémon
        
        Args:
            pokemon_type: Tipo de Pokémon (Fire, Water, Grass, etc)
        
        Returns:
            Diccionario con color HEX y nombre del tipo
        """
        type_colors = {
            "Normal": {"hex": "#A8A878", "rgb": "168, 168, 120"},
            "Fire": {"hex": "#F08030", "rgb": "240, 128, 48"},
            "Water": {"hex": "#6890F0", "rgb": "104, 144, 240"},
            "Grass": {"hex": "#78C850", "rgb": "120, 200, 80"},
            "Electric": {"hex": "#F8D030", "rgb": "248, 208, 48"},
            "Ice": {"hex": "#98D8D8", "rgb": "152, 216, 216"},
            "Fighting": {"hex": "#A05038", "rgb": "160, 80, 56"},
            "Poison": {"hex": "#A040A0", "rgb": "160, 64, 160"},
            "Ground": {"hex": "#E0C068", "rgb": "224, 192, 104"},
            "Flying": {"hex": "#A890F0", "rgb": "168, 144, 240"},
            "Psychic": {"hex": "#F85888", "rgb": "248, 88, 136"},
            "Bug": {"hex": "#A8B820", "rgb": "168, 184, 32"},
            "Rock": {"hex": "#B8A038", "rgb": "184, 160, 56"},
            "Ghost": {"hex": "#705898", "rgb": "112, 88, 152"},
            "Dragon": {"hex": "#7038F8", "rgb": "112, 56, 248"},
            "Dark": {"hex": "#705848", "rgb": "112, 88, 72"},
            "Steel": {"hex": "#B8B8D0", "rgb": "184, 184, 208"},
            "Fairy": {"hex": "#EE99AC", "rgb": "238, 153, 172"},
        }
        
        return type_colors.get(pokemon_type, {"hex": "#999999", "rgb": "153, 153, 153"})

    @staticmethod
    def _best_pricing_variant(pricing_block: Dict, preferred_order: Optional[List[str]] = None) -> Dict:
        if not isinstance(pricing_block, dict):
            return {}

        preferred_order = preferred_order or ["normal", "holo", "reverse"]
        direct_keys = {"trend", "avg7", "avg30", "low", "marketPrice", "lowPrice", "midPrice", "highPrice"}
        if any(key in pricing_block for key in direct_keys):
            return pricing_block

        best_variant: Dict = {}
        best_score = -1
        for variant_name in preferred_order:
            variant = pricing_block.get(variant_name)
            if not isinstance(variant, dict):
                continue
            score = sum(1 for value in variant.values() if value not in (None, ""))
            if score > best_score:
                best_variant = variant
                best_score = score

        if best_variant:
            return best_variant

        for variant in pricing_block.values():
            if isinstance(variant, dict):
                return variant
        return {}

    def _extract_market_data(self, pricing: Dict) -> Dict:
        pricing = pricing if isinstance(pricing, dict) else {}

        cardmarket_raw = pricing.get("cardmarket", {})
        tcgplayer_raw = pricing.get("tcgplayer", {})

        cardmarket = self._best_pricing_variant(cardmarket_raw, ["normal", "reverse", "holo"])
        tcgplayer = self._best_pricing_variant(tcgplayer_raw, ["normal", "holo", "reverse"])

        return {
            "cardmarket": {
                "currency": "EUR",
                "trend": cardmarket.get("trend"),
                "avg7": cardmarket.get("avg7"),
                "avg30": cardmarket.get("avg30"),
                "low": cardmarket.get("low"),
            } if cardmarket else {},
            "tcgplayer": {
                "currency": "USD",
                "marketPrice": tcgplayer.get("marketPrice"),
                "lowPrice": tcgplayer.get("lowPrice"),
                "midPrice": tcgplayer.get("midPrice"),
                "highPrice": tcgplayer.get("highPrice"),
            } if tcgplayer else {},
        }
    
    def format_card_info(self, card_info: CardInfo) -> Dict:
        """
        Formatea la información de la carta de manera legible
        
        Args:
            card_info: Objeto CardInfo
        
        Returns:
            Diccionario con información formateada
        """
        type_color = self.get_pokemon_type_color(card_info.pokemon_type or "")
        
        return {
            "id": card_info.id,
            "nombre": card_info.name,
            "numero": card_info.card_number,
            "set": card_info.set_name,
            "rareza": card_info.rarity,
            "tipo_pokemon": card_info.pokemon_type,
            "tipo_color": type_color,
            "hp": card_info.hp,
            "ilustrador": card_info.illustrator,
            "descripcion": card_info.description,
            "imagen_url": card_info.image_url,
            "habilidades": card_info.abilities,
            "ataques": card_info.attacks,
            "mercado": self._extract_market_data(card_info.pricing),
        }


# Singleton global
_detector: Optional[PokemonDetector] = None


def get_pokemon_detector() -> PokemonDetector:
    """Obtiene la instancia singleton del detector"""
    global _detector
    if _detector is None:
        _detector = PokemonDetector(language="es")
    return _detector
