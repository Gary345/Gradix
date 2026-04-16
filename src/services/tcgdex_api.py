"""
Servicio para consultar la API de TCGdex.
Documentación: https://tcgdex.dev/rest
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class TCGdexAPIError(Exception):
    """Excepción personalizada para errores de TCGdex API"""
    pass


class TCGdexClient:
    """Cliente para interactuar con TCGdex API"""
    
    BASE_URL = "https://api.tcgdex.net/v2"
    
    def __init__(self, timeout: int = 10, language: str = "es"):
        """
        Inicializa el cliente de TCGdex
        
        Args:
            timeout: Timeout para las requests (segundos)
            language: Idioma para los resultados ('es', 'en', 'fr', 'de', 'ja', etc.)
        """
        self.timeout = timeout
        self.language = language
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Crea una sesión con reintentos automáticos"""
        session = requests.Session()
        
        # Configurar reintentos
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _build_url(self, endpoint: str) -> str:
        normalized_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        return f"{self.BASE_URL}/{self.language}{normalized_endpoint}"

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """
        Realiza una request GET
        
        Args:
            endpoint: Endpoint sin la URL base
            params: Parámetros de query
        
        Returns:
            Respuesta JSON
        
        Raises:
            TCGdexAPIError: Si hay error en la API
        """
        try:
            response = self.session.get(
                self._build_url(endpoint),
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en TCGdex API: {e}")
            raise TCGdexAPIError(f"Error consultando TCGdex: {e}")

    @staticmethod
    def _extract_list_payload(data: Any) -> List[Dict]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            payload = data.get("data", [])
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
        return []
    
    def search_card(self, query: str) -> List[Dict]:
        """
        Busca cartas por nombre
        
        Args:
            query: Nombre de la carta a buscar
        
        Returns:
            Lista de cartas encontradas
        """
        try:
            normalized_query = query.strip()
            if not normalized_query:
                return []

            endpoint = "/cards"
            params = {
                "name": normalized_query,
            }
            data = self._get(endpoint, params)
            return self._extract_list_payload(data)
        except TCGdexAPIError:
            return []
    
    def get_card_by_id(self, card_id: str) -> Optional[Dict]:
        """
        Obtiene información de una carta por ID
        
        Args:
            card_id: ID de la carta en TCGdex
        
        Returns:
            Información de la carta o None si no existe
        """
        try:
            endpoint = f"/cards/{card_id}"
            return self._get(endpoint)
        except TCGdexAPIError:
            return None
    
    def search_pokemon(self, pokemon_name: str) -> List[Dict]:
        """
        Busca todas las cartas de un Pokémon específico
        
        Args:
            pokemon_name: Nombre del Pokémon (ej: "Pikachu", "Charizard")
        
        Returns:
            Lista de cartas del Pokémon
        """
        try:
            normalized_name = pokemon_name.strip()
            if not normalized_name:
                return []

            endpoint = "/cards"
            params = {
                "name": f"eq:{normalized_name}",
            }
            data = self._get(endpoint, params)
            return self._extract_list_payload(data)
        except TCGdexAPIError:
            return []
    
    def get_sets(self) -> List[Dict]:
        """
        Obtiene todos los sets disponibles
        
        Returns:
            Lista de sets
        """
        try:
            endpoint = "/sets"
            data = self._get(endpoint)
            return self._extract_list_payload(data)
        except TCGdexAPIError:
            return []
    
    def get_set_cards(self, set_id: str) -> List[Dict]:
        """
        Obtiene todas las cartas de un set
        
        Args:
            set_id: ID del set
        
        Returns:
            Lista de cartas del set
        """
        try:
            endpoint = f"/sets/{set_id}/cards"
            data = self._get(endpoint)
            return self._extract_list_payload(data)
        except TCGdexAPIError:
            return []
    
    def get_series(self) -> List[Dict]:
        """
        Obtiene todas las series disponibles
        
        Returns:
            Lista de series
        """
        try:
            endpoint = "/series"
            data = self._get(endpoint)
            return self._extract_list_payload(data)
        except TCGdexAPIError:
            return []
    
    def get_rarities(self) -> List[Dict]:
        """
        Obtiene todas las raridades disponibles
        
        Returns:
            Lista de raridades
        """
        try:
            endpoint = "/rarities"
            data = self._get(endpoint)
            return self._extract_list_payload(data)
        except TCGdexAPIError:
            return []
    
    def get_card_prices(self, card_id: str) -> Dict:
        """
        Obtiene información de precios de una carta (si está disponible)
        
        Args:
            card_id: ID de la carta
        
        Returns:
            Información de precios
        """
        try:
            # TCGdex integra precios de varias fuentes
            card = self.get_card_by_id(card_id)
            if card:
                return card.get("pricing", card.get("prices", {}))
            return {}
        except TCGdexAPIError:
            return {}


class CardInfo:
    """Clase para organizar información de una carta"""
    
    def __init__(self, card_data: Dict):
        self.data = card_data
    
    @property
    def id(self) -> str:
        return self.data.get("id", "")
    
    @property
    def name(self) -> str:
        return self.data.get("name", "")
    
    @property
    def set_id(self) -> str:
        set_data = self.data.get("set", {})
        if isinstance(set_data, dict):
            return set_data.get("id", "")
        return self.data.get("setId", "")
    
    @property
    def set_name(self) -> str:
        set_data = self.data.get("set", {})
        if isinstance(set_data, dict):
            return set_data.get("name", "")
        return self.data.get("setName", "")
    
    @property
    def card_number(self) -> str:
        return self.data.get("localId", self.data.get("cardNumber", ""))
    
    @property
    def rarity(self) -> str:
        return self.data.get("rarity", "")
    
    @property
    def pokemon_type(self) -> Optional[str]:
        """Tipo de Pokémon (Grass, Fire, Water, etc)"""
        pokemon_types = self.data.get("types")
        if isinstance(pokemon_types, list) and pokemon_types:
            return pokemon_types[0]
        return self.data.get("type")
    
    @property
    def hp(self) -> Optional[int]:
        """Puntos de vida"""
        return self.data.get("hp")
    
    @property
    def description(self) -> str:
        return self.data.get("description", "")
    
    @property
    def image_url(self) -> str:
        """URL de la imagen de la carta"""
        image_data = self.data.get("image")
        if isinstance(image_data, str):
            return image_data
        if isinstance(image_data, dict):
            return image_data.get("high", "")
        return ""
    
    @property
    def illustrator(self) -> str:
        return self.data.get("illustrator", "")
    
    @property
    def abilities(self) -> List[Dict]:
        """Habilidades de la carta"""
        return self.data.get("abilities", [])
    
    @property
    def attacks(self) -> List[Dict]:
        """Ataques de la carta"""
        return self.data.get("attacks", [])

    @property
    def pricing(self) -> Dict:
        return self.data.get("pricing", self.data.get("prices", {}))
    
    def to_dict(self) -> Dict:
        """Convierte la información a diccionario"""
        return {
            "id": self.id,
            "name": self.name,
            "set_id": self.set_id,
            "set_name": self.set_name,
            "card_number": self.card_number,
            "rarity": self.rarity,
            "pokemon_type": self.pokemon_type,
            "hp": self.hp,
            "illustrator": self.illustrator,
            "description": self.description,
            "image_url": self.image_url,
            "abilities": self.abilities,
            "attacks": self.attacks,
            "pricing": self.pricing
        }
