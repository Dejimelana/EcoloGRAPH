"""
FishBase API client for fish species data enrichment.

Provides:
- Species biology (habitat, diet, reproduction)
- Common names and synonyms
- Distribution data
- Ecological information
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


logger = logging.getLogger(__name__)

FISHBASE_API_BASE = "https://fishbase.ropensci.org"


@dataclass
class FishData:
    """Fish species data from FishBase."""
    
    # Identity
    species_code: int | None = None
    scientific_name: str | None = None
    genus: str | None = None
    species: str | None = None
    author: str | None = None
    
    # Common names
    common_names: list[dict[str, str]] = field(default_factory=list)  # [{name, language}]
    fb_name: str | None = None  # FishBase common name
    
    # Biology
    max_length_cm: float | None = None
    max_weight_g: float | None = None
    max_age_years: float | None = None
    
    # Habitat
    depth_range_shallow: int | None = None
    depth_range_deep: int | None = None
    habitat: str | None = None  # pelagic, demersal, reef, etc.
    climate: str | None = None  # tropical, subtropical, temperate
    
    # Ecology
    feeding_type: str | None = None  # herbivore, carnivore, omnivore
    trophic_level: float | None = None
    food_items: list[str] = field(default_factory=list)
    predators: list[str] = field(default_factory=list)
    
    # Distribution
    distribution: str | None = None
    countries: list[str] = field(default_factory=list)
    fao_areas: list[str] = field(default_factory=list)
    
    # Conservation
    iucn_status: str | None = None
    vulnerability: float | None = None  # 0-100
    
    # Reproduction
    spawning_season: str | None = None
    maturity_length_cm: float | None = None
    
    # Metadata
    source: str = "fishbase"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "species_code": self.species_code,
            "scientific_name": self.scientific_name,
            "genus": self.genus,
            "species": self.species,
            "author": self.author,
            "common_names": self.common_names,
            "fb_name": self.fb_name,
            "max_length_cm": self.max_length_cm,
            "max_weight_g": self.max_weight_g,
            "max_age_years": self.max_age_years,
            "depth_range": {
                "shallow": self.depth_range_shallow,
                "deep": self.depth_range_deep
            },
            "habitat": self.habitat,
            "climate": self.climate,
            "feeding_type": self.feeding_type,
            "trophic_level": self.trophic_level,
            "food_items": self.food_items,
            "predators": self.predators,
            "distribution": self.distribution,
            "countries": self.countries,
            "fao_areas": self.fao_areas,
            "iucn_status": self.iucn_status,
            "vulnerability": self.vulnerability,
            "spawning_season": self.spawning_season,
            "maturity_length_cm": self.maturity_length_cm,
            "source": self.source
        }


class FishBaseClient:
    """
    Client for FishBase API.
    
    Uses the rfishbase API endpoint for data retrieval.
    """
    
    def __init__(self, rate_limit: float = 1.0, timeout: float = 30.0):
        """
        Initialize FishBase client.
        
        Args:
            rate_limit: Minimum seconds between requests
            timeout: Request timeout in seconds
        """
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._last_request = 0.0
        # SSL verification disabled — ropensci cert is broken/self-signed
        self._client = httpx.Client(timeout=timeout, verify=False)
        
        logger.info("FishBaseClient initialized")
    
    def get_species(self, scientific_name: str) -> FishData | None:
        """
        Get species data by scientific name.
        
        Args:
            scientific_name: Scientific name (e.g., "Gadus morhua")
            
        Returns:
            FishData or None if not found
        """
        # Split into genus and species
        parts = scientific_name.strip().split()
        if len(parts) < 2:
            logger.warning(f"Invalid species name: {scientific_name}")
            return None
        
        # FishBase API is case-sensitive: Genus capitalized, species lowercase
        genus = parts[0].capitalize()
        species = parts[1].lower()
        
        # Search for species
        species_data = self._search_species(genus, species)
        if not species_data:
            return None
        
        # Get additional data
        species_code = species_data.get("SpecCode")
        if species_code:
            # Get ecology data
            ecology = self._get_ecology(species_code)
            # Get common names
            common_names = self._get_common_names(species_code)
        else:
            ecology = {}
            common_names = []
        
        return self._build_fish_data(species_data, ecology, common_names)
    
    def get_species_by_code(self, species_code: int) -> FishData | None:
        """Get species data by FishBase species code."""
        self._rate_limit_wait()
        
        try:
            response = self._client.get(
                f"{FISHBASE_API_BASE}/species",
                params={"SpecCode": species_code}
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data.get("data"):
                return None
            
            species_data = data["data"][0]
            ecology = self._get_ecology(species_code)
            common_names = self._get_common_names(species_code)
            
            return self._build_fish_data(species_data, ecology, common_names)
            
        except Exception as e:
            logger.error(f"FishBase error: {e}")
            return None
    
    def _search_species(self, genus: str, species: str) -> dict | None:
        """Search for species by genus and species name."""
        self._rate_limit_wait()
        
        try:
            response = self._client.get(
                f"{FISHBASE_API_BASE}/species",
                params={"Genus": genus, "Species": species}
            )
            
            if response.status_code != 200:
                logger.debug(f"FishBase search failed: {response.status_code}")
                return None
            
            data = response.json()
            if not data.get("data"):
                return None
            
            return data["data"][0]
            
        except Exception as e:
            logger.error(f"FishBase search error: {e}")
            return None
    
    def _get_ecology(self, species_code: int) -> dict:
        """Get ecology data for a species."""
        self._rate_limit_wait()
        
        try:
            response = self._client.get(
                f"{FISHBASE_API_BASE}/ecology",
                params={"SpecCode": species_code}
            )
            
            if response.status_code != 200:
                return {}
            
            data = response.json()
            if not data.get("data"):
                return {}
            
            return data["data"][0]
            
        except Exception as e:
            logger.debug(f"FishBase ecology error: {e}")
            return {}
    
    def _get_common_names(self, species_code: int, limit: int = 10) -> list[dict]:
        """Get common names for a species."""
        self._rate_limit_wait()
        
        try:
            response = self._client.get(
                f"{FISHBASE_API_BASE}/comnames",
                params={"SpecCode": species_code, "limit": limit}
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            names = []
            for item in data.get("data", []):
                names.append({
                    "name": item.get("ComName"),
                    "language": item.get("Language")
                })
            
            return names
            
        except Exception as e:
            logger.debug(f"FishBase common names error: {e}")
            return []
    
    def _build_fish_data(
        self, 
        species: dict, 
        ecology: dict, 
        common_names: list
    ) -> FishData:
        """Build FishData from API responses."""
        return FishData(
            species_code=species.get("SpecCode"),
            scientific_name=f"{species.get('Genus', '')} {species.get('Species', '')}".strip(),
            genus=species.get("Genus"),
            species=species.get("Species"),
            author=species.get("Author"),
            common_names=common_names,
            fb_name=species.get("FBname"),
            max_length_cm=species.get("Length"),
            max_weight_g=species.get("Weight"),
            max_age_years=species.get("Longevity"),
            depth_range_shallow=species.get("DepthRangeShallow"),
            depth_range_deep=species.get("DepthRangeDeep"),
            habitat=ecology.get("Habitat"),
            climate=species.get("Climate"),
            feeding_type=ecology.get("FeedingType"),
            trophic_level=ecology.get("FoodTroph"),
            iucn_status=species.get("IUCN_Code"),
            vulnerability=species.get("Vulnerability"),
            distribution=species.get("Distribution"),
        )
    
    def _rate_limit_wait(self):
        """Wait for rate limit."""
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()
    
    def close(self):
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
