"""
IUCN Red List API client for conservation status.

Provides:
- Conservation status (CR, EN, VU, etc.)
- Population trends
- Threats and habitats
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


logger = logging.getLogger(__name__)

IUCN_API_BASE = "https://apiv3.iucnredlist.org/api/v3"


@dataclass
class ConservationStatus:
    """Conservation status data from IUCN."""
    
    scientific_name: str
    taxon_id: int | None = None
    
    # Status
    category: str | None = None  # CR, EN, VU, NT, LC, DD, NE
    category_name: str | None = None  # Full name
    population_trend: str | None = None  # increasing, decreasing, stable, unknown
    
    # Assessment
    assessment_date: str | None = None
    criteria: str | None = None  # e.g., "A2cd"
    
    # Narrative
    rationale: str | None = None
    
    # Threats
    threats: list[str] = field(default_factory=list)
    
    # Habitats
    habitats: list[str] = field(default_factory=list)
    
    # Geographic
    range_description: str | None = None
    countries: list[str] = field(default_factory=list)
    
    # Source
    source: str = "iucn"
    
    def is_threatened(self) -> bool:
        """Check if species is in threatened category."""
        return self.category in ["CR", "EN", "VU"]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "scientific_name": self.scientific_name,
            "taxon_id": self.taxon_id,
            "category": self.category,
            "category_name": self.category_name,
            "population_trend": self.population_trend,
            "assessment_date": self.assessment_date,
            "criteria": self.criteria,
            "is_threatened": self.is_threatened(),
            "threats": self.threats,
            "habitats": self.habitats,
            "range_description": self.range_description,
            "countries": self.countries,
            "source": self.source
        }


# Category code to full name mapping
IUCN_CATEGORIES = {
    "EX": "Extinct",
    "EW": "Extinct in the Wild",
    "CR": "Critically Endangered",
    "EN": "Endangered",
    "VU": "Vulnerable",
    "NT": "Near Threatened",
    "LC": "Least Concern",
    "DD": "Data Deficient",
    "NE": "Not Evaluated"
}


class IUCNClient:
    """
    Client for IUCN Red List API.
    
    Note: Requires an API token from https://apiv3.iucnredlist.org/api/v3/token
    """
    
    def __init__(
        self, 
        api_token: str | None = None,
        rate_limit: float = 1.0, 
        timeout: float = 30.0
    ):
        """
        Initialize IUCN client.
        
        Args:
            api_token: IUCN API token (required for most endpoints)
            rate_limit: Minimum seconds between requests
            timeout: Request timeout in seconds
        """
        self.api_token = api_token
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._last_request = 0.0
        self._client = httpx.Client(timeout=timeout)
        
        if not api_token:
            logger.warning("IUCN API token not provided - API access will be limited")
        
        logger.info("IUCNClient initialized")
    
    def get_species(self, scientific_name: str) -> ConservationStatus | None:
        """
        Get conservation status for a species.
        
        Uses IUCN API if token is available, otherwise falls back to
        free GBIF Species API which includes IUCN Red List category.
        
        Args:
            scientific_name: Scientific name (e.g., "Gadus morhua")
            
        Returns:
            ConservationStatus or None if not found
        """
        if not self.api_token:
            # Try free GBIF fallback
            return self._gbif_fallback(scientific_name)
        
        self._rate_limit_wait()
        
        try:
            # URL encode the species name
            name = scientific_name.replace(" ", "%20")
            
            response = self._client.get(
                f"{IUCN_API_BASE}/species/{name}",
                params={"token": self.api_token}
            )
            
            if response.status_code != 200:
                logger.debug(f"IUCN request failed: {response.status_code}")
                # Try GBIF fallback
                return self._gbif_fallback(scientific_name)
            
            data = response.json()
            
            if not data.get("result"):
                return self._gbif_fallback(scientific_name)
            
            result = data["result"][0]
            
            # Get additional data
            taxon_id = result.get("taxonid")
            threats = self._get_threats(taxon_id) if taxon_id else []
            habitats = self._get_habitats(taxon_id) if taxon_id else []
            
            return self._build_status(result, threats, habitats)
            
        except Exception as e:
            logger.error(f"IUCN error: {e}")
            return self._gbif_fallback(scientific_name)

    def _gbif_fallback(self, scientific_name: str) -> ConservationStatus | None:
        """
        Free fallback: use GBIF Species API to get IUCN category.
        
        GBIF species/match returns iucnRedListCategory for many species.
        No API token required.
        """
        self._rate_limit_wait()
        
        try:
            response = self._client.get(
                "https://api.gbif.org/v1/species/match",
                params={"name": scientific_name, "verbose": "true"}
            )
            
            if response.status_code != 200:
                logger.debug(f"GBIF species match failed: {response.status_code}")
                return None
            
            data = response.json()
            
            if data.get("matchType") == "NONE":
                return None
            
            # Get the GBIF species key for additional lookups
            species_key = data.get("usageKey")
            
            # Try to get IUCN status from GBIF species detail
            iucn_category = None
            if species_key:
                try:
                    detail_resp = self._client.get(
                        f"https://api.gbif.org/v1/species/{species_key}/iucnRedListCategory"
                    )
                    if detail_resp.status_code == 200:
                        detail = detail_resp.json()
                        iucn_category = detail.get("category")
                except Exception:
                    pass
            
            # Also check the direct match response
            if not iucn_category:
                iucn_category = data.get("iucnRedListCategory")
            
            if not iucn_category:
                # Still return basic species info without IUCN status
                return ConservationStatus(
                    scientific_name=data.get("canonicalName", scientific_name),
                    category=None,
                    category_name="Not available (no IUCN token)",
                    source="gbif_fallback",
                )
            
            return ConservationStatus(
                scientific_name=data.get("canonicalName", scientific_name),
                category=iucn_category,
                category_name=IUCN_CATEGORIES.get(iucn_category, iucn_category),
                source="gbif_fallback",
            )
            
        except Exception as e:
            logger.error(f"GBIF fallback error: {e}")
            return None
    
    def _get_threats(self, taxon_id: int) -> list[str]:
        """Get threats for a taxon."""
        if not self.api_token:
            return []
        
        self._rate_limit_wait()
        
        try:
            response = self._client.get(
                f"{IUCN_API_BASE}/threats/species/id/{taxon_id}",
                params={"token": self.api_token}
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            return [t.get("title", "") for t in data.get("result", []) if t.get("title")]
            
        except Exception as e:
            logger.debug(f"IUCN threats error: {e}")
            return []
    
    def _get_habitats(self, taxon_id: int) -> list[str]:
        """Get habitats for a taxon."""
        if not self.api_token:
            return []
        
        self._rate_limit_wait()
        
        try:
            response = self._client.get(
                f"{IUCN_API_BASE}/habitats/species/id/{taxon_id}",
                params={"token": self.api_token}
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            return [h.get("habitat", "") for h in data.get("result", []) if h.get("habitat")]
            
        except Exception as e:
            logger.debug(f"IUCN habitats error: {e}")
            return []
    
    def _build_status(
        self, 
        data: dict, 
        threats: list[str], 
        habitats: list[str]
    ) -> ConservationStatus:
        """Build ConservationStatus from API data."""
        category = data.get("category")
        
        return ConservationStatus(
            scientific_name=data.get("scientific_name", ""),
            taxon_id=data.get("taxonid"),
            category=category,
            category_name=IUCN_CATEGORIES.get(category),
            population_trend=data.get("population_trend"),
            assessment_date=data.get("assessment_date"),
            criteria=data.get("criteria"),
            rationale=data.get("rationale"),
            threats=threats,
            habitats=habitats
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
