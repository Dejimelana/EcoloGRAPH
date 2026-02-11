"""
Taxonomy resolver using WoRMS and GBIF APIs.

Provides:
- Species name validation
- Taxonomic hierarchy (Kingdom → Species)
- Synonyms and accepted names
- Marine vs terrestrial classification
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any
import hashlib
import re

import requests
from diskcache import Cache

logger = logging.getLogger(__name__)

# API base URLs
WORMS_API_BASE = "https://www.marinespecies.org/rest"
GBIF_API_BASE = "https://api.gbif.org/v1"


@dataclass
class TaxonomicInfo:
    """Resolved taxonomic information for a species."""
    # Input
    original_name: str
    
    # Resolved name
    scientific_name: str | None = None
    canonical_name: str | None = None  # Without author
    accepted_name: str | None = None  # If input was a synonym
    
    # Status
    is_valid: bool = False
    is_marine: bool = False
    is_synonym: bool = False
    match_type: str | None = None  # exact, fuzzy, partial
    confidence: float = 0.0
    
    # Taxonomic hierarchy
    kingdom: str | None = None
    phylum: str | None = None
    class_name: str | None = None  # 'class' is reserved
    order: str | None = None
    family: str | None = None
    genus: str | None = None
    species: str | None = None
    
    # Identifiers
    aphia_id: int | None = None  # WoRMS ID
    gbif_key: int | None = None  # GBIF ID
    
    # Common names
    common_names: list[str] = field(default_factory=list)
    
    # Source
    source: str | None = None  # "worms" or "gbif"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "original_name": self.original_name,
            "scientific_name": self.scientific_name,
            "canonical_name": self.canonical_name,
            "accepted_name": self.accepted_name,
            "is_valid": self.is_valid,
            "is_marine": self.is_marine,
            "is_synonym": self.is_synonym,
            "match_type": self.match_type,
            "confidence": self.confidence,
            "kingdom": self.kingdom,
            "phylum": self.phylum,
            "class": self.class_name,
            "order": self.order,
            "family": self.family,
            "genus": self.genus,
            "species": self.species,
            "aphia_id": self.aphia_id,
            "gbif_key": self.gbif_key,
            "common_names": self.common_names,
            "source": self.source,
        }
    
    def get_hierarchy_string(self) -> str:
        """Get formatted taxonomic hierarchy."""
        parts = []
        for level in [self.kingdom, self.phylum, self.class_name, 
                      self.order, self.family, self.genus, self.species]:
            if level:
                parts.append(level)
        return " > ".join(parts)


class TaxonomyResolver:
    """
    Resolves species names using WoRMS (marine) and GBIF (all taxa).
    
    Strategy:
    1. Try WoRMS first for marine species
    2. Fall back to GBIF for terrestrial/freshwater
    3. Use fuzzy matching when exact match fails
    """
    
    def __init__(
        self,
        cache_dir: str | None = None,
        worms_rate_limit: float = 1.0,
        gbif_rate_limit: float = 1.0
    ):
        """
        Initialize taxonomy resolver.
        
        Args:
            cache_dir: Directory for caching responses
            worms_rate_limit: Seconds between WoRMS requests
            gbif_rate_limit: Seconds between GBIF requests
        """
        self.worms_rate_limit = worms_rate_limit
        self.gbif_rate_limit = gbif_rate_limit
        self._last_worms_request = 0.0
        self._last_gbif_request = 0.0
        
        # Setup cache
        if cache_dir:
            self.cache = Cache(cache_dir)
        else:
            self.cache = None
        
        self.session = requests.Session()
        
        logger.info(f"TaxonomyResolver initialized (cache={'enabled' if cache_dir else 'disabled'})")
    
    def resolve(self, species_name: str, prefer_marine: bool = True) -> TaxonomicInfo:
        """
        Resolve a species name to taxonomic information.
        
        Args:
            species_name: Scientific name to resolve
            prefer_marine: If True, try WoRMS first
            
        Returns:
            TaxonomicInfo with resolved data
        """
        # Clean the name
        clean_name = self._clean_species_name(species_name)
        
        if not clean_name:
            return TaxonomicInfo(original_name=species_name, is_valid=False)
        
        # Check cache
        cache_key = f"taxonomy:{hashlib.md5(clean_name.lower().encode()).hexdigest()}"
        if self.cache and cache_key in self.cache:
            logger.debug(f"Cache hit for species: {clean_name}")
            return self.cache[cache_key]
        
        result = TaxonomicInfo(original_name=species_name)
        
        # Try WoRMS first for marine species
        if prefer_marine:
            worms_result = self._resolve_worms(clean_name)
            if worms_result and worms_result.is_valid:
                result = worms_result
                result.original_name = species_name
        
        # If WoRMS failed or not preferred, try GBIF
        if not result.is_valid:
            gbif_result = self._resolve_gbif(clean_name)
            if gbif_result and gbif_result.is_valid:
                result = gbif_result
                result.original_name = species_name
        
        # Cache result
        if self.cache:
            self.cache[cache_key] = result
        
        return result
    
    def resolve_batch(
        self, 
        species_names: list[str],
        prefer_marine: bool = True
    ) -> list[TaxonomicInfo]:
        """Resolve multiple species names."""
        return [self.resolve(name, prefer_marine) for name in species_names]
    
    def _resolve_worms(self, name: str) -> TaxonomicInfo | None:
        """Resolve species using WoRMS API."""
        # Rate limiting
        elapsed = time.time() - self._last_worms_request
        if elapsed < self.worms_rate_limit:
            time.sleep(self.worms_rate_limit - elapsed)
        
        try:
            # First try exact match
            url = f"{WORMS_API_BASE}/AphiaRecordsByMatchNames"
            params = {"scientificnames[]": name, "marine_only": "false"}
            
            response = self.session.get(url, params=params, timeout=30)
            self._last_worms_request = time.time()
            
            if response.status_code != 200:
                logger.debug(f"WoRMS request failed: {response.status_code}")
                return None
            
            data = response.json()
            
            # Response is [[matches], ...] for each name
            if not data or not data[0]:
                # Try fuzzy match
                return self._resolve_worms_fuzzy(name)
            
            # Get best match
            match = data[0][0]  # First match for first name
            
            return self._parse_worms_record(match, name)
            
        except Exception as e:
            logger.error(f"WoRMS error: {e}")
            return None
    
    def _resolve_worms_fuzzy(self, name: str) -> TaxonomicInfo | None:
        """Try fuzzy matching in WoRMS."""
        try:
            url = f"{WORMS_API_BASE}/AphiaRecordsByName/{name}"
            params = {"like": "true", "marine_only": "false"}
            
            response = self.session.get(url, params=params, timeout=30)
            self._last_worms_request = time.time()
            
            if response.status_code != 200 or not response.content:
                return None
            
            data = response.json()
            if not data:
                return None
            
            # Get first match
            match = data[0]
            result = self._parse_worms_record(match, name)
            if result:
                result.match_type = "fuzzy"
                result.confidence = 0.7
            
            return result
            
        except Exception as e:
            logger.debug(f"WoRMS fuzzy search failed: {e}")
            return None
    
    def _parse_worms_record(self, record: dict, original_name: str) -> TaxonomicInfo | None:
        """Parse WoRMS record into TaxonomicInfo."""
        if not record:
            return None
        
        try:
            status = record.get("status", "").lower()
            is_valid = status in ["accepted", "valid"]
            is_synonym = status in ["synonym", "unaccepted"]
            
            info = TaxonomicInfo(
                original_name=original_name,
                scientific_name=record.get("scientificname"),
                canonical_name=record.get("valid_name"),
                accepted_name=record.get("valid_name") if is_synonym else None,
                is_valid=True,  # Found in WoRMS
                is_marine=record.get("isMarine", 0) == 1,
                is_synonym=is_synonym,
                match_type="exact",
                confidence=1.0,
                kingdom=record.get("kingdom"),
                phylum=record.get("phylum"),
                class_name=record.get("class"),
                order=record.get("order"),
                family=record.get("family"),
                genus=record.get("genus"),
                species=record.get("valid_name"),
                aphia_id=record.get("AphiaID"),
                source="worms"
            )
            
            return info
            
        except Exception as e:
            logger.error(f"Error parsing WoRMS record: {e}")
            return None
    
    def _resolve_gbif(self, name: str) -> TaxonomicInfo | None:
        """Resolve species using GBIF API."""
        # Rate limiting
        elapsed = time.time() - self._last_gbif_request
        if elapsed < self.gbif_rate_limit:
            time.sleep(self.gbif_rate_limit - elapsed)
        
        try:
            url = f"{GBIF_API_BASE}/species/match"
            params = {"name": name, "verbose": "true"}
            
            response = self.session.get(url, params=params, timeout=30)
            self._last_gbif_request = time.time()
            
            if response.status_code != 200:
                logger.debug(f"GBIF request failed: {response.status_code}")
                return None
            
            data = response.json()
            
            if data.get("matchType") == "NONE":
                return None
            
            return self._parse_gbif_record(data, name)
            
        except Exception as e:
            logger.error(f"GBIF error: {e}")
            return None
    
    def _parse_gbif_record(self, record: dict, original_name: str) -> TaxonomicInfo | None:
        """Parse GBIF record into TaxonomicInfo."""
        if not record:
            return None
        
        try:
            match_type = record.get("matchType", "").lower()
            confidence = record.get("confidence", 0) / 100.0
            
            is_synonym = record.get("synonym", False)
            
            info = TaxonomicInfo(
                original_name=original_name,
                scientific_name=record.get("scientificName"),
                canonical_name=record.get("canonicalName"),
                accepted_name=record.get("species") if is_synonym else None,
                is_valid=match_type != "none" and confidence > 0.5,
                is_marine=False,  # GBIF doesn't provide this
                is_synonym=is_synonym,
                match_type=match_type,
                confidence=confidence,
                kingdom=record.get("kingdom"),
                phylum=record.get("phylum"),
                class_name=record.get("class"),
                order=record.get("order"),
                family=record.get("family"),
                genus=record.get("genus"),
                species=record.get("species"),
                gbif_key=record.get("usageKey"),
                source="gbif"
            )
            
            return info
            
        except Exception as e:
            logger.error(f"Error parsing GBIF record: {e}")
            return None
    
    def _clean_species_name(self, name: str) -> str | None:
        """Clean and validate species name."""
        if not name:
            return None
        
        # Remove extra whitespace
        name = " ".join(name.split())
        
        # Remove common artifacts
        name = re.sub(r'\([^)]*\)', '', name)  # Remove parentheticals
        name = re.sub(r'\[[^\]]*\]', '', name)  # Remove brackets
        name = name.strip()
        
        # Should have at least 2 words (Genus species)
        if len(name.split()) < 2:
            return None
        
        return name
