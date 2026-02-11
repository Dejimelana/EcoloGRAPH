"""
GBIF Occurrence API client for species distribution data.

Provides:
- Occurrence records (where species have been observed)
- Geographic distribution patterns
- Data quality filters
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime

import httpx


logger = logging.getLogger(__name__)

GBIF_API_BASE = "https://api.gbif.org/v1"


@dataclass
class OccurrenceRecord:
    """A single occurrence record from GBIF."""
    
    occurrence_id: str | None = None
    gbif_id: int | None = None
    
    # Taxonomy
    scientific_name: str | None = None
    species_key: int | None = None
    
    # Location
    decimal_latitude: float | None = None
    decimal_longitude: float | None = None
    country: str | None = None
    country_code: str | None = None
    locality: str | None = None
    depth_meters: float | None = None
    elevation_meters: float | None = None
    
    # Time
    event_date: str | None = None
    year: int | None = None
    month: int | None = None
    
    # Source
    dataset_key: str | None = None
    institution_code: str | None = None
    basis_of_record: str | None = None
    
    # Quality
    has_coordinate: bool = False
    has_geospatial_issues: bool = False
    coordinate_uncertainty_m: float | None = None


@dataclass
class SpeciesDistribution:
    """Aggregated distribution data for a species."""
    
    scientific_name: str
    species_key: int | None = None
    
    # Counts
    total_occurrences: int = 0
    occurrences_with_coords: int = 0
    
    # Geographic
    countries: list[str] = field(default_factory=list)
    bounding_box: dict | None = None  # {min_lat, max_lat, min_lon, max_lon}
    
    # Temporal
    min_year: int | None = None
    max_year: int | None = None
    
    # Sample records
    sample_records: list[OccurrenceRecord] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "scientific_name": self.scientific_name,
            "species_key": self.species_key,
            "total_occurrences": self.total_occurrences,
            "occurrences_with_coords": self.occurrences_with_coords,
            "countries": self.countries,
            "bounding_box": self.bounding_box,
            "year_range": {
                "min": self.min_year,
                "max": self.max_year
            },
            "sample_records": [
                {
                    "lat": r.decimal_latitude,
                    "lon": r.decimal_longitude,
                    "date": r.event_date,
                    "country": r.country
                }
                for r in self.sample_records
            ]
        }


class GBIFOccurrenceClient:
    """
    Client for GBIF Occurrence API.
    
    Retrieves species occurrence data for distribution analysis.
    """
    
    def __init__(self, rate_limit: float = 0.5, timeout: float = 30.0):
        """
        Initialize GBIF client.
        
        Args:
            rate_limit: Minimum seconds between requests
            timeout: Request timeout in seconds
        """
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._last_request = 0.0
        self._client = httpx.Client(timeout=timeout)
        
        logger.info("GBIFOccurrenceClient initialized")
    
    def get_species_key(self, scientific_name: str) -> int | None:
        """
        Get GBIF species key for a scientific name.
        
        Args:
            scientific_name: Scientific name (e.g., "Gadus morhua")
            
        Returns:
            GBIF species key or None
        """
        self._rate_limit_wait()
        
        try:
            response = self._client.get(
                f"{GBIF_API_BASE}/species/match",
                params={"name": scientific_name}
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            return data.get("usageKey")
            
        except Exception as e:
            logger.error(f"GBIF species match error: {e}")
            return None
    
    def get_occurrences(
        self,
        scientific_name: str | None = None,
        species_key: int | None = None,
        country: str | None = None,
        year_range: tuple[int, int] | None = None,
        has_coordinate: bool = True,
        limit: int = 100
    ) -> list[OccurrenceRecord]:
        """
        Get occurrence records for a species.
        
        Args:
            scientific_name: Scientific name to search
            species_key: GBIF species key (faster if known)
            country: ISO country code filter
            year_range: (min_year, max_year) filter
            has_coordinate: Only return georeferenced records
            limit: Maximum records to return
            
        Returns:
            List of OccurrenceRecord
        """
        # Get species key if not provided
        if not species_key and scientific_name:
            species_key = self.get_species_key(scientific_name)
        
        if not species_key:
            logger.warning(f"Could not find species key for: {scientific_name}")
            return []
        
        self._rate_limit_wait()
        
        try:
            params = {
                "speciesKey": species_key,
                "hasCoordinate": str(has_coordinate).lower(),
                "hasGeospatialIssue": "false",
                "limit": min(limit, 300)  # GBIF max is 300
            }
            
            if country:
                params["country"] = country
            
            if year_range:
                params["year"] = f"{year_range[0]},{year_range[1]}"
            
            response = self._client.get(
                f"{GBIF_API_BASE}/occurrence/search",
                params=params
            )
            
            if response.status_code != 200:
                logger.debug(f"GBIF occurrence search failed: {response.status_code}")
                return []
            
            data = response.json()
            
            records = []
            for item in data.get("results", []):
                records.append(self._parse_occurrence(item))
            
            return records
            
        except Exception as e:
            logger.error(f"GBIF occurrence error: {e}")
            return []
    
    def get_distribution(
        self,
        scientific_name: str,
        sample_size: int = 20
    ) -> SpeciesDistribution | None:
        """
        Get aggregated distribution data for a species.
        
        Args:
            scientific_name: Scientific name
            sample_size: Number of sample records to include
            
        Returns:
            SpeciesDistribution with aggregated data
        """
        species_key = self.get_species_key(scientific_name)
        if not species_key:
            return None
        
        # Get occurrence count
        self._rate_limit_wait()
        
        try:
            # Get total count
            response = self._client.get(
                f"{GBIF_API_BASE}/occurrence/search",
                params={"speciesKey": species_key, "limit": 0}
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            total = data.get("count", 0)
            
            # Get sample records with coordinates
            records = self.get_occurrences(
                species_key=species_key,
                has_coordinate=True,
                limit=sample_size
            )
            
            # Get country facets
            countries = self._get_country_facets(species_key)
            
            # Calculate bounding box from records
            bbox = self._calculate_bbox(records)
            
            # Calculate year range
            years = [r.year for r in records if r.year]
            
            return SpeciesDistribution(
                scientific_name=scientific_name,
                species_key=species_key,
                total_occurrences=total,
                occurrences_with_coords=len(records),
                countries=countries,
                bounding_box=bbox,
                min_year=min(years) if years else None,
                max_year=max(years) if years else None,
                sample_records=records
            )
            
        except Exception as e:
            logger.error(f"GBIF distribution error: {e}")
            return None
    
    def _get_country_facets(self, species_key: int) -> list[str]:
        """Get list of countries where species occurs."""
        self._rate_limit_wait()
        
        try:
            response = self._client.get(
                f"{GBIF_API_BASE}/occurrence/search",
                params={
                    "speciesKey": species_key,
                    "facet": "country",
                    "limit": 0
                }
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            facets = data.get("facets", [])
            
            for facet in facets:
                if facet.get("field") == "COUNTRY":
                    return [c["name"] for c in facet.get("counts", [])]
            
            return []
            
        except Exception as e:
            logger.debug(f"GBIF country facets error: {e}")
            return []
    
    def _calculate_bbox(self, records: list[OccurrenceRecord]) -> dict | None:
        """Calculate bounding box from records."""
        lats = [r.decimal_latitude for r in records if r.decimal_latitude is not None]
        lons = [r.decimal_longitude for r in records if r.decimal_longitude is not None]
        
        if not lats or not lons:
            return None
        
        return {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons)
        }
    
    def _parse_occurrence(self, data: dict) -> OccurrenceRecord:
        """Parse GBIF occurrence record."""
        return OccurrenceRecord(
            occurrence_id=data.get("occurrenceID"),
            gbif_id=data.get("gbifID"),
            scientific_name=data.get("scientificName"),
            species_key=data.get("speciesKey"),
            decimal_latitude=data.get("decimalLatitude"),
            decimal_longitude=data.get("decimalLongitude"),
            country=data.get("country"),
            country_code=data.get("countryCode"),
            locality=data.get("locality"),
            depth_meters=data.get("depth"),
            elevation_meters=data.get("elevation"),
            event_date=data.get("eventDate"),
            year=data.get("year"),
            month=data.get("month"),
            dataset_key=data.get("datasetKey"),
            institution_code=data.get("institutionCode"),
            basis_of_record=data.get("basisOfRecord"),
            has_coordinate=data.get("hasCoordinate", False),
            has_geospatial_issues=data.get("hasGeospatialIssues", False),
            coordinate_uncertainty_m=data.get("coordinateUncertaintyInMeters")
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
