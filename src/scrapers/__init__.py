"""
Scrapers module - External API clients for species data enrichment.

Provides clients for:
- FishBase: Fish biology and ecology
- GBIF: Species occurrences and distribution  
- IUCN: Conservation status
"""
from .fishbase_client import FishBaseClient, FishData
from .gbif_occurrence_client import GBIFOccurrenceClient, OccurrenceRecord, SpeciesDistribution
from .iucn_client import IUCNClient, ConservationStatus

__all__ = [
    "FishBaseClient",
    "FishData",
    "GBIFOccurrenceClient", 
    "OccurrenceRecord",
    "SpeciesDistribution",
    "IUCNClient",
    "ConservationStatus"
]
