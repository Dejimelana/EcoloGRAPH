"""
Core module - Configuration, schemas, and LLM client.
"""
from .config import Settings, get_settings
from .llm_client import LLMClient, LLMResponse
from .schemas import (
    SourceReference,
    SpeciesMention,
    Measurement,
    Location,
    TemporalRange,
    EcologicalRelation,
    SpeciesTrait,
    ExtractionResult,
)

__all__ = [
    "Settings",
    "get_settings",
    "LLMClient",
    "LLMResponse",
    "SourceReference",
    "SpeciesMention",
    "Measurement",
    "Location",
    "TemporalRange",
    "EcologicalRelation",
    "SpeciesTrait",
    "ExtractionResult",
]
