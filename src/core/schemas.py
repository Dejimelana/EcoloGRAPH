"""
Pydantic schemas for ecological data extraction.

These schemas define the structure of extracted entities with
full traceability to source text.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


# ============================================================
# Traceability / Source Tracking
# ============================================================

class SourceReference(BaseModel):
    """Reference to the exact source of extracted data."""
    doc_id: str = Field(..., description="Document identifier")
    chunk_id: str | None = Field(None, description="Chunk containing the data")
    page: int | None = Field(None, description="Page number if available")
    section: str | None = Field(None, description="Section title if available")
    text_snippet: str | None = Field(None, description="Exact text that supports this extraction")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    is_inferred: bool = Field(default=False, description="True if inferred, not explicitly stated")


class InferenceType(str, Enum):
    """Types of inferences the system can make."""
    EXPLICIT = "explicit"           # Directly stated in text
    TEMPORAL_CONTEXT = "temporal"   # Inferred from study period
    SPATIAL_CONTEXT = "spatial"     # Inferred from study location
    TAXONOMIC = "taxonomic"         # Inferred from species taxonomy
    METHODOLOGICAL = "method"       # Inferred from methods section


# ============================================================
# Geographic / Spatial
# ============================================================

class Coordinates(BaseModel):
    """Geographic coordinates."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class Location(BaseModel):
    """Geographic location with optional coordinates."""
    name: str = Field(..., description="Location name")
    coordinates: Coordinates | None = None
    region: str | None = Field(None, description="Larger region (e.g., North Sea)")
    country: str | None = None
    habitat_type: str | None = Field(None, description="e.g., 'coastal', 'pelagic', 'freshwater'")
    source: SourceReference | None = None


# ============================================================
# Temporal
# ============================================================

class TemporalRange(BaseModel):
    """Time period or range."""
    start_date: datetime | str | None = None
    end_date: datetime | str | None = None
    season: str | None = Field(None, description="e.g., 'summer', 'winter'")
    description: str | None = Field(None, description="e.g., 'during spawning period'")
    source: SourceReference | None = None


# ============================================================
# Species / Taxonomy
# ============================================================

class SpeciesMention(BaseModel):
    """A species mentioned in the text."""
    original_name: str = Field(..., description="Name as it appears in text")
    scientific_name: str | None = Field(None, description="Validated scientific name")
    common_name: str | None = None
    
    # Taxonomic classification (filled by enrichment)
    kingdom: str | None = None
    phylum: str | None = None
    class_name: str | None = None
    order: str | None = None
    family: str | None = None
    genus: str | None = None
    
    # External IDs
    aphia_id: int | None = Field(None, description="WoRMS ID")
    gbif_key: int | None = Field(None, description="GBIF ID")
    
    is_validated: bool = Field(default=False, description="True if validated against WoRMS/GBIF")
    source: SourceReference | None = None


# ============================================================
# Measurements / Quantitative Data
# ============================================================

class MeasurementUnit(str, Enum):
    """Common measurement units in ecology."""
    # Length
    METERS = "m"
    CENTIMETERS = "cm"
    MILLIMETERS = "mm"
    KILOMETERS = "km"
    
    # Mass
    KILOGRAMS = "kg"
    GRAMS = "g"
    MILLIGRAMS = "mg"
    
    # Temperature
    CELSIUS = "°C"
    KELVIN = "K"
    
    # Time
    SECONDS = "s"
    MINUTES = "min"
    HOURS = "h"
    DAYS = "d"
    YEARS = "yr"
    
    # Concentration
    PPM = "ppm"
    PPB = "ppb"
    MG_PER_L = "mg/L"
    
    # Other
    PERCENT = "%"
    INDIVIDUALS = "ind"
    PER_M2 = "/m²"
    PER_KM2 = "/km²"


class Measurement(BaseModel):
    """A quantitative measurement extracted from text."""
    parameter: str = Field(..., description="What is being measured (e.g., 'depth', 'temperature')")
    value: float = Field(..., description="Numeric value")
    unit: str = Field(..., description="Unit of measurement")
    
    # Range support
    value_min: float | None = Field(None, description="Minimum value if range")
    value_max: float | None = Field(None, description="Maximum value if range")
    
    # Statistical info
    is_mean: bool = Field(default=False)
    is_median: bool = Field(default=False)
    std_dev: float | None = None
    sample_size: int | None = None
    
    # Context
    species: str | None = Field(None, description="Species this measurement applies to")
    life_stage: str | None = Field(None, description="e.g., 'adult', 'juvenile', 'larvae'")
    location: str | None = None
    temporal_context: str | None = None
    
    source: SourceReference | None = None


# ============================================================
# Ecological Relationships
# ============================================================

class RelationType(str, Enum):
    """Types of ecological relationships."""
    PREDATION = "predation"         # A eats B
    PARASITISM = "parasitism"       # A is parasite of B
    COMPETITION = "competition"     # A competes with B
    SYMBIOSIS = "symbiosis"         # A lives with B
    HABITAT_USE = "habitat_use"     # A lives in B
    ASSOCIATED_WITH = "associated"  # General association


class EcologicalRelation(BaseModel):
    """A relationship between two entities."""
    relation_type: RelationType
    subject: str = Field(..., description="First entity (e.g., species name)")
    object: str = Field(..., description="Second entity")
    description: str | None = Field(None, description="Details of the relationship")
    source: SourceReference | None = None


# ============================================================
# Traits / Characteristics
# ============================================================

class SpeciesTrait(BaseModel):
    """An ecological trait of a species."""
    species: str
    trait_name: str = Field(..., description="e.g., 'preferred_depth', 'diet_type', 'migration_pattern'")
    trait_value: str | Any = Field(..., description="Value of the trait")
    
    # Context
    life_stage: str | None = None
    season: str | None = None
    location: str | None = None
    
    source: SourceReference | None = None


# ============================================================
# Generic / Domain-Agnostic Entities (for non-ecological papers)
# ============================================================

class GenericEntity(BaseModel):
    """Domain-agnostic entity (algorithm, device, material, software, etc.)."""
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="e.g., 'algorithm', 'device', 'material', 'software'")
    description: str | None = Field(None, description="Brief description of the entity")
    source: SourceReference | None = None


class Method(BaseModel):
    """Research method or technique."""
    name: str = Field(..., description="Method name")
    description: str | None = Field(None, description="Detailed description")
    source: SourceReference | None = None


class Dataset(BaseModel):
    """Dataset used in research."""
    name: str = Field(..., description="Dataset name")
    size: int | None = Field(None, description="Number of samples/records")
    description: str | None = Field(None, description="Dataset description")
    source: SourceReference | None = None


# ============================================================
# Citations / References
# ============================================================

class Citation(BaseModel):
    """A bibliographic citation/reference from a paper."""
    # Core citation info
    authors: list[str] = Field(default_factory=list, description="Author names")
    title: str = Field(..., description="Paper title")
    year: int | None = Field(None, description="Publication year")
    journal: str | None = Field(None, description="Journal/venue name")
    
    # Identifiers
    doi: str | None = Field(None, description="Digital Object Identifier")
    url: str | None = Field(None, description="URL if available")
    
    # Linking
    cited_by_doc_id: str = Field(..., description="Paper that contains this citation")
    
    # Optional: Match to existing papers in database
    matches_doc_id: str | None = Field(None, description="If citation matches a paper in DB")
    match_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Match confidence score")
    
    # Source tracing
    source: SourceReference | None = None


# ============================================================
# Extraction Results
# ============================================================

class ExtractionResult(BaseModel):
    """Complete extraction result from a single chunk or document."""
    # Source info
    doc_id: str
    chunk_id: str | None = None
    
    # Extracted entities (ecological)
    species: list[SpeciesMention] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    temporal_info: list[TemporalRange] = Field(default_factory=list)
    relations: list[EcologicalRelation] = Field(default_factory=list)
    traits: list[SpeciesTrait] = Field(default_factory=list)
    
    # Generic entities (for non-ecological papers)
    key_entities: list[GenericEntity] = Field(default_factory=list)
    methods: list[Method] = Field(default_factory=list)
    datasets: list[Dataset] = Field(default_factory=list)
    
    # Citations (bibliographic references)
    citations: list[Citation] = Field(default_factory=list)
    
    # Metadata
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    model_used: str | None = None
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    
    def entity_count(self) -> int:
        """Count total extracted entities (ecological + generic)."""
        return (
            len(self.species) + 
            len(self.locations) + 
            len(self.measurements) + 
            len(self.relations) + 
            len(self.traits) +
            len(self.key_entities) +
            len(self.methods) +
            len(self.datasets)
        )
