"""
Enrichment module - Metadata enrichment from CrossRef, Semantic Scholar, WoRMS, GBIF.
"""
from .crossref_client import CrossRefClient, CrossRefMetadata
from .semantic_scholar_client import SemanticScholarClient, SemanticScholarMetadata
from .taxonomy_resolver import TaxonomyResolver, TaxonomicInfo
from .metadata_enricher import MetadataEnricher, EnrichedMetadata

__all__ = [
    "CrossRefClient",
    "CrossRefMetadata",
    "SemanticScholarClient",
    "SemanticScholarMetadata",
    "TaxonomyResolver",
    "TaxonomicInfo",
    "MetadataEnricher",
    "EnrichedMetadata",
]
