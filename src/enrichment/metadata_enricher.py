"""
Metadata enricher - orchestrates all enrichment sources.

Combines data from:
- CrossRef (bibliographic data, affiliations, funding)
- Semantic Scholar (fields of study, citations, TLDR)
- WoRMS/GBIF (taxonomy validation)
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .crossref_client import CrossRefClient, CrossRefMetadata
from .semantic_scholar_client import SemanticScholarClient, SemanticScholarMetadata
from .taxonomy_resolver import TaxonomyResolver, TaxonomicInfo

logger = logging.getLogger(__name__)


@dataclass
class EnrichedMetadata:
    """
    Combined metadata from all enrichment sources.
    
    Provides a unified view of paper metadata with traceability
    to original sources.
    """
    # Core identifiers
    doi: str | None = None
    title: str | None = None
    
    # Authors with affiliations
    authors: list[dict[str, Any]] = field(default_factory=list)
    # Format: [{"name": "John Smith", "affiliations": ["Univ A"], "orcid": "..."}]
    
    # Publication info
    year: int | None = None
    journal: str | None = None
    publisher: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    
    # Classification
    subjects: list[str] = field(default_factory=list)
    fields_of_study: list[str] = field(default_factory=list)
    
    # Funding
    funders: list[dict[str, Any]] = field(default_factory=list)
    
    # Citations
    citation_count: int = 0
    reference_count: int = 0
    
    # Abstract and summary
    abstract: str | None = None
    tldr: str | None = None  # AI-generated summary from S2
    
    # Open access
    is_open_access: bool = False
    open_access_url: str | None = None
    
    # Taxonomic entities found (validated species names)
    validated_species: list[TaxonomicInfo] = field(default_factory=list)
    
    # Source tracking
    sources: list[str] = field(default_factory=list)  # ["crossref", "semantic_scholar"]
    
    # Raw data for advanced users
    crossref_raw: dict | None = None
    s2_raw: dict | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "doi": self.doi,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "journal": self.journal,
            "publisher": self.publisher,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "subjects": self.subjects,
            "fields_of_study": self.fields_of_study,
            "funders": self.funders,
            "citation_count": self.citation_count,
            "reference_count": self.reference_count,
            "abstract": self.abstract,
            "tldr": self.tldr,
            "is_open_access": self.is_open_access,
            "open_access_url": self.open_access_url,
            "validated_species": [s.to_dict() for s in self.validated_species],
            "sources": self.sources,
        }


class MetadataEnricher:
    """
    Orchestrates metadata enrichment from multiple sources.
    
    Features:
    - Combines data from CrossRef, Semantic Scholar, and taxonomy DBs
    - Handles missing data gracefully
    - Caches all responses
    - Rate limits to respect API policies
    """
    
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        crossref_email: str | None = None,
        s2_api_key: str | None = None
    ):
        """
        Initialize the metadata enricher.
        
        Args:
            cache_dir: Directory for caching API responses
            crossref_email: Email for CrossRef polite pool
            s2_api_key: Optional Semantic Scholar API key
        """
        cache_path = str(cache_dir) if cache_dir else None
        
        # Initialize clients
        self.crossref = CrossRefClient(
            mailto=crossref_email,
            cache_dir=f"{cache_path}/crossref" if cache_path else None,
            rate_limit=1.0
        )
        
        self.semantic_scholar = SemanticScholarClient(
            api_key=s2_api_key,
            cache_dir=f"{cache_path}/s2" if cache_path else None,
            rate_limit=1.0 if s2_api_key else 3.0
        )
        
        self.taxonomy = TaxonomyResolver(
            cache_dir=f"{cache_path}/taxonomy" if cache_path else None,
            worms_rate_limit=1.0,
            gbif_rate_limit=1.0
        )
        
        logger.info("MetadataEnricher initialized with all clients")
    
    def enrich_by_doi(self, doi: str) -> EnrichedMetadata:
        """
        Enrich metadata using DOI lookup.
        
        This is the most reliable method as DOI provides exact matching.
        
        Args:
            doi: The DOI of the paper
            
        Returns:
            EnrichedMetadata with combined data
        """
        result = EnrichedMetadata(doi=doi)
        
        # Get CrossRef data
        cr_data = self.crossref.get_by_doi(doi)
        if cr_data:
            self._merge_crossref(result, cr_data)
            result.sources.append("crossref")
            result.crossref_raw = cr_data.raw
        
        # Get Semantic Scholar data
        s2_data = self.semantic_scholar.get_by_doi(doi)
        if s2_data:
            self._merge_semantic_scholar(result, s2_data)
            result.sources.append("semantic_scholar")
        
        logger.info(f"Enriched DOI {doi}: sources={result.sources}")
        
        return result
    
    def enrich_by_title(
        self,
        title: str,
        authors: list[str] | None = None,
        year: int | None = None
    ) -> EnrichedMetadata:
        """
        Enrich metadata using title search.
        
        Less reliable than DOI but works when DOI is unavailable.
        
        Args:
            title: Paper title
            authors: Optional author list to improve matching
            year: Optional publication year
            
        Returns:
            EnrichedMetadata with combined data
        """
        result = EnrichedMetadata(title=title)
        
        # Search CrossRef
        cr_results = self.crossref.search_by_title(title, authors, year, limit=1)
        if cr_results:
            cr_data = cr_results[0]
            self._merge_crossref(result, cr_data)
            result.sources.append("crossref")
            result.crossref_raw = cr_data.raw
        
        # Search Semantic Scholar
        s2_results = self.semantic_scholar.search_by_title(title, year, limit=1)
        if s2_results:
            s2_data = s2_results[0]
            self._merge_semantic_scholar(result, s2_data)
            result.sources.append("semantic_scholar")
        
        logger.info(f"Enriched title '{title[:50]}...': sources={result.sources}")
        
        return result
    
    def validate_species(
        self,
        species_names: list[str],
        prefer_marine: bool = True
    ) -> list[TaxonomicInfo]:
        """
        Validate and enrich species names.
        
        Args:
            species_names: List of species names to validate
            prefer_marine: Try WoRMS first if True
            
        Returns:
            List of TaxonomicInfo for each species
        """
        results = []
        for name in species_names:
            info = self.taxonomy.resolve(name, prefer_marine)
            results.append(info)
            
            if info.is_valid:
                logger.debug(f"Validated species: {name} -> {info.scientific_name}")
            else:
                logger.debug(f"Could not validate species: {name}")
        
        return results
    
    def enrich_full(
        self,
        doi: str | None = None,
        title: str | None = None,
        authors: list[str] | None = None,
        year: int | None = None,
        species_names: list[str] | None = None
    ) -> EnrichedMetadata:
        """
        Full enrichment with all available data.
        
        Args:
            doi: DOI if available (preferred)
            title: Paper title
            authors: Author names
            year: Publication year
            species_names: Species mentioned in paper
            
        Returns:
            Fully enriched metadata
        """
        # Start with DOI if available
        if doi:
            result = self.enrich_by_doi(doi)
        elif title:
            result = self.enrich_by_title(title, authors, year)
        else:
            result = EnrichedMetadata()
        
        # Add species validation
        if species_names:
            result.validated_species = self.validate_species(species_names)
        
        return result
    
    def _merge_crossref(self, result: EnrichedMetadata, cr: CrossRefMetadata) -> None:
        """Merge CrossRef data into result."""
        if not result.doi:
            result.doi = cr.doi
        if not result.title:
            result.title = cr.title
        
        result.year = cr.year
        result.journal = cr.journal
        result.publisher = cr.publisher
        result.volume = cr.volume
        result.issue = cr.issue
        result.pages = cr.pages
        result.subjects = cr.subjects
        result.funders = cr.funders
        result.reference_count = cr.references_count
        result.citation_count = cr.is_referenced_by_count
        
        # Convert authors to unified format
        for author in cr.authors:
            unified = {
                "name": f"{author.get('given', '')} {author.get('family', '')}".strip(),
                "given": author.get("given"),
                "family": author.get("family"),
                "affiliations": [a.get("name") for a in author.get("affiliation", [])],
                "orcid": author.get("ORCID"),
            }
            result.authors.append(unified)
    
    def _merge_semantic_scholar(self, result: EnrichedMetadata, s2: SemanticScholarMetadata) -> None:
        """Merge Semantic Scholar data into result."""
        if not result.doi:
            result.doi = s2.doi
        if not result.title:
            result.title = s2.title
        
        result.abstract = s2.abstract
        result.tldr = s2.tldr
        result.fields_of_study = s2.fields_of_study
        result.is_open_access = s2.is_open_access
        result.open_access_url = s2.open_access_pdf_url
        
        # Update citation count if higher (S2 is often more current)
        if s2.citation_count > result.citation_count:
            result.citation_count = s2.citation_count
        
        # Add authors if not already present
        if not result.authors and s2.authors:
            for author in s2.authors:
                unified = {
                    "name": author.get("name", ""),
                    "s2_id": author.get("authorId"),
                }
                result.authors.append(unified)
