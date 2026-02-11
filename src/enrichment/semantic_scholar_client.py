"""
Semantic Scholar API client for enriching paper metadata.

Semantic Scholar provides:
- Normalized keywords and topics
- Field of study classification
- Citation context
- Paper embeddings for similarity
- Open access links
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any
import hashlib

import requests
from diskcache import Cache

logger = logging.getLogger(__name__)

# Semantic Scholar API base URL
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"


@dataclass
class SemanticScholarMetadata:
    """Metadata retrieved from Semantic Scholar."""
    # Core identifiers
    paper_id: str | None = None  # S2 internal ID
    doi: str | None = None
    arxiv_id: str | None = None
    
    # Basic info
    title: str | None = None
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None  # Conference/Journal
    
    # Authors
    authors: list[dict[str, Any]] = field(default_factory=list)
    # Format: [{"authorId": "123", "name": "John Smith"}]
    
    # Topics and classification
    fields_of_study: list[str] = field(default_factory=list)
    # e.g., ["Biology", "Environmental Science", "Ecology"]
    
    s2_fields_of_study: list[dict] = field(default_factory=list)
    # More detailed: [{"category": "Biology", "source": "s2-fos-model"}]
    
    # Citations
    citation_count: int = 0
    influential_citation_count: int = 0
    reference_count: int = 0
    
    # Open access
    is_open_access: bool = False
    open_access_pdf_url: str | None = None
    
    # Embedding for similarity search
    embedding: list[float] | None = None
    
    # TLDR summary
    tldr: str | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "paper_id": self.paper_id,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "venue": self.venue,
            "authors": self.authors,
            "fields_of_study": self.fields_of_study,
            "citation_count": self.citation_count,
            "influential_citation_count": self.influential_citation_count,
            "reference_count": self.reference_count,
            "is_open_access": self.is_open_access,
            "open_access_pdf_url": self.open_access_pdf_url,
            "tldr": self.tldr,
        }


class SemanticScholarClient:
    """
    Client for the Semantic Scholar API.
    
    Features:
    - DOI/ArXiv/S2 ID lookup
    - Title-based search
    - Field of study classification
    - Citation data
    - Caching and rate limiting
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | None = None,
        rate_limit: float = 1.0
    ):
        """
        Initialize Semantic Scholar client.
        
        Args:
            api_key: Optional API key for higher rate limits
            cache_dir: Directory for caching responses
            rate_limit: Seconds between requests (3.0 without key, 1.0 with key)
        """
        self.api_key = api_key
        self.rate_limit = rate_limit if api_key else max(rate_limit, 3.0)  # Free tier is slower
        self._last_request_time = 0.0
        
        # Setup cache
        if cache_dir:
            self.cache = Cache(cache_dir)
        else:
            self.cache = None
        
        # Session for connection reuse
        self.session = requests.Session()
        headers = {"User-Agent": "EcoloGRAPH/0.1.0"}
        if api_key:
            headers["x-api-key"] = api_key
        self.session.headers.update(headers)
        
        logger.info(f"SemanticScholarClient initialized (api_key={'set' if api_key else 'none'}, cache={'enabled' if cache_dir else 'disabled'})")
    
    # Fields to request from API
    PAPER_FIELDS = [
        "paperId", "externalIds", "title", "abstract", "year", "venue",
        "authors", "fieldsOfStudy", "s2FieldsOfStudy",
        "citationCount", "influentialCitationCount", "referenceCount",
        "isOpenAccess", "openAccessPdf", "tldr"
    ]
    
    def get_by_doi(self, doi: str) -> SemanticScholarMetadata | None:
        """
        Get metadata for a DOI.
        
        Args:
            doi: The DOI
            
        Returns:
            SemanticScholarMetadata or None if not found
        """
        doi = self._clean_doi(doi)
        if not doi:
            return None
        
        return self._get_paper(f"DOI:{doi}")
    
    def get_by_paper_id(self, paper_id: str) -> SemanticScholarMetadata | None:
        """Get metadata by Semantic Scholar paper ID."""
        return self._get_paper(paper_id)
    
    def search_by_title(
        self,
        title: str,
        year: int | None = None,
        limit: int = 5
    ) -> list[SemanticScholarMetadata]:
        """
        Search for papers by title.
        
        Args:
            title: Paper title to search
            year: Optional publication year filter
            limit: Maximum results
            
        Returns:
            List of matching papers
        """
        # Build query
        query = title
        if year:
            query = f"{title} {year}"
        
        # Check cache
        cache_key = f"s2:search:{hashlib.md5(f'{query}:{limit}'.encode()).hexdigest()}"
        if self.cache and cache_key in self.cache:
            logger.debug(f"Cache hit for title search: {title[:50]}...")
            return self.cache[cache_key]
        
        # Make request
        url = f"{S2_API_BASE}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": ",".join(self.PAPER_FIELDS)
        }
        
        data = self._make_request(url, params=params)
        
        if not data:
            return []
        
        # Parse results
        results = []
        for item in data.get("data", []):
            metadata = self._parse_paper(item)
            if metadata:
                results.append(metadata)
        
        # Apply year filter if specified
        if year:
            results = [r for r in results if r.year == year]
        
        # Cache results
        if self.cache:
            self.cache[cache_key] = results
        
        return results
    
    def _get_paper(self, paper_id: str) -> SemanticScholarMetadata | None:
        """Get paper by any identifier."""
        # Check cache
        cache_key = f"s2:paper:{paper_id}"
        if self.cache and cache_key in self.cache:
            logger.debug(f"Cache hit for paper: {paper_id}")
            return self.cache[cache_key]
        
        # Make request
        url = f"{S2_API_BASE}/paper/{paper_id}"
        params = {"fields": ",".join(self.PAPER_FIELDS)}
        
        data = self._make_request(url, params=params)
        
        if not data:
            return None
        
        # Parse response
        metadata = self._parse_paper(data)
        
        # Cache result
        if self.cache and metadata:
            self.cache[cache_key] = metadata
        
        return metadata
    
    def _make_request(self, url: str, params: dict | None = None) -> dict | None:
        """Make rate-limited request to Semantic Scholar."""
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        
        try:
            logger.debug(f"S2 request: {url}")
            response = self.session.get(url, params=params, timeout=30)
            self._last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.debug("Paper not found in Semantic Scholar")
                return None
            elif response.status_code == 429:
                logger.warning("Semantic Scholar rate limit hit, waiting...")
                time.sleep(10)
                return self._make_request(url, params)  # Retry once
            else:
                logger.warning(f"Semantic Scholar error: {response.status_code}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Semantic Scholar request failed: {e}")
            return None
    
    def _parse_paper(self, paper: dict) -> SemanticScholarMetadata | None:
        """Parse Semantic Scholar paper response."""
        if not paper:
            return None
        
        try:
            # Extract external IDs
            external_ids = paper.get("externalIds", {}) or {}
            
            # Extract open access PDF
            open_access = paper.get("openAccessPdf", {}) or {}
            
            # Extract TLDR
            tldr_data = paper.get("tldr", {}) or {}
            
            # Extract fields of study
            fields = paper.get("fieldsOfStudy") or []
            s2_fields = paper.get("s2FieldsOfStudy") or []
            
            return SemanticScholarMetadata(
                paper_id=paper.get("paperId"),
                doi=external_ids.get("DOI"),
                arxiv_id=external_ids.get("ArXiv"),
                title=paper.get("title"),
                abstract=paper.get("abstract"),
                year=paper.get("year"),
                venue=paper.get("venue"),
                authors=paper.get("authors", []),
                fields_of_study=fields,
                s2_fields_of_study=s2_fields,
                citation_count=paper.get("citationCount", 0),
                influential_citation_count=paper.get("influentialCitationCount", 0),
                reference_count=paper.get("referenceCount", 0),
                is_open_access=paper.get("isOpenAccess", False),
                open_access_pdf_url=open_access.get("url"),
                tldr=tldr_data.get("text"),
            )
        except Exception as e:
            logger.error(f"Error parsing S2 response: {e}")
            return None
    
    def _clean_doi(self, doi: str) -> str | None:
        """Clean and validate DOI."""
        if not doi:
            return None
        
        doi = doi.strip()
        for prefix in ["https://doi.org/", "http://doi.org/", "doi:", "DOI:"]:
            if doi.startswith(prefix):
                doi = doi[len(prefix):]
        
        if not doi or "/" not in doi:
            return None
        
        return doi
