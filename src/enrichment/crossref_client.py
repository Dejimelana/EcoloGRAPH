"""
CrossRef API client for enriching paper metadata.

CrossRef provides:
- Complete bibliographic data
- Author affiliations
- Funding information
- Citations and references
- DOI resolution
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any
import hashlib

import requests
from diskcache import Cache

logger = logging.getLogger(__name__)

# CrossRef API base URL
CROSSREF_API_BASE = "https://api.crossref.org"


@dataclass
class CrossRefMetadata:
    """Metadata retrieved from CrossRef."""
    # Core identifiers
    doi: str | None = None
    title: str | None = None
    
    # Authors with affiliations
    authors: list[dict[str, Any]] = field(default_factory=list)
    # Format: [{"given": "John", "family": "Smith", "affiliation": [{"name": "..."}]}]
    
    # Publication info
    journal: str | None = None
    publisher: str | None = None
    year: int | None = None
    month: int | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    
    # Subject and type
    subjects: list[str] = field(default_factory=list)
    document_type: str | None = None  # e.g., "journal-article"
    
    # Funding
    funders: list[dict[str, Any]] = field(default_factory=list)
    # Format: [{"name": "NSF", "award": ["123456"]}]
    
    # References
    references_count: int = 0
    is_referenced_by_count: int = 0  # Citation count
    
    # URLs
    resource_url: str | None = None
    
    # Raw response for additional fields
    raw: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "doi": self.doi,
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "publisher": self.publisher,
            "year": self.year,
            "month": self.month,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "subjects": self.subjects,
            "document_type": self.document_type,
            "funders": self.funders,
            "references_count": self.references_count,
            "is_referenced_by_count": self.is_referenced_by_count,
            "resource_url": self.resource_url,
        }


class CrossRefClient:
    """
    Client for the CrossRef API.
    
    Features:
    - DOI lookup
    - Title-based search
    - Caching to avoid repeated requests
    - Rate limiting
    """
    
    def __init__(
        self,
        mailto: str | None = None,
        cache_dir: str | None = None,
        rate_limit: float = 1.0
    ):
        """
        Initialize CrossRef client.
        
        Args:
            mailto: Email for polite pool (gets higher rate limits)
            cache_dir: Directory for caching responses
            rate_limit: Seconds between requests
        """
        self.mailto = mailto
        self.rate_limit = rate_limit
        self._last_request_time = 0.0
        
        # Setup cache
        if cache_dir:
            self.cache = Cache(cache_dir)
        else:
            self.cache = None
        
        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"EcoloGRAPH/0.1.0 (mailto:{mailto})" if mailto else "EcoloGRAPH/0.1.0"
        })
        
        logger.info(f"CrossRefClient initialized (mailto={mailto}, cache={'enabled' if cache_dir else 'disabled'})")
    
    def get_by_doi(self, doi: str) -> CrossRefMetadata | None:
        """
        Get metadata for a DOI.
        
        Args:
            doi: The DOI (with or without https://doi.org/ prefix)
            
        Returns:
            CrossRefMetadata or None if not found
        """
        # Clean DOI
        doi = self._clean_doi(doi)
        if not doi:
            return None
        
        # Check cache
        cache_key = f"crossref:doi:{doi}"
        if self.cache and cache_key in self.cache:
            logger.debug(f"Cache hit for DOI: {doi}")
            return self.cache[cache_key]
        
        # Make request
        url = f"{CROSSREF_API_BASE}/works/{doi}"
        data = self._make_request(url)
        
        if not data:
            return None
        
        # Parse response
        metadata = self._parse_work(data.get("message", {}))
        
        # Cache result
        if self.cache and metadata:
            self.cache[cache_key] = metadata
        
        return metadata
    
    def search_by_title(
        self,
        title: str,
        authors: list[str] | None = None,
        year: int | None = None,
        limit: int = 5
    ) -> list[CrossRefMetadata]:
        """
        Search for papers by title.
        
        Args:
            title: Paper title to search
            authors: Optional author names to filter
            year: Optional publication year
            limit: Maximum results to return
            
        Returns:
            List of matching CrossRefMetadata
        """
        # Build query
        query_parts = [f'query.title="{title}"']
        
        if authors:
            # Use first author's family name
            first_author = authors[0].split(",")[0].split()[-1]  # Get last name
            query_parts.append(f'query.author={first_author}')
        
        if year:
            query_parts.append(f'filter=from-pub-date:{year},until-pub-date:{year}')
        
        query_parts.append(f'rows={limit}')
        
        # Check cache
        cache_key = f"crossref:search:{hashlib.md5('&'.join(query_parts).encode()).hexdigest()}"
        if self.cache and cache_key in self.cache:
            logger.debug(f"Cache hit for title search: {title[:50]}...")
            return self.cache[cache_key]
        
        # Make request
        url = f"{CROSSREF_API_BASE}/works?{'&'.join(query_parts)}"
        data = self._make_request(url)
        
        if not data:
            return []
        
        # Parse results
        results = []
        items = data.get("message", {}).get("items", [])
        for item in items:
            metadata = self._parse_work(item)
            if metadata:
                results.append(metadata)
        
        # Cache results
        if self.cache:
            self.cache[cache_key] = results
        
        return results
    
    def _make_request(self, url: str) -> dict | None:
        """Make rate-limited request to CrossRef."""
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        
        try:
            logger.debug(f"CrossRef request: {url[:100]}...")
            response = self.session.get(url, timeout=30)
            self._last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.debug("DOI not found in CrossRef")
                return None
            else:
                logger.warning(f"CrossRef error: {response.status_code}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"CrossRef request failed: {e}")
            return None
    
    def _parse_work(self, work: dict) -> CrossRefMetadata | None:
        """Parse CrossRef work response into CrossRefMetadata."""
        if not work:
            return None
        
        try:
            # Extract year from date parts
            year = None
            month = None
            date_parts = work.get("published-print", work.get("published-online", {}))
            if date_parts:
                parts = date_parts.get("date-parts", [[]])[0]
                if parts:
                    year = parts[0] if len(parts) > 0 else None
                    month = parts[1] if len(parts) > 1 else None
            
            # Extract journal title
            journal = None
            container = work.get("container-title", [])
            if container:
                journal = container[0]
            
            # Build metadata
            return CrossRefMetadata(
                doi=work.get("DOI"),
                title=work.get("title", [""])[0] if work.get("title") else None,
                authors=work.get("author", []),
                journal=journal,
                publisher=work.get("publisher"),
                year=year,
                month=month,
                volume=work.get("volume"),
                issue=work.get("issue"),
                pages=work.get("page"),
                subjects=work.get("subject", []),
                document_type=work.get("type"),
                funders=work.get("funder", []),
                references_count=work.get("references-count", 0),
                is_referenced_by_count=work.get("is-referenced-by-count", 0),
                resource_url=work.get("URL"),
                raw=work
            )
        except Exception as e:
            logger.error(f"Error parsing CrossRef response: {e}")
            return None
    
    def _clean_doi(self, doi: str) -> str | None:
        """Clean and validate DOI."""
        if not doi:
            return None
        
        # Remove common prefixes
        doi = doi.strip()
        for prefix in ["https://doi.org/", "http://doi.org/", "doi:", "DOI:"]:
            if doi.startswith(prefix):
                doi = doi[len(prefix):]
        
        # Basic validation
        if not doi or "/" not in doi:
            return None
        
        return doi
