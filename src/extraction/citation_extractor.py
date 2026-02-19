"""
Citation extractor for building citation networks from papers.

Extracts bibliographic references from papers' References sections
using LLM-based parsing.
"""
import json
import logging
from pathlib import Path
from typing import Any

from ..core.llm_client import LLMClient
from ..core.schemas import Citation, SourceReference
from ..ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class CitationExtractor:
    """
    Extract citations from References sections of papers.
    
    Features:
    - Detects References/Bibliography sections
    - LLM-based citation parsing
    - Handles varied citation formats
    - Optional matching to existing papers in database
    """
    
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompts_dir: Path | str | None = None
    ):
        """
        Initialize citation extractor.
        
        Args:
            llm_client: LLM client instance (creates default if None)
            prompts_dir: Directory containing prompt templates
        """
        self.llm = llm_client or LLMClient(role="ingestion")
        
        # Load prompt
        if prompts_dir:
            self.prompts_dir = Path(prompts_dir)
        else:
            self.prompts_dir = Path(__file__).parent.parent.parent / "config" / "prompts"
        
        self.citation_prompt_template = self._load_prompt("extract_citations.txt")
        
        logger.info("CitationExtractor initialized")
    
    def extract_citations(
        self,
        chunks: list[DocumentChunk],
        doc_id: str
    ) -> list[Citation]:
        """
        Extract all citations from paper's References section.
        
        Args:
            chunks: All chunks from the paper
            doc_id: Document ID of the paper
            
        Returns:
            List of Citation objects found in References section
        """
        # Find References section
        ref_chunks = self._find_references_section(chunks)
        
        if not ref_chunks:
            logger.info(f"No References section found in {doc_id[:12]}")
            return []
        
        logger.info(f"Found References section with {len(ref_chunks)} chunks")
        
        # Combine reference chunks
        refs_text = "\n\n".join(chunk.text for chunk in ref_chunks)
        
        # Extract with LLM
        try:
            citations_raw = self._extract_with_llm(refs_text, doc_id)
            logger.info(f"Extracted {len(citations_raw)} citations from {doc_id[:12]}")
            return citations_raw
        except Exception as e:
            logger.error(f"Citation extraction failed for {doc_id[:12]}: {e}")
            return []
    
    def _find_references_section(
        self,
        chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        """
        Find chunks containing References/Bibliography section.
        
        Looks for section headers or content patterns indicating references.
        """
        keywords = [
            "references",
            "bibliography",
            "cited literature",
            "literature cited",
            "works cited"
        ]
        
        ref_chunks = []
        in_references = False
        
        for chunk in chunks:
            # Check section name
            section = (chunk.section or "").lower()
            
            # Check if this is start of references
            if any(kw in section for kw in keywords):
                in_references = True
                ref_chunks.append(chunk)
                continue
            
            # If in references section, continue collecting chunks
            # until we hit another section
            if in_references:
                # Stop if we hit another major section
                if section and not any(kw in section for kw in keywords):
                    # Check if section name indicates end (Acknowledgments, Appendix, etc.)
                    end_sections = ["acknowledgment", "appendix", "supplement", "author"]
                    if any(end in section for end in end_sections):
                        break
                
                ref_chunks.append(chunk)
        
        return ref_chunks
    
    def _extract_with_llm(
        self,
        refs_text: str,
        doc_id: str
    ) -> list[Citation]:
        """
        Extract citations using LLM.
        
        Args:
            refs_text: Combined text from References section
            doc_id: Document ID for source reference
            
        Returns:
            List of parsed Citation objects
        """
        # Fill prompt template
        filled_prompt = self.citation_prompt_template.format(
            references_text=refs_text[:8000]  # Limit to avoid context overflow
        )
        
        # Call LLM
        response = self.llm.generate(
            prompt=filled_prompt,
            temperature=0.1  # Low temperature for consistent extraction
        )
        
        # Parse JSON response
        try:
            citations_data = self._parse_response(response.content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error in citation extraction: {e}")
            return []
        
        # Convert to Citation objects
        citations = []
        for cit_data in citations_data:
            try:
                citation = Citation(
                    authors=cit_data.get('authors', []),
                    title=cit_data.get('title', 'Unknown'),
                    year=cit_data.get('year'),
                    journal=cit_data.get('journal'),
                    doi=cit_data.get('doi'),
                    cited_by_doc_id=doc_id,
                    source=SourceReference(
                        doc_id=doc_id,
                        section="References",
                        confidence=0.9,
                        is_inferred=False
                    )
                )
                citations.append(citation)
            except Exception as e:
                logger.warning(f"Failed to parse citation: {e}")
                continue
        
        return citations
    
    def _parse_response(self, content: str) -> list[dict]:
        """
        Parse LLM JSON response.
        
        Handles various JSON formats (with/without markdown, etc.)
        """
        import re
        # Remove markdown code blocks if present
        content = content.strip()
        # Remove Qwen3 thinking tags
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        if content.startswith("```"):
            # Extract content between ```
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        
        # Try to find JSON array
        start = content.find("[")
        end = content.rfind("]") + 1
        
        if start >= 0 and end > start:
            json_str = content[start:end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Fallback: try parsing entire content
        return json.loads(content)
    
    def _load_prompt(self, filename: str) -> str:
        """Load prompt template from file."""
        prompt_path = self.prompts_dir / filename
        if not prompt_path.exists():
            logger.error(f"Prompt file not found: {prompt_path}")
            return ""
        return prompt_path.read_text(encoding='utf-8')
