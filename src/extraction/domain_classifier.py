"""
Domain Classifier - Automatically classifies documents by scientific domain.

Uses LLM + keyword matching for robust classification.
Supports multi-label classification with domain scores and study type detection.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..core.llm_client import LLMClient
from ..core.domain_registry import DomainType, DomainRegistry, DomainConfig
from ..ingestion.pdf_parser import ParsedDocument
from ..ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class StudyType(str, Enum):
    """Type of scientific study methodology."""
    LABORATORY = "laboratory"           # In vitro, controlled experiments
    FIELD = "field"                     # Natural environment observations
    FIELD_EXPERIMENT = "field_experiment"  # Manipulative field study
    META_ANALYSIS = "meta_analysis"     # Statistical synthesis of studies
    MODELING = "modeling"               # Computational/mathematical models
    REVIEW = "review"                   # Literature review
    MIXED = "mixed"                     # Combination of approaches
    UNKNOWN = "unknown"


# Keywords for study type detection
STUDY_TYPE_KEYWORDS = {
    StudyType.LABORATORY: [
        "in vitro", "laboratory", "culture", "cultured", "incubation",
        "controlled conditions", "experiment", "experimental", "petri",
        "flask", "bioreactor", "grown in", "maintained at"
    ],
    StudyType.FIELD: [
        "field study", "field survey", "sampling", "sampled", "collected",
        "in situ", "natural habitat", "wild population", "observed in",
        "transect", "plot", "station", "site"
    ],
    StudyType.FIELD_EXPERIMENT: [
        "field experiment", "manipulative", "enclosure", "exclusion",
        "transplant", "treatment", "control plot", "experimental plot"
    ],
    StudyType.META_ANALYSIS: [
        "meta-analysis", "meta analysis", "systematic review",
        "pooled", "effect size", "heterogeneity", "publication bias"
    ],
    StudyType.MODELING: [
        "model", "simulation", "predict", "projection", "scenario",
        "parameter", "sensitivity analysis", "monte carlo"
    ],
    StudyType.REVIEW: [
        "review", "literature", "synthesis", "overview", "state of the art"
    ]
}


@dataclass
class ClassificationResult:
    """Result of domain classification with multi-label support."""
    # Primary classification
    primary_domain: DomainType
    confidence: float
    
    # Multi-label: all domains with scores
    domain_scores: dict[DomainType, float] = field(default_factory=dict)
    
    # Study methodology
    study_type: StudyType = StudyType.UNKNOWN
    study_type_confidence: float = 0.0
    
    # Secondary domains (for backward compatibility)
    secondary_domains: list[tuple[DomainType, float]] = field(default_factory=list)
    
    # Metadata
    reasoning: str = ""
    method: str = ""  # "llm", "keyword", "hybrid"


class DomainClassifier:
    """
    Classifies documents into scientific domains.
    
    Uses a hybrid approach:
    1. Keyword matching for fast initial classification
    2. LLM for disambiguation and confidence scoring
    """
    
    CLASSIFICATION_PROMPT = """You are a scientific document classifier. Analyze the following text and classify it into one of these domains:

DOMAINS:
{domains_list}

TEXT TO CLASSIFY:
---
{text}
---

Respond with ONLY a JSON object:
{{
    "domain": "domain_code",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Choose the most specific domain that applies. Use "unknown" only if the text is not scientific or doesn't match any domain."""

    def __init__(self, llm_client: LLMClient | None = None):
        """
        Initialize classifier.
        
        Args:
            llm_client: LLM client for classification (optional, uses keyword-only if None)
        """
        self.llm = llm_client
        logger.info("DomainClassifier initialized")
    
    def classify_document(
        self,
        document: ParsedDocument,
        use_llm: bool = True
    ) -> ClassificationResult:
        """
        Classify a parsed document.
        
        Args:
            document: Parsed document to classify
            use_llm: Whether to use LLM for classification
            
        Returns:
            ClassificationResult with domain and confidence
        """
        # Build text for classification (title + abstract + first sections)
        text_parts = []
        
        if document.title:
            text_parts.append(f"Title: {document.title}")
        
        if document.abstract:
            text_parts.append(f"Abstract: {document.abstract}")
        
        # Add first few sections
        for section in document.sections[:3]:
            text_parts.append(f"{section.title}: {section.text[:500]}")
        
        text = "\n\n".join(text_parts)
        
        return self.classify_text(text, use_llm=use_llm)
    
    def classify_chunk(
        self,
        chunk: DocumentChunk,
        use_llm: bool = False
    ) -> ClassificationResult:
        """
        Classify a single chunk (lightweight, keyword-only by default).
        
        Args:
            chunk: Document chunk to classify
            use_llm: Whether to use LLM
            
        Returns:
            ClassificationResult
        """
        return self.classify_text(chunk.text, use_llm=use_llm)
    
    def classify_text(
        self,
        text: str,
        use_llm: bool = True
    ) -> ClassificationResult:
        """
        Classify text into a scientific domain.
        
        Args:
            text: Text to classify
            use_llm: Whether to use LLM for classification
            
        Returns:
            ClassificationResult
        """
        # Step 1: Keyword matching
        keyword_scores = DomainRegistry.get_by_keyword(text)
        
        # Detect study type (always do this)
        study_type, study_conf = self._detect_study_type(text)
        
        # Build domain_scores dict from keyword results
        domain_scores = {dt: score for dt, score in keyword_scores}
        
        # If strong keyword match and no LLM requested
        if keyword_scores and keyword_scores[0][1] >= 0.15:
            if not use_llm or not self.llm:
                return ClassificationResult(
                    primary_domain=keyword_scores[0][0],
                    confidence=min(keyword_scores[0][1] * 1.2, 1.0),
                    domain_scores=domain_scores,
                    study_type=study_type,
                    study_type_confidence=study_conf,
                    secondary_domains=keyword_scores[1:3],
                    reasoning=f"Keyword match: {keyword_scores[0][1]:.2f}",
                    method="keyword"
                )
        
        # Step 2: LLM classification
        if use_llm and self.llm:
            llm_result = self._classify_with_llm(text)
            
            if llm_result:
                # Combine with keyword scores (passes text for study_type)
                return self._merge_results(llm_result, keyword_scores, text)
        
        # Fallback: use keyword results or unknown
        if keyword_scores:
            return ClassificationResult(
                primary_domain=keyword_scores[0][0],
                confidence=keyword_scores[0][1],
                domain_scores=domain_scores,
                study_type=study_type,
                study_type_confidence=study_conf,
                secondary_domains=keyword_scores[1:3],
                reasoning="Keyword-based classification",
                method="keyword"
            )
        
        return ClassificationResult(
            primary_domain=DomainType.UNKNOWN,
            confidence=0.0,
            domain_scores={},
            study_type=study_type,
            study_type_confidence=study_conf,
            secondary_domains=[],
            reasoning="No matching domain found",
            method="fallback"
        )
    
    def _classify_with_llm(self, text: str) -> ClassificationResult | None:
        """Use LLM to classify text."""
        try:
            # Build domains list
            domains = DomainRegistry.get_all()
            domains_list = "\n".join([
                f"- {d.domain_type.value}: {d.display_name} - {d.description}"
                for d in domains
            ])
            
            # Build prompt
            prompt = self.CLASSIFICATION_PROMPT.format(
                domains_list=domains_list,
                text=text[:2000]  # Limit text length
            )
            
            # Call LLM
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.1
            )
            
            # Parse response
            import json
            content = response.content.strip()
            
            # Extract JSON
            if "{" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                data = json.loads(json_str)
                
                domain_str = data.get("domain", "unknown")
                try:
                    domain_type = DomainType(domain_str)
                except ValueError:
                    domain_type = DomainType.UNKNOWN
                
                return ClassificationResult(
                    primary_domain=domain_type,
                    confidence=float(data.get("confidence", 0.5)),
                    secondary_domains=[],
                    reasoning=data.get("reasoning", "LLM classification"),
                    method="llm"
                )
                
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
        
        return None
    
    def _merge_results(
        self,
        llm_result: ClassificationResult,
        keyword_scores: list[tuple[DomainType, float]],
        text: str = ""
    ) -> ClassificationResult:
        """Merge LLM and keyword results."""
        
        # Check if LLM and keyword agree
        keyword_top = keyword_scores[0][0] if keyword_scores else None
        
        if keyword_top == llm_result.primary_domain:
            # Agreement - boost confidence
            combined_confidence = min(
                (llm_result.confidence + keyword_scores[0][1]) / 1.5,
                1.0
            )
            
            # Build domain_scores from keyword results
            domain_scores = {dt: score for dt, score in keyword_scores}
            domain_scores[llm_result.primary_domain] = combined_confidence
            
            # Detect study type
            study_type, study_conf = self._detect_study_type(text)
            
            return ClassificationResult(
                primary_domain=llm_result.primary_domain,
                confidence=combined_confidence,
                domain_scores=domain_scores,
                study_type=study_type,
                study_type_confidence=study_conf,
                secondary_domains=keyword_scores[1:3],
                reasoning=f"LLM + keyword agreement: {llm_result.reasoning}",
                method="hybrid"
            )
        
        # Disagreement - use LLM but note secondary
        secondary = [(keyword_top, keyword_scores[0][1])] if keyword_top else []
        secondary.extend(keyword_scores[1:2])
        
        # Build domain_scores
        domain_scores = {dt: score for dt, score in keyword_scores}
        domain_scores[llm_result.primary_domain] = llm_result.confidence
        
        # Detect study type
        study_type, study_conf = self._detect_study_type(text)
        
        return ClassificationResult(
            primary_domain=llm_result.primary_domain,
            confidence=llm_result.confidence * 0.9,
            domain_scores=domain_scores,
            study_type=study_type,
            study_type_confidence=study_conf,
            secondary_domains=secondary,
            reasoning=f"LLM override: {llm_result.reasoning}",
            method="llm"
        )
    
    def _detect_study_type(self, text: str) -> tuple[StudyType, float]:
        """
        Detect the type of study from text.
        
        Returns:
            Tuple of (StudyType, confidence)
        """
        text_lower = text.lower()
        
        scores = {}
        for study_type, keywords in STUDY_TYPE_KEYWORDS.items():
            matched = sum(1 for kw in keywords if kw in text_lower)
            if matched > 0:
                scores[study_type] = matched / len(keywords)
        
        if not scores:
            return StudyType.UNKNOWN, 0.0
        
        # Get best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # Check for mixed methodology
        high_scores = [st for st, s in scores.items() if s >= best_score * 0.7]
        if len(high_scores) > 1:
            return StudyType.MIXED, best_score * 0.8
        
        return best_type, min(best_score * 1.5, 1.0)
    
    def get_domain_config(self, domain_type: DomainType) -> DomainConfig | None:
        """Get configuration for a classified domain."""
        return DomainRegistry.get(domain_type)
    
    def get_top_domains(
        self, 
        result: ClassificationResult, 
        threshold: float = 0.1
    ) -> list[tuple[DomainType, float]]:
        """
        Get all domains above threshold from a classification result.
        
        Args:
            result: Classification result
            threshold: Minimum score to include
            
        Returns:
            List of (domain_type, score) tuples sorted by score
        """
        if not result.domain_scores:
            return [(result.primary_domain, result.confidence)]
        
        filtered = [
            (dt, score) for dt, score in result.domain_scores.items()
            if score >= threshold
        ]
        return sorted(filtered, key=lambda x: x[1], reverse=True)
