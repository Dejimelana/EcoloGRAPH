"""
Inference Proposer for generating scientific hypotheses.

Uses cross-domain links and knowledge patterns to:
- Generate novel research hypotheses
- Identify knowledge gaps
- Suggest experimental designs
- Predict ecological effects
"""
import logging
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import json

from ..core.llm_client import LLMClient
from ..core.domain_registry import DomainType, DomainRegistry
from .cross_domain_linker import CrossDomainLinker, CrossDomainLink, LinkType

logger = logging.getLogger(__name__)


class HypothesisType(str, Enum):
    """Types of generated hypotheses."""
    ECOLOGICAL_EFFECT = "ecological_effect"  # Predicting cascade effects
    CONSERVATION_IMPLICATION = "conservation_implication"  # Conservation relevance
    METHODOLOGICAL_TRANSFER = "methodological_transfer"  # Method applicability
    KNOWLEDGE_GAP = "knowledge_gap"  # Identified gaps
    DATA_SYNTHESIS = "data_synthesis"  # Combining measurements
    NOVEL_INTERACTION = "novel_interaction"  # Undocumented interactions


class ConfidenceLevel(str, Enum):
    """Confidence levels for hypotheses."""
    HIGH = "high"  # Strong evidence, multiple sources
    MEDIUM = "medium"  # Some evidence, plausible
    LOW = "low"  # Speculative but interesting
    EXPLORATORY = "exploratory"  # Novel, needs investigation


@dataclass
class Hypothesis:
    """A generated scientific hypothesis."""
    
    hypothesis_id: str
    hypothesis_type: HypothesisType
    
    # The hypothesis itself
    statement: str
    rationale: str
    
    # Evidence and confidence
    confidence: ConfidenceLevel
    confidence_score: float  # 0-1
    supporting_evidence: list[str] = field(default_factory=list)
    
    # Source links
    source_links: list[str] = field(default_factory=list)  # Link IDs
    source_domains: list[DomainType] = field(default_factory=list)
    
    # Suggested actions
    suggested_experiments: list[str] = field(default_factory=list)
    key_questions: list[str] = field(default_factory=list)
    
    # Metadata
    species_involved: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "type": self.hypothesis_type.value,
            "statement": self.statement,
            "rationale": self.rationale,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "supporting_evidence": self.supporting_evidence,
            "source_links": self.source_links,
            "source_domains": [d.value for d in self.source_domains],
            "suggested_experiments": self.suggested_experiments,
            "key_questions": self.key_questions,
            "species_involved": self.species_involved
        }
    
    def to_markdown(self) -> str:
        """Format hypothesis as markdown."""
        emoji = {
            HypothesisType.ECOLOGICAL_EFFECT: "🌿",
            HypothesisType.CONSERVATION_IMPLICATION: "🛡️",
            HypothesisType.METHODOLOGICAL_TRANSFER: "🔬",
            HypothesisType.KNOWLEDGE_GAP: "❓",
            HypothesisType.DATA_SYNTHESIS: "📊",
            HypothesisType.NOVEL_INTERACTION: "🔗"
        }.get(self.hypothesis_type, "💡")
        
        conf_emoji = {
            ConfidenceLevel.HIGH: "🟢",
            ConfidenceLevel.MEDIUM: "🟡",
            ConfidenceLevel.LOW: "🟠",
            ConfidenceLevel.EXPLORATORY: "🔵"
        }.get(self.confidence, "⚪")
        
        md = f"""### {emoji} {self.hypothesis_type.value.replace('_', ' ').title()}

**Hypothesis:** {self.statement}

**Rationale:** {self.rationale}

**Confidence:** {conf_emoji} {self.confidence.value} ({self.confidence_score:.0%})
"""
        
        if self.supporting_evidence:
            md += "\n**Evidence:**\n"
            for ev in self.supporting_evidence[:3]:
                md += f"- {ev}\n"
        
        if self.suggested_experiments:
            md += "\n**Suggested Experiments:**\n"
            for exp in self.suggested_experiments:
                md += f"- {exp}\n"
        
        if self.key_questions:
            md += "\n**Key Questions:**\n"
            for q in self.key_questions:
                md += f"- {q}\n"
        
        return md


HYPOTHESIS_PROMPT = """You are a scientific hypothesis generator for ecological research.

Given the following cross-domain connection between two scientific fields, generate a novel research hypothesis.

## Cross-Domain Link
- **Type:** {link_type}
- **Source Domain:** {source_domain}
- **Source Entity:** {source_entity}
- **Target Domain:** {target_domain}
- **Target Entity:** {target_entity}
- **Description:** {description}

## Additional Context
{context}

## Task
Generate ONE specific, testable hypothesis that emerges from this cross-domain connection.

Respond in JSON format:
{{
    "hypothesis_type": "ecological_effect|conservation_implication|methodological_transfer|knowledge_gap|data_synthesis|novel_interaction",
    "statement": "Clear, specific hypothesis statement",
    "rationale": "Scientific reasoning for this hypothesis",
    "confidence": "high|medium|low|exploratory",
    "confidence_score": 0.0-1.0,
    "supporting_evidence": ["evidence point 1", "evidence point 2"],
    "suggested_experiments": ["experiment 1", "experiment 2"],
    "key_questions": ["research question 1", "research question 2"],
    "species_involved": ["species1", "species2"]
}}
"""


class InferenceProposer:
    """
    Generates scientific hypotheses from cross-domain links.
    
    Uses LLM reasoning to:
    1. Analyze cross-domain connections
    2. Identify potential implications
    3. Generate testable hypotheses
    4. Suggest experimental approaches
    """
    
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        cross_domain_linker: CrossDomainLinker | None = None
    ):
        """
        Initialize inference proposer.
        
        Args:
            llm_client: LLM client for reasoning
            cross_domain_linker: Linker for discovering connections
        """
        self.llm = llm_client
        self.linker = cross_domain_linker
        
        logger.info("InferenceProposer initialized")
    
    def generate_hypothesis_from_link(
        self,
        link: CrossDomainLink,
        additional_context: str = ""
    ) -> Hypothesis | None:
        """
        Generate a hypothesis from a cross-domain link.
        
        Args:
            link: The cross-domain connection to analyze
            additional_context: Extra context from retrieval
            
        Returns:
            Generated Hypothesis or None if failed
        """
        if not self.llm:
            return self._generate_rule_based_hypothesis(link)
        
        # Build prompt
        prompt = HYPOTHESIS_PROMPT.format(
            link_type=link.link_type.value,
            source_domain=link.source_domain.value,
            source_entity=link.source_entity,
            target_domain=link.target_domain.value,
            target_entity=link.target_entity,
            description=link.description,
            context=additional_context or "No additional context available."
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt="You are a scientific hypothesis generator. Respond only in valid JSON.",
                temperature=0.7
            )
            
            # Parse JSON response
            # Handle markdown code blocks
            text = response.content
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            data = json.loads(text.strip())
            
            return self._parse_hypothesis_response(data, link)
            
        except Exception as e:
            logger.error(f"Hypothesis generation failed: {e}")
            return self._generate_rule_based_hypothesis(link)
    
    def _parse_hypothesis_response(
        self, 
        data: dict, 
        link: CrossDomainLink
    ) -> Hypothesis:
        """Parse LLM response into Hypothesis object."""
        import hashlib
        
        hypothesis_id = hashlib.md5(
            f"{link.link_id}:{data.get('statement', '')}".encode()
        ).hexdigest()[:12]
        
        # Map hypothesis type
        try:
            h_type = HypothesisType(data.get("hypothesis_type", "ecological_effect"))
        except ValueError:
            h_type = HypothesisType.ECOLOGICAL_EFFECT
        
        # Map confidence
        try:
            conf = ConfidenceLevel(data.get("confidence", "medium"))
        except ValueError:
            conf = ConfidenceLevel.MEDIUM
        
        return Hypothesis(
            hypothesis_id=hypothesis_id,
            hypothesis_type=h_type,
            statement=data.get("statement", ""),
            rationale=data.get("rationale", ""),
            confidence=conf,
            confidence_score=float(data.get("confidence_score", 0.5)),
            supporting_evidence=data.get("supporting_evidence", []),
            source_links=[link.link_id],
            source_domains=[link.source_domain, link.target_domain],
            suggested_experiments=data.get("suggested_experiments", []),
            key_questions=data.get("key_questions", []),
            species_involved=data.get("species_involved", [])
        )
    
    def _generate_rule_based_hypothesis(
        self, 
        link: CrossDomainLink
    ) -> Hypothesis:
        """Generate hypothesis using rules when LLM is unavailable."""
        import hashlib
        
        hypothesis_id = hashlib.md5(
            f"rule:{link.link_id}".encode()
        ).hexdigest()[:12]
        
        # Template-based hypothesis generation
        templates = {
            LinkType.SHARED_SPECIES: {
                "type": HypothesisType.DATA_SYNTHESIS,
                "statement": f"Data on {link.source_entity} from {link.source_domain.value} "
                           f"and {link.target_domain.value} can be synthesized to build "
                           f"a more complete ecological profile.",
                "rationale": "Same species studied in different domains often yields "
                           "complementary data that can enhance understanding.",
                "experiments": [
                    f"Meta-analysis combining {link.source_domain.value} and {link.target_domain.value} data",
                    "Cross-validation of measurements between studies"
                ]
            },
            LinkType.ECOLOGICAL_CASCADE: {
                "type": HypothesisType.ECOLOGICAL_EFFECT,
                "statement": f"Changes affecting {link.source_entity} may cascade through "
                           f"ecological relationships to impact {link.target_entity}.",
                "rationale": "Ecological networks transmit effects through species interactions.",
                "experiments": [
                    "Network modeling of cascade effects",
                    "Field observations of indirect interactions"
                ]
            },
            LinkType.COMPLEMENTARY_DATA: {
                "type": HypothesisType.KNOWLEDGE_GAP,
                "statement": f"The semantic similarity between {link.source_domain.value} and "
                           f"{link.target_domain.value} research suggests unexplored "
                           f"interdisciplinary connections.",
                "rationale": "High semantic similarity may indicate shared mechanisms or phenomena.",
                "experiments": [
                    "Literature review of shared concepts",
                    "Comparative analysis of methodologies"
                ]
            }
        }
        
        template = templates.get(link.link_type, {
            "type": HypothesisType.NOVEL_INTERACTION,
            "statement": f"Connection between {link.source_domain.value} and {link.target_domain.value} "
                       f"via {link.source_entity} warrants further investigation.",
            "rationale": "Cross-domain connections may reveal novel ecological patterns.",
            "experiments": ["Exploratory data analysis"]
        })
        
        return Hypothesis(
            hypothesis_id=hypothesis_id,
            hypothesis_type=template["type"],
            statement=template["statement"],
            rationale=template["rationale"],
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=link.confidence,
            source_links=[link.link_id],
            source_domains=[link.source_domain, link.target_domain],
            suggested_experiments=template.get("experiments", []),
            species_involved=[link.source_entity] if link.source_entity else []
        )
    
    def generate_hypotheses_batch(
        self,
        links: list[CrossDomainLink],
        max_hypotheses: int = 10
    ) -> list[Hypothesis]:
        """Generate hypotheses for multiple links."""
        hypotheses = []
        
        for link in links[:max_hypotheses]:
            hypothesis = self.generate_hypothesis_from_link(link)
            if hypothesis:
                hypotheses.append(hypothesis)
        
        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence_score, reverse=True)
        
        return hypotheses
    
    def identify_knowledge_gaps(
        self,
        domain: DomainType,
        species_list: list[str]
    ) -> list[Hypothesis]:
        """
        Identify knowledge gaps for species in a domain.
        
        Compares what is known about species across domains
        and identifies missing information.
        """
        gaps = []
        
        # This would query the graph to find:
        # 1. Species with data in some domains but not others
        # 2. Missing measurement types
        # 3. Geographic gaps
        
        # For now, return template hypothesis
        if species_list:
            import hashlib
            gap_id = hashlib.md5(f"gap:{domain.value}".encode()).hexdigest()[:12]
            
            gap = Hypothesis(
                hypothesis_id=gap_id,
                hypothesis_type=HypothesisType.KNOWLEDGE_GAP,
                statement=f"Systematic review needed to identify data gaps for "
                         f"{', '.join(species_list[:3])} in {domain.value}.",
                rationale="Comprehensive data coverage enables better ecological modeling.",
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.8,
                source_domains=[domain],
                suggested_experiments=[
                    "Systematic literature review",
                    "Database completeness analysis"
                ],
                species_involved=species_list[:5]
            )
            gaps.append(gap)
        
        return gaps
    
    def generate_report(
        self,
        hypotheses: list[Hypothesis],
        title: str = "Cross-Domain Inference Report"
    ) -> str:
        """Generate a markdown report of hypotheses."""
        md = f"# {title}\n\n"
        md += f"**Generated:** {len(hypotheses)} hypotheses\n\n"
        
        # Group by type
        by_type: dict[HypothesisType, list[Hypothesis]] = {}
        for h in hypotheses:
            if h.hypothesis_type not in by_type:
                by_type[h.hypothesis_type] = []
            by_type[h.hypothesis_type].append(h)
        
        for h_type, hyps in by_type.items():
            md += f"## {h_type.value.replace('_', ' ').title()}\n\n"
            for h in hyps:
                md += h.to_markdown() + "\n---\n\n"
        
        return md
