"""
Inference module - Cross-domain knowledge linking and hypothesis generation.

Provides:
- CrossDomainLinker: Discover connections between scientific domains
- InferenceProposer: Generate research hypotheses from patterns
"""
from .cross_domain_linker import CrossDomainLinker, CrossDomainLink, LinkType
from .inference_proposer import (
    InferenceProposer, 
    Hypothesis, 
    HypothesisType, 
    ConfidenceLevel
)

__all__ = [
    "CrossDomainLinker",
    "CrossDomainLink",
    "LinkType",
    "InferenceProposer",
    "Hypothesis",
    "HypothesisType",
    "ConfidenceLevel"
]
