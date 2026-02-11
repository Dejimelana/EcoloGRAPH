"""
Extraction module - LLM-based ecological data extraction.
"""
from .entity_extractor import EntityExtractor
from .domain_classifier import DomainClassifier, ClassificationResult, StudyType

__all__ = ["EntityExtractor", "DomainClassifier", "ClassificationResult", "StudyType"]
