"""
Entity highlighting utility for chunk visualization.

Highlights species, locations, and other entities in text
for better readability in the UI.
"""
import re
from typing import Dict, List


def highlight_entities(text: str, entities: Dict[str, List[str]]) -> str:
    """
    Highlight entities in text using HTML spans.
    
    Args:
        text: Chunk text to highlight
        entities: Dictionary with entity types and values
                 {"species": [...], "locations": [...]}
    
    Returns:
        HTML-formatted text with colored highlights
    """
    highlighted = text
    
    # Highlight species in red/coral
    for species in entities.get("species", []):
        if not species:
            continue
        pattern = re.compile(re.escape(species), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<span style="background-color:#FF6B6B;color:white;padding:2px 6px;'
            f'border-radius:4px;font-weight:500;margin:0 2px">{species}</span>',
            highlighted
        )
    
    # Highlight locations in green/teal
    for location in entities.get("locations", []):
        if not location:
            continue
        pattern = re.compile(re.escape(location), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<span style="background-color:#95E1D3;color:#1a1a1a;padding:2px 6px;'
            f'border-radius:4px;font-weight:500;margin:0 2px">{location}</span>',
            highlighted
        )
    
    # Highlight methods/techniques in blue (optional)
    for method in entities.get("methods", []):
        if not method:
            continue
        pattern = re.compile(re.escape(method), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<span style="background-color:#4ECDC4;color:white;padding:2px 6px;'
            f'border-radius:4px;font-weight:500;margin:0 2px">{method}</span>',
            highlighted
        )
    
    return highlighted


def extract_entities_from_chunk(chunk: dict) -> Dict[str, List[str]]:
    """
    Extract entities from chunk metadata/payload.
    
    Args:
        chunk: Chunk dictionary with potential entity fields
        
    Returns:
        Dictionary of entity types and values
    """
    entities = {
        "species": [],
        "locations": [],
        "methods": []
    }
    
    # Extract from various possible fields
    if "species" in chunk and isinstance(chunk["species"], list):
        entities["species"] = chunk["species"]
    
    if "locations" in chunk and isinstance(chunk["locations"], list):
        entities["locations"] = chunk["locations"]
    
    if "methods" in chunk and isinstance(chunk["methods"], list):
        entities["methods"] = chunk["methods"]
    
    return entities


def create_legend() -> str:
    """
    Create HTML legend for entity color coding.
    
    Returns:
        HTML string with legend
    """
    return """
    <div style="margin-bottom:1rem;padding:0.5rem;background:#f8f9fa;border-radius:4px">
        <small style="font-weight:600;margin-right:1rem">Entity Types:</small>
        <span style="background-color:#FF6B6B;color:white;padding:2px 8px;border-radius:4px;margin-right:0.5rem">Species</span>
        <span style="background-color:#95E1D3;color:#1a1a1a;padding:2px 8px;border-radius:4px;margin-right:0.5rem">Locations</span>
        <span style="background-color:#4ECDC4;color:white;padding:2px 8px;border-radius:4px">Methods</span>
    </div>
    """
