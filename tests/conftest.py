"""
Pytest configuration and fixtures.
"""
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_text():
    """Sample text for testing extraction."""
    return """
    The Atlantic cod (Gadus morhua) is distributed across the North Atlantic.
    Adult specimens are typically found at depths of 150-200m during summer months.
    Temperature tolerance ranges from 0°C to 20°C, with optimal temperatures between 4-7°C.
    This species shows seasonal migration patterns, moving to shallower waters (30-80m) 
    for spawning in late winter.
    """


@pytest.fixture
def sample_paper_metadata():
    """Sample paper metadata for testing."""
    return {
        "title": "Depth distribution of Atlantic cod in the North Sea",
        "authors": ["Smith, J.", "García, M.", "Johnson, K."],
        "year": 2023,
        "doi": "10.1234/example.2023.001",
        "journal": "Marine Ecology Progress Series"
    }
