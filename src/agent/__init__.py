"""
EcoloGRAPH Agent Module.

Provides the Query Agent and its tools for interactive ecological research.
"""

from .query_agent import QueryAgent, detect_loaded_model
from .tool_registry import ALL_TOOLS, get_tool_descriptions

__all__ = [
    "QueryAgent",
    "detect_loaded_model",
    "ALL_TOOLS",
    "get_tool_descriptions",
]
