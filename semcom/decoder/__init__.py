"""Semantic decoder block: soft symbols -> CNN + FiLM -> reconstruction and classification heads."""
from .decoder import SemanticDecoder

__all__ = ["SemanticDecoder"]
