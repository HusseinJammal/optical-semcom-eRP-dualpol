"""Conditioning block: c = [P, L] and FiLM layers (used by both the encoder and the decoder)."""
from .film import FiLM, FiLMGenerator, link_condition

__all__ = ["FiLM", "FiLMGenerator", "link_condition"]
