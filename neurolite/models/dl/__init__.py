"""
Deep learning models for NeuroLite.

Provides implementations of popular deep learning architectures
for computer vision, NLP, and other domains.
"""

from .vision import register_vision_models
from .nlp import register_nlp_models

__all__ = [
    "register_vision_models",
    "register_nlp_models"
]