"""
Deep learning models for NeuroLite.

Provides implementations of popular deep learning architectures
for computer vision, NLP, and other domains.
"""

from .vision import register_vision_models

__all__ = [
    "register_vision_models"
]