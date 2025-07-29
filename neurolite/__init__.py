"""
NeuroLite - AI/ML/DL/NLP Productivity Library

A high-level abstraction layer that provides a unified, minimal-code interface 
for machine learning workflows.
"""

__version__ = "0.1.0"
__author__ = "NeuroLite Team"
__email__ = "team@neurolite.ai"

from .api import train, deploy
from .core.exceptions import NeuroLiteError

__all__ = ["train", "deploy", "NeuroLiteError"]