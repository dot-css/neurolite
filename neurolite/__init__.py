"""
NeuroLite - AI/ML/DL/NLP Productivity Library

A high-level abstraction layer that provides a unified, minimal-code interface 
for machine learning workflows.
"""

from ._version import __version__, get_version, get_version_info

__author__ = "NeuroLite Team"
__email__ = "team@neurolite.ai"

from .api import train, deploy
from .core.exceptions import NeuroLiteError

__all__ = ["train", "deploy", "NeuroLiteError", "__version__", "get_version", "get_version_info"]