"""
Version management for NeuroLite library.
"""

__version__ = "0.2.0"
__version_info__ = (0, 2, 0)

def get_version():
    """Get the current version string."""
    return __version__

def get_version_info():
    """Get the current version as a tuple of integers."""
    return __version_info__