"""
Version management for NeuroLite library.
"""

__version__ = "0.1.0"
__version_info__ = tuple(map(int, __version__.split(".")))

def get_version():
    """Get the current version string."""
    return __version__

def get_version_info():
    """Get the current version as a tuple of integers."""
    return __version_info__