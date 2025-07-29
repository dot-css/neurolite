"""
Unit tests for data preprocessing functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from neurolite.data import (
    Dataset, DatasetInfo, DataType,
    BasePreprocessor, ImagePreprocessor, TextPreprocessor, TabularPreprocessor,
    PreprocessorFactory, PreprocessingConfig, preprocess_data
)
from neurolite.core import DataError


class TestPreprocessingConfig:
    """Test preprocessing configuration.