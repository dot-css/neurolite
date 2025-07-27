# NeuroLite

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/neurolite/neurolite)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://github.com/neurolite/neurolite)

**NeuroLite** is an automated AI/ML library that intelligently detects data types and automatically applies the best models with minimal code. It simplifies the machine learning workflow by providing comprehensive data analysis, quality assessment, and model recommendations.

## 🚀 Features

### 🔍 Intelligent Data Detection
- **File Format Detection**: Supports 20+ formats (CSV, JSON, XML, Excel, Parquet, HDF5, images, audio, video)
- **Data Structure Analysis**: Automatically identifies tabular, time series, image, text, and audio data
- **Column Type Classification**: Numerical, categorical, temporal, text, and binary data types
- **Domain-Specific Detection**: Computer vision, NLP, and time series task identification

### 📊 Comprehensive Quality Assessment
- **Missing Data Analysis**: MCAR, MAR, MNAR pattern detection with imputation recommendations
- **Data Consistency Validation**: Duplicate detection, format consistency, range validation
- **Statistical Properties**: Distribution analysis, correlation detection, outlier identification
- **Quality Metrics**: Completeness, consistency, validity, and uniqueness scoring

### 🤖 Automated Model Recommendations
- **Traditional ML Models**: Decision Trees, Random Forest, SVM, Linear models
- **Deep Learning**: CNN, RNN/LSTM, Transformers, AutoEncoders
- **Preprocessing Suggestions**: Normalization, encoding, feature scaling strategies
- **Performance Estimation**: Resource requirements, complexity assessment

## 📦 Installation

### Basic Installation
```bash
pip install neurolite
```

### Development Installation
```bash
git clone https://github.com/dot-css/neurolite
cd neurolite
pip install -e .[dev]
```

### Full Installation (with all optional dependencies)
```bash
pip install neurolite[all]
```

## 🏃‍♂️ Quick Start

### Basic Usage (3 lines of code!)
```python
from neurolite import DataProfiler

# Analyze any dataset with a single function call
profiler = DataProfiler()
report = profiler.analyze('your_data.csv')

# Get comprehensive insights
print(f"Data type: {report.data_structure.structure_type}")
print(f"Quality score: {report.quality_metrics.completeness:.2f}")
print(f"Recommended models: {[r.model_name for r in report.model_recommendations[:3]]}")
```

### Advanced Usage
```python
from neurolite import DataProfiler
from neurolite.detectors import QualityDetector, DataTypeDetector
import pandas as pd

# Load your data
df = pd.read_csv('your_dataset.csv')

# Initialize profiler
profiler = DataProfiler()

# Perform comprehensive analysis
report = profiler.analyze(df)

# Access detailed results
print("=== File Information ===")
print(f"Format: {report.file_info.format_type}")
print(f"Structure: {report.data_structure.structure_type}")
print(f"Dimensions: {report.data_structure.dimensions}")

print("\n=== Quality Assessment ===")
print(f"Completeness: {report.quality_metrics.completeness:.2%}")
print(f"Consistency: {report.quality_metrics.consistency:.2%}")
print(f"Missing Pattern: {report.quality_metrics.missing_pattern}")

print("\n=== Column Analysis ===")
for col, analysis in report.column_analysis.items():
    print(f"{col}: {analysis.primary_type} ({analysis.subtype})")

print("\n=== Model Recommendations ===")
for rec in report.model_recommendations[:5]:
    print(f"- {rec.model_name} ({rec.confidence:.2%} confidence)")
    print(f"  Rationale: {rec.rationale}")
```

### Specific Detectors
```python
from neurolite.detectors import QualityDetector, DataTypeDetector, FileDetector

# Quality assessment
quality_detector = QualityDetector()
quality_report = quality_detector.analyze_quality(df)
missing_analysis = quality_detector.detect_missing_patterns(df)
duplicate_analysis = quality_detector.find_duplicates(df)

# Data type detection
type_detector = DataTypeDetector()
column_types = type_detector.classify_columns(df)

# File format detection
file_detector = FileDetector()
file_format = file_detector.detect_format('data.csv')
data_structure = file_detector.detect_structure(df)
```

## 📋 Supported Data Types

### File Formats
- **Tabular**: CSV, TSV, Excel (.xlsx, .xls), Parquet, HDF5
- **Structured**: JSON, XML, YAML
- **Images**: PNG, JPG, JPEG, TIFF, BMP, GIF
- **Audio**: WAV, MP3, FLAC, OGG
- **Video**: MP4, AVI, MOV, MKV
- **Text**: TXT, MD, PDF, DOC

### Data Structures
- **Tabular Data**: Structured datasets with rows and columns
- **Time Series**: Sequential data with temporal patterns
- **Image Data**: Computer vision datasets
- **Text Corpus**: Natural language processing datasets
- **Audio Signals**: Speech and audio analysis datasets

### Column Types
- **Numerical**: Integer, float, continuous, discrete
- **Categorical**: Nominal, ordinal, high/low cardinality
- **Temporal**: Dates, timestamps, time series
- **Text**: Natural language, categorical text, structured text
- **Binary**: Boolean, binary encoded data

## 🔧 Configuration

### Environment Variables
```bash
export NEUROLITE_CACHE_DIR="/path/to/cache"
export NEUROLITE_LOG_LEVEL="INFO"
export NEUROLITE_MAX_MEMORY="8GB"
```

### Configuration File
Create `~/.neurolite/config.yaml`:
```yaml
cache:
  enabled: true
  directory: "~/.neurolite/cache"
  max_size: "1GB"

analysis:
  max_file_size: "1GB"
  timeout: 300
  confidence_threshold: 0.8

models:
  enable_deep_learning: true
  enable_traditional_ml: true
  max_recommendations: 10
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=neurolite --cov-report=html

# Run specific test categories
pytest tests/test_quality_detector.py
pytest tests/test_data_type_detector.py
```

## 📚 Documentation

- **API Reference**: [https://neurolite.readthedocs.io/](https://neurolite.readthedocs.io/)
- **User Guide**: [docs/user_guide.md](docs/user_guide.md)
- **Examples**: [examples/](examples/)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/dot-css/neurolite
cd neurolite
pip install -e .[dev]
pre-commit install
```

### Running Tests
```bash
pytest tests/
black neurolite/ tests/
flake8 neurolite/ tests/
mypy neurolite/
```

## 📈 Performance

NeuroLite is designed for performance:
- **Fast Analysis**: < 5 seconds for datasets up to 1GB
- **Memory Efficient**: Streaming and lazy loading for large datasets
- **Parallel Processing**: Multi-core support for complex analyses
- **Caching**: Intelligent caching for repeated analyses

## 🛣️ Roadmap

- [ ] **v0.2.0**: Enhanced deep learning model recommendations
- [ ] **v0.3.0**: Real-time data stream analysis
- [ ] **v0.4.0**: AutoML pipeline integration
- [ ] **v0.5.0**: Distributed processing support
- [ ] **v1.0.0**: Production-ready release

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ by the NeuroLite team
- Inspired by the need for accessible AI/ML tools
- Thanks to all contributors and the open-source community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/dot-css/neurolite/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dot-css/neurolite/discussions)
- **Email**: support@neurolite.ai
- **Documentation**: [https://neurolite.readthedocs.io/](https://neurolite.readthedocs.io/)

---

**Made with ❤️ for the AI/ML community**