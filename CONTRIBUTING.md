# Contributing to NeuroLite

Thank you for your interest in contributing to NeuroLite! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues
- Use the [GitHub Issues](https://github.com/dot-css/neurolite/issues) page
- Search existing issues before creating a new one
- Provide detailed information including:
  - Python version
  - NeuroLite version
  - Operating system
  - Steps to reproduce
  - Expected vs actual behavior
  - Code samples and error messages

### Suggesting Features
- Open a [GitHub Discussion](https://github.com/dot-css/neurolite/discussions) for feature requests
- Describe the use case and expected behavior
- Consider implementation complexity and maintenance burden

### Code Contributions

#### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/dot-css/neurolite.git
   cd neurolite
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```
5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

#### Making Changes
1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes following our coding standards
3. Add tests for new functionality
4. Run the test suite:
   ```bash
   pytest
   ```
5. Run code quality checks:
   ```bash
   black neurolite/ tests/
   flake8 neurolite/ tests/
   mypy neurolite/
   ```

#### Submitting Changes
1. Commit your changes with clear messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```
2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
3. Create a Pull Request with:
   - Clear title and description
   - Reference to related issues
   - Test coverage information
   - Documentation updates if needed

## 📝 Coding Standards

### Code Style
- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length: 88 characters
- Use type hints for all public functions

### Documentation
- Write docstrings for all public functions and classes
- Use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings
- Update README.md for user-facing changes
- Add examples for new features

### Testing
- Write unit tests for all new functionality
- Aim for >90% test coverage
- Use descriptive test names
- Include edge cases and error conditions
- Use pytest fixtures for common test data

### Example Code Style
```python
def analyze_data_quality(
    df: pd.DataFrame, 
    confidence_threshold: float = 0.8
) -> QualityMetrics:
    """
    Analyze data quality metrics for a DataFrame.
    
    Args:
        df: Input DataFrame to analyze
        confidence_threshold: Minimum confidence for classifications
        
    Returns:
        QualityMetrics object with analysis results
        
    Raises:
        InsufficientDataError: If dataset is too small for analysis
        
    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        >>> metrics = analyze_data_quality(df)
        >>> print(f"Completeness: {metrics.completeness:.2%}")
    """
    if df.empty:
        raise InsufficientDataError(0, 1, "dataset")
    
    # Implementation here...
    return QualityMetrics(...)
```

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── test_detectors/
│   ├── test_quality_detector.py
│   ├── test_data_type_detector.py
│   └── test_file_detector.py
├── test_analyzers/
├── test_recommenders/
└── test_integration/
```

### Test Categories
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test speed and memory usage

### Test Naming Convention
```python
class TestQualityDetector:
    def test_analyze_quality_empty_dataframe(self):
        """Test quality analysis with empty DataFrame."""
        
    def test_analyze_quality_with_missing_data(self):
        """Test quality analysis with missing values."""
        
    def test_analyze_quality_raises_insufficient_data_error(self):
        """Test that insufficient data raises appropriate error."""
```

## 📚 Documentation

### API Documentation
- Use Sphinx for API documentation
- Include examples in docstrings
- Document all parameters and return values
- Add type information

### User Documentation
- Update README.md for user-facing changes
- Add examples to `examples/` directory
- Update user guide in `docs/`

## 🏗️ Architecture Guidelines

### Module Organization
```
neurolite/
├── core/           # Core data models and exceptions
├── detectors/      # Data detection components
├── analyzers/      # Statistical analysis components
├── recommenders/   # Model recommendation components
└── utils/          # Utility functions
```

### Design Principles
- **Modularity**: Components should be loosely coupled
- **Extensibility**: Easy to add new detectors and analyzers
- **Performance**: Optimize for large datasets
- **Reliability**: Graceful error handling and fallbacks

### Error Handling
- Use custom exceptions from `neurolite.core.exceptions`
- Provide helpful error messages
- Include suggestions for fixing issues
- Log warnings for non-critical issues

## 🚀 Release Process

### Version Numbering
- Follow [Semantic Versioning](https://semver.org/)
- Format: `MAJOR.MINOR.PATCH`
- Pre-release: `MAJOR.MINOR.PATCH-alpha.N`

### Release Checklist
- [ ] Update version in `setup.py`
- [ ] Update `CHANGELOG.md`
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release in Git
- [ ] Deploy to PyPI

## 🎯 Areas for Contribution

### High Priority
- Performance optimizations for large datasets
- Additional file format support
- Enhanced model recommendations
- Better error messages and documentation

### Medium Priority
- New statistical analysis methods
- Domain-specific detectors
- Visualization components
- CLI improvements

### Good First Issues
- Documentation improvements
- Test coverage increases
- Code style fixes
- Example notebooks

## 📞 Getting Help

- **Questions**: Use [GitHub Discussions](https://github.com/dot-css/neurolite/discussions)
- **Chat**: Join our [Discord server](https://discord.gg/neurolite)
- **Email**: dev@neurolite.ai

## 🏆 Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Invited to join the core team (for significant contributions)

Thank you for contributing to NeuroLite! 🎉