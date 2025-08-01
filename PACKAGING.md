# NeuroLite Packaging and Distribution Guide

This document describes the packaging and distribution setup for the NeuroLite library.

## Package Structure

The package follows modern Python packaging standards using `pyproject.toml` as the primary configuration file:

- `pyproject.toml` - Main package configuration
- `setup.py` - Minimal setup script for backward compatibility
- `neurolite/_version.py` - Centralized version management
- `scripts/release.py` - Automated release script
- `Makefile` - Common packaging tasks
- `.pypirc.template` - PyPI configuration template

## Version Management

Version information is centralized in `neurolite/_version.py` and dynamically loaded into `pyproject.toml`. This ensures consistency across all package metadata.

### Updating Version

```bash
# Update version in neurolite/_version.py
# Then run release script
python scripts/release.py --version 0.2.0
```

## Building the Package

### Prerequisites

Install build dependencies:

```bash
pip install -e ".[dev]"
```

### Build Commands

```bash
# Clean previous builds
make clean

# Build wheel and source distribution
make build

# Check package integrity
make check
```

## Testing

### Package Tests

Run packaging-specific tests:

```bash
pytest tests/test_packaging.py -v
```

### Full Test Suite

```bash
make test
```

## Distribution

### Test PyPI

Upload to Test PyPI for validation:

```bash
# Using make
make upload-test

# Using release script
python scripts/release.py --test-only
```

Install from Test PyPI:

```bash
make install-test
```

### PyPI Release

Full release process:

```bash
# Interactive release
python scripts/release.py

# Or using make
make release
```

## Automated Release Process

The release script (`scripts/release.py`) automates the entire release process:

1. Version validation and update
2. Clean build artifacts
3. Run test suite
4. Build package
5. Check package integrity
6. Upload to Test PyPI
7. Upload to PyPI (with confirmation)
8. Create and push git tag

### Release Script Options

```bash
# Release with new version
python scripts/release.py --version 1.0.0

# Test PyPI only
python scripts/release.py --test-only

# Skip tests (not recommended)
python scripts/release.py --skip-tests

# Skip git tagging
python scripts/release.py --skip-tag
```

## Continuous Integration

GitHub Actions workflow (`.github/workflows/test-and-release.yml`) provides:

- Multi-platform testing (Ubuntu, Windows, macOS)
- Multi-version Python testing (3.8-3.11)
- Code quality checks (linting, formatting, type checking)
- Automated Test PyPI uploads on main branch
- Automated PyPI uploads on GitHub releases

### Setting up CI/CD

1. Add secrets to GitHub repository:
   - `PYPI_API_TOKEN` - PyPI API token
   - `TEST_PYPI_API_TOKEN` - Test PyPI API token

2. Create tokens at:
   - PyPI: https://pypi.org/manage/account/token/
   - Test PyPI: https://test.pypi.org/manage/account/token/

## PyPI Configuration

### Setup .pypirc

Copy the template and add your tokens:

```bash
cp .pypirc.template ~/.pypirc
# Edit ~/.pypirc with your tokens
```

### Token-based Authentication

Use API tokens instead of username/password:

```ini
[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-api-token-here
```

## Quality Assurance

### Pre-release Checklist

- [ ] All tests pass
- [ ] Version number updated
- [ ] CHANGELOG updated
- [ ] Documentation updated
- [ ] Package builds successfully
- [ ] Package passes twine check
- [ ] Test PyPI upload successful
- [ ] Test installation from Test PyPI

### Package Validation

The packaging tests (`tests/test_packaging.py`) validate:

- Version consistency across files
- Package configuration validity
- Build system functionality
- Dependency management
- Installation process

## Troubleshooting

### Common Issues

1. **Build fails**: Check dependencies and Python version compatibility
2. **Upload fails**: Verify PyPI tokens and network connectivity
3. **Import fails after install**: Check package structure and __init__.py files
4. **Version conflicts**: Ensure version is updated in _version.py

### Debug Commands

```bash
# Check current version
make version

# Validate package metadata
python -m build --check

# Test import after build
python -c "import neurolite; print(neurolite.__version__)"

# Check installed packages
pip list | grep neurolite
```

## Dependencies

### Core Dependencies

Managed in `pyproject.toml` under `dependencies`:
- Runtime dependencies required for basic functionality

### Optional Dependencies

Organized by feature groups:
- `dev` - Development and testing tools
- `tensorflow` - TensorFlow integration
- `xgboost` - XGBoost integration  
- `all` - All optional dependencies

### Dependency Updates

Regularly update dependencies for security and compatibility:

```bash
# Check outdated packages
pip list --outdated

# Update specific dependency in pyproject.toml
# Then test compatibility
make test
```