.PHONY: clean build test check upload-test upload install-dev lint typecheck quality help

# Default target
help:
	@echo "Available targets:"
	@echo "  clean        - Clean build artifacts"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests with coverage"
	@echo "  test-fast    - Run tests without coverage"
	@echo "  lint         - Run code formatting and linting"
	@echo "  typecheck    - Run type checking"
	@echo "  quality      - Run all quality checks"
	@echo "  build        - Build package"
	@echo "  check        - Check built package"
	@echo "  validate     - Run comprehensive package validation"
	@echo "  upload-test  - Upload to Test PyPI"
	@echo "  upload       - Upload to PyPI"
	@echo "  release-test - Full release to Test PyPI"
	@echo "  release      - Full release to PyPI"
	@echo "  release-dry  - Dry run release (no upload)"
	@echo "  release-enhanced - Enhanced release with verbose output"
	@echo "  release-version - Release with version bump (use VERSION=x.y.z)"
	@echo "  release-quick - Quick release for development (Test PyPI only)"
	@echo "  install-test - Install from Test PyPI"
	@echo "  version      - Show current version"
	@echo "  docs         - Build documentation"

# Clean build artifacts (cross-platform)
clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').glob('build') if p.is_dir()]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').glob('dist') if p.is_dir()]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').glob('*.egg-info') if p.is_dir()]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]"
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"

# Install development dependencies
install-dev:
	pip install -e ".[dev]"

# Run tests with coverage
test:
	python -m pytest tests/ -v --cov=neurolite --cov-report=html --cov-report=term-missing

# Run tests without coverage (faster)
test-fast:
	python -m pytest tests/ -v

# Run parallel tests
test-parallel:
	python -m pytest tests/ -v -n auto --cov=neurolite

# Run specific test
test-one:
	python -m pytest tests/$(TEST) -v

# Build package
build: clean
	python -m build

# Check package
check: build
	python -m twine check dist/*

# Run comprehensive package validation
validate: build
	python scripts/validate_package.py

# Upload to Test PyPI
upload-test: check
	python -m twine upload --repository testpypi dist/*

# Upload to PyPI
upload: check
	python -m twine upload dist/*

# Full release to Test PyPI
release-test:
	python scripts/release.py --test-only

# Full release
release:
	python scripts/release.py

# Dry run release
release-dry:
	python scripts/release.py --dry-run

# Enhanced release with comprehensive validation
release-enhanced:
	python scripts/release.py --verbose

# Release with version bump
release-version:
	@echo "Usage: make release-version VERSION=x.y.z"
	@if [ -z "$(VERSION)" ]; then echo "ERROR: VERSION not specified"; exit 1; fi
	python scripts/release.py --version $(VERSION) --verbose

# Quick release (skip some checks for development)
release-quick:
	python scripts/release.py --skip-quality --skip-git-checks --test-only

# Install from Test PyPI
install-test:
	pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ neurolite

# Show current version
version:
	python -c "import neurolite; print(neurolite.__version__)"

# Format and lint code
lint:
	python -m black neurolite/ tests/
	python -m ruff check neurolite/ tests/ --fix
	python -m flake8 neurolite/ tests/

# Type check
typecheck:
	python -m mypy neurolite/

# All quality checks
quality: lint typecheck test

# Build documentation
docs:
	cd docs && python -m sphinx -b html . _build/html

# Serve documentation locally
docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Security check
security:
	python -m pip-audit

# Dependency check
deps-check:
	python -m pip check

# Update dependencies
deps-update:
	pip install --upgrade pip setuptools wheel
	pip install --upgrade -e ".[dev]"