.PHONY: clean build test check upload-test upload install-dev

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Install development dependencies
install-dev:
	pip install -e ".[dev]"

# Run tests
test:
	python -m pytest tests/ -v --cov=neurolite

# Build package
build: clean
	python -m build

# Check package
check: build
	python -m twine check dist/*

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

# Install from Test PyPI
install-test:
	pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ neurolite

# Show current version
version:
	python -c "import neurolite; print(neurolite.__version__)"

# Lint code
lint:
	black neurolite/ tests/
	flake8 neurolite/ tests/

# Type check
typecheck:
	mypy neurolite/

# All quality checks
quality: lint typecheck test