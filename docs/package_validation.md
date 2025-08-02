# Package Validation Workflow

The NeuroLite package includes a comprehensive validation workflow to ensure package quality and integrity before distribution to PyPI.

## Overview

The validation workflow performs the following checks:

1. **Project Structure** - Validates required files and directories
2. **Version Consistency** - Ensures version information is consistent across files
3. **Dependencies** - Validates dependency specifications in pyproject.toml
4. **Build Artifacts** - Checks that wheel and source distributions are properly built
5. **Package Metadata** - Runs `twine check` to validate package metadata
6. **Package Import** - Tests that the package can be imported from the built wheel
7. **Entry Points** - Validates console script entry points
8. **Test Suite** - Runs the test suite to ensure functionality

## Usage

### Manual Validation

Run the validation script directly:

```bash
python scripts/validate_package.py
```

### With Verbose Output

```bash
python scripts/validate_package.py --verbose
```

### Generate Report

```bash
python scripts/validate_package.py --report validation_report.json
```

### Using Makefile (Unix/Linux/macOS)

```bash
make validate
```

### Integration with Release Process

The validation is automatically integrated into the release script:

```bash
python scripts/release.py --dry-run
```

## Validation Steps

### 1. Project Structure Validation

Checks for the presence of required files:
- `pyproject.toml`
- `setup.py`
- `README.md`
- `neurolite/__init__.py`
- `neurolite/_version.py`

### 2. Version Consistency

- Reads version from `neurolite/_version.py`
- Validates semantic versioning format
- Ensures version consistency across configuration files

### 3. Dependencies Validation

- Parses `pyproject.toml` for dependency specifications
- Validates core dependencies format
- Checks optional dependency groups

### 4. Build Artifacts Validation

- Verifies presence of wheel (`.whl`) and source distribution (`.tar.gz`) files
- Validates internal structure of wheel files
- Checks for required metadata files

### 5. Package Metadata Validation

- Runs `twine check` on built distributions
- Validates package metadata completeness
- Ensures PyPI compatibility

### 6. Package Import Testing

- Installs the wheel in a temporary environment
- Tests basic package import
- Validates core module accessibility
- Checks version information availability

### 7. Entry Points Validation

- Tests console script entry points
- Validates CLI module accessibility

### 8. Test Suite Execution

- Runs the full test suite using pytest
- Reports test results (warnings only, doesn't fail validation)

## Output Format

The validation script provides clear, color-coded output:

- ✅ **Success**: Step passed validation
- ⚠️ **Warning**: Non-critical issues found
- ❌ **Error**: Critical issues that prevent package upload

## Exit Codes

- `0`: Validation passed successfully
- `1`: Validation failed with errors

## Integration Points

### Release Script Integration

The validation is automatically run as part of the release process:

```python
# In scripts/release.py
def run_package_validation():
    """Run comprehensive package validation."""
    print("Running comprehensive package validation...")
    run_command("python scripts/validate_package.py")
```

### Makefile Integration

```makefile
# Run comprehensive package validation
validate: build
	python scripts/validate_package.py
```

## Troubleshooting

### Common Issues

1. **Missing Build Artifacts**
   - Run `python -m build` to create distributions
   - Ensure `dist/` directory contains `.whl` and `.tar.gz` files

2. **Import Errors**
   - Check that all required dependencies are properly specified
   - Verify package structure and `__init__.py` files

3. **Metadata Validation Failures**
   - Review `pyproject.toml` and `setup.py` for completeness
   - Ensure all required metadata fields are present

4. **Version Inconsistencies**
   - Update version in `neurolite/_version.py`
   - Ensure version follows semantic versioning (e.g., `1.0.0`)

### Skipping Validation Steps

For development purposes, you can modify the validation script to skip certain steps by commenting out the corresponding validation functions in the `run_full_validation()` method.

## Best Practices

1. **Run validation before every release**
2. **Address all errors before uploading to PyPI**
3. **Review warnings and fix when possible**
4. **Keep build artifacts clean** - run `make clean` between builds
5. **Test in isolated environments** when possible

## Continuous Integration

The validation workflow can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Validate Package
  run: |
    python -m build
    python scripts/validate_package.py
```

This ensures that every commit and pull request maintains package quality standards.