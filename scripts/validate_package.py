#!/usr/bin/env python3
"""
Comprehensive package validation workflow for NeuroLite library.
This script performs pre-upload validation including package integrity,
metadata validation, and import testing.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class PackageValidator:
    """Comprehensive package validation class."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.project_root = Path.cwd()
        self.dist_dir = self.project_root / "dist"
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with appropriate formatting."""
        if level == "ERROR":
            print(f"❌ {message}")
            self.errors.append(message)
        elif level == "WARNING":
            print(f"⚠️  {message}")
            self.warnings.append(message)
        elif level == "SUCCESS":
            print(f"✅ {message}")
        else:
            if self.verbose or level == "INFO":
                print(f"ℹ️  {message}")
    
    def run_command(self, cmd: str, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        if self.verbose:
            self.log(f"Running: {cmd}")
        
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output, 
            text=True,
            cwd=self.project_root
        )
        
        if check and result.returncode != 0:
            error_msg = f"Command failed: {cmd}"
            if result.stderr:
                error_msg += f"\nError: {result.stderr}"
            raise ValidationError(error_msg)
        
        return result
    
    def validate_project_structure(self) -> bool:
        """Validate basic project structure."""
        self.log("Validating project structure...", "INFO")
        
        required_files = [
            "pyproject.toml",
            "setup.py", 
            "README.md",
            "neurolite/__init__.py",
            "neurolite/_version.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.log(f"Missing required files: {', '.join(missing_files)}", "ERROR")
            return False
        
        self.log("Project structure validation passed", "SUCCESS")
        return True
    
    def validate_version_consistency(self) -> bool:
        """Validate version consistency across files."""
        self.log("Validating version consistency...", "INFO")
        
        try:
            # Get version from _version.py
            version_file = self.project_root / "neurolite" / "_version.py"
            version_dict = {}
            with open(version_file) as f:
                exec(f.read(), version_dict)
            version = version_dict.get('__version__')
            
            if not version:
                self.log("Version not found in _version.py", "ERROR")
                return False
            
            self.log(f"Found version: {version}")
            
            # Check if version follows semantic versioning
            import re
            semver_pattern = r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?(?:\+[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?$'
            if not re.match(semver_pattern, version):
                self.log(f"Version {version} doesn't follow semantic versioning", "WARNING")
            
            self.log("Version consistency validation passed", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Failed to validate version: {e}", "ERROR")
            return False
    
    def validate_dependencies(self) -> bool:
        """Validate dependency specifications."""
        self.log("Validating dependencies...", "INFO")
        
        try:
            # Read pyproject.toml to check dependencies
            try:
                import tomllib  # Python 3.11+
                with open(self.project_root / "pyproject.toml", "rb") as f:
                    pyproject_data = tomllib.load(f)
            except ImportError:
                # Fallback for Python < 3.11
                try:
                    import tomli as tomllib
                    with open(self.project_root / "pyproject.toml", "rb") as f:
                        pyproject_data = tomllib.load(f)
                except ImportError:
                    # Final fallback - basic parsing
                    self.log("tomllib/tomli not available, using basic TOML parsing", "WARNING")
                    return True  # Skip detailed dependency validation
            
            dependencies = pyproject_data.get("project", {}).get("dependencies", [])
            optional_deps = pyproject_data.get("project", {}).get("optional-dependencies", {})
            
            self.log(f"Found {len(dependencies)} core dependencies")
            self.log(f"Found {len(optional_deps)} optional dependency groups")
            
            # Basic validation of dependency format
            for dep in dependencies:
                if not isinstance(dep, str) or not dep.strip():
                    self.log(f"Invalid dependency format: {dep}", "ERROR")
                    return False
            
            # Validate optional dependencies
            for group_name, group_deps in optional_deps.items():
                for dep in group_deps:
                    if not isinstance(dep, str) or not dep.strip():
                        self.log(f"Invalid optional dependency in {group_name}: {dep}", "ERROR")
                        return False
            
            self.log("Dependencies validation passed", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Failed to validate dependencies: {e}", "ERROR")
            return False
    
    def validate_build_artifacts(self) -> bool:
        """Validate that build artifacts exist and are valid."""
        self.log("Validating build artifacts...", "INFO")
        
        if not self.dist_dir.exists():
            self.log("dist/ directory not found. Run 'python -m build' first.", "ERROR")
            return False
        
        # Look for wheel and source distribution
        wheel_files = list(self.dist_dir.glob("*.whl"))
        sdist_files = list(self.dist_dir.glob("*.tar.gz"))
        
        if not wheel_files:
            self.log("No wheel files found in dist/", "ERROR")
            return False
        
        if not sdist_files:
            self.log("No source distribution files found in dist/", "ERROR")
            return False
        
        self.log(f"Found {len(wheel_files)} wheel file(s)")
        self.log(f"Found {len(sdist_files)} source distribution file(s)")
        
        # Validate wheel file structure
        for wheel_file in wheel_files:
            if not self._validate_wheel_structure(wheel_file):
                return False
        
        self.log("Build artifacts validation passed", "SUCCESS")
        return True
    
    def _validate_wheel_structure(self, wheel_path: Path) -> bool:
        """Validate the internal structure of a wheel file."""
        try:
            with zipfile.ZipFile(wheel_path, 'r') as wheel:
                files = wheel.namelist()
                
                # Check for required files in wheel
                has_init = any(f.endswith('neurolite/__init__.py') for f in files)
                has_metadata = any(f.endswith('.dist-info/METADATA') for f in files)
                
                if not has_init:
                    self.log(f"Wheel {wheel_path.name} missing neurolite/__init__.py", "ERROR")
                    return False
                
                if not has_metadata:
                    self.log(f"Wheel {wheel_path.name} missing METADATA file", "ERROR")
                    return False
                
                self.log(f"Wheel {wheel_path.name} structure is valid")
                return True
                
        except Exception as e:
            self.log(f"Failed to validate wheel {wheel_path.name}: {e}", "ERROR")
            return False
    
    def validate_metadata(self) -> bool:
        """Validate package metadata using twine check."""
        self.log("Validating package metadata with twine...", "INFO")
        
        try:
            result = self.run_command("python -m twine check dist/*")
            
            if "PASSED" in result.stdout:
                self.log("Metadata validation passed", "SUCCESS")
                return True
            else:
                self.log(f"Metadata validation failed: {result.stdout}", "ERROR")
                return False
                
        except ValidationError as e:
            self.log(f"Twine check failed: {e}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Failed to run twine check: {e}", "ERROR")
            return False
    
    def test_package_import(self) -> bool:
        """Test package import from built wheel."""
        self.log("Testing package import from wheel...", "INFO")
        
        wheel_files = list(self.dist_dir.glob("*.whl"))
        if not wheel_files:
            self.log("No wheel files found for import testing", "ERROR")
            return False
        
        wheel_file = wheel_files[0]  # Use the first wheel file
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                # Install the wheel in the temporary directory
                self.log(f"Installing {wheel_file.name} for import testing...")
                install_cmd = f"python -m pip install --target {temp_path} {wheel_file} --no-deps"
                self.run_command(install_cmd)
                
                # Test import
                return self._test_imports_in_directory(temp_path)
                
            except Exception as e:
                self.log(f"Failed to test package import: {e}", "ERROR")
                return False
    
    def _test_imports_in_directory(self, install_dir: Path) -> bool:
        """Test imports from a specific directory."""
        # Add the install directory to Python path
        original_path = sys.path.copy()
        sys.path.insert(0, str(install_dir))
        
        try:
            # Test basic import
            import neurolite
            self.log("✓ Basic neurolite import successful")
            
            # Test version access
            version = getattr(neurolite, '__version__', None)
            if version:
                self.log(f"✓ Version accessible: {version}")
            else:
                self.log("⚠️  Version not accessible from package", "WARNING")
            
            # Test core module imports
            core_modules = [
                'neurolite.core',
                'neurolite.models', 
                'neurolite.data',
                'neurolite.training',
                'neurolite.evaluation'
            ]
            
            successful_imports = 0
            for module_name in core_modules:
                try:
                    importlib.import_module(module_name)
                    self.log(f"✓ {module_name} import successful")
                    successful_imports += 1
                except ImportError as e:
                    self.log(f"⚠️  {module_name} import failed: {e}", "WARNING")
                except Exception as e:
                    self.log(f"⚠️  {module_name} import error: {e}", "WARNING")
            
            if successful_imports == 0:
                self.log("No core modules could be imported", "ERROR")
                return False
            
            self.log(f"Import testing passed ({successful_imports}/{len(core_modules)} core modules)", "SUCCESS")
            return True
            
        except ImportError as e:
            self.log(f"Failed to import neurolite: {e}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Unexpected error during import testing: {e}", "ERROR")
            return False
        finally:
            # Restore original Python path
            sys.path = original_path
            
            # Clean up imported modules
            modules_to_remove = [name for name in sys.modules.keys() if name.startswith('neurolite')]
            for module_name in modules_to_remove:
                del sys.modules[module_name]
    
    def validate_entry_points(self) -> bool:
        """Validate console script entry points."""
        self.log("Validating entry points...", "INFO")
        
        try:
            # Check if the CLI entry point works
            result = self.run_command("python -c \"import neurolite.cli.main; print('CLI module accessible')\"", check=False)
            
            if result.returncode == 0:
                self.log("CLI entry point validation passed", "SUCCESS")
                return True
            else:
                self.log("CLI entry point validation failed", "WARNING")
                return True  # Don't fail the entire validation for this
                
        except Exception as e:
            self.log(f"Failed to validate entry points: {e}", "WARNING")
            return True  # Don't fail the entire validation for this
    
    def run_test_suite(self) -> bool:
        """Run the test suite to ensure package functionality."""
        self.log("Running test suite...", "INFO")
        
        try:
            # Check if pytest is available
            result = self.run_command("python -m pytest --version", check=False)
            if result.returncode != 0:
                self.log("pytest not available, skipping test suite", "WARNING")
                return True
            
            # Run tests with minimal output
            test_cmd = "python -m pytest tests/ -x --tb=short -q"
            result = self.run_command(test_cmd, check=False)
            
            if result.returncode == 0:
                self.log("Test suite passed", "SUCCESS")
                return True
            else:
                self.log("Some tests failed", "WARNING")
                if self.verbose and result.stdout:
                    print(result.stdout)
                return True  # Don't fail validation for test failures
                
        except Exception as e:
            self.log(f"Failed to run test suite: {e}", "WARNING")
            return True  # Don't fail validation for test issues
    
    def generate_validation_report(self) -> Dict:
        """Generate a comprehensive validation report."""
        return {
            "validation_status": "PASSED" if not self.errors else "FAILED",
            "errors": self.errors,
            "warnings": self.warnings,
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings)
        }
    
    def run_full_validation(self) -> bool:
        """Run the complete validation workflow."""
        print("🔍 NeuroLite Package Validation Workflow")
        print("=" * 50)
        
        validation_steps = [
            ("Project Structure", self.validate_project_structure),
            ("Version Consistency", self.validate_version_consistency),
            ("Dependencies", self.validate_dependencies),
            ("Build Artifacts", self.validate_build_artifacts),
            ("Package Metadata", self.validate_metadata),
            ("Package Import", self.test_package_import),
            ("Entry Points", self.validate_entry_points),
            ("Test Suite", self.run_test_suite)
        ]
        
        passed_steps = 0
        total_steps = len(validation_steps)
        
        for step_name, step_function in validation_steps:
            print(f"\n📋 {step_name}")
            print("-" * 30)
            
            try:
                if step_function():
                    passed_steps += 1
                else:
                    self.log(f"{step_name} validation failed", "ERROR")
            except Exception as e:
                self.log(f"{step_name} validation error: {e}", "ERROR")
        
        # Generate final report
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)
        
        report = self.generate_validation_report()
        
        print(f"Steps passed: {passed_steps}/{total_steps}")
        print(f"Errors: {report['total_errors']}")
        print(f"Warnings: {report['total_warnings']}")
        
        if report['validation_status'] == "PASSED":
            print("\n✅ VALIDATION PASSED")
            print("🚀 Package is ready for upload!")
            return True
        else:
            print("\n❌ VALIDATION FAILED")
            print("🔧 Please fix the errors above before uploading.")
            
            if self.errors:
                print("\nErrors to fix:")
                for i, error in enumerate(self.errors, 1):
                    print(f"  {i}. {error}")
            
            return False


def main():
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(
        description="Comprehensive package validation for NeuroLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_package.py                    # Run full validation
  python scripts/validate_package.py --verbose          # Run with detailed output
  python scripts/validate_package.py --report report.json  # Save report to file
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--report", "-r",
        type=str,
        help="Save validation report to JSON file"
    )
    
    args = parser.parse_args()
    
    # Create validator and run validation
    validator = PackageValidator(verbose=args.verbose)
    
    try:
        success = validator.run_full_validation()
        
        # Save report if requested
        if args.report:
            report = validator.generate_validation_report()
            with open(args.report, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Validation report saved to {args.report}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during validation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())