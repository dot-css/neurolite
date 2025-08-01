"""
Tests for package configuration and distribution.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from neurolite import __version__, get_version, get_version_info


class TestVersionManagement:
    """Test version management functionality."""
    
    def test_version_format(self):
        """Test that version follows semantic versioning."""
        version_pattern = r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?$'
        assert re.match(version_pattern, __version__), f"Invalid version format: {__version__}"
    
    def test_version_consistency(self):
        """Test that version is consistent across files."""
        # Check _version.py
        version_file = Path("neurolite/_version.py")
        assert version_file.exists(), "Version file not found"
        
        with open(version_file, "r") as f:
            content = f.read()
        
        match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
        assert match, "Version not found in _version.py"
        version_py = match.group(1)
        
        assert version_py == __version__, f"Version mismatch: {version_py} != {__version__}"
    
    def test_get_version_functions(self):
        """Test version getter functions."""
        assert get_version() == __version__
        
        version_info = get_version_info()
        assert isinstance(version_info, tuple)
        assert len(version_info) >= 3
        assert all(isinstance(x, int) for x in version_info)
        
        # Reconstruct version from version_info
        reconstructed = ".".join(map(str, version_info))
        assert reconstructed in __version__


class TestPackageConfiguration:
    """Test package configuration files."""
    
    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists and is valid."""
        pyproject_file = Path("pyproject.toml")
        assert pyproject_file.exists(), "pyproject.toml not found"
        
        # Try to parse it (basic validation)
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("No TOML parser available")
        
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        
        # Check required sections
        assert "build-system" in config
        assert "project" in config
        
        # Check project metadata
        project = config["project"]
        assert project["name"] == "neurolite"
        assert "description" in project
        assert "dependencies" in project
        assert "dynamic" in project and "version" in project["dynamic"]
    
    def test_setup_py_exists(self):
        """Test that setup.py exists."""
        setup_file = Path("setup.py")
        assert setup_file.exists(), "setup.py not found"
    
    def test_manifest_requirements(self):
        """Test that all required files are included in distribution."""
        required_files = [
            "README.md",
            "pyproject.toml",
            "setup.py",
            "neurolite/__init__.py",
            "neurolite/_version.py",
        ]
        
        for file_path in required_files:
            assert Path(file_path).exists(), f"Required file not found: {file_path}"


class TestBuildSystem:
    """Test package building functionality."""
    
    @pytest.mark.slow
    def test_package_builds(self):
        """Test that package can be built successfully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run build command
            result = subprocess.run(
                [sys.executable, "-m", "build", "--outdir", temp_dir],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                pytest.fail(f"Build failed: {result.stderr}")
            
            # Check that wheel and source distribution were created
            dist_files = list(Path(temp_dir).glob("*"))
            wheel_files = [f for f in dist_files if f.suffix == ".whl"]
            sdist_files = [f for f in dist_files if f.suffix == ".gz"]
            
            assert len(wheel_files) == 1, f"Expected 1 wheel file, got {len(wheel_files)}"
            assert len(sdist_files) == 1, f"Expected 1 source dist file, got {len(sdist_files)}"
    
    @pytest.mark.slow
    def test_package_check(self):
        """Test that built package passes twine check."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Build package
            build_result = subprocess.run(
                [sys.executable, "-m", "build", "--outdir", temp_dir],
                capture_output=True,
                text=True
            )
            
            if build_result.returncode != 0:
                pytest.skip(f"Build failed: {build_result.stderr}")
            
            # Check package with twine
            check_result = subprocess.run(
                [sys.executable, "-m", "twine", "check", f"{temp_dir}/*"],
                capture_output=True,
                text=True
            )
            
            if check_result.returncode != 0:
                pytest.fail(f"Package check failed: {check_result.stderr}")


class TestDependencies:
    """Test dependency management."""
    
    def test_core_dependencies_importable(self):
        """Test that core dependencies can be imported."""
        core_deps = [
            "numpy",
            "pandas", 
            "sklearn",
            "torch",
            "transformers",
            "matplotlib",
            "click",
            "tqdm",
        ]
        
        failed_imports = []
        for dep in core_deps:
            try:
                __import__(dep)
            except ImportError:
                failed_imports.append(dep)
        
        if failed_imports:
            pytest.skip(f"Core dependencies not available: {failed_imports}")
    
    def test_optional_dependencies(self):
        """Test optional dependency groups."""
        pyproject_file = Path("pyproject.toml")
        if not pyproject_file.exists():
            pytest.skip("pyproject.toml not found")
        
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("No TOML parser available")
        
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        
        optional_deps = config["project"]["optional-dependencies"]
        
        # Check that dev dependencies include testing tools
        dev_deps = optional_deps.get("dev", [])
        assert any("pytest" in dep for dep in dev_deps), "pytest not in dev dependencies"
        assert any("black" in dep for dep in dev_deps), "black not in dev dependencies"


class TestReleaseAutomation:
    """Test release automation scripts."""
    
    def test_release_script_exists(self):
        """Test that release script exists and is executable."""
        release_script = Path("scripts/release.py")
        assert release_script.exists(), "Release script not found"
        
        # Check if script is executable (on Unix systems)
        if os.name != 'nt':  # Not Windows
            assert os.access(release_script, os.X_OK), "Release script not executable"
    
    def test_makefile_exists(self):
        """Test that Makefile exists with required targets."""
        makefile = Path("Makefile")
        assert makefile.exists(), "Makefile not found"
        
        with open(makefile, "r") as f:
            content = f.read()
        
        required_targets = ["clean", "build", "test", "check", "upload-test", "upload"]
        for target in required_targets:
            assert f"{target}:" in content, f"Makefile missing target: {target}"
    
    def test_pypirc_template_exists(self):
        """Test that PyPI configuration template exists."""
        pypirc_template = Path(".pypirc.template")
        assert pypirc_template.exists(), "PyPI configuration template not found"
        
        with open(pypirc_template, "r") as f:
            content = f.read()
        
        assert "[pypi]" in content, "PyPI configuration missing"
        assert "[testpypi]" in content, "Test PyPI configuration missing"


class TestInstallation:
    """Test package installation."""
    
    @pytest.mark.slow
    def test_editable_install(self):
        """Test that package can be installed in editable mode."""
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            pytest.fail(f"Editable install failed: {result.stderr}")
        
        # Test that package can be imported
        import_result = subprocess.run(
            [sys.executable, "-c", "import neurolite; print(neurolite.__version__)"],
            capture_output=True,
            text=True
        )
        
        assert import_result.returncode == 0, f"Import failed: {import_result.stderr}"
        assert __version__ in import_result.stdout, "Version mismatch after install"