#!/usr/bin/env python3
"""
Release automation script for NeuroLite library.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    return result


def get_current_version():
    """Get the current version from _version.py."""
    version_file = Path("neurolite/_version.py")
    if not version_file.exists():
        raise FileNotFoundError("Version file not found")
    
    with open(version_file, "r") as f:
        content = f.read()
    
    match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Version not found in _version.py")
    
    return match.group(1)


def update_version(new_version):
    """Update the version in _version.py."""
    version_file = Path("neurolite/_version.py")
    
    with open(version_file, "r") as f:
        content = f.read()
    
    # Update version string
    content = re.sub(
        r'__version__ = ["\'][^"\']+["\']',
        f'__version__ = "{new_version}"',
        content
    )
    
    # Update version_info tuple
    version_parts = new_version.split(".")
    version_tuple = ", ".join(version_parts)
    content = re.sub(
        r'__version_info__ = tuple\(map\(int, __version__\.split\("\."\)\)\)',
        f'__version_info__ = ({version_tuple})',
        content
    )
    
    with open(version_file, "w") as f:
        f.write(content)
    
    print(f"Updated version to {new_version}")


def validate_version(version):
    """Validate version format (semantic versioning)."""
    pattern = r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?$'
    if not re.match(pattern, version):
        raise ValueError(f"Invalid version format: {version}")


def clean_build():
    """Clean build artifacts."""
    print("Cleaning build artifacts...")
    run_command("rm -rf build/ dist/ *.egg-info/")


def run_tests():
    """Run the test suite."""
    print("Running tests...")
    run_command("python -m pytest tests/ -v")


def build_package():
    """Build wheel and source distribution."""
    print("Building package...")
    run_command("python -m build")


def check_package():
    """Check the built package."""
    print("Checking package...")
    run_command("python -m twine check dist/*")


def upload_to_test_pypi():
    """Upload to Test PyPI."""
    print("Uploading to Test PyPI...")
    run_command("python -m twine upload --repository testpypi dist/*")


def upload_to_pypi():
    """Upload to PyPI."""
    print("Uploading to PyPI...")
    run_command("python -m twine upload dist/*")


def create_git_tag(version):
    """Create and push git tag."""
    print(f"Creating git tag v{version}...")
    run_command(f"git tag v{version}")
    run_command(f"git push origin v{version}")


def main():
    parser = argparse.ArgumentParser(description="Release automation for NeuroLite")
    parser.add_argument("--version", help="New version to release")
    parser.add_argument("--test-only", action="store_true", 
                       help="Upload to Test PyPI only")
    parser.add_argument("--skip-tests", action="store_true", 
                       help="Skip running tests")
    parser.add_argument("--skip-tag", action="store_true", 
                       help="Skip creating git tag")
    
    args = parser.parse_args()
    
    if args.version:
        validate_version(args.version)
        current_version = get_current_version()
        print(f"Current version: {current_version}")
        print(f"New version: {args.version}")
        
        # Update version
        update_version(args.version)
    else:
        args.version = get_current_version()
        print(f"Using current version: {args.version}")
    
    # Clean previous builds
    clean_build()
    
    # Run tests
    if not args.skip_tests:
        run_tests()
    
    # Build package
    build_package()
    
    # Check package
    check_package()
    
    # Upload
    if args.test_only:
        upload_to_test_pypi()
    else:
        # First upload to test PyPI
        try:
            upload_to_test_pypi()
            print("Successfully uploaded to Test PyPI")
        except:
            print("Warning: Failed to upload to Test PyPI (might already exist)")
        
        # Then upload to PyPI
        response = input("Upload to PyPI? (y/N): ")
        if response.lower() == 'y':
            upload_to_pypi()
            
            # Create git tag
            if not args.skip_tag:
                create_git_tag(args.version)
    
    print(f"Release {args.version} completed successfully!")


if __name__ == "__main__":
    main()