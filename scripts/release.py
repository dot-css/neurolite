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
    
    # Update version_info tuple - handle semantic versioning properly
    version_parts = new_version.split("-")[0].split("+")[0].split(".")
    version_tuple = ", ".join(version_parts)
    content = re.sub(
        r'__version_info__ = \([^)]+\)',
        f'__version_info__ = ({version_tuple})',
        content
    )
    
    with open(version_file, "w") as f:
        f.write(content)
    
    print(f"Updated version to {new_version}")


def validate_version(version):
    """Validate version format (semantic versioning)."""
    pattern = r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?(?:\+[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?$'
    if not re.match(pattern, version):
        raise ValueError(f"Invalid version format: {version}")


def check_git_status():
    """Check if git working directory is clean."""
    result = run_command("git status --porcelain", check=False)
    if result.stdout.strip():
        print("Warning: Git working directory is not clean")
        print("Uncommitted changes:")
        print(result.stdout)
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)


def check_git_branch():
    """Check if we're on the main/master branch."""
    result = run_command("git branch --show-current", check=False)
    current_branch = result.stdout.strip()
    
    if current_branch not in ['main', 'master']:
        print(f"Warning: Not on main/master branch (current: {current_branch})")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)


def clean_build():
    """Clean build artifacts."""
    print("Cleaning build artifacts...")
    # Use cross-platform commands
    import shutil
    
    dirs_to_remove = ['build', 'dist']
    for dir_name in dirs_to_remove:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name}/")
    
    # Remove egg-info directories
    for egg_info in Path('.').glob('*.egg-info'):
        if egg_info.is_dir():
            shutil.rmtree(egg_info)
            print(f"Removed {egg_info}")


def run_tests():
    """Run the test suite."""
    print("Running tests...")
    run_command("python -m pytest tests/ -v --tb=short")


def run_quality_checks():
    """Run code quality checks."""
    print("Running quality checks...")
    
    # Check if tools are available
    tools = ['black', 'ruff', 'mypy']
    available_tools = []
    
    for tool in tools:
        result = run_command(f"python -m {tool} --version", check=False)
        if result.returncode == 0:
            available_tools.append(tool)
    
    if 'black' in available_tools:
        print("Running black...")
        run_command("python -m black --check neurolite/ tests/")
    
    if 'ruff' in available_tools:
        print("Running ruff...")
        run_command("python -m ruff check neurolite/ tests/")
    
    if 'mypy' in available_tools:
        print("Running mypy...")
        run_command("python -m mypy neurolite/", check=False)  # Don't fail on mypy errors


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


def update_changelog(version):
    """Update CHANGELOG.md with new version."""
    changelog_file = Path("CHANGELOG.md")
    if not changelog_file.exists():
        print("CHANGELOG.md not found, skipping changelog update")
        return
    
    with open(changelog_file, "r") as f:
        content = f.read()
    
    # Add new version entry at the top
    today = subprocess.run(['date', '+%Y-%m-%d'], capture_output=True, text=True).stdout.strip()
    if not today:  # Fallback for Windows
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
    
    new_entry = f"\n## [{version}] - {today}\n\n### Added\n- \n\n### Changed\n- \n\n### Fixed\n- \n\n"
    
    # Insert after the first line (usually the title)
    lines = content.split('\n')
    if len(lines) > 1:
        lines.insert(2, new_entry)
        new_content = '\n'.join(lines)
        
        with open(changelog_file, "w") as f:
            f.write(new_content)
        
        print(f"Updated CHANGELOG.md with version {version}")


def main():
    parser = argparse.ArgumentParser(description="Release automation for NeuroLite")
    parser.add_argument("--version", help="New version to release")
    parser.add_argument("--test-only", action="store_true", 
                       help="Upload to Test PyPI only")
    parser.add_argument("--skip-tests", action="store_true", 
                       help="Skip running tests")
    parser.add_argument("--skip-quality", action="store_true", 
                       help="Skip quality checks")
    parser.add_argument("--skip-tag", action="store_true", 
                       help="Skip creating git tag")
    parser.add_argument("--skip-git-checks", action="store_true", 
                       help="Skip git status checks")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Perform a dry run without uploading")
    
    args = parser.parse_args()
    
    # Git checks
    if not args.skip_git_checks:
        check_git_status()
        check_git_branch()
    
    if args.version:
        validate_version(args.version)
        current_version = get_current_version()
        print(f"Current version: {current_version}")
        print(f"New version: {args.version}")
        
        # Update version
        update_version(args.version)
        update_changelog(args.version)
    else:
        args.version = get_current_version()
        print(f"Using current version: {args.version}")
    
    # Clean previous builds
    clean_build()
    
    # Run quality checks
    if not args.skip_quality:
        run_quality_checks()
    
    # Run tests
    if not args.skip_tests:
        run_tests()
    
    # Build package
    build_package()
    
    # Check package
    check_package()
    
    if args.dry_run:
        print("Dry run completed successfully!")
        return
    
    # Upload
    if args.test_only:
        upload_to_test_pypi()
    else:
        # First upload to test PyPI
        try:
            upload_to_test_pypi()
            print("Successfully uploaded to Test PyPI")
        except SystemExit:
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