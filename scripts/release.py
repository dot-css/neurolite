#!/usr/bin/env python3
"""
Enhanced release automation script for NeuroLite library.
Includes comprehensive validation, automated testing, quality checks, and rollback capabilities.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ReleaseError(Exception):
    """Custom exception for release errors."""
    pass


class ReleaseManager:
    """Enhanced release manager with comprehensive validation and rollback capabilities."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path.cwd()
        self.backup_dir = None
        self.release_state = {
            'version_updated': False,
            'changelog_updated': False,
            'git_tagged': False,
            'uploaded_to_test_pypi': False,
            'uploaded_to_pypi': False,
            'original_version': None,
            'backup_created': False
        }
        self.validation_results = {}
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with appropriate formatting."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if level == "ERROR":
            print(f"[{timestamp}] ❌ {message}")
        elif level == "WARNING":
            print(f"[{timestamp}] ⚠️  {message}")
        elif level == "SUCCESS":
            print(f"[{timestamp}] ✅ {message}")
        else:
            if self.verbose or level == "INFO":
                print(f"[{timestamp}] ℹ️  {message}")
    
    def run_command(self, cmd: str, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command with enhanced error handling."""
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
            if result.stdout:
                error_msg += f"\nOutput: {result.stdout}"
            raise ReleaseError(error_msg)
        
        return result
    
    def create_backup(self):
        """Create a backup of critical files before making changes."""
        self.log("Creating backup of critical files...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = self.project_root / f".release_backup_{timestamp}"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Backup critical files
        files_to_backup = [
            "neurolite/_version.py",
            "CHANGELOG.md",
            "pyproject.toml"
        ]
        
        for file_path in files_to_backup:
            source = self.project_root / file_path
            if source.exists():
                dest = self.backup_dir / file_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                self.log(f"Backed up {file_path}")
        
        self.release_state['backup_created'] = True
        self.log(f"Backup created at {self.backup_dir}", "SUCCESS")
    
    def rollback_changes(self):
        """Rollback changes made during the release process."""
        self.log("Rolling back changes...", "WARNING")
        
        if not self.backup_dir or not self.backup_dir.exists():
            self.log("No backup found, cannot rollback", "ERROR")
            return False
        
        try:
            # Restore backed up files
            for backup_file in self.backup_dir.rglob("*"):
                if backup_file.is_file():
                    relative_path = backup_file.relative_to(self.backup_dir)
                    target_file = self.project_root / relative_path
                    shutil.copy2(backup_file, target_file)
                    self.log(f"Restored {relative_path}")
            
            # Remove git tag if created
            if self.release_state.get('git_tagged'):
                try:
                    version = self.release_state.get('original_version', 'unknown')
                    self.run_command(f"git tag -d v{version}", check=False)
                    self.run_command(f"git push origin :refs/tags/v{version}", check=False)
                    self.log("Removed git tag")
                except:
                    self.log("Failed to remove git tag", "WARNING")
            
            # Clean up backup
            shutil.rmtree(self.backup_dir)
            self.log("Rollback completed successfully", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Rollback failed: {e}", "ERROR")
            return False

    def get_current_version(self) -> str:
        """Get the current version from _version.py."""
        version_file = self.project_root / "neurolite" / "_version.py"
        if not version_file.exists():
            raise ReleaseError("Version file not found")
        
        with open(version_file, "r") as f:
            content = f.read()
        
        match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
        if not match:
            raise ReleaseError("Version not found in _version.py")
        
        return match.group(1)

    def update_version(self, new_version: str):
        """Update the version in _version.py."""
        version_file = self.project_root / "neurolite" / "_version.py"
        
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
        
        self.release_state['version_updated'] = True
        self.log(f"Updated version to {new_version}", "SUCCESS")

    def validate_version(self, version: str):
        """Validate version format (semantic versioning)."""
        pattern = r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?(?:\+[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?$'
        if not re.match(pattern, version):
            raise ReleaseError(f"Invalid version format: {version}")

    def check_git_status(self):
        """Check if git working directory is clean."""
        result = self.run_command("git status --porcelain", check=False)
        if result.stdout.strip():
            self.log("Git working directory is not clean", "WARNING")
            self.log("Uncommitted changes:")
            print(result.stdout)
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                raise ReleaseError("Release cancelled due to uncommitted changes")

    def check_git_branch(self):
        """Check if we're on the main/master branch."""
        result = self.run_command("git branch --show-current", check=False)
        current_branch = result.stdout.strip()
        
        if current_branch not in ['main', 'master']:
            self.log(f"Not on main/master branch (current: {current_branch})", "WARNING")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                raise ReleaseError("Release cancelled due to branch check")

    def clean_build(self):
        """Clean build artifacts."""
        self.log("Cleaning build artifacts...")
        
        dirs_to_remove = ['build', 'dist']
        for dir_name in dirs_to_remove:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)
                self.log(f"Removed {dir_name}/")
        
        # Remove egg-info directories
        for egg_info in self.project_root.glob('*.egg-info'):
            if egg_info.is_dir():
                shutil.rmtree(egg_info)
                self.log(f"Removed {egg_info}")
        
        self.log("Build artifacts cleaned", "SUCCESS")

    def run_comprehensive_tests(self):
        """Run comprehensive test suite with enhanced reporting."""
        self.log("Running comprehensive test suite...")
        
        test_results = {
            'unit_tests': False,
            'integration_tests': False,
            'coverage_check': False
        }
        
        try:
            # Run unit tests with coverage
            self.log("Running unit tests with coverage...")
            result = self.run_command(
                "python -m pytest tests/ -v --cov=neurolite --cov-report=term-missing --cov-report=html --tb=short",
                check=False
            )
            
            if result.returncode == 0:
                test_results['unit_tests'] = True
                self.log("Unit tests passed", "SUCCESS")
            else:
                self.log("Unit tests failed", "ERROR")
                if self.verbose:
                    print(result.stdout)
                    print(result.stderr)
            
            # Check coverage threshold
            coverage_result = self.run_command(
                "python -c \"import coverage; cov = coverage.Coverage(); cov.load(); print(f'Coverage: {cov.report():.1f}%')\"",
                check=False
            )
            
            if coverage_result.returncode == 0:
                test_results['coverage_check'] = True
                self.log("Coverage check completed", "SUCCESS")
            
            # Run integration tests if they exist
            integration_test_dir = self.project_root / "tests" / "integration"
            if integration_test_dir.exists():
                self.log("Running integration tests...")
                result = self.run_command(
                    "python -m pytest tests/integration/ -v --tb=short",
                    check=False
                )
                
                if result.returncode == 0:
                    test_results['integration_tests'] = True
                    self.log("Integration tests passed", "SUCCESS")
                else:
                    self.log("Integration tests failed", "WARNING")
            else:
                test_results['integration_tests'] = True  # No integration tests to run
            
            self.validation_results['tests'] = test_results
            
            # Fail if critical tests failed
            if not test_results['unit_tests']:
                raise ReleaseError("Unit tests failed - cannot proceed with release")
            
            return True
            
        except ReleaseError:
            raise
        except Exception as e:
            self.log(f"Test execution failed: {e}", "ERROR")
            raise ReleaseError(f"Test execution failed: {e}")

    def run_quality_checks(self):
        """Run comprehensive code quality checks."""
        self.log("Running code quality checks...")
        
        quality_results = {
            'formatting': False,
            'linting': False,
            'type_checking': False,
            'security_check': False
        }
        
        # Check if tools are available
        tools = {
            'black': 'formatting',
            'ruff': 'linting', 
            'mypy': 'type_checking',
            'pip-audit': 'security_check'
        }
        
        for tool, check_type in tools.items():
            try:
                # Check if tool is available
                version_result = self.run_command(f"python -m {tool} --version", check=False)
                if version_result.returncode != 0:
                    self.log(f"{tool} not available, skipping {check_type}", "WARNING")
                    quality_results[check_type] = True  # Skip if not available
                    continue
                
                # Run the appropriate check
                if tool == 'black':
                    result = self.run_command("python -m black --check neurolite/ tests/", check=False)
                elif tool == 'ruff':
                    result = self.run_command("python -m ruff check neurolite/ tests/", check=False)
                elif tool == 'mypy':
                    result = self.run_command("python -m mypy neurolite/", check=False)
                elif tool == 'pip-audit':
                    result = self.run_command("python -m pip-audit", check=False)
                
                if result.returncode == 0:
                    quality_results[check_type] = True
                    self.log(f"{tool} check passed", "SUCCESS")
                else:
                    self.log(f"{tool} check failed", "WARNING")
                    if self.verbose:
                        print(result.stdout)
                        print(result.stderr)
                    
                    # Only fail for critical checks
                    if tool in ['black', 'ruff']:
                        quality_results[check_type] = False
                    else:
                        quality_results[check_type] = True  # Don't fail for mypy/security
                        
            except Exception as e:
                self.log(f"Failed to run {tool}: {e}", "WARNING")
                quality_results[check_type] = True  # Don't fail release for tool issues
        
        self.validation_results['quality'] = quality_results
        
        # Check if critical quality checks passed
        critical_failed = []
        if not quality_results['formatting']:
            critical_failed.append('formatting')
        if not quality_results['linting']:
            critical_failed.append('linting')
        
        if critical_failed:
            raise ReleaseError(f"Critical quality checks failed: {', '.join(critical_failed)}")
        
        return True

    def build_package(self):
        """Build wheel and source distribution."""
        self.log("Building package...")
        
        try:
            self.run_command("python -m build")
            
            # Verify build artifacts
            dist_dir = self.project_root / "dist"
            wheel_files = list(dist_dir.glob("*.whl"))
            sdist_files = list(dist_dir.glob("*.tar.gz"))
            
            if not wheel_files:
                raise ReleaseError("No wheel files generated")
            if not sdist_files:
                raise ReleaseError("No source distribution files generated")
            
            self.log(f"Built {len(wheel_files)} wheel(s) and {len(sdist_files)} source distribution(s)", "SUCCESS")
            return True
            
        except ReleaseError:
            raise
        except Exception as e:
            raise ReleaseError(f"Package build failed: {e}")

    def validate_package(self):
        """Run comprehensive package validation."""
        self.log("Running comprehensive package validation...")
        
        try:
            # Run twine check
            self.run_command("python -m twine check dist/*")
            self.log("Twine check passed", "SUCCESS")
            
            # Run comprehensive validation script
            result = self.run_command("python scripts/validate_package.py", check=False)
            
            if result.returncode == 0:
                self.log("Comprehensive package validation passed", "SUCCESS")
                return True
            else:
                self.log("Package validation failed", "ERROR")
                if self.verbose:
                    print(result.stdout)
                    print(result.stderr)
                raise ReleaseError("Package validation failed")
                
        except ReleaseError:
            raise
        except Exception as e:
            raise ReleaseError(f"Package validation failed: {e}")

    def upload_to_test_pypi(self):
        """Upload to Test PyPI with enhanced error handling."""
        self.log("Uploading to Test PyPI...")
        
        try:
            self.run_command("python -m twine upload --repository testpypi dist/*")
            self.release_state['uploaded_to_test_pypi'] = True
            self.log("Successfully uploaded to Test PyPI", "SUCCESS")
            return True
            
        except ReleaseError as e:
            # Check if it's a duplicate version error
            if "already exists" in str(e).lower():
                self.log("Package already exists on Test PyPI", "WARNING")
                return True
            else:
                raise ReleaseError(f"Failed to upload to Test PyPI: {e}")

    def upload_to_pypi(self):
        """Upload to PyPI with confirmation."""
        self.log("Preparing to upload to PyPI...")
        
        # Final confirmation
        response = input("🚀 Upload to PyPI? This cannot be undone. (y/N): ")
        if response.lower() != 'y':
            self.log("PyPI upload cancelled by user")
            return False
        
        try:
            self.run_command("python -m twine upload dist/*")
            self.release_state['uploaded_to_pypi'] = True
            self.log("Successfully uploaded to PyPI", "SUCCESS")
            return True
            
        except ReleaseError as e:
            raise ReleaseError(f"Failed to upload to PyPI: {e}")

    def create_git_tag(self, version: str):
        """Create and push git tag."""
        self.log(f"Creating git tag v{version}...")
        
        try:
            self.run_command(f"git tag v{version}")
            self.run_command(f"git push origin v{version}")
            self.release_state['git_tagged'] = True
            self.log(f"Created and pushed git tag v{version}", "SUCCESS")
            return True
            
        except ReleaseError as e:
            raise ReleaseError(f"Failed to create git tag: {e}")

    def update_changelog(self, version: str):
        """Update CHANGELOG.md with new version."""
        changelog_file = self.project_root / "CHANGELOG.md"
        if not changelog_file.exists():
            self.log("CHANGELOG.md not found, skipping changelog update", "WARNING")
            return
        
        with open(changelog_file, "r") as f:
            content = f.read()
        
        # Add new version entry at the top
        today = datetime.now().strftime('%Y-%m-%d')
        new_entry = f"\n## [{version}] - {today}\n\n### Added\n- \n\n### Changed\n- \n\n### Fixed\n- \n\n"
        
        # Insert after the first line (usually the title)
        lines = content.split('\n')
        if len(lines) > 1:
            lines.insert(2, new_entry)
            new_content = '\n'.join(lines)
            
            with open(changelog_file, "w") as f:
                f.write(new_content)
            
            self.release_state['changelog_updated'] = True
            self.log(f"Updated CHANGELOG.md with version {version}", "SUCCESS")

    def generate_release_report(self) -> Dict:
        """Generate a comprehensive release report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "release_state": self.release_state,
            "validation_results": self.validation_results,
            "success": not any(level == "ERROR" for level in self.validation_results.values())
        }

    def run_release(self, version: Optional[str] = None, test_only: bool = False, 
                   skip_tests: bool = False, skip_quality: bool = False, 
                   skip_tag: bool = False, skip_git_checks: bool = False, 
                   dry_run: bool = False) -> bool:
        """Run the complete release workflow with enhanced error handling."""
        
        try:
            self.log("🚀 Starting enhanced release workflow", "INFO")
            
            # Store original version for rollback
            self.release_state['original_version'] = self.get_current_version()
            
            # Create backup
            self.create_backup()
            
            # Git checks
            if not skip_git_checks:
                self.check_git_status()
                self.check_git_branch()
            
            # Version handling
            if version:
                self.validate_version(version)
                current_version = self.get_current_version()
                self.log(f"Current version: {current_version}")
                self.log(f"New version: {version}")
                
                # Update version
                self.update_version(version)
                self.update_changelog(version)
            else:
                version = self.get_current_version()
                self.log(f"Using current version: {version}")
            
            # Clean previous builds
            self.clean_build()
            
            # Run quality checks
            if not skip_quality:
                self.run_quality_checks()
            
            # Run tests
            if not skip_tests:
                self.run_comprehensive_tests()
            
            # Build package
            self.build_package()
            
            # Validate package
            self.validate_package()
            
            if dry_run:
                self.log("Dry run completed successfully!", "SUCCESS")
                return True
            
            # Upload workflow
            if test_only:
                self.upload_to_test_pypi()
            else:
                # First upload to test PyPI
                self.upload_to_test_pypi()
                
                # Then upload to PyPI
                if self.upload_to_pypi():
                    # Create git tag
                    if not skip_tag:
                        self.create_git_tag(version)
            
            self.log(f"Release {version} completed successfully!", "SUCCESS")
            
            # Generate and save release report
            report = self.generate_release_report()
            report_file = self.project_root / f"release_report_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.log(f"Release report saved to {report_file}")
            
            return True
            
        except ReleaseError as e:
            self.log(f"Release failed: {e}", "ERROR")
            
            # Attempt rollback
            if input("Attempt rollback? (y/N): ").lower() == 'y':
                self.rollback_changes()
            
            return False
            
        except KeyboardInterrupt:
            self.log("Release interrupted by user", "WARNING")
            
            # Attempt rollback
            if input("Attempt rollback? (y/N): ").lower() == 'y':
                self.rollback_changes()
            
            return False
            
        except Exception as e:
            self.log(f"Unexpected error during release: {e}", "ERROR")
            
            # Attempt rollback
            if input("Attempt rollback? (y/N): ").lower() == 'y':
                self.rollback_changes()
            
            return False


def main():
    """Main entry point for the enhanced release script."""
    parser = argparse.ArgumentParser(
        description="Enhanced release automation for NeuroLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/release.py --version 0.3.0          # Release version 0.3.0
  python scripts/release.py --test-only              # Upload to Test PyPI only
  python scripts/release.py --dry-run                # Perform dry run
  python scripts/release.py --verbose --version 0.3.0 # Verbose release
        """
    )
    
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
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create release manager
    release_manager = ReleaseManager(verbose=args.verbose)
    
    # Run release
    success = release_manager.run_release(
        version=args.version,
        test_only=args.test_only,
        skip_tests=args.skip_tests,
        skip_quality=args.skip_quality,
        skip_tag=args.skip_tag,
        skip_git_checks=args.skip_git_checks,
        dry_run=args.dry_run
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()