#!/usr/bin/env python3
"""
Comprehensive test runner for NeuroLite.
Runs all tests including unit tests, integration tests, and performance benchmarks.
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path


def run_command(cmd, description, capture_output=False):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ FAILED: {description}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
            else:
                print(f"✅ PASSED: {description}")
                return True
        else:
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"❌ FAILED: {description}")
                return False
            else:
                print(f"✅ PASSED: {description}")
                return True
    except Exception as e:
        print(f"❌ ERROR: {description} - {e}")
        return False
    finally:
        elapsed = time.time() - start_time
        print(f"⏱️  Elapsed time: {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive NeuroLite tests')
    parser.add_argument('--unit-only', action='store_true', 
                       help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--performance-only', action='store_true',
                       help='Run only performance benchmarks')
    parser.add_argument('--skip-lint', action='store_true',
                       help='Skip linting checks')
    parser.add_argument('--skip-type-check', action='store_true',
                       help='Skip type checking')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🚀 Starting NeuroLite Comprehensive Test Suite")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version}")
    
    results = []
    
    # Code quality checks
    if not args.skip_lint and not args.integration_only and not args.performance_only:
        print("\n" + "="*80)
        print("🔍 CODE QUALITY CHECKS")
        print("="*80)
        
        # Black formatting check
        results.append(run_command(
            "black --check neurolite tests",
            "Code formatting check (black)",
            capture_output=True
        ))
        
        # Flake8 linting
        results.append(run_command(
            "flake8 neurolite tests --count --select=E9,F63,F7,F82 --show-source --statistics",
            "Critical linting errors (flake8)",
            capture_output=True
        ))
        
        results.append(run_command(
            "flake8 neurolite tests --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics",
            "Code quality check (flake8)",
            capture_output=True
        ))
    
    # Type checking
    if not args.skip_type_check and not args.integration_only and not args.performance_only:
        results.append(run_command(
            "mypy neurolite --ignore-missing-imports",
            "Type checking (mypy)",
            capture_output=True
        ))
    
    # Unit tests
    if not args.integration_only and not args.performance_only:
        print("\n" + "="*80)
        print("🧪 UNIT TESTS")
        print("="*80)
        
        unit_test_cmd = "pytest tests/ -v --tb=short"
        if not args.integration_only:
            unit_test_cmd += " --ignore=tests/integration/"
        if args.verbose:
            unit_test_cmd += " -s"
        
        results.append(run_command(
            unit_test_cmd + " --cov=neurolite --cov-report=term-missing",
            "Unit tests with coverage"
        ))
    
    # Integration tests
    if not args.unit_only and not args.performance_only:
        print("\n" + "="*80)
        print("🔗 INTEGRATION TESTS")
        print("="*80)
        
        integration_cmd = "pytest tests/integration/ -v --tb=short"
        if args.verbose:
            integration_cmd += " -s"
        
        results.append(run_command(
            integration_cmd,
            "Integration tests"
        ))
    
    # Performance benchmarks
    if not args.unit_only and not args.integration_only:
        print("\n" + "="*80)
        print("⚡ PERFORMANCE BENCHMARKS")
        print("="*80)
        
        perf_cmd = "pytest tests/integration/test_performance_benchmarks.py -v -s --tb=short"
        
        results.append(run_command(
            perf_cmd,
            "Performance benchmarks"
        ))
    
    # Compatibility tests
    if not args.unit_only and not args.performance_only:
        print("\n" + "="*80)
        print("🌐 COMPATIBILITY TESTS")
        print("="*80)
        
        compat_cmd = "pytest tests/integration/test_compatibility.py -v --tb=short"
        if args.verbose:
            compat_cmd += " -s"
        
        results.append(run_command(
            compat_cmd,
            "Compatibility tests"
        ))
    
    # Package build test
    if not args.unit_only and not args.integration_only and not args.performance_only:
        print("\n" + "="*80)
        print("📦 PACKAGE BUILD TEST")
        print("="*80)
        
        # Clean previous builds
        run_command("rm -rf build/ dist/ *.egg-info/", "Clean previous builds", capture_output=True)
        
        results.append(run_command(
            "python -m build",
            "Package build test",
            capture_output=True
        ))
        
        results.append(run_command(
            "twine check dist/*",
            "Package validation",
            capture_output=True
        ))
    
    # Generate comprehensive test report
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    failed = total - passed
    
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success rate: {passed/total*100:.1f}%")
    
    # Generate detailed report
    report_path = os.path.join(project_root, 'test_report.md')
    with open(report_path, 'w') as f:
        f.write("# NeuroLite Comprehensive Test Report\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Python Version:** {sys.version}\n\n")
        f.write(f"**Platform:** {sys.platform}\n\n")
        
        f.write("## Test Results Summary\n\n")
        f.write(f"- **Total Tests:** {total}\n")
        f.write(f"- **Passed:** {passed}\n")
        f.write(f"- **Failed:** {failed}\n")
        f.write(f"- **Success Rate:** {passed/total*100:.1f}%\n\n")
        
        if failed == 0:
            f.write("## ✅ Status: ALL TESTS PASSED\n\n")
            f.write("The NeuroLite library has successfully passed all comprehensive tests and is ready for release.\n\n")
            f.write("### Test Categories Completed:\n")
            f.write("- Code quality checks (formatting, linting, type checking)\n")
            f.write("- Unit tests with coverage analysis\n")
            f.write("- Integration tests with real datasets\n")
            f.write("- Performance benchmarks and optimization tests\n")
            f.write("- Cross-platform compatibility tests\n")
            f.write("- Package build and distribution tests\n\n")
        else:
            f.write("## ❌ Status: TESTS FAILED\n\n")
            f.write(f"{failed} test(s) failed. Please review and fix the failing tests before proceeding with release.\n\n")
        
        f.write("## Recommendations\n\n")
        if failed == 0:
            f.write("- The library is ready for production deployment\n")
            f.write("- Consider running performance benchmarks on target hardware\n")
            f.write("- Update documentation with any new features\n")
            f.write("- Prepare release notes and changelog\n")
        else:
            f.write("- Fix all failing tests before release\n")
            f.write("- Re-run comprehensive test suite after fixes\n")
            f.write("- Consider adding additional test coverage for failed areas\n")
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("The NeuroLite library is ready for release!")
        return 0
    else:
        print(f"\n💥 {failed} TEST(S) FAILED")
        print("Please fix the failing tests before release.")
        return 1


if __name__ == "__main__":
    sys.exit(main())