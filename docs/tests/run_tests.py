#!/usr/bin/env python3
"""
Documentation Test Runner

This script runs all documentation tests to ensure examples and tutorials
remain functional and up-to-date.

Usage:
    python run_tests.py [--verbose] [--test-type TYPE]
    
Options:
    --verbose: Enable verbose output
    --test-type: Run specific test type (docs, examples, all)
"""

import sys
import os
import argparse
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import test modules
from test_documentation import run_documentation_tests
from test_examples import run_example_tests


def run_all_tests(verbose=False):
    """Run all documentation tests."""
    print("Running NeuroLite Documentation Tests")
    print("=" * 50)
    
    success = True
    
    # Run documentation tests
    print("\n1. Running Documentation Tests...")
    print("-" * 30)
    doc_success = run_documentation_tests()
    if not doc_success:
        print("❌ Documentation tests failed")
        success = False
    else:
        print("✅ Documentation tests passed")
    
    # Run example tests
    print("\n2. Running Example Tests...")
    print("-" * 30)
    example_success = run_example_tests()
    if not example_success:
        print("❌ Example tests failed")
        success = False
    else:
        print("✅ Example tests passed")
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 All documentation tests passed!")
        return True
    else:
        print("💥 Some documentation tests failed!")
        return False


def run_specific_tests(test_type, verbose=False):
    """Run specific type of tests."""
    if test_type == "docs":
        print("Running Documentation Tests Only...")
        return run_documentation_tests()
    elif test_type == "examples":
        print("Running Example Tests Only...")
        return run_example_tests()
    else:
        print(f"Unknown test type: {test_type}")
        return False


def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        'pandas',
        'numpy',
        'pathlib'
    ]
    
    optional_packages = [
        'neurolite',
        'matplotlib',
        'seaborn'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        print(f"❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("Some tests may be skipped. Install with: pip install " + " ".join(missing_optional))
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run NeuroLite documentation tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tests.py                    # Run all tests
    python run_tests.py --verbose          # Run with verbose output
    python run_tests.py --test-type docs   # Run only documentation tests
    python run_tests.py --test-type examples  # Run only example tests
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--test-type', '-t',
        choices=['docs', 'examples', 'all'],
        default='all',
        help='Type of tests to run (default: all)'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies and exit'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        if not args.check_deps:
            print("\nContinuing with available packages...")
        else:
            sys.exit(1)
    elif args.check_deps:
        print("✅ All dependencies are available")
        sys.exit(0)
    
    # Set verbosity
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Run tests
    if args.test_type == 'all':
        success = run_all_tests(args.verbose)
    else:
        success = run_specific_tests(args.test_type, args.verbose)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()