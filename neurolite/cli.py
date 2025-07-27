"""
Command-line interface for NeuroLite.

Provides a simple CLI for analyzing datasets and generating reports.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from .detectors import QualityDetector, DataTypeDetector, FileDetector
from .core.exceptions import NeuroLiteException


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog='neurolite',
        description='NeuroLite: Automated AI/ML data analysis and model recommendation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neurolite analyze data.csv
  neurolite analyze data.csv --output report.json
  neurolite analyze data.csv --quality-only
  neurolite analyze data.csv --format json
  neurolite version
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze a dataset and generate a report'
    )
    analyze_parser.add_argument(
        'input_file',
        help='Path to the input data file'
    )
    analyze_parser.add_argument(
        '--output', '-o',
        help='Output file path (default: stdout)'
    )
    analyze_parser.add_argument(
        '--format', '-f',
        choices=['json', 'text'],
        default='text',
        help='Output format (default: text)'
    )
    analyze_parser.add_argument(
        '--quality-only',
        action='store_true',
        help='Only perform quality analysis'
    )
    analyze_parser.add_argument(
        '--types-only',
        action='store_true',
        help='Only perform type detection'
    )
    analyze_parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.8,
        help='Confidence threshold for classifications (default: 0.8)'
    )
    
    # Version command
    version_parser = subparsers.add_parser(
        'version',
        help='Show version information'
    )
    
    return parser


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Detect file format and load accordingly
    suffix = path.suffix.lower()
    
    try:
        if suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif suffix == '.json':
            return pd.read_json(file_path)
        elif suffix == '.parquet':
            return pd.read_parquet(file_path)
        else:
            # Try CSV as default
            return pd.read_csv(file_path)
    except Exception as e:
        raise NeuroLiteException(f"Failed to load data from {file_path}: {e}")


def analyze_quality(df: pd.DataFrame, confidence_threshold: float) -> dict:
    """Perform quality analysis."""
    detector = QualityDetector(confidence_threshold=confidence_threshold)
    
    # Comprehensive quality analysis
    quality_metrics = detector.analyze_quality(df)
    missing_analysis = detector.detect_missing_patterns(df)
    duplicate_analysis = detector.find_duplicates(df)
    consistency_report = detector.validate_consistency(df)
    
    return {
        'quality_metrics': {
            'completeness': quality_metrics.completeness,
            'consistency': quality_metrics.consistency,
            'validity': quality_metrics.validity,
            'uniqueness': quality_metrics.uniqueness,
            'missing_pattern': quality_metrics.missing_pattern,
            'duplicate_count': quality_metrics.duplicate_count,
        },
        'missing_analysis': {
            'missing_percentage': missing_analysis.missing_percentage,
            'missing_pattern_type': missing_analysis.missing_pattern_type,
            'missing_columns': missing_analysis.missing_columns,
            'imputation_strategy': missing_analysis.imputation_strategy,
        },
        'duplicate_analysis': {
            'duplicate_count': duplicate_analysis.duplicate_count,
            'duplicate_percentage': duplicate_analysis.duplicate_percentage,
            'exact_duplicates': duplicate_analysis.exact_duplicates,
            'partial_duplicates': duplicate_analysis.partial_duplicates,
        },
        'consistency_report': {
            'format_consistency_score': consistency_report.format_consistency_score,
            'range_consistency_score': consistency_report.range_consistency_score,
            'referential_integrity_score': consistency_report.referential_integrity_score,
            'inconsistent_formats': consistency_report.inconsistent_formats,
            'integrity_violations': consistency_report.integrity_violations,
        }
    }


def analyze_types(df: pd.DataFrame, confidence_threshold: float) -> dict:
    """Perform type analysis."""
    detector = DataTypeDetector(confidence_threshold=confidence_threshold)
    
    column_types = detector.classify_columns(df)
    
    return {
        'column_types': {
            col: {
                'primary_type': col_type.primary_type,
                'subtype': col_type.subtype,
                'confidence': col_type.confidence,
                'properties': col_type.properties,
            }
            for col, col_type in column_types.items()
        }
    }


def analyze_file_structure(df: pd.DataFrame) -> dict:
    """Analyze file structure."""
    detector = FileDetector()
    
    data_structure = detector.detect_structure(df)
    
    return {
        'data_structure': {
            'structure_type': data_structure.structure_type,
            'dimensions': data_structure.dimensions,
            'sample_size': data_structure.sample_size,
            'memory_usage': data_structure.memory_usage,
        }
    }


def format_text_output(results: dict) -> str:
    """Format results as human-readable text."""
    output = []
    
    output.append("NeuroLite Analysis Report")
    output.append("=" * 50)
    output.append("")
    
    # File structure
    if 'data_structure' in results:
        ds = results['data_structure']
        output.append("Data Structure:")
        output.append(f"  Type: {ds['structure_type']}")
        output.append(f"  Dimensions: {ds['dimensions']}")
        output.append(f"  Sample Size: {ds['sample_size']:,}")
        output.append(f"  Memory Usage: {ds['memory_usage']:,} bytes")
        output.append("")
    
    # Quality metrics
    if 'quality_metrics' in results:
        qm = results['quality_metrics']
        output.append("Quality Metrics:")
        output.append(f"  Completeness: {qm['completeness']:.2%}")
        output.append(f"  Consistency: {qm['consistency']:.2%}")
        output.append(f"  Validity: {qm['validity']:.2%}")
        output.append(f"  Uniqueness: {qm['uniqueness']:.2%}")
        output.append(f"  Missing Pattern: {qm['missing_pattern']}")
        output.append(f"  Duplicate Count: {qm['duplicate_count']}")
        output.append("")
    
    # Missing data analysis
    if 'missing_analysis' in results:
        ma = results['missing_analysis']
        output.append("Missing Data Analysis:")
        output.append(f"  Missing Percentage: {ma['missing_percentage']:.2%}")
        output.append(f"  Pattern Type: {ma['missing_pattern_type']}")
        output.append(f"  Missing Columns: {', '.join(ma['missing_columns'])}")
        output.append(f"  Imputation Strategy: {ma['imputation_strategy']}")
        output.append("")
    
    # Column types
    if 'column_types' in results:
        output.append("Column Types:")
        for col, col_info in results['column_types'].items():
            output.append(f"  {col}: {col_info['primary_type']} ({col_info['subtype']}) "
                         f"[{col_info['confidence']:.2%}]")
        output.append("")
    
    return "\n".join(output)


def handle_analyze_command(args) -> int:
    """Handle the analyze command."""
    try:
        # Load data
        print(f"Loading data from {args.input_file}...", file=sys.stderr)
        df = load_data(args.input_file)
        print(f"Loaded dataset with shape {df.shape}", file=sys.stderr)
        
        # Perform analysis
        results = {}
        
        if not args.types_only:
            print("Performing quality analysis...", file=sys.stderr)
            results.update(analyze_quality(df, args.confidence_threshold))
        
        if not args.quality_only:
            print("Performing type analysis...", file=sys.stderr)
            results.update(analyze_types(df, args.confidence_threshold))
            
            print("Analyzing file structure...", file=sys.stderr)
            results.update(analyze_file_structure(df))
        
        # Format output
        if args.format == 'json':
            output = json.dumps(results, indent=2, default=str)
        else:
            output = format_text_output(results)
        
        # Write output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Report saved to {args.output}", file=sys.stderr)
        else:
            print(output)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def handle_version_command(args) -> int:
    """Handle the version command."""
    from . import __version__
    print(f"NeuroLite version {__version__}")
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'analyze':
        return handle_analyze_command(args)
    elif args.command == 'version':
        return handle_version_command(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())