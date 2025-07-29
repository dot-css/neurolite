"""
Visualization and formatting utilities for NeuroLite analysis results.

This module provides functions to format and visualize analysis results
in various formats including text summaries, HTML reports, and plots.
"""

from typing import Union, Dict, Any, List, Optional
import pandas as pd
from .core.data_models import ProfileReport, QuickReport, ModelRecommendation
from .core.exceptions import NeuroLiteException


def format_summary(report: Union[ProfileReport, QuickReport], 
                  format_type: str = 'text') -> str:
    """
    Format analysis results into a human-readable summary.
    
    Args:
        report: Analysis report to format
        format_type: Output format ('text', 'markdown', 'html')
        
    Returns:
        Formatted summary string
        
    Examples:
        >>> import neurolite as nl
        >>> report = nl.analyze('data.csv')
        >>> summary = nl.format_summary(report, 'text')
        >>> print(summary)
    """
    try:
        if isinstance(report, QuickReport):
            return _format_quick_summary(report, format_type)
        else:
            return _format_full_summary(report, format_type)
    except Exception as e:
        raise NeuroLiteException(f"Failed to format summary: {str(e)}")


def _format_quick_summary(report: QuickReport, format_type: str) -> str:
    """Format quick report summary."""
    if format_type == 'text':
        return f"""
NeuroLite Quick Analysis Summary
===============================

File Information:
- Format: {report.file_info.format_type}
- Confidence: {report.file_info.confidence:.1%}

Data Structure:
- Type: {report.data_structure.structure_type}
- Dimensions: {report.data_structure.dimensions}
- Sample Size: {report.data_structure.sample_size:,}
- Memory Usage: {report.data_structure.memory_usage / (1024*1024):.1f} MB

Basic Statistics:
{_format_basic_stats(report.basic_stats)}

Quick Recommendations:
{_format_recommendations_list(report.quick_recommendations)}

Analysis completed in {report.execution_time:.2f} seconds
"""
    
    elif format_type == 'markdown':
        return f"""
# NeuroLite Quick Analysis Summary

## File Information
- **Format:** {report.file_info.format_type}
- **Confidence:** {report.file_info.confidence:.1%}

## Data Structure
- **Type:** {report.data_structure.structure_type}
- **Dimensions:** {report.data_structure.dimensions}
- **Sample Size:** {report.data_structure.sample_size:,}
- **Memory Usage:** {report.data_structure.memory_usage / (1024*1024):.1f} MB

## Basic Statistics
{_format_basic_stats_markdown(report.basic_stats)}

## Quick Recommendations
{_format_recommendations_markdown(report.quick_recommendations)}

*Analysis completed in {report.execution_time:.2f} seconds*
"""
    
    elif format_type == 'html':
        return f"""
<div class="neurolite-summary">
    <h1>NeuroLite Quick Analysis Summary</h1>
    
    <h2>File Information</h2>
    <ul>
        <li><strong>Format:</strong> {report.file_info.format_type}</li>
        <li><strong>Confidence:</strong> {report.file_info.confidence:.1%}</li>
    </ul>
    
    <h2>Data Structure</h2>
    <ul>
        <li><strong>Type:</strong> {report.data_structure.structure_type}</li>
        <li><strong>Dimensions:</strong> {report.data_structure.dimensions}</li>
        <li><strong>Sample Size:</strong> {report.data_structure.sample_size:,}</li>
        <li><strong>Memory Usage:</strong> {report.data_structure.memory_usage / (1024*1024):.1f} MB</li>
    </ul>
    
    <h2>Basic Statistics</h2>
    {_format_basic_stats_html(report.basic_stats)}
    
    <h2>Quick Recommendations</h2>
    {_format_recommendations_html(report.quick_recommendations)}
    
    <p><em>Analysis completed in {report.execution_time:.2f} seconds</em></p>
</div>
"""
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def _format_full_summary(report: ProfileReport, format_type: str) -> str:
    """Format full report summary."""
    if format_type == 'text':
        return f"""
NeuroLite Comprehensive Analysis Report
======================================

File Information:
- Format: {report.file_info.format_type}
- Confidence: {report.file_info.confidence:.1%}
- Encoding: {report.file_info.encoding or 'Unknown'}

Data Structure:
- Type: {report.data_structure.structure_type}
- Dimensions: {report.data_structure.dimensions}
- Sample Size: {report.data_structure.sample_size:,}
- Memory Usage: {report.data_structure.memory_usage / (1024*1024):.1f} MB

Column Analysis:
{_format_column_analysis(report.column_analysis)}

Data Quality:
- Completeness: {report.quality_metrics.completeness:.1%}
- Consistency: {report.quality_metrics.consistency:.1%}
- Validity: {report.quality_metrics.validity:.1%}
- Uniqueness: {report.quality_metrics.uniqueness:.1%}
- Missing Pattern: {report.quality_metrics.missing_pattern}
- Duplicate Count: {report.quality_metrics.duplicate_count:,}

Task Identification:
- Task Type: {report.task_identification.task_type}
- Task Subtype: {report.task_identification.task_subtype}
- Complexity: {report.task_identification.complexity}
- Confidence: {report.task_identification.confidence:.1%}

Model Recommendations:
{_format_model_recommendations(report.model_recommendations)}

Preprocessing Recommendations:
{_format_recommendations_list(report.preprocessing_recommendations)}

Resource Requirements:
{_format_resource_requirements(report.resource_requirements)}

Analysis completed in {report.execution_time:.2f} seconds
"""
    
    elif format_type == 'markdown':
        return f"""
# NeuroLite Comprehensive Analysis Report

## File Information
- **Format:** {report.file_info.format_type}
- **Confidence:** {report.file_info.confidence:.1%}
- **Encoding:** {report.file_info.encoding or 'Unknown'}

## Data Structure
- **Type:** {report.data_structure.structure_type}
- **Dimensions:** {report.data_structure.dimensions}
- **Sample Size:** {report.data_structure.sample_size:,}
- **Memory Usage:** {report.data_structure.memory_usage / (1024*1024):.1f} MB

## Column Analysis
{_format_column_analysis_markdown(report.column_analysis)}

## Data Quality
- **Completeness:** {report.quality_metrics.completeness:.1%}
- **Consistency:** {report.quality_metrics.consistency:.1%}
- **Validity:** {report.quality_metrics.validity:.1%}
- **Uniqueness:** {report.quality_metrics.uniqueness:.1%}
- **Missing Pattern:** {report.quality_metrics.missing_pattern}
- **Duplicate Count:** {report.quality_metrics.duplicate_count:,}

## Task Identification
- **Task Type:** {report.task_identification.task_type}
- **Task Subtype:** {report.task_identification.task_subtype}
- **Complexity:** {report.task_identification.complexity}
- **Confidence:** {report.task_identification.confidence:.1%}

## Model Recommendations
{_format_model_recommendations_markdown(report.model_recommendations)}

## Preprocessing Recommendations
{_format_recommendations_markdown(report.preprocessing_recommendations)}

## Resource Requirements
{_format_resource_requirements_markdown(report.resource_requirements)}

*Analysis completed in {report.execution_time:.2f} seconds*
"""
    
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def _format_basic_stats(stats: Dict[str, Any]) -> str:
    """Format basic statistics for text output."""
    lines = []
    
    if 'shape' in stats:
        lines.append(f"- Shape: {stats['shape']}")
    
    if 'missing_values' in stats:
        missing = stats['missing_values']
        total_missing = sum(missing.values()) if missing else 0
        lines.append(f"- Total Missing Values: {total_missing:,}")
    
    if 'memory_usage_mb' in stats:
        lines.append(f"- Memory Usage: {stats['memory_usage_mb']:.1f} MB")
    
    return '\n'.join(lines) if lines else "No basic statistics available"


def _format_basic_stats_markdown(stats: Dict[str, Any]) -> str:
    """Format basic statistics for markdown output."""
    lines = []
    
    if 'shape' in stats:
        lines.append(f"- **Shape:** {stats['shape']}")
    
    if 'missing_values' in stats:
        missing = stats['missing_values']
        total_missing = sum(missing.values()) if missing else 0
        lines.append(f"- **Total Missing Values:** {total_missing:,}")
    
    if 'memory_usage_mb' in stats:
        lines.append(f"- **Memory Usage:** {stats['memory_usage_mb']:.1f} MB")
    
    return '\n'.join(lines) if lines else "*No basic statistics available*"


def _format_basic_stats_html(stats: Dict[str, Any]) -> str:
    """Format basic statistics for HTML output."""
    lines = ["<ul>"]
    
    if 'shape' in stats:
        lines.append(f"<li><strong>Shape:</strong> {stats['shape']}</li>")
    
    if 'missing_values' in stats:
        missing = stats['missing_values']
        total_missing = sum(missing.values()) if missing else 0
        lines.append(f"<li><strong>Total Missing Values:</strong> {total_missing:,}</li>")
    
    if 'memory_usage_mb' in stats:
        lines.append(f"<li><strong>Memory Usage:</strong> {stats['memory_usage_mb']:.1f} MB</li>")
    
    lines.append("</ul>")
    return '\n'.join(lines) if len(lines) > 2 else "<p><em>No basic statistics available</em></p>"


def _format_column_analysis(column_analysis: Dict[str, Any]) -> str:
    """Format column analysis for text output."""
    if not column_analysis:
        return "No column analysis available"
    
    lines = []
    for col_name, col_type in column_analysis.items():
        confidence_str = f" ({col_type.confidence:.1%})" if hasattr(col_type, 'confidence') else ""
        subtype_str = f" - {col_type.subtype}" if hasattr(col_type, 'subtype') and col_type.subtype else ""
        lines.append(f"- {col_name}: {col_type.primary_type}{subtype_str}{confidence_str}")
    
    return '\n'.join(lines)


def _format_column_analysis_markdown(column_analysis: Dict[str, Any]) -> str:
    """Format column analysis for markdown output."""
    if not column_analysis:
        return "*No column analysis available*"
    
    lines = []
    for col_name, col_type in column_analysis.items():
        confidence_str = f" ({col_type.confidence:.1%})" if hasattr(col_type, 'confidence') else ""
        subtype_str = f" - {col_type.subtype}" if hasattr(col_type, 'subtype') and col_type.subtype else ""
        lines.append(f"- **{col_name}:** {col_type.primary_type}{subtype_str}{confidence_str}")
    
    return '\n'.join(lines)


def _format_model_recommendations(recommendations: List[ModelRecommendation]) -> str:
    """Format model recommendations for text output."""
    if not recommendations:
        return "No model recommendations available"
    
    lines = []
    for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
        lines.append(f"{i}. {rec.model_name} ({rec.model_type})")
        lines.append(f"   Confidence: {rec.confidence:.1%}")
        lines.append(f"   Rationale: {rec.rationale}")
        if rec.expected_performance:
            perf_str = ", ".join([f"{k}: {v:.3f}" for k, v in rec.expected_performance.items()])
            lines.append(f"   Expected Performance: {perf_str}")
        lines.append("")
    
    return '\n'.join(lines)


def _format_model_recommendations_markdown(recommendations: List[ModelRecommendation]) -> str:
    """Format model recommendations for markdown output."""
    if not recommendations:
        return "*No model recommendations available*"
    
    lines = []
    for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
        lines.append(f"{i}. **{rec.model_name}** ({rec.model_type})")
        lines.append(f"   - **Confidence:** {rec.confidence:.1%}")
        lines.append(f"   - **Rationale:** {rec.rationale}")
        if rec.expected_performance:
            perf_str = ", ".join([f"{k}: {v:.3f}" for k, v in rec.expected_performance.items()])
            lines.append(f"   - **Expected Performance:** {perf_str}")
        lines.append("")
    
    return '\n'.join(lines)


def _format_recommendations_list(recommendations: List[str]) -> str:
    """Format recommendation list for text output."""
    if not recommendations:
        return "No recommendations available"
    
    return '\n'.join([f"- {rec}" for rec in recommendations])


def _format_recommendations_markdown(recommendations: List[str]) -> str:
    """Format recommendation list for markdown output."""
    if not recommendations:
        return "*No recommendations available*"
    
    return '\n'.join([f"- {rec}" for rec in recommendations])


def _format_recommendations_html(recommendations: List[str]) -> str:
    """Format recommendation list for HTML output."""
    if not recommendations:
        return "<p><em>No recommendations available</em></p>"
    
    items = '\n'.join([f"<li>{rec}</li>" for rec in recommendations])
    return f"<ul>\n{items}\n</ul>"


def _format_resource_requirements(requirements: Dict[str, Any]) -> str:
    """Format resource requirements for text output."""
    if not requirements:
        return "No resource requirements available"
    
    lines = []
    
    if 'estimated_memory_mb' in requirements:
        lines.append(f"- Estimated Memory: {requirements['estimated_memory_mb']:.1f} MB")
    
    if 'estimated_processing_time_seconds' in requirements:
        time_sec = requirements['estimated_processing_time_seconds']
        if time_sec < 60:
            lines.append(f"- Estimated Processing Time: {time_sec:.1f} seconds")
        else:
            lines.append(f"- Estimated Processing Time: {time_sec/60:.1f} minutes")
    
    if 'recommended_hardware' in requirements:
        lines.append(f"- Recommended Hardware: {requirements['recommended_hardware'].upper()}")
    
    return '\n'.join(lines) if lines else "No resource requirements available"


def _format_resource_requirements_markdown(requirements: Dict[str, Any]) -> str:
    """Format resource requirements for markdown output."""
    if not requirements:
        return "*No resource requirements available*"
    
    lines = []
    
    if 'estimated_memory_mb' in requirements:
        lines.append(f"- **Estimated Memory:** {requirements['estimated_memory_mb']:.1f} MB")
    
    if 'estimated_processing_time_seconds' in requirements:
        time_sec = requirements['estimated_processing_time_seconds']
        if time_sec < 60:
            lines.append(f"- **Estimated Processing Time:** {time_sec:.1f} seconds")
        else:
            lines.append(f"- **Estimated Processing Time:** {time_sec/60:.1f} minutes")
    
    if 'recommended_hardware' in requirements:
        lines.append(f"- **Recommended Hardware:** {requirements['recommended_hardware'].upper()}")
    
    return '\n'.join(lines) if lines else "*No resource requirements available*"


def create_dataframe_summary(report: Union[ProfileReport, QuickReport]) -> pd.DataFrame:
    """
    Create a pandas DataFrame summary of the analysis results.
    
    Args:
        report: Analysis report to summarize
        
    Returns:
        DataFrame with key metrics and findings
        
    Examples:
        >>> import neurolite as nl
        >>> report = nl.analyze('data.csv')
        >>> summary_df = nl.create_dataframe_summary(report)
        >>> print(summary_df)
    """
    try:
        if isinstance(report, QuickReport):
            data = {
                'Metric': ['Format', 'Structure Type', 'Dimensions', 'Sample Size', 'Memory (MB)', 'Execution Time (s)'],
                'Value': [
                    report.file_info.format_type,
                    report.data_structure.structure_type,
                    str(report.data_structure.dimensions),
                    f"{report.data_structure.sample_size:,}",
                    f"{report.data_structure.memory_usage / (1024*1024):.1f}",
                    f"{report.execution_time:.2f}"
                ]
            }
        else:
            data = {
                'Metric': [
                    'Format', 'Structure Type', 'Dimensions', 'Sample Size', 'Memory (MB)',
                    'Completeness', 'Consistency', 'Validity', 'Uniqueness',
                    'Task Type', 'Task Subtype', 'Task Confidence', 'Execution Time (s)'
                ],
                'Value': [
                    report.file_info.format_type,
                    report.data_structure.structure_type,
                    str(report.data_structure.dimensions),
                    f"{report.data_structure.sample_size:,}",
                    f"{report.data_structure.memory_usage / (1024*1024):.1f}",
                    f"{report.quality_metrics.completeness:.1%}",
                    f"{report.quality_metrics.consistency:.1%}",
                    f"{report.quality_metrics.validity:.1%}",
                    f"{report.quality_metrics.uniqueness:.1%}",
                    report.task_identification.task_type,
                    report.task_identification.task_subtype,
                    f"{report.task_identification.confidence:.1%}",
                    f"{report.execution_time:.2f}"
                ]
            }
        
        return pd.DataFrame(data)
        
    except Exception as e:
        raise NeuroLiteException(f"Failed to create DataFrame summary: {str(e)}")


def export_report(report: Union[ProfileReport, QuickReport], 
                 filepath: str, format_type: str = 'json') -> None:
    """
    Export analysis report to file.
    
    Args:
        report: Analysis report to export
        filepath: Output file path
        format_type: Export format ('json', 'csv', 'html', 'markdown')
        
    Examples:
        >>> import neurolite as nl
        >>> report = nl.analyze('data.csv')
        >>> nl.export_report(report, 'analysis_report.html', 'html')
    """
    try:
        if format_type == 'json':
            import json
            # Convert report to dict (simplified)
            report_dict = {
                'file_info': {
                    'format_type': report.file_info.format_type,
                    'confidence': report.file_info.confidence,
                    'encoding': report.file_info.encoding
                },
                'data_structure': {
                    'structure_type': report.data_structure.structure_type,
                    'dimensions': report.data_structure.dimensions,
                    'sample_size': report.data_structure.sample_size,
                    'memory_usage': report.data_structure.memory_usage
                },
                'execution_time': report.execution_time,
                'timestamp': report.timestamp.isoformat()
            }
            
            if isinstance(report, ProfileReport):
                report_dict.update({
                    'quality_metrics': {
                        'completeness': report.quality_metrics.completeness,
                        'consistency': report.quality_metrics.consistency,
                        'validity': report.quality_metrics.validity,
                        'uniqueness': report.quality_metrics.uniqueness
                    },
                    'task_identification': {
                        'task_type': report.task_identification.task_type,
                        'task_subtype': report.task_identification.task_subtype,
                        'confidence': report.task_identification.confidence
                    },
                    'model_recommendations': [
                        {
                            'model_name': rec.model_name,
                            'model_type': rec.model_type,
                            'confidence': rec.confidence,
                            'rationale': rec.rationale
                        } for rec in report.model_recommendations
                    ]
                })
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2)
                
        elif format_type == 'csv':
            summary_df = create_dataframe_summary(report)
            summary_df.to_csv(filepath, index=False)
            
        elif format_type == 'html':
            html_content = format_summary(report, 'html')
            with open(filepath, 'w') as f:
                f.write(html_content)
                
        elif format_type == 'markdown':
            md_content = format_summary(report, 'markdown')
            with open(filepath, 'w') as f:
                f.write(md_content)
                
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
            
    except Exception as e:
        raise NeuroLiteException(f"Failed to export report: {str(e)}")


# Add visualization functions to the public API
__all__ = [
    'format_summary',
    'create_dataframe_summary', 
    'export_report'
]