"""
Visualization and reporting engine for NeuroLite.

Provides comprehensive visualization capabilities for training progress,
model performance, data analysis, and reporting with export functionality.
"""

from .plots import (
    TrainingPlotter,
    PerformancePlotter,
    DataPlotter,
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_data_distribution
)
from .dashboard import VisualizationDashboard, LiveTrainingMonitor
from .reports import ReportGenerator
from .exporters import (
    VisualizationExporter,
    export_figure,
    export_multiple_figures,
    get_supported_formats
)

__all__ = [
    # Core plotting classes
    "TrainingPlotter",
    "PerformancePlotter", 
    "DataPlotter",
    
    # Dashboard and monitoring
    "VisualizationDashboard",
    "LiveTrainingMonitor",
    
    # Reporting
    "ReportGenerator",
    
    # Export functionality
    "VisualizationExporter",
    "export_figure",
    "export_multiple_figures",
    "get_supported_formats",
    
    # Convenience functions
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_data_distribution"
]