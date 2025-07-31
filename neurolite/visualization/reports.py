"""
Comprehensive reporting system for NeuroLite.

Provides automated report generation with training summaries,
model performance analysis, and data insights.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import asdict
import base64
from io import BytesIO

try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..training.trainer import TrainingHistory
from ..evaluation.evaluator import EvaluationResults
from ..models.base import TaskType
from ..data.detector import DataType
from ..core import get_logger
from ..core.exceptions import VisualizationError
from .plots import TrainingPlotter, PerformancePlotter, DataPlotter

logger = get_logger(__name__)


class ReportGenerator:
    """
    Comprehensive report generator for training and evaluation results.
    
    Creates detailed HTML, PDF, and JSON reports with visualizations,
    metrics, and analysis summaries.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize report generator.
        
        Args:
            template_dir: Directory containing custom report templates
        """
        self.template_dir = template_dir or self._get_default_template_dir()
        
        if JINJA2_AVAILABLE:
            self.env = Environment(
                loader=FileSystemLoader(self.template_dir),
                autoescape=True
            )
        else:
            self.env = None
            logger.warning("Jinja2 not available. HTML reports will be basic.")
        
        # Initialize plotters
        self.training_plotter = TrainingPlotter(backend="plotly" if PLOTLY_AVAILABLE else "matplotlib")
        self.performance_plotter = PerformancePlotter(backend="plotly" if PLOTLY_AVAILABLE else "matplotlib")
        self.data_plotter = DataPlotter(backend="plotly" if PLOTLY_AVAILABLE else "matplotlib")
        
        logger.debug(f"Report generator initialized with template dir: {self.template_dir}")
    
    def _get_default_template_dir(self) -> str:
        """Get default template directory."""
        current_dir = Path(__file__).parent
        template_dir = current_dir / "templates"
        template_dir.mkdir(exist_ok=True)
        return str(template_dir)
    
    def generate_training_report(
        self,
        history: TrainingHistory,
        results: Optional[EvaluationResults] = None,
        model_info: Optional[Dict[str, Any]] = None,
        data_info: Optional[Dict[str, Any]] = None,
        output_path: str = "training_report.html",
        format: str = "html"
    ) -> str:
        """
        Generate comprehensive training report.
        
        Args:
            history: Training history
            results: Evaluation results (optional)
            model_info: Model information (optional)
            data_info: Data information (optional)
            output_path: Output file path
            format: Report format ("html", "json", "pdf")
            
        Returns:
            Path to generated report
        """
        logger.info(f"Generating training report in {format} format")
        
        # Prepare report data
        report_data = self._prepare_training_report_data(
            history, results, model_info, data_info
        )
        
        # Generate visualizations
        visualizations = self._generate_training_visualizations(history, results)
        report_data["visualizations"] = visualizations
        
        # Generate report based on format
        if format.lower() == "html":
            return self._generate_html_report(report_data, output_path, "training_report.html")
        elif format.lower() == "json":
            return self._generate_json_report(report_data, output_path)
        elif format.lower() == "pdf":
            return self._generate_pdf_report(report_data, output_path)
        else:
            raise VisualizationError(f"Unsupported report format: {format}")
    
    def generate_evaluation_report(
        self,
        results: EvaluationResults,
        model_info: Optional[Dict[str, Any]] = None,
        data_info: Optional[Dict[str, Any]] = None,
        output_path: str = "evaluation_report.html",
        format: str = "html"
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            model_info: Model information (optional)
            data_info: Data information (optional)
            output_path: Output file path
            format: Report format ("html", "json", "pdf")
            
        Returns:
            Path to generated report
        """
        logger.info(f"Generating evaluation report in {format} format")
        
        # Prepare report data
        report_data = self._prepare_evaluation_report_data(results, model_info, data_info)
        
        # Generate visualizations
        visualizations = self._generate_evaluation_visualizations(results)
        report_data["visualizations"] = visualizations
        
        # Generate report based on format
        if format.lower() == "html":
            return self._generate_html_report(report_data, output_path, "evaluation_report.html")
        elif format.lower() == "json":
            return self._generate_json_report(report_data, output_path)
        elif format.lower() == "pdf":
            return self._generate_pdf_report(report_data, output_path)
        else:
            raise VisualizationError(f"Unsupported report format: {format}")
    
    def generate_data_report(
        self,
        data: Any,
        data_type: DataType,
        data_info: Optional[Dict[str, Any]] = None,
        output_path: str = "data_report.html",
        format: str = "html"
    ) -> str:
        """
        Generate data analysis report.
        
        Args:
            data: Input data
            data_type: Type of data
            data_info: Additional data information
            output_path: Output file path
            format: Report format ("html", "json", "pdf")
            
        Returns:
            Path to generated report
        """
        logger.info(f"Generating data report in {format} format")
        
        # Prepare report data
        report_data = self._prepare_data_report_data(data, data_type, data_info)
        
        # Generate visualizations
        visualizations = self._generate_data_visualizations(data, data_type)
        report_data["visualizations"] = visualizations
        
        # Generate report based on format
        if format.lower() == "html":
            return self._generate_html_report(report_data, output_path, "data_report.html")
        elif format.lower() == "json":
            return self._generate_json_report(report_data, output_path)
        elif format.lower() == "pdf":
            return self._generate_pdf_report(report_data, output_path)
        else:
            raise VisualizationError(f"Unsupported report format: {format}")
    
    def _prepare_training_report_data(
        self,
        history: TrainingHistory,
        results: Optional[EvaluationResults],
        model_info: Optional[Dict[str, Any]],
        data_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare data for training report."""
        report_data = {
            "title": "Training Report",
            "generated_at": datetime.now().isoformat(),
            "training_summary": self._summarize_training(history),
            "model_info": model_info or {},
            "data_info": data_info or {},
            "training_history": asdict(history) if history else None,
            "evaluation_results": asdict(results) if results else None
        }
        
        # Add performance summary if evaluation results available
        if results:
            report_data["performance_summary"] = self._summarize_performance(results)
        
        return report_data
    
    def _prepare_evaluation_report_data(
        self,
        results: EvaluationResults,
        model_info: Optional[Dict[str, Any]],
        data_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare data for evaluation report."""
        return {
            "title": "Evaluation Report",
            "generated_at": datetime.now().isoformat(),
            "performance_summary": self._summarize_performance(results),
            "model_info": model_info or {},
            "data_info": data_info or {},
            "evaluation_results": asdict(results)
        }
    
    def _prepare_data_report_data(
        self,
        data: Any,
        data_type: DataType,
        data_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare data for data analysis report."""
        return {
            "title": "Data Analysis Report",
            "generated_at": datetime.now().isoformat(),
            "data_summary": self._summarize_data(data, data_type),
            "data_type": data_type.value,
            "data_info": data_info or {}
        }
    
    def _summarize_training(self, history: TrainingHistory) -> Dict[str, Any]:
        """Create training summary."""
        if not history.epochs:
            return {"status": "No training data available"}
        
        summary = {
            "total_epochs": len(history.epochs),
            "final_train_loss": history.train_loss[-1] if history.train_loss else None,
            "final_val_loss": history.val_loss[-1] if history.val_loss and any(v != 0 for v in history.val_loss) else None,
            "best_train_loss": min(history.train_loss) if history.train_loss else None,
            "best_val_loss": min(v for v in history.val_loss if v != 0) if history.val_loss and any(v != 0 for v in history.val_loss) else None,
            "metrics": {}
        }
        
        # Summarize metrics
        for metric_name, values in history.metrics.items():
            if values and not metric_name.startswith('val_'):
                summary["metrics"][metric_name] = {
                    "final": values[-1],
                    "best": max(values) if "accuracy" in metric_name.lower() or "f1" in metric_name.lower() else min(values)
                }
        
        return summary
    
    def _summarize_performance(self, results: EvaluationResults) -> Dict[str, Any]:
        """Create performance summary."""
        summary = {
            "task_type": results.task_type.value,
            "model_name": results.model_name,
            "prediction_time": results.prediction_time,
            "samples_per_second": results.samples_per_second,
            "metrics": {}
        }
        
        # Extract key metrics
        if hasattr(results.metrics, 'accuracy') and results.metrics.accuracy is not None:
            summary["metrics"]["accuracy"] = results.metrics.accuracy
        
        if hasattr(results.metrics, 'f1_score') and results.metrics.f1_score is not None:
            summary["metrics"]["f1_score"] = results.metrics.f1_score
        
        if hasattr(results.metrics, 'precision') and results.metrics.precision is not None:
            summary["metrics"]["precision"] = results.metrics.precision
        
        if hasattr(results.metrics, 'recall') and results.metrics.recall is not None:
            summary["metrics"]["recall"] = results.metrics.recall
        
        return summary
    
    def _summarize_data(self, data: Any, data_type: DataType) -> Dict[str, Any]:
        """Create data summary."""
        summary = {
            "data_type": data_type.value,
            "shape": getattr(data, 'shape', None),
            "size": len(data) if hasattr(data, '__len__') else None
        }
        
        # Add type-specific summaries
        if data_type == DataType.TABULAR:
            try:
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    summary.update({
                        "columns": list(data.columns),
                        "dtypes": data.dtypes.to_dict(),
                        "missing_values": data.isnull().sum().to_dict(),
                        "numeric_columns": list(data.select_dtypes(include=['number']).columns)
                    })
            except ImportError:
                pass
        
        return summary
    
    def _generate_training_visualizations(
        self,
        history: TrainingHistory,
        results: Optional[EvaluationResults]
    ) -> Dict[str, str]:
        """Generate visualizations for training report."""
        visualizations = {}
        
        try:
            # Training history plot
            if history and history.epochs:
                fig = self.training_plotter.plot_training_history(history)
                visualizations["training_history"] = self._figure_to_base64(fig)
            
            # Performance plots if evaluation results available
            if results:
                if results.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                    # Confusion matrix
                    try:
                        fig = self.performance_plotter.plot_confusion_matrix(results)
                        visualizations["confusion_matrix"] = self._figure_to_base64(fig)
                    except Exception as e:
                        logger.warning(f"Failed to generate confusion matrix: {e}")
                    
                    # ROC curve for binary classification
                    if results.task_type == TaskType.BINARY_CLASSIFICATION and results.probabilities is not None:
                        try:
                            fig = self.performance_plotter.plot_roc_curve(results)
                            visualizations["roc_curve"] = self._figure_to_base64(fig)
                        except Exception as e:
                            logger.warning(f"Failed to generate ROC curve: {e}")
        
        except Exception as e:
            logger.error(f"Error generating training visualizations: {e}")
        
        return visualizations
    
    def _generate_evaluation_visualizations(self, results: EvaluationResults) -> Dict[str, str]:
        """Generate visualizations for evaluation report."""
        visualizations = {}
        
        try:
            if results.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                # Confusion matrix
                try:
                    fig = self.performance_plotter.plot_confusion_matrix(results)
                    visualizations["confusion_matrix"] = self._figure_to_base64(fig)
                except Exception as e:
                    logger.warning(f"Failed to generate confusion matrix: {e}")
                
                # ROC curve for binary classification
                if results.task_type == TaskType.BINARY_CLASSIFICATION and results.probabilities is not None:
                    try:
                        fig = self.performance_plotter.plot_roc_curve(results)
                        visualizations["roc_curve"] = self._figure_to_base64(fig)
                    except Exception as e:
                        logger.warning(f"Failed to generate ROC curve: {e}")
        
        except Exception as e:
            logger.error(f"Error generating evaluation visualizations: {e}")
        
        return visualizations
    
    def _generate_data_visualizations(self, data: Any, data_type: DataType) -> Dict[str, str]:
        """Generate visualizations for data report."""
        visualizations = {}
        
        try:
            # Data distribution plot
            fig = self.data_plotter.plot_data_distribution(data, data_type)
            visualizations["data_distribution"] = self._figure_to_base64(fig)
        
        except Exception as e:
            logger.error(f"Error generating data visualizations: {e}")
        
        return visualizations
    
    def _figure_to_base64(self, fig: Any) -> str:
        """Convert figure to base64 string for embedding in HTML."""
        try:
            if PLOTLY_AVAILABLE and hasattr(fig, 'to_html'):
                # For Plotly figures, return HTML directly
                return fig.to_html(include_plotlyjs='cdn', div_id=None)
            elif MATPLOTLIB_AVAILABLE:
                # For matplotlib figures
                buffer = BytesIO()
                fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close(fig)  # Clean up
                return f"data:image/png;base64,{image_base64}"
            else:
                return ""
        except Exception as e:
            logger.error(f"Error converting figure to base64: {e}")
            return ""
    
    def _generate_html_report(
        self,
        report_data: Dict[str, Any],
        output_path: str,
        template_name: str
    ) -> str:
        """Generate HTML report."""
        if self.env:
            try:
                template = self.env.get_template(template_name)
                html_content = template.render(**report_data)
            except Exception as e:
                logger.warning(f"Failed to use template {template_name}: {e}")
                html_content = self._generate_basic_html_report(report_data)
        else:
            html_content = self._generate_basic_html_report(report_data)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
        return output_path
    
    def _generate_basic_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate basic HTML report without templates."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            f"<title>{report_data.get('title', 'Report')}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "h1, h2 { color: #333; }",
            ".metric { margin: 10px 0; }",
            ".visualization { margin: 20px 0; text-align: center; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "</style>",
            "</head><body>",
            f"<h1>{report_data.get('title', 'Report')}</h1>",
            f"<p>Generated at: {report_data.get('generated_at', 'Unknown')}</p>"
        ]
        
        # Add summaries
        if 'training_summary' in report_data:
            html_parts.extend(self._format_training_summary_html(report_data['training_summary']))
        
        if 'performance_summary' in report_data:
            html_parts.extend(self._format_performance_summary_html(report_data['performance_summary']))
        
        if 'data_summary' in report_data:
            html_parts.extend(self._format_data_summary_html(report_data['data_summary']))
        
        # Add visualizations
        if 'visualizations' in report_data:
            html_parts.append("<h2>Visualizations</h2>")
            for name, viz in report_data['visualizations'].items():
                if viz:
                    html_parts.append(f"<div class='visualization'>")
                    html_parts.append(f"<h3>{name.replace('_', ' ').title()}</h3>")
                    if viz.startswith('data:image'):
                        html_parts.append(f"<img src='{viz}' alt='{name}' style='max-width: 100%;'>")
                    else:
                        html_parts.append(viz)  # Plotly HTML
                    html_parts.append("</div>")
        
        html_parts.extend(["</body></html>"])
        
        return "\n".join(html_parts)
    
    def _format_training_summary_html(self, summary: Dict[str, Any]) -> List[str]:
        """Format training summary as HTML."""
        html_parts = ["<h2>Training Summary</h2>"]
        
        if summary.get('status'):
            html_parts.append(f"<p>{summary['status']}</p>")
            return html_parts
        
        html_parts.append("<table>")
        html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
        
        for key, value in summary.items():
            if key != 'metrics' and value is not None:
                html_parts.append(f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>")
        
        html_parts.append("</table>")
        
        if summary.get('metrics'):
            html_parts.append("<h3>Final Metrics</h3>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>Metric</th><th>Final</th><th>Best</th></tr>")
            
            for metric, values in summary['metrics'].items():
                html_parts.append(f"<tr><td>{metric}</td><td>{values.get('final', 'N/A')}</td><td>{values.get('best', 'N/A')}</td></tr>")
            
            html_parts.append("</table>")
        
        return html_parts
    
    def _format_performance_summary_html(self, summary: Dict[str, Any]) -> List[str]:
        """Format performance summary as HTML."""
        html_parts = ["<h2>Performance Summary</h2>"]
        
        html_parts.append("<table>")
        html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
        
        for key, value in summary.items():
            if key != 'metrics' and value is not None:
                html_parts.append(f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>")
        
        html_parts.append("</table>")
        
        if summary.get('metrics'):
            html_parts.append("<h3>Model Metrics</h3>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
            
            for metric, value in summary['metrics'].items():
                html_parts.append(f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value:.4f}</td></tr>")
            
            html_parts.append("</table>")
        
        return html_parts
    
    def _format_data_summary_html(self, summary: Dict[str, Any]) -> List[str]:
        """Format data summary as HTML."""
        html_parts = ["<h2>Data Summary</h2>"]
        
        html_parts.append("<table>")
        html_parts.append("<tr><th>Property</th><th>Value</th></tr>")
        
        for key, value in summary.items():
            if key not in ['columns', 'dtypes', 'missing_values'] and value is not None:
                html_parts.append(f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>")
        
        html_parts.append("</table>")
        
        return html_parts
    
    def _generate_json_report(self, report_data: Dict[str, Any], output_path: str) -> str:
        """Generate JSON report."""
        # Remove visualizations from JSON (they're too large)
        json_data = {k: v for k, v in report_data.items() if k != 'visualizations'}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info(f"JSON report generated: {output_path}")
        return output_path
    
    def _generate_pdf_report(self, report_data: Dict[str, Any], output_path: str) -> str:
        """Generate PDF report."""
        try:
            import weasyprint
            
            # Generate HTML first
            html_content = self._generate_basic_html_report(report_data)
            
            # Convert to PDF
            weasyprint.HTML(string=html_content).write_pdf(output_path)
            
            logger.info(f"PDF report generated: {output_path}")
            return output_path
            
        except ImportError:
            logger.error("WeasyPrint not available for PDF generation. Install with: pip install weasyprint")
            raise VisualizationError("PDF generation requires WeasyPrint. Install with: pip install weasyprint")
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            raise VisualizationError(f"PDF generation failed: {e}")


def create_default_templates():
    """Create default report templates."""
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)
    
    # Basic training report template
    training_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1, h2, h3 { color: #333; }
        .summary { background: #f5f5f5; padding: 20px; margin: 20px 0; }
        .metric { margin: 10px 0; }
        .visualization { margin: 20px 0; text-align: center; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <p>Generated at: {{ generated_at }}</p>
    
    {% if training_summary %}
    <div class="summary">
        <h2>Training Summary</h2>
        <p>Total Epochs: {{ training_summary.total_epochs }}</p>
        <p>Final Training Loss: {{ training_summary.final_train_loss }}</p>
        {% if training_summary.final_val_loss %}
        <p>Final Validation Loss: {{ training_summary.final_val_loss }}</p>
        {% endif %}
    </div>
    {% endif %}
    
    {% if visualizations %}
    <h2>Visualizations</h2>
    {% for name, viz in visualizations.items() %}
    <div class="visualization">
        <h3>{{ name.replace('_', ' ').title() }}</h3>
        {% if viz.startswith('data:image') %}
        <img src="{{ viz }}" alt="{{ name }}" style="max-width: 100%;">
        {% else %}
        {{ viz|safe }}
        {% endif %}
    </div>
    {% endfor %}
    {% endif %}
</body>
</html>
    """
    
    with open(template_dir / "training_report.html", 'w') as f:
        f.write(training_template.strip())
    
    # Copy for other report types
    with open(template_dir / "evaluation_report.html", 'w') as f:
        f.write(training_template.strip())
    
    with open(template_dir / "data_report.html", 'w') as f:
        f.write(training_template.strip())
    
    logger.info(f"Default templates created in {template_dir}")


# Create default templates on import
try:
    create_default_templates()
except Exception as e:
    logger.warning(f"Failed to create default templates: {e}")