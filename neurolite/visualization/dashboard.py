"""
Interactive visualization dashboard for NeuroLite.

Provides real-time training monitoring and comprehensive model analysis
through interactive web-based dashboards.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json
import webbrowser
from dataclasses import asdict

try:
    import dash
    from dash import dcc, html, Input, Output, State
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

from ..training.trainer import TrainingHistory
from ..evaluation.evaluator import EvaluationResults
from ..models.base import TaskType
from ..data.detector import DataType
from ..core import get_logger
from ..core.exceptions import VisualizationError

logger = get_logger(__name__)


class VisualizationDashboard:
    """
    Interactive dashboard for real-time training monitoring and analysis.
    
    Provides web-based interface for monitoring training progress,
    analyzing model performance, and exploring data characteristics.
    """
    
    def __init__(
        self,
        port: int = 8050,
        host: str = "127.0.0.1",
        debug: bool = False
    ):
        """
        Initialize visualization dashboard.
        
        Args:
            port: Port number for the dashboard server
            host: Host address for the dashboard server
            debug: Enable debug mode
        """
        if not DASH_AVAILABLE:
            raise VisualizationError(
                "Dash is required for interactive dashboards. "
                "Install with: pip install dash plotly"
            )
        
        self.port = port
        self.host = host
        self.debug = debug
        
        # Dashboard state
        self.training_history: Optional[TrainingHistory] = None
        self.evaluation_results: Optional[EvaluationResults] = None
        self.is_training = False
        self.update_callbacks: List[Callable] = []
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
        logger.info(f"Dashboard initialized on {host}:{port}")
    
    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("NeuroLite Training Dashboard", className="header-title"),
                html.Div(id="status-indicator", className="status-indicator")
            ], className="header"),
            
            # Main content
            html.Div([
                # Training progress section
                html.Div([
                    html.H2("Training Progress"),
                    dcc.Graph(id="training-loss-plot"),
                    dcc.Graph(id="training-metrics-plot"),
                    dcc.Interval(
                        id="training-interval",
                        interval=2000,  # Update every 2 seconds
                        n_intervals=0
                    )
                ], className="training-section"),
                
                # Model performance section
                html.Div([
                    html.H2("Model Performance"),
                    dcc.Graph(id="confusion-matrix-plot"),
                    dcc.Graph(id="roc-curve-plot")
                ], className="performance-section"),
                
                # Data analysis section
                html.Div([
                    html.H2("Data Analysis"),
                    dcc.Graph(id="data-distribution-plot"),
                    dcc.Graph(id="data-quality-plot")
                ], className="data-section")
            ], className="main-content")
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks for interactivity."""
        
        @self.app.callback(
            [Output("training-loss-plot", "figure"),
             Output("training-metrics-plot", "figure"),
             Output("status-indicator", "children")],
            [Input("training-interval", "n_intervals")]
        )
        def update_training_plots(n_intervals):
            """Update training progress plots."""
            if self.training_history is None:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No training data available")
                return empty_fig, empty_fig, "No training in progress"
            
            # Create loss plot
            loss_fig = self._create_loss_plot()
            
            # Create metrics plot
            metrics_fig = self._create_metrics_plot()
            
            # Update status
            status = "Training in progress..." if self.is_training else "Training completed"
            
            return loss_fig, metrics_fig, status
        
        @self.app.callback(
            [Output("confusion-matrix-plot", "figure"),
             Output("roc-curve-plot", "figure")],
            [Input("training-interval", "n_intervals")]
        )
        def update_performance_plots(n_intervals):
            """Update model performance plots."""
            if self.evaluation_results is None:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No evaluation results available")
                return empty_fig, empty_fig
            
            # Create confusion matrix plot
            cm_fig = self._create_confusion_matrix_plot()
            
            # Create ROC curve plot
            roc_fig = self._create_roc_curve_plot()
            
            return cm_fig, roc_fig
    
    def _create_loss_plot(self) -> go.Figure:
        """Create training loss plot."""
        if not self.training_history or not self.training_history.epochs:
            fig = go.Figure()
            fig.update_layout(title="Training Loss")
            return fig
        
        fig = go.Figure()
        
        # Add training loss
        fig.add_trace(go.Scatter(
            x=self.training_history.epochs,
            y=self.training_history.train_loss,
            mode='lines',
            name='Training Loss',
            line=dict(color='blue', width=2)
        ))
        
        # Add validation loss if available
        if any(val != 0.0 for val in self.training_history.val_loss):
            fig.add_trace(go.Scatter(
                x=self.training_history.epochs,
                y=self.training_history.val_loss,
                mode='lines',
                name='Validation Loss',
                line=dict(color='red', width=2)
            ))
        
        fig.update_layout(
            title="Training and Validation Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode='x unified'
        )
        
        return fig
    
    def _create_metrics_plot(self) -> go.Figure:
        """Create training metrics plot."""
        if not self.training_history or not self.training_history.metrics:
            fig = go.Figure()
            fig.update_layout(title="Training Metrics")
            return fig
        
        # Get available metrics (excluding validation metrics)
        train_metrics = {k: v for k, v in self.training_history.metrics.items() 
                        if not k.startswith('val_')}
        
        if not train_metrics:
            fig = go.Figure()
            fig.update_layout(title="No metrics available")
            return fig
        
        # Create subplots for multiple metrics
        n_metrics = len(train_metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=list(train_metrics.keys())
        )
        
        for i, (metric_name, values) in enumerate(train_metrics.items()):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            # Add training metric
            fig.add_trace(
                go.Scatter(
                    x=self.training_history.epochs,
                    y=values,
                    mode='lines',
                    name=f'Train {metric_name}',
                    line=dict(width=2)
                ),
                row=row, col=col
            )
            
            # Add validation metric if available
            val_metric_name = f"val_{metric_name}"
            if val_metric_name in self.training_history.metrics:
                fig.add_trace(
                    go.Scatter(
                        x=self.training_history.epochs,
                        y=self.training_history.metrics[val_metric_name],
                        mode='lines',
                        name=f'Val {metric_name}',
                        line=dict(width=2, dash='dash')
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=400 * n_rows,
            title="Training Metrics",
            showlegend=True
        )
        
        return fig
    
    def _create_confusion_matrix_plot(self) -> go.Figure:
        """Create confusion matrix plot."""
        if (self.evaluation_results is None or 
            self.evaluation_results.task_type not in [
                TaskType.BINARY_CLASSIFICATION, 
                TaskType.MULTICLASS_CLASSIFICATION
            ]):
            fig = go.Figure()
            fig.update_layout(title="Confusion Matrix (Not available for this task)")
            return fig
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(
            self.evaluation_results.ground_truth,
            self.evaluation_results.predictions
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title='Count')
        ))
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                fig.add_annotation(
                    x=j, y=i,
                    text=str(cm[i, j]),
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black")
                )
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label'
        )
        
        return fig
    
    def _create_roc_curve_plot(self) -> go.Figure:
        """Create ROC curve plot."""
        if (self.evaluation_results is None or 
            self.evaluation_results.task_type != TaskType.BINARY_CLASSIFICATION or
            self.evaluation_results.probabilities is None):
            fig = go.Figure()
            fig.update_layout(title="ROC Curve (Not available for this task)")
            return fig
        
        from sklearn.metrics import roc_curve, auc
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(
            self.evaluation_results.ground_truth,
            self.evaluation_results.probabilities[:, 1]
        )
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {roc_auc:.2f})',
            line=dict(color='darkorange', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05])
        )
        
        return fig
    
    def update_training_history(self, history: TrainingHistory):
        """
        Update training history data.
        
        Args:
            history: Updated training history
        """
        self.training_history = history
        logger.debug("Training history updated in dashboard")
    
    def update_evaluation_results(self, results: EvaluationResults):
        """
        Update evaluation results data.
        
        Args:
            results: Updated evaluation results
        """
        self.evaluation_results = results
        logger.debug("Evaluation results updated in dashboard")
    
    def set_training_status(self, is_training: bool):
        """
        Update training status.
        
        Args:
            is_training: Whether training is currently in progress
        """
        self.is_training = is_training
        logger.debug(f"Training status updated: {is_training}")
    
    def start_server(self, open_browser: bool = True):
        """
        Start the dashboard server.
        
        Args:
            open_browser: Whether to automatically open browser
        """
        if open_browser:
            # Open browser after a short delay
            def open_browser_delayed():
                time.sleep(1.5)
                webbrowser.open(f"http://{self.host}:{self.port}")
            
            threading.Thread(target=open_browser_delayed, daemon=True).start()
        
        logger.info(f"Starting dashboard server on http://{self.host}:{self.port}")
        
        try:
            self.app.run_server(
                host=self.host,
                port=self.port,
                debug=self.debug,
                use_reloader=False  # Disable reloader to avoid issues
            )
        except Exception as e:
            logger.error(f"Failed to start dashboard server: {e}")
            raise VisualizationError(f"Dashboard server failed to start: {e}")
    
    def stop_server(self):
        """Stop the dashboard server."""
        # Note: Dash doesn't provide a clean way to stop the server
        # This is a limitation of the current implementation
        logger.info("Dashboard server stop requested")
    
    def export_dashboard_data(self, output_path: str):
        """
        Export dashboard data to JSON file.
        
        Args:
            output_path: Path to save the data
        """
        data = {
            "training_history": asdict(self.training_history) if self.training_history else None,
            "evaluation_results": asdict(self.evaluation_results) if self.evaluation_results else None,
            "is_training": self.is_training,
            "timestamp": time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Dashboard data exported to {output_path}")


class LiveTrainingMonitor:
    """
    Live training monitor that updates dashboard in real-time.
    
    Provides integration between training process and visualization dashboard.
    """
    
    def __init__(self, dashboard: VisualizationDashboard):
        """
        Initialize live training monitor.
        
        Args:
            dashboard: Visualization dashboard to update
        """
        self.dashboard = dashboard
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        logger.debug("Live training monitor initialized")
    
    def start_monitoring(self, training_history: TrainingHistory):
        """
        Start monitoring training progress.
        
        Args:
            training_history: Training history object to monitor
        """
        if self.is_monitoring:
            logger.warning("Training monitoring already in progress")
            return
        
        self.is_monitoring = True
        self.dashboard.set_training_status(True)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_training,
            args=(training_history,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Started live training monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring training progress."""
        self.is_monitoring = False
        self.dashboard.set_training_status(False)
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        logger.info("Stopped live training monitoring")
    
    def _monitor_training(self, training_history: TrainingHistory):
        """
        Monitor training progress in background thread.
        
        Args:
            training_history: Training history to monitor
        """
        while self.is_monitoring:
            try:
                # Update dashboard with current training history
                self.dashboard.update_training_history(training_history)
                
                # Sleep before next update
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in training monitoring: {e}")
                break
        
        logger.debug("Training monitoring thread stopped")