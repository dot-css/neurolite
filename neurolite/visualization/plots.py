"""
Core plotting utilities for NeuroLite visualization.

Provides plotting functions for training progress, model performance,
and data analysis with support for multiple backends.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import warnings

from ..training.trainer import TrainingHistory
from ..evaluation.evaluator import EvaluationResults
from ..models.base import TaskType
from ..data.detector import DataType
from ..core import get_logger
from ..core.exceptions import VisualizationError

logger = get_logger(__name__)

# Set default style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


class TrainingPlotter:
    """
    Plotter for training progress visualization.
    
    Provides live and static plotting of training metrics, loss curves,
    and learning rate schedules.
    """
    
    def __init__(self, backend: str = "matplotlib"):
        """
        Initialize training plotter.
        
        Args:
            backend: Plotting backend ("matplotlib" or "plotly")
        """
        self.backend = backend.lower()
        if self.backend not in ["matplotlib", "plotly"]:
            raise VisualizationError(f"Unsupported backend: {backend}")
        
        logger.debug(f"Initialized TrainingPlotter with {self.backend} backend")
    
    def plot_training_history(
        self,
        history: TrainingHistory,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> Any:
        """
        Plot complete training history with loss and metrics.
        
        Args:
            history: Training history object
            metrics: Specific metrics to plot (plots all if None)
            figsize: Figure size for matplotlib
            save_path: Path to save the plot
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        if not history.epochs:
            raise VisualizationError("Training history is empty")
        
        # Determine which metrics to plot
        available_metrics = list(history.metrics.keys())
        if metrics is None:
            metrics = available_metrics
        else:
            metrics = [m for m in metrics if m in available_metrics]
        
        if self.backend == "matplotlib":
            return self._plot_training_history_matplotlib(
                history, metrics, figsize, save_path
            )
        else:
            return self._plot_training_history_plotly(
                history, metrics, save_path
            )
    
    def _plot_training_history_matplotlib(
        self,
        history: TrainingHistory,
        metrics: List[str],
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> plt.Figure:
        """Plot training history using matplotlib."""
        # Calculate subplot layout
        n_plots = 1 + len(metrics)  # Loss + metrics
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        epochs = history.epochs
        
        # Plot loss
        ax = axes[0]
        ax.plot(epochs, history.train_loss, label='Training Loss', linewidth=2)
        if any(val != 0.0 for val in history.val_loss):
            ax.plot(epochs, history.val_loss, label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot metrics
        for i, metric_name in enumerate(metrics, 1):
            if i >= len(axes):
                break
                
            ax = axes[i]
            metric_values = history.metrics[metric_name]
            
            # Check if we have validation version of this metric
            val_metric_name = f"val_{metric_name}"
            has_val_metric = val_metric_name in history.metrics
            
            ax.plot(epochs, metric_values, label=f'Training {metric_name.title()}', linewidth=2)
            
            if has_val_metric:
                val_metric_values = history.metrics[val_metric_name]
                ax.plot(epochs, val_metric_values, label=f'Validation {metric_name.title()}', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.title())
            ax.set_title(f'{metric_name.title()} Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        return fig
    
    def _plot_training_history_plotly(
        self,
        history: TrainingHistory,
        metrics: List[str],
        save_path: Optional[str]
    ) -> go.Figure:
        """Plot training history using plotly."""
        # Calculate subplot layout
        n_plots = 1 + len(metrics)
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplot titles
        subplot_titles = ['Training and Validation Loss']
        subplot_titles.extend([f'{metric.title()} Over Time' for metric in metrics])
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles[:n_plots]
        )
        
        epochs = history.epochs
        
        # Plot loss
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history.train_loss,
                mode='lines',
                name='Training Loss',
                line=dict(width=2)
            ),
            row=1, col=1
        )
        
        if any(val != 0.0 for val in history.val_loss):
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history.val_loss,
                    mode='lines',
                    name='Validation Loss',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Plot metrics
        for i, metric_name in enumerate(metrics):
            row = (i + 1) // n_cols + 1
            col = (i + 1) % n_cols + 1
            if col == 1 and i + 1 >= n_cols:
                col = n_cols
            
            metric_values = history.metrics[metric_name]
            
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=metric_values,
                    mode='lines',
                    name=f'Training {metric_name.title()}',
                    line=dict(width=2)
                ),
                row=row, col=col
            )
            
            # Check for validation metric
            val_metric_name = f"val_{metric_name}"
            if val_metric_name in history.metrics:
                val_metric_values = history.metrics[val_metric_name]
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=val_metric_values,
                        mode='lines',
                        name=f'Validation {metric_name.title()}',
                        line=dict(width=2)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=400 * n_rows,
            showlegend=True,
            title_text="Training History"
        )
        
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Value")
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        
        return fig
    
    def plot_live_training(
        self,
        history: TrainingHistory,
        update_frequency: int = 1
    ) -> Any:
        """
        Create live updating training plot.
        
        Args:
            history: Training history object (updated externally)
            update_frequency: Update frequency in epochs
            
        Returns:
            Live plot object
        """
        if self.backend == "plotly":
            return self._create_live_plotly_training(history, update_frequency)
        else:
            logger.warning("Live plotting not fully supported with matplotlib backend")
            return self.plot_training_history(history)
    
    def _create_live_plotly_training(
        self,
        history: TrainingHistory,
        update_frequency: int
    ) -> go.FigureWidget:
        """Create live updating plotly training plot."""
        fig = go.FigureWidget()
        
        # Add initial traces
        fig.add_trace(go.Scatter(
            x=[], y=[], mode='lines', name='Training Loss'
        ))
        fig.add_trace(go.Scatter(
            x=[], y=[], mode='lines', name='Validation Loss'
        ))
        
        fig.update_layout(
            title="Live Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss"
        )
        
        return fig


class PerformancePlotter:
    """
    Plotter for model performance visualization.
    
    Provides plotting functions for confusion matrices, ROC curves,
    precision-recall curves, and other performance metrics.
    """
    
    def __init__(self, backend: str = "matplotlib"):
        """
        Initialize performance plotter.
        
        Args:
            backend: Plotting backend ("matplotlib" or "plotly")
        """
        self.backend = backend.lower()
        if self.backend not in ["matplotlib", "plotly"]:
            raise VisualizationError(f"Unsupported backend: {backend}")
        
        logger.debug(f"Initialized PerformancePlotter with {self.backend} backend")
    
    def plot_confusion_matrix(
        self,
        results: EvaluationResults,
        normalize: bool = False,
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None
    ) -> Any:
        """
        Plot confusion matrix for classification results.
        
        Args:
            results: Evaluation results containing predictions
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size for matplotlib
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        if results.task_type not in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            raise VisualizationError("Confusion matrix only available for classification tasks")
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(results.ground_truth, results.predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        if self.backend == "matplotlib":
            return self._plot_confusion_matrix_matplotlib(cm, normalize, figsize, save_path)
        else:
            return self._plot_confusion_matrix_plotly(cm, normalize, save_path)
    
    def _plot_confusion_matrix_matplotlib(
        self,
        cm: np.ndarray,
        normalize: bool,
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> plt.Figure:
        """Plot confusion matrix using matplotlib."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def _plot_confusion_matrix_plotly(
        self,
        cm: np.ndarray,
        normalize: bool,
        save_path: Optional[str]
    ) -> go.Figure:
        """Plot confusion matrix using plotly."""
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title='Proportion' if normalize else 'Count')
        ))
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]}",
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black")
                )
        
        fig.update_layout(
            title='Normalized Confusion Matrix' if normalize else 'Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label'
        )
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curve(
        self,
        results: EvaluationResults,
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None
    ) -> Any:
        """
        Plot ROC curve for binary classification results.
        
        Args:
            results: Evaluation results with probabilities
            figsize: Figure size for matplotlib
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        if results.task_type != TaskType.BINARY_CLASSIFICATION:
            raise VisualizationError("ROC curve only available for binary classification")
        
        if results.probabilities is None:
            raise VisualizationError("ROC curve requires prediction probabilities")
        
        from sklearn.metrics import roc_curve, auc
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(results.ground_truth, results.probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        
        if self.backend == "matplotlib":
            return self._plot_roc_curve_matplotlib(fpr, tpr, roc_auc, figsize, save_path)
        else:
            return self._plot_roc_curve_plotly(fpr, tpr, roc_auc, save_path)
    
    def _plot_roc_curve_matplotlib(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        roc_auc: float,
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> plt.Figure:
        """Plot ROC curve using matplotlib."""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        return fig
    
    def _plot_roc_curve_plotly(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        roc_auc: float,
        save_path: Optional[str]
    ) -> go.Figure:
        """Plot ROC curve using plotly."""
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
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"ROC curve saved to {save_path}")
        
        return fig


class DataPlotter:
    """
    Plotter for data analysis and quality visualization.
    
    Provides plotting functions for data distributions, correlations,
    and quality assessments.
    """
    
    def __init__(self, backend: str = "matplotlib"):
        """
        Initialize data plotter.
        
        Args:
            backend: Plotting backend ("matplotlib" or "plotly")
        """
        self.backend = backend.lower()
        if self.backend not in ["matplotlib", "plotly"]:
            raise VisualizationError(f"Unsupported backend: {backend}")
        
        logger.debug(f"Initialized DataPlotter with {self.backend} backend")
    
    def plot_data_distribution(
        self,
        data: Union[np.ndarray, Any],
        data_type: DataType,
        columns: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> Any:
        """
        Plot data distribution based on data type.
        
        Args:
            data: Input data
            data_type: Type of data
            columns: Column names for tabular data
            figsize: Figure size for matplotlib
            save_path: Path to save the plot
            
        Returns:
            Figure object
        """
        if data_type == DataType.TABULAR:
            return self._plot_tabular_distribution(data, columns, figsize, save_path)
        elif data_type == DataType.IMAGE:
            return self._plot_image_distribution(data, figsize, save_path)
        elif data_type == DataType.TEXT:
            return self._plot_text_distribution(data, figsize, save_path)
        else:
            raise VisualizationError(f"Distribution plotting not supported for {data_type}")
    
    def _plot_tabular_distribution(
        self,
        data: Any,
        columns: Optional[List[str]],
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> Any:
        """Plot distribution for tabular data."""
        import pandas as pd
        
        # Convert to DataFrame if needed
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data, columns=columns)
        else:
            df = data
        
        if self.backend == "matplotlib":
            return self._plot_tabular_distribution_matplotlib(df, figsize, save_path)
        else:
            return self._plot_tabular_distribution_plotly(df, save_path)
    
    def _plot_tabular_distribution_matplotlib(
        self,
        df: Any,
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> plt.Figure:
        """Plot tabular data distribution using matplotlib."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        
        if n_cols == 0:
            raise VisualizationError("No numeric columns found for distribution plotting")
        
        # Calculate subplot layout
        n_plot_cols = min(3, n_cols)
        n_plot_rows = (n_cols + n_plot_cols - 1) // n_plot_cols
        
        fig, axes = plt.subplots(n_plot_rows, n_plot_cols, figsize=figsize)
        if n_cols == 1:
            axes = [axes]
        elif n_plot_rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i >= len(axes):
                break
                
            ax = axes[i]
            df[col].hist(bins=30, ax=ax, alpha=0.7)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Data distribution plot saved to {save_path}")
        
        return fig
    
    def _plot_tabular_distribution_plotly(
        self,
        df: Any,
        save_path: Optional[str]
    ) -> go.Figure:
        """Plot tabular data distribution using plotly."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        
        if n_cols == 0:
            raise VisualizationError("No numeric columns found for distribution plotting")
        
        # Calculate subplot layout
        n_plot_cols = min(3, n_cols)
        n_plot_rows = (n_cols + n_plot_cols - 1) // n_plot_cols
        
        fig = make_subplots(
            rows=n_plot_rows,
            cols=n_plot_cols,
            subplot_titles=[f'Distribution of {col}' for col in numeric_cols]
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // n_plot_cols + 1
            col_idx = i % n_plot_cols + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, showlegend=False),
                row=row, col=col_idx
            )
        
        fig.update_layout(
            height=400 * n_plot_rows,
            title_text="Data Distribution"
        )
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)
            logger.info(f"Data distribution plot saved to {save_path}")
        
        return fig
    
    def _plot_image_distribution(
        self,
        data: Any,
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> Any:
        """Plot distribution for image data."""
        # For image data, plot pixel intensity distributions
        if hasattr(data, 'shape') and len(data.shape) >= 3:
            # Flatten image data to get pixel intensities
            pixel_values = data.flatten()
            
            if self.backend == "matplotlib":
                fig, ax = plt.subplots(figsize=figsize)
                ax.hist(pixel_values, bins=50, alpha=0.7)
                ax.set_title('Pixel Intensity Distribution')
                ax.set_xlabel('Pixel Intensity')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Image distribution plot saved to {save_path}")
                
                return fig
            else:
                fig = go.Figure(data=[go.Histogram(x=pixel_values)])
                fig.update_layout(
                    title='Pixel Intensity Distribution',
                    xaxis_title='Pixel Intensity',
                    yaxis_title='Frequency'
                )
                
                if save_path:
                    if save_path.endswith('.html'):
                        fig.write_html(save_path)
                    else:
                        fig.write_image(save_path)
                    logger.info(f"Image distribution plot saved to {save_path}")
                
                return fig
        else:
            raise VisualizationError("Invalid image data format")
    
    def _plot_text_distribution(
        self,
        data: Any,
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> Any:
        """Plot distribution for text data."""
        # For text data, plot length distributions
        if isinstance(data, (list, np.ndarray)):
            text_lengths = [len(str(text)) for text in data]
            
            if self.backend == "matplotlib":
                fig, ax = plt.subplots(figsize=figsize)
                ax.hist(text_lengths, bins=30, alpha=0.7)
                ax.set_title('Text Length Distribution')
                ax.set_xlabel('Text Length (characters)')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Text distribution plot saved to {save_path}")
                
                return fig
            else:
                fig = go.Figure(data=[go.Histogram(x=text_lengths)])
                fig.update_layout(
                    title='Text Length Distribution',
                    xaxis_title='Text Length (characters)',
                    yaxis_title='Frequency'
                )
                
                if save_path:
                    if save_path.endswith('.html'):
                        fig.write_html(save_path)
                    else:
                        fig.write_image(save_path)
                    logger.info(f"Text distribution plot saved to {save_path}")
                
                return fig
        else:
            raise VisualizationError("Invalid text data format")


# Convenience functions
def plot_training_history(
    history: TrainingHistory,
    backend: str = "matplotlib",
    **kwargs
) -> Any:
    """
    Convenience function to plot training history.
    
    Args:
        history: Training history object
        backend: Plotting backend
        **kwargs: Additional plotting arguments
        
    Returns:
        Figure object
    """
    plotter = TrainingPlotter(backend=backend)
    return plotter.plot_training_history(history, **kwargs)


def plot_confusion_matrix(
    results: EvaluationResults,
    backend: str = "matplotlib",
    **kwargs
) -> Any:
    """
    Convenience function to plot confusion matrix.
    
    Args:
        results: Evaluation results
        backend: Plotting backend
        **kwargs: Additional plotting arguments
        
    Returns:
        Figure object
    """
    plotter = PerformancePlotter(backend=backend)
    return plotter.plot_confusion_matrix(results, **kwargs)


def plot_roc_curve(
    results: EvaluationResults,
    backend: str = "matplotlib",
    **kwargs
) -> Any:
    """
    Convenience function to plot ROC curve.
    
    Args:
        results: Evaluation results
        backend: Plotting backend
        **kwargs: Additional plotting arguments
        
    Returns:
        Figure object
    """
    plotter = PerformancePlotter(backend=backend)
    return plotter.plot_roc_curve(results, **kwargs)


def plot_data_distribution(
    data: Any,
    data_type: DataType,
    backend: str = "matplotlib",
    **kwargs
) -> Any:
    """
    Convenience function to plot data distribution.
    
    Args:
        data: Input data
        data_type: Type of data
        backend: Plotting backend
        **kwargs: Additional plotting arguments
        
    Returns:
        Figure object
    """
    plotter = DataPlotter(backend=backend)
    return plotter.plot_data_distribution(data, data_type, **kwargs)