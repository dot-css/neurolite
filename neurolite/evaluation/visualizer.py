"""
Visualization generators for NeuroLite evaluation results.

Provides comprehensive visualization capabilities for confusion matrices,
ROC curves, performance plots, and other evaluation visualizations.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from pathlib import Path
import warnings

from ..models.base import TaskType
from ..data.detector import DataType
from ..core.exceptions import EvaluationError
from ..core import get_logger
from .evaluator import EvaluationResults
from .validator import CrossValidationResults

logger = get_logger(__name__)


class VisualizationEngine:
    """
    Comprehensive visualization engine for evaluation results.
    
    Generates publication-ready plots for model evaluation including
    confusion matrices, ROC curves, performance plots, and more.
    """
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize the visualization engine.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self._check_dependencies()
        self._setup_style()
        logger.debug("Initialized VisualizationEngine")
    
    def _check_dependencies(self) -> None:
        """Check if required visualization dependencies are available."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            self.plt = plt
            self.sns = sns
            self._matplotlib_available = True
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, visualizations will be disabled")
            self._matplotlib_available = False
    
    def _setup_style(self) -> None:
        """Setup matplotlib style and seaborn settings."""
        if not self._matplotlib_available:
            return
        
        try:
            self.plt.style.use(self.style)
            self.sns.set_palette("husl")
            self.sns.set_context("notebook", font_scale=1.1)
        except Exception as e:
            logger.debug(f"Failed to setup style: {e}")
    
    def generate_confusion_matrix(
        self,
        evaluation_result: EvaluationResults,
        normalize: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        Generate confusion matrix visualization.
        
        Args:
            evaluation_result: Evaluation results containing predictions
            normalize: Normalization method ('true', 'pred', 'all', or None)
            class_names: Names of classes for labels
            title: Custom title for the plot
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object or None if matplotlib not available
        """
        if not self._matplotlib_available:
            logger.warning("Matplotlib not available, cannot generate confusion matrix")
            return None
        
        try:
            from sklearn.metrics import confusion_matrix
            
            # Calculate confusion matrix
            cm = confusion_matrix(
                evaluation_result.ground_truth,
                evaluation_result.predictions,
                normalize=normalize
            )
            
            # Create figure
            fig, ax = self.plt.subplots(figsize=self.figsize)
            
            # Generate heatmap
            im = self.sns.heatmap(
                cm,
                annot=True,
                fmt='.2f' if normalize else 'd',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax
            )
            
            # Set labels and title
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            
            if title is None:
                title = f'Confusion Matrix - {evaluation_result.model_name}'
                if normalize:
                    title += f' (normalized by {normalize})'
            ax.set_title(title)
            
            # Adjust layout
            self.plt.tight_layout()
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            # Show if requested
            if show:
                self.plt.show()
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to generate confusion matrix: {str(e)}")
    
    def generate_roc_curve(
        self,
        evaluation_result: EvaluationResults,
        class_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        Generate ROC curve visualization.
        
        Args:
            evaluation_result: Evaluation results containing predictions and probabilities
            class_names: Names of classes for multi-class ROC
            title: Custom title for the plot
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object or None if matplotlib not available
        """
        if not self._matplotlib_available:
            logger.warning("Matplotlib not available, cannot generate ROC curve")
            return None
        
        if evaluation_result.probabilities is None:
            logger.warning("No probabilities available, cannot generate ROC curve")
            return None
        
        try:
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize
            
            y_true = evaluation_result.ground_truth
            y_proba = evaluation_result.probabilities
            
            # Create figure
            fig, ax = self.plt.subplots(figsize=self.figsize)
            
            # Determine if binary or multi-class
            unique_classes = np.unique(y_true)
            n_classes = len(unique_classes)
            
            if n_classes == 2:
                # Binary classification
                if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                    y_score = y_proba[:, 1]  # Probability of positive class
                else:
                    y_score = y_proba.ravel()
                
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, linewidth=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
            
            else:
                # Multi-class classification
                # Binarize the output
                y_true_bin = label_binarize(y_true, classes=unique_classes)
                
                # Compute ROC curve for each class
                colors = self.plt.cm.Set1(np.linspace(0, 1, n_classes))
                
                for i, color in zip(range(n_classes), colors):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    class_name = class_names[i] if class_names else f'Class {unique_classes[i]}'
                    ax.plot(fpr, tpr, color=color, linewidth=2,
                           label=f'{class_name} (AUC = {roc_auc:.3f})')
                
                # Compute micro-average ROC curve
                fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
                roc_auc_micro = auc(fpr_micro, tpr_micro)
                ax.plot(fpr_micro, tpr_micro, color='deeppink', linestyle='--', linewidth=2,
                       label=f'Micro-average (AUC = {roc_auc_micro:.3f})')
            
            # Plot diagonal line
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.8)
            
            # Set labels and title
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            
            if title is None:
                title = f'ROC Curve - {evaluation_result.model_name}'
            ax.set_title(title)
            
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            # Adjust layout
            self.plt.tight_layout()
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            # Show if requested
            if show:
                self.plt.show()
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to generate ROC curve: {str(e)}")
    
    def generate_precision_recall_curve(
        self,
        evaluation_result: EvaluationResults,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        Generate precision-recall curve visualization.
        
        Args:
            evaluation_result: Evaluation results containing predictions and probabilities
            title: Custom title for the plot
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object or None if matplotlib not available
        """
        if not self._matplotlib_available:
            logger.warning("Matplotlib not available, cannot generate PR curve")
            return None
        
        if evaluation_result.probabilities is None:
            logger.warning("No probabilities available, cannot generate PR curve")
            return None
        
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            
            y_true = evaluation_result.ground_truth
            y_proba = evaluation_result.probabilities
            
            # For binary classification
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                y_score = y_proba[:, 1]  # Probability of positive class
            else:
                y_score = y_proba.ravel()
            
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            avg_precision = average_precision_score(y_true, y_score)
            
            # Create figure
            fig, ax = self.plt.subplots(figsize=self.figsize)
            
            # Plot curve
            ax.plot(recall, precision, linewidth=2,
                   label=f'PR curve (AP = {avg_precision:.3f})')
            
            # Set labels and title
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            
            if title is None:
                title = f'Precision-Recall Curve - {evaluation_result.model_name}'
            ax.set_title(title)
            
            ax.legend(loc="lower left")
            ax.grid(True, alpha=0.3)
            
            # Adjust layout
            self.plt.tight_layout()
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            # Show if requested
            if show:
                self.plt.show()
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to generate precision-recall curve: {str(e)}")
    
    def generate_performance_plots(
        self,
        cv_results: CrossValidationResults,
        metrics_to_plot: Optional[List[str]] = None,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        Generate performance plots for cross-validation results.
        
        Args:
            cv_results: Cross-validation results
            metrics_to_plot: List of metrics to plot (plots all if None)
            title: Custom title for the plot
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object or None if matplotlib not available
        """
        if not self._matplotlib_available:
            logger.warning("Matplotlib not available, cannot generate performance plots")
            return None
        
        try:
            # Determine metrics to plot
            if metrics_to_plot is None:
                metrics_to_plot = list(cv_results.mean_metrics.keys())
            
            # Filter to available metrics
            available_metrics = [m for m in metrics_to_plot if m in cv_results.mean_metrics]
            
            if not available_metrics:
                logger.warning("No valid metrics found for plotting")
                return None
            
            # Create subplots
            n_metrics = len(available_metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = self.plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
            if n_metrics == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            # Plot each metric
            for i, metric_name in enumerate(available_metrics):
                ax = axes[i]
                
                # Extract fold scores for this metric
                fold_scores = []
                for fold_result in cv_results.fold_results:
                    metric_result = fold_result.metrics.get_metric(metric_name)
                    if metric_result:
                        fold_scores.append(metric_result.value)
                
                if not fold_scores:
                    continue
                
                # Create box plot
                box_plot = ax.boxplot([fold_scores], labels=[metric_name], patch_artist=True)
                box_plot['boxes'][0].set_facecolor('lightblue')
                
                # Add individual points
                x_pos = np.random.normal(1, 0.04, len(fold_scores))
                ax.scatter(x_pos, fold_scores, alpha=0.6, color='red', s=30)
                
                # Add mean line
                mean_score = cv_results.mean_metrics[metric_name]
                ax.axhline(y=mean_score, color='green', linestyle='--', alpha=0.8,
                          label=f'Mean: {mean_score:.3f}')
                
                # Set labels
                ax.set_ylabel('Score')
                ax.set_title(f'{metric_name}')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # Hide unused subplots
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)
            
            # Set main title
            if title is None:
                title = f'Cross-Validation Performance - {cv_results.model_name}'
            fig.suptitle(title, fontsize=16)
            
            # Adjust layout
            self.plt.tight_layout()
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            # Show if requested
            if show:
                self.plt.show()
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to generate performance plots: {str(e)}")
    
    def generate_model_comparison(
        self,
        results: Dict[str, Union[EvaluationResults, CrossValidationResults]],
        metric_name: str,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        Generate model comparison visualization.
        
        Args:
            results: Dictionary of model name -> evaluation/CV results
            metric_name: Name of metric to compare
            title: Custom title for the plot
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object or None if matplotlib not available
        """
        if not self._matplotlib_available:
            logger.warning("Matplotlib not available, cannot generate model comparison")
            return None
        
        try:
            # Extract scores and errors
            model_names = []
            scores = []
            errors = []
            
            for model_name, result in results.items():
                if isinstance(result, CrossValidationResults):
                    if metric_name in result.mean_metrics:
                        model_names.append(model_name)
                        scores.append(result.mean_metrics[metric_name])
                        errors.append(result.std_metrics[metric_name])
                elif isinstance(result, EvaluationResults):
                    metric_value = result.get_metric(metric_name)
                    if metric_value is not None:
                        model_names.append(model_name)
                        scores.append(metric_value)
                        errors.append(0)  # No error for single evaluation
            
            if not model_names:
                logger.warning(f"No results found for metric {metric_name}")
                return None
            
            # Create figure
            fig, ax = self.plt.subplots(figsize=self.figsize)
            
            # Create bar plot
            x_pos = np.arange(len(model_names))
            bars = ax.bar(x_pos, scores, yerr=errors, capsize=5, 
                         color='skyblue', edgecolor='navy', alpha=0.7)
            
            # Customize plot
            ax.set_xlabel('Models')
            ax.set_ylabel(metric_name)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            
            if title is None:
                title = f'Model Comparison - {metric_name}'
            ax.set_title(title)
            
            # Add value labels on bars
            for i, (bar, score, error) in enumerate(zip(bars, scores, errors)):
                height = bar.get_height()
                label = f'{score:.3f}'
                if error > 0:
                    label += f'±{error:.3f}'
                ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                       label, ha='center', va='bottom', fontsize=10)
            
            ax.grid(True, alpha=0.3, axis='y')
            
            # Adjust layout
            self.plt.tight_layout()
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            # Show if requested
            if show:
                self.plt.show()
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to generate model comparison: {str(e)}")
    
    def generate_learning_curve(
        self,
        train_scores: List[float],
        val_scores: List[float],
        train_sizes: Optional[List[int]] = None,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Optional[Any]:
        """
        Generate learning curve visualization.
        
        Args:
            train_scores: Training scores over epochs/iterations
            val_scores: Validation scores over epochs/iterations
            train_sizes: Training set sizes (if None, uses indices)
            title: Custom title for the plot
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure object or None if matplotlib not available
        """
        if not self._matplotlib_available:
            logger.warning("Matplotlib not available, cannot generate learning curve")
            return None
        
        try:
            # Create figure
            fig, ax = self.plt.subplots(figsize=self.figsize)
            
            # Use indices if train_sizes not provided
            if train_sizes is None:
                train_sizes = list(range(1, len(train_scores) + 1))
            
            # Plot curves
            ax.plot(train_sizes, train_scores, 'o-', color='blue', 
                   label='Training Score', linewidth=2, markersize=6)
            ax.plot(train_sizes, val_scores, 'o-', color='red', 
                   label='Validation Score', linewidth=2, markersize=6)
            
            # Customize plot
            ax.set_xlabel('Training Set Size' if train_sizes != list(range(1, len(train_scores) + 1)) else 'Epoch')
            ax.set_ylabel('Score')
            
            if title is None:
                title = 'Learning Curve'
            ax.set_title(title)
            
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Adjust layout
            self.plt.tight_layout()
            
            # Save if requested
            if save_path:
                self._save_figure(fig, save_path)
            
            # Show if requested
            if show:
                self.plt.show()
            
            return fig
            
        except Exception as e:
            raise EvaluationError(f"Failed to generate learning curve: {str(e)}")
    
    def _save_figure(self, fig: Any, save_path: Union[str, Path]) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib figure object
            save_path: Path to save the figure
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format from extension
        format_map = {
            '.png': 'png',
            '.jpg': 'jpeg',
            '.jpeg': 'jpeg',
            '.pdf': 'pdf',
            '.svg': 'svg',
            '.eps': 'eps'
        }
        
        file_format = format_map.get(save_path.suffix.lower(), 'png')
        
        # Save with high DPI for publication quality
        fig.savefig(save_path, format=file_format, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")


# Convenience functions
def generate_confusion_matrix(
    evaluation_result: EvaluationResults,
    normalize: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    **kwargs
) -> Optional[Any]:
    """
    Generate confusion matrix visualization.
    
    Args:
        evaluation_result: Evaluation results containing predictions
        normalize: Normalization method ('true', 'pred', 'all', or None)
        class_names: Names of classes for labels
        **kwargs: Additional arguments for VisualizationEngine
        
    Returns:
        Matplotlib figure object or None if matplotlib not available
    """
    engine = VisualizationEngine()
    return engine.generate_confusion_matrix(
        evaluation_result=evaluation_result,
        normalize=normalize,
        class_names=class_names,
        **kwargs
    )


def generate_roc_curve(
    evaluation_result: EvaluationResults,
    class_names: Optional[List[str]] = None,
    **kwargs
) -> Optional[Any]:
    """
    Generate ROC curve visualization.
    
    Args:
        evaluation_result: Evaluation results containing predictions and probabilities
        class_names: Names of classes for multi-class ROC
        **kwargs: Additional arguments for VisualizationEngine
        
    Returns:
        Matplotlib figure object or None if matplotlib not available
    """
    engine = VisualizationEngine()
    return engine.generate_roc_curve(
        evaluation_result=evaluation_result,
        class_names=class_names,
        **kwargs
    )


def generate_performance_plots(
    cv_results: CrossValidationResults,
    metrics_to_plot: Optional[List[str]] = None,
    **kwargs
) -> Optional[Any]:
    """
    Generate performance plots for cross-validation results.
    
    Args:
        cv_results: Cross-validation results
        metrics_to_plot: List of metrics to plot (plots all if None)
        **kwargs: Additional arguments for VisualizationEngine
        
    Returns:
        Matplotlib figure object or None if matplotlib not available
    """
    engine = VisualizationEngine()
    return engine.generate_performance_plots(
        cv_results=cv_results,
        metrics_to_plot=metrics_to_plot,
        **kwargs
    )