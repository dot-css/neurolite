"""
Unit tests for evaluation visualization engine.

Tests the VisualizationEngine and related functionality for generating
confusion matrices, ROC curves, and performance plots.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from neurolite.evaluation.visualizer import (
    VisualizationEngine, generate_confusion_matrix, generate_roc_curve, generate_performance_plots
)
from neurolite.evaluation.evaluator import EvaluationResults
from neurolite.evaluation.validator import CrossValidationResults
from neurolite.evaluation.metrics import MetricsCollection, MetricResult
from neurolite.models.base import TaskType
from neurolite.data.detector import DataType
from neurolite.core.exceptions import EvaluationError


class TestVisualizationEngine:
    """Test VisualizationEngine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock matplotlib to avoid import issues in testing
        self.mock_plt = Mock()
        self.mock_sns = Mock()
        
        with patch('neurolite.evaluation.visualizer.plt', self.mock_plt), \
             patch('neurolite.evaluation.visualizer.sns', self.mock_sns):
            self.engine = VisualizationEngine()
            self.engine.plt = self.mock_plt
            self.engine.sns = self.mock_sns
            self.engine._matplotlib_available = True
        
        # Create test evaluation results
        metrics = {
            'accuracy': MetricResult('Accuracy', 0.85, 'Test accuracy'),
            'precision': MetricResult('Precision', 0.80, 'Test precision'),
            'recall': MetricResult('Recall', 0.90, 'Test recall')
        }
        
        self.metrics_collection = MetricsCollection(
            task_type=TaskType.BINARY_CLASSIFICATION,
            metrics=metrics,
            primary_metric='accuracy'
        )
        
        self.evaluation_result = EvaluationResults(
            metrics=self.metrics_collection,
            predictions=np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0]),
            ground_truth=np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0]),
            probabilities=np.array([
                [0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1], [0.2, 0.8],
                [0.4, 0.6], [0.1, 0.9], [0.3, 0.7], [0.7, 0.3], [0.8, 0.2]
            ]),
            model_name="TestModel"
        )
    
    def test_init_without_matplotlib(self):
        """Test initialization without matplotlib."""
        with patch('neurolite.evaluation.visualizer.plt', side_effect=ImportError):
            engine = VisualizationEngine()
            assert not engine._matplotlib_available
    
    def test_check_dependencies(self):
        """Test dependency checking."""
        with patch('neurolite.evaluation.visualizer.plt') as mock_plt, \
             patch('neurolite.evaluation.visualizer.sns') as mock_sns:
            engine = VisualizationEngine()
            assert engine._matplotlib_available
            assert engine.plt == mock_plt
            assert engine.sns == mock_sns
    
    @patch('sklearn.metrics.confusion_matrix')
    def test_generate_confusion_matrix(self, mock_cm):
        """Test confusion matrix generation."""
        mock_cm.return_value = np.array([[4, 1], [2, 3]])
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        self.mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = self.engine.generate_confusion_matrix(
            evaluation_result=self.evaluation_result,
            show=False
        )
        
        assert result == mock_fig
        mock_cm.assert_called_once()
        self.mock_sns.heatmap.assert_called_once()
        mock_ax.set_xlabel.assert_called_with('Predicted Label')
        mock_ax.set_ylabel.assert_called_with('True Label')
    
    def test_generate_confusion_matrix_no_matplotlib(self):
        """Test confusion matrix generation without matplotlib."""
        self.engine._matplotlib_available = False
        
        result = self.engine.generate_confusion_matrix(
            evaluation_result=self.evaluation_result
        )
        
        assert result is None
    
    @patch('sklearn.metrics.roc_curve')
    @patch('sklearn.metrics.auc')
    def test_generate_roc_curve_binary(self, mock_auc, mock_roc_curve):
        """Test ROC curve generation for binary classification."""
        mock_roc_curve.return_value = (
            np.array([0, 0.2, 0.4, 1.0]),  # fpr
            np.array([0, 0.6, 0.8, 1.0]),  # tpr
            np.array([0.9, 0.7, 0.3, 0.1])  # thresholds
        )
        mock_auc.return_value = 0.85
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        self.mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = self.engine.generate_roc_curve(
            evaluation_result=self.evaluation_result,
            show=False
        )
        
        assert result == mock_fig
        mock_roc_curve.assert_called_once()
        mock_auc.assert_called_once()
        mock_ax.plot.assert_called()
    
    def test_generate_roc_curve_no_probabilities(self):
        """Test ROC curve generation without probabilities."""
        # Remove probabilities
        self.evaluation_result.probabilities = None
        
        result = self.engine.generate_roc_curve(
            evaluation_result=self.evaluation_result,
            show=False
        )
        
        assert result is None
    
    @patch('sklearn.metrics.precision_recall_curve')
    @patch('sklearn.metrics.average_precision_score')
    def test_generate_precision_recall_curve(self, mock_ap, mock_pr_curve):
        """Test precision-recall curve generation."""
        mock_pr_curve.return_value = (
            np.array([1.0, 0.8, 0.6, 0.4]),  # precision
            np.array([0.0, 0.4, 0.6, 0.8]),  # recall
            np.array([0.9, 0.7, 0.3, 0.1])   # thresholds
        )
        mock_ap.return_value = 0.75
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        self.mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = self.engine.generate_precision_recall_curve(
            evaluation_result=self.evaluation_result,
            show=False
        )
        
        assert result == mock_fig
        mock_pr_curve.assert_called_once()
        mock_ap.assert_called_once()
        mock_ax.plot.assert_called()
    
    def test_generate_performance_plots(self):
        """Test performance plots generation for cross-validation results."""
        # Create mock cross-validation results
        fold_results = []
        for i in range(3):
            metrics = {
                'accuracy': MetricResult('Accuracy', 0.8 + i * 0.05, 'Test accuracy'),
                'precision': MetricResult('Precision', 0.75 + i * 0.05, 'Test precision')
            }
            
            metrics_collection = MetricsCollection(
                task_type=TaskType.BINARY_CLASSIFICATION,
                metrics=metrics,
                primary_metric='accuracy'
            )
            
            fold_result = EvaluationResults(
                metrics=metrics_collection,
                predictions=np.array([0, 1, 0, 1]),
                ground_truth=np.array([0, 1, 1, 1])
            )
            
            fold_results.append(fold_result)
        
        cv_results = CrossValidationResults(
            fold_results=fold_results,
            mean_metrics={'accuracy': 0.825, 'precision': 0.775},
            std_metrics={'accuracy': 0.025, 'precision': 0.025},
            num_folds=3,
            task_type=TaskType.BINARY_CLASSIFICATION,
            data_type=DataType.TABULAR,
            model_name="TestModel"
        )
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        self.mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        result = self.engine.generate_performance_plots(
            cv_results=cv_results,
            show=False
        )
        
        assert result == mock_fig
        self.mock_plt.subplots.assert_called_once()
    
    def test_generate_performance_plots_no_metrics(self):
        """Test performance plots with no valid metrics."""
        cv_results = Mock()
        cv_results.mean_metrics = {}
        
        result = self.engine.generate_performance_plots(
            cv_results=cv_results,
            show=False
        )
        
        assert result is None
    
    def test_generate_model_comparison(self):
        """Test model comparison visualization."""
        # Create mock results
        result1 = Mock()
        result1.get_metric.return_value = 0.85
        
        result2 = Mock()
        result2.get_metric.return_value = 0.90
        
        results = {'Model1': result1, 'Model2': result2}
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_bars = [Mock(), Mock()]
        self.mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_ax.bar.return_value = mock_bars
        
        result = self.engine.generate_model_comparison(
            results=results,
            metric_name='accuracy',
            show=False
        )
        
        assert result == mock_fig
        mock_ax.bar.assert_called_once()
    
    def test_generate_model_comparison_no_results(self):
        """Test model comparison with no valid results."""
        result1 = Mock()
        result1.get_metric.return_value = None  # No metric available
        
        results = {'Model1': result1}
        
        result = self.engine.generate_model_comparison(
            results=results,
            metric_name='accuracy',
            show=False
        )
        
        assert result is None
    
    def test_generate_learning_curve(self):
        """Test learning curve generation."""
        train_scores = [0.6, 0.7, 0.8, 0.85, 0.87]
        val_scores = [0.55, 0.65, 0.75, 0.78, 0.80]
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        self.mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = self.engine.generate_learning_curve(
            train_scores=train_scores,
            val_scores=val_scores,
            show=False
        )
        
        assert result == mock_fig
        assert mock_ax.plot.call_count == 2  # Two lines (train and val)
    
    def test_generate_learning_curve_with_sizes(self):
        """Test learning curve generation with custom training sizes."""
        train_scores = [0.6, 0.7, 0.8]
        val_scores = [0.55, 0.65, 0.75]
        train_sizes = [100, 200, 300]
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        self.mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = self.engine.generate_learning_curve(
            train_scores=train_scores,
            val_scores=val_scores,
            train_sizes=train_sizes,
            show=False
        )
        
        assert result == mock_fig
        # Verify that custom train_sizes were used
        call_args = mock_ax.plot.call_args_list
        assert len(call_args) == 2
        assert call_args[0][0][0] == train_sizes  # First plot call
    
    def test_save_figure(self):
        """Test saving figure to file."""
        mock_fig = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_plot.png"
            
            self.engine._save_figure(mock_fig, save_path)
            
            mock_fig.savefig.assert_called_once_with(
                save_path, format='png', dpi=300, bbox_inches='tight'
            )
    
    def test_save_figure_different_formats(self):
        """Test saving figure in different formats."""
        mock_fig = Mock()
        
        formats = {
            '.png': 'png',
            '.jpg': 'jpeg',
            '.pdf': 'pdf',
            '.svg': 'svg'
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for ext, expected_format in formats.items():
                save_path = Path(temp_dir) / f"test_plot{ext}"
                
                self.engine._save_figure(mock_fig, save_path)
                
                mock_fig.savefig.assert_called_with(
                    save_path, format=expected_format, dpi=300, bbox_inches='tight'
                )
    
    def test_visualization_with_exception(self):
        """Test visualization with exception during generation."""
        self.mock_plt.subplots.side_effect = Exception("Plotting failed")
        
        with pytest.raises(EvaluationError):
            self.engine.generate_confusion_matrix(
                evaluation_result=self.evaluation_result,
                show=False
            )


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test evaluation results
        metrics = {
            'accuracy': MetricResult('Accuracy', 0.85, 'Test accuracy')
        }
        
        self.metrics_collection = MetricsCollection(
            task_type=TaskType.BINARY_CLASSIFICATION,
            metrics=metrics,
            primary_metric='accuracy'
        )
        
        self.evaluation_result = EvaluationResults(
            metrics=self.metrics_collection,
            predictions=np.array([0, 1, 1, 0, 1]),
            ground_truth=np.array([0, 1, 0, 0, 1]),
            probabilities=np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1], [0.2, 0.8]]),
            model_name="TestModel"
        )
    
    @patch('neurolite.evaluation.visualizer.VisualizationEngine')
    def test_generate_confusion_matrix_function(self, mock_engine_class):
        """Test generate_confusion_matrix convenience function."""
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        mock_engine.generate_confusion_matrix.return_value = Mock()
        
        result = generate_confusion_matrix(
            evaluation_result=self.evaluation_result,
            normalize='true'
        )
        
        mock_engine.generate_confusion_matrix.assert_called_once_with(
            evaluation_result=self.evaluation_result,
            normalize='true'
        )
    
    @patch('neurolite.evaluation.visualizer.VisualizationEngine')
    def test_generate_roc_curve_function(self, mock_engine_class):
        """Test generate_roc_curve convenience function."""
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        mock_engine.generate_roc_curve.return_value = Mock()
        
        result = generate_roc_curve(
            evaluation_result=self.evaluation_result,
            class_names=['Class 0', 'Class 1']
        )
        
        mock_engine.generate_roc_curve.assert_called_once_with(
            evaluation_result=self.evaluation_result,
            class_names=['Class 0', 'Class 1']
        )
    
    @patch('neurolite.evaluation.visualizer.VisualizationEngine')
    def test_generate_performance_plots_function(self, mock_engine_class):
        """Test generate_performance_plots convenience function."""
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        mock_engine.generate_performance_plots.return_value = Mock()
        
        cv_results = Mock()
        
        result = generate_performance_plots(
            cv_results=cv_results,
            metrics_to_plot=['accuracy', 'precision']
        )
        
        mock_engine.generate_performance_plots.assert_called_once_with(
            cv_results=cv_results,
            metrics_to_plot=['accuracy', 'precision']
        )


if __name__ == '__main__':
    pytest.main([__file__])