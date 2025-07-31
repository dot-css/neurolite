"""
Unit tests for visualization plotting functionality.

Tests the core plotting classes and convenience functions for training,
performance, and data visualization.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, List, Optional

# Mock the optional dependencies
matplotlib_mock = MagicMock()
plotly_mock = MagicMock()
seaborn_mock = MagicMock()

with patch.dict('sys.modules', {
    'matplotlib': matplotlib_mock,
    'matplotlib.pyplot': matplotlib_mock.pyplot,
    'seaborn': seaborn_mock,
    'plotly': plotly_mock,
    'plotly.graph_objects': plotly_mock.graph_objects,
    'plotly.express': plotly_mock.express,
    'plotly.subplots': plotly_mock.subplots,
    'sklearn.metrics': MagicMock()
}):
    from neurolite.visualization.plots import (
        TrainingPlotter,
        PerformancePlotter,
        DataPlotter,
        plot_training_history,
        plot_confusion_matrix,
        plot_roc_curve,
        plot_data_distribution
    )
    from neurolite.training.trainer import TrainingHistory
    from neurolite.evaluation.evaluator import EvaluationResults
    from neurolite.models.base import TaskType
    from neurolite.data.detector import DataType
    from neurolite.core.exceptions import VisualizationError


class TestTrainingPlotter:
    """Test cases for TrainingPlotter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plotter_matplotlib = TrainingPlotter(backend="matplotlib")
        self.plotter_plotly = TrainingPlotter(backend="plotly")
        
        # Create mock training history
        self.history = TrainingHistory()
        self.history.epochs = [1, 2, 3, 4, 5]
        self.history.train_loss = [1.0, 0.8, 0.6, 0.5, 0.4]
        self.history.val_loss = [1.2, 0.9, 0.7, 0.6, 0.5]
        self.history.metrics = {
            'accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
            'val_accuracy': [0.55, 0.65, 0.75, 0.8, 0.85],
            'f1_score': [0.5, 0.6, 0.7, 0.75, 0.8]
        }
    
    def test_init_valid_backend(self):
        """Test initialization with valid backends."""
        plotter_mpl = TrainingPlotter(backend="matplotlib")
        assert plotter_mpl.backend == "matplotlib"
        
        plotter_plotly = TrainingPlotter(backend="plotly")
        assert plotter_plotly.backend == "plotly"
    
    def test_init_invalid_backend(self):
        """Test initialization with invalid backend."""
        with pytest.raises(VisualizationError):
            TrainingPlotter(backend="invalid")
    
    def test_plot_training_history_empty(self):
        """Test plotting with empty history."""
        empty_history = TrainingHistory()
        
        with pytest.raises(VisualizationError):
            self.plotter_matplotlib.plot_training_history(empty_history)
    
    @patch('neurolite.visualization.plots.plt')
    def test_plot_training_history_matplotlib(self, mock_plt):
        """Test matplotlib training history plotting."""
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        result = self.plotter_matplotlib.plot_training_history(self.history)
        
        # Verify matplotlib calls
        mock_plt.subplots.assert_called_once()
        assert result == mock_fig
        
        # Verify plot calls on axes
        for ax in mock_axes:
            ax.plot.assert_called()
            ax.set_xlabel.assert_called()
            ax.set_ylabel.assert_called()
            ax.set_title.assert_called()
            ax.legend.assert_called()
            ax.grid.assert_called()
    
    @patch('neurolite.visualization.plots.make_subplots')
    @patch('neurolite.visualization.plots.go')
    def test_plot_training_history_plotly(self, mock_go, mock_make_subplots):
        """Test plotly training history plotting."""
        mock_fig = Mock()
        mock_make_subplots.return_value = mock_fig
        
        result = self.plotter_plotly.plot_training_history(self.history)
        
        # Verify plotly calls
        mock_make_subplots.assert_called_once()
        mock_fig.add_trace.assert_called()
        mock_fig.update_layout.assert_called()
        assert result == mock_fig
    
    def test_plot_training_history_with_save(self):
        """Test plotting with save path."""
        with patch('neurolite.visualization.plots.plt') as mock_plt:
            mock_fig = Mock()
            mock_plt.subplots.return_value = (mock_fig, [Mock(), Mock()])
            
            self.plotter_matplotlib.plot_training_history(
                self.history, 
                save_path="test_plot.png"
            )
            
            mock_plt.savefig.assert_called_with(
                "test_plot.png", 
                dpi=300, 
                bbox_inches='tight'
            )
    
    def test_plot_training_history_specific_metrics(self):
        """Test plotting with specific metrics."""
        with patch('neurolite.visualization.plots.plt') as mock_plt:
            mock_fig = Mock()
            mock_plt.subplots.return_value = (mock_fig, [Mock(), Mock()])
            
            self.plotter_matplotlib.plot_training_history(
                self.history, 
                metrics=['accuracy']
            )
            
            # Should only plot loss + accuracy (2 subplots)
            mock_plt.subplots.assert_called_with(1, 2, figsize=(12, 8))
    
    @patch('neurolite.visualization.plots.go')
    def test_plot_live_training_plotly(self, mock_go):
        """Test live training plot creation."""
        mock_fig = Mock()
        mock_go.FigureWidget.return_value = mock_fig
        
        result = self.plotter_plotly.plot_live_training(self.history)
        
        mock_go.FigureWidget.assert_called_once()
        mock_fig.add_trace.assert_called()
        mock_fig.update_layout.assert_called()
        assert result == mock_fig
    
    def test_plot_live_training_matplotlib_warning(self):
        """Test live training with matplotlib shows warning."""
        with patch('neurolite.visualization.plots.logger') as mock_logger:
            result = self.plotter_matplotlib.plot_live_training(self.history)
            
            mock_logger.warning.assert_called_with(
                "Live plotting not fully supported with matplotlib backend"
            )


class TestPerformancePlotter:
    """Test cases for PerformancePlotter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plotter = PerformancePlotter(backend="matplotlib")
        
        # Create mock evaluation results
        self.results = Mock(spec=EvaluationResults)
        self.results.task_type = TaskType.BINARY_CLASSIFICATION
        self.results.ground_truth = np.array([0, 1, 0, 1, 0])
        self.results.predictions = np.array([0, 1, 1, 1, 0])
        self.results.probabilities = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.4, 0.6], 
            [0.2, 0.8], [0.9, 0.1]
        ])
    
    def test_init_valid_backend(self):
        """Test initialization with valid backend."""
        plotter = PerformancePlotter(backend="plotly")
        assert plotter.backend == "plotly"
    
    def test_init_invalid_backend(self):
        """Test initialization with invalid backend."""
        with pytest.raises(VisualizationError):
            PerformancePlotter(backend="invalid")
    
    @patch('sklearn.metrics.confusion_matrix')
    @patch('neurolite.visualization.plots.sns')
    @patch('neurolite.visualization.plots.plt')
    def test_plot_confusion_matrix_matplotlib(self, mock_plt, mock_sns, mock_cm):
        """Test matplotlib confusion matrix plotting."""
        mock_cm.return_value = np.array([[2, 1], [0, 2]])
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = self.plotter.plot_confusion_matrix(self.results)
        
        mock_cm.assert_called_once()
        mock_sns.heatmap.assert_called_once()
        mock_ax.set_xlabel.assert_called_with('Predicted Label')
        mock_ax.set_ylabel.assert_called_with('True Label')
        assert result == mock_fig
    
    def test_plot_confusion_matrix_wrong_task_type(self):
        """Test confusion matrix with wrong task type."""
        self.results.task_type = TaskType.REGRESSION
        
        with pytest.raises(VisualizationError):
            self.plotter.plot_confusion_matrix(self.results)
    
    @patch('sklearn.metrics.roc_curve')
    @patch('sklearn.metrics.auc')
    @patch('neurolite.visualization.plots.plt')
    def test_plot_roc_curve_matplotlib(self, mock_plt, mock_auc, mock_roc):
        """Test matplotlib ROC curve plotting."""
        mock_roc.return_value = ([0, 0.5, 1], [0, 0.8, 1], [0.5, 0.3, 0.1])
        mock_auc.return_value = 0.85
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = self.plotter.plot_roc_curve(self.results)
        
        mock_roc.assert_called_once()
        mock_auc.assert_called_once()
        mock_ax.plot.assert_called()
        assert result == mock_fig
    
    def test_plot_roc_curve_wrong_task_type(self):
        """Test ROC curve with wrong task type."""
        self.results.task_type = TaskType.MULTICLASS_CLASSIFICATION
        
        with pytest.raises(VisualizationError):
            self.plotter.plot_roc_curve(self.results)
    
    def test_plot_roc_curve_no_probabilities(self):
        """Test ROC curve without probabilities."""
        self.results.probabilities = None
        
        with pytest.raises(VisualizationError):
            self.plotter.plot_roc_curve(self.results)


class TestDataPlotter:
    """Test cases for DataPlotter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plotter = DataPlotter(backend="matplotlib")
        
        # Create test data
        self.tabular_data = np.random.randn(100, 3)
        self.image_data = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        self.text_data = ["hello world", "test text", "sample data"] * 10
    
    def test_init_valid_backend(self):
        """Test initialization with valid backend."""
        plotter = DataPlotter(backend="plotly")
        assert plotter.backend == "plotly"
    
    @patch('neurolite.visualization.plots.plt')
    def test_plot_tabular_distribution(self, mock_plt):
        """Test tabular data distribution plotting."""
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        with patch('pandas.DataFrame') as mock_df:
            mock_df_instance = Mock()
            mock_df_instance.select_dtypes.return_value.columns = ['col1', 'col2', 'col3']
            mock_df.return_value = mock_df_instance
            
            result = self.plotter.plot_data_distribution(
                self.tabular_data, 
                DataType.TABULAR,
                columns=['col1', 'col2', 'col3']
            )
            
            mock_plt.subplots.assert_called_once()
            assert result == mock_fig
    
    @patch('neurolite.visualization.plots.plt')
    def test_plot_image_distribution(self, mock_plt):
        """Test image data distribution plotting."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = self.plotter.plot_data_distribution(
            self.image_data, 
            DataType.IMAGE
        )
        
        mock_ax.hist.assert_called_once()
        mock_ax.set_title.assert_called_with('Pixel Intensity Distribution')
        assert result == mock_fig
    
    @patch('neurolite.visualization.plots.plt')
    def test_plot_text_distribution(self, mock_plt):
        """Test text data distribution plotting."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = self.plotter.plot_data_distribution(
            self.text_data, 
            DataType.TEXT
        )
        
        mock_ax.hist.assert_called_once()
        mock_ax.set_title.assert_called_with('Text Length Distribution')
        assert result == mock_fig
    
    def test_plot_unsupported_data_type(self):
        """Test plotting with unsupported data type."""
        with pytest.raises(VisualizationError):
            self.plotter.plot_data_distribution(
                self.tabular_data, 
                DataType.AUDIO
            )


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.history = TrainingHistory()
        self.history.epochs = [1, 2, 3]
        self.history.train_loss = [1.0, 0.8, 0.6]
        self.history.val_loss = [1.2, 0.9, 0.7]
        
        self.results = Mock(spec=EvaluationResults)
        self.results.task_type = TaskType.BINARY_CLASSIFICATION
        self.results.ground_truth = np.array([0, 1, 0, 1])
        self.results.predictions = np.array([0, 1, 1, 1])
        self.results.probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])
    
    @patch('neurolite.visualization.plots.TrainingPlotter')
    def test_plot_training_history_function(self, mock_plotter_class):
        """Test plot_training_history convenience function."""
        mock_plotter = Mock()
        mock_plotter_class.return_value = mock_plotter
        mock_plotter.plot_training_history.return_value = "mock_figure"
        
        result = plot_training_history(self.history, backend="matplotlib")
        
        mock_plotter_class.assert_called_with(backend="matplotlib")
        mock_plotter.plot_training_history.assert_called_with(self.history)
        assert result == "mock_figure"
    
    @patch('neurolite.visualization.plots.PerformancePlotter')
    def test_plot_confusion_matrix_function(self, mock_plotter_class):
        """Test plot_confusion_matrix convenience function."""
        mock_plotter = Mock()
        mock_plotter_class.return_value = mock_plotter
        mock_plotter.plot_confusion_matrix.return_value = "mock_figure"
        
        result = plot_confusion_matrix(self.results, backend="plotly")
        
        mock_plotter_class.assert_called_with(backend="plotly")
        mock_plotter.plot_confusion_matrix.assert_called_with(self.results)
        assert result == "mock_figure"
    
    @patch('neurolite.visualization.plots.PerformancePlotter')
    def test_plot_roc_curve_function(self, mock_plotter_class):
        """Test plot_roc_curve convenience function."""
        mock_plotter = Mock()
        mock_plotter_class.return_value = mock_plotter
        mock_plotter.plot_roc_curve.return_value = "mock_figure"
        
        result = plot_roc_curve(self.results, backend="matplotlib")
        
        mock_plotter_class.assert_called_with(backend="matplotlib")
        mock_plotter.plot_roc_curve.assert_called_with(self.results)
        assert result == "mock_figure"
    
    @patch('neurolite.visualization.plots.DataPlotter')
    def test_plot_data_distribution_function(self, mock_plotter_class):
        """Test plot_data_distribution convenience function."""
        mock_plotter = Mock()
        mock_plotter_class.return_value = mock_plotter
        mock_plotter.plot_data_distribution.return_value = "mock_figure"
        
        data = np.random.randn(100, 3)
        result = plot_data_distribution(data, DataType.TABULAR, backend="plotly")
        
        mock_plotter_class.assert_called_with(backend="plotly")
        mock_plotter.plot_data_distribution.assert_called_with(data, DataType.TABULAR)
        assert result == "mock_figure"


class TestErrorHandling:
    """Test cases for error handling in visualization."""
    
    def test_invalid_backend_error(self):
        """Test error handling for invalid backends."""
        with pytest.raises(VisualizationError, match="Unsupported backend"):
            TrainingPlotter(backend="invalid_backend")
        
        with pytest.raises(VisualizationError, match="Unsupported backend"):
            PerformancePlotter(backend="invalid_backend")
        
        with pytest.raises(VisualizationError, match="Unsupported backend"):
            DataPlotter(backend="invalid_backend")
    
    def test_empty_data_error(self):
        """Test error handling for empty data."""
        empty_history = TrainingHistory()
        plotter = TrainingPlotter()
        
        with pytest.raises(VisualizationError, match="Training history is empty"):
            plotter.plot_training_history(empty_history)
    
    def test_wrong_task_type_error(self):
        """Test error handling for wrong task types."""
        results = Mock(spec=EvaluationResults)
        results.task_type = TaskType.REGRESSION
        plotter = PerformancePlotter()
        
        with pytest.raises(VisualizationError, match="only available for classification"):
            plotter.plot_confusion_matrix(results)
    
    def test_missing_probabilities_error(self):
        """Test error handling for missing probabilities."""
        results = Mock(spec=EvaluationResults)
        results.task_type = TaskType.BINARY_CLASSIFICATION
        results.probabilities = None
        plotter = PerformancePlotter()
        
        with pytest.raises(VisualizationError, match="requires prediction probabilities"):
            plotter.plot_roc_curve(results)


if __name__ == "__main__":
    pytest.main([__file__])