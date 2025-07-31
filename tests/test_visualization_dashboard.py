"""
Unit tests for visualization dashboard functionality.

Tests the interactive dashboard and live training monitoring components.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Mock the optional dependencies
dash_mock = MagicMock()
plotly_mock = MagicMock()

with patch.dict('sys.modules', {
    'dash': dash_mock,
    'dash.dcc': dash_mock.dcc,
    'dash.html': dash_mock.html,
    'plotly': plotly_mock,
    'plotly.graph_objects': plotly_mock.graph_objects,
    'plotly.subplots': plotly_mock.subplots
}):
    from neurolite.visualization.dashboard import (
        VisualizationDashboard,
        LiveTrainingMonitor
    )
    from neurolite.training.trainer import TrainingHistory
    from neurolite.evaluation.evaluator import EvaluationResults
    from neurolite.models.base import TaskType
    from neurolite.core.exceptions import VisualizationError


class TestVisualizationDashboard:
    """Test cases for VisualizationDashboard class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock Dash availability
        with patch('neurolite.visualization.dashboard.DASH_AVAILABLE', True):
            self.dashboard = VisualizationDashboard(port=8051, debug=True)
        
        # Create mock training history
        self.history = TrainingHistory()
        self.history.epochs = [1, 2, 3, 4, 5]
        self.history.train_loss = [1.0, 0.8, 0.6, 0.5, 0.4]
        self.history.val_loss = [1.2, 0.9, 0.7, 0.6, 0.5]
        self.history.metrics = {
            'accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
            'val_accuracy': [0.55, 0.65, 0.75, 0.8, 0.85]
        }
        
        # Create mock evaluation results
        self.results = Mock(spec=EvaluationResults)
        self.results.task_type = TaskType.BINARY_CLASSIFICATION
        self.results.ground_truth = [0, 1, 0, 1, 0]
        self.results.predictions = [0, 1, 1, 1, 0]
        self.results.probabilities = [[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]]
    
    @patch('neurolite.visualization.dashboard.DASH_AVAILABLE', False)
    def test_init_dash_not_available(self):
        """Test initialization when Dash is not available."""
        with pytest.raises(VisualizationError, match="Dash is required"):
            VisualizationDashboard()
    
    @patch('neurolite.visualization.dashboard.DASH_AVAILABLE', True)
    def test_init_with_parameters(self):
        """Test initialization with custom parameters."""
        dashboard = VisualizationDashboard(port=9000, host="0.0.0.0", debug=True)
        
        assert dashboard.port == 9000
        assert dashboard.host == "0.0.0.0"
        assert dashboard.debug is True
        assert dashboard.training_history is None
        assert dashboard.evaluation_results is None
        assert dashboard.is_training is False
    
    @patch('neurolite.visualization.dashboard.DASH_AVAILABLE', True)
    def test_setup_layout(self):
        """Test dashboard layout setup."""
        dashboard = VisualizationDashboard()
        
        # Check that app layout is set
        assert dashboard.app.layout is not None
        
        # Verify Dash components were called
        dash_mock.html.Div.assert_called()
        dash_mock.html.H1.assert_called()
        dash_mock.dcc.Graph.assert_called()
        dash_mock.dcc.Interval.assert_called()
    
    def test_update_training_history(self):
        """Test updating training history."""
        self.dashboard.update_training_history(self.history)
        
        assert self.dashboard.training_history == self.history
    
    def test_update_evaluation_results(self):
        """Test updating evaluation results."""
        self.dashboard.update_evaluation_results(self.results)
        
        assert self.dashboard.evaluation_results == self.results
    
    def test_set_training_status(self):
        """Test setting training status."""
        self.dashboard.set_training_status(True)
        assert self.dashboard.is_training is True
        
        self.dashboard.set_training_status(False)
        assert self.dashboard.is_training is False
    
    def test_create_loss_plot_with_data(self):
        """Test loss plot creation with data."""
        self.dashboard.update_training_history(self.history)
        
        with patch('neurolite.visualization.dashboard.go') as mock_go:
            mock_figure = Mock()
            mock_go.Figure.return_value = mock_figure
            
            result = self.dashboard._create_loss_plot()
            
            mock_go.Figure.assert_called_once()
            mock_figure.add_trace.assert_called()
            mock_figure.update_layout.assert_called()
            assert result == mock_figure
    
    def test_create_loss_plot_without_data(self):
        """Test loss plot creation without data."""
        with patch('neurolite.visualization.dashboard.go') as mock_go:
            mock_figure = Mock()
            mock_go.Figure.return_value = mock_figure
            
            result = self.dashboard._create_loss_plot()
            
            mock_go.Figure.assert_called_once()
            mock_figure.update_layout.assert_called_with(title="Training Loss")
            assert result == mock_figure
    
    def test_create_metrics_plot_with_data(self):
        """Test metrics plot creation with data."""
        self.dashboard.update_training_history(self.history)
        
        with patch('neurolite.visualization.dashboard.make_subplots') as mock_subplots:
            with patch('neurolite.visualization.dashboard.go') as mock_go:
                mock_figure = Mock()
                mock_subplots.return_value = mock_figure
                
                result = self.dashboard._create_metrics_plot()
                
                mock_subplots.assert_called_once()
                mock_figure.add_trace.assert_called()
                mock_figure.update_layout.assert_called()
                assert result == mock_figure
    
    def test_create_metrics_plot_without_data(self):
        """Test metrics plot creation without data."""
        with patch('neurolite.visualization.dashboard.go') as mock_go:
            mock_figure = Mock()
            mock_go.Figure.return_value = mock_figure
            
            result = self.dashboard._create_metrics_plot()
            
            mock_go.Figure.assert_called_once()
            mock_figure.update_layout.assert_called_with(title="Training Metrics")
            assert result == mock_figure
    
    def test_create_confusion_matrix_plot_with_data(self):
        """Test confusion matrix plot creation with data."""
        self.dashboard.update_evaluation_results(self.results)
        
        with patch('sklearn.metrics.confusion_matrix') as mock_cm:
            with patch('neurolite.visualization.dashboard.go') as mock_go:
                mock_cm.return_value = [[2, 1], [0, 2]]
                mock_figure = Mock()
                mock_go.Figure.return_value = mock_figure
                
                result = self.dashboard._create_confusion_matrix_plot()
                
                mock_go.Figure.assert_called_once()
                mock_figure.add_annotation.assert_called()
                mock_figure.update_layout.assert_called()
                assert result == mock_figure
    
    def test_create_confusion_matrix_plot_wrong_task(self):
        """Test confusion matrix plot with wrong task type."""
        self.results.task_type = TaskType.REGRESSION
        self.dashboard.update_evaluation_results(self.results)
        
        with patch('neurolite.visualization.dashboard.go') as mock_go:
            mock_figure = Mock()
            mock_go.Figure.return_value = mock_figure
            
            result = self.dashboard._create_confusion_matrix_plot()
            
            mock_figure.update_layout.assert_called_with(
                title="Confusion Matrix (Not available for this task)"
            )
    
    def test_create_roc_curve_plot_with_data(self):
        """Test ROC curve plot creation with data."""
        self.dashboard.update_evaluation_results(self.results)
        
        with patch('sklearn.metrics.roc_curve') as mock_roc:
            with patch('sklearn.metrics.auc') as mock_auc:
                with patch('neurolite.visualization.dashboard.go') as mock_go:
                    mock_roc.return_value = ([0, 0.5, 1], [0, 0.8, 1], [0.5, 0.3, 0.1])
                    mock_auc.return_value = 0.85
                    mock_figure = Mock()
                    mock_go.Figure.return_value = mock_figure
                    
                    result = self.dashboard._create_roc_curve_plot()
                    
                    mock_go.Figure.assert_called_once()
                    mock_figure.add_trace.assert_called()
                    mock_figure.update_layout.assert_called()
                    assert result == mock_figure
    
    def test_create_roc_curve_plot_wrong_task(self):
        """Test ROC curve plot with wrong task type."""
        self.results.task_type = TaskType.MULTICLASS_CLASSIFICATION
        self.dashboard.update_evaluation_results(self.results)
        
        with patch('neurolite.visualization.dashboard.go') as mock_go:
            mock_figure = Mock()
            mock_go.Figure.return_value = mock_figure
            
            result = self.dashboard._create_roc_curve_plot()
            
            mock_figure.update_layout.assert_called_with(
                title="ROC Curve (Not available for this task)"
            )
    
    @patch('webbrowser.open')
    @patch('threading.Thread')
    def test_start_server_with_browser(self, mock_thread, mock_browser):
        """Test starting server with browser opening."""
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        with patch.object(self.dashboard.app, 'run_server') as mock_run:
            self.dashboard.start_server(open_browser=True)
            
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()
            mock_run.assert_called_once_with(
                host=self.dashboard.host,
                port=self.dashboard.port,
                debug=self.dashboard.debug,
                use_reloader=False
            )
    
    def test_start_server_without_browser(self):
        """Test starting server without browser opening."""
        with patch.object(self.dashboard.app, 'run_server') as mock_run:
            with patch('threading.Thread') as mock_thread:
                self.dashboard.start_server(open_browser=False)
                
                mock_thread.assert_not_called()
                mock_run.assert_called_once()
    
    def test_start_server_error(self):
        """Test server start error handling."""
        with patch.object(self.dashboard.app, 'run_server') as mock_run:
            mock_run.side_effect = Exception("Server failed")
            
            with pytest.raises(VisualizationError, match="Dashboard server failed to start"):
                self.dashboard.start_server(open_browser=False)
    
    def test_export_dashboard_data(self):
        """Test dashboard data export."""
        import tempfile
        import json
        
        self.dashboard.update_training_history(self.history)
        self.dashboard.update_evaluation_results(self.results)
        self.dashboard.set_training_status(True)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_path = f.name
        
        try:
            self.dashboard.export_dashboard_data(output_path)
            
            # Verify file was created and contains expected data
            assert os.path.exists(output_path)
            
            with open(output_path, 'r') as f:
                data = json.load(f)
                
                assert data["is_training"] is True
                assert "training_history" in data
                assert "evaluation_results" in data
                assert "timestamp" in data
        
        finally:
            os.unlink(output_path)


class TestLiveTrainingMonitor:
    """Test cases for LiveTrainingMonitor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dashboard = Mock(spec=VisualizationDashboard)
        self.monitor = LiveTrainingMonitor(self.mock_dashboard)
        
        # Create mock training history
        self.history = TrainingHistory()
        self.history.epochs = [1, 2, 3]
        self.history.train_loss = [1.0, 0.8, 0.6]
    
    def test_init(self):
        """Test monitor initialization."""
        assert self.monitor.dashboard == self.mock_dashboard
        assert self.monitor.is_monitoring is False
        assert self.monitor.monitor_thread is None
    
    def test_start_monitoring(self):
        """Test starting monitoring."""
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance
            
            self.monitor.start_monitoring(self.history)
            
            assert self.monitor.is_monitoring is True
            self.mock_dashboard.set_training_status.assert_called_with(True)
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()
    
    def test_start_monitoring_already_running(self):
        """Test starting monitoring when already running."""
        self.monitor.is_monitoring = True
        
        with patch('threading.Thread') as mock_thread:
            self.monitor.start_monitoring(self.history)
            
            # Should not create new thread
            mock_thread.assert_not_called()
    
    def test_stop_monitoring(self):
        """Test stopping monitoring."""
        # Set up monitoring state
        self.monitor.is_monitoring = True
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        self.monitor.monitor_thread = mock_thread
        
        self.monitor.stop_monitoring()
        
        assert self.monitor.is_monitoring is False
        self.mock_dashboard.set_training_status.assert_called_with(False)
        mock_thread.join.assert_called_with(timeout=1.0)
    
    def test_stop_monitoring_no_thread(self):
        """Test stopping monitoring when no thread exists."""
        self.monitor.is_monitoring = True
        self.monitor.monitor_thread = None
        
        self.monitor.stop_monitoring()
        
        assert self.monitor.is_monitoring is False
        self.mock_dashboard.set_training_status.assert_called_with(False)
    
    @patch('time.sleep')
    def test_monitor_training_loop(self, mock_sleep):
        """Test monitoring training loop."""
        # Set up monitoring state
        self.monitor.is_monitoring = True
        
        # Mock sleep to control loop iterations
        call_count = 0
        def sleep_side_effect(duration):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:  # Stop after 3 iterations
                self.monitor.is_monitoring = False
        
        mock_sleep.side_effect = sleep_side_effect
        
        # Run monitoring loop
        self.monitor._monitor_training(self.history)
        
        # Verify dashboard was updated
        assert self.mock_dashboard.update_training_history.call_count >= 3
        assert mock_sleep.call_count >= 3
    
    @patch('time.sleep')
    def test_monitor_training_with_error(self, mock_sleep):
        """Test monitoring training loop with error."""
        # Set up monitoring state
        self.monitor.is_monitoring = True
        
        # Make dashboard update raise an exception
        self.mock_dashboard.update_training_history.side_effect = Exception("Update failed")
        
        # Run monitoring loop (should exit on error)
        self.monitor._monitor_training(self.history)
        
        # Verify it tried to update at least once
        self.mock_dashboard.update_training_history.assert_called()


class TestDashboardIntegration:
    """Test cases for dashboard integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('neurolite.visualization.dashboard.DASH_AVAILABLE', True):
            self.dashboard = VisualizationDashboard(port=8052)
        self.monitor = LiveTrainingMonitor(self.dashboard)
    
    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        history = TrainingHistory()
        history.epochs = [1, 2]
        history.train_loss = [1.0, 0.8]
        
        # Start monitoring
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance
            
            self.monitor.start_monitoring(history)
            
            assert self.monitor.is_monitoring is True
            assert self.dashboard.is_training is True
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        assert self.monitor.is_monitoring is False
        assert self.dashboard.is_training is False
    
    def test_dashboard_callback_simulation(self):
        """Test dashboard callback behavior simulation."""
        # Set up data
        history = TrainingHistory()
        history.epochs = [1, 2, 3]
        history.train_loss = [1.0, 0.8, 0.6]
        history.val_loss = [1.2, 0.9, 0.7]
        history.metrics = {'accuracy': [0.6, 0.7, 0.8]}
        
        results = Mock(spec=EvaluationResults)
        results.task_type = TaskType.BINARY_CLASSIFICATION
        results.ground_truth = [0, 1, 0, 1]
        results.predictions = [0, 1, 1, 1]
        results.probabilities = [[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]]
        
        # Update dashboard
        self.dashboard.update_training_history(history)
        self.dashboard.update_evaluation_results(results)
        self.dashboard.set_training_status(True)
        
        # Test plot creation methods
        with patch('neurolite.visualization.dashboard.go'):
            loss_fig = self.dashboard._create_loss_plot()
            metrics_fig = self.dashboard._create_metrics_plot()
            
            assert loss_fig is not None
            assert metrics_fig is not None
        
        with patch('sklearn.metrics.confusion_matrix'):
            with patch('neurolite.visualization.dashboard.go'):
                cm_fig = self.dashboard._create_confusion_matrix_plot()
                assert cm_fig is not None
        
        with patch('sklearn.metrics.roc_curve'):
            with patch('sklearn.metrics.auc'):
                with patch('neurolite.visualization.dashboard.go'):
                    roc_fig = self.dashboard._create_roc_curve_plot()
                    assert roc_fig is not None


class TestErrorHandling:
    """Test cases for error handling in dashboard functionality."""
    
    def test_dashboard_init_no_dash(self):
        """Test dashboard initialization without Dash."""
        with patch('neurolite.visualization.dashboard.DASH_AVAILABLE', False):
            with pytest.raises(VisualizationError, match="Dash is required"):
                VisualizationDashboard()
    
    def test_server_start_failure(self):
        """Test server start failure handling."""
        with patch('neurolite.visualization.dashboard.DASH_AVAILABLE', True):
            dashboard = VisualizationDashboard()
            
            with patch.object(dashboard.app, 'run_server') as mock_run:
                mock_run.side_effect = Exception("Port already in use")
                
                with pytest.raises(VisualizationError, match="Dashboard server failed to start"):
                    dashboard.start_server(open_browser=False)
    
    def test_monitoring_thread_error_handling(self):
        """Test error handling in monitoring thread."""
        mock_dashboard = Mock(spec=VisualizationDashboard)
        monitor = LiveTrainingMonitor(mock_dashboard)
        
        history = TrainingHistory()
        
        # Make dashboard update fail
        mock_dashboard.update_training_history.side_effect = Exception("Dashboard error")
        
        # Set monitoring state
        monitor.is_monitoring = True
        
        # Run monitoring (should handle error gracefully)
        with patch('time.sleep'):
            monitor._monitor_training(history)
        
        # Should have tried to update at least once
        mock_dashboard.update_training_history.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])