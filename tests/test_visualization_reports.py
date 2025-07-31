"""
Unit tests for visualization reporting functionality.

Tests the report generation system for training, evaluation, and data analysis.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from dataclasses import dataclass
from typing import Dict, Any

# Mock the optional dependencies
jinja2_mock = MagicMock()
matplotlib_mock = MagicMock()
plotly_mock = MagicMock()

with patch.dict('sys.modules', {
    'jinja2': jinja2_mock,
    'matplotlib': matplotlib_mock,
    'matplotlib.pyplot': matplotlib_mock.pyplot,
    'plotly': plotly_mock,
    'plotly.graph_objects': plotly_mock.graph_objects,
    'plotly.io': plotly_mock.io,
    'weasyprint': MagicMock()
}):
    from neurolite.visualization.reports import (
        ReportGenerator,
        create_default_templates
    )
    from neurolite.training.trainer import TrainingHistory
    from neurolite.evaluation.evaluator import EvaluationResults
    from neurolite.models.base import TaskType
    from neurolite.data.detector import DataType
    from neurolite.core.exceptions import VisualizationError


class TestReportGenerator:
    """Test cases for ReportGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ReportGenerator(template_dir=self.temp_dir)
        
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
        self.results.model_name = "test_model"
        self.results.prediction_time = 0.1
        self.results.samples_per_second = 100.0
        self.results.ground_truth = [0, 1, 0, 1, 0]
        self.results.predictions = [0, 1, 1, 1, 0]
        self.results.probabilities = [[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]]
        
        # Mock metrics
        mock_metrics = Mock()
        mock_metrics.accuracy = 0.85
        mock_metrics.f1_score = 0.82
        mock_metrics.precision = 0.80
        mock_metrics.recall = 0.85
        self.results.metrics = mock_metrics
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_with_template_dir(self):
        """Test initialization with custom template directory."""
        generator = ReportGenerator(template_dir=self.temp_dir)
        assert generator.template_dir == self.temp_dir
    
    def test_init_without_template_dir(self):
        """Test initialization without template directory."""
        generator = ReportGenerator()
        assert generator.template_dir is not None
        assert os.path.exists(generator.template_dir)
    
    @patch('neurolite.visualization.reports.JINJA2_AVAILABLE', False)
    def test_init_without_jinja2(self):
        """Test initialization without Jinja2."""
        generator = ReportGenerator()
        assert generator.env is None
    
    def test_prepare_training_report_data(self):
        """Test training report data preparation."""
        data = self.generator._prepare_training_report_data(
            self.history, self.results, {"model": "test"}, {"data": "test"}
        )
        
        assert data["title"] == "Training Report"
        assert "generated_at" in data
        assert "training_summary" in data
        assert "performance_summary" in data
        assert data["model_info"] == {"model": "test"}
        assert data["data_info"] == {"data": "test"}
    
    def test_prepare_evaluation_report_data(self):
        """Test evaluation report data preparation."""
        data = self.generator._prepare_evaluation_report_data(
            self.results, {"model": "test"}, {"data": "test"}
        )
        
        assert data["title"] == "Evaluation Report"
        assert "generated_at" in data
        assert "performance_summary" in data
        assert data["model_info"] == {"model": "test"}
        assert data["data_info"] == {"data": "test"}
    
    def test_prepare_data_report_data(self):
        """Test data report data preparation."""
        test_data = [[1, 2, 3], [4, 5, 6]]
        data = self.generator._prepare_data_report_data(
            test_data, DataType.TABULAR, {"info": "test"}
        )
        
        assert data["title"] == "Data Analysis Report"
        assert "generated_at" in data
        assert "data_summary" in data
        assert data["data_type"] == DataType.TABULAR.value
        assert data["data_info"] == {"info": "test"}
    
    def test_summarize_training(self):
        """Test training summary generation."""
        summary = self.generator._summarize_training(self.history)
        
        assert summary["total_epochs"] == 5
        assert summary["final_train_loss"] == 0.4
        assert summary["final_val_loss"] == 0.5
        assert summary["best_train_loss"] == 0.4
        assert summary["best_val_loss"] == 0.5
        assert "accuracy" in summary["metrics"]
        assert summary["metrics"]["accuracy"]["final"] == 0.9
        assert summary["metrics"]["accuracy"]["best"] == 0.9
    
    def test_summarize_training_empty(self):
        """Test training summary with empty history."""
        empty_history = TrainingHistory()
        summary = self.generator._summarize_training(empty_history)
        
        assert summary["status"] == "No training data available"
    
    def test_summarize_performance(self):
        """Test performance summary generation."""
        summary = self.generator._summarize_performance(self.results)
        
        assert summary["task_type"] == TaskType.BINARY_CLASSIFICATION.value
        assert summary["model_name"] == "test_model"
        assert summary["prediction_time"] == 0.1
        assert summary["samples_per_second"] == 100.0
        assert summary["metrics"]["accuracy"] == 0.85
        assert summary["metrics"]["f1_score"] == 0.82
    
    def test_summarize_data_tabular(self):
        """Test data summary for tabular data."""
        test_data = [[1, 2, 3], [4, 5, 6]]
        summary = self.generator._summarize_data(test_data, DataType.TABULAR)
        
        assert summary["data_type"] == DataType.TABULAR.value
        assert summary["size"] == 2
    
    @patch('neurolite.visualization.reports.pd')
    def test_summarize_data_pandas(self, mock_pd):
        """Test data summary with pandas DataFrame."""
        mock_df = Mock()
        mock_df.columns = ['col1', 'col2']
        mock_df.dtypes.to_dict.return_value = {'col1': 'int64', 'col2': 'float64'}
        mock_df.isnull.return_value.sum.return_value.to_dict.return_value = {'col1': 0, 'col2': 1}
        mock_df.select_dtypes.return_value.columns = ['col1', 'col2']
        
        mock_pd.DataFrame = Mock(return_value=mock_df)
        
        summary = self.generator._summarize_data(mock_df, DataType.TABULAR)
        
        assert summary["data_type"] == DataType.TABULAR.value
        assert summary["columns"] == ['col1', 'col2']
        assert summary["dtypes"] == {'col1': 'int64', 'col2': 'float64'}
    
    @patch.object(ReportGenerator, '_generate_training_visualizations')
    @patch.object(ReportGenerator, '_generate_html_report')
    def test_generate_training_report_html(self, mock_html, mock_viz):
        """Test HTML training report generation."""
        mock_viz.return_value = {"plot1": "viz_data"}
        mock_html.return_value = "report.html"
        
        result = self.generator.generate_training_report(
            self.history, self.results, output_path="test_report.html", format="html"
        )
        
        mock_viz.assert_called_once()
        mock_html.assert_called_once()
        assert result == "report.html"
    
    @patch.object(ReportGenerator, '_generate_json_report')
    def test_generate_training_report_json(self, mock_json):
        """Test JSON training report generation."""
        mock_json.return_value = "report.json"
        
        result = self.generator.generate_training_report(
            self.history, output_path="test_report.json", format="json"
        )
        
        mock_json.assert_called_once()
        assert result == "report.json"
    
    def test_generate_training_report_unsupported_format(self):
        """Test training report with unsupported format."""
        with pytest.raises(VisualizationError, match="Unsupported report format"):
            self.generator.generate_training_report(
                self.history, output_path="test.xyz", format="xyz"
            )
    
    @patch.object(ReportGenerator, '_generate_evaluation_visualizations')
    @patch.object(ReportGenerator, '_generate_html_report')
    def test_generate_evaluation_report(self, mock_html, mock_viz):
        """Test evaluation report generation."""
        mock_viz.return_value = {"plot1": "viz_data"}
        mock_html.return_value = "report.html"
        
        result = self.generator.generate_evaluation_report(
            self.results, output_path="eval_report.html"
        )
        
        mock_viz.assert_called_once()
        mock_html.assert_called_once()
        assert result == "report.html"
    
    @patch.object(ReportGenerator, '_generate_data_visualizations')
    @patch.object(ReportGenerator, '_generate_html_report')
    def test_generate_data_report(self, mock_html, mock_viz):
        """Test data analysis report generation."""
        mock_viz.return_value = {"plot1": "viz_data"}
        mock_html.return_value = "report.html"
        
        test_data = [[1, 2, 3], [4, 5, 6]]
        result = self.generator.generate_data_report(
            test_data, DataType.TABULAR, output_path="data_report.html"
        )
        
        mock_viz.assert_called_once()
        mock_html.assert_called_once()
        assert result == "report.html"
    
    @patch('neurolite.visualization.reports.TrainingPlotter')
    def test_generate_training_visualizations(self, mock_plotter_class):
        """Test training visualization generation."""
        mock_plotter = Mock()
        mock_plotter_class.return_value = mock_plotter
        mock_plotter.plot_training_history.return_value = Mock()
        
        with patch.object(self.generator, '_figure_to_base64', return_value="base64_data"):
            visualizations = self.generator._generate_training_visualizations(
                self.history, self.results
            )
            
            assert "training_history" in visualizations
            assert visualizations["training_history"] == "base64_data"
    
    @patch('neurolite.visualization.reports.PerformancePlotter')
    def test_generate_evaluation_visualizations(self, mock_plotter_class):
        """Test evaluation visualization generation."""
        mock_plotter = Mock()
        mock_plotter_class.return_value = mock_plotter
        mock_plotter.plot_confusion_matrix.return_value = Mock()
        mock_plotter.plot_roc_curve.return_value = Mock()
        
        with patch.object(self.generator, '_figure_to_base64', return_value="base64_data"):
            visualizations = self.generator._generate_evaluation_visualizations(self.results)
            
            assert "confusion_matrix" in visualizations
            assert "roc_curve" in visualizations
    
    @patch('neurolite.visualization.reports.DataPlotter')
    def test_generate_data_visualizations(self, mock_plotter_class):
        """Test data visualization generation."""
        mock_plotter = Mock()
        mock_plotter_class.return_value = mock_plotter
        mock_plotter.plot_data_distribution.return_value = Mock()
        
        with patch.object(self.generator, '_figure_to_base64', return_value="base64_data"):
            test_data = [[1, 2, 3], [4, 5, 6]]
            visualizations = self.generator._generate_data_visualizations(
                test_data, DataType.TABULAR
            )
            
            assert "data_distribution" in visualizations
            assert visualizations["data_distribution"] == "base64_data"
    
    @patch('neurolite.visualization.reports.PLOTLY_AVAILABLE', True)
    def test_figure_to_base64_plotly(self):
        """Test figure to base64 conversion for Plotly."""
        mock_figure = Mock()
        mock_figure.to_html.return_value = "<div>plotly html</div>"
        
        result = self.generator._figure_to_base64(mock_figure)
        
        mock_figure.to_html.assert_called_once()
        assert result == "<div>plotly html</div>"
    
    @patch('neurolite.visualization.reports.MATPLOTLIB_AVAILABLE', True)
    @patch('neurolite.visualization.reports.plt')
    def test_figure_to_base64_matplotlib(self, mock_plt):
        """Test figure to base64 conversion for matplotlib."""
        mock_figure = Mock()
        mock_figure.savefig = Mock()
        
        # Mock BytesIO
        with patch('neurolite.visualization.reports.BytesIO') as mock_bytesio:
            mock_buffer = Mock()
            mock_buffer.getvalue.return_value = b'fake_image_data'
            mock_bytesio.return_value = mock_buffer
            
            with patch('base64.b64encode') as mock_b64:
                mock_b64.return_value = b'encoded_data'
                
                result = self.generator._figure_to_base64(mock_figure)
                
                mock_figure.savefig.assert_called_once()
                mock_plt.close.assert_called_once()
                assert result == "data:image/png;base64,encoded_data"
    
    def test_generate_json_report(self):
        """Test JSON report generation."""
        report_data = {
            "title": "Test Report",
            "data": {"key": "value"},
            "visualizations": {"plot": "data"}  # Should be removed
        }
        
        output_path = os.path.join(self.temp_dir, "test.json")
        
        result = self.generator._generate_json_report(report_data, output_path)
        
        assert result == output_path
        assert os.path.exists(output_path)
        
        # Check that visualizations were removed
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
            assert "visualizations" not in saved_data
            assert saved_data["title"] == "Test Report"
    
    def test_generate_basic_html_report(self):
        """Test basic HTML report generation."""
        report_data = {
            "title": "Test Report",
            "generated_at": "2023-01-01",
            "training_summary": {
                "total_epochs": 5,
                "final_train_loss": 0.4,
                "metrics": {"accuracy": {"final": 0.9, "best": 0.9}}
            },
            "visualizations": {
                "plot1": "data:image/png;base64,fake_data"
            }
        }
        
        html_content = self.generator._generate_basic_html_report(report_data)
        
        assert "<title>Test Report</title>" in html_content
        assert "Generated at: 2023-01-01" in html_content
        assert "Training Summary" in html_content
        assert "Visualizations" in html_content
        assert "data:image/png;base64,fake_data" in html_content
    
    @patch('neurolite.visualization.reports.JINJA2_AVAILABLE', True)
    def test_generate_html_report_with_template(self):
        """Test HTML report generation with Jinja2 template."""
        # Create a simple template
        template_content = "<html><body><h1>{{ title }}</h1></body></html>"
        template_path = os.path.join(self.temp_dir, "test_template.html")
        
        with open(template_path, 'w') as f:
            f.write(template_content)
        
        # Mock Jinja2 environment
        mock_template = Mock()
        mock_template.render.return_value = "<html><body><h1>Test Report</h1></body></html>"
        
        self.generator.env = Mock()
        self.generator.env.get_template.return_value = mock_template
        
        report_data = {"title": "Test Report"}
        output_path = os.path.join(self.temp_dir, "output.html")
        
        result = self.generator._generate_html_report(
            report_data, output_path, "test_template.html"
        )
        
        assert result == output_path
        assert os.path.exists(output_path)
        
        with open(output_path, 'r') as f:
            content = f.read()
            assert "<h1>Test Report</h1>" in content
    
    @patch('weasyprint.HTML')
    def test_generate_pdf_report(self, mock_weasyprint):
        """Test PDF report generation."""
        mock_html = Mock()
        mock_weasyprint.return_value = mock_html
        
        report_data = {"title": "Test Report"}
        output_path = os.path.join(self.temp_dir, "test.pdf")
        
        result = self.generator._generate_pdf_report(report_data, output_path)
        
        mock_weasyprint.assert_called_once()
        mock_html.write_pdf.assert_called_with(output_path)
        assert result == output_path
    
    def test_generate_pdf_report_no_weasyprint(self):
        """Test PDF report generation without WeasyPrint."""
        with patch('neurolite.visualization.reports.weasyprint', side_effect=ImportError):
            report_data = {"title": "Test Report"}
            output_path = os.path.join(self.temp_dir, "test.pdf")
            
            with pytest.raises(VisualizationError, match="PDF generation requires WeasyPrint"):
                self.generator._generate_pdf_report(report_data, output_path)
    
    def test_format_training_summary_html(self):
        """Test training summary HTML formatting."""
        summary = {
            "total_epochs": 5,
            "final_train_loss": 0.4,
            "metrics": {"accuracy": {"final": 0.9, "best": 0.95}}
        }
        
        html_parts = self.generator._format_training_summary_html(summary)
        
        html_content = "\n".join(html_parts)
        assert "Training Summary" in html_content
        assert "5" in html_content  # total_epochs
        assert "0.4" in html_content  # final_train_loss
        assert "accuracy" in html_content
        assert "0.9" in html_content  # final accuracy
        assert "0.95" in html_content  # best accuracy
    
    def test_format_performance_summary_html(self):
        """Test performance summary HTML formatting."""
        summary = {
            "task_type": "binary_classification",
            "model_name": "test_model",
            "metrics": {"accuracy": 0.85, "f1_score": 0.82}
        }
        
        html_parts = self.generator._format_performance_summary_html(summary)
        
        html_content = "\n".join(html_parts)
        assert "Performance Summary" in html_content
        assert "binary_classification" in html_content
        assert "test_model" in html_content
        assert "0.8500" in html_content  # accuracy formatted
        assert "0.8200" in html_content  # f1_score formatted
    
    def test_format_data_summary_html(self):
        """Test data summary HTML formatting."""
        summary = {
            "data_type": "tabular",
            "shape": [100, 5],
            "size": 100
        }
        
        html_parts = self.generator._format_data_summary_html(summary)
        
        html_content = "\n".join(html_parts)
        assert "Data Summary" in html_content
        assert "tabular" in html_content
        assert "[100, 5]" in html_content
        assert "100" in html_content


class TestCreateDefaultTemplates:
    """Test cases for default template creation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('neurolite.visualization.reports.Path')
    def test_create_default_templates(self, mock_path):
        """Test default template creation."""
        mock_template_dir = Mock()
        mock_template_dir.mkdir = Mock()
        mock_path.return_value.parent = Mock()
        mock_path.return_value.parent.__truediv__ = Mock(return_value=mock_template_dir)
        
        with patch('builtins.open', mock_open()) as mock_file:
            create_default_templates()
            
            mock_template_dir.mkdir.assert_called_with(exist_ok=True)
            # Should create 3 template files
            assert mock_file.call_count == 3


class TestErrorHandling:
    """Test cases for error handling in report generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ReportGenerator(template_dir=self.temp_dir)
        self.history = TrainingHistory()
        self.history.epochs = [1, 2, 3]
        self.history.train_loss = [1.0, 0.8, 0.6]
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_unsupported_format_error(self):
        """Test error handling for unsupported formats."""
        with pytest.raises(VisualizationError, match="Unsupported report format"):
            self.generator.generate_training_report(
                self.history, output_path="test.xyz", format="xyz"
            )
    
    def test_visualization_generation_error(self):
        """Test error handling in visualization generation."""
        with patch.object(self.generator, 'training_plotter') as mock_plotter:
            mock_plotter.plot_training_history.side_effect = Exception("Plot failed")
            
            # Should not raise exception, just log warning
            visualizations = self.generator._generate_training_visualizations(
                self.history, None
            )
            
            # Should return empty dict or handle gracefully
            assert isinstance(visualizations, dict)
    
    def test_figure_to_base64_error(self):
        """Test error handling in figure conversion."""
        mock_figure = Mock()
        mock_figure.to_html.side_effect = Exception("Conversion failed")
        
        result = self.generator._figure_to_base64(mock_figure)
        
        # Should return empty string on error
        assert result == ""


if __name__ == "__main__":
    pytest.main([__file__])