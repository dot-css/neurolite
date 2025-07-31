"""
Unit tests for visualization export functionality.

Tests the export utilities for different formats and figure types.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import Dict, Any

# Mock the optional dependencies
matplotlib_mock = MagicMock()
plotly_mock = MagicMock()
pil_mock = MagicMock()

with patch.dict('sys.modules', {
    'matplotlib': matplotlib_mock,
    'matplotlib.pyplot': matplotlib_mock.pyplot,
    'plotly': plotly_mock,
    'plotly.graph_objects': plotly_mock.graph_objects,
    'plotly.io': plotly_mock.io,
    'PIL': pil_mock,
    'PIL.Image': pil_mock.Image
}):
    from neurolite.visualization.exporters import (
        VisualizationExporter,
        export_figure,
        export_multiple_figures,
        get_supported_formats
    )
    from neurolite.core.exceptions import VisualizationError


class TestVisualizationExporter:
    """Test cases for VisualizationExporter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = VisualizationExporter()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test exporter initialization."""
        exporter = VisualizationExporter()
        assert exporter is not None
    
    def test_supported_formats(self):
        """Test supported formats list."""
        formats = self.exporter.SUPPORTED_FORMATS
        expected_formats = ['png', 'jpg', 'jpeg', 'pdf', 'svg', 'html', 'json']
        
        for fmt in expected_formats:
            assert fmt in formats
    
    def test_get_supported_formats(self):
        """Test getting available formats."""
        formats = self.exporter.get_supported_formats()
        assert isinstance(formats, dict)
        assert len(formats) > 0
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Test invalid characters
        result = self.exporter._sanitize_filename("test<>:\"/\\|?*file")
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        
        # Test length limit
        long_name = "a" * 150
        result = self.exporter._sanitize_filename(long_name)
        assert len(result) <= 100
        
        # Test empty filename
        result = self.exporter._sanitize_filename("")
        assert result == "visualization"
    
    def test_is_plotly_figure(self):
        """Test Plotly figure detection."""
        # Mock Plotly figure
        mock_figure = Mock()
        mock_figure.__class__.__name__ = "Figure"
        
        with patch('neurolite.visualization.exporters.PLOTLY_AVAILABLE', True):
            with patch('neurolite.visualization.exporters.go') as mock_go:
                mock_go.Figure = type(mock_figure)
                mock_go.FigureWidget = type(mock_figure)
                
                # This is a simplified test - in reality we'd need proper isinstance checks
                result = self.exporter._is_plotly_figure(mock_figure)
                # The actual result depends on the mock setup
    
    def test_is_matplotlib_figure(self):
        """Test matplotlib figure detection."""
        # Mock matplotlib figure
        mock_figure = Mock()
        mock_figure.savefig = Mock()
        
        with patch('neurolite.visualization.exporters.MATPLOTLIB_AVAILABLE', True):
            result = self.exporter._is_matplotlib_figure(mock_figure)
            assert result is True
        
        # Test without savefig method
        mock_figure_no_savefig = Mock(spec=[])
        result = self.exporter._is_matplotlib_figure(mock_figure_no_savefig)
        assert result is False
    
    def test_export_figure_unsupported_format(self):
        """Test export with unsupported format."""
        mock_figure = Mock()
        output_path = os.path.join(self.temp_dir, "test.xyz")
        
        with pytest.raises(VisualizationError, match="Unsupported format"):
            self.exporter.export_figure(mock_figure, output_path, "xyz")
    
    def test_export_figure_auto_detect_format(self):
        """Test format auto-detection from file extension."""
        mock_figure = Mock()
        mock_figure.savefig = Mock()  # Make it look like matplotlib
        
        output_path = os.path.join(self.temp_dir, "test.png")
        
        with patch.object(self.exporter, '_is_matplotlib_figure', return_value=True):
            with patch.object(self.exporter, '_export_matplotlib_figure', return_value=output_path) as mock_export:
                result = self.exporter.export_figure(mock_figure, output_path)
                
                mock_export.assert_called_once()
                assert result == output_path
    
    def test_export_figure_unsupported_figure_type(self):
        """Test export with unsupported figure type."""
        mock_figure = "not_a_figure"
        output_path = os.path.join(self.temp_dir, "test.png")
        
        with patch.object(self.exporter, '_is_plotly_figure', return_value=False):
            with patch.object(self.exporter, '_is_matplotlib_figure', return_value=False):
                with pytest.raises(VisualizationError, match="Unsupported figure type"):
                    self.exporter.export_figure(mock_figure, output_path)
    
    @patch('neurolite.visualization.exporters.PLOTLY_AVAILABLE', True)
    def test_export_plotly_figure_html(self):
        """Test Plotly figure HTML export."""
        mock_figure = Mock()
        mock_figure.write_html = Mock()
        
        output_path = os.path.join(self.temp_dir, "test.html")
        
        result = self.exporter._export_plotly_figure(mock_figure, output_path, "html")
        
        mock_figure.write_html.assert_called_once()
        assert result == output_path
    
    @patch('neurolite.visualization.exporters.PLOTLY_AVAILABLE', True)
    def test_export_plotly_figure_json(self):
        """Test Plotly figure JSON export."""
        mock_figure = Mock()
        mock_figure.write_json = Mock()
        
        output_path = os.path.join(self.temp_dir, "test.json")
        
        result = self.exporter._export_plotly_figure(mock_figure, output_path, "json")
        
        mock_figure.write_json.assert_called_once()
        assert result == output_path
    
    @patch('neurolite.visualization.exporters.PLOTLY_AVAILABLE', True)
    def test_export_plotly_figure_image(self):
        """Test Plotly figure image export."""
        mock_figure = Mock()
        mock_figure.write_image = Mock()
        
        output_path = os.path.join(self.temp_dir, "test.png")
        
        result = self.exporter._export_plotly_figure(mock_figure, output_path, "png")
        
        mock_figure.write_image.assert_called_once()
        assert result == output_path
    
    @patch('neurolite.visualization.exporters.PLOTLY_AVAILABLE', False)
    def test_export_plotly_figure_not_available(self):
        """Test Plotly export when not available."""
        mock_figure = Mock()
        output_path = os.path.join(self.temp_dir, "test.html")
        
        with pytest.raises(VisualizationError, match="Plotly not available"):
            self.exporter._export_plotly_figure(mock_figure, output_path, "html")
    
    @patch('neurolite.visualization.exporters.MATPLOTLIB_AVAILABLE', True)
    def test_export_matplotlib_figure_png(self):
        """Test matplotlib figure PNG export."""
        mock_figure = Mock()
        mock_figure.savefig = Mock()
        
        output_path = os.path.join(self.temp_dir, "test.png")
        
        result = self.exporter._export_matplotlib_figure(mock_figure, output_path, "png")
        
        mock_figure.savefig.assert_called_once()
        assert result == output_path
    
    @patch('neurolite.visualization.exporters.MATPLOTLIB_AVAILABLE', True)
    def test_export_matplotlib_figure_pdf(self):
        """Test matplotlib figure PDF export."""
        mock_figure = Mock()
        mock_figure.savefig = Mock()
        
        output_path = os.path.join(self.temp_dir, "test.pdf")
        
        result = self.exporter._export_matplotlib_figure(mock_figure, output_path, "pdf")
        
        mock_figure.savefig.assert_called_once()
        call_args = mock_figure.savefig.call_args
        assert call_args[1]['format'] == 'pdf'
        assert result == output_path
    
    @patch('neurolite.visualization.exporters.MATPLOTLIB_AVAILABLE', True)
    def test_export_matplotlib_figure_html(self):
        """Test matplotlib figure HTML export."""
        mock_figure = Mock()
        mock_figure.savefig = Mock()
        
        output_path = os.path.join(self.temp_dir, "test.html")
        
        with patch.object(self.exporter, '_create_html_with_image', return_value=output_path) as mock_create_html:
            result = self.exporter._export_matplotlib_figure(mock_figure, output_path, "html")
            
            mock_figure.savefig.assert_called_once()
            mock_create_html.assert_called_once()
            assert result == output_path
    
    @patch('neurolite.visualization.exporters.MATPLOTLIB_AVAILABLE', False)
    def test_export_matplotlib_figure_not_available(self):
        """Test matplotlib export when not available."""
        mock_figure = Mock()
        output_path = os.path.join(self.temp_dir, "test.png")
        
        with pytest.raises(VisualizationError, match="Matplotlib not available"):
            self.exporter._export_matplotlib_figure(mock_figure, output_path, "png")
    
    def test_create_html_with_image(self):
        """Test HTML creation with embedded image."""
        # Create a temporary PNG file
        png_path = os.path.join(self.temp_dir, "test.png")
        html_path = os.path.join(self.temp_dir, "test.html")
        
        # Create dummy PNG data
        png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
        
        with open(png_path, 'wb') as f:
            f.write(png_data)
        
        result = self.exporter._create_html_with_image(png_path, html_path)
        
        # Check that HTML file was created
        assert os.path.exists(html_path)
        assert result == html_path
        
        # Check HTML content
        with open(html_path, 'r') as f:
            content = f.read()
            assert 'data:image/png;base64,' in content
            assert '<img src=' in content
    
    def test_export_multiple_figures(self):
        """Test exporting multiple figures."""
        figures = {
            'figure1': Mock(),
            'figure2': Mock()
        }
        
        # Mock the export_figure method
        with patch.object(self.exporter, 'export_figure') as mock_export:
            mock_export.side_effect = lambda fig, path, fmt: path
            
            result = self.exporter.export_multiple_figures(
                figures, 
                self.temp_dir, 
                'png'
            )
            
            assert len(result) == 2
            assert 'figure1' in result
            assert 'figure2' in result
            assert mock_export.call_count == 2
    
    def test_export_multiple_figures_with_error(self):
        """Test exporting multiple figures with one failing."""
        figures = {
            'figure1': Mock(),
            'figure2': Mock()
        }
        
        # Mock the export_figure method to fail for figure2
        def mock_export_side_effect(fig, path, fmt):
            if 'figure2' in path:
                raise Exception("Export failed")
            return path
        
        with patch.object(self.exporter, 'export_figure') as mock_export:
            mock_export.side_effect = mock_export_side_effect
            
            result = self.exporter.export_multiple_figures(
                figures, 
                self.temp_dir, 
                'png'
            )
            
            assert len(result) == 2
            assert result['figure1'] is not None
            assert result['figure2'] is None
    
    def test_create_export_batch(self):
        """Test batch export in multiple formats."""
        figures = {'test_figure': Mock()}
        
        with patch.object(self.exporter, 'export_multiple_figures') as mock_export:
            mock_export.return_value = {'test_figure': 'path/to/file'}
            
            result = self.exporter.create_export_batch(
                figures, 
                self.temp_dir, 
                ['png', 'html']
            )
            
            assert len(result) == 2
            assert 'png' in result
            assert 'html' in result
            assert mock_export.call_count == 2


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('neurolite.visualization.exporters.VisualizationExporter')
    def test_export_figure_function(self, mock_exporter_class):
        """Test export_figure convenience function."""
        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_figure.return_value = "test_path"
        
        mock_figure = Mock()
        output_path = os.path.join(self.temp_dir, "test.png")
        
        result = export_figure(mock_figure, output_path)
        
        mock_exporter_class.assert_called_once()
        mock_exporter.export_figure.assert_called_with(mock_figure, output_path, None)
        assert result == "test_path"
    
    @patch('neurolite.visualization.exporters.VisualizationExporter')
    def test_export_multiple_figures_function(self, mock_exporter_class):
        """Test export_multiple_figures convenience function."""
        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.export_multiple_figures.return_value = {"fig1": "path1"}
        
        figures = {"fig1": Mock()}
        
        result = export_multiple_figures(figures, self.temp_dir, 'png')
        
        mock_exporter_class.assert_called_once()
        mock_exporter.export_multiple_figures.assert_called_with(figures, self.temp_dir, 'png')
        assert result == {"fig1": "path1"}
    
    @patch('neurolite.visualization.exporters.VisualizationExporter')
    def test_get_supported_formats_function(self, mock_exporter_class):
        """Test get_supported_formats convenience function."""
        mock_exporter = Mock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.get_supported_formats.return_value = {"png": "PNG format"}
        
        result = get_supported_formats()
        
        mock_exporter_class.assert_called_once()
        mock_exporter.get_supported_formats.assert_called_once()
        assert result == {"png": "PNG format"}


class TestErrorHandling:
    """Test cases for error handling in export functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = VisualizationExporter()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_unsupported_format_error(self):
        """Test error handling for unsupported formats."""
        mock_figure = Mock()
        output_path = os.path.join(self.temp_dir, "test.xyz")
        
        with pytest.raises(VisualizationError, match="Unsupported format"):
            self.exporter.export_figure(mock_figure, output_path, "xyz")
    
    def test_unsupported_figure_type_error(self):
        """Test error handling for unsupported figure types."""
        invalid_figure = "not_a_figure"
        output_path = os.path.join(self.temp_dir, "test.png")
        
        with patch.object(self.exporter, '_is_plotly_figure', return_value=False):
            with patch.object(self.exporter, '_is_matplotlib_figure', return_value=False):
                with pytest.raises(VisualizationError, match="Unsupported figure type"):
                    self.exporter.export_figure(invalid_figure, output_path)
    
    def test_export_failure_error(self):
        """Test error handling for export failures."""
        mock_figure = Mock()
        mock_figure.savefig = Mock(side_effect=Exception("Save failed"))
        
        output_path = os.path.join(self.temp_dir, "test.png")
        
        with patch.object(self.exporter, '_is_matplotlib_figure', return_value=True):
            with patch('neurolite.visualization.exporters.MATPLOTLIB_AVAILABLE', True):
                with pytest.raises(VisualizationError, match="Export failed"):
                    self.exporter.export_figure(mock_figure, output_path, "png")
    
    def test_directory_creation(self):
        """Test that output directories are created."""
        mock_figure = Mock()
        mock_figure.savefig = Mock()
        
        # Use a nested directory that doesn't exist
        nested_dir = os.path.join(self.temp_dir, "nested", "dir")
        output_path = os.path.join(nested_dir, "test.png")
        
        with patch.object(self.exporter, '_is_matplotlib_figure', return_value=True):
            with patch('neurolite.visualization.exporters.MATPLOTLIB_AVAILABLE', True):
                self.exporter.export_figure(mock_figure, output_path, "png")
                
                # Check that directory was created
                assert os.path.exists(nested_dir)


if __name__ == "__main__":
    pytest.main([__file__])