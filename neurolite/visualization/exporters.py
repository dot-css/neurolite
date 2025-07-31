"""
Visualization export utilities for NeuroLite.

Provides export functionality for visualizations in multiple formats
including PNG, PDF, SVG, and HTML with proper error handling.
"""

import os
from pathlib import Path
from typing import Any, Optional, Dict, List, Union
import base64
from io import BytesIO
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from ..core import get_logger
from ..core.exceptions import VisualizationError

logger = get_logger(__name__)


class VisualizationExporter:
    """
    Comprehensive visualization export utility.
    
    Handles export of matplotlib and plotly figures to various formats
    with proper error handling and format validation.
    """
    
    SUPPORTED_FORMATS = {
        'png': 'Portable Network Graphics',
        'jpg': 'JPEG Image',
        'jpeg': 'JPEG Image',
        'pdf': 'Portable Document Format',
        'svg': 'Scalable Vector Graphics',
        'html': 'HTML with embedded JavaScript',
        'json': 'Plotly JSON format'
    }
    
    def __init__(self):
        """Initialize visualization exporter."""
        self._check_dependencies()
        logger.debug("Visualization exporter initialized")
    
    def _check_dependencies(self):
        """Check available dependencies and log warnings."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Some export formats may not work.")
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Interactive exports will not work.")
        
        if not PIL_AVAILABLE:
            logger.warning("PIL not available. Some image processing features may not work.")
    
    def export_figure(
        self,
        figure: Any,
        output_path: str,
        format: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Export figure to specified format.
        
        Args:
            figure: Figure object (matplotlib or plotly)
            output_path: Output file path
            format: Export format (auto-detected from extension if None)
            **kwargs: Additional export parameters
            
        Returns:
            Path to exported file
            
        Raises:
            VisualizationError: If export fails
        """
        # Determine format from file extension if not specified
        if format is None:
            format = Path(output_path).suffix.lower().lstrip('.')
        
        format = format.lower()
        
        # Validate format
        if format not in self.SUPPORTED_FORMATS:
            raise VisualizationError(
                f"Unsupported format: {format}. "
                f"Supported formats: {list(self.SUPPORTED_FORMATS.keys())}"
            )
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export based on figure type and format
        try:
            if self._is_plotly_figure(figure):
                return self._export_plotly_figure(figure, output_path, format, **kwargs)
            elif self._is_matplotlib_figure(figure):
                return self._export_matplotlib_figure(figure, output_path, format, **kwargs)
            else:
                raise VisualizationError(f"Unsupported figure type: {type(figure)}")
        
        except Exception as e:
            logger.error(f"Failed to export figure to {output_path}: {e}")
            raise VisualizationError(f"Export failed: {e}")
    
    def export_multiple_figures(
        self,
        figures: Dict[str, Any],
        output_dir: str,
        format: str = 'png',
        **kwargs
    ) -> Dict[str, str]:
        """
        Export multiple figures to specified directory.
        
        Args:
            figures: Dictionary of figure_name -> figure_object
            output_dir: Output directory
            format: Export format for all figures
            **kwargs: Additional export parameters
            
        Returns:
            Dictionary of figure_name -> output_path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        for name, figure in figures.items():
            try:
                # Create safe filename
                safe_name = self._sanitize_filename(name)
                output_path = output_dir / f"{safe_name}.{format}"
                
                exported_path = self.export_figure(figure, str(output_path), format, **kwargs)
                exported_files[name] = exported_path
                
                logger.debug(f"Exported {name} to {exported_path}")
                
            except Exception as e:
                logger.error(f"Failed to export figure {name}: {e}")
                exported_files[name] = None
        
        logger.info(f"Exported {len([p for p in exported_files.values() if p])} figures to {output_dir}")
        return exported_files
    
    def _is_plotly_figure(self, figure: Any) -> bool:
        """Check if figure is a Plotly figure."""
        if not PLOTLY_AVAILABLE:
            return False
        
        return isinstance(figure, (go.Figure, go.FigureWidget))
    
    def _is_matplotlib_figure(self, figure: Any) -> bool:
        """Check if figure is a matplotlib figure."""
        if not MATPLOTLIB_AVAILABLE:
            return False
        
        return hasattr(figure, 'savefig')  # Duck typing for matplotlib figures
    
    def _export_plotly_figure(
        self,
        figure: Any,
        output_path: str,
        format: str,
        **kwargs
    ) -> str:
        """Export Plotly figure."""
        if not PLOTLY_AVAILABLE:
            raise VisualizationError("Plotly not available for export")
        
        # Set default parameters
        export_kwargs = {
            'width': kwargs.get('width', 1200),
            'height': kwargs.get('height', 800),
            'scale': kwargs.get('scale', 2)
        }
        
        try:
            if format == 'html':
                # HTML export with embedded JavaScript
                html_kwargs = {
                    'include_plotlyjs': kwargs.get('include_plotlyjs', True),
                    'div_id': kwargs.get('div_id', None),
                    'config': kwargs.get('config', {'displayModeBar': True})
                }
                figure.write_html(output_path, **html_kwargs)
                
            elif format == 'json':
                # JSON export
                figure.write_json(output_path)
                
            elif format in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
                # Image/vector export
                if format in ['jpg', 'jpeg']:
                    format = 'jpeg'
                
                figure.write_image(output_path, format=format, **export_kwargs)
                
            else:
                raise VisualizationError(f"Unsupported Plotly export format: {format}")
            
            logger.debug(f"Plotly figure exported to {output_path}")
            return output_path
            
        except Exception as e:
            # Try alternative export methods
            if format in ['png', 'jpg', 'jpeg'] and 'kaleido' in str(e).lower():
                logger.warning("Kaleido not available, trying alternative export method")
                return self._export_plotly_alternative(figure, output_path, format, **kwargs)
            else:
                raise
    
    def _export_plotly_alternative(
        self,
        figure: Any,
        output_path: str,
        format: str,
        **kwargs
    ) -> str:
        """Alternative Plotly export method when kaleido is not available."""
        try:
            # Try using orca (legacy)
            figure.write_image(output_path, format=format, engine='orca', **kwargs)
            logger.debug(f"Plotly figure exported using orca to {output_path}")
            return output_path
            
        except Exception:
            # Fallback to HTML export
            logger.warning(f"Cannot export to {format}, falling back to HTML")
            html_path = output_path.rsplit('.', 1)[0] + '.html'
            figure.write_html(html_path)
            logger.debug(f"Plotly figure exported as HTML to {html_path}")
            return html_path
    
    def _export_matplotlib_figure(
        self,
        figure: Any,
        output_path: str,
        format: str,
        **kwargs
    ) -> str:
        """Export matplotlib figure."""
        if not MATPLOTLIB_AVAILABLE:
            raise VisualizationError("Matplotlib not available for export")
        
        # Set default parameters
        export_kwargs = {
            'dpi': kwargs.get('dpi', 300),
            'bbox_inches': kwargs.get('bbox_inches', 'tight'),
            'facecolor': kwargs.get('facecolor', 'white'),
            'edgecolor': kwargs.get('edgecolor', 'none')
        }
        
        # Format-specific parameters
        if format in ['jpg', 'jpeg']:
            export_kwargs['format'] = 'jpeg'
            # JPEG doesn't support transparency
            export_kwargs['facecolor'] = 'white'
        elif format == 'png':
            export_kwargs['format'] = 'png'
            export_kwargs['transparent'] = kwargs.get('transparent', False)
        elif format == 'pdf':
            export_kwargs['format'] = 'pdf'
        elif format == 'svg':
            export_kwargs['format'] = 'svg'
        elif format == 'html':
            # For HTML, we'll save as PNG and embed
            png_path = output_path.rsplit('.', 1)[0] + '.png'
            figure.savefig(png_path, format='png', **export_kwargs)
            return self._create_html_with_image(png_path, output_path)
        else:
            raise VisualizationError(f"Unsupported matplotlib export format: {format}")
        
        try:
            figure.savefig(output_path, **export_kwargs)
            logger.debug(f"Matplotlib figure exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Matplotlib export failed: {e}")
            raise
    
    def _create_html_with_image(self, image_path: str, html_path: str) -> str:
        """Create HTML file with embedded image."""
        try:
            # Read image and convert to base64
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            image_base64 = base64.b64encode(image_data).decode()
            
            # Create HTML content
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Visualization</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            text-align: center; 
            margin: 20px; 
        }}
        img {{ 
            max-width: 100%; 
            height: auto; 
        }}
    </style>
</head>
<body>
    <h1>Visualization</h1>
    <img src="data:image/png;base64,{image_base64}" alt="Visualization">
</body>
</html>
            """
            
            # Write HTML file
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content.strip())
            
            # Clean up temporary PNG file
            try:
                os.remove(image_path)
            except Exception:
                pass
            
            logger.debug(f"HTML with embedded image created: {html_path}")
            return html_path
            
        except Exception as e:
            logger.error(f"Failed to create HTML with image: {e}")
            raise
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage."""
        # Replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove extra spaces and dots
        filename = filename.strip('. ')
        
        # Limit length
        if len(filename) > 100:
            filename = filename[:100]
        
        return filename or 'visualization'
    
    def get_supported_formats(self) -> Dict[str, str]:
        """
        Get supported export formats.
        
        Returns:
            Dictionary of format -> description
        """
        available_formats = {}
        
        for format, description in self.SUPPORTED_FORMATS.items():
            # Check if format is actually available
            if format in ['html', 'json']:
                if PLOTLY_AVAILABLE:
                    available_formats[format] = description
            elif format in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
                if MATPLOTLIB_AVAILABLE or PLOTLY_AVAILABLE:
                    available_formats[format] = description
        
        return available_formats
    
    def create_export_batch(
        self,
        figures: Dict[str, Any],
        output_dir: str,
        formats: List[str] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Export figures in multiple formats.
        
        Args:
            figures: Dictionary of figure_name -> figure_object
            output_dir: Output directory
            formats: List of formats to export (default: ['png', 'html'])
            
        Returns:
            Dictionary of format -> {figure_name -> output_path}
        """
        if formats is None:
            formats = ['png', 'html']
        
        results = {}
        
        for format in formats:
            try:
                format_results = self.export_multiple_figures(
                    figures, 
                    os.path.join(output_dir, format), 
                    format
                )
                results[format] = format_results
                
            except Exception as e:
                logger.error(f"Failed to export in {format} format: {e}")
                results[format] = {}
        
        logger.info(f"Batch export completed for {len(formats)} formats")
        return results


# Convenience functions
def export_figure(
    figure: Any,
    output_path: str,
    format: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convenience function to export a single figure.
    
    Args:
        figure: Figure object
        output_path: Output file path
        format: Export format (auto-detected if None)
        **kwargs: Additional export parameters
        
    Returns:
        Path to exported file
    """
    exporter = VisualizationExporter()
    return exporter.export_figure(figure, output_path, format, **kwargs)


def export_multiple_figures(
    figures: Dict[str, Any],
    output_dir: str,
    format: str = 'png',
    **kwargs
) -> Dict[str, str]:
    """
    Convenience function to export multiple figures.
    
    Args:
        figures: Dictionary of figure_name -> figure_object
        output_dir: Output directory
        format: Export format
        **kwargs: Additional export parameters
        
    Returns:
        Dictionary of figure_name -> output_path
    """
    exporter = VisualizationExporter()
    return exporter.export_multiple_figures(figures, output_dir, format, **kwargs)


def get_supported_formats() -> Dict[str, str]:
    """
    Get supported export formats.
    
    Returns:
        Dictionary of format -> description
    """
    exporter = VisualizationExporter()
    return exporter.get_supported_formats()