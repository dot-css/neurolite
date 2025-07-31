"""
Integration tests for API deployment functionality.

Tests the web API server generation, input validation, error handling,
and monitoring endpoints for deployed models.
"""

import pytest
import json
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from neurolite.deployment.server import (
    ModelAPIServer,
    APIConfig,
    ServerFramework,
    DataType,
    ServerMetrics,
    create_api_server
)
from neurolite.core.exceptions import DeploymentError, DependencyError
from neurolite.models.base import BaseModel, ModelCapabilities, TaskType


class MockModel(BaseModel):
    """Mock model for testing."""
    
    def __init__(self, framework="pytorch", task_type=TaskType.CLASSIFICATION):
        self._capabilities = ModelCapabilities(
            supported_tasks=[task_type],
            supported_data_types=[],
            framework=framework,
            supports_probability_prediction=True
        )
        self._is_trained = True
        self.metadata = Mock()
        self.metadata.model_size_mb = 10.5
    
    @property
    def capabilities(self):
        return self._capabilities
    
    @property
    def is_trained(self) -> bool:
        return self._is_trained
    
    def fit(self, X, y, **kwargs):
        """Mock fit method."""
        self._is_trained = True
        return self
    
    def predict(self, X):
        """Mock prediction method."""
        if isinstance(X, str):
            # Text input
            return Mock(predictions=[0], probabilities=[[0.8, 0.2]])
        
        # Numeric input
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        batch_size = X.shape[0] if len(X.shape) > 1 else 1
        
        # Return mock predictions
        predictions = np.random.randint(0, 2, batch_size)
        probabilities = np.random.rand(batch_size, 2)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        return Mock(predictions=predictions, probabilities=probabilities)
    
    def save(self, path):
        """Mock save method."""
        pass
    
    def load(self, path):
        """Mock load method."""
        pass


class TestServerMetrics:
    """Test server metrics functionality."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ServerMetrics()
        
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.average_response_time_ms == 0.0
        assert len(metrics.response_times) == 0
        assert len(metrics.error_counts) == 0
    
    def test_add_successful_request(self):
        """Test adding successful request to metrics."""
        metrics = ServerMetrics()
        
        metrics.add_request(response_time_ms=100.0, success=True)
        
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.average_response_time_ms == 100.0
        assert metrics.get_success_rate() == 100.0
    
    def test_add_failed_request(self):
        """Test adding failed request to metrics."""
        metrics = ServerMetrics()
        
        metrics.add_request(response_time_ms=50.0, success=False, error_type="validation_error")
        
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 1
        assert metrics.average_response_time_ms == 50.0
        assert metrics.get_success_rate() == 0.0
        assert metrics.error_counts["validation_error"] == 1
    
    def test_rolling_average(self):
        """Test rolling average calculation."""
        metrics = ServerMetrics()
        
        # Add multiple requests
        for i in range(5):
            metrics.add_request(response_time_ms=float(i * 10), success=True)
        
        expected_avg = (0 + 10 + 20 + 30 + 40) / 5
        assert metrics.average_response_time_ms == expected_avg
    
    def test_uptime_calculation(self):
        """Test uptime calculation."""
        metrics = ServerMetrics()
        time.sleep(0.1)  # Wait a bit
        
        uptime = metrics.get_uptime_seconds()
        assert uptime > 0.05  # Should be at least 50ms


class TestAPIConfig:
    """Test API configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = APIConfig()
        
        assert config.framework == ServerFramework.FASTAPI
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.debug is False
        assert config.prediction_endpoint == "/predict"
        assert config.health_endpoint == "/health"
        assert config.info_endpoint == "/info"
        assert config.metrics_endpoint == "/metrics"
        assert config.enable_cors is True
        assert config.api_key_required is False
        assert config.enable_rate_limiting is False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = APIConfig(
            framework=ServerFramework.FLASK,
            port=9000,
            debug=True,
            api_key_required=True,
            valid_api_keys=["test-key-123"]
        )
        
        assert config.framework == ServerFramework.FLASK
        assert config.port == 9000
        assert config.debug is True
        assert config.api_key_required is True
        assert "test-key-123" in config.valid_api_keys


class TestModelAPIServer:
    """Test ModelAPIServer functionality."""
    
    def test_server_initialization(self):
        """Test server initialization."""
        model = MockModel()
        config = APIConfig()
        
        server = ModelAPIServer(model, config)
        
        assert server.model == model
        assert server.config == config
        assert isinstance(server.metrics, ServerMetrics)
        assert server.app is None
    
    def test_untrained_model_error(self):
        """Test error when using untrained model."""
        model = MockModel()
        model._is_trained = False
        
        with pytest.raises(DeploymentError) as exc_info:
            ModelAPIServer(model)
        
        assert "must be trained" in str(exc_info.value)
    
    def test_input_validation_valid_data(self):
        """Test input validation with valid data."""
        model = MockModel()
        server = ModelAPIServer(model)
        
        # Test list input
        is_valid, error_msg, processed_data = server._validate_input_data([[1.0, 2.0, 3.0]])
        assert is_valid
        assert error_msg == ""
        assert isinstance(processed_data, np.ndarray)
        assert processed_data.shape == (1, 3)
        
        # Test dictionary input
        is_valid, error_msg, processed_data = server._validate_input_data({"data": [1.0, 2.0]})
        assert is_valid
        assert error_msg == ""
        assert isinstance(processed_data, np.ndarray)
        
        # Test string input
        is_valid, error_msg, processed_data = server._validate_input_data("test text")
        assert is_valid
        assert error_msg == ""
        assert processed_data == "test text"
    
    def test_input_validation_invalid_data(self):
        """Test input validation with invalid data."""
        model = MockModel()
        config = APIConfig(expected_input_shape=(3,))
        server = ModelAPIServer(model, config)
        
        # Test wrong shape
        is_valid, error_msg, processed_data = server._validate_input_data([[1.0, 2.0]])  # Wrong shape
        assert not is_valid
        assert "Expected dimension" in error_msg
        
        # Test dictionary without 'data' key
        is_valid, error_msg, processed_data = server._validate_input_data({"wrong_key": [1.0, 2.0]})
        assert not is_valid
        assert "must contain 'data' key" in error_msg
    
    def test_make_prediction(self):
        """Test making predictions."""
        model = MockModel()
        server = ModelAPIServer(model)
        
        result = server._make_prediction([[1.0, 2.0, 3.0]], return_probabilities=True)
        
        assert "predictions" in result
        assert "processing_time_ms" in result
        assert "model_info" in result
        assert "probabilities" in result
        assert isinstance(result["processing_time_ms"], float)
        assert result["processing_time_ms"] > 0
    
    def test_api_key_validation(self):
        """Test API key validation."""
        model = MockModel()
        config = APIConfig(api_key_required=True, valid_api_keys=["valid-key"])
        server = ModelAPIServer(model, config)
        
        # Test valid key
        headers = {"X-API-Key": "valid-key"}
        assert server._validate_api_key(headers) is True
        
        # Test invalid key
        headers = {"X-API-Key": "invalid-key"}
        assert server._validate_api_key(headers) is False
        
        # Test missing key
        headers = {}
        assert server._validate_api_key(headers) is False
        
        # Test disabled authentication
        config.api_key_required = False
        assert server._validate_api_key({}) is True
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        model = MockModel()
        config = APIConfig(enable_rate_limiting=True, rate_limit_requests_per_minute=2)
        server = ModelAPIServer(model, config)
        
        client_ip = "127.0.0.1"
        
        # First two requests should be allowed
        assert server._check_rate_limit(client_ip) is True
        assert server._check_rate_limit(client_ip) is True
        
        # Third request should be blocked
        assert server._check_rate_limit(client_ip) is False
    
    @patch('neurolite.deployment.server.Flask')
    def test_create_flask_app(self, mock_flask):
        """Test Flask app creation."""
        model = MockModel()
        config = APIConfig(framework=ServerFramework.FLASK)
        server = ModelAPIServer(model, config)
        
        # Mock Flask and its methods
        mock_app = Mock()
        mock_flask.return_value = mock_app
        
        app = server._create_flask_app()
        
        # Verify Flask was called
        mock_flask.assert_called_once()
        assert app == mock_app
    
    @patch('neurolite.deployment.server.FastAPI')
    def test_create_fastapi_app(self, mock_fastapi):
        """Test FastAPI app creation."""
        model = MockModel()
        config = APIConfig(framework=ServerFramework.FASTAPI)
        server = ModelAPIServer(model, config)
        
        # Mock FastAPI
        mock_app = Mock()
        mock_fastapi.return_value = mock_app
        
        app = server._create_fastapi_app()
        
        # Verify FastAPI was called with correct parameters
        mock_fastapi.assert_called_once_with(
            title=config.api_title,
            version=config.api_version,
            description=config.api_description
        )
        assert app == mock_app
    
    def test_openapi_spec_generation(self):
        """Test OpenAPI specification generation."""
        model = MockModel()
        server = ModelAPIServer(model)
        
        spec = server.get_openapi_spec()
        
        assert isinstance(spec, dict)
        assert "openapi" in spec
        assert "info" in spec
        assert "paths" in spec


class TestCreateAPIServer:
    """Test create_api_server function."""
    
    def test_create_fastapi_server(self):
        """Test creating FastAPI server."""
        model = MockModel()
        
        server = create_api_server(model, framework="fastapi")
        
        assert isinstance(server, ModelAPIServer)
        assert server.config.framework == ServerFramework.FASTAPI
    
    def test_create_flask_server(self):
        """Test creating Flask server."""
        model = MockModel()
        
        server = create_api_server(model, framework="flask")
        
        assert isinstance(server, ModelAPIServer)
        assert server.config.framework == ServerFramework.FLASK
    
    def test_create_server_with_custom_config(self):
        """Test creating server with custom configuration."""
        model = MockModel()
        config = APIConfig(port=9000, debug=True)
        
        server = create_api_server(model, config=config, host="localhost")
        
        assert server.config.port == 9000
        assert server.config.debug is True
        assert server.config.host == "localhost"
    
    def test_unsupported_framework_error(self):
        """Test error with unsupported framework."""
        model = MockModel()
        
        with pytest.raises(DeploymentError) as exc_info:
            create_api_server(model, framework="unsupported")
        
        assert "Unsupported framework" in str(exc_info.value)


class TestAPIEndpoints:
    """Test API endpoint functionality."""
    
    def test_health_check_response(self):
        """Test health check endpoint response."""
        model = MockModel()
        server = ModelAPIServer(model)
        
        response = server._health_check_fastapi()
        
        assert "status" in response
        assert response["status"] == "healthy"
        assert "timestamp" in response
        assert "uptime_seconds" in response
        assert "model_loaded" in response
        assert response["model_loaded"] is True
    
    def test_model_info_response(self):
        """Test model info endpoint response."""
        model = MockModel()
        server = ModelAPIServer(model)
        
        response = server._model_info_fastapi()
        
        assert "model_type" in response
        assert "framework" in response
        assert "task_type" in response
        assert "api_version" in response
        assert response["framework"] == "pytorch"
    
    def test_metrics_response(self):
        """Test metrics endpoint response."""
        model = MockModel()
        server = ModelAPIServer(model)
        
        # Add some test metrics
        server.metrics.add_request(100.0, True)
        server.metrics.add_request(200.0, False, "test_error")
        
        response = server._metrics_fastapi()
        
        assert "total_requests" in response
        assert "successful_requests" in response
        assert "failed_requests" in response
        assert "success_rate_percent" in response
        assert "average_response_time_ms" in response
        assert "uptime_seconds" in response
        assert "error_counts" in response
        
        assert response["total_requests"] == 2
        assert response["successful_requests"] == 1
        assert response["failed_requests"] == 1
        assert response["success_rate_percent"] == 50.0


class TestErrorHandling:
    """Test error handling in API server."""
    
    def test_prediction_error_handling(self):
        """Test error handling during prediction."""
        model = MockModel()
        
        # Mock model to raise exception
        def failing_predict(X):
            raise ValueError("Prediction failed")
        
        model.predict = failing_predict
        server = ModelAPIServer(model)
        
        with pytest.raises(DeploymentError) as exc_info:
            server._make_prediction([[1.0, 2.0, 3.0]])
        
        assert "Prediction failed" in str(exc_info.value)
    
    def test_dependency_error_handling(self):
        """Test handling of missing dependencies."""
        model = MockModel()
        config = APIConfig(framework=ServerFramework.FLASK)
        server = ModelAPIServer(model, config)
        
        # Mock missing Flask dependency
        with patch('neurolite.deployment.server.Flask', side_effect=ImportError("No module named 'flask'")):
            with pytest.raises(DependencyError) as exc_info:
                server._create_flask_app()
            
            assert "Flask" in str(exc_info.value)


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def test_complete_server_workflow(self):
        """Test complete server creation and configuration workflow."""
        # Create model
        model = MockModel()
        
        # Create custom configuration
        config = APIConfig(
            framework=ServerFramework.FASTAPI,
            port=8080,
            api_title="Test Model API",
            enable_cors=True,
            max_request_size_mb=5.0
        )
        
        # Create server
        server = ModelAPIServer(model, config)
        
        # Verify configuration
        assert server.config.port == 8080
        assert server.config.api_title == "Test Model API"
        assert server.config.enable_cors is True
        assert server.config.max_request_size_mb == 5.0
        
        # Test input validation
        is_valid, _, processed_data = server._validate_input_data([[1.0, 2.0, 3.0, 4.0]])
        assert is_valid
        assert processed_data.shape == (1, 4)
        
        # Test prediction
        result = server._make_prediction(processed_data, return_probabilities=True)
        assert "predictions" in result
        assert "probabilities" in result
        assert "processing_time_ms" in result
        assert "model_info" in result
        
        # Test metrics
        server.metrics.add_request(150.0, True)
        metrics = server._metrics_fastapi()
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 1
        assert metrics["average_response_time_ms"] == 150.0
    
    def test_multi_framework_support(self):
        """Test support for multiple ML frameworks."""
        frameworks = ["pytorch", "tensorflow", "sklearn"]
        
        for framework in frameworks:
            model = MockModel(framework=framework)
            server = ModelAPIServer(model)
            
            # Test model info
            info = server._model_info_fastapi()
            assert info["framework"] == framework
            
            # Test prediction
            result = server._make_prediction([[1.0, 2.0]])
            assert "predictions" in result
            assert result["model_info"]["framework"] == framework
    
    def test_different_task_types(self):
        """Test support for different task types."""
        task_types = [TaskType.CLASSIFICATION, TaskType.REGRESSION]
        
        for task_type in task_types:
            model = MockModel(task_type=task_type)
            server = ModelAPIServer(model)
            
            # Test model info
            info = server._model_info_fastapi()
            assert task_type.value in info["task_type"] or str(task_type) in info["task_type"]
            
            # Test prediction
            result = server._make_prediction([[1.0, 2.0]])
            assert "predictions" in result


if __name__ == "__main__":
    pytest.main([__file__])