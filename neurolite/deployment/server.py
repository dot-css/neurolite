"""
Web API server generation for NeuroLite.

Provides functionality to create Flask/FastAPI servers for trained models
with automatic input validation, error handling, and monitoring endpoints.
"""

from typing import Any, Dict, Optional, Union, List, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import threading
from datetime import datetime
import numpy as np

from ..core import get_logger, DeploymentError, DependencyError
from ..models.base import BaseModel


logger = get_logger(__name__)


class ServerFramework(Enum):
    """Supported web server frameworks."""
    
    FLASK = "flask"
    FASTAPI = "fastapi"


class DataType(Enum):
    """Supported input data types."""
    
    JSON = "json"
    IMAGE = "image"
    TEXT = "text"
    BINARY = "binary"
    MULTIPART = "multipart"


@dataclass
class APIConfig:
    """
    Configuration for API server generation.
    
    Defines server settings, endpoints, and validation rules.
    """
    
    # Server settings
    framework: ServerFramework = ServerFramework.FASTAPI
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # API settings
    api_title: str = "NeuroLite Model API"
    api_version: str = "1.0.0"
    api_description: str = "Auto-generated API for machine learning model"
    
    # Endpoint settings
    prediction_endpoint: str = "/predict"
    health_endpoint: str = "/health"
    info_endpoint: str = "/info"
    metrics_endpoint: str = "/metrics"
    
    # Input validation
    max_request_size_mb: float = 10.0
    request_timeout_seconds: float = 30.0
    enable_cors: bool = True
    
    # Authentication (optional)
    api_key_required: bool = False
    api_key_header: str = "X-API-Key"
    valid_api_keys: List[str] = field(default_factory=list)
    
    # Rate limiting
    enable_rate_limiting: bool = False
    rate_limit_requests_per_minute: int = 60
    
    # Monitoring
    enable_request_logging: bool = True
    enable_metrics_collection: bool = True
    
    # Model-specific settings
    expected_input_shape: Optional[Tuple[int, ...]] = None
    expected_input_type: DataType = DataType.JSON
    output_format: str = "json"  # "json", "binary", "image"


@dataclass
class ServerMetrics:
    """
    Server metrics for monitoring.
    
    Tracks request statistics and model performance.
    """
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    
    # Request timing
    response_times: List[float] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    
    # Error tracking
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    def add_request(self, response_time_ms: float, success: bool, error_type: str = None):
        """Add a request to metrics."""
        self.total_requests += 1
        self.response_times.append(response_time_ms)
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Update average response time (rolling average)
        if len(self.response_times) > 1000:  # Keep only last 1000 requests
            self.response_times = self.response_times[-1000:]
        
        self.average_response_time_ms = sum(self.response_times) / len(self.response_times)
    
    def get_uptime_seconds(self) -> float:
        """Get server uptime in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0


class ModelAPIServer:
    """
    Web API server for trained models.
    
    Creates Flask or FastAPI servers with automatic input validation,
    error handling, and monitoring capabilities.
    """
    
    def __init__(
        self,
        model: BaseModel,
        config: APIConfig = None
    ):
        """
        Initialize the API server.
        
        Args:
            model: Trained model to serve
            config: Server configuration
        """
        if not model.is_trained:
            raise DeploymentError("Model must be trained before creating API server")
        
        self.model = model
        self.config = config or APIConfig()
        self.metrics = ServerMetrics()
        self.app = None
        self.server_thread = None
        self._rate_limiter = None
        
        # Initialize rate limiter if enabled
        if self.config.enable_rate_limiting:
            self._rate_limiter = self._create_rate_limiter()
        
        logger.info(f"Initialized {self.config.framework.value} API server for model")
    
    def create_app(self) -> Any:
        """
        Create the web application.
        
        Returns:
            Flask or FastAPI application instance
        """
        if self.config.framework == ServerFramework.FLASK:
            return self._create_flask_app()
        elif self.config.framework == ServerFramework.FASTAPI:
            return self._create_fastapi_app()
        else:
            raise DeploymentError(f"Unsupported framework: {self.config.framework}")
    
    def _create_flask_app(self) -> Any:
        """Create Flask application."""
        try:
            from flask import Flask, request, jsonify
            from flask_cors import CORS
        except ImportError:
            raise DependencyError("Flask and flask-cors are required for Flask API server")
        
        app = Flask(__name__)
        
        # Enable CORS if configured
        if self.config.enable_cors:
            CORS(app)
        
        # Configure request size limit
        app.config['MAX_CONTENT_LENGTH'] = int(self.config.max_request_size_mb * 1024 * 1024)
        
        # Add middleware
        app.before_request(self._before_request_flask)
        app.after_request(self._after_request_flask)
        
        # Add routes
        app.route(self.config.prediction_endpoint, methods=['POST'])(self._predict_flask)
        app.route(self.config.health_endpoint, methods=['GET'])(self._health_check_flask)
        app.route(self.config.info_endpoint, methods=['GET'])(self._model_info_flask)
        app.route(self.config.metrics_endpoint, methods=['GET'])(self._metrics_flask)
        
        return app
    
    def _create_fastapi_app(self) -> Any:
        """Create FastAPI application."""
        try:
            from fastapi import FastAPI, HTTPException, Depends, Request
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import JSONResponse
            from pydantic import BaseModel as PydanticBaseModel
        except ImportError:
            raise DependencyError("FastAPI is required for FastAPI server")
        
        app = FastAPI(
            title=self.config.api_title,
            version=self.config.api_version,
            description=self.config.api_description
        )
        
        # Enable CORS if configured
        if self.config.enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Add middleware
        app.middleware("http")(self._middleware_fastapi)
        
        # Create request/response models
        PredictionRequest, PredictionResponse = self._create_pydantic_models()
        
        # Add routes
        @app.post(self.config.prediction_endpoint, response_model=PredictionResponse)
        async def predict(request: PredictionRequest, http_request: Request):
            return await self._predict_fastapi(request, http_request)
        
        @app.get(self.config.health_endpoint)
        async def health_check():
            return self._health_check_fastapi()
        
        @app.get(self.config.info_endpoint)
        async def model_info():
            return self._model_info_fastapi()
        
        @app.get(self.config.metrics_endpoint)
        async def metrics():
            return self._metrics_fastapi()
        
        return app
    
    def _create_pydantic_models(self) -> Tuple[Any, Any]:
        """Create Pydantic models for request/response validation."""
        try:
            from pydantic import BaseModel as PydanticBaseModel, Field
            from typing import Any, List, Union
        except ImportError:
            raise DependencyError("Pydantic is required for FastAPI request validation")
        
        class PredictionRequest(PydanticBaseModel):
            data: Union[List[List[float]], List[float], str, dict] = Field(
                ..., 
                description="Input data for prediction"
            )
            return_probabilities: bool = Field(
                False, 
                description="Whether to return class probabilities"
            )
            
            class Config:
                schema_extra = {
                    "example": {
                        "data": [[1.0, 2.0, 3.0, 4.0]],
                        "return_probabilities": False
                    }
                }
        
        class PredictionResponse(PydanticBaseModel):
            predictions: Union[List[float], List[int], List[List[float]]] = Field(
                ..., 
                description="Model predictions"
            )
            probabilities: Optional[List[List[float]]] = Field(
                None, 
                description="Class probabilities (if requested)"
            )
            processing_time_ms: float = Field(
                ..., 
                description="Processing time in milliseconds"
            )
            model_info: dict = Field(
                ..., 
                description="Model metadata"
            )
            
            class Config:
                schema_extra = {
                    "example": {
                        "predictions": [0],
                        "probabilities": [[0.8, 0.2]],
                        "processing_time_ms": 15.2,
                        "model_info": {"framework": "pytorch", "task": "classification"}
                    }
                }
        
        return PredictionRequest, PredictionResponse
    
    def _validate_api_key(self, request_headers: Dict[str, str]) -> bool:
        """Validate API key if authentication is enabled."""
        if not self.config.api_key_required:
            return True
        
        api_key = request_headers.get(self.config.api_key_header)
        if not api_key:
            return False
        
        return api_key in self.config.valid_api_keys
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if request is within rate limits."""
        if not self.config.enable_rate_limiting or not self._rate_limiter:
            return True
        
        return self._rate_limiter.is_allowed(client_ip)
    
    def _create_rate_limiter(self) -> Any:
        """Create rate limiter instance."""
        # Simple in-memory rate limiter
        class SimpleRateLimiter:
            def __init__(self, requests_per_minute: int):
                self.requests_per_minute = requests_per_minute
                self.requests = {}  # client_ip -> list of timestamps
            
            def is_allowed(self, client_ip: str) -> bool:
                now = time.time()
                minute_ago = now - 60
                
                # Clean old requests
                if client_ip in self.requests:
                    self.requests[client_ip] = [
                        req_time for req_time in self.requests[client_ip] 
                        if req_time > minute_ago
                    ]
                else:
                    self.requests[client_ip] = []
                
                # Check if under limit
                if len(self.requests[client_ip]) >= self.requests_per_minute:
                    return False
                
                # Add current request
                self.requests[client_ip].append(now)
                return True
        
        return SimpleRateLimiter(self.config.rate_limit_requests_per_minute)
    
    def _validate_input_data(self, data: Any) -> Tuple[bool, str, Any]:
        """
        Validate input data format and convert to model-compatible format.
        
        Returns:
            Tuple of (is_valid, error_message, processed_data)
        """
        try:
            # Convert to numpy array if needed
            if isinstance(data, (list, tuple)):
                processed_data = np.array(data, dtype=np.float32)
            elif isinstance(data, dict):
                # Handle dictionary input (e.g., for structured data)
                if 'data' in data:
                    processed_data = np.array(data['data'], dtype=np.float32)
                else:
                    return False, "Dictionary input must contain 'data' key", None
            elif isinstance(data, str):
                # Handle text input
                processed_data = data
            else:
                processed_data = np.array(data, dtype=np.float32)
            
            # Validate shape if expected shape is configured
            if (self.config.expected_input_shape is not None and 
                hasattr(processed_data, 'shape')):
                
                expected_shape = self.config.expected_input_shape
                actual_shape = processed_data.shape
                
                # Allow batch dimension flexibility
                if len(expected_shape) == len(actual_shape) - 1:
                    expected_shape = (None,) + expected_shape
                
                if len(actual_shape) != len(expected_shape):
                    return False, f"Expected {len(expected_shape)} dimensions, got {len(actual_shape)}", None
                
                # Check non-batch dimensions
                for i, (expected, actual) in enumerate(zip(expected_shape[1:], actual_shape[1:]), 1):
                    if expected is not None and expected != actual:
                        return False, f"Expected dimension {i} to be {expected}, got {actual}", None
            
            return True, "", processed_data
            
        except Exception as e:
            return False, f"Invalid input data format: {str(e)}", None
    
    def _make_prediction(self, data: Any, return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Make prediction with the model.
        
        Args:
            data: Input data
            return_probabilities: Whether to return probabilities
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            # Make prediction
            result = self.model.predict(data)
            
            # Extract predictions
            if hasattr(result, 'predictions'):
                predictions = result.predictions
                probabilities = getattr(result, 'probabilities', None)
            else:
                predictions = result
                probabilities = None
            
            # Convert to serializable format
            if isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()
            
            if probabilities is not None and isinstance(probabilities, np.ndarray):
                probabilities = probabilities.tolist()
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            response = {
                "predictions": predictions,
                "processing_time_ms": processing_time,
                "model_info": {
                    "framework": self.model.capabilities.framework,
                    "task": [task.value if hasattr(task, 'value') else str(task) for task in self.model.capabilities.supported_tasks],
                    "model_type": type(self.model).__name__
                }
            }
            
            if return_probabilities and probabilities is not None:
                response["probabilities"] = probabilities
            
            return response
            
        except Exception as e:
            raise DeploymentError(f"Prediction failed: {str(e)}")
    
    # Flask-specific methods
    def _before_request_flask(self):
        """Flask before request middleware."""
        from flask import request, jsonify
        
        # Check API key
        if not self._validate_api_key(dict(request.headers)):
            return jsonify({"error": "Invalid or missing API key"}), 401
        
        # Check rate limit
        client_ip = request.environ.get('REMOTE_ADDR', 'unknown')
        if not self._check_rate_limit(client_ip):
            return jsonify({"error": "Rate limit exceeded"}), 429
    
    def _after_request_flask(self, response):
        """Flask after request middleware."""
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        return response
    
    def _predict_flask(self):
        """Flask prediction endpoint."""
        from flask import request, jsonify
        
        start_time = time.time()
        success = False
        error_type = None
        
        try:
            # Get request data
            if not request.is_json:
                return jsonify({"error": "Content-Type must be application/json"}), 400
            
            request_data = request.get_json()
            if not request_data or 'data' not in request_data:
                return jsonify({"error": "Request must contain 'data' field"}), 400
            
            # Validate input
            is_valid, error_msg, processed_data = self._validate_input_data(request_data['data'])
            if not is_valid:
                return jsonify({"error": error_msg}), 400
            
            # Make prediction
            return_probabilities = request_data.get('return_probabilities', False)
            result = self._make_prediction(processed_data, return_probabilities)
            
            success = True
            return jsonify(result)
            
        except DeploymentError as e:
            error_type = "prediction_error"
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            error_type = "internal_error"
            logger.error(f"Unexpected error in prediction endpoint: {e}")
            return jsonify({"error": "Internal server error"}), 500
        finally:
            # Update metrics
            response_time = (time.time() - start_time) * 1000
            self.metrics.add_request(response_time, success, error_type)
    
    def _health_check_flask(self):
        """Flask health check endpoint."""
        from flask import jsonify
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": self.metrics.get_uptime_seconds(),
            "model_loaded": self.model.is_trained
        })
    
    def _model_info_flask(self):
        """Flask model info endpoint."""
        from flask import jsonify
        
        return jsonify({
            "model_type": type(self.model).__name__,
            "framework": self.model.capabilities.framework,
            "task_type": [task.value if hasattr(task, 'value') else str(task) for task in self.model.capabilities.supported_tasks],
            "input_shape": self.config.expected_input_shape,
            "api_version": self.config.api_version
        })
    
    def _metrics_flask(self):
        """Flask metrics endpoint."""
        from flask import jsonify
        
        return jsonify({
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate_percent": self.metrics.get_success_rate(),
            "average_response_time_ms": self.metrics.average_response_time_ms,
            "uptime_seconds": self.metrics.get_uptime_seconds(),
            "error_counts": self.metrics.error_counts
        })
    
    # FastAPI-specific methods
    async def _middleware_fastapi(self, request, call_next):
        """FastAPI middleware."""
        from fastapi import HTTPException
        
        # Check API key
        if not self._validate_api_key(dict(request.headers)):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        
        # Check rate limit
        client_ip = request.client.host if request.client else 'unknown'
        if not self._check_rate_limit(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        return response
    
    async def _predict_fastapi(self, request, http_request):
        """FastAPI prediction endpoint."""
        from fastapi import HTTPException
        
        start_time = time.time()
        success = False
        error_type = None
        
        try:
            # Validate input
            is_valid, error_msg, processed_data = self._validate_input_data(request.data)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Make prediction
            result = self._make_prediction(processed_data, request.return_probabilities)
            
            success = True
            return result
            
        except DeploymentError as e:
            error_type = "prediction_error"
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            error_type = "internal_error"
            logger.error(f"Unexpected error in prediction endpoint: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            # Update metrics
            response_time = (time.time() - start_time) * 1000
            self.metrics.add_request(response_time, success, error_type)
    
    def _health_check_fastapi(self):
        """FastAPI health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": self.metrics.get_uptime_seconds(),
            "model_loaded": self.model.is_trained
        }
    
    def _model_info_fastapi(self):
        """FastAPI model info endpoint."""
        return {
            "model_type": type(self.model).__name__,
            "framework": self.model.capabilities.framework,
            "task_type": [task.value if hasattr(task, 'value') else str(task) for task in self.model.capabilities.supported_tasks],
            "input_shape": self.config.expected_input_shape,
            "api_version": self.config.api_version
        }
    
    def _metrics_fastapi(self):
        """FastAPI metrics endpoint."""
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate_percent": self.metrics.get_success_rate(),
            "average_response_time_ms": self.metrics.average_response_time_ms,
            "uptime_seconds": self.metrics.get_uptime_seconds(),
            "error_counts": self.metrics.error_counts
        }
    
    def run(
        self,
        host: str = None,
        port: int = None,
        debug: bool = None,
        threaded: bool = True
    ) -> None:
        """
        Run the API server.
        
        Args:
            host: Host to bind to (overrides config)
            port: Port to bind to (overrides config)
            debug: Debug mode (overrides config)
            threaded: Whether to run in threaded mode
        """
        # Use provided values or fall back to config
        host = host or self.config.host
        port = port or self.config.port
        debug = debug if debug is not None else self.config.debug
        
        # Create app if not already created
        if self.app is None:
            self.app = self.create_app()
        
        logger.info(f"Starting {self.config.framework.value} server on {host}:{port}")
        
        try:
            if self.config.framework == ServerFramework.FLASK:
                self.app.run(
                    host=host,
                    port=port,
                    debug=debug,
                    threaded=threaded
                )
            elif self.config.framework == ServerFramework.FASTAPI:
                import uvicorn
                uvicorn.run(
                    self.app,
                    host=host,
                    port=port,
                    log_level="debug" if debug else "info"
                )
        except ImportError as e:
            if "uvicorn" in str(e):
                raise DependencyError("uvicorn is required to run FastAPI server")
            raise
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise DeploymentError(f"Failed to start server: {e}")
    
    def run_async(
        self,
        host: str = None,
        port: int = None,
        debug: bool = None
    ) -> threading.Thread:
        """
        Run the API server in a separate thread.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Debug mode
            
        Returns:
            Thread object running the server
        """
        def run_server():
            self.run(host, port, debug, threaded=True)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        logger.info("Server started in background thread")
        return self.server_thread
    
    def stop(self) -> None:
        """Stop the API server."""
        if self.server_thread and self.server_thread.is_alive():
            logger.info("Stopping API server...")
            # Note: Proper server shutdown would require more sophisticated handling
            # This is a simplified implementation
    
    def get_openapi_spec(self) -> Dict[str, Any]:
        """
        Generate OpenAPI specification for the API.
        
        Returns:
            OpenAPI specification dictionary
        """
        if self.config.framework == ServerFramework.FASTAPI:
            if self.app is None:
                self.app = self.create_app()
            return self.app.openapi()
        else:
            # Generate basic OpenAPI spec for Flask
            return {
                "openapi": "3.0.0",
                "info": {
                    "title": self.config.api_title,
                    "version": self.config.api_version,
                    "description": self.config.api_description
                },
                "paths": {
                    self.config.prediction_endpoint: {
                        "post": {
                            "summary": "Make prediction",
                            "requestBody": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "data": {"type": "array"},
                                                "return_probabilities": {"type": "boolean"}
                                            },
                                            "required": ["data"]
                                        }
                                    }
                                }
                            },
                            "responses": {
                                "200": {
                                    "description": "Prediction result",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "predictions": {"type": "array"},
                                                    "probabilities": {"type": "array"},
                                                    "processing_time_ms": {"type": "number"},
                                                    "model_info": {"type": "object"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    self.config.health_endpoint: {
                        "get": {
                            "summary": "Health check",
                            "responses": {
                                "200": {
                                    "description": "Health status"
                                }
                            }
                        }
                    }
                }
            }


def create_api_server(
    model: BaseModel,
    framework: str = "fastapi",
    config: APIConfig = None,
    **kwargs
) -> ModelAPIServer:
    """
    Create an API server for a trained model.
    
    Args:
        model: Trained model to serve
        framework: Web framework to use ("flask" or "fastapi")
        config: Server configuration
        **kwargs: Additional configuration options
        
    Returns:
        ModelAPIServer instance
    """
    # Create config if not provided
    if config is None:
        config = APIConfig()
    
    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Set framework
    if framework.lower() == "flask":
        config.framework = ServerFramework.FLASK
    elif framework.lower() == "fastapi":
        config.framework = ServerFramework.FASTAPI
    else:
        raise DeploymentError(f"Unsupported framework: {framework}")
    
    return ModelAPIServer(model, config)