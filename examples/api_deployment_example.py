#!/usr/bin/env python3
"""
Example demonstrating the Web API Deployment System for NeuroLite.

This example shows how to:
1. Create a trained model
2. Set up a web API server with automatic input validation
3. Configure monitoring and health check endpoints
4. Handle different data types and error scenarios
"""

import numpy as np
from neurolite.deployment.server import (
    ModelAPIServer,
    APIConfig,
    ServerFramework,
    create_api_server
)
from neurolite.models.base import BaseModel, ModelCapabilities, TaskType
from neurolite.data.detector import DataType


class ExampleModel(BaseModel):
    """Example trained model for demonstration."""
    
    def __init__(self):
        self._capabilities = ModelCapabilities(
            supported_tasks=[TaskType.CLASSIFICATION],
            supported_data_types=[DataType.TABULAR],
            framework="sklearn",
            supports_probability_prediction=True
        )
        self._is_trained = True
    
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
        """Mock prediction method that returns random predictions."""
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        batch_size = X.shape[0] if len(X.shape) > 1 else 1
        
        # Return mock predictions and probabilities
        predictions = np.random.randint(0, 3, batch_size)  # 3 classes
        probabilities = np.random.rand(batch_size, 3)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        # Return result with both predictions and probabilities
        class MockResult:
            def __init__(self, pred, prob):
                self.predictions = pred
                self.probabilities = prob
        
        return MockResult(predictions, probabilities)
    
    def save(self, path):
        """Mock save method."""
        pass
    
    def load(self, path):
        """Mock load method."""
        pass


def main():
    """Demonstrate the Web API deployment system."""
    print("NeuroLite Web API Deployment System Example")
    print("=" * 50)
    
    # 1. Create a trained model
    print("\n1. Creating a trained model...")
    model = ExampleModel()
    print(f"   Model framework: {model.capabilities.framework}")
    print(f"   Model is trained: {model.is_trained}")
    
    # 2. Create API configuration
    print("\n2. Setting up API configuration...")
    config = APIConfig(
        framework=ServerFramework.FASTAPI,
        host="127.0.0.1",
        port=8080,
        api_title="Example ML Model API",
        api_description="Demonstration of NeuroLite's automatic API generation",
        prediction_endpoint="/predict",
        health_endpoint="/health",
        info_endpoint="/info",
        metrics_endpoint="/metrics",
        enable_cors=True,
        max_request_size_mb=5.0,
        expected_input_shape=(4,),  # Expecting 4 features
        enable_request_logging=True,
        enable_metrics_collection=True
    )
    
    print(f"   Framework: {config.framework.value}")
    print(f"   Host: {config.host}:{config.port}")
    print(f"   API Title: {config.api_title}")
    
    # 3. Create API server
    print("\n3. Creating API server...")
    server = ModelAPIServer(model, config)
    print(f"   Server created successfully")
    print(f"   Available endpoints:")
    print(f"     - POST {config.prediction_endpoint} (predictions)")
    print(f"     - GET  {config.health_endpoint} (health check)")
    print(f"     - GET  {config.info_endpoint} (model info)")
    print(f"     - GET  {config.metrics_endpoint} (server metrics)")
    
    # 4. Test input validation
    print("\n4. Testing input validation...")
    
    # Valid input
    valid_input = [[1.0, 2.0, 3.0, 4.0]]
    is_valid, error_msg, processed_data = server._validate_input_data(valid_input)
    print(f"   Valid input {valid_input}: {'✓' if is_valid else '✗'}")
    if is_valid:
        print(f"     Processed shape: {processed_data.shape}")
    
    # Invalid input (wrong shape)
    invalid_input = [[1.0, 2.0]]  # Only 2 features instead of 4
    is_valid, error_msg, processed_data = server._validate_input_data(invalid_input)
    print(f"   Invalid input {invalid_input}: {'✓' if is_valid else '✗'}")
    if not is_valid:
        print(f"     Error: {error_msg}")
    
    # 5. Test prediction
    print("\n5. Testing prediction functionality...")
    try:
        result = server._make_prediction([[1.0, 2.0, 3.0, 4.0]], return_probabilities=True)
        print(f"   Prediction successful: ✓")
        print(f"     Predictions: {result['predictions']}")
        print(f"     Probabilities: {result['probabilities']}")
        print(f"     Processing time: {result['processing_time_ms']:.2f}ms")
        print(f"     Model info: {result['model_info']}")
    except Exception as e:
        print(f"   Prediction failed: ✗ ({e})")
    
    # 6. Test health check
    print("\n6. Testing health check...")
    health_response = server._health_check_fastapi()
    print(f"   Health status: {health_response['status']}")
    print(f"   Model loaded: {health_response['model_loaded']}")
    print(f"   Uptime: {health_response['uptime_seconds']:.2f}s")
    
    # 7. Test model info
    print("\n7. Testing model info...")
    info_response = server._model_info_fastapi()
    print(f"   Model type: {info_response['model_type']}")
    print(f"   Framework: {info_response['framework']}")
    print(f"   Task type: {info_response['task_type']}")
    
    # 8. Test metrics
    print("\n8. Testing metrics collection...")
    # Add some mock requests to metrics
    server.metrics.add_request(100.0, True)
    server.metrics.add_request(150.0, True)
    server.metrics.add_request(200.0, False, "validation_error")
    
    metrics_response = server._metrics_fastapi()
    print(f"   Total requests: {metrics_response['total_requests']}")
    print(f"   Success rate: {metrics_response['success_rate_percent']:.1f}%")
    print(f"   Average response time: {metrics_response['average_response_time_ms']:.2f}ms")
    print(f"   Error counts: {metrics_response['error_counts']}")
    
    # 9. Show OpenAPI specification
    print("\n9. OpenAPI specification...")
    try:
        openapi_spec = server.get_openapi_spec()
        print(f"   OpenAPI version: {openapi_spec.get('openapi', 'N/A')}")
        print(f"   API title: {openapi_spec.get('info', {}).get('title', 'N/A')}")
        print(f"   Available paths: {list(openapi_spec.get('paths', {}).keys())}")
    except Exception as e:
        print(f"   OpenAPI spec generation failed: {e}")
    
    # 10. Alternative server creation method
    print("\n10. Alternative server creation...")
    try:
        alt_server = create_api_server(
            model=model,
            framework="flask",
            port=8081,
            debug=True,
            api_title="Alternative Flask API"
        )
        print(f"    Alternative server created: ✓")
        print(f"    Framework: {alt_server.config.framework.value}")
        print(f"    Port: {alt_server.config.port}")
        print(f"    Debug mode: {alt_server.config.debug}")
    except Exception as e:
        print(f"    Alternative server creation failed: ✗ ({e})")
    
    print("\n" + "=" * 50)
    print("Web API Deployment System demonstration completed!")
    print("\nTo run the actual server, you would call:")
    print("  server.run()  # This would start the server on the configured port")
    print("\nThe server would then be accessible at:")
    print(f"  http://{config.host}:{config.port}{config.prediction_endpoint}")
    print(f"  http://{config.host}:{config.port}{config.health_endpoint}")
    print(f"  http://{config.host}:{config.port}{config.info_endpoint}")
    print(f"  http://{config.host}:{config.port}{config.metrics_endpoint}")


if __name__ == "__main__":
    main()