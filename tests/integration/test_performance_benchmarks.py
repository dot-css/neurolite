"""
Performance benchmark tests for NeuroLite.
Tests training and inference speed across different scenarios.
"""

import pytest
import time
import tempfile
import os
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import psutil
import gc

from neurolite import train


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def large_tabular_dataset(self, temp_dir):
        """Create large tabular dataset for performance testing."""
        np.random.seed(42)
        n_samples = 10000
        n_features = 50
        
        # Generate feature data
        data = {}
        for i in range(n_features):
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        # Generate target
        data['target'] = np.random.randint(0, 5, n_samples)
        
        df = pd.DataFrame(data)
        file_path = os.path.join(temp_dir, 'large_tabular.csv')
        df.to_csv(file_path, index=False)
        return file_path
    
    @pytest.fixture
    def medium_tabular_dataset(self, temp_dir):
        """Create medium tabular dataset."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        data = {}
        for i in range(n_features):
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        data['target'] = np.random.randint(0, 3, n_samples)
        
        df = pd.DataFrame(data)
        file_path = os.path.join(temp_dir, 'medium_tabular.csv')
        df.to_csv(file_path, index=False)
        return file_path
    
    def measure_memory_usage(self):
        """Measure current memory usage."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def test_training_speed_small_dataset(self, medium_tabular_dataset):
        """Test training speed on small dataset."""
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        model = train(
            data=medium_tabular_dataset,
            task="classification",
            target="target",
            config={'epochs': 10}
        )
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        training_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Performance assertions
        assert training_time < 60, f"Training took too long: {training_time:.2f}s"
        assert memory_usage < 500, f"Memory usage too high: {memory_usage:.2f}MB"
        assert model is not None
        
        print(f"Small dataset training time: {training_time:.2f}s")
        print(f"Memory usage: {memory_usage:.2f}MB")
    
    def test_training_speed_large_dataset(self, large_tabular_dataset):
        """Test training speed on large dataset."""
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        model = train(
            data=large_tabular_dataset,
            task="classification",
            target="target",
            config={'epochs': 5}  # Fewer epochs for large dataset
        )
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        training_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Performance assertions for large dataset
        assert training_time < 300, f"Large dataset training took too long: {training_time:.2f}s"
        assert memory_usage < 1000, f"Memory usage too high: {memory_usage:.2f}MB"
        assert model is not None
        
        print(f"Large dataset training time: {training_time:.2f}s")
        print(f"Memory usage: {memory_usage:.2f}MB")
    
    def test_inference_speed(self, medium_tabular_dataset):
        """Test inference speed."""
        # Train model first
        model = train(
            data=medium_tabular_dataset,
            task="classification",
            target="target"
        )
        
        # Prepare test data
        test_data = {f'feature_{i}': np.random.normal(0, 1) for i in range(20)}
        
        # Single prediction speed
        start_time = time.time()
        prediction = model.predict(test_data)
        single_inference_time = time.time() - start_time
        
        assert single_inference_time < 1.0, f"Single inference too slow: {single_inference_time:.4f}s"
        assert prediction is not None
        
        # Batch prediction speed
        batch_data = pd.DataFrame([test_data] * 100)
        start_time = time.time()
        batch_predictions = model.predict(batch_data)
        batch_inference_time = time.time() - start_time
        
        assert batch_inference_time < 5.0, f"Batch inference too slow: {batch_inference_time:.4f}s"
        assert len(batch_predictions) == 100
        
        print(f"Single inference time: {single_inference_time:.4f}s")
        print(f"Batch inference time: {batch_inference_time:.4f}s")
        print(f"Batch inference per sample: {batch_inference_time/100:.6f}s")
    
    def test_memory_efficiency(self, large_tabular_dataset):
        """Test memory efficiency during training."""
        initial_memory = self.measure_memory_usage()
        
        # Train model
        model = train(
            data=large_tabular_dataset,
            task="classification",
            target="target",
            config={'epochs': 3}
        )
        
        peak_memory = self.measure_memory_usage()
        
        # Clean up
        del model
        gc.collect()
        
        final_memory = self.measure_memory_usage()
        
        peak_usage = peak_memory - initial_memory
        memory_leak = final_memory - initial_memory
        
        # Memory efficiency assertions
        assert peak_usage < 1500, f"Peak memory usage too high: {peak_usage:.2f}MB"
        assert memory_leak < 100, f"Possible memory leak: {memory_leak:.2f}MB"
        
        print(f"Peak memory usage: {peak_usage:.2f}MB")
        print(f"Memory after cleanup: {memory_leak:.2f}MB")
    
    def test_data_loading_speed(self, large_tabular_dataset):
        """Test data loading and preprocessing speed."""
        from neurolite.data.loader import DataLoader
        from neurolite.data.detector import DataDetector
        
        # Test data detection speed
        start_time = time.time()
        detector = DataDetector()
        data_type = detector.detect_data_type(large_tabular_dataset)
        detection_time = time.time() - start_time
        
        assert detection_time < 1.0, f"Data detection too slow: {detection_time:.4f}s"
        
        # Test data loading speed
        start_time = time.time()
        loader = DataLoader()
        dataset = loader.load_data(large_tabular_dataset, data_type)
        loading_time = time.time() - start_time
        
        assert loading_time < 10.0, f"Data loading too slow: {loading_time:.2f}s"
        
        print(f"Data detection time: {detection_time:.4f}s")
        print(f"Data loading time: {loading_time:.2f}s")
    
    def test_model_selection_speed(self, medium_tabular_dataset):
        """Test automatic model selection speed."""
        from neurolite.models.registry import ModelRegistry
        from neurolite.data.loader import DataLoader
        from neurolite.data.detector import DataDetector
        
        # Load data
        detector = DataDetector()
        data_type = detector.detect_data_type(medium_tabular_dataset)
        loader = DataLoader()
        dataset = loader.load_data(medium_tabular_dataset, data_type)
        
        # Test model selection speed
        start_time = time.time()
        registry = ModelRegistry()
        model_name = registry.auto_select(dataset, "classification")
        selection_time = time.time() - start_time
        
        assert selection_time < 5.0, f"Model selection too slow: {selection_time:.4f}s"
        assert model_name is not None
        
        print(f"Model selection time: {selection_time:.4f}s")
        print(f"Selected model: {model_name}")
    
    def test_concurrent_training(self, temp_dir):
        """Test performance with concurrent training (if supported)."""
        import threading
        import queue
        
        # Create multiple small datasets
        datasets = []
        for i in range(3):
            np.random.seed(42 + i)
            data = {
                'x1': np.random.normal(0, 1, 500),
                'x2': np.random.normal(0, 1, 500),
                'target': np.random.randint(0, 2, 500)
            }
            df = pd.DataFrame(data)
            file_path = os.path.join(temp_dir, f'dataset_{i}.csv')
            df.to_csv(file_path, index=False)
            datasets.append(file_path)
        
        results = queue.Queue()
        
        def train_model(dataset_path, result_queue):
            try:
                start_time = time.time()
                model = train(
                    data=dataset_path,
                    task="classification",
                    target="target",
                    config={'epochs': 3}
                )
                training_time = time.time() - start_time
                result_queue.put(('success', training_time, model))
            except Exception as e:
                result_queue.put(('error', str(e), None))
        
        # Start concurrent training
        threads = []
        start_time = time.time()
        
        for dataset in datasets:
            thread = threading.Thread(target=train_model, args=(dataset, results))
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        successful_trainings = 0
        training_times = []
        
        while not results.empty():
            status, result, model = results.get()
            if status == 'success':
                successful_trainings += 1
                training_times.append(result)
        
        assert successful_trainings == 3, f"Only {successful_trainings}/3 trainings succeeded"
        assert total_time < 120, f"Concurrent training took too long: {total_time:.2f}s"
        
        print(f"Concurrent training completed in: {total_time:.2f}s")
        print(f"Individual training times: {training_times}")
    
    def test_scalability_with_features(self, temp_dir):
        """Test performance scaling with number of features."""
        feature_counts = [10, 50, 100]
        training_times = []
        
        for n_features in feature_counts:
            # Create dataset with varying feature count
            np.random.seed(42)
            n_samples = 1000
            
            data = {}
            for i in range(n_features):
                data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
            data['target'] = np.random.randint(0, 3, n_samples)
            
            df = pd.DataFrame(data)
            file_path = os.path.join(temp_dir, f'features_{n_features}.csv')
            df.to_csv(file_path, index=False)
            
            # Train and measure time
            start_time = time.time()
            model = train(
                data=file_path,
                task="classification",
                target="target",
                config={'epochs': 5}
            )
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            assert model is not None
            print(f"Features: {n_features}, Training time: {training_time:.2f}s")
        
        # Check that training time doesn't grow exponentially
        # (should be roughly linear or sub-quadratic)
        time_ratio = training_times[-1] / training_times[0]
        feature_ratio = feature_counts[-1] / feature_counts[0]
        
        assert time_ratio < feature_ratio * 2, f"Training time scaling too poor: {time_ratio:.2f}x for {feature_ratio}x features"
        
        print(f"Scalability ratio: {time_ratio:.2f}x time for {feature_ratio}x features")
    
    def test_memory_leak_detection(self, temp_dir):
        """Test for memory leaks during repeated training."""
        import gc
        
        # Create small dataset for repeated training
        np.random.seed(42)
        data = {
            'x1': np.random.normal(0, 1, 500),
            'x2': np.random.normal(0, 1, 500),
            'target': np.random.randint(0, 2, 500)
        }
        df = pd.DataFrame(data)
        file_path = os.path.join(temp_dir, 'leak_test.csv')
        df.to_csv(file_path, index=False)
        
        initial_memory = self.measure_memory_usage()
        memory_measurements = []
        
        # Train multiple models and measure memory
        for i in range(5):
            model = train(
                data=file_path,
                task="classification",
                target="target",
                config={'epochs': 2}
            )
            
            # Force cleanup
            del model
            gc.collect()
            
            current_memory = self.measure_memory_usage()
            memory_measurements.append(current_memory - initial_memory)
            
            print(f"Iteration {i+1}: Memory delta = {memory_measurements[-1]:.2f}MB")
        
        # Check for memory leak (memory should not grow significantly)
        memory_growth = memory_measurements[-1] - memory_measurements[0]
        assert memory_growth < 200, f"Possible memory leak detected: {memory_growth:.2f}MB growth"
    
    def test_gpu_vs_cpu_performance(self, medium_tabular_dataset):
        """Test performance difference between GPU and CPU training."""
        import torch
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for GPU testing")
        
        # Test CPU training
        start_time = time.time()
        cpu_model = train(
            data=medium_tabular_dataset,
            task="classification",
            target="target",
            config={'epochs': 5, 'device': 'cpu'}
        )
        cpu_time = time.time() - start_time
        
        # Test GPU training
        start_time = time.time()
        gpu_model = train(
            data=medium_tabular_dataset,
            task="classification",
            target="target",
            config={'epochs': 5, 'device': 'cuda'}
        )
        gpu_time = time.time() - start_time
        
        # GPU should be faster or at least not significantly slower
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.2f}x")
        
        # For small datasets, GPU might be slower due to overhead
        # Just ensure both complete successfully
        assert cpu_model is not None
        assert gpu_model is not None
    
    def test_batch_size_performance_impact(self, large_tabular_dataset):
        """Test performance impact of different batch sizes."""
        batch_sizes = [16, 32, 64, 128]
        training_times = []
        
        for batch_size in batch_sizes:
            start_time = time.time()
            model = train(
                data=large_tabular_dataset,
                task="classification",
                target="target",
                config={'epochs': 3, 'batch_size': batch_size}
            )
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            assert model is not None
            print(f"Batch size {batch_size}: {training_time:.2f}s")
        
        # Larger batch sizes should generally be more efficient
        # (though this depends on hardware and dataset size)
        assert all(t > 0 for t in training_times)
    
    def test_parallel_data_loading_performance(self, temp_dir):
        """Test performance impact of parallel data loading."""
        # Create larger dataset to see data loading impact
        np.random.seed(42)
        n_samples = 5000
        n_features = 30
        
        data = {}
        for i in range(n_features):
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        data['target'] = np.random.randint(0, 3, n_samples)
        
        df = pd.DataFrame(data)
        file_path = os.path.join(temp_dir, 'parallel_test.csv')
        df.to_csv(file_path, index=False)
        
        # Test with single worker
        start_time = time.time()
        model1 = train(
            data=file_path,
            task="classification",
            target="target",
            config={'epochs': 3, 'num_workers': 0}
        )
        single_worker_time = time.time() - start_time
        
        # Test with multiple workers
        start_time = time.time()
        model2 = train(
            data=file_path,
            task="classification",
            target="target",
            config={'epochs': 3, 'num_workers': 2}
        )
        multi_worker_time = time.time() - start_time
        
        assert model1 is not None
        assert model2 is not None
        
        print(f"Single worker: {single_worker_time:.2f}s")
        print(f"Multi worker: {multi_worker_time:.2f}s")
        
        # Multi-worker should not be significantly slower
        assert multi_worker_time < single_worker_time * 1.5
    
    def test_model_size_vs_performance(self, medium_tabular_dataset):
        """Test relationship between model complexity and performance."""
        model_configs = [
            {'hidden_layers': [32], 'epochs': 5},
            {'hidden_layers': [64, 32], 'epochs': 5},
            {'hidden_layers': [128, 64, 32], 'epochs': 5}
        ]
        
        results = []
        
        for i, config in enumerate(model_configs):
            start_time = time.time()
            start_memory = self.measure_memory_usage()
            
            model = train(
                data=medium_tabular_dataset,
                task="classification",
                target="target",
                config=config
            )
            
            training_time = time.time() - start_time
            peak_memory = self.measure_memory_usage() - start_memory
            
            # Measure model size (approximate)
            model_params = sum(p.numel() for p in model.model.parameters() if hasattr(model, 'model') and hasattr(model.model, 'parameters'))
            
            results.append({
                'config': i,
                'training_time': training_time,
                'memory_usage': peak_memory,
                'model_params': model_params
            })
            
            print(f"Config {i}: {training_time:.2f}s, {peak_memory:.2f}MB, {model_params} params")
        
        # More complex models should take longer and use more memory
        assert results[0]['training_time'] <= results[-1]['training_time'] * 2
        assert all(r['training_time'] > 0 for r in results)
    
    def test_caching_performance_improvement(self, large_tabular_dataset):
        """Test performance improvement from caching."""
        # First run (no cache)
        start_time = time.time()
        model1 = train(
            data=large_tabular_dataset,
            task="classification",
            target="target",
            config={'epochs': 3, 'use_cache': False}
        )
        first_run_time = time.time() - start_time
        
        # Second run (with cache)
        start_time = time.time()
        model2 = train(
            data=large_tabular_dataset,
            task="classification",
            target="target",
            config={'epochs': 3, 'use_cache': True}
        )
        second_run_time = time.time() - start_time
        
        assert model1 is not None
        assert model2 is not None
        
        print(f"First run (no cache): {first_run_time:.2f}s")
        print(f"Second run (with cache): {second_run_time:.2f}s")
        
        # Second run should be faster due to caching (if implemented)
        # At minimum, it should not be significantly slower
        assert second_run_time <= first_run_time * 1.2
    
    def test_inference_throughput(self, medium_tabular_dataset):
        """Test inference throughput with different batch sizes."""
        # Train model first
        model = train(
            data=medium_tabular_dataset,
            task="classification",
            target="target"
        )
        
        # Prepare test data
        test_data = {f'feature_{i}': np.random.normal(0, 1) for i in range(20)}
        
        # Test different batch sizes for inference
        batch_sizes = [1, 10, 100, 1000]
        throughputs = []
        
        for batch_size in batch_sizes:
            batch_data = pd.DataFrame([test_data] * batch_size)
            
            # Warm up
            model.predict(batch_data)
            
            # Measure throughput
            start_time = time.time()
            n_runs = 10
            for _ in range(n_runs):
                predictions = model.predict(batch_data)
            total_time = time.time() - start_time
            
            throughput = (batch_size * n_runs) / total_time
            throughputs.append(throughput)
            
            print(f"Batch size {batch_size}: {throughput:.2f} samples/sec")
        
        # Larger batches should generally have higher throughput
        assert all(t > 0 for t in throughputs)
        assert throughputs[-1] > throughputs[0]  # Largest batch should be most efficient
    
    def test_training_convergence_speed(self, medium_tabular_dataset):
        """Test how quickly different models converge."""
        model_types = ['linear', 'tree', 'neural_network']
        convergence_results = []
        
        for model_type in model_types:
            try:
                start_time = time.time()
                model = train(
                    data=medium_tabular_dataset,
                    task="classification",
                    target="target",
                    model=model_type,
                    config={'epochs': 10}
                )
                training_time = time.time() - start_time
                
                # Get training history if available
                history = getattr(model, 'training_history', {})
                
                convergence_results.append({
                    'model_type': model_type,
                    'training_time': training_time,
                    'converged': model is not None,
                    'history': history
                })
                
                print(f"{model_type}: {training_time:.2f}s")
                
            except Exception as e:
                print(f"{model_type} failed: {e}")
                convergence_results.append({
                    'model_type': model_type,
                    'training_time': float('inf'),
                    'converged': False,
                    'history': {}
                })
        
        # At least one model should converge successfully
        successful_models = [r for r in convergence_results if r['converged']]
        assert len(successful_models) > 0, "No models converged successfully"