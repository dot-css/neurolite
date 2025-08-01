# Troubleshooting Guide

This guide helps you resolve common issues when using NeuroLite.

## Installation Issues

### Problem: `pip install neurolite` fails

**Symptoms:**
- Package not found error
- Permission denied errors
- Dependency conflicts

**Solutions:**

1. **Update pip and try again:**
   ```bash
   pip install --upgrade pip
   pip install neurolite
   ```

2. **Use virtual environment:**
   ```bash
   python -m venv neurolite_env
   source neurolite_env/bin/activate  # On Windows: neurolite_env\Scripts\activate
   pip install neurolite
   ```

3. **Install from source:**
   ```bash
   git clone https://github.com/dot-css/neurolite.git
   cd neurolite
   pip install -e .
   ```

### Problem: Missing dependencies

**Symptoms:**
- ImportError for specific packages
- ModuleNotFoundError

**Solutions:**

1. **Install optional dependencies:**
   ```bash
   pip install neurolite[vision]  # For computer vision
   pip install neurolite[nlp]     # For NLP
   pip install neurolite[all]     # For all features
   ```

2. **Manual dependency installation:**
   ```bash
   pip install torch torchvision  # For PyTorch models
   pip install transformers       # For NLP models
   pip install opencv-python      # For computer vision
   ```

## Data Loading Issues

### Problem: "Data path does not exist"

**Symptoms:**
```
ConfigurationError: Data path does not exist: /path/to/data
```

**Solutions:**

1. **Check file path:**
   ```python
   import os
   print(os.path.exists('your_data_path'))
   print(os.listdir('.'))  # List current directory
   ```

2. **Use absolute paths:**
   ```python
   import os
   data_path = os.path.abspath('data/my_dataset.csv')
   model = neurolite.train(data=data_path)
   ```

3. **Verify file permissions:**
   ```bash
   ls -la your_data_file.csv
   chmod 644 your_data_file.csv  # Fix permissions if needed
   ```

### Problem: "Cannot detect data type"

**Symptoms:**
```
DataError: Cannot automatically detect data type for: /path/to/data
```

**Solutions:**

1. **Specify task explicitly:**
   ```python
   model = neurolite.train(
       data='data.csv',
       task='classification',  # Specify task
       target='label'
   )
   ```

2. **Check data format:**
   - **Images:** Organize in subdirectories by class
   - **Text:** Use CSV with text and label columns
   - **Tabular:** Use CSV with proper headers

3. **Verify file formats:**
   ```python
   import pandas as pd
   df = pd.read_csv('your_data.csv')
   print(df.head())
   print(df.columns)
   ```

### Problem: "Empty dataset after loading"

**Symptoms:**
```
DataError: Dataset is empty after loading and preprocessing
```

**Solutions:**

1. **Check data file content:**
   ```python
   import pandas as pd
   df = pd.read_csv('data.csv')
   print(f"Dataset shape: {df.shape}")
   print(f"Null values: {df.isnull().sum()}")
   ```

2. **Verify image directory structure:**
   ```
   data/
   ├── class1/
   │   ├── image1.jpg
   │   └── image2.jpg
   └── class2/
       ├── image1.jpg
       └── image2.jpg
   ```

3. **Check file extensions:**
   - Images: `.jpg`, `.jpeg`, `.png`, `.bmp`
   - Text: `.csv`, `.tsv`

## Training Issues

### Problem: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

1. **Reduce batch size:**
   ```python
   model = neurolite.train(
       data='data/',
       batch_size=16,  # Reduce from default 32
   )
   ```

2. **Use smaller image size:**
   ```python
   model = neurolite.train(
       data='images/',
       image_size=128,  # Reduce from default 224
   )
   ```

3. **Enable gradient checkpointing:**
   ```python
   model = neurolite.train(
       data='data/',
       gradient_checkpointing=True,
   )
   ```

4. **Use CPU training:**
   ```python
   model = neurolite.train(
       data='data/',
       device='cpu',
   )
   ```

### Problem: Training is very slow

**Symptoms:**
- Training takes much longer than expected
- Low GPU utilization

**Solutions:**

1. **Increase batch size:**
   ```python
   model = neurolite.train(
       data='data/',
       batch_size=64,  # Increase if memory allows
   )
   ```

2. **Use multiple workers:**
   ```python
   model = neurolite.train(
       data='data/',
       num_workers=4,  # Parallel data loading
   )
   ```

3. **Enable mixed precision:**
   ```python
   model = neurolite.train(
       data='data/',
       mixed_precision=True,
   )
   ```

4. **Use faster model:**
   ```python
   model = neurolite.train(
       data='data/',
       model='distilbert',  # Instead of 'bert'
   )
   ```

### Problem: "Model not converging"

**Symptoms:**
- Loss not decreasing
- Accuracy stuck at low values
- Training metrics not improving

**Solutions:**

1. **Check learning rate:**
   ```python
   model = neurolite.train(
       data='data/',
       learning_rate=1e-4,  # Try different values
   )
   ```

2. **Enable hyperparameter optimization:**
   ```python
   model = neurolite.train(
       data='data/',
       optimize=True,  # Let NeuroLite find best parameters
   )
   ```

3. **Increase training epochs:**
   ```python
   model = neurolite.train(
       data='data/',
       epochs=50,  # More training time
   )
   ```

4. **Check data quality:**
   ```python
   # Verify labels are correct
   import pandas as pd
   df = pd.read_csv('data.csv')
   print(df['label'].value_counts())
   ```

## Model Performance Issues

### Problem: Low accuracy on test data

**Symptoms:**
- High training accuracy but low test accuracy
- Model performs poorly on new data

**Solutions:**

1. **Check for overfitting:**
   ```python
   model = neurolite.train(
       data='data/',
       validation_split=0.2,  # Monitor validation performance
       early_stopping=True,   # Stop when validation stops improving
   )
   ```

2. **Increase dataset size:**
   - Collect more training data
   - Use data augmentation for images
   - Apply text augmentation for NLP

3. **Try different models:**
   ```python
   # For tabular data
   model = neurolite.train(data='data.csv', model='xgboost')
   
   # For images
   model = neurolite.train(data='images/', model='efficientnet')
   
   # For text
   model = neurolite.train(data='text.csv', model='roberta')
   ```

4. **Adjust data splits:**
   ```python
   model = neurolite.train(
       data='data/',
       validation_split=0.15,  # More data for training
       test_split=0.15,
   )
   ```

### Problem: Inconsistent predictions

**Symptoms:**
- Same input gives different outputs
- Model predictions vary between runs

**Solutions:**

1. **Set random seed:**
   ```python
   import random
   import numpy as np
   import torch
   
   random.seed(42)
   np.random.seed(42)
   torch.manual_seed(42)
   
   model = neurolite.train(data='data/', random_state=42)
   ```

2. **Use deterministic algorithms:**
   ```python
   model = neurolite.train(
       data='data/',
       deterministic=True,
   )
   ```

## Deployment Issues

### Problem: "Port already in use"

**Symptoms:**
```
OSError: [Errno 98] Address already in use
```

**Solutions:**

1. **Use different port:**
   ```python
   api_server = neurolite.deploy(model, format='api', port=8081)
   ```

2. **Kill existing process:**
   ```bash
   # Find process using port 8000
   lsof -i :8000
   
   # Kill the process
   kill -9 <PID>
   ```

3. **Use automatic port selection:**
   ```python
   api_server = neurolite.deploy(model, format='api', port=0)  # Auto-select port
   ```

### Problem: Model export fails

**Symptoms:**
```
DeploymentError: Failed to export model to ONNX format
```

**Solutions:**

1. **Check model compatibility:**
   ```python
   # Some models may not support all export formats
   try:
       exported = neurolite.deploy(model, format='onnx')
   except Exception as e:
       print(f"ONNX export failed: {e}")
       # Try different format
       exported = neurolite.deploy(model, format='pickle')
   ```

2. **Install export dependencies:**
   ```bash
   pip install onnx onnxruntime  # For ONNX export
   pip install tensorflow        # For TFLite export
   ```

3. **Use alternative export format:**
   ```python
   # If ONNX fails, try TorchScript
   exported = neurolite.deploy(model, format='torchscript')
   ```

## Performance Issues

### Problem: Slow inference

**Symptoms:**
- Predictions take too long
- High latency in production

**Solutions:**

1. **Optimize model:**
   ```python
   optimized_model = neurolite.deploy(
       model,
       format='onnx',
       optimize=True,  # Apply optimizations
   )
   ```

2. **Use quantization:**
   ```python
   quantized_model = neurolite.deploy(
       model,
       format='tflite',
       quantize=True,  # Reduce model size
   )
   ```

3. **Batch predictions:**
   ```python
   # Instead of single predictions
   predictions = model.predict([text1, text2, text3])  # Batch processing
   ```

4. **Use GPU for inference:**
   ```python
   model = neurolite.train(data='data/', device='cuda')
   # Model will use GPU for predictions automatically
   ```

## Memory Issues

### Problem: "Out of memory during data loading"

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Enable lazy loading:**
   ```python
   model = neurolite.train(
       data='large_dataset/',
       lazy_loading=True,  # Load data on-demand
   )
   ```

2. **Reduce batch size:**
   ```python
   model = neurolite.train(
       data='data/',
       batch_size=8,  # Smaller batches
   )
   ```

3. **Use data streaming:**
   ```python
   model = neurolite.train(
       data='data/',
       streaming=True,  # Stream data from disk
   )
   ```

## Common Error Messages

### ConfigurationError

**Cause:** Invalid parameters or configuration
**Solution:** Check parameter values and types

### DataError

**Cause:** Issues with data loading or preprocessing
**Solution:** Verify data format and file paths

### ModelError

**Cause:** Model-related issues
**Solution:** Try different model or check model availability

### TrainingError

**Cause:** Training process failures
**Solution:** Check data quality and training parameters

### DeploymentError

**Cause:** Deployment failures
**Solution:** Verify export format compatibility and dependencies

## Getting Help

If you're still experiencing issues:

1. **Check the logs:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   # Run your code to see detailed logs
   ```

2. **Enable verbose mode:**
   ```python
   model = neurolite.train(data='data/', verbose=True)
   ```

3. **Create minimal reproducible example:**
   ```python
   import neurolite
   
   # Minimal code that reproduces the issue
   model = neurolite.train('simple_data.csv')
   ```

4. **Check system requirements:**
   - Python 3.7+
   - Sufficient RAM (8GB+ recommended)
   - GPU with CUDA support (optional but recommended)

5. **Update NeuroLite:**
   ```bash
   pip install --upgrade neurolite
   ```

6. **Report the issue:**
   - GitHub Issues: https://github.com/neurolite/neurolite/issues
   - Include error message, code snippet, and system info
   - Provide sample data if possible

## Performance Tips

### General Optimization

1. **Use appropriate hardware:**
   - GPU for deep learning models
   - Multi-core CPU for traditional ML
   - Sufficient RAM for large datasets

2. **Optimize data pipeline:**
   - Use efficient data formats (Parquet instead of CSV)
   - Preprocess data once and cache results
   - Use parallel data loading

3. **Choose right model:**
   - Start with simple models for small datasets
   - Use pre-trained models when available
   - Consider model size vs. accuracy trade-offs

4. **Monitor resource usage:**
   ```python
   import psutil
   import GPUtil
   
   # Check CPU and memory usage
   print(f"CPU: {psutil.cpu_percent()}%")
   print(f"Memory: {psutil.virtual_memory().percent}%")
   
   # Check GPU usage (if available)
   gpus = GPUtil.getGPUs()
   for gpu in gpus:
       print(f"GPU {gpu.id}: {gpu.load*100:.1f}%")
   ```

### Domain-Specific Tips

**Computer Vision:**
- Use appropriate image sizes (224x224 for most models)
- Apply data augmentation for small datasets
- Consider transfer learning for better performance

**NLP:**
- Use appropriate sequence lengths
- Consider model size vs. speed trade-offs
- Use caching for tokenized data

**Tabular Data:**
- Handle missing values appropriately
- Use feature engineering
- Consider ensemble methods for better accuracy

Remember: NeuroLite is designed to handle most issues automatically, but understanding these troubleshooting steps will help you resolve edge cases and optimize performance for your specific use case.