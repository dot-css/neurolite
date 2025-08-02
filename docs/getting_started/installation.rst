Installation Guide
==================

This guide will help you install NeuroLite and get it running on your system.

System Requirements
-------------------

Minimum Requirements
~~~~~~~~~~~~~~~~~~~~
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **OS**: Windows 10, macOS 10.14, Ubuntu 18.04

Recommended Requirements
~~~~~~~~~~~~~~~~~~~~~~~~
- **Python**: 3.9 or higher
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support
- **Storage**: 10GB free space (for model cache)
- **OS**: Latest stable versions

Installation Methods
--------------------

Quick Install (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install neurolite

This will install NeuroLite with all core dependencies.

Development Install
~~~~~~~~~~~~~~~~~~~

If you want to contribute to NeuroLite or use the latest features:

.. code-block:: bash

   git clone https://github.com/dot-css/neurolite.git
   cd neurolite
   pip install -e ".[dev]"

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

NeuroLite supports optional dependencies for extended functionality:

.. code-block:: bash

   # For TensorFlow support
   pip install neurolite[tensorflow]

   # For XGBoost support  
   pip install neurolite[xgboost]

   # For documentation building
   pip install neurolite[docs]

   # Install everything
   pip install neurolite[all]

GPU Support
-----------

CUDA Installation
~~~~~~~~~~~~~~~~~

For GPU acceleration, install CUDA-compatible PyTorch:

.. code-block:: bash

   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Verify GPU Support
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA devices: {torch.cuda.device_count()}")

Docker Installation
-------------------

You can also run NeuroLite in a Docker container:

.. code-block:: bash

   # Pull the official image
   docker pull neurolite/neurolite:latest

   # Run with GPU support
   docker run --gpus all -it neurolite/neurolite:latest

Verification
------------

Test your installation:

.. code-block:: python

   import neurolite
   print(f"NeuroLite version: {neurolite.__version__}")

   # Test basic functionality
   from neurolite import train
   print("Installation successful!")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'neurolite'**

- Make sure you've activated the correct Python environment
- Try reinstalling: ``pip uninstall neurolite && pip install neurolite``

**CUDA out of memory**

- Reduce batch size in your training configuration
- Use CPU training: ``neurolite.config.set_device("cpu")``

**Permission denied errors**

- Use ``pip install --user neurolite`` for user-level installation
- Or use a virtual environment

Virtual Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend using a virtual environment:

.. code-block:: bash

   # Create virtual environment
   python -m venv neurolite-env

   # Activate (Windows)
   neurolite-env\Scripts\activate

   # Activate (macOS/Linux)
   source neurolite-env/bin/activate

   # Install NeuroLite
   pip install neurolite

Next Steps
----------

Now that you have NeuroLite installed, check out:

- :doc:`quickstart` - Your first model in 5 minutes
- :doc:`basic_concepts` - Understanding NeuroLite concepts
- :doc:`../tutorials/image_classification` - Hands-on tutorials