NeuroLite Documentation
=======================

Welcome to NeuroLite's documentation! NeuroLite is an automated AI/ML library that intelligently detects data types and automatically applies the best models with minimal code.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api_reference
   examples
   contributing
   changelog

Features
--------

* **Intelligent Data Detection**: Automatic file format and data structure identification
* **Quality Assessment**: Comprehensive data quality analysis with missing data pattern detection
* **Type Classification**: Smart column type detection with confidence scoring
* **Model Recommendations**: Automated ML model suggestions based on data characteristics
* **Performance Optimized**: Handles datasets up to 1GB with sub-5-second analysis time

Quick Start
-----------

Install NeuroLite:

.. code-block:: bash

   pip install neurolite

Analyze your data in 3 lines:

.. code-block:: python

   from neurolite import DataProfiler
   
   profiler = DataProfiler()
   report = profiler.analyze('your_data.csv')
   
   print(f"Quality score: {report.quality_metrics.completeness:.2%}")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`