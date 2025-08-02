NeuroLite Documentation
=======================

.. raw:: html

   <div class="quick-start">
   <h2>🚀 Welcome to NeuroLite</h2>
   <p>The revolutionary AI/ML/DL/NLP productivity library that enables you to build, train, and deploy machine learning models with <strong>minimal code</strong>.</p>
   </div>

⚡ Quick Start
--------------

Transform complex ML workflows into simple operations:

.. code-block:: python

   import neurolite

   # Train any model with just one line
   model = neurolite.train('your_data.csv', task='classification')

   # Deploy it instantly
   neurolite.deploy(model, format='api', port=8080)

.. raw:: html

   <div class="installation-section">
   <h3>📦 Installation</h3>
   <pre><code>pip install neurolite</code></pre>
   <p>Or for the latest development version:</p>
   <pre><code>pip install git+https://github.com/dot-css/neurolite.git</code></pre>
   </div>

🌟 Key Features
---------------

.. raw:: html

   <div class="feature-card">
   <h3>🎯 Minimal Code Interface</h3>
   <p>Train state-of-the-art models with less than 10 lines of code. No complex configurations or boilerplate required.</p>
   </div>

   <div class="feature-card">
   <h3>🤖 Auto-Everything</h3>
   <p>Automatic data processing, model selection, hyperparameter tuning, and feature engineering.</p>
   </div>

   <div class="feature-card">
   <h3>🌍 Multi-Domain Support</h3>
   <p>Unified interface for Computer Vision, NLP, and Traditional ML with consistent APIs.</p>
   </div>

   <div class="feature-card">
   <h3>⚡ Production Ready</h3>
   <p>One-click deployment to multiple platforms including AWS, GCP, Azure, and Docker.</p>
   </div>

📚 Documentation Contents
-------------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/basic_concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/computer_vision
   user_guide/nlp
   user_guide/traditional_ml
   user_guide/deployment
   user_guide/hyperparameter_optimization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/models
   api/data
   api/deployment
   api/evaluation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/image_classification
   tutorials/text_classification
   tutorials/tabular_data
   tutorials/custom_models
   tutorials/production_deployment

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic_examples
   examples/advanced_examples
   examples/business_cases
   examples/research_applications

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/plugin_development
   advanced/performance_optimization
   advanced/custom_workflows
   advanced/monitoring

.. toctree::
   :maxdepth: 1
   :caption: Help & Support

   troubleshooting
   faq
   contributing
   changelog

📈 Performance Benchmarks
-------------------------

.. raw:: html

   <table class="performance-table">
   <thead>
   <tr>
   <th>Task Type</th>
   <th>Dataset Size</th>
   <th>Training Time</th>
   <th>Accuracy</th>
   </tr>
   </thead>
   <tbody>
   <tr>
   <td>Image Classification</td>
   <td>10K images</td>
   <td>15 minutes</td>
   <td>94.2%</td>
   </tr>
   <tr>
   <td>Text Classification</td>
   <td>50K documents</td>
   <td>8 minutes</td>
   <td>91.7%</td>
   </tr>
   <tr>
   <td>Tabular Classification</td>
   <td>100K rows</td>
   <td>3 minutes</td>
   <td>89.5%</td>
   </tr>
   <tr>
   <td>Object Detection</td>
   <td>5K images</td>
   <td>25 minutes</td>
   <td>87.3%</td>
   </tr>
   <tr>
   <td>Sentiment Analysis</td>
   <td>25K reviews</td>
   <td>12 minutes</td>
   <td>93.1%</td>
   </tr>
   </tbody>
   </table>

   <p><em>Benchmarks run on NVIDIA RTX 3080, Intel i7-10700K, 32GB RAM</em></p>

🎯 Use Cases by Industry
------------------------

Business & E-commerce
~~~~~~~~~~~~~~~~~~~~~
- Customer sentiment analysis
- Product recommendation systems
- Fraud detection
- Price optimization
- Demand forecasting

Healthcare & Life Sciences
~~~~~~~~~~~~~~~~~~~~~~~~~
- Medical image analysis
- Drug discovery
- Disease prediction
- Clinical trial optimization
- Genomics analysis

Technology & Software
~~~~~~~~~~~~~~~~~~~~
- Code analysis and generation
- Bug detection
- Performance optimization
- User behavior analysis
- Automated testing

Research & Academia
~~~~~~~~~~~~~~~~~~
- Scientific paper classification
- Experiment analysis
- Literature review automation
- Data mining
- Statistical modeling

🛠️ Supported Technologies
--------------------------

Machine Learning Frameworks
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **PyTorch** - Deep learning models
- **TensorFlow** - Neural networks and deployment
- **Scikit-learn** - Traditional ML algorithms
- **XGBoost** - Gradient boosting
- **Hugging Face Transformers** - NLP models

Data Formats
~~~~~~~~~~~~
- **Images**: JPEG, PNG, BMP, TIFF
- **Text**: CSV, TSV, JSON, TXT
- **Tabular**: CSV, Excel, Parquet
- **Audio**: WAV, MP3, FLAC (coming soon)
- **Video**: MP4, AVI, MOV (coming soon)

Deployment Options
~~~~~~~~~~~~~~~~~
- **REST API** - Flask/FastAPI servers
- **ONNX** - Cross-platform inference
- **TensorFlow Lite** - Mobile deployment
- **TorchScript** - PyTorch production
- **Docker** - Containerized deployment

🤝 Community & Support
----------------------

Getting Help
~~~~~~~~~~~~
1. **Documentation**: Start with this documentation
2. **Examples**: Check the examples gallery
3. **Troubleshooting**: Read the troubleshooting guide
4. **GitHub Issues**: Report bugs and request features
5. **Discussions**: Join community discussions

Contributing
~~~~~~~~~~~~
We welcome contributions! See our :doc:`contributing` guide for:

- Code contributions
- Documentation improvements
- Bug reports
- Feature requests
- Example submissions

License
~~~~~~~
NeuroLite is released under the MIT License. See `LICENSE <https://github.com/dot-css/neurolite/blob/main/LICENSE>`_ for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`