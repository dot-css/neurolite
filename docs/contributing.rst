Contributing to NeuroLite
========================

We welcome contributions to NeuroLite! This guide will help you get started with contributing to the project.

Getting Started
---------------

Types of Contributions
~~~~~~~~~~~~~~~~~~~~~~

We welcome several types of contributions:

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit bug fixes or new features
- **Documentation**: Improve or add to our documentation
- **Examples**: Create tutorials and example code
- **Testing**: Help improve our test coverage

How to Contribute
~~~~~~~~~~~~~~~~~

1. **Fork the Repository**: Create a fork of the NeuroLite repository
2. **Create a Branch**: Create a feature branch for your changes
3. **Make Changes**: Implement your changes with tests
4. **Submit a Pull Request**: Submit your changes for review

Development Setup
-----------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

Setting Up Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Clone your fork**:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/neurolite.git
      cd neurolite

2. **Create a virtual environment**:

   .. code-block:: bash

      python -m venv neurolite-dev
      source neurolite-dev/bin/activate  # On Windows: neurolite-dev\Scripts\activate

3. **Install in development mode**:

   .. code-block:: bash

      pip install -e ".[dev]"

4. **Install pre-commit hooks**:

   .. code-block:: bash

      pre-commit install

5. **Verify installation**:

   .. code-block:: python

      import neurolite
      print(f"NeuroLite version: {neurolite.__version__}")

Development Workflow
--------------------

Creating a Feature Branch
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number-description

Making Changes
~~~~~~~~~~~~~~

1. **Write Code**: Implement your changes following our coding standards
2. **Add Tests**: Write tests for your new functionality
3. **Update Documentation**: Update relevant documentation
4. **Run Tests**: Ensure all tests pass

.. code-block:: bash

   # Run tests
   pytest tests/ -v
   
   # Run specific test file
   pytest tests/test_core.py -v
   
   # Run with coverage
   pytest tests/ --cov=neurolite --cov-report=html

Code Quality Checks
~~~~~~~~~~~~~~~~~~~

We use several tools to maintain code quality:

.. code-block:: bash

   # Format code with Black
   black neurolite/ tests/
   
   # Check code style with Flake8
   flake8 neurolite/ tests/
   
   # Type checking with MyPy
   mypy neurolite/
   
   # Run all checks
   pre-commit run --all-files

Submitting Changes
~~~~~~~~~~~~~~~~~~

1. **Commit your changes**:

   .. code-block:: bash

      git add .
      git commit -m "feat: add new feature description"

2. **Push to your fork**:

   .. code-block:: bash

      git push origin feature/your-feature-name

3. **Create a Pull Request**: Go to GitHub and create a pull request

Coding Standards
----------------

Code Style
~~~~~~~~~~

We follow PEP 8 with some modifications:

- **Line Length**: 88 characters (Black default)
- **Imports**: Use absolute imports, group by standard/third-party/local
- **Naming**: Use descriptive names, follow Python conventions
- **Comments**: Write clear, concise comments and docstrings

Example Code Style
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   """Module docstring describing the module purpose."""

   import os
   from typing import Dict, List, Optional, Union

   import numpy as np
   import pandas as pd
   import torch

   from neurolite.core.base import BaseModel
   from neurolite.utils.validation import validate_data


   class ExampleModel(BaseModel):
       """Example model class with proper documentation.
       
       Args:
           param1: Description of parameter 1
           param2: Description of parameter 2
           
       Attributes:
           attribute1: Description of attribute 1
           attribute2: Description of attribute 2
       """
       
       def __init__(self, param1: str, param2: Optional[int] = None):
           super().__init__()
           self.param1 = param1
           self.param2 = param2 or 10
           
       def train(self, data: Union[str, pd.DataFrame]) -> Dict[str, float]:
           """Train the model on provided data.
           
           Args:
               data: Training data as file path or DataFrame
               
           Returns:
               Dictionary containing training metrics
               
           Raises:
               ValueError: If data is invalid
           """
           validated_data = validate_data(data)
           
           # Implementation here
           metrics = {"accuracy": 0.95, "loss": 0.05}
           
           return metrics

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Include type hints for all functions
- **Examples**: Include usage examples in docstrings
- **API Documentation**: Document all public APIs

Testing Guidelines
------------------

Test Structure
~~~~~~~~~~~~~~

We use pytest for testing. Tests are organized as follows:

.. code-block:: text

   tests/
   ├── unit/           # Unit tests
   ├── integration/    # Integration tests
   ├── fixtures/       # Test fixtures and data
   └── conftest.py     # Pytest configuration

Writing Tests
~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   import pandas as pd
   from neurolite import train
   from neurolite.exceptions import DataError


   class TestTrainFunction:
       """Test cases for the train function."""
       
       def test_train_with_csv_data(self, sample_csv_data):
           """Test training with CSV data."""
           model = train(data=sample_csv_data, task="classification")
           
           assert model is not None
           assert hasattr(model, 'predict')
           
       def test_train_with_invalid_data(self):
           """Test training with invalid data raises appropriate error."""
           with pytest.raises(DataError):
               train(data="nonexistent.csv", task="classification")
               
       @pytest.mark.parametrize("task", ["classification", "regression"])
       def test_train_different_tasks(self, sample_data, task):
           """Test training with different task types."""
           model = train(data=sample_data, task=task)
           assert model.task == task

Test Fixtures
~~~~~~~~~~~~~

Create reusable test fixtures:

.. code-block:: python

   # conftest.py
   import pytest
   import pandas as pd
   import tempfile
   import os


   @pytest.fixture
   def sample_csv_data():
       """Create sample CSV data for testing."""
       data = pd.DataFrame({
           'feature1': [1, 2, 3, 4, 5],
           'feature2': [2, 4, 6, 8, 10],
           'label': ['A', 'B', 'A', 'B', 'A']
       })
       
       with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
           data.to_csv(f.name, index=False)
           yield f.name
           
       os.unlink(f.name)

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run with verbose output
   pytest -v
   
   # Run specific test file
   pytest tests/unit/test_core.py
   
   # Run tests matching pattern
   pytest -k "test_train"
   
   # Run with coverage
   pytest --cov=neurolite
   
   # Run only fast tests (skip slow integration tests)
   pytest -m "not slow"

Documentation Contributions
---------------------------

Documentation Types
~~~~~~~~~~~~~~~~~~~

- **API Documentation**: Automatically generated from docstrings
- **User Guides**: Step-by-step guides for users
- **Tutorials**: Hands-on learning materials
- **Examples**: Code examples and use cases

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install documentation dependencies
   pip install -e ".[docs]"
   
   # Build documentation
   cd docs
   sphinx-build -b html . _build/html
   
   # Serve documentation locally
   python -m http.server 8000 -d _build/html

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

- **Clear Structure**: Use clear headings and organization
- **Code Examples**: Include working code examples
- **Screenshots**: Add screenshots for UI elements
- **Cross-References**: Link to related sections

Example Documentation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rst

   Example Feature
   ===============
   
   This section describes how to use the example feature.
   
   Basic Usage
   -----------
   
   Here's how to use the basic functionality:
   
   .. code-block:: python
   
      import neurolite
      
      # Create and train a model
      model = neurolite.train('data.csv', task='classification')
      
      # Make predictions
      predictions = model.predict(new_data)
   
   Advanced Usage
   --------------
   
   For more advanced use cases:
   
   .. code-block:: python
   
      model = neurolite.train(
          data='data.csv',
          task='classification',
          config={
              'model_type': 'neural_network',
              'epochs': 100
          }
      )

Bug Reports
-----------

How to Report Bugs
~~~~~~~~~~~~~~~~~~

1. **Check Existing Issues**: Search for existing bug reports
2. **Create Detailed Report**: Include all relevant information
3. **Provide Reproduction Steps**: Clear steps to reproduce the bug
4. **Include Environment Info**: Python version, OS, etc.

Bug Report Template
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   **Bug Description**
   A clear description of the bug.
   
   **To Reproduce**
   Steps to reproduce the behavior:
   1. Go to '...'
   2. Click on '....'
   3. Scroll down to '....'
   4. See error
   
   **Expected Behavior**
   What you expected to happen.
   
   **Screenshots**
   If applicable, add screenshots.
   
   **Environment:**
   - OS: [e.g. Windows 10, macOS 11.0, Ubuntu 20.04]
   - Python Version: [e.g. 3.9.0]
   - NeuroLite Version: [e.g. 0.3.0]
   - GPU: [e.g. NVIDIA RTX 3080, None]
   
   **Additional Context**
   Any other context about the problem.

Feature Requests
----------------

How to Request Features
~~~~~~~~~~~~~~~~~~~~~~~

1. **Check Existing Requests**: Look for similar feature requests
2. **Describe the Problem**: Explain what problem the feature solves
3. **Propose Solution**: Suggest how the feature might work
4. **Consider Alternatives**: Think about alternative solutions

Feature Request Template
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   **Is your feature request related to a problem?**
   A clear description of what the problem is.
   
   **Describe the solution you'd like**
   A clear description of what you want to happen.
   
   **Describe alternatives you've considered**
   Alternative solutions or features you've considered.
   
   **Additional context**
   Any other context or screenshots about the feature request.

Community Guidelines
--------------------

Code of Conduct
~~~~~~~~~~~~~~~

We are committed to providing a welcoming and inclusive environment:

- **Be Respectful**: Treat everyone with respect and kindness
- **Be Inclusive**: Welcome people of all backgrounds and experience levels
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Patient**: Remember that everyone is learning

Communication Channels
~~~~~~~~~~~~~~~~~~~~~~

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code contributions and reviews

Review Process
--------------

Pull Request Review
~~~~~~~~~~~~~~~~~~~

All pull requests go through a review process:

1. **Automated Checks**: CI/CD pipeline runs tests and checks
2. **Code Review**: Maintainers review the code
3. **Feedback**: Reviewers provide feedback and suggestions
4. **Iteration**: Make changes based on feedback
5. **Approval**: Once approved, the PR is merged

Review Criteria
~~~~~~~~~~~~~~~

We look for:

- **Functionality**: Does the code work as intended?
- **Tests**: Are there adequate tests?
- **Documentation**: Is the code well-documented?
- **Style**: Does it follow our coding standards?
- **Performance**: Does it maintain or improve performance?

Release Process
---------------

Version Numbering
~~~~~~~~~~~~~~~~~

We follow Semantic Versioning (SemVer):

- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

Release Schedule
~~~~~~~~~~~~~~~~

- **Major Releases**: Every 6-12 months
- **Minor Releases**: Every 2-3 months
- **Patch Releases**: As needed for critical bugs

Recognition
-----------

Contributors
~~~~~~~~~~~~

We recognize contributors in several ways:

- **Contributors File**: Listed in CONTRIBUTORS.md
- **Release Notes**: Mentioned in changelog
- **GitHub Recognition**: GitHub contributor statistics
- **Special Thanks**: Recognition for significant contributions

Maintainer Path
~~~~~~~~~~~~~~~

Active contributors may be invited to become maintainers:

- **Consistent Contributions**: Regular, high-quality contributions
- **Community Involvement**: Active in discussions and reviews
- **Technical Expertise**: Deep understanding of the codebase
- **Leadership**: Helping other contributors

Getting Help
------------

If you need help with contributing:

- **Documentation**: Read this contributing guide
- **GitHub Discussions**: Ask questions in discussions
- **Issues**: Create an issue for specific problems
- **Email**: Contact maintainers directly for sensitive issues

Resources
---------

Useful Links
~~~~~~~~~~~~

- `GitHub Repository <https://github.com/dot-css/neurolite>`_
- `Documentation <https://neurolite.readthedocs.io>`_
- `PyPI Package <https://pypi.org/project/neurolite/>`_
- `Issue Tracker <https://github.com/dot-css/neurolite/issues>`_

Development Tools
~~~~~~~~~~~~~~~~~

- **IDE**: VS Code, PyCharm, or your preferred editor
- **Git**: Version control
- **pytest**: Testing framework
- **Black**: Code formatting
- **Flake8**: Code linting
- **MyPy**: Type checking

Thank you for contributing to NeuroLite! 🧠⚡