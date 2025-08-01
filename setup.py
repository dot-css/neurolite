"""
Setup script for neurolite library.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read version from _version.py
version_file = os.path.join(this_directory, 'neurolite', '_version.py')
version_dict = {}
with open(version_file) as f:
    exec(f.read(), version_dict)

setup(
    name="neurolite",
    version=version_dict['__version__'],
    author="NeuroLite Team",
    author_email="team@neurolite.ai",
    description="AI/ML/DL/NLP productivity library for minimal-code machine learning workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dot-css/neurolite",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "transformers>=4.20.0",
        "optuna>=3.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "flask>=2.0.0",
        "fastapi>=0.70.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "click>=8.0.0",
        "tqdm>=4.60.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "librosa>=0.9.0",
        "onnx>=1.12.0",
        "onnxruntime>=1.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
            "build>=0.8.0",
            "twine>=4.0.0",
        ],
        "tensorflow": [
            "tensorflow>=2.9.0",
        ],
        "xgboost": [
            "xgboost>=1.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neurolite=neurolite.cli.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)