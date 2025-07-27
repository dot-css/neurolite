"""
Setup configuration for NeuroLite package.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="neurolite",
    version="0.1.0",
    author="NeuroLite Team",
    author_email="team@neurolite.ai",
    description="Automated AI/ML library for intelligent data detection and model recommendation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neurolite/neurolite",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Scientists",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.22.0",
        ],
        "all": [
            "opencv-python>=4.7.0",
            "librosa>=0.10.0",
            "transformers>=4.25.0",
            "torch>=2.0.0",
            "tensorflow>=2.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "neurolite=neurolite.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="machine learning, data analysis, automated ml, data detection, ai",
    project_urls={
        "Bug Reports": "https://github.com/dot-css/neurolite/issues",
        "Source": "https://github.com/dot-css/neurolite",
        "Documentation": "https://neurolite.readthedocs.io/",
    },
)