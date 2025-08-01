# Contributing to NeuroLite

We love your input! We want to make contributing to NeuroLite as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

### Development Setup

```bash
git clone https://github.com/dot-css/neurolite.git
cd neurolite
pip install -e ".[dev]"
pre-commit install
```

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Pytest** for testing

Run these before submitting:

```bash
black neurolite/ tests/
flake8 neurolite/ tests/
mypy neurolite/
pytest tests/ -v
```

### Testing

- Write tests for new features
- Ensure all tests pass
- Maintain or improve code coverage
- Use descriptive test names

### Documentation

- Update docstrings for new/modified functions
- Add examples for new features
- Update README.md if needed
- Keep documentation clear and concise

## Any contributions you make will be under the MIT Software License

When you submit code changes, your submissions are understood to be under the same [MIT License](LICENSE) that covers the project.

## Report bugs using GitHub's [issue tracker](https://github.com/dot-css/neurolite/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/dot-css/neurolite/issues/new).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## License

By contributing, you agree that your contributions will be licensed under its MIT License.