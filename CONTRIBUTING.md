# Contributing to Amazon Deep Insights

First off, thank you for considering contributing to this project! Your help is greatly appreciated.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please open an issue on GitHub and provide the following information:

*   A clear and descriptive title.
*   A detailed description of the bug, including steps to reproduce it.
*   The expected behavior and what happened instead.
*   Your operating system, Python version, and any other relevant information.

### Suggesting Enhancements

If you have an idea for an enhancement, please open an issue on GitHub and provide the following information:

*   A clear and descriptive title.
*   A detailed description of the enhancement you're proposing.
*   Any relevant code snippets or mockups.

### Pull Requests

If you'd like to contribute code, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (e.g., `feature/my-new-feature` or `bugfix/my-bug-fix`).
3.  Make your changes and commit them with a clear and descriptive commit message.
4.  Push your changes to your fork.
5.  Open a pull request to the `main` branch of the original repository.

## Styleguides

### Git Commit Messages

*   Use the present tense ("Add feature" not "Added feature").
*   Use the imperative mood ("Move file to..." not "Moves file to...").
*   Limit the first line to 72 characters or less.
*   Reference issues and pull requests liberally after the first line.

### Python Styleguide

*   All Python code should adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/).
*   We use [Black](https://github.com/psf/black) to format our code.

### Development Setup

1. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```
2. Run tests with coverage:
   ```bash
   pytest --cov=src --cov=run_pipeline.py
   ```
3. Use `tqdm` for progress bars but guard imports with a fallback so pipelines run without optional deps.
