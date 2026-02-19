# Contributing to EcoloGRAPH

Thank you for your interest in contributing to EcoloGRAPH! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)

---

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in all interactions.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

---

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.11 or higher
- Git
- Docker Desktop (for full features)
- A code editor (VS Code, PyCharm recommended)
- Familiarity with:
  - Python
  - Neo4j/Cypher (for graph contributions)
  - Streamlit (for UI contributions)
  - LangChain/LangGraph (for agent contributions)

---

## 💻 Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/EcoloGRAPH.git
cd EcoloGRAPH

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/EcoloGRAPH.git
```

### 2. Create Development Environment

```bash
# Create conda environment
conda create -n ecolograph-dev python=3.11
conda activate ecolograph-dev

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black isort flake8 mypy
```

### 3. Start Services

```bash
# Start Docker services
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password neo4j
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with local LLM settings
```

### 5. Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

---

## 🛠️ Making Contributions

### Types of Contributions

We welcome contributions in several areas:

1. **🐛 Bug Fixes**
   - Fix issues in existing code
   - Improve error handling
   - Fix edge cases

2. **✨ New Features**
   - Add new agent tools
   - Implement new visualization types
   - Add support for new data sources
   - Expand domain registry

3. **📚 Documentation**
   - Improve README and guides
   - Add code comments
   - Write tutorials and examples

4. **🧪 Tests**
   - Add unit tests
   - Add integration tests
   - Improve test coverage

5. **🎨 UI/UX Improvements**
   - Enhance Streamlit pages
   - Improve theme and styling
   - Add new interactive components

### Workflow

1. **Create a branch** for your contribution:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make your changes** following the code style guidelines

3. **Test your changes** thoroughly

4. **Commit your changes** with clear messages:
   ```bash
   git commit -m "feat: add support for marine species enrichment"
   # or
   git commit -m "fix: resolve PDF parsing error for tables"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub

---

## 📝 Code Style

We follow PEP 8 with some modifications. Please adhere to these guidelines:

### Python Style

```python
# Use type hints
def process_paper(doc_id: str, metadata: dict) -> PaperResult:
    ...

# Prefer f-strings for formatting
message = f"Processing paper: {doc_id}"

# Use descriptive variable names
species_mentions = extract_species(text)  # Good
sm = extract_species(text)  # Bad

# Constants in UPPER_CASE
MAX_CHUNK_SIZE = 1000
DEFAULT_MODEL = "qwen2.5:7b"

# Classes in PascalCase
class GraphBuilder:
    ...

# Functions in snake_case
def search_papers(query: str) -> list[Paper]:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def search_papers(query: str, limit: int = 10) -> list[Paper]:
    """
    Search for papers using hybrid BM25 + semantic search.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return
        
    Returns:
        List of Paper objects matching the query
        
    Raises:
        ValueError: If query is empty
        ConnectionError: If database is unavailable
    """
    ...
```

### Import Organization

```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
import streamlit as st
from neo4j import GraphDatabase

# Local
from src.core.config import Config
from src.search.paper_index import PaperIndex
```

### Code Formatting Tools

Before committing, run:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

---

## 🧪 Testing

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/`
- Use descriptive test names:

```python
def test_paper_search_returns_relevant_results():
    """Test that paper search returns papers matching the query."""
    ...

def test_graph_builder_creates_species_nodes():
    """Test that GraphBuilder correctly creates species nodes in Neo4j."""
    ...
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_integration.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Coverage

- Aim for at least 80% coverage for new code
- All public functions should have tests
- Critical paths (ingestion, extraction) should have comprehensive tests

---

## 🔄 Pull Request Process

### Before Submitting

1. **Update documentation** if you've added/changed features
2. **Add tests** for new functionality
3. **Run the full test suite** and ensure it passes
4. **Update CHANGELOG.md** with your changes
5. **Ensure code is formatted** with Black and isort

### PR Title Format

Use conventional commit format:

- `feat: add species validation tool`
- `fix: resolve Neo4j connection timeout`
- `docs: update installation guide`
- `refactor: simplify domain classifier logic`
- `test: add tests for chunk extraction`
- `chore: update dependencies`

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- List of specific changes made
- Each change as a bullet point

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots (if UI changes)
Add screenshots here

## Checklist
- [ ] Code follows project style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. Maintainer will review your PR within 3-5 days
2. Address any requested changes
3. Once approved, PR will be merged
4. Your contribution will be credited in the release notes

---

## 🐛 Reporting Bugs

### Before Reporting

1. **Check existing issues** to avoid duplicates
2. **Update to latest version** and test if bug persists
3. **Collect error messages** and logs

### Bug Report Template

```markdown
**Describe the bug**
A clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen

**Actual behavior**
What actually happened

**Screenshots**
Add screenshots if applicable

**Environment:**
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python version: [e.g., 3.11.5]
- EcoloGRAPH version: [e.g., 1.2.0]
- LLM provider: [e.g., Ollama, OpenAI-compatible]

**Additional context**
Any other relevant information
```

---

## 💡 Requesting Features

### Feature Request Template

```markdown
**Feature description**
A clear description of the feature

**Problem it solves**
What problem does this feature address?

**Proposed solution**
How would you like this implemented?

**Alternatives considered**
Other solutions you've thought about

**Additional context**
Any other relevant information
```

---

## 🌟 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in the project README

---

## 📞 Questions?

If you have questions about contributing:
- Open a discussion on GitHub
- Check existing documentation
- Review closed issues for similar questions

---

**Thank you for contributing to EcoloGRAPH! 🌿**
