# Development Guide

## Overview

This guide covers setting up a development environment for the Knowledge Base Nova Pipeline Models service, including code organization, testing, and contribution guidelines.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, or virtualenv)
- Code editor (VS Code, PyCharm, etc.)

### Local Environment Setup

#### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd kb-nova-pipeline-models

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest black flake8 mypy pre-commit
```

#### 2. Environment Configuration

Create a `.env` file for development:

```bash
# .env
LOG_LEVEL=DEBUG
HF_HUB_DISABLE_PROGRESS_BARS=1
HF_HUB_OFFLINE=0  # Allow downloads in development
ENVIRONMENT=development
```

#### 3. Run the Service

```bash
# Start development server with auto-reload
python -m src.main

# Or with uvicorn directly
uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
```

#### 4. Verify Setup

```bash
# Test API endpoints
curl http://localhost:8080/docs

# Test troubleshoot endpoint
curl -X POST http://localhost:8080/troubleshoot \
  -H "Content-Type: application/json" \
  -d '{"text": "test query"}'
```

## Code Organization

### Project Structure

```
src/
├── __init__.py                     # Main package
├── main.py                         # FastAPI application entry point
├── api/                            # API layer
│   ├── __init__.py
│   └── routes.py                   # Endpoint handlers
├── models/                         # Data models and ML models
│   ├── __init__.py
│   ├── schemas.py                  # Pydantic models
│   └── sentence_transformer.py    # Model management
├── data/                           # Data access layer
│   ├── __init__.py
│   └── knowledge_base.py          # Knowledge base data
└── utils/                          # Utility functions
    ├── __init__.py
    └── file_utils.py              # File operations
```

### Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Avoid tight coupling between modules
3. **Error Handling**: Comprehensive exception handling
4. **Thread Safety**: Proper locking for concurrent operations
5. **Logging**: Structured logging throughout the application

### Code Style Guidelines

#### Python Style

Follow PEP 8 with these specific guidelines:

```python
# Import organization
import os
import json
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.models.schemas import Query
from src.utils.file_utils import load_pairs_from_disk

# Function definitions
def process_query(query: str) -> dict:
    """
    Process a troubleshooting query.
    
    Args:
        query: The user's query text
        
    Returns:
        Dictionary containing response and similarity score
        
    Raises:
        ValueError: If query is empty or invalid
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    # Implementation here
    return {"response": "solution", "score": 0.85}

# Class definitions
class ModelManager:
    """Manages sentence transformer models."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
    
    def load_model(self) -> None:
        """Load the sentence transformer model."""
        # Implementation here
        pass
```

#### Documentation Standards

```python
def fine_tune_model(
    training_pairs: List[TrainingPair],
    epochs: int = 1,
    learning_rate: float = 1e-5
) -> str:
    """
    Fine-tune the sentence transformer model.
    
    This function takes training pairs and fine-tunes the current model
    using cosine similarity loss. The trained model is saved with a
    timestamp and automatically loaded for future inference.
    
    Args:
        training_pairs: List of input-target pairs for training
        epochs: Number of training epochs (default: 1)
        learning_rate: Learning rate for optimization (default: 1e-5)
        
    Returns:
        Path to the saved fine-tuned model
        
    Raises:
        RuntimeError: If training fails due to insufficient memory or data
        ValueError: If training_pairs is empty or invalid
        
    Example:
        >>> pairs = [TrainingPair(input="error", target="solution")]
        >>> model_path = fine_tune_model(pairs, epochs=2)
        >>> print(f"Model saved to: {model_path}")
    """
    # Implementation here
    pass
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/new-endpoint

# Make changes
# ... edit files ...

# Test changes
python -m pytest tests/

# Format code
black src/
flake8 src/

# Commit changes
git add .
git commit -m "Add new endpoint for model statistics"

# Push and create PR
git push origin feature/new-endpoint
```

### 2. Testing

#### Unit Tests

```python
# tests/unit/test_models.py
import pytest
from src.models.schemas import Query, TrainingPair

def test_query_validation():
    """Test query model validation."""
    # Valid query
    query = Query(text="test query")
    assert query.text == "test query"
    
    # Invalid query (empty)
    with pytest.raises(ValueError):
        Query(text="")

def test_training_pair_creation():
    """Test training pair model creation."""
    pair = TrainingPair(input="problem", target="solution")
    assert pair.input == "problem"
    assert pair.target == "solution"
```

#### Integration Tests

```python
# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_troubleshoot_endpoint():
    """Test the troubleshoot endpoint."""
    response = client.post(
        "/troubleshoot",
        json={"text": "npm install error"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "response" in data
    assert "similarity_score" in data

def test_train_endpoint():
    """Test the training endpoint."""
    training_data = {
        "data": [
            {"input": "test problem", "target": "test solution"}
        ]
    }
    response = client.post("/train", json=training_data)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "note" in data
```

#### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_models.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_query"
```

### 3. Code Quality Tools

#### Black (Code Formatting)

```bash
# Format all Python files
black src/ tests/

# Check formatting without changes
black --check src/

# Format specific file
black src/models/schemas.py
```

#### Flake8 (Linting)

```bash
# Lint all files
flake8 src/

# Lint with specific config
flake8 --max-line-length=88 src/

# Ignore specific errors
flake8 --ignore=E203,W503 src/
```

#### MyPy (Type Checking)

```bash
# Type check all files
mypy src/

# Type check specific module
mypy src/models/

# Generate type coverage report
mypy --html-report mypy-report src/
```

### 4. Pre-commit Hooks

Set up pre-commit hooks for automatic code quality checks:

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.4.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
EOF

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Adding New Features

### 1. API Endpoints

To add a new endpoint:

```python
# 1. Add Pydantic model in src/models/schemas.py
class NewRequest(BaseModel):
    parameter: str
    optional_param: Optional[int] = None

class NewResponse(BaseModel):
    result: str
    metadata: dict

# 2. Add business logic in src/api/routes.py
def new_endpoint_handler(request: NewRequest) -> NewResponse:
    """Handle new endpoint logic."""
    # Implementation here
    return NewResponse(result="success", metadata={})

# 3. Add endpoint in src/main.py
@app.post("/new-endpoint", response_model=NewResponse)
def new_endpoint(request: NewRequest):
    return new_endpoint_handler(request)

# 4. Add tests in tests/
def test_new_endpoint():
    response = client.post("/new-endpoint", json={"parameter": "test"})
    assert response.status_code == 200
```

### 2. Model Enhancements

To enhance the sentence transformer functionality:

```python
# 1. Add new methods in src/models/sentence_transformer.py
def get_model_info() -> dict:
    """Get information about the current model."""
    global model
    with model_lock:
        return {
            "model_name": str(model),
            "device": str(model.device),
            "max_seq_length": model.max_seq_length
        }

# 2. Add corresponding API endpoint
@app.get("/model/info")
def model_info():
    return get_model_info()
```

### 3. Data Management

To add new data sources or storage:

```python
# 1. Create new module in src/data/
# src/data/database.py
class DatabaseManager:
    """Manage database connections and operations."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def save_training_pair(self, pair: TrainingPair) -> None:
        """Save training pair to database."""
        # Implementation here
        pass

# 2. Update knowledge_base.py to use new storage
from src.data.database import DatabaseManager

db_manager = DatabaseManager("sqlite:///kb.db")
```

## Debugging

### 1. Logging

Use structured logging for debugging:

```python
import logging

logger = logging.getLogger(__name__)

def troubleshoot_endpoint(q: Query):
    logger.info(f"Processing query: {q.text}")
    
    try:
        # Processing logic
        result = process_query(q.text)
        logger.info(f"Query processed successfully, score: {result['score']}")
        return result
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        raise
```

### 2. Interactive Debugging

```python
# Add breakpoints for debugging
import pdb

def problematic_function():
    # ... some code ...
    pdb.set_trace()  # Debugger will stop here
    # ... more code ...
```

### 3. Performance Profiling

```python
import cProfile
import pstats

def profile_function():
    """Profile a specific function."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    result = expensive_operation()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
    
    return result
```

## Contributing Guidelines

### 1. Code Review Process

1. **Create Feature Branch**: Always work on feature branches
2. **Write Tests**: Include unit and integration tests
3. **Update Documentation**: Update relevant documentation
4. **Code Quality**: Ensure code passes all quality checks
5. **Create Pull Request**: Provide clear description and context

### 2. Commit Message Format

```
type(scope): brief description

Detailed explanation of the change, including:
- What was changed and why
- Any breaking changes
- References to issues or tickets

Examples:
feat(api): add model statistics endpoint
fix(training): resolve memory leak in fine-tuning
docs(readme): update installation instructions
refactor(models): simplify schema validation
```

### 3. Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings or errors
```

### 4. Issue Reporting

When reporting issues, include:

```markdown
## Bug Report

**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Package versions: [output of pip freeze]

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Logs:**
```
[Include relevant log output]
```

**Additional Context:**
Any other relevant information
```

## Performance Optimization

### 1. Model Optimization

```python
# Use model quantization for faster inference
from sentence_transformers import SentenceTransformer
import torch

def load_optimized_model(model_path: str):
    """Load model with optimizations."""
    model = SentenceTransformer(model_path)
    
    # Enable half precision if GPU available
    if torch.cuda.is_available():
        model = model.half()
    
    # Set to evaluation mode
    model.eval()
    
    return model
```

### 2. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_encode(text: str) -> list:
    """Cache encoded embeddings for frequently used texts."""
    with model_lock:
        return model.encode(text).tolist()
```

### 3. Batch Processing

```python
def batch_troubleshoot(queries: List[str]) -> List[dict]:
    """Process multiple queries in batch for efficiency."""
    with model_lock:
        query_embeddings = model.encode(queries, convert_to_tensor=True)
        kb_embeddings = model.encode(corpus_inputs, convert_to_tensor=True)
    
    # Process all similarities at once
    scores = util.cos_sim(query_embeddings, kb_embeddings)
    
    results = []
    for i, query in enumerate(queries):
        best_idx = int(scores[i].argmax())
        results.append({
            "query": query,
            "response": corpus_answers[best_idx],
            "similarity_score": float(scores[i][best_idx])
        })
    
    return results
```

## Troubleshooting Development Issues

### Common Problems

1. **Import Errors**
   ```bash
   # Ensure PYTHONPATH includes src directory
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   
   # Or run with module flag
   python -m src.main
   ```

2. **Model Loading Issues**
   ```python
   # Check model directory exists
   import os
   print(os.path.exists("breakfix-kb-model/all-mpnet-base-v2"))
   
   # Verify PyTorch installation
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   ```

3. **Memory Issues**
   ```python
   # Monitor memory usage
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   ```

### Development Tools

1. **VS Code Configuration**
   ```json
   // .vscode/settings.json
   {
       "python.defaultInterpreterPath": "./venv/bin/python",
       "python.linting.enabled": true,
       "python.linting.flake8Enabled": true,
       "python.formatting.provider": "black",
       "python.testing.pytestEnabled": true,
       "python.testing.pytestArgs": ["tests/"]
   }
   ```

2. **PyCharm Configuration**
   - Set interpreter to virtual environment
   - Enable code inspections
   - Configure test runner to pytest
   - Set up run configurations for FastAPI

This development guide provides a comprehensive foundation for contributing to the Knowledge Base Nova Pipeline Models project. Follow these guidelines to maintain code quality and ensure smooth collaboration. 