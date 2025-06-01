# Python Version Compatibility Guide

## Overview
This knowledge base project is designed to work with both **Python 3.9** (production) and **Python 3.12** (development) environments.

## Compatibility Requirements

### Supported Python Versions
- **Minimum**: Python 3.9
- **Maximum Tested**: Python 3.12
- **Recommended Development**: Python 3.12
- **Production**: Python 3.9

## Code Compatibility Checklist

### ✅ Features Used (Python 3.9 Compatible)
- Type annotations with `typing` module (`List`, `Dict`, `Tuple`, `Optional`)
- `async`/`await` syntax
- F-strings
- Dataclasses
- Context managers
- `pathlib` module

### ❌ Features NOT Used (Python 3.10+ Only)
- ~~Union types with `|` syntax~~ (Use `Union[str, None]` instead of `str | None`)
- ~~`match`/`case` statements~~ (Use `if`/`elif` instead)
- ~~Parenthesized context managers~~ (Use nested `with` statements)
- ~~PEP 585 built-in generics~~ (Use `typing.List` instead of `list`)

## Installation Instructions

### Python 3.9 (Production)
```bash
# Verify Python version
python --version  # Should show 3.9.x

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import fastapi; print('FastAPI version:', fastapi.__version__)"
```

### Python 3.12 (Development)
```bash
# Verify Python version
python --version  # Should show 3.12.x

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import fastapi; print('FastAPI version:', fastapi.__version__)"
```

## Testing Compatibility

### Run Tests on Both Versions
```bash
# Python 3.9
python -m pytest tests/

# Python 3.12  
python -m pytest tests/
```

### Check Import Compatibility
```bash
# Test core imports
python -c "
from src.api.routes import router
from src.models.schemas import Query
from src.models.langchain_hybrid_kb import LangChainHybridKnowledgeBase
print('✅ All imports successful')
"
```

## Dependency Version Strategy

### Version Ranges
Our `requirements.txt` uses version ranges to ensure compatibility:
- `fastapi>=0.104.1,<0.116.0` - Supports both Python versions
- `torch>=2.1.0,<2.5.0` - Avoids versions that drop Python 3.9 support
- `numpy>=1.24.3,<2.0.0` - Stable across Python versions

### Version Pinning for Production
For production deployments, consider pinning exact versions:
```bash
pip freeze > requirements-locked.txt
```

## Environment Variables

### Python Version Detection
```python
import sys

def check_python_compatibility():
    version = sys.version_info
    if version < (3, 9):
        raise RuntimeError("Python 3.9+ required")
    if version >= (3, 13):
        print("Warning: Python 3.13+ not fully tested")
    print(f"✅ Python {version.major}.{version.minor} is supported")
```

## Common Issues & Solutions

### 1. Import Errors
**Problem**: `ModuleNotFoundError`
**Solution**: Check Python version and reinstall dependencies

### 2. Type Annotation Issues  
**Problem**: Syntax errors with type annotations
**Solution**: Use `typing` module imports consistently

### 3. Dependency Conflicts
**Problem**: Package version incompatibilities
**Solution**: Use virtual environments and version ranges

## Continuous Integration

### GitHub Actions Example
```yaml
strategy:
  matrix:
    python-version: ["3.9", "3.12"]

steps:
- uses: actions/setup-python@v4
  with:
    python-version: ${{ matrix.python-version }}
- run: pip install -r requirements.txt
- run: python -m pytest
```

## Migration Path

### From Python 3.8 → 3.9
1. Update Python installation
2. Reinstall dependencies 
3. Test application thoroughly
4. Update CI/CD pipelines

### From Python 3.12 → 3.9 (Dev to Prod)
1. Test with Python 3.9 locally
2. Verify no 3.10+ features used
3. Deploy to production environment
4. Monitor for runtime issues

## Best Practices

1. **Always test on both versions** before deployment
2. **Use virtual environments** to isolate dependencies
3. **Pin dependency versions** in production
4. **Document Python version requirements** clearly
5. **Set up CI** to test multiple Python versions
6. **Avoid bleeding-edge features** that aren't widely supported

## Quick Verification Script

```python
#!/usr/bin/env python3
"""
Quick compatibility verification script
"""
import sys
import importlib

def verify_compatibility():
    # Check Python version
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 9):
        print("❌ Python 3.9+ required")
        return False
    
    if version >= (3, 13):
        print("⚠️  Python 3.13+ not fully tested")
    
    # Test core imports
    try:
        import fastapi
        import uvicorn
        import pydantic
        import sentence_transformers
        import torch
        import numpy
        import chromadb
        print("✅ All core dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    print("✅ Environment is compatible")
    return True

if __name__ == "__main__":
    verify_compatibility()
```

Save this as `verify_compatibility.py` and run: `python verify_compatibility.py` 