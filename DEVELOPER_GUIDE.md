# Developer Guide: How to Navigate & Test the KB Nova Pipeline

## 🎯 **Quick Start for Developers**

This guide helps you understand and test the codebase step by step, from simple examples to full system testing.

## 📁 **Project Structure Overview**

```
kb-nova-pipeline-models/
├── 🚀 START HERE
│   ├── README.md                    # Project overview
│   ├── DEVELOPER_GUIDE.md          # This file - your roadmap
│   └── requirements.txt             # Dependencies
├── 🧪 TESTING & EXAMPLES
│   ├── examples/
│   │   ├── simple_sentence_transformer_demo.py  # ⭐ START HERE
│   │   ├── sentence_transformer_demo.py         # Full integration
│   │   └── chroma_demo.py                       # Vector database
│   └── tests/                       # Unit tests
├── 📚 DOCUMENTATION
│   ├── docs/                        # Comprehensive guides
│   └── notebooks/                   # Analysis notebooks
├── ⚙️ CORE CODE
│   └── src/
│       ├── models/                  # AI/ML models
│       ├── data/                    # Data services
│       ├── api/                     # REST API
│       └── utils/                   # Utilities
└── 🔧 CONFIGURATION
    ├── configs/                     # Settings
    ├── pyproject.toml              # Project config
    └── Dockerfile                  # Deployment
```

## 🚀 **Step-by-Step Testing Guide**

### **Step 1: Run the Simple Demo (No Dependencies)**

Start here to understand the concepts without installing anything:

```bash
# This works immediately - no installations needed
python examples/simple_sentence_transformer_demo.py
```

**What this shows:**
- How your SentenceTransformer code integrates
- Mock implementations of the architecture
- Core patterns and workflows

### **Step 2: Explore the Code Structure**

Let's look at the key files in order of importance:

```bash
# 1. Your SentenceTransformer integration
code src/models/kb_model.py

# 2. API endpoints (how it's exposed)
code src/api/main.py

# 3. Vector database integration
code src/data/chroma_service.py

# 4. Knowledge base management
code src/data/knowledge_base.py
```

### **Step 3: Test Individual Components**

Test each component separately before running the full system:

```bash
# Test configuration loading
python -c "
import sys
sys.path.append('src')
from utils.config import load_config
config = load_config()
print('✅ Config loaded successfully')
print(f'Project: {config.project.name}')
"

# Test data schemas
python -c "
import sys
sys.path.append('src')
from models.schemas import KnowledgeBaseItem
item = KnowledgeBaseItem(
    id='test_001',
    description='Test problem',
    resolution='Test solution',
    category='test'
)
print('✅ Schemas working')
print(f'Item: {item.description}')
"
```

### **Step 4: Install Dependencies (When Ready)**

```bash
# Install core dependencies
pip install fastapi uvicorn pydantic

# Test basic API
python -c "
import sys
sys.path.append('src')
from api.main import app
print('✅ FastAPI app created successfully')
"
```

### **Step 5: Test with Real SentenceTransformers**

```bash
# Install SentenceTransformers
pip install sentence-transformers

# Test your exact model
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(['Hello world'])
print('✅ SentenceTransformer working')
print(f'Embedding shape: {embeddings.shape}')
"
```

## 🧪 **Quick Test Scripts**

### **Comprehensive Component Testing**

```bash
# Test all components at once
python test_components.py
```

This script tests:
- ✅ Basic Python imports
- ✅ Project structure
- ✅ Configuration loading
- ✅ Data schemas
- ✅ Knowledge base service
- ✅ FastAPI application
- ✅ Mock SentenceTransformer
- ⚠️ Real SentenceTransformer (requires installation)
- ⚠️ KB Model Manager (requires dependencies)

### **Code Exploration Helper**

```bash
# Understand the codebase structure
python explore_code.py
```

This shows:
- 📁 Directory structure with explanations
- 🔍 Core files and their purposes
- 🎓 Recommended learning path
- 🧠 Key concepts explained
- ⚡ Quick commands to get started

## 🎯 **Navigation Tips**

### **For Visual Studio Code Users**

```bash
# Open key files in tabs
code src/models/kb_model.py src/api/main.py src/data/chroma_service.py

# View project structure
code . 
```

### **For Command Line Users**

```bash
# Quick file viewing
cat src/models/kb_model.py | head -50    # First 50 lines
grep -n "class\|def" src/models/kb_model.py  # Find classes and functions

# Directory exploration
find src -name "*.py" | head -10         # List Python files
tree src/                                # Show directory tree (if installed)
```

## 🔧 **Development Workflow**

### **1. Understanding Phase**
```bash
python explore_code.py                   # Understand structure
python examples/simple_sentence_transformer_demo.py  # See concepts
```

### **2. Testing Phase**
```bash
python test_components.py               # Test components
pip install fastapi sentence-transformers  # Install deps
python test_components.py               # Test again
```

### **3. Development Phase**
```bash
# Start the API server
python src/api/main.py

# In another terminal, test the API
curl http://localhost:8000/docs         # View API documentation
```

### **4. Integration Phase**
```bash
python examples/sentence_transformer_demo.py  # Full integration demo
python examples/chroma_demo.py               # Vector database demo
```

## 🚨 **Common Issues & Solutions**

### **Import Errors**
```bash
# If you get import errors, add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/Mac
$env:PYTHONPATH += ";$(pwd)\src"               # Windows PowerShell
```

### **Missing Dependencies**
```bash
# Install minimal dependencies first
pip install pydantic fastapi

# Then add AI dependencies
pip install sentence-transformers

# Finally add vector database
pip install chromadb
```

### **Configuration Issues**
```bash
# Check if config file exists
ls configs/config.yaml

# Test config loading
python -c "
import sys; sys.path.append('src')
from utils.config import load_config
print(load_config())
"
```

## 📚 **Learning Resources**

### **Start Here (No Dependencies)**
1. `examples/simple_sentence_transformer_demo.py` - Mock implementation
2. `explore_code.py` - Code structure guide
3. `test_components.py` - Component testing

### **Core Implementation**
1. `src/models/kb_model.py` - Your SentenceTransformer integration
2. `src/api/main.py` - FastAPI web service
3. `src/data/chroma_service.py` - Vector database

### **Advanced Features**
1. `examples/chroma_demo.py` - Vector database demo
2. `notebooks/` - Jupyter analysis notebooks
3. `docs/` - Comprehensive documentation

## 🎉 **Success Indicators**

You'll know you're on the right track when:

✅ `python test_components.py` shows mostly green checkmarks  
✅ `python examples/simple_sentence_transformer_demo.py` runs without errors  
✅ You can read and understand `src/models/kb_model.py`  
✅ `python src/api/main.py` starts a web server  
✅ You can navigate the code structure confidently  

## 🚀 **Next Steps**

Once you're comfortable with the codebase:

1. **Customize the knowledge base** - Add your own troubleshooting data
2. **Experiment with models** - Try different SentenceTransformer models
3. **Extend the API** - Add new endpoints for your use cases
4. **Deploy the system** - Use Docker or cloud platforms
5. **Scale the solution** - Add monitoring, logging, and performance optimization

---

**Need Help?** 
- Run `python explore_code.py` for guidance
- Check `docs/README.md` for comprehensive documentation
- Look at `examples/` for working code samples