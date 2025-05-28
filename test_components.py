#!/usr/bin/env python3
"""
Quick Component Testing Script

This script helps you test individual components of the KB Nova Pipeline
without needing to install all dependencies or run the full system.

Run: python test_components.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_basic_imports():
    """Test if basic Python modules work"""
    print("🧪 Testing Basic Imports...")
    
    try:
        import json
        import logging
        from pathlib import Path
        from typing import List, Dict, Any
        print("✅ Basic Python imports working")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_project_structure():
    """Test if project structure is correct"""
    print("\n🧪 Testing Project Structure...")
    
    required_dirs = [
        "src",
        "src/models", 
        "src/data",
        "src/api",
        "src/utils",
        "configs",
        "examples",
        "docs"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ Project structure is correct")
        return True

def test_config_loading():
    """Test configuration loading"""
    print("\n🧪 Testing Configuration Loading...")
    
    try:
        from utils.config import load_config
        config = load_config()
        print(f"✅ Config loaded successfully")
        print(f"   Project: {config.project.name}")
        print(f"   Version: {config.project.version}")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_data_schemas():
    """Test Pydantic data schemas"""
    print("\n🧪 Testing Data Schemas...")
    
    try:
        from models.schemas import KnowledgeBaseItem, TroubleshootRequest
        
        # Test KnowledgeBaseItem
        item = KnowledgeBaseItem(
            id="test_001",
            description="Test problem description",
            resolution="Test solution",
            category="test"
        )
        print(f"✅ KnowledgeBaseItem created: {item.description}")
        
        # Test TroubleshootRequest
        request = TroubleshootRequest(
            description="npm install error",
            category="npm"
        )
        print(f"✅ TroubleshootRequest created: {request.description}")
        return True
    except Exception as e:
        print(f"❌ Schema testing failed: {e}")
        return False

def test_knowledge_base_service():
    """Test Knowledge Base Service"""
    print("\n🧪 Testing Knowledge Base Service...")
    
    try:
        from data.knowledge_base import KnowledgeBaseService
        
        # Initialize service
        kb_service = KnowledgeBaseService(data_dir="data")
        print("✅ KnowledgeBaseService initialized")
        
        # Test getting items
        items = kb_service.get_all_items()
        print(f"✅ Retrieved {len(items)} knowledge base items")
        
        if items:
            print(f"   Example item: {items[0].description}")
        
        return True
    except Exception as e:
        print(f"❌ Knowledge Base Service failed: {e}")
        print("   Note: This is expected if data files don't exist yet")
        return False

def test_fastapi_app():
    """Test FastAPI application creation"""
    print("\n🧪 Testing FastAPI Application...")
    
    try:
        from api.main import app
        print("✅ FastAPI app created successfully")
        
        # Check if app has routes
        routes = [route.path for route in app.routes]
        print(f"✅ App has {len(routes)} routes")
        print(f"   Routes: {routes[:5]}...")  # Show first 5 routes
        return True
    except Exception as e:
        print(f"❌ FastAPI app creation failed: {e}")
        return False

def test_mock_sentence_transformer():
    """Test mock SentenceTransformer (no dependencies needed)"""
    print("\n🧪 Testing Mock SentenceTransformer...")
    
    try:
        # Import from the simple demo
        sys.path.append('examples')
        from simple_sentence_transformer_demo import MockSentenceTransformer
        
        model = MockSentenceTransformer("all-MiniLM-L6-v2")
        
        sentences = ["Hello world", "Test sentence"]
        embeddings = model.encode(sentences)
        
        print(f"✅ Mock model created and working")
        print(f"   Model: {model.model_name}")
        print(f"   Embeddings shape: {embeddings.shape}")
        return True
    except Exception as e:
        print(f"❌ Mock SentenceTransformer failed: {e}")
        return False

def test_real_sentence_transformer():
    """Test real SentenceTransformer (requires installation)"""
    print("\n🧪 Testing Real SentenceTransformer...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("   Loading all-MiniLM-L6-v2 model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        sentences = ["Hello world", "Test sentence"]
        embeddings = model.encode(sentences)
        
        print(f"✅ Real SentenceTransformer working")
        print(f"   Model: all-MiniLM-L6-v2")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        return True
    except ImportError:
        print("⚠️  SentenceTransformers not installed")
        print("   Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ Real SentenceTransformer failed: {e}")
        return False

def test_kb_model_manager():
    """Test KB Model Manager (requires SentenceTransformers)"""
    print("\n🧪 Testing KB Model Manager...")
    
    try:
        from models.kb_model import KBModelManager
        
        # Try to initialize (might fail without dependencies)
        kb_manager = KBModelManager(
            base_model_name="all-MiniLM-L6-v2",
            models_dir="models"
        )
        
        print(f"✅ KB Model Manager created")
        print(f"   Model loaded: {kb_manager.is_model_loaded()}")
        
        if kb_manager.is_model_loaded():
            model_info = kb_manager.get_model_info()
            print(f"   Model: {model_info.model_name}")
        
        return True
    except Exception as e:
        print(f"❌ KB Model Manager failed: {e}")
        print("   Note: This requires sentence-transformers installation")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 KB Nova Pipeline - Component Testing")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Project Structure", test_project_structure),
        ("Configuration", test_config_loading),
        ("Data Schemas", test_data_schemas),
        ("Knowledge Base Service", test_knowledge_base_service),
        ("FastAPI App", test_fastapi_app),
        ("Mock SentenceTransformer", test_mock_sentence_transformer),
        ("Real SentenceTransformer", test_real_sentence_transformer),
        ("KB Model Manager", test_kb_model_manager),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is working perfectly.")
    elif passed >= total * 0.7:
        print("👍 Most tests passed! System is mostly working.")
        print("💡 Install missing dependencies to get remaining tests working.")
    else:
        print("⚠️  Several tests failed. Check the error messages above.")
        print("💡 Start with the simple demo: python examples/simple_sentence_transformer_demo.py")
    
    return passed, total

def show_next_steps():
    """Show what to do next"""
    print("\n🚀 Next Steps:")
    print("1. Run simple demo: python examples/simple_sentence_transformer_demo.py")
    print("2. Install dependencies: pip install fastapi sentence-transformers")
    print("3. Explore code: Start with src/models/kb_model.py")
    print("4. Read docs: Check docs/README.md for comprehensive guides")
    print("5. Run full system: python src/api/main.py (after installing deps)")

if __name__ == "__main__":
    passed, total = run_all_tests()
    show_next_steps() 