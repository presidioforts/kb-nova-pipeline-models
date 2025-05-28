#!/usr/bin/env python3
"""
Code Explorer - Navigate the KB Nova Pipeline Codebase

This script helps you understand what each file does and how to explore the code.

Run: python explore_code.py
"""

import os
from pathlib import Path

def show_file_info(file_path, description, key_functions=None):
    """Show information about a file"""
    path = Path(file_path)
    if path.exists():
        size = path.stat().st_size
        lines = len(path.read_text(encoding='utf-8').splitlines()) if size < 100000 else "Large file"
        status = "✅"
    else:
        size = 0
        lines = 0
        status = "❌"
    
    print(f"{status} {file_path}")
    print(f"   📝 {description}")
    if path.exists():
        print(f"   📊 {lines} lines, {size} bytes")
    if key_functions:
        print(f"   🔧 Key functions: {', '.join(key_functions)}")
    print()

def explore_core_files():
    """Explore the core files in order of importance"""
    print("🔍 Core Files Exploration")
    print("=" * 50)
    
    # Start with the most important files
    files_to_explore = [
        {
            "path": "examples/simple_sentence_transformer_demo.py",
            "description": "🌟 START HERE - Shows how everything works with mock implementations",
            "functions": ["demo_your_original_code", "demo_kb_model_manager_integration"]
        },
        {
            "path": "src/models/kb_model.py", 
            "description": "🧠 Your SentenceTransformer integration - the heart of the system",
            "functions": ["KBModelManager", "encode", "find_similar", "fine_tune"]
        },
        {
            "path": "src/api/main.py",
            "description": "🚀 FastAPI endpoints - how your model is exposed as a web service",
            "functions": ["troubleshoot_issue", "search_knowledge_base", "chat_with_kb"]
        },
        {
            "path": "src/data/chroma_service.py",
            "description": "🗄️ Vector database integration - persistent storage for embeddings",
            "functions": ["ChromaService", "add_document", "search_documents"]
        },
        {
            "path": "src/data/knowledge_base.py",
            "description": "📚 Knowledge base management - handles troubleshooting data",
            "functions": ["KnowledgeBaseService", "get_all_items", "search_by_category"]
        },
        {
            "path": "src/models/schemas.py",
            "description": "📋 Data structures - Pydantic models for API requests/responses",
            "functions": ["KnowledgeBaseItem", "TroubleshootRequest", "ChatRequest"]
        },
        {
            "path": "src/utils/config.py",
            "description": "⚙️ Configuration management - loads settings from config files",
            "functions": ["load_config", "get_config"]
        },
        {
            "path": "configs/config.yaml",
            "description": "🔧 Main configuration file - all system settings",
            "functions": ["model settings", "API config", "database config"]
        }
    ]
    
    for file_info in files_to_explore:
        show_file_info(
            file_info["path"], 
            file_info["description"], 
            file_info.get("functions")
        )

def show_directory_structure():
    """Show the directory structure with explanations"""
    print("📁 Directory Structure Guide")
    print("=" * 50)
    
    structure = {
        "src/": "🏗️ Main source code",
        "src/models/": "🧠 AI/ML models and schemas",
        "src/data/": "📊 Data services and management", 
        "src/api/": "🌐 Web API endpoints",
        "src/utils/": "🔧 Utility functions",
        "examples/": "💡 Demo scripts and examples",
        "docs/": "📚 Documentation and guides",
        "notebooks/": "📓 Jupyter notebooks for analysis",
        "tests/": "🧪 Unit and integration tests",
        "configs/": "⚙️ Configuration files",
        "data/": "💾 Data storage (created at runtime)",
        "models/": "🤖 Trained model storage (created at runtime)"
    }
    
    for path, description in structure.items():
        exists = "✅" if Path(path).exists() else "❌"
        print(f"{exists} {path:<15} {description}")
    print()

def show_learning_path():
    """Show the recommended learning path"""
    print("🎓 Recommended Learning Path")
    print("=" * 50)
    
    steps = [
        {
            "step": 1,
            "title": "Understand the Concept",
            "action": "Run: python examples/simple_sentence_transformer_demo.py",
            "why": "See how your SentenceTransformer code fits into the bigger picture"
        },
        {
            "step": 2, 
            "title": "Explore Your Model Integration",
            "action": "Read: src/models/kb_model.py",
            "why": "Understand how your all-MiniLM-L6-v2 model is wrapped and enhanced"
        },
        {
            "step": 3,
            "title": "See the API Layer", 
            "action": "Read: src/api/main.py",
            "why": "Learn how your model becomes a web service"
        },
        {
            "step": 4,
            "title": "Understand Data Flow",
            "action": "Read: src/data/knowledge_base.py and src/data/chroma_service.py", 
            "why": "See how data flows through the system"
        },
        {
            "step": 5,
            "title": "Test Components",
            "action": "Run: python test_components.py",
            "why": "Verify each component works independently"
        },
        {
            "step": 6,
            "title": "Install Dependencies",
            "action": "pip install fastapi sentence-transformers uvicorn",
            "why": "Get the full system working"
        },
        {
            "step": 7,
            "title": "Run the Full System",
            "action": "python src/api/main.py",
            "why": "See your SentenceTransformer serving real requests"
        }
    ]
    
    for step in steps:
        print(f"Step {step['step']}: {step['title']}")
        print(f"   🎯 Action: {step['action']}")
        print(f"   💡 Why: {step['why']}")
        print()

def show_key_concepts():
    """Explain key concepts in the codebase"""
    print("🧠 Key Concepts to Understand")
    print("=" * 50)
    
    concepts = {
        "SentenceTransformer": {
            "what": "Your all-MiniLM-L6-v2 model that converts text to 384-dimensional vectors",
            "where": "src/models/kb_model.py",
            "why": "Core AI component that enables semantic search and similarity"
        },
        "Vector Database (Chroma)": {
            "what": "Stores and searches through text embeddings efficiently", 
            "where": "src/data/chroma_service.py",
            "why": "Enables fast similarity search across large knowledge bases"
        },
        "FastAPI": {
            "what": "Modern Python web framework for building APIs",
            "where": "src/api/main.py", 
            "why": "Exposes your AI model as REST endpoints for web applications"
        },
        "Knowledge Base": {
            "what": "Collection of troubleshooting problems and solutions",
            "where": "src/data/knowledge_base.py",
            "why": "The data your AI model searches through to help users"
        },
        "Embeddings": {
            "what": "Numerical representations of text (384 numbers per sentence)",
            "where": "Throughout the system",
            "why": "Allows computers to understand and compare text meaning"
        }
    }
    
    for concept, info in concepts.items():
        print(f"🔍 {concept}")
        print(f"   What: {info['what']}")
        print(f"   Where: {info['where']}")
        print(f"   Why: {info['why']}")
        print()

def show_quick_commands():
    """Show quick commands to explore the code"""
    print("⚡ Quick Commands")
    print("=" * 50)
    
    commands = [
        ("Test everything", "python test_components.py"),
        ("Run simple demo", "python examples/simple_sentence_transformer_demo.py"),
        ("View main API", "code src/api/main.py"),
        ("View your model", "code src/models/kb_model.py"), 
        ("View config", "code configs/config.yaml"),
        ("Install deps", "pip install fastapi sentence-transformers uvicorn"),
        ("Start API server", "python src/api/main.py"),
        ("View docs", "code docs/README.md")
    ]
    
    for description, command in commands:
        print(f"📝 {description:<20} → {command}")
    print()

def main():
    """Main exploration function"""
    print("🚀 KB Nova Pipeline - Code Explorer")
    print("Welcome! This script helps you navigate and understand the codebase.")
    print()
    
    show_directory_structure()
    explore_core_files()
    show_learning_path()
    show_key_concepts()
    show_quick_commands()
    
    print("🎯 Ready to Start?")
    print("Run this command to begin: python examples/simple_sentence_transformer_demo.py")
    print()
    print("💡 Need help? Check the docs: docs/README.md")

if __name__ == "__main__":
    main() 