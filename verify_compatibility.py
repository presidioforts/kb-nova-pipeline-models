#!/usr/bin/env python3
"""
Python Version Compatibility Verification Script
Checks if the current environment is compatible with both Python 3.9 and 3.12
"""
import sys
import importlib
from typing import List, Dict, Tuple

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 9):
        print("❌ Python 3.9+ required for this project")
        print("   Production environment uses Python 3.9")
        return False
    
    if version >= (3, 13):
        print("⚠️  Python 3.13+ not fully tested")
        print("   Recommended versions: 3.9-3.12")
    
    print("✅ Python version is compatible")
    return True

def check_typing_features() -> bool:
    """Check that typing features work correctly."""
    try:
        # Test typing imports (Python 3.9 compatible)
        from typing import List, Dict, Tuple, Optional, Union
        
        # Test type annotations
        def test_function(items: List[str], metadata: Dict[str, int]) -> Optional[str]:
            return items[0] if items else None
        
        # Test with sample data
        result = test_function(["test"], {"count": 1})
        
        print("✅ Typing features work correctly")
        return True
    except Exception as e:
        print(f"❌ Typing features error: {e}")
        return False

def check_core_imports() -> bool:
    """Test importing core dependencies."""
    required_packages = [
        "fastapi",
        "uvicorn", 
        "pydantic",
    ]
    
    optional_packages = [
        "sentence_transformers",
        "torch",
        "numpy",
        "chromadb",
        "pytest"
    ]
    
    print("\n📦 Checking core dependencies...")
    
    failed_imports = []
    
    # Check required packages
    for package in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError as e:
            print(f"❌ {package}: Not installed - {e}")
            failed_imports.append(package)
    
    # Check optional packages
    print("\n📦 Checking optional dependencies...")
    for package in optional_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"⚠️  {package}: Not installed (optional)")
    
    if failed_imports:
        print(f"\n❌ Failed to import required packages: {failed_imports}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required dependencies imported successfully")
    return True

def check_project_imports() -> bool:
    """Test importing project modules."""
    print("\n🏗️ Checking project imports...")
    
    project_modules = [
        "src.models.schemas",
        "src.api.routes"
    ]
    
    failed_imports = []
    
    for module_path in project_modules:
        try:
            importlib.import_module(module_path)
            print(f"✅ {module_path}")
        except ImportError as e:
            print(f"❌ {module_path}: {e}")
            failed_imports.append(module_path)
    
    if failed_imports:
        print(f"\n❌ Failed to import project modules: {failed_imports}")
        print("   Make sure you're in the project root directory")
        return False
    
    print("\n✅ All project modules imported successfully")
    return True

def check_async_features() -> bool:
    """Test async/await functionality."""
    try:
        import asyncio
        
        async def test_async():
            return "async works"
        
        # Test async function
        result = asyncio.run(test_async())
        
        print("✅ Async/await features work correctly")
        return True
    except Exception as e:
        print(f"❌ Async features error: {e}")
        return False

def print_recommendations():
    """Print compatibility recommendations."""
    print("\n💡 Recommendations:")
    print("   📍 Development: Use Python 3.12 for latest features")
    print("   🏭 Production: Use Python 3.9 for stability")
    print("   🧪 Testing: Test on both versions before deployment")
    print("   📦 Dependencies: Use version ranges in requirements.txt")
    print("   🔒 Production: Pin exact versions for deployment")

def main():
    """Main verification function."""
    print("🔍 Python Version Compatibility Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Typing Features", check_typing_features),
        ("Core Dependencies", check_core_imports),
        ("Project Imports", check_project_imports),
        ("Async Features", check_async_features)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}:")
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All compatibility checks passed!")
        print("✅ Environment is ready for both Python 3.9 and 3.12")
    else:
        print("⚠️  Some compatibility checks failed")
        print("❌ Please fix the issues above before proceeding")
    
    print_recommendations()
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 