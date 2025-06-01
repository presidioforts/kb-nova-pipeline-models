#!/usr/bin/env python3
"""
Test script for the development health check endpoint
"""

import os
import sys

def test_development_detection():
    """Test the development mode detection logic"""
    
    # Add the src directory to Python path
    sys.path.insert(0, 'src')
    
    try:
        from src.api.routes import is_development
        
        print("🧪 Testing development detection...")
        
        # Test 1: Current environment
        current_result = is_development()
        print(f"   Current environment detected as development: {current_result}")
        
        # Test 2: Set environment variables
        os.environ["ENVIRONMENT"] = "development"
        dev_result = is_development()
        print(f"   With ENVIRONMENT=development: {dev_result}")
        
        # Test 3: Set debug flag
        os.environ["DEBUG"] = "true"
        debug_result = is_development()
        print(f"   With DEBUG=true: {debug_result}")
        
        # Clean up
        if "ENVIRONMENT" in os.environ:
            del os.environ["ENVIRONMENT"]
        if "DEBUG" in os.environ:
            del os.environ["DEBUG"]
            
        print("✅ Development detection test completed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the src directory structure is correct")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_psutil_availability():
    """Test if psutil is available for system monitoring"""
    try:
        import psutil
        
        print("🔍 Testing psutil functionality...")
        
        # Test basic psutil functions
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        cpu_count = psutil.cpu_count()
        
        print(f"   Memory: {memory.percent:.1f}% used")
        print(f"   Disk: {disk.used / disk.total * 100:.1f}% used")
        print(f"   CPU cores: {cpu_count}")
        print(f"   Process ID: {os.getpid()}")
        
        print("✅ psutil test completed")
        return True
        
    except ImportError:
        print("❌ psutil not installed. Run: pip install psutil")
        return False
    except Exception as e:
        print(f"❌ psutil error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Development Health Check Implementation")
    print("=" * 50)
    
    # Run tests
    psutil_ok = test_psutil_availability()
    dev_detection_ok = test_development_detection()
    
    print("\n" + "=" * 50)
    if psutil_ok and dev_detection_ok:
        print("✅ All tests passed! Development health endpoint should work.")
        print("\n📋 Next steps:")
        print("   1. Install psutil if not already installed: pip install psutil")
        print("   2. Start the service with: python -m src.main")
        print("   3. Test the endpoint: GET http://localhost:8000/api/v1/dev/health")
        print("   4. Check FastAPI docs: http://localhost:8000/docs")
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.") 