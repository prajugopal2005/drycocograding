#!/usr/bin/env python3
"""
System Test Script for Coconut Purity Grading System
Validates all components and functionality
"""

import os
import sys
import tempfile
from PIL import Image
import numpy as np

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import flask
        print(f"✅ Flask {flask.__version__}")
    except ImportError:
        print("❌ Flask not available")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
    except ImportError:
        print("⚠️ TensorFlow not available - simulation mode will be used")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not available")
        return False
    
    try:
        from PIL import Image
        print(f"✅ Pillow {Image.__version__}")
    except ImportError:
        print("❌ Pillow not available")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError:
        print("❌ NumPy not available")
        return False
    
    return True

def test_flask_app():
    """Test if Flask app can be created"""
    print("\n🌐 Testing Flask app...")
    
    try:
        from app import app
        print("✅ Flask app created successfully")
        
        # Test routes
        with app.test_client() as client:
            # Test home route
            response = client.get('/')
            if response.status_code == 200:
                print("✅ Home route working")
            else:
                print(f"❌ Home route failed: {response.status_code}")
                return False
            
            # Test about route
            response = client.get('/about')
            if response.status_code == 200:
                print("✅ About route working")
            else:
                print(f"❌ About route failed: {response.status_code}")
                return False
            
            # Test health check
            response = client.get('/api/health')
            if response.status_code == 200:
                print("✅ Health check working")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def test_prediction_module():
    """Test prediction module"""
    print("\n🧠 Testing prediction module...")
    
    try:
        from predict import predict_purity, predict_with_cloud_api
        print("✅ Prediction module imported successfully")
        
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='brown')
        test_path = 'test_coconut.jpg'
        test_image.save(test_path)
        
        # Test prediction
        label, confidence = predict_purity(test_path)
        print(f"✅ Prediction working: {label} ({confidence}%)")
        
        # Clean up
        os.remove(test_path)
        
        return True
    except Exception as e:
        print(f"❌ Prediction module test failed: {e}")
        return False

def test_training_module():
    """Test training module"""
    print("\n🎓 Testing training module...")
    
    try:
        from train_model import create_model, prepare_data
        print("✅ Training module imported successfully")
        
        # Test model creation
        model = create_model()
        print("✅ Model creation working")
        
        return True
    except Exception as e:
        print(f"❌ Training module test failed: {e}")
        return False

def test_directories():
    """Test if required directories exist"""
    print("\n📁 Testing directories...")
    
    required_dirs = [
        'static',
        'static/uploads',
        'templates',
        'model'
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory exists: {directory}")
        else:
            print(f"❌ Directory missing: {directory}")
            return False
    
    return True

def test_templates():
    """Test if template files exist"""
    print("\n📄 Testing templates...")
    
    required_templates = [
        'templates/index.html',
        'templates/result.html',
        'templates/about.html'
    ]
    
    for template in required_templates:
        if os.path.exists(template):
            print(f"✅ Template exists: {template}")
        else:
            print(f"❌ Template missing: {template}")
            return False
    
    return True

def test_file_upload():
    """Test file upload functionality"""
    print("\n📤 Testing file upload...")
    
    try:
        from app import app
        
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='brown')
        test_path = 'test_upload.jpg'
        test_image.save(test_path)
        
        with app.test_client() as client:
            with open(test_path, 'rb') as f:
                response = client.post('/predict', 
                                    data={'image': (f, 'test_upload.jpg')},
                                    content_type='multipart/form-data')
                
                if response.status_code == 200:
                    print("✅ File upload working")
                else:
                    print(f"❌ File upload failed: {response.status_code}")
                    return False
        
        # Clean up
        os.remove(test_path)
        return True
    except Exception as e:
        print(f"❌ File upload test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🥥 Coconut Purity Grading System - System Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Directory Test", test_directories),
        ("Template Test", test_templates),
        ("Flask App Test", test_flask_app),
        ("Prediction Test", test_prediction_module),
        ("Training Test", test_training_module),
        ("File Upload Test", test_file_upload)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Run: python app.py")
        print("2. Open: http://127.0.0.1:5000")
        print("3. Upload a coconut image to test!")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("💡 Try running: python setup.py")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)