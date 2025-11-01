#!/usr/bin/env python3
"""
Setup script for Coconut Purity Grading System
Automates the installation and setup process
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("🥥 Coconut Purity Grading System - Setup")
    print("=" * 60)
    print("Automated setup for the ML-powered coconut purity classifier")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    directories = [
        "static/uploads",
        "model",
        "templates"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_tensorflow():
    """Check if TensorFlow is working"""
    print("\n🧠 Testing TensorFlow...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} is working!")
        return True
    except ImportError:
        print("❌ TensorFlow not available!")
        return False

def create_sample_files():
    """Create sample configuration files"""
    print("\n📝 Creating sample files...")
    
    # Create .env file
    env_content = """# Coconut Purity Grading System - Environment Variables
# Uncomment and set these if needed:

# Google Cloud Vision API (optional)
# GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# Flask Configuration (optional)
# FLASK_ENV=development
# FLASK_DEBUG=True
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    print("✅ Created .env file")
    
    # Create .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Flask
instance/
.webassets-cache

# Model files
model/*.h5
model/*.pkl

# Uploads
static/uploads/*
!static/uploads/.gitkeep

# Environment
.env
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("✅ Created .gitignore file")

def test_application():
    """Test if the application can start"""
    print("\n🧪 Testing application...")
    try:
        # Test imports
        from app import app
        print("✅ Flask app imports successfully!")
        
        # Test prediction module
        from predict import predict_purity
        print("✅ Prediction module imports successfully!")
        
        return True
    except Exception as e:
        print(f"❌ Application test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. Run the application:")
    print("   python app.py")
    print("\n2. Open your browser:")
    print("   http://127.0.0.1:5000")
    print("\n3. To train your own model:")
    print("   python train_model.py")
    print("   (You'll need to organize your dataset first)")
    print("\n📚 Documentation:")
    print("- Quick Start: QUICK_START.md")
    print("- Full Docs: PROJECT_DOCUMENTATION.md")
    print("- Project Summary: PROJECT_SUMMARY.md")
    print("\n🔧 Configuration:")
    print("- Edit .env file for environment variables")
    print("- Place your model in model/ directory")
    print("- Organize dataset for training")
    print("\n" + "=" * 60)

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check TensorFlow
    if not check_tensorflow():
        print("⚠️ TensorFlow not available - using simulation mode")
    
    # Create sample files
    create_sample_files()
    
    # Test application
    if not test_application():
        print("❌ Setup failed at application test")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
