"""
Setup script for the Document Q&A system.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def install_requirements():
    """Install required packages."""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False


def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        try:
            with open(env_example, 'r') as src, open(env_file, 'w') as dst:
                dst.write(src.read())
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file and add your OpenAI API key")
            return True
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            return False
    else:
        print("❌ env_example.txt not found")
        return False


def test_imports():
    """Test if all required modules can be imported."""
    print("\n🧪 Testing imports...")
    
    required_modules = [
        "streamlit",
        "openai",
        "sentence_transformers",
        "faiss",
        "PyPDF2",
        "pdfplumber",
        "beautifulsoup4",
        "markdown",
        "html2text",
        "numpy",
        "pandas",
        "tiktoken",
        "langchain"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            if module == "beautifulsoup4":
                __import__("bs4")
            elif module == "html2text":
                __import__("html2text")
            else:
                __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful")
    return True


def run_system_test():
    """Run the system test."""
    print("\n🧪 Running system test...")
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ System test passed")
            print(result.stdout)
            return True
        else:
            print("❌ System test failed")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ System test timed out")
        return False
    except Exception as e:
        print(f"❌ Error running system test: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Document Q&A System Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Create .env file
    if not create_env_file():
        return False
    
    # Test imports
    if not test_imports():
        print("\n❌ Some required packages are missing.")
        print("   Try running: pip install -r requirements.txt")
        return False
    
    # Run system test
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key (optional)")
    print("2. Run the web interface: streamlit run app.py")
    print("3. Or run the test: python test_system.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
