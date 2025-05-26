#!/usr/bin/env python3
"""
Setup script for KB Nova Pipeline Models project.
This script initializes the project environment and creates necessary files.
"""

import os
import sys
import subprocess
from pathlib import Path


def create_directory_structure():
    """Create the complete directory structure for the project."""
    directories = [
        "src/data",
        "src/models", 
        "src/features",
        "src/visualization",
        "src/utils",
        "src/api",
        "src/training",
        "src/inference",
        "src/evaluation",
        "data/raw",
        "data/processed",
        "data/external", 
        "data/interim",
        "models/trained",
        "models/artifacts",
        "notebooks/exploratory",
        "notebooks/experiments",
        "tests/unit",
        "tests/integration",
        "configs",
        "scripts",
        "docs",
        "logs",
        "reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def create_init_files():
    """Create __init__.py files for Python packages."""
    init_files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/features/__init__.py", 
        "src/visualization/__init__.py",
        "src/utils/__init__.py",
        "src/api/__init__.py",
        "src/training/__init__.py",
        "src/inference/__init__.py",
        "src/evaluation/__init__.py",
        "tests/__init__.py",
        "tests/unit/__init__.py",
        "tests/integration/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✓ Created __init__.py: {init_file}")


def create_gitkeep_files():
    """Create .gitkeep files to preserve empty directories."""
    gitkeep_files = [
        "data/raw/.gitkeep",
        "data/processed/.gitkeep", 
        "data/external/.gitkeep",
        "data/interim/.gitkeep",
        "models/trained/.gitkeep",
        "models/artifacts/.gitkeep",
        "logs/.gitkeep",
        "reports/.gitkeep"
    ]
    
    for gitkeep_file in gitkeep_files:
        Path(gitkeep_file).touch()
        print(f"✓ Created .gitkeep: {gitkeep_file}")


def setup_git_hooks():
    """Setup pre-commit hooks if git repository exists."""
    if Path(".git").exists():
        try:
            subprocess.run(["pre-commit", "install"], check=True)
            print("✓ Pre-commit hooks installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠ Pre-commit not installed or failed to setup hooks")
    else:
        print("⚠ Not a git repository, skipping pre-commit setup")


def create_environment_file():
    """Create .env file from template if it doesn't exist."""
    if not Path(".env").exists() and Path("env.example").exists():
        import shutil
        shutil.copy("env.example", ".env")
        print("✓ Created .env file from template")
        print("⚠ Please update .env file with your actual configuration")


def install_dependencies():
    """Install Python dependencies."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("⚠ Failed to install dependencies")
        print("Please run: pip install -r requirements.txt")


def main():
    """Main setup function."""
    print("🚀 Setting up KB Nova Pipeline Models project...")
    print("=" * 50)
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    create_directory_structure()
    
    # Create __init__.py files
    print("\n📄 Creating Python package files...")
    create_init_files()
    
    # Create .gitkeep files
    print("\n🔒 Creating .gitkeep files...")
    create_gitkeep_files()
    
    # Setup environment file
    print("\n⚙️ Setting up environment configuration...")
    create_environment_file()
    
    # Setup git hooks
    print("\n🔧 Setting up git hooks...")
    setup_git_hooks()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    install_dependencies()
    
    print("\n" + "=" * 50)
    print("✅ Project setup completed successfully!")
    print("\nNext steps:")
    print("1. Update .env file with your configuration")
    print("2. Activate your virtual environment")
    print("3. Run 'make test' to verify setup")
    print("4. Start developing your AI/ML pipeline!")


if __name__ == "__main__":
    main() 