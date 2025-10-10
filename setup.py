#!/usr/bin/env python3
"""
Setup script for RL Trading Environment
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "logs",
        "tensorboard_logs",
        "plots"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def main():
    print("🚀 Setting up RL Trading Environment...")
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed!")
        return
    
    # Create directories
    create_directories()
    
    print("\n✅ Setup completed successfully!")
    print("\n🎯 Next steps:")
    print("1. Run 'python demo.py' to test the environment")
    print("2. Run 'python train_agent.py' to train the RL agent")
    print("3. Check the generated plots and models")

if __name__ == "__main__":
    main()
