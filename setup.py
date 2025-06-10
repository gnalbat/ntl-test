#!/usr/bin/env python3
"""
Setup script for Getis-Ord Gi* Analysis project
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("✗ Error: Python 3.7 or higher is required")
        return False
    
    print("✓ Python version is compatible")
    return True

def install_packages():
    """Install required packages."""
    packages = [
        "earthengine-api>=0.1.300",
        "pyyaml>=6.0", 
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ]
    
    print("\nInstalling required packages...")
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    
    return True

def authenticate_gee():
    """Authenticate Google Earth Engine."""
    print("\nGoogle Earth Engine Authentication")
    print("=" * 40)
    print("You need to authenticate with Google Earth Engine to use this script.")
    print("This will open a web browser for authentication.")
    
    response = input("\nDo you want to authenticate now? (y/n): ").lower().strip()
    
    if response == 'y' or response == 'yes':
        return run_command("earthengine authenticate", "Authenticating Google Earth Engine")
    else:
        print("Skipping authentication. You can run 'earthengine authenticate' later.")
        return True

def create_directories():
    """Create necessary directories."""
    directories = ["./data", "./exports", "./figures"]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Error creating directory {directory}: {e}")
            return False
    
    return True

def test_imports():
    """Test if all required packages can be imported."""
    print("\nTesting package imports...")
    
    packages = {
        'ee': 'Google Earth Engine API',
        'yaml': 'PyYAML',
        'numpy': 'NumPy', 
        'matplotlib.pyplot': 'Matplotlib',
        'seaborn': 'Seaborn'
    }
    
    all_good = True
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {name}: {e}")
            all_good = False
    
    return all_good

def main():
    """Main setup function."""
    print("Getis-Ord Gi* Analysis - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install packages
    if not install_packages():
        print("\n✗ Package installation failed. Please check the errors above.")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n✗ Some packages failed to import. Please check the installation.")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("\n✗ Failed to create some directories.")
        sys.exit(1)
    
    # Authenticate GEE
    if not authenticate_gee():
        print("\n✗ Google Earth Engine authentication failed.")
        print("You can run 'earthengine authenticate' manually later.")
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("=" * 50)
    
    print("\nNext steps:")
    print("1. Review and modify config.yaml for your specific analysis")
    print("2. Run the analysis: python getis_ord_analysis.py")
    print("3. Visualize results: python visualize_results.py")
    
    print("\nFiles created:")
    print("- config.yaml: Configuration file")
    print("- getis_ord_analysis.py: Main analysis script")
    print("- visualize_results.py: Visualization utilities")
    print("- requirements.txt: Package dependencies")
    print("- README.md: Documentation")
    
    print("\nDirectories created:")
    print("- ./data: For local data storage")
    print("- ./exports: For exported results")
    print("- ./figures: For generated figures")

if __name__ == "__main__":
    main()
