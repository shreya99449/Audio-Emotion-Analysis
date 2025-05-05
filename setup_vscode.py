#!/usr/bin/env python
"""
Setup script for VS Code users to prepare the environment for the Voice Analysis Application.
This script will:
1. Check Python version
2. Create necessary directories
3. Check for dependencies
4. Create a test directory structure

Run this script before starting the application in VS Code.
"""

import os
import sys
import subprocess
import platform
import shutil
from datetime import datetime

# Configuration
PYTHON_MIN_VERSION = (3, 8)
REQUIRED_DIRS = ['uploads', 'static/plots', 'static/css', 'static/js', 'templates']


def print_header(message):
    """Print a formatted header message"""
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80)


def print_status(message, success=True):
    """Print a status message with success indicator"""
    status = "[[32m✓[0m]" if success else "[[31m✗[0m]"
    print(f"{status} {message}")
    return success


def check_python_version():
    """Check if Python version meets requirements"""
    current_python = sys.version_info
    print(f"Current Python version: {current_python.major}.{current_python.minor}.{current_python.micro}")
    
    if current_python.major < PYTHON_MIN_VERSION[0] or \
       (current_python.major == PYTHON_MIN_VERSION[0] and current_python.minor < PYTHON_MIN_VERSION[1]):
        return print_status(
            f"Python version must be at least {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]}", 
            False
        )
    return print_status("Python version is adequate")


def create_directories():
    """Create necessary directories if they don't exist"""
    success = True
    for directory in REQUIRED_DIRS:
        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print_status(f"Created directory: {directory}")
            else:
                print_status(f"Directory already exists: {directory}")
        except Exception as e:
            print_status(f"Failed to create directory {directory}: {str(e)}", False)
            success = False
    return success


def check_dependencies():
    """Check if dependencies can be installed"""
    requirements_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements_for_vscode.txt')
    
    if not os.path.exists(requirements_file):
        return print_status(f"Requirements file not found: {requirements_file}", False)
    
    print("Checking if pip is available...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return print_status("pip is not available", False)
    
    print("\nTo install dependencies, run:")
    print(f"  {sys.executable} -m pip install -r {requirements_file}")
    
    return print_status("Dependencies file check passed")


def prepare_test_environment():
    """Create a sample test environment structure"""
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_samples')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    readme_path = os.path.join(test_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("""# Test Audio Samples Directory

Place test audio files (WAV, MP3, MP4) in this directory to use them for testing.

Recommended test files should contain:
- Clear speech samples
- Different emotional states
- Both male and female voices
- Various audio qualities

You can download free audio samples from websites like:
- FreeSound.org
- Archive.org
- Kaggle datasets
""")
    
    return print_status("Test environment prepared")


def create_vs_code_settings():
    """Create VS Code settings directory and files if they don't exist"""
    vs_code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.vscode')
    if not os.path.exists(vs_code_dir):
        os.makedirs(vs_code_dir)
    
    # Create launch.json
    launch_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Flask",
                "type": "python",
                "request": "launch",
                "module": "flask",
                "env": {
                    "FLASK_APP": "run.py",
                    "FLASK_ENV": "development",
                    "FLASK_DEBUG": "1"
                },
                "args": [
                    "run",
                    "--no-debugger",
                    "--host=0.0.0.0",
                    "--port=5000"
                ],
                "jinja": True
            },
            {
                "name": "Python: Run App",
                "type": "python",
                "request": "launch",
                "program": "${workspaceFolder}/run.py",
                "console": "integratedTerminal"
            }
        ]
    }
    
    # Write to disk as JSON string
    launch_path = os.path.join(vs_code_dir, 'launch.json')
    with open(launch_path, 'w') as f:
        import json
        json.dump(launch_config, f, indent=4)
    
    # Create settings.json with recommended settings
    settings_config = {
        "python.linting.enabled": True,
        "python.linting.pylintEnabled": True,
        "python.formatting.provider": "black",
        "editor.formatOnSave": True,
        "python.analysis.extraPaths": ["${workspaceFolder}"],
        "[python]": {
            "editor.rulers": [88],
            "editor.tabSize": 4
        },
        "files.exclude": {
            "**/__pycache__": True,
            "**/*.pyc": True,
            "**/.DS_Store": True,
            "**/.pytest_cache": True,
            "**/.coverage": True
        }
    }
    
    settings_path = os.path.join(vs_code_dir, 'settings.json') 
    with open(settings_path, 'w') as f:
        import json
        json.dump(settings_config, f, indent=4)
    
    return print_status("VS Code configuration files created")


def main():
    """Main function to run all setup tasks"""
    print_header("Voice Analysis Application Setup")
    print(f"Running setup on {platform.system()} {platform.release()}")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.path.dirname(os.path.abspath(__file__))}\n")
    
    tasks = [
        ("Checking Python version", check_python_version),
        ("Creating required directories", create_directories),
        ("Checking dependencies", check_dependencies),
        ("Preparing test environment", prepare_test_environment),
        ("Creating VS Code settings", create_vs_code_settings)
    ]
    
    all_successful = True
    for task_name, task_func in tasks:
        print_header(task_name)
        if not task_func():
            all_successful = False
    
    print_header("Setup Complete")
    if all_successful:
        print_status("All setup tasks completed successfully")
        print("\nYou can now run the application with:")
        print("  python run.py")
        print("\nOr use the VS Code debugger with the 'Python: Run App' configuration.")
    else:
        print_status("Some setup tasks failed. Please review the errors above.", False)
    
    return 0 if all_successful else 1


if __name__ == "__main__":
    sys.exit(main())
