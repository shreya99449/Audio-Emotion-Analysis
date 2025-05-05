# VS Code Setup Guide

## Python Version Compatibility

This application has been tested and works with Python 3.8-3.11. It can also work with Python 3.12 with some additional configuration.

## Setting Up in VS Code

### Step 1: Create a Virtual Environment

```bash
# For Python 3.8-3.11 (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# For Python 3.8-3.11
pip install -r requirements_for_vscode.txt

# For Python 3.12
pip install --pre -r requirements_for_vscode.txt
```

### Step 3: Run Setup Script

```bash
python setup_vscode.py
```

### Step 4: Run the Application

```bash
python run.py
```

## Troubleshooting

### Error: AttributeError: module 'pkgutil' has no attribute 'ImpImporter'

This error typically occurs with Python 3.12 due to compatibility issues with some packages. To fix:

1. Make sure you're using the latest setuptools:
   ```bash
   pip install --upgrade setuptools pip wheel
   ```

2. Try installing dependencies with the `--pre` flag to get pre-release versions:
   ```bash
   pip install --pre -r requirements_for_vscode.txt
   ```

3. If issues persist, consider using Python 3.11 instead, which has better compatibility with all the required packages.

### Error: Failed to build librosa/matplotlib/numpy

On Windows, you might need additional build tools:

1. Install Microsoft C++ Build Tools:
   - Download from [Microsoft's website](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Select "Desktop development with C++" during installation

2. Try installing specific wheel files instead:
   ```bash
   pip install --only-binary=:all: librosa matplotlib numpy
   ```

### Error: No module named 'audio_processor'

Make sure you're running the application from the project root directory:

```bash
# Navigate to the project root
cd path/to/audio-emotion-detector

# Then run the application
python run.py
```

## Database Setup

By default, the application uses SQLite for VS Code environments. To use PostgreSQL instead:

1. Install PostgreSQL on your system
2. Create a database for the application
3. Set the `DATABASE_URL` environment variable:
   ```
   # In your terminal before running the app
   # Windows:
   set DATABASE_URL=postgresql://username:password@localhost:5432/dbname
   # macOS/Linux:
   export DATABASE_URL=postgresql://username:password@localhost:5432/dbname
   ```

## Audio Recording

Audio recording requires a browser with microphone access. Make sure to:

1. Run the application on a device with a microphone
2. Allow microphone access when prompted by the browser
3. Use a secure context (https or localhost) as microphone access is restricted in non-secure contexts
