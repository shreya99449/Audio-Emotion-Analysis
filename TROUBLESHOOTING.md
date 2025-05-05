# Troubleshooting Guide

This document provides solutions to common issues you might encounter when running the Voice Analysis Application in VS Code or other environments.

## Installation Issues

### Python Dependencies

**Issue**: Error installing dependencies from `requirements_for_vscode.txt`

**Solution**: 
1. Make sure you have Python 3.8 or higher installed
2. If you're having issues with a specific package, try installing it separately:
   ```
   pip install librosa==0.10.1
   ```
3. On Windows, some packages like `librosa` may require additional build tools. Install Visual C++ Build Tools or try:
   ```
   pip install librosa --no-cache-dir
   ```

### Python 3.12 Compatibility Issues

**Issue**: Error like `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'` when installing with Python 3.12

**Solution**:
1. Use Python 3.11 instead (recommended)
2. If you must use Python 3.12, try these fixes:
   ```
   # Update setuptools first
   pip install --upgrade setuptools>=69.0.0
   
   # Then try installing with the --pre flag
   pip install --pre -r requirements_for_vscode.txt
   ```
3. See the detailed VSCODE_SETUP.md file for complete instructions

### Missing FFmpeg (Audio Processing)

**Issue**: Error like `RuntimeError: FFmpeg not found`

**Solution**:
1. Install FFmpeg on your system:
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg`
2. Restart your terminal/command prompt after installation

## Runtime Errors

### Plot Generation Issues

**Issue**: Error creating audio plots or `Matplotlib backend error`

**Solution**:
1. Make sure you have a non-interactive backend set for matplotlib:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # Add this before any matplotlib imports
   ```
2. Check that the `static/plots` directory exists and is writable

### Audio File Processing Errors

**Issue**: Error processing uploaded audio files

**Solution**:
1. Make sure the file is a valid audio file (MP3, WAV, MP4)
2. Check if the file is not corrupted by playing it with an audio player
3. Try with a shorter audio clip (10-20 seconds) - very long files might cause memory issues
4. Some very low-quality recordings might not work well - try with clearer audio

### Flask Application Not Starting

**Issue**: Error when running `python run.py`

**Solution**:
1. Check if port 5000 is already in use by another application
2. Try changing the port in `run.py`:
   ```python
   app.run(host='0.0.0.0', port=5001, debug=True)
   ```
3. Make sure you're in the correct directory when running the command

## Performance Issues

### Slow Audio Processing

**Issue**: Audio processing takes too long

**Solution**:
1. Use shorter audio clips - 10-20 seconds is optimal
2. The first run will be slower as models are loaded into memory
3. Check your CPU usage - the application is CPU-intensive

## Model Accuracy Issues

### Gender Detection Issues

**Issue**: Incorrect gender detection

**Solution**:
1. The model works best with clear voice recordings with minimal background noise
2. Certain voice types in the overlap range (170-210 Hz) may be more challenging to classify
3. Try different segments of the audio that have clearer speech

### Emotion Detection Issues

**Issue**: Emotion detection does not match expected emotion

**Solution**:
1. Emotion detection is based on vocal characteristics, not content
2. The model works best with exaggerated emotional expressions
3. Cultural and personal speaking styles may affect results
4. Try segments with stronger emotional content

## File Storage Issues

**Issue**: Files not being saved or accessed correctly

**Solution**:
1. Check that the `uploads` directory exists and has write permissions
2. Verify that you're not filling up disk space with large audio files
3. If running on a network drive, try using a local path instead

## Browser Issues

**Issue**: Web interface not displaying correctly

**Solution**:
1. Try a different browser (Chrome, Firefox, Edge)
2. Clear browser cache and reload
3. Check for JavaScript errors in the browser console
4. Make sure your browser supports CSS gradients and transitions

**Issue**: Gradient effects not showing or displaying incorrectly

**Solution**:
1. Update your browser to the latest version
2. If using Internet Explorer, switch to a modern browser like Chrome or Firefox
3. Some older mobile browsers may not support advanced CSS effects

**Issue**: Charts not displaying properly on the results page

**Solution**:
1. Make sure JavaScript is enabled in your browser
2. Check the browser console for any Chart.js related errors
3. Try reloading the page after all elements have loaded
4. If specific chart tabs don't work, try clicking other tabs first, then return

## Getting Additional Help

If you continue to experience issues after trying these solutions:

1. Check the application logs in `app.log`
2. Take a screenshot of any error messages
3. Note the exact steps to reproduce the issue
4. Create a detailed report including:
   - Your operating system and version
   - Python version (`python --version`)
   - The exact error message
   - Steps to reproduce the issue

## Common Library-Specific Issues

### Librosa Issues

**Issue**: `NoBackendError: AudioFile failed`

**Solution**:
1. Install additional codecs:
   - **Windows**: Install [K-Lite Codec Pack](https://codecguide.com/download_kl.htm)
   - **Linux**: `sudo apt-get install ubuntu-restricted-extras`
2. Try converting your audio to WAV format before uploading

### Matplotlib Issues

**Issue**: `OSError: [Errno 30] Read-only file system`

**Solution**:
1. Set a custom cache directory:
   ```python
   import os
   os.environ['MPLCONFIGDIR'] = os.path.join(os.path.expanduser('~'), '.matplotlib')
   ```
2. Add this code before importing matplotlib

### SQLAlchemy Issues

**Issue**: Database connection errors

**Solution**:
1. Check that the `instance` directory exists and has write permissions
2. If using PostgreSQL, verify your connection string and credentials
3. Try using SQLite for local development
