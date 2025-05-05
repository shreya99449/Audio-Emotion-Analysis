# Advanced Voice Analysis Application with Machine Learning

This Flask web application allows users to upload audio files and perform sophisticated voice analysis using machine learning models. The application detects emotional content, speaker gender, and detailed voice characteristics with high accuracy. It stores the uploaded files in a local 'uploads' directory and uses advanced audio processing techniques to extract a wide range of vocal features.

## Features

- Upload audio files (MP3, WAV, MP4) via an intuitive web interface
- Advanced ML-based emotion detection for 9 distinct emotions
  - Happy, sad, angry, neutral, fearful, surprised, disgusted, calm, excited
  - RandomForest classifier with confidence scores
- Sophisticated gender detection using multiple voice characteristics
  - Considers pitch, formants, spectral properties, and acoustic features
  - Handles ambiguous pitch ranges (170-210 Hz) with specialized algorithms
  - 7-feature RandomForest classifier with 600+ synthetic training samples
  - Provides confidence scores and enhanced accuracy
- Comprehensive voice characteristic analysis
  - Pitch, speech rate, energy, clarity, tone variation, harmonic ratio
  - Formant structure and spectral analysis
- Beautiful audio visualizations (waveform, spectrogram, MFCC)
- Voice profile interpretation based on detected characteristics
- Interactive visualization of emotion distribution with charts
- History tracking of previous analyses
- Mobile-responsive design with Bootstrap

## Project Structure

- `app.py`: Main Flask application with routes and request handling
- `audio_processor.py`: Audio processing logic for emotion and gender detection
- `templates/`: HTML templates for the web interface
- `static/`: CSS and JavaScript files
- `uploads/`: Directory for storing uploaded audio files
- `run.py`: Simple script to run the application directly

## Setup in VS Code

### Prerequisites

- Python 3.8 or higher
- VS Code with Python extension
- pip package manager

### Installation

1. Clone the repository or download the source code
2. Open the project folder in VS Code
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
5. Install the required packages:
   ```bash
   pip install -r requirements_for_vscode.txt
   ```
6. Run the setup script to prepare your environment:
   ```bash
   python setup_vscode.py
   ```
   This script will:
   - Check your Python version
   - Create all necessary directories
   - Set up VS Code configuration files
   - Verify dependencies
   - Create a test environment

### Running the Application

1. Make sure your virtual environment is activated
2. Run the application:
   ```bash
   python run.py
   ```
3. Open a web browser and navigate to `http://localhost:5000`

> **Having issues?** Check the [Troubleshooting Guide](TROUBLESHOOTING.md) for solutions to common problems.

## Usage

1. On the home page, click the "Choose File" button and select an audio file (MP3, WAV, or MP4)
2. Click the "Upload" button to process the file
3. View the emotion analysis results and gender detection on the results page
4. The pie chart shows the distribution of detected emotions
5. Click "Back to Home" to analyze another file

## Future Enhancements

- Train models on real-world labeled datasets instead of synthetic data
- Add support for batch processing multiple files at once
- Implement user accounts for persistent history tracking
- Add voice recognition for speaker identification
- Extract additional voice quality features (jitter, shimmer, etc.)
- Implement deep learning models (CNNs, RNNs) for even better accuracy
- Add audio playback functionality on the results page
- Export results to PDF or CSV formats

## Technical Notes

### Machine Learning Implementation
- Gender detection uses a RandomForest classifier with 200 trees and optimized hyperparameters
- The gender model is trained on 600+ synthetic samples covering the full vocal range
- Special handling is implemented for the ambiguous pitch range (170-210 Hz) where male/female voices overlap
- 7 voice characteristics are used for gender classification: pitch, formant1, formant2, pitch_variation, spectral_centroid, formant_ratio, and clarity
- Emotion detection uses a separate RandomForest classifier trained on acoustic patterns associated with different emotional states

### Audio Processing
- The application extracts features using librosa's audio processing capabilities
- MFCCs (Mel-Frequency Cepstral Coefficients) are used to estimate formant structures
- Plot images are saved to the static/plots directory rather than being embedded in the session cookie to avoid size limitations
- Voice characteristics are calculated using both time-domain and frequency-domain features
- The application is designed to gracefully handle audio files of various quality and formats

### Design Considerations
- Feature scaling is applied to normalize input features for the ML models
- Post-processing rules improve accuracy in ambiguous cases
- Detailed logging helps with debugging and performance analysis
- Session-based storage allows for history tracking without database integration
