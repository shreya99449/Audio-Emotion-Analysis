# Advanced Voice Analysis Application with Machine Learning

**UPDATE**: This application is now fully compatible with VS Code! Use the included setup tools and instructions to run it outside of Replit.

This Flask web application allows users to upload audio files and perform sophisticated voice analysis using machine learning models. The application detects emotional content, speaker gender, and detailed voice characteristics with high accuracy. It stores the uploaded files in a local 'uploads' directory and uses advanced audio processing techniques to extract a wide range of vocal features.

## Features

- Upload audio files (MP3, WAV, MP4) via an intuitive web interface
- Record audio directly in the browser for instant analysis
- Advanced ML-based emotion detection for 9 distinct emotions
  - Happy, sad, angry, neutral, fearful, surprised, disgusted, calm, excited
  - RandomForest classifier with confidence scores
- Age estimation based on voice characteristics
  - Identifies age range (child, teenager, young adult, etc.)
  - Provides confidence scores for the estimation
- Comprehensive voice characteristic analysis
  - Pitch, speech rate, energy, clarity, tone variation, harmonic ratio
  - Formant structure and spectral analysis
- Beautiful audio visualizations (waveform, spectrogram, MFCC)
- Voice profile interpretation based on detected characteristics
- Interactive visualization of emotion distribution with both pie and radar charts
- Personalized activity recommendations based on detected emotions
- History tracking of previous analyses
- Mobile-responsive design with Bootstrap
- Modern colorful UI with gradient effects and visually appealing transitions

## Project Structure

- `app.py`: Main Flask application with routes and request handling
- `audio_processor.py`: Audio processing logic for emotion and gender detection
- `templates/`: HTML templates for the web interface
- `static/`: CSS and JavaScript files
- `uploads/`: Directory for storing uploaded audio files
- `run.py`: Simple script to run the application directly

## Setup in VS Code

### Prerequisites

- Python 3.8 or higher (3.8 - 3.11 recommended)
- VS Code with Python extension
- pip package manager
- FFmpeg for audio processing (optional but recommended)

### Installation

1. Clone or download the repository from Replit
2. Open the project folder in VS Code
3. Create a virtual environment (strongly recommended):
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
   - Create all necessary directories (uploads, static/plots, etc.)
   - Set up VS Code configuration files
   - Verify dependencies
   - Create a test environment
   - Configure debugging settings

### Database Setup

The application uses SQLite by default in VS Code, which doesn't require additional configuration. If you want to use PostgreSQL instead:

1. Install PostgreSQL on your machine
2. Create a database for the application
3. Set the `DATABASE_URL` environment variable:
   - On Windows: `set DATABASE_URL=postgresql://username:password@localhost:5432/dbname`
   - On macOS/Linux: `export DATABASE_URL=postgresql://username:password@localhost:5432/dbname`

### Running the Application

1. Make sure your virtual environment is activated
2. Run the application:
   ```bash
   python run.py
   ```
3. Open a web browser and navigate to `http://localhost:5000`

> **Having issues?** Check the [Troubleshooting Guide](TROUBLESHOOTING.md) for solutions to common problems.

## Usage

### Uploading Audio
1. On the home page, click the "Choose File" button or simply drag and drop an audio file (MP3, WAV, or MP4)
2. Click the "Upload and Analyze" button to process the file
3. Wait for the analysis to complete - this may take a few seconds depending on the file size

### Recording Audio
1. Switch to the "Record Voice" tab on the home page
2. Allow microphone access when prompted by your browser
3. Click the "Start Recording" button and speak into your microphone
4. Click "Stop Recording" when finished
5. Click "Analyze Recording" to process your recording

### Viewing Results
1. The results page shows multiple visualizations of your audio analysis:
   - Top emotion badges with confidence scores
   - Interactive pie and radar charts showing emotion distribution
   - Age estimation with confidence level
   - Detailed voice characteristics (pitch, speech rate, etc.)
   - Audio waveform, spectrogram and MFCC visualizations
   - Personalized activity recommendations based on mood
2. Use the tabs to switch between different visualization types
3. Click "Back to Home" to analyze another file

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
- Emotion detection uses a RandomForest classifier trained on acoustic patterns associated with different emotional states
- Age estimation utilizes voice characteristics like pitch, speech rate, and formants
- Both models are powered by scikit-learn's RandomForest implementation with optimized hyperparameters
- The models are trained on synthetic data with variations to improve generalization
- Feature importance analysis is used to determine the most relevant voice characteristics

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

### UI Design
- Modern, colorful interface with gradient backgrounds and card effects
- Responsive design using Bootstrap 5 framework
- Interactive visualizations with Chart.js for both pie and radar charts
- Custom CSS animations for a more engaging user experience
- Semantic use of colors to indicate emotions and confidence levels
- Touch-friendly interface elements for mobile users
- Real-time audio recording and visualization

## Replit vs. VS Code Environment

This application is designed to run smoothly in both Replit and VS Code environments, with some key differences:

### Replit Environment
- **Workflow Management**: Uses Replit's built-in workflow system to start the server
- **Database**: Configured to use PostgreSQL database via environment variables
- **Package Management**: Dependencies are managed through pyproject.toml
- **Web Access**: Automatically exposed through Replit's domain
- **Starting the App**: Click the Run button in Replit's interface

### VS Code Environment
- **Workflow Management**: Uses the included run.py script to start the server
- **Database**: Defaults to SQLite unless PostgreSQL is configured manually
- **Package Management**: Dependencies are installed through requirements_for_vscode.txt
- **Web Access**: Accessible at localhost:5000 by default
- **Starting the App**: Use `python run.py` command in terminal

### Switching Between Environments
If you've been developing in Replit and want to switch to VS Code:
1. Export your project from Replit
2. Follow the VS Code setup instructions above
3. Run setup_vscode.py to configure the local environment

If you've been developing in VS Code and want to use Replit:
1. Import your project to Replit
2. Make sure the correct packages are listed in pyproject.toml
3. Use the Replit workflow to start the application
