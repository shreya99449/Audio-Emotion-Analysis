# Advanced Voice Analysis Application

This Flask web application allows users to upload audio files and perform comprehensive voice analysis including emotional content, speaker gender detection, and detailed voice characteristics. The application stores the uploaded files in a local 'uploads' directory and uses advanced audio processing techniques to extract a wide range of vocal features.

## Features

- Upload audio files (MP3, WAV, MP4) via web interface
- Emotion detection (happy, sad, angry, neutral, fearful)
- Gender detection based on voice pitch analysis
- Voice characteristic analysis (pitch, speech rate, energy, clarity, tone variation)
- Advanced audio visualizations (waveform, spectrogram, MFCC)
- Voice profile interpretation based on detected characteristics
- Comprehensive visualization of emotion distribution with charts
- Session-based results storage
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

### Running the Application

1. Make sure your virtual environment is activated
2. Run the application:
   ```bash
   python run.py
   ```
3. Open a web browser and navigate to `http://localhost:5000`

## Usage

1. On the home page, click the "Choose File" button and select an audio file (MP3, WAV, or MP4)
2. Click the "Upload" button to process the file
3. View the emotion analysis results and gender detection on the results page
4. The pie chart shows the distribution of detected emotions
5. Click "Back to Home" to analyze another file

## Future Enhancements

- Implement actual machine learning models for more accurate emotion detection
- Add batch processing capability for multiple files
- Include user history tracking
- Implement more sophisticated gender detection algorithms
- Add audio playback on the results page
