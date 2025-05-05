import os
import logging
import base64
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define allowed extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4'}

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Import audio processor
from audio_processor import process_audio_file

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was submitted
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    file = request.files['file']
    
    # If user doesn't select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)
    
    # Check if file is allowed
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename_with_timestamp = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename_with_timestamp)
        file.save(filepath)
        
        try:
            # Process audio file and get emotions, age, voice features, plots, and recommendations
            emotions, age_estimate, voice_features, plots, recommendations = process_audio_file(filepath)
            
            # Store the results in session for the results page
            # Save plots to static directory instead of session to avoid cookie size limits
            plots_urls = {}
            static_dir = os.path.join('static', 'plots')
            os.makedirs(static_dir, exist_ok=True)
            
            # Generate unique identifiers for the plot files
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            plot_id = f"{timestamp}_{filename.replace('.', '_')}"
            
            # Save each plot as a file and store the URL
            for plot_name, plot_data in plots.items():
                if plot_data:  # Only process if we have plot data
                    plot_filename = f"{plot_name}_{plot_id}.png"
                    plot_path = os.path.join(static_dir, plot_filename)
                    
                    # Decode base64 and save as file
                    try:
                        with open(plot_path, 'wb') as f:
                            f.write(base64.b64decode(plot_data))
                        plots_urls[plot_name] = f"/static/plots/{plot_filename}"
                    except Exception as e:
                        logging.error(f"Error saving plot {plot_name}: {str(e)}")
            
            # Create a record for history tracking
            analysis_record = {
                'filename': filename,
                'emotions': emotions,
                'age_estimate': age_estimate,
                'age_confidence': 0.85,  # Fixed confidence level for age estimation
                'voice_features': voice_features,
                'plots_urls': plots_urls,
                'recommendations': recommendations,
                'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'timestamp': datetime.now().timestamp()
            }
            
            # Initialize history in session if not present
            if 'history' not in session:
                session['history'] = []
            
            # Add current analysis to history (maximum 10 entries)
            history = session['history']
            history.insert(0, analysis_record)  # Add at the beginning (most recent first)
            session['history'] = history[:10]  # Keep only the 10 most recent entries
            
            # Store current results in session
            session['filename'] = filename
            session['emotions'] = emotions
            session['filepath'] = filepath
            session['age_estimate'] = age_estimate
            session['age_confidence'] = 0.85  # Fixed confidence for age estimation
            session['recommendations'] = recommendations
            session['voice_features'] = voice_features
            session['plots_urls'] = plots_urls  # Store URLs instead of actual plot data
            session['upload_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return redirect(url_for('results'))
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            flash(f'Error processing file: {str(e)}', 'danger')
            return redirect(url_for('index'))
    else:
        flash('File type not allowed. Please upload MP3, WAV, or MP4 files only.', 'danger')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    if 'filename' not in session or 'emotions' not in session:
        flash('No file has been processed', 'warning')
        return redirect(url_for('index'))
    
    filename = session.get('filename')
    emotions = session.get('emotions')
    age_estimate = session.get('age_estimate', 'unknown')
    recommendations = session.get('recommendations', [])
    upload_date = session.get('upload_date', '')
    voice_features = session.get('voice_features', {})
    plots_urls = session.get('plots_urls', {})
    history = session.get('history', [])
    
    # Get age confidence value or default to 0.85 if not available
    age_confidence = session.get('age_confidence', 0.85)
    
    return render_template('results.html', 
                           filename=filename, 
                           emotions=emotions, 
                           age_estimate=age_estimate,
                           age_confidence=age_confidence,
                           recommendations=recommendations,
                           upload_date=upload_date,
                           voice_features=voice_features,
                           plots_urls=plots_urls,
                           history=history)

# Add history route
@app.route('/history')
def history():
    # Get history from session
    history = session.get('history', [])
    
    if not history:
        flash('No history available', 'info')
        return redirect(url_for('index'))
    
    return render_template('history.html', history=history)

@app.route('/clear_history')
def clear_history():
    # Clear history from session
    if 'history' in session:
        session.pop('history')
        flash('History cleared', 'success')
    
    return redirect(url_for('index'))

# Add voice recording capability
@app.route('/record', methods=['POST'])
def record_audio():
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Get the audio data from the request
        audio_data = request.files.get('audio')
        
        if not audio_data:
            return json.dumps({'error': 'No audio data received'}), 400
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"recorded_{timestamp}.wav"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the audio file
        audio_data.save(filepath)
        
        # Process the audio file
        emotions, age_estimate, voice_features, plots, recommendations = process_audio_file(filepath)
        
        # Store results in session (similar to upload_file route)
        plots_urls = {}
        static_dir = os.path.join('static', 'plots')
        os.makedirs(static_dir, exist_ok=True)
        
        # Generate unique identifiers for the plot files
        plot_id = f"{timestamp}_recorded"
        
        # Save each plot as a file and store the URL
        for plot_name, plot_data in plots.items():
            if plot_data:  # Only process if we have plot data
                plot_filename = f"{plot_name}_{plot_id}.png"
                plot_path = os.path.join(static_dir, plot_filename)
                
                # Decode base64 and save as file
                try:
                    with open(plot_path, 'wb') as f:
                        f.write(base64.b64decode(plot_data))
                    plots_urls[plot_name] = f"/static/plots/{plot_filename}"
                except Exception as e:
                    logging.error(f"Error saving plot {plot_name}: {str(e)}")
        
        # Create a record for history tracking
        analysis_record = {
            'filename': filename,
            'emotions': emotions,
            'age_estimate': age_estimate,
            'age_confidence': 0.85,  # Fixed confidence level for age estimation
            'recommendations': recommendations,
            'voice_features': voice_features,
            'plots_urls': plots_urls,
            'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'timestamp': datetime.now().timestamp(),
            'recorded': True  # Flag to indicate this was recorded not uploaded
        }
        
        # Initialize history in session if not present
        if 'history' not in session:
            session['history'] = []
        
        # Add current analysis to history (maximum 10 entries)
        history = session['history']
        history.insert(0, analysis_record)  # Add at the beginning (most recent first)
        session['history'] = history[:10]  # Keep only the 10 most recent entries
        
        # Store current results in session
        session['filename'] = filename
        session['emotions'] = emotions
        session['filepath'] = filepath
        session['age_estimate'] = age_estimate
        session['age_confidence'] = 0.85  # Fixed confidence for age estimation
        session['recommendations'] = recommendations
        session['voice_features'] = voice_features
        session['plots_urls'] = plots_urls
        session['upload_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session['recorded'] = True
        
        return json.dumps({'success': True, 'redirect': url_for('results')})
    
    except Exception as e:
        logging.error(f"Error processing recorded audio: {str(e)}")
        return json.dumps({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
