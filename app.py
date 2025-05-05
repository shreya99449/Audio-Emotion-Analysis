import os
import logging
import base64
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
            # Process audio file and get emotions, gender, voice features and plots
            emotions, gender, voice_features, plots = process_audio_file(filepath)
            
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
            
            # Store other results in session
            session['filename'] = filename
            session['emotions'] = emotions
            session['filepath'] = filepath
            session['gender'] = gender
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
    gender = session.get('gender', 'unknown')
    upload_date = session.get('upload_date', '')
    voice_features = session.get('voice_features', {})
    plots_urls = session.get('plots_urls', {})
    
    return render_template('results.html', 
                           filename=filename, 
                           emotions=emotions, 
                           gender=gender, 
                           upload_date=upload_date,
                           voice_features=voice_features,
                           plots_urls=plots_urls)

# No database initialization needed

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
