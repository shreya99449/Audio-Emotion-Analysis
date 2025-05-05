import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define allowed extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4'}

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///audio_emotions.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize the app with the extension
db.init_app(app)

# Import after app initialization to avoid circular imports
from models import AudioFile, EmotionResult
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
            # Process audio file and get emotions and gender
            emotions, gender = process_audio_file(filepath)
            
            # Store file info and results in database
            audio_file = AudioFile(
                filename=filename,
                filepath=filepath,
                upload_date=datetime.now(),
                gender=gender
            )
            db.session.add(audio_file)
            db.session.flush()  # To get the audio_file.id
            
            # Store each emotion result
            for emotion, score in emotions.items():
                emotion_result = EmotionResult(
                    audio_file_id=audio_file.id,
                    emotion=emotion,
                    score=score
                )
                db.session.add(emotion_result)
            
            db.session.commit()
            
            # Store the filename, emotions and gender in session for the results page
            session['filename'] = filename
            session['emotions'] = emotions
            session['filepath'] = filepath
            session['gender'] = gender
            
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
    
    return render_template('results.html', filename=filename, emotions=emotions, gender=gender)

# Initialize database within app context
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
