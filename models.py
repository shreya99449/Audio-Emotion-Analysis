from app import db
from datetime import datetime

class AudioFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(512), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    gender = db.Column(db.String(50), nullable=True, default='unknown')
    emotion_results = db.relationship('EmotionResult', backref='audio_file', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<AudioFile {self.filename}>'

class EmotionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    audio_file_id = db.Column(db.Integer, db.ForeignKey('audio_file.id'), nullable=False)
    emotion = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<EmotionResult {self.emotion}: {self.score}>'
