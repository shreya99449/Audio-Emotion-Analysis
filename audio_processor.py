import os
import logging
import numpy as np
import hashlib
import librosa

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# This implementation uses a more consistent approach for emotion detection
# It uses the file content hash rather than the file path for deterministic results
def process_audio_file(file_path):
    """
    Process an audio file and detect emotions.
    This uses file content to consistently generate the same emotion values for the same audio file.
    
    In a production environment, this would be replaced with a real ML model.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        dict: Dictionary of emotions and their scores
    """
    logging.info(f"Processing audio file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get a hash of the file content to ensure the same file always produces the same results
    file_hash = get_file_hash(file_path)
    
    # A more advanced implementation would use librosa for real audio analysis
    try:
        # Load the audio file using librosa for basic analysis
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract some basic audio features
        # These features will be used to influence the emotion scores
        # without actually doing real emotion detection
        
        # Average amplitude (volume) might correlate with intensity
        amplitude = np.abs(y).mean()
        
        # Spectral centroid indicates brightness/sharpness
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Zero crossing rate can relate to noisiness/speech
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Use features to seed random generator along with file hash
        seed_value = int(hashlib.sha256(f"{file_hash}{amplitude}{spectral_centroid}{zero_crossing}".encode()).hexdigest(), 16) % 2**32
        np.random.seed(seed_value)
        
        # Give some audio-based bias to emotions based on extracted features
        # Higher amplitude might increase probability of intense emotions (angry, happy)
        # Higher spectral centroid might relate to happiness
        # These are simplistic correlations, not real emotion detection
        
        happy_bias = min(0.3, spectral_centroid / 5000) + min(0.2, amplitude * 5)
        sad_bias = min(0.3, (1 - spectral_centroid / 5000)) - min(0.2, amplitude * 3)
        angry_bias = min(0.3, amplitude * 8) + min(0.1, zero_crossing)
        fearful_bias = min(0.2, zero_crossing * 2)
        neutral_bias = 0.2  # Base neutral bias
        
        # Generate random scores with bias
        emotions = {
            "happy": round(float(np.random.uniform(0, 1) + happy_bias), 2),
            "sad": round(float(np.random.uniform(0, 1) + sad_bias), 2),
            "angry": round(float(np.random.uniform(0, 1) + angry_bias), 2),
            "neutral": round(float(np.random.uniform(0, 1) + neutral_bias), 2),
            "fearful": round(float(np.random.uniform(0, 1) + fearful_bias), 2)
        }
        
    except Exception as e:
        logging.warning(f"Error in audio processing: {str(e)}. Falling back to basic method.")
        # Fallback to basic method if librosa processing fails
        np.random.seed(int(file_hash, 16) % 2**32)
        
        # Simple emotion scores without audio analysis
        emotions = {
            "happy": round(float(np.random.uniform(0, 1)), 2),
            "sad": round(float(np.random.uniform(0, 1)), 2),
            "angry": round(float(np.random.uniform(0, 1)), 2),
            "neutral": round(float(np.random.uniform(0, 1)), 2),
            "fearful": round(float(np.random.uniform(0, 1)), 2)
        }
    
    # Normalize the scores so they sum to 1
    total = sum(emotions.values())
    for emotion in emotions:
        emotions[emotion] = round(emotions[emotion] / total, 2)
    
    logging.info(f"Detected emotions: {emotions}")
    return emotions


def get_file_hash(file_path, read_size=8192):
    """
    Calculate SHA-256 hash of a file to use as a consistent identifier.
    Reads the file in chunks to handle large files efficiently.
    
    Args:
        file_path (str): Path to the file
        read_size (int): Chunk size for reading the file
        
    Returns:
        str: Hexadecimal digest of the file hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks
        for byte_block in iter(lambda: f.read(read_size), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# Future improvements:
# 1. Implement actual emotion detection model using deep learning
# 2. Add support for batch processing
# 3. Implement caching for processed files
# 4. Add more sophisticated audio feature extraction
