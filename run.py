# This file allows running the app directly in VS Code or any IDE
import os
import logging
import sys
from app import app

# Configure logging for better debugging in VS Code
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

# Ensure uploads directory exists
uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)
    logging.info(f"Created uploads directory at {uploads_dir}")

# Create directory for plots if it doesn't exist
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    logging.info(f"Created plots directory at {plots_dir}")

if __name__ == '__main__':
    # Get port from environment variable or use default 5000
    port = int(os.environ.get('PORT', 5000))
    
    logging.info(f"Starting application on port {port}")
    logging.info("Running in debug mode - not recommended for production")
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=port, debug=True)
