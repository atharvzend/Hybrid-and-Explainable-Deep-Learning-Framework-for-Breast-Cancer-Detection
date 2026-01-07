import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model configuration
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'hybrid_cnn_vit_model_epoch_10.pth')
NUM_CLASSES = 3

# Class labels and their meanings
CLASS_LABELS = {
    0: 'Benign',
    1: 'Malignant',
    2: 'Normal'
}

CLASS_DESCRIPTIONS = {
    'Benign': 'Non-cancerous growth. Regular monitoring recommended.',
    'Malignant': 'Cancerous tissue detected. Immediate medical consultation required.',
    'Normal': 'No abnormalities detected. Routine screening recommended.'
}

# Color mapping for visualization (BGR format for OpenCV)
HEATMAP_COLORS = {
    'high_attention': (0, 0, 255),      # Red - High suspicion
    'medium_attention': (0, 165, 255),  # Orange - Medium suspicion
    'low_attention': (0, 255, 0)        # Green - Low suspicion
}

# Image preprocessing
IMAGE_SIZE = (224, 224)
GRAYSCALE = True

# Upload configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask configuration
SECRET_KEY = 'your-secret-key-change-this-in-production'
DEBUG = True

# Simple authentication (for demo - replace with proper auth in production)
DEMO_CREDENTIALS = {
    'doctor': 'password123',
    'admin': 'admin123'
}

# Explainability settings
GRADCAM_ALPHA = 0.4  # Transparency for heatmap overlay
LIME_NUM_SAMPLES = 1000  # Number of samples for LIME
LIME_NUM_FEATURES = 10   # Top features to show