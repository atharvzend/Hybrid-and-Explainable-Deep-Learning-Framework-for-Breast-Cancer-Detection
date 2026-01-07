from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import torch
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

# Import custom modules
from config import *
from models.architecture import load_model
from utils.preprocessing import preprocess_image, load_image_for_display
from utils.visualization import (
    create_gradcam_visualization, 
    create_lime_visualization,
    save_visualization
)
from explainability.gradcam import generate_gradcam_explanation
from explainability.lime_explain import generate_lime_explanation

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading model on device: {device}")
model = load_model(MODEL_PATH, NUM_CLASSES, device)
print("Model loaded successfully!")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def pil_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def numpy_to_base64(image_array):
    """Convert numpy array to base64 string"""
    image = Image.fromarray(image_array.astype(np.uint8))
    return pil_to_base64(image)


@app.route('/')
def index():
    """Landing page - redirect to login"""
    return redirect(url_for('login'))


@app.route('/login', methods=['GET'])
def login():
    """Login page"""
    return render_template('login.html')


@app.route('/authenticate', methods=['POST'])
def authenticate():
    """Handle login authentication"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    # Simple authentication (replace with proper auth in production)
    if username in DEMO_CREDENTIALS and DEMO_CREDENTIALS[username] == password:
        session['user'] = username
        return jsonify({'success': True, 'message': 'Login successful'})
    else:
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401


@app.route('/logout')
def logout():
    """Logout user"""
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/home')
def home():
    """Main application page"""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['user'])


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image for model
        input_tensor = preprocess_image(filepath, IMAGE_SIZE)
        input_tensor = input_tensor.to(device)
        
        # Load original image for display
        original_image = load_image_for_display(filepath)
        original_size = original_image.size  # (width, height)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item() * 100
        
        # Get class label and description
        class_label = CLASS_LABELS[predicted_class]
        class_description = CLASS_DESCRIPTIONS[class_label]
        
        # Get all class probabilities
        all_probs = {CLASS_LABELS[i]: float(probabilities[0, i].item() * 100) for i in range(NUM_CLASSES)}
        
        # Convert original image to base64
        original_base64 = pil_to_base64(original_image)
        
        # Store filepath in session for explainability
        session['last_upload'] = filepath
        session['last_prediction'] = predicted_class
        
        return jsonify({
            'success': True,
            'prediction': {
                'class': class_label,
                'class_id': predicted_class,
                'confidence': round(confidence, 2),
                'description': class_description,
                'all_probabilities': all_probs
            },
            'original_image': original_base64
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'}), 500


@app.route('/explain', methods=['POST'])
def explain():
    """Generate explainability visualizations (Grad-CAM + LIME)"""
    if 'user' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    if 'last_upload' not in session:
        return jsonify({'success': False, 'message': 'No image to explain'}), 400
    
    try:
        filepath = session['last_upload']
        predicted_class = session.get('last_prediction')
        
        # Load original image
        original_image = load_image_for_display(filepath)
        original_size = (original_image.size[1], original_image.size[0])  # (height, width)
        
        # Preprocess for model
        input_tensor = preprocess_image(filepath, IMAGE_SIZE)
        input_tensor = input_tensor.to(device)
        
        # Generate Grad-CAM explanation
        print("Generating Grad-CAM...")
        gradcam_result = generate_gradcam_explanation(model, input_tensor, original_size)
        gradcam_viz = create_gradcam_visualization(
            original_image, 
            gradcam_result['heatmap'], 
            alpha=GRADCAM_ALPHA
        )
        
        # Generate LIME explanation
        print("Generating LIME explanation...")
        lime_result = generate_lime_explanation(
            model, 
            original_image, 
            device=device,
            num_samples=LIME_NUM_SAMPLES,
            num_features=LIME_NUM_FEATURES
        )
        lime_viz = lime_result['visualization']
        
        # Convert visualizations to base64
        gradcam_base64 = numpy_to_base64(gradcam_viz)
        lime_base64 = numpy_to_base64(lime_viz)
        
        # Save visualizations (optional)
        gradcam_save_path = os.path.join(UPLOAD_FOLDER, 'gradcam_' + os.path.basename(filepath))
        lime_save_path = os.path.join(UPLOAD_FOLDER, 'lime_' + os.path.basename(filepath))
        save_visualization(gradcam_viz, gradcam_save_path)
        save_visualization(lime_viz, lime_save_path)
        
        return jsonify({
            'success': True,
            'gradcam': gradcam_base64,
            'lime': lime_base64,
            'prediction': {
                'class': CLASS_LABELS[predicted_class],
                'confidence': round(gradcam_result['confidence'], 2)
            }
        })
    
    except Exception as e:
        print(f"Error in explain: {str(e)}")
        return jsonify({'success': False, 'message': f'Error generating explanations: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })


if __name__ == '__main__':
    # Ensure upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Run the app
    print(f"\n{'='*50}")
    print("🏥 Breast Cancer Detection System")
    print(f"{'='*50}")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {device}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"\n🚀 Starting server...")
    print(f"{'='*50}\n")
    
    app.run(debug=DEBUG, host='0.0.0.0', port=5000)