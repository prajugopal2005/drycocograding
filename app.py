"""
Flask Backend for Coconut Purity Grading System
Handles file uploads and prediction routing with enhanced error handling
"""

from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from predict import predict_purity, predict_with_cloud_api
import os
import uuid
from werkzeug.utils import secure_filename
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'coconut_purity_grading_secret_key_2024'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('model', exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file):
    """Validate uploaded image file"""
    try:
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            return False, "File size too large. Maximum size is 16MB."
        
        # Try to open with PIL to validate image
        img = Image.open(file)
        img.verify()
        file.seek(0)  # Reset to beginning
        return True, "Valid image"
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

@app.route('/')
def home():
    """Displays upload page (HTML form)"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    """
    Accepts uploaded image → Saves → Runs predict_purity() → Returns results
    Enhanced with better error handling and file validation
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            flash('No image file selected', 'error')
            return redirect(url_for('home'))
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            flash('No image file selected', 'error')
            return redirect(url_for('home'))
        
        # Check file extension
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, TIFF)', 'error')
            return redirect(url_for('home'))
        
        # Validate image
        is_valid, message = validate_image(file)
        if not is_valid:
            flash(message, 'error')
            return redirect(url_for('home'))
        
        # Generate unique filename to avoid conflicts
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
        
        # Save the uploaded file
        path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(path)
        
        # Run prediction
        try:
            label, confidence = predict_purity(path)
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'error')
            return redirect(url_for('home'))
        
        # Return result page
        return render_template('result.html', 
                             label=label, 
                             confidence=confidence, 
                             image_path=path,
                             filename=unique_filename)
    
    except Exception as e:
        flash(f'Upload error: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/predict-cloud', methods=['POST'])
def predict_with_cloud():
    """
    Alternative prediction using cloud API (Google Cloud Vision)
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file selected'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Validate image
        is_valid, message = validate_image(file)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
        path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(path)
        
        # Run cloud prediction
        try:
            label, confidence = predict_with_cloud_api(path)
        except Exception as e:
            return jsonify({'error': f'Cloud prediction error: {str(e)}'}), 500
        
        return jsonify({
            'label': label,
            'confidence': confidence,
            'image_path': path,
            'filename': unique_filename
        })
    
    except Exception as e:
        return jsonify({'error': f'Cloud prediction error: {str(e)}'}), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Coconut Purity Grading System',
        'version': '1.0.0'
    })

if __name__ == "__main__":
    print("=" * 60)
    print("🥥 Automated Purity Grading System for Dry Coconuts")
    print("=" * 60)
    print("Starting Flask server...")
    print("Access the application at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True)
