from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import requests
import os
import logging
from datetime import datetime
from PIL import Image
import io
import pathlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
BASE_DIR = pathlib.Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR.parent / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = f'{UPLOADS_DIR}'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
AI_BACKEND_URL = os.getenv('AI_BACKEND_URL', 'http://127.0.0.1:8001')
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Startup log
logger.info("Starting UI Backend Server...")
logger.info(f"AI Backend URL: {AI_BACKEND_URL}")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_image(file_path):
    """Validate that the file is a proper image"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            return True, img.format, img.size
    except Exception as e:
        return False, None, None

@app.route('/')
def index():
    """Serve the web interface"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        response = requests.get(f"{AI_BACKEND_URL}/health", timeout=5)
        ai_status = response.json() if response.status_code == 200 else {"status": "unreachable"}
    except Exception as e:
        logger.error(f"AI backend health check failed: {str(e)}")
        ai_status = {"status": "unreachable", "error": str(e)}
    
    return jsonify({
        "status": "healthy",
        "service": "UI Backend",
        "timestamp": datetime.now().isoformat(),
        "ai_backend": ai_status
    })

@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle image upload and forward to AI backend
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({
                "success": False,
                "error": "No image file provided"
            }), 400
        
        file = request.files['image']
        
        # Check if file was selected
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({
                "success": False,
                "error": "Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, webp"
            }), 400
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        base_name, ext = os.path.splitext(filename)
        unique_filename = f"{timestamp}_{base_name}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(filepath)
        logger.info(f"File saved: {filepath}")
        
        # Validate image
        is_valid, img_format, img_size = validate_image(filepath)
        if not is_valid:
            os.remove(filepath)
            logger.error("Invalid image file")
            return jsonify({
                "success": False,
                "error": "Invalid image file"
            }), 400
        
        logger.info(f"Image validated: {img_format}, {img_size}")
        
        # Forward to AI backend
        try:
            with open(filepath, 'rb') as f:
                files = {'file': (filename, f, file.content_type or 'application/octet-stream')}
                logger.info(f"Sending to AI backend: {AI_BACKEND_URL}/detect")
                
                response = requests.post(
                    f"{AI_BACKEND_URL}/detect",
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Detection successful: {result.get('detections_count', 0)} objects found")
                
                # Ensure the response has all required fields
                if 'success' not in result:
                    result['success'] = True
                if 'image_name' not in result:
                    result['image_name'] = filename
                if 'detections_count' not in result:
                    result['detections_count'] = len(result.get('detections', []))
                
                return jsonify(result)
            else:
                logger.error(f"AI backend error: {response.status_code} - {response.text}")
                return jsonify({
                    "success": False,
                    "error": f"AI backend error: {response.text}"
                }), response.status_code
                
        except requests.exceptions.Timeout:
            logger.error("AI backend timeout")
            return jsonify({
                "success": False,
                "error": "AI backend request timeout"
            }), 504
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to AI backend")
            return jsonify({
                "success": False,
                "error": "Cannot connect to AI backend service. Please ensure it's running."
            }), 503
        except Exception as e:
            logger.error(f"Error forwarding to AI backend: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Error communicating with AI backend: {str(e)}"
            }), 500
            
    except Exception as e:
        logger.error(f"Error during upload/detection: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/object-detection-microservice/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)