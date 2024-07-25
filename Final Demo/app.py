import os
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from PIL import Image
import atexit
import shutil
import webbrowser
import threading
import time

app = Flask(__name__)

# Configure upload folder and allowed extensions
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
MODEL_FOLDER = os.path.join(BASE_DIR, 'model')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the necessary folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Set the model path
MODEL_PATH = os.path.join(MODEL_FOLDER, "generator_epoch_1.h5")

# Load the model (you'll need to place the model file in the 'model' folder)
g_model = load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    norm_img = (img - 127.5) / 127.5
    return img, norm_img

def generate_image(model, norm_img):
    g_img = model.predict(np.expand_dims(norm_img, 0))[0]
    g_img = g_img * 127.5 + 127.5
    return np.clip(g_img, 0, 255).astype('uint8')

def compute_l2(img1, img2):
    return np.mean((np.square(img1 - img2)))

def compute_ssim(img1, img2, win_size=7, channel_axis=-1):
    return ssim(img1, img2, win_size=win_size, channel_axis=channel_axis)

def clear_upload_folder():
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Clear upload folder at startup
clear_upload_folder()

# Register function to clear upload folder at shutdown
atexit.register(clear_upload_folder)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'sketch' not in request.files or 'target' not in request.files:
            return jsonify({'error': 'Both sketch and target files are required'}), 400
        
        sketch_file = request.files['sketch']
        target_file = request.files['target']
        
        if sketch_file.filename == '' or target_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if sketch_file and allowed_file(sketch_file.filename) and target_file and allowed_file(target_file.filename):
            sketch_filename = secure_filename(sketch_file.filename)
            target_filename = secure_filename(target_file.filename)
            
            sketch_path = os.path.join(app.config['UPLOAD_FOLDER'], sketch_filename)
            target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_filename)
            
            # Resize and save sketch
            sketch_img = Image.open(sketch_file)
            sketch_img = sketch_img.resize((256, 256), Image.LANCZOS)
            sketch_img.save(sketch_path)
            
            # Resize and save target
            target_img = Image.open(target_file)
            target_img = target_img.resize((256, 256), Image.LANCZOS)
            target_img.save(target_path)
            
            # Process images
            img, norm_img = load_and_preprocess_image(sketch_path)
            g_img = generate_image(g_model, norm_img)
            
            # Compute scores
            target = cv2.imread(target_path)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            l2 = compute_l2(g_img, target)
            ssim_score = compute_ssim(g_img, target, win_size=7, channel_axis=-1)
            
            # Save generated image
            generated_filename = 'generated_' + sketch_filename
            generated_path = os.path.join(app.config['UPLOAD_FOLDER'], generated_filename)
            plt.imsave(generated_path, g_img.astype(np.uint8))
            
            return jsonify({
                'sketch': os.path.join('static/uploads', sketch_filename),
                'generated': os.path.join('static/uploads', generated_filename),
                'target': os.path.join('static/uploads', target_filename),
                'l2': float(l2),
                'ssim': float(ssim_score),
                'order': ['sketch', 'generated', 'target']
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

def open_browser():
    time.sleep(1)  # Wait for the server to start
    webbrowser.open('http://127.0.0.1:5000/')

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(debug=True, use_reloader=False)
