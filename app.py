import os
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# Flask app initialization
app = Flask(__name__ ,template_folder=r'F:\TpDeVops\Front-end\templates')

# Resolve the model path dynamically
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model1.keras')

# Verify that the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the model
model = load_model(MODEL_PATH)
print(f'Model loaded from {MODEL_PATH}. Check http://127.0.0.1:5000/')

# Class labels
LABELS = ['Normal', 'Pneumonia']  # Update as needed

# Ensure uploads directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_class_name(class_no):
    """Return the class name based on class index."""
    return LABELS[class_no]

def preprocess_image(img_path):
    """Preprocess the image for prediction."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    if img is None:
        raise ValueError("Failed to load image. Ensure the file is a valid image format.")
    img_resized = cv2.resize(img, (128, 128))  # Resize to match model input size
    img_normalized = img_resized.astype('float32') / 255.0  # Normalize pixel values
    img_normalized = np.expand_dims(img_normalized, axis=-1)  # Add channel dimension
    img_normalized = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
    return img_normalized

def predict_result(img_path):
    """Run the prediction on the preprocessed image."""
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)
    class_idx = np.argmax(prediction, axis=1)[0]  # Get the class index
    confidence = float(prediction[0][class_idx])  # Get confidence score
    return class_idx, confidence

@app.route('/', methods=['GET'])
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(file_path)

    try:
        # Get the prediction result
        class_idx, confidence = predict_result(file_path)
        result = {
            'class': get_class_name(class_idx),
            'confidence': confidence
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
