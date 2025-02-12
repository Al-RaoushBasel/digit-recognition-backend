from flask import Flask, request, jsonify
from flask_cors import CORS  # Allows frontend to make requests
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS so frontend can call the API

# Get absolute path for model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "digit_model.h5")

def load_model():
    """Load the model only when needed (to save memory)"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    img = Image.open(io.BytesIO(image)).convert("L")  # Keep grayscale
    img = ImageOps.invert(img)  # Invert colors (black digit on white background)
    img = ImageOps.pad(img, (280, 280), color=255)  # Ensure square padding
    img = img.resize((32, 32))  # Resize to model input size
    
    img = np.array(img).astype("float32") / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    img = np.stack([img] * 3, axis=-1)  # Shape becomes (1, 32, 32, 3)

    return img


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to predict the digit."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = file.read()

    try:
        model = load_model()  # Load model inside the function (lazy loading)
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_digit = int(np.argmax(predictions))

        return jsonify({"prediction": predicted_digit})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is running"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=True)
