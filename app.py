from flask import Flask, request, jsonify
from flask_cors import CORS  # Allows frontend to make requests
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io
import os

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS so frontend can call the API

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "digit_model.h5")
  # Model is in backend root
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    """Enhanced preprocessing to match Colab results."""
    img = Image.open(io.BytesIO(image)).convert("L")  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors (ensure black digit on white background)
    img = ImageOps.pad(img, (280, 280), color=255)  # Add padding to make it square
    img = img.resize((32, 32))  # Resize to match the model's input size
    img = np.stack([img] * 3, axis=-1)  # Convert grayscale to RGB (3 channels)
    img = np.array(img).astype("float32") / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to predict the digit."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Get the uploaded image
    file = request.files["file"]
    image = file.read()

    try:
        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        predictions = model.predict(processed_image)
        predicted_digit = int(np.argmax(predictions))

        # Return the result as JSON
        return jsonify({"prediction": predicted_digit})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is running"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
