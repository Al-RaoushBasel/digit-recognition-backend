from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import io
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "digit_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

def preprocess_image(image):
    img = Image.open(io.BytesIO(image)).convert("L")
    img = ImageOps.invert(img)
    img = ImageOps.pad(img, (280, 280), color=255)
    img = img.resize((32, 32))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.stack([img] * 3, axis=-1)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = file.read()

    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_digit = int(np.argmax(predictions))
        return jsonify({"prediction": predicted_digit})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is running"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
