from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load trained model
model = tf.keras.models.load_model("model/digit_recognition_model.h5")


def preprocess_image(image):
    """ Convert image to MNIST format """
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dimension
    return image


@app.route("/predict", methods=["POST"])
def predict():
    """ Handle image prediction request """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    digit = np.argmax(prediction)

    return jsonify({"digit": int(digit)})


if __name__ == "__main__":
    app.run(debug=True)
