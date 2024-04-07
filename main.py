from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import json

app = Flask(__name__)

# Load the YOLO model
model = YOLO("facialexpressionmodel.pt")

# Function to preprocess image from base64 data
def preprocess_image(base64_img):
    try:
        if base64_img is None:
            raise ValueError("Image data is None")

        image_data = base64.b64decode(base64_img)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image data")

        if img.size == 0:
            raise ValueError("Empty image data")

        return img
    except Exception as e:
        print("Error preprocessing image:", str(e))
        return None

# Function to perform YOLO prediction
def predict_yolo(image):
    try:
        # Perform YOLO prediction
        results = model.predict(image)
        xx = results[0].tojson()
        class_names_list = json.loads(xx)

        # Extract only the names
        class_names2 = [obj['name'] for obj in class_names_list]

        class_names_string = ' and '.join(class_names2)
        return class_names_string
    except Exception as e:
        return str(e)

@app.route('/api', methods=['POST'])
def predict():
    try:
        # Check if image is included in the request
        if 'image' not in request.json:
            return jsonify({'error': 'No image provided.'}), 400

        # Read the base64 image from the request
        base64_img = request.json['image']

        # Preprocess the image
        img = preprocess_image(base64_img)
        if img is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400

        # Perform YOLO prediction
        prediction_result = predict_yolo(img)

        # Return prediction result
        return jsonify({'prediction': prediction_result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=3000)



