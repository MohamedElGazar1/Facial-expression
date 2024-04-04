import json
from PIL import Image
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import io

app = Flask(__name__)

# Initialize YOLO model
try:
    model = YOLO('FacialExpression.pt')
except Exception as e:
    print("Error initializing YOLO model:", e)
    model = None

def predict_yolo(image):
    try:
        # Check if model is initialized
        if model is None:
            return "Error: YOLO model is not initialized."

        # Perform YOLO prediction
        results = model.predict(image)
        xx = results[0].to_json()
        class_names_list = json.loads(xx)

        # Extract only the names
        class_names2 = [obj['name'] for obj in class_names_list]

        # Modify to handle output to count the objects
        word_counts = {}
        for word in class_names2:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

        output = []
        for word, count in word_counts.items():
            output.append(f"{count} {word}")

        class_names_string = ' and '.join(output)
        return class_names_string
    except Exception as e:
        return str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided.'}), 400

        image = request.files['image']

        if image.filename == '':
            return jsonify({'error': 'No image selected.'}), 400

        img_data = image.read()
        img = Image.open(io.BytesIO(img_data))

        # Perform YOLO prediction
        prediction_result = predict_yolo(img)

        return jsonify({'result': prediction_result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Handle 404 Not Found error
@app.errorhandler(404)
def not_found_error(error):
    return render_template('test.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
