import json
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO
import io


app = Flask(__name__)
model = YOLO('FacialExpression.pt')

def predict_yolo(image):
    try:
        # Perform YOLO prediction
        results = model.predict(image)
        # cv2.waitKey(0)
        xx=results[0].tojson()
        class_names_list = json.loads(xx)

        # Extract only the names
        class_names2 = [obj['name'] for obj in class_names_list]
        #modify to handle output to count the objects
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
        return  str(e)

@app.route('/', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'No image selected.'}), 400

    if image:
        img_data = image.read()
        img = Image.open(io.BytesIO(img_data))

        # Perform YOLO prediction
        prediction_result = predict_yolo(img)

        return jsonify(prediction_result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
