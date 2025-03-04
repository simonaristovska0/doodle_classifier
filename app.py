from flask import Flask, request, jsonify, render_template
import base64
import io
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model("full_doodle_model.h5")
print("Model loaded successfully on the server.")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    image_data = data.get('image')
    if image_data is None:
        return jsonify({'error': 'No image provided'}), 400

    if ',' in image_data:
        header, encoded = image_data.split(',', 1)
    else:
        encoded = image_data

    try:
        img_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        return jsonify({'error': f'Error processing image: {e}'}), 400

    predictions = model.predict(img_array)
    probabilities = predictions[0].tolist()
    predicted_class = int(np.argmax(predictions[0]))

    return jsonify({'predictions': probabilities, 'predicted_class': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)
