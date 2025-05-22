import os
import gdown
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = "trashnet_model.keras"
GOOGLE_DRIVE_ID = "1jcL6I7JPSxoEkPf9O019_xCjR2dghRPr"

# 若本地尚未有模型，從 Google Drive 自動下載
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)
    file.save(file_path)

    try:
        img_array = prepare_image(file_path)
        preds = model.predict(img_array)
        pred_class = CLASS_NAMES[np.argmax(preds[0])]
        return jsonify({'result': pred_class})
    finally:
        os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
