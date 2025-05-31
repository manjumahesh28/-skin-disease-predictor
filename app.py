
from flask import Flask, request, render_template
from PIL import Image
from io import BytesIO
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import keras
from keras import backend as K
import os

app = Flask(__name__)

SKIN_CLASSES = {
    0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion'
}

# Load model ONCE at startup
with open('modelnew.json', 'r') as j_file:
    loaded_json_model = j_file.read()
model = model_from_json(loaded_json_model)
model.load_weights('modelnew.h5')
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def login():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    f = request.files['file']
    if f.filename == '':
        return "No selected file", 400

    if f.mimetype not in ['image/jpeg', 'image/png']:
        return "Invalid file type, only JPEG and PNG are allowed", 400

    if len(f.read()) == 0:
        return "Uploaded file is empty.", 400
    f.seek(0)

    try:
        img_bytes = BytesIO(f.read())
        img = Image.open(img_bytes)
        img.verify()
        f.seek(0)
    except (IOError, SyntaxError) as e:
        return f"Invalid image file: {e}", 400

    img1 = image.load_img(img_bytes, target_size=(224, 224))
    img1 = image.img_to_array(img1)
    img1 = np.expand_dims(img1, axis=0)
    img1 = img1 / 255.0

    prediction = model.predict(img1)
    pred = np.argmax(prediction)
    disease = SKIN_CLASSES[pred]
    accuracy = prediction[0][pred] * 100

    K.clear_session()
    print(disease)
    print(accuracy)
    return render_template('result.html', title='Result', predictions=disease, acc=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
