from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications import efficientnet

import os

image = tf.keras.preprocessing.image
preprocess_input = efficientnet.preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "food_classifier_50.keras")

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)




model = tf.keras.models.load_model(MODEL_PATH)



class_names = ['apple_pie', 'baklava', 'beet_salad', 'caprese_salad', 'carrot_cake', 'chicken_quesadilla', 'clam_chowder', 'crab_cakes', 'deviled_eggs', 'edamame', 'fish_and_chips', 'fried_calamari', 'frozen_yogurt', 'greek_salad', 'grilled_salmon', 'hamburger', 'ice_cream', 'miso_soup', 'peking_duck', 'pork_chop', 'ravioli', 'risotto', 'strawberry_shortcake', 'sushi', 'tuna_tartare']


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)
    return pred_class, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded'
        file = request.files['file']
        if file.filename == '':
            return 'No file selected'
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Make prediction
        label, conf = model_predict(file_path, model)
        return render_template('index.html', filename=file.filename, label=label, confidence=conf, classes=class_names)

    # return render_template('index.html', filename=None)
    return render_template("index.html", classes=class_names)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

