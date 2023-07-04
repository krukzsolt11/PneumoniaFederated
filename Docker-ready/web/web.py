from flask import Flask, request
import flasgger
from flasgger import Swagger
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import pandas 
from PIL import Image

MODEL_PATH = "/app/data/model_round_10.h5"

model = load_model(MODEL_PATH)

app = Flask(__name__)
Swagger(app)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    """Let's Predict Pneumonia
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    img = Image.open(request.files.get("file"))
    img = img.convert('RGB')  # Convert to RGB mode
    img = img.resize((180, 180))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    img_data = tf.keras.applications.vgg16.preprocess_input(x)
    
    classes = model.predict(img_data)
    A = np.squeeze(np.asarray(classes))
    if A[1] == 1:
        return "PNEUMONIA"
    else:
        return "NORMAL"

    
if __name__ == '__main__':
    app.run()