import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    return gray

# Define the Neural Network
def create_ann(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # Assuming 3 stages of psoriasis
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load trained model (assuming the model is already trained and saved)
model = create_ann((128, 128, 1))
model.load_weights('model_weights.h5')  # Adjust the path as necessary

# Predict stage of psoriasis
def predict_stage(model, image_path):
    gray = preprocess_image(image_path)
    gray = cv2.resize(gray, (128, 128))
    gray = gray.reshape((1, 128, 128, 1))
    gray = gray / 255.0  # Normalize image
    prediction = model.predict(gray)
    stage = np.argmax(prediction)
    return stage

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            stage = predict_stage(model, filepath)
            return render_template('result.html', stage=stage)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
