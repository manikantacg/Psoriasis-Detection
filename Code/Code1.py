import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

# Image Preprocessing
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    return image, gray, edges

# Load and preprocess images
def load_data(image_paths, labels):
    images = []
    for image_path in image_paths:
        _, gray, _ = preprocess_image(image_path)
        images.append(gray)
    images = np.array(images)
    images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))
    images = images / 255.0  # Normalize images
    labels = np.array(labels)
    return images, labels

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

# Training the ANN
def train_ann(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)
    model = create_ann(X_train[0].shape)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    return model

# Predicting Psoriasis Stage
def predict_stage(model, image_path):
    _, gray, _ = preprocess_image(image_path)
    gray = gray.reshape((1, gray.shape[0], gray.shape[1], 1))
    gray = gray / 255.0  # Normalize image
    prediction = model.predict(gray)
    stage = np.argmax(prediction)
    return stage

# Main script
if __name__ == "__main__":
    # Example image paths and labels
    image_paths = ["path_to_image1.jpg", "path_to_image2.jpg", "path_to_image3.jpg"]
    labels = [0, 1, 2]  # Replace with actual labels

    images, labels = load_data(image_paths, labels)
    model = train_ann(images, labels)
    
    # Test prediction
    test_image_path = "path_to_test_image.jpg"
    stage = predict_stage(model, test_image_path)
    print(f"The predicted stage of psoriasis is: {stage}")
