# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:51:11 2024

@author: justinjoseph
"""

from keras.models import load_model
import os
import glob
import cv2
import numpy as np

# Define your paths here
model_path = '*.hdf5'  # Update this to the filepath and name of your model (include the file extension 'hdf5')
input_directory = '*'  # Update this path to the location of your 'Input_images'
output_directory = '*'  # Update this path to the location of your "Prediction_output"

def load_model_from_path(model_path):
    return load_model(model_path)

def predict_images(input_directory, output_directory, model):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Create folders for original and visualized predictions
    original_pred_folder = os.path.join(output_directory, 'Original_Predictions')
    visualized_pred_folder = os.path.join(output_directory, 'Visualized_Predictions')

    if not os.path.exists(original_pred_folder):
        os.makedirs(original_pred_folder)
    if not os.path.exists(visualized_pred_folder):
        os.makedirs(visualized_pred_folder)

    images = []
    image_names = []

    for img_path in glob.glob(os.path.join(input_directory, "*.tif*")):  # Will match both .tif and .tiff
        img = cv2.imread(img_path, 1)  # 1 means reading as RGB

        if img is None:
            print(f"Skipping {img_path}: Could not read image.")
            continue

        if img.shape[2] != 3:  # Checking if the image is RGB
            print(f"Skipping {img_path}: Not an RGB image.")
            continue

        # Resize images if not the expected size
        if img.shape[:2] != (192, 256):
            print(f"Resizing {img_path}: Image size is not 256x192, resizing.")
            img = cv2.resize(img, (256, 192))

        images.append(img)
        image_names.append(os.path.basename(img_path))

    images = np.array(images)
    images = images / 255.0  # Normalize images to [0, 1]

    for i, img in enumerate(images):
        img_input = np.expand_dims(img, 0)  # Expand dims to fit model input
        prediction = model.predict(img_input)
        predicted_img = np.argmax(prediction, axis=-1)[0, :, :]  # Get prediction

        # Save original prediction
        original_output_path = os.path.join(original_pred_folder, f'prediction_{image_names[i]}')
        cv2.imwrite(original_output_path, predicted_img)

        # Normalize and save visualized prediction
        normalized_img = ((predicted_img / np.max(predicted_img)) * 255).astype(np.uint8)
        visualized_output_path = os.path.join(visualized_pred_folder, f'visualized_prediction_{image_names[i]}')
        cv2.imwrite(visualized_output_path, normalized_img)

# Loading the model
model = load_model_from_path(model_path)

# Running prediction on images
predict_images(input_directory, output_directory, model)
