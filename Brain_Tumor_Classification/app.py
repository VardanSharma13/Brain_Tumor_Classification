from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# Load the trained model (make sure the model file is in the correct directory)
model = load_model('model.h5')  # Adjust the path if needed

# Class names as per your model (ensure these match your training classes)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Treatment information for each class
treatment_info = {
    'glioma': {
        'treatment': [
            "Surgery: Removal of the tumor.",
            "Radiation Therapy: High-energy rays to target cancer cells.",
            "Chemotherapy: Using drugs to kill cancer cells.",
            "Targeted Therapy: Drugs that target specific cancer cell proteins."
        ],
        'description': "Gliomas are a type of tumor that starts in the brain or spine. They are often aggressive."
    },
    'meningioma': {
        'treatment': [
            "Surgery: Most common treatment for meningiomas.",
            "Radiation Therapy: For inoperable or recurrent tumors.",
            "Observation: If the tumor is small and not growing, regular monitoring is an option."
        ],
        'description': "Meningiomas are typically benign tumors that form in the meninges, the protective layers of tissue covering the brain and spinal cord."
    },
    'notumor': {
        'treatment': [
            "No treatment needed, as there is no tumor.",
            "Routine monitoring to ensure there are no changes."
        ],
        'description': "No tumor detected. Continue with regular check-ups if needed."
    },
    'pituitary': {
        'treatment': [
            "Surgery: Removing the tumor from the pituitary gland.",
            "Radiation Therapy: Used when surgery isn't possible or to shrink the tumor.",
            "Medication: To control hormone levels produced by the tumor."
        ],
        'description': "Pituitary tumors affect the pituitary gland at the base of the brain, influencing hormone production."
    }
}

# Preprocessing function for images
def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((128, 128))  # Resize to match the model input size
    img = np.array(img) / 255.0  # Normalize pixel values
    if img.shape[-1] != 3:
        img = np.stack([img] * 3, axis=-1)  # Convert grayscale to RGB if needed
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def index():
    return render_template('index.html')  # Home page with file upload form

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure the request has a file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file to the uploads folder
    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)

    # Preprocess the image for prediction
    img = preprocess_image(file_path)

    # Make the prediction using the loaded model
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]  # Get the predicted class

    # Get the treatment info for the predicted class
    treatment = treatment_info.get(predicted_class, {}).get('treatment', [])
    description = treatment_info.get(predicted_class, {}).get('description', "")

    # Render the prediction result page with the prediction and treatment information
    return render_template('prediction.html', 
                           prediction=predicted_class,
                           treatment=treatment,
                           description=description)

if __name__ == "__main__":
    # Make sure the uploads directory exists
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')

    app.run(debug=True)
