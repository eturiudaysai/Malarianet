from flask import Flask, render_template, request, jsonify
from keras.preprocessing import image
import numpy as np
import os
import keras

app = Flask(__name__)


loaded_model = keras.models.load_model('malaria_model.h5')

# Function to load the model
def load_model():
    global loaded_model
    if loaded_model is None:
        loaded_model = keras.models.load_model('malaria_model.h5')

# Define a function to make predictions on a single image
def predict_single_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Ensure target size matches the model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Rescale to match the preprocessing used during training

    prediction = loaded_model.predict(img_array)

    if prediction[0][0] > 0.5:
        return "Malaria Detected"
    else:
        return "Malaria Not Detected"

# Define precautions and suggestions
precautions = [
    "Seek immediate medical attention.",
    "Stay hydrated by drinking plenty of fluids.",
    "Rest and avoid physical exertion.",
    "Use mosquito repellent and wear long-sleeved clothing.",
    "Sleep under a mosquito net.",
    "Follow the prescribed medication regimen.",
    "Avoid self-medication.",
    "Prevent mosquito breeding by eliminating standing water around your home."
]

suggestions = [
    "Maintain good hygiene practices.",
    "Use insect repellent containing DEET.",
    "Wear protective clothing, such as long-sleeved shirts and pants, especially during peak mosquito activity times.",
    "Install window and door screens to keep mosquitoes out.",
    "Use mosquito nets while sleeping, especially in areas with high malaria transmission.",
    "Consider taking antimalarial medication if traveling to areas where malaria is prevalent."
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    load_model()  # Load the model
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the file to a temporary location
        if not os.path.exists('temp'):
            os.makedirs('temp')
        file_path = os.path.join('temp', file.filename)
        file.save(file_path)

        # Make prediction on the input image
        result = predict_single_image(file_path)

        # Prepare additional information based on prediction result
        additional_info = ""
        if result == "Malaria Detected":
            additional_info += "Precautions to be taken:\n\n"
            additional_info += "\n".join(precautions)
            additional_info += "\nSuggestions to reduce risk:\n"
            additional_info += "\n".join(suggestions)
        else:
            additional_info += "\nCongratulations! Malaria Not Detected."
            additional_info += "\nSuggestions to prevent malaria:\n"
            additional_info += "\n".join(suggestions)

        return jsonify({'result': result, 'additional_info': additional_info}), 200, {'Content-Type': 'text/html'}
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Remove the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
