from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os
from flask_cors import CORS
from utils import (preprocess_image, get_class_labels, 
                   format_prediction_result, load_model_with_custom_objects)

# model VGG19 model H5 file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'VGG19', 'vgg19_fetal_brain.h5')

# Load the model with custom objects 
print(f"Loading model from: {MODEL_PATH}")
model = load_model_with_custom_objects(MODEL_PATH)

# model information
input_shape = model.input_shape
input_dims = input_shape[1:3] if input_shape[1:3] != (None, None) else (224, 224)
num_classes = model.output_shape[-1]
print(f"Model loaded successfully with {num_classes} output classes")
print(f"Input shape: {input_shape}")

app = Flask(__name__)
CORS(app)

@app.route('/predict-image', methods=['POST'])
def predict_image():
    try:
        if 'file' in request.files:
            try:
                file = request.files['file']
                image_data = file.read()
                import io
                from PIL import Image
                image = Image.open(io.BytesIO(image_data))
                
                processed_image = preprocess_image(image, target_size=input_dims)
            except Exception as e:
                return jsonify({'error': f'Error processing uploaded file: {str(e)}'}), 400
      
        prediction = model.predict(processed_image)
        
        result = format_prediction_result(prediction, get_class_labels())
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'up'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
