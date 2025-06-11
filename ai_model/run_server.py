"""
Run script for the deep learning classification API server
"""
from app import app

if __name__ == "__main__":
    print("Starting VGG16 Classification API server...")
    print("API will be available at: http://localhost:5000")
    print("Available endpoints:")
    print(" - GET /health - Health check")
    print(" - POST /predict-image - Predict from image")
    
    app.run(debug=True, host='0.0.0.0', port=5000)