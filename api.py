import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
import traceback

# --- Initialize the Flask App ---
app = Flask(__name__)
CORS(app)

# --- This is now the only function we need ---
def extract_text_with_google_vision(image_content):
    """Uses Google Cloud Vision API for superior OCR."""
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(response.error.message)
    # Return the full, raw text found in the image
    return response.text_annotations[0].description if response.text_annotations else ""

# --- Define the API Endpoint ---
@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        image_content = file.read()
        
        # Get the raw text from the image
        ocr_text = extract_text_with_google_vision(image_content)
        
        # Return the raw text directly
        return app.response_class(
            response=ocr_text,
            status=200,
            mimetype='text/plain' # Send it as plain text
        )
    except Exception as e:
        print("!!! A SERVER ERROR OCCURRED !!!")
        print(traceback.format_exc())
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
