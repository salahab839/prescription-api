import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
import vertexai
from vertexai.generative_models import GenerativeModel
import traceback
import json

# --- Initialize the Flask App ---
app = Flask(__name__)
CORS(app)

# --- Define Google Cloud Project Details ---
GCP_PROJECT_ID = "acoustic-bridge-465311-v9"
GCP_LOCATION = "us-central1" # A common, stable region for AI models

# --- Function to use Google Vision API for OCR ---
def extract_text_with_google_vision(image_content):
    """Uses Google Cloud Vision API for superior OCR."""
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(response.error.message)
    return response.text_annotations[0].description if response.text_annotations else ""

# --- Function to use Gemini AI for Data Extraction ---
def extract_data_with_gemini(text_content):
    """Uses the Gemini model via Vertex AI to understand and structure the text."""
    
    # Initialize Vertex AI
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    
    # Use a stable and widely available Gemini model
    model = GenerativeModel("gemini-1.0-pro")

    # The prompt asking the AI what to do
    prompt = f"""
    You are an expert medical data extraction assistant. From the following medical prescription text, extract the information into a valid JSON object. The prescription is in French. Do not include the ```json markdown wrapper or any text other than the JSON object itself in your response.

    Here are the fields to extract:
    - "doctor_name": The full name of the doctor. Find the most specific name, not just the title.
    - "patient_name": The full name of the patient (first and last).
    - "prescription_date": The date the prescription was written.
    - "patient_age": The patient's age, if available.
    - "patient_dob": The patient's date of birth, if available.
    - "medications": A list of all medications. Each item in the list should be an object with two keys: "name" (the full name and dosage of the medication) and "dosage_and_frequency" (the instructions on how to take it).

    Prescription Text to Analyze:
    ---
    {text_content}
    ---
    """

    response = model.generate_content(prompt)
    
    # Clean up the response to get only the JSON part
    cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
    
    return cleaned_response

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
        
        # Step 1: Use Google Vision to get perfect text
        ocr_text = extract_text_with_google_vision(image_content)
        
        # Step 2: Use Gemini to understand the text and extract data
        structured_data_json_string = extract_data_with_gemini(ocr_text)
        
        # The AI returns a JSON string, so we return it directly
        # We set the content type to application/json
        return app.response_class(
            response=structured_data_json_string,
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        print("!!! A SERVER ERROR OCCURRED !!!")
        print(traceback.format_exc())
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
