import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
import vertexai
from vertexai.generative_models import GenerativeModel
from groq import Groq
import traceback
import json
from google.api_core import exceptions as google_exceptions

# --- Initialize the Flask App & API Clients ---
app = Flask(__name__)
CORS(app)

# Google Vision client will use the secret file from its default location
vision_client = vision.ImageAnnotatorClient() 
# Groq client will use the environment variable set on Render
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- Define Google Cloud Project Details ---
GCP_PROJECT_ID = "acoustic-bridge-465311-v9"
GCP_LOCATION = "us-central1"

# --- API Functions ---
def extract_text_with_google_vision(image_content):
    """Uses Google Cloud Vision API for superior OCR."""
    print("Extracting text with Google Vision...")
    image = vision.Image(content=image_content)
    response = vision_client.text_detection(image=image)
    if response.error.message:
        raise Exception(response.error.message)
    print("OCR successful.")
    return response.text_annotations[0].description if response.text_annotations else ""

def extract_data_with_gemini(text_content):
    """Tries to use the Gemini model via Vertex AI to structure the text."""
    print("Attempting to use Gemini AI...")
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    model = GenerativeModel("gemini-1.0-pro")
    prompt = f"""
    You are an expert medical data extraction assistant. From the following medical prescription text, extract the information into a valid JSON object. The prescription is in French. Do not include the ```json markdown wrapper or any text other than the JSON object itself in your response.
    The JSON object must have keys: "doctor_name", "patient_name", "prescription_date", "patient_age", "patient_dob", and "medications" (a list of objects with "name" and "dosage_and_frequency").
    Text to Analyze: --- {text_content} ---
    """
    response = model.generate_content(prompt)
    print("Gemini AI processing successful.")
    return response.text

def extract_data_with_groq_ai(text_content):
    """Uses a Groq language model as a fallback."""
    print("Falling back to Groq AI...")
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are an expert medical data extraction assistant. From the user's text, which is from a French medical prescription, extract the information into a valid JSON object and nothing else.
                The JSON object must have keys: "doctor_name", "patient_name", "prescription_date", "patient_age", "patient_dob", and "medications" (a list of objects with "name" and "dosage_and_frequency").
                """
            },
            {
                "role": "user",
                "content": f"Here is the prescription text:\n\n---\n{text_content}\n---",
            }
        ],
        model="llama3-8b-8192",
    )
    print("Groq AI processing successful.")
    return chat_completion.choices[0].message.content

# --- API Endpoint ---
@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        image_content = file.read()
        ocr_text = extract_text_with_google_vision(image_content)
        
        structured_data_json_string = ""
        try:
            # First, try Gemini
            structured_data_json_string = extract_data_with_gemini(ocr_text)
        except google_exceptions.NotFound as e:
            # If Gemini fails with the permission error, fall back to Groq
            print(f"Gemini permission error: {e}. Falling back to Groq.")
            structured_data_json_string = extract_data_with_groq_ai(ocr_text)
        
        # Clean and return the response from whichever AI succeeded
        # This robustly finds the JSON object in the AI's response
        json_start = structured_data_json_string.find('{')
        json_end = structured_data_json_string.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            cleaned_response = structured_data_json_string[json_start:json_end]
        else:
            # If no JSON is found, return an error
            raise ValueError("AI response did not contain a valid JSON object.")

        return app.response_class(
            response=cleaned_response,
            status=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        print("!!! A SERVER ERROR OCCURRED !!!")
        print(traceback.format_exc())
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
