import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
from groq import Groq
import traceback
import json

# --- Initialize the Flask App & API Clients ---
app = Flask(__name__)
CORS(app)

vision_client = vision.ImageAnnotatorClient() 
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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

# --- NEW FUNCTION FOR VIGNETTE SCANNING ---
def extract_vignette_data_with_groq(text_content):
    """Uses Groq AI to extract specific fields from a medication vignette."""
    print("Sending vignette text to Groq for analysis...")
    
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are an expert at reading French medication vignettes (the small price stickers). From the user's text, extract the information into a valid JSON object and nothing else.

                The JSON object must have these exact keys:
                - "nom": The commercial name of the medication.
                - "dosage": The dosage information (e.g., "500MG", "100 MG/5 ML").
                - "conditionnement": The packaging information, which is the number of units in the package (e.g., "B/30 COMP", "FL/150ML").
                - "ppa": The Public Pharmacy Price (Prix Public Algérie), which is a number, possibly with decimals.
                
                If a field is not present, return an empty string "" for its value.
                """
            },
            {
                "role": "user",
                "content": f"Here is the vignette text:\n\n---\n{text_content}\n---",
            }
        ],
        model="llama3-8b-8192",
        response_format={"type": "json_object"}, # Force JSON output
    )
    
    response_text = chat_completion.choices[0].message.content
    print("Received structured vignette data from Groq.")
    return response_text

# --- API Endpoint for Prescriptions (Unchanged) ---
@app.route('/process_prescription', methods=['POST'])
def process_prescription_endpoint():
    # This endpoint remains for processing full prescriptions
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        image_content = file.read()
        ocr_text = extract_text_with_google_vision(image_content)
        # For now, we'll just return a placeholder for the prescription part
        # You would add your full Gemini/Groq logic here for prescriptions
        return jsonify({"message": "Prescription endpoint called. Implement prescription logic here.", "text": ocr_text})
    except Exception as e:
        print(f"!!! ERROR IN PRESCRIPTION PROCESSING: {e} !!!")
        return jsonify({"error": "An internal server error occurred."}), 500


# --- NEW API ENDPOINT FOR VIGNETTES ---
@app.route('/process_vignette', methods=['POST'])
def process_vignette_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        image_content = file.read()
        
        # Step 1: Use Google Vision to get perfect text
        ocr_text = extract_text_with_google_vision(image_content)
        
        # Step 2: Use Groq AI to understand the vignette text
        structured_data_json_string = extract_vignette_data_with_groq(ocr_text)
        
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
