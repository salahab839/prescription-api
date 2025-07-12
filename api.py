import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
import re

# --- Initialize the Flask App ---
app = Flask(__name__)
CORS(app)

# --- Data Extraction Code ---
def extract_text_with_google_vision(image_content):
    """Uses Google Cloud Vision API for superior OCR."""
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(response.error.message)
    return response.text_annotations[0].description if response.text_annotations else ""

def parse_medical_data_locally(text):
    """
    Parses clean text with the most effective rules we developed.
    """
    structured_data = {
        "doctor_name": "Not found",
        "patient_name": "Not found",
        "prescription_date": "Not found",
        "patient_age": "Not found",
        "medications": []
    }
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    patient_parts = {}
    headers_to_ignore = ["Docteur", "Spécialiste", "Tél", "Nom:", "Prénom:", "Age", "ORDONNANCE", "Jdioula"]

    # --- Data Extraction Logic ---
    for i, line in enumerate(lines):
        # Doctor Extraction
        if line.lower().startswith("docteur ") or line.lower().startswith("dr "):
            structured_data["doctor_name"] = line
        
        # Date Extraction
        date_match = re.search(r'\d{2}/\d{2}/\d{4}', line)
        if date_match:
            structured_data["prescription_date"] = date_match.group(0)

        # Patient Name Extraction (handles different formats)
        if "Nom & Prénom :" in line:
            structured_data["patient_name"] = line.split(":")[-1].strip()
        elif line.lower().startswith("nom :") and ":" in line:
            patient_parts['last_name'] = line.split(":")[-1].strip()
        elif line.lower().startswith("prénom :") and ":" in line:
            patient_parts['first_name'] = line.split(":")[-1].strip()
        
        # Age Extraction
        if line.lower().startswith("age :"):
            structured_data["patient_age"] = line.split(":")[-1].strip()

    # Combine patient name parts if found separately
    if structured_data["patient_name"] == "Not found":
        if 'last_name' in patient_parts or 'first_name' in patient_parts:
            structured_data["patient_name"] = f"{patient_parts.get('first_name', '')} {patient_parts.get('last_name', '')}".strip()
    
    if structured_data["patient_name"] != "Not found":
        headers_to_ignore.extend(structured_data["patient_name"].upper().split())

    # Medication Extraction
    try:
        start_index = lines.index("ORDONNANCE") + 1
    except ValueError:
        start_index = 0

    for i in range(start_index, len(lines)):
        line = lines[i]
        # Regex to find lines that are likely medications
        med_pattern = re.compile(r'([A-Z]{3,}.*\s(\d|CP|GEL|UI|B\.O\.N))')
        
        if med_pattern.search(line) and not any(header in line for header in headers_to_ignore):
            name = re.sub(r'\s+01\s+boite\.?$', '', line).strip()
            instruction = "Not specified"
            if i + 1 < len(lines):
                next_line = lines[i+1]
                if next_line and (next_line[0].isdigit() or next_line.lower().startswith(("par", "à", "au"))):
                    instruction = next_line
            
            structured_data["medications"].append({
                "name": name, 
                "dosage_and_frequency": instruction
            })
            
    return structured_data

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
        ocr_text = extract_text_with_google_vision(image_content)
        structured_data = parse_medical_data_locally(ocr_text)
        return jsonify(structured_data), 200
    except Exception as e:
        import traceback
        print("!!! A SERVER ERROR OCCURRED !!!")
        print(traceback.format_exc())
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True)
