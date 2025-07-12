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
    Parses clean text with more flexible and intelligent rules to handle multiple formats.
    """
    structured_data = {
        "doctor_name": "Not found",
        "patient_name": "Not found",
        "prescription_date": "Not found",
        "patient_age": "Not found",
        "patient_dob": "Not found",
        "medications": []
    }
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    patient_parts = {}
    headers_to_ignore = ["Docteur", "Spécialiste", "Tél", "Nom:", "Prénom:", "Age", "ORDONNANCE", "Jdioula", "Date naissance"]

    for i, line in enumerate(lines):
        # Doctor Extraction: More generic
        if line.lower().startswith("docteur ") or line.lower().startswith("dr "):
            structured_data["doctor_name"] = line
        
        # Date Extraction
        date_match = re.search(r'\d{2}/\d{2}/\d{4}', line)
        if date_match:
            structured_data["prescription_date"] = date_match.group(0)

        # Patient Name Extraction: Handle multiple formats
        if "Nom & Prénom :" in line:
            structured_data["patient_name"] = line.split(":")[-1].strip()
        elif line.lower().startswith("nom :") and ":" in line:
            patient_parts['last_name'] = line.split(":")[-1].strip()
        elif line.lower().startswith("prénom :") and ":" in line:
            patient_parts['first_name'] = line.split(":")[-1].strip()
        
        # Age and Date of Birth Extraction
        if line.lower().startswith("age :"):
            structured_data["patient_age"] = line.split(":")[-1].strip()
        
        dob_match = re.search(r'(\d{4}-\d{2}-\d{2})\s*\((\d+\s*ans)\)', line)
        if dob_match:
            structured_data["patient_dob"] = dob_match.group(1)
            structured_data["patient_age"] = dob_match.group(2)
        elif "date naissance" in line.lower():
             structured_data["patient_dob"] = line.split(":")[-1].strip()

    # Combine patient name parts if they were found separately
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
        # Regex to find lines starting with a number, like "1.) medication name. instruction"
        med_match = re.match(r'^\d+\)?\s*(.*?)(?:\.\s*(.*))?$', line, re.IGNORECASE)
        if med_match:
            name = med_match.group(1).strip()
            instruction = med_match.group(2) or ""
            instruction = instruction.strip()

            if "QSP" in name:
                name = name.split("QSP")[0].strip()

            # If instruction is empty on the same line, check the next line
            if not instruction and i + 1 < len(lines):
                next_line = lines[i+1]
                # Ensure the next line isn't just another medication
                if not (re.match(r'^\d+\)', next_line.lstrip()) or "QSP" in next_line):
                    instruction = next_line
            
            # Final check to avoid adding footer text
            if "@" not in name and "tel:" not in name.lower() and "mob:" not in name.lower():
                 structured_data["medications"].append({
                    "name": name, 
                    "dosage_and_frequency": instruction if instruction else "Not specified"
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
