import os
from flask import Flask, request, jsonify
from google.cloud import vision
import re

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Your Working Data Extraction Code ---
def extract_text_with_google_vision(image_content):
    """Uses Google Cloud Vision API on image content."""
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(response.error.message)
    return response.text_annotations[0].description if response.text_annotations else ""

def parse_medical_data_locally(text):
    """
    Parses clean text with high-accuracy rules tailored to the prescription layout.
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

    for i, line in enumerate(lines):
        # Doctor and Date Extraction
        if "Docteur YOUCEF Samir" in line:
            structured_data["doctor_name"] = "Docteur YOUCEF Samir"
        
        date_match = re.search(r'\d{2}/\d{2}/\d{4}', line)
        if date_match:
            structured_data["prescription_date"] = date_match.group(0)

        # Patient Name Extraction (looks at the next line)
        if "Nom:" in line and i + 1 < len(lines):
            patient_parts['last_name'] = lines[i+1]
        if "Prénom:" in line and i + 1 < len(lines):
            patient_parts['first_name'] = lines[i+1]
        
        # Age Extraction
        if "Age" in line:
            if i > 0 and re.search(r'\d+', lines[i-1]):
                structured_data["patient_age"] = lines[i-1].strip()
            elif i + 1 < len(lines) and re.search(r'\d+', lines[i+1]):
                structured_data["patient_age"] = lines[i+1].strip()

    # Combine patient name parts and add to ignore list
    if 'last_name' in patient_parts or 'first_name' in patient_parts:
        structured_data["patient_name"] = f"{patient_parts.get('last_name', '')} {patient_parts.get('first_name', '')}".strip()
        headers_to_ignore.extend(structured_data["patient_name"].split())
    
    # Medication Extraction
    for i, line in enumerate(lines):
        med_pattern = re.compile(r'([A-Z]{3,}.*\s(\d|CP|GEL|UI|B\.O\.N))')
        
        if med_pattern.search(line) and not any(header in line for header in headers_to_ignore):
            name = re.sub(r'\s+01\s+boite\.?$', '', line).strip()
            instruction = "Not specified"
            if i + 1 < len(lines):
                next_line = lines[i+1]
                if next_line and (next_line[0].isdigit() or next_line.lower().startswith(("par", "à", "au"))):
                    instruction = next_line
            structured_data["medications"].append({"name": name, "dosage_and_frequency": instruction})
            
    return structured_data

# --- Define the API Endpoint ---
@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    """This function is called when a file is uploaded to the server."""
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
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # This part is for local testing only
    app.run(debug=True)