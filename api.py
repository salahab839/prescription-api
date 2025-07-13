import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
import re
import traceback

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
    Parses clean text with the most advanced and flexible rules based on all provided examples.
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
    
    # --- Data Extraction Logic ---
    patient_name_parts = {}
    
    # --- Phase 1: Extract Header Information (Patient, Doctor, Dates) ---
    for i, line in enumerate(lines):
        # Doctor Name (handles various formats and titles)
        if re.search(r'^(Dr|Docteur)[\.\s]', line, re.IGNORECASE):
            structured_data["doctor_name"] = line

        # Date
        date_match = re.search(r'\d{2}/\d{2}/\d{4}', line)
        if date_match:
            structured_data["prescription_date"] = date_match.group(0)

        # Patient Name (handles same-line and next-line formats)
        if re.search(r'Nom\s*&\s*Prénom\s*:', line, re.IGNORECASE):
            structured_data["patient_name"] = line.split(":")[-1].strip()
        elif re.search(r'^Nom\s*:', line, re.IGNORECASE):
            name_part = line.split(":")[-1].strip()
            if name_part: patient_name_parts['last'] = name_part
            elif i + 1 < len(lines): patient_name_parts['last'] = lines[i+1]
        elif re.search(r'^Prénom\s*:', line, re.IGNORECASE):
            name_part = line.split(":")[-1].strip()
            if name_part: patient_name_parts['first'] = name_part
            elif i + 1 < len(lines): patient_name_parts['first'] = lines[i+1]
        
        # Age and DOB
        if re.search(r'^Age\s*:', line, re.IGNORECASE):
            structured_data["patient_age"] = line.split(":")[-1].strip()
        if re.search(r'Date naissance\s*:', line, re.IGNORECASE):
            dob_part = line.split(":")[-1].strip()
            # Also capture age if it's in parentheses on the same line
            age_in_dob = re.search(r'\(.*\)', dob_part)
            if age_in_dob:
                structured_data["patient_age"] = age_in_dob.group(0)
            structured_data["patient_dob"] = dob_part.split('(')[0].strip()

    # Assemble Patient Name
    if structured_data["patient_name"] == "Not found":
        if 'first' in patient_name_parts or 'last' in patient_name_parts:
            structured_data["patient_name"] = f"{patient_name_parts.get('first', '')} {patient_name_parts.get('last', '')}".strip()

    # --- Phase 2: Medication Extraction ---
    try:
        start_index = lines.index("ORDONNANCE") + 1
    except ValueError:
        start_index = 0 # If "ORDONNANCE" isn't found, try to find meds anyway

    medication_lines = lines[start_index:]
    
    # List of known non-medication keywords to ignore
    ignore_keywords = ["docteur", "spécialiste", "tél", "nom", "prénom", "age", "ordonnance", "jdioula", "date", "qsp", "servi", "scanned with", "page"]
    if structured_data["patient_name"] != "Not found":
        ignore_keywords.extend(structured_data["patient_name"].lower().split())

    for i, line in enumerate(medication_lines):
        # A medication line usually starts with a number or an uppercase word, and contains letters.
        if (line and (line[0].isdigit() or line[0].isupper()) and re.search('[a-zA-Z]', line)):
            
            # Check if the line contains any words from our ignore list
            if any(keyword in line.lower() for keyword in ignore_keywords):
                continue

            # This is likely a medication. Let's find its instruction.
            instruction = ""
            if i + 1 < len(medication_lines):
                next_line = medication_lines[i+1]
                # An instruction line often starts with a number (dosage) or a lowercase word (preposition)
                # and is not another medication line itself.
                if next_line and (next_line[0].isdigit() or next_line[0].islower()) and not re.search(r'^\d+\)', next_line):
                    instruction = next_line
            
            # Clean up the name
            name = re.sub(r'^\d+\)?\s*\.?\s*', '', line) # Remove leading numbers like "1)"
            name = re.sub(r'\s+01\s+boite\.?$', '', name, flags=re.IGNORECASE).strip()

            # Add to list if it seems like a valid medication (avoids adding random numbers/short codes)
            if len(name) > 3:
                structured_data["medications"].append({
                    "name": name, 
                    "dosage_and_frequency": instruction if instruction else "Not specified"
                })

    return structured_data

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
        structured_data = parse_medical_data_locally(ocr_text)
        return jsonify(structured_data), 200
    except Exception as e:
        print("!!! A SERVER ERROR OCCURRED !!!")
        print(traceback.format_exc())
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
