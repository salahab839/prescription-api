import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
import re
import spacy
from spacy.matcher import Matcher

# --- Initialize Flask App and SpaCy ---
app = Flask(__name__)
CORS(app)

# Load the French spaCy model
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    # This will run on Render during the build process
    print("Downloading spaCy model...")
    spacy.cli.download("fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

# --- API Functions ---
def extract_text_with_google_vision(image_content):
    """Uses Google Cloud Vision API for superior OCR."""
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(response.error.message)
    return response.text_annotations[0].description if response.text_annotations else ""

def parse_data_with_spacy(text):
    """Parses clean text with spaCy's Matcher for high accuracy."""
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    
    structured_data = {
        "doctor_name": "Not found",
        "patient_name": "Not found",
        "prescription_date": "Not found",
        "patient_age": "Not found",
        "patient_dob": "Not found",
        "medications": []
    }

    # Pattern for Doctor: "Docteur" or "Dr." followed by Proper Nouns
    doctor_pattern = [{"LOWER": {"in": ["docteur", "dr", "dr."]}}, {"POS": "PROPN", "OP": "+"}]
    # Pattern for Patient: "Nom" or "Prénom" followed by a colon and Proper Nouns
    patient_pattern = [{"LOWER": {"in": ["nom", "prénom"]}}, {"TEXT": ":"}, {"POS": "PROPN", "OP": "+"}]
    # Pattern for Date: A specific regex match
    date_pattern = [{"TEXT": {"REGEX": r"\d{2}/\d{2}/\d{4}"}}]

    matcher.add("DOCTOR", [doctor_pattern])
    matcher.add("PATIENT", [patient_pattern])
    matcher.add("DATE", [date_pattern])

    matches = matcher(doc)
    
    patient_parts = []
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]
        span = doc[start:end]
        if string_id == "DOCTOR":
            structured_data["doctor_name"] = span.text
        elif string_id == "PATIENT":
            patient_parts.append(span.text.split(":")[-1].strip())
        elif string_id == "DATE":
            structured_data["prescription_date"] = span.text

    if patient_parts:
        structured_data["patient_name"] = " ".join(patient_parts)

    # Use regex for medications as it's often more flexible for drug names
    lines = text.split('\n')
    for i, line in enumerate(lines):
        med_match = re.match(r'^\d+\)?\s*(.*?)(?:\.\s*(.*))?$', line, re.IGNORECASE)
        if med_match:
            name = med_match.group(1).strip()
            instruction = med_match.group(2) or ""
            if "QSP" in name: name = name.split("QSP")[0].strip()
            if not instruction.strip() and i + 1 < len(lines):
                next_line = lines[i+1]
                if not (re.match(r'^\d+\)', next_line.lstrip()) or "QSP" in next_line):
                    instruction = next_line
            if "@" not in name:
                structured_data["medications"].append({"name": name, "dosage_and_frequency": instruction.strip() if instruction else "Not specified"})

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
        structured_data = parse_data_with_spacy(ocr_text)
        return jsonify(structured_data), 200
    except Exception as e:
        import traceback
        print("!!! A SERVER ERROR OCCURRED !!!")
        print(traceback.format_exc())
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
