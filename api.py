import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
import re

# --- Initialize the Flask App ---
app = Flask(__name__)
CORS(app)

# --- NEW FUNCTION using Document Text Detection ---
def extract_structured_text_with_google_vision(image_content):
    """
    Uses Google Vision's Document Text Detection to understand blocks and paragraphs.
    Returns the full text and a list of all detected text blocks.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    
    # Use document_text_detection instead of text_detection
    response = client.document_text_detection(image=image)
    
    if response.error.message:
        raise Exception(response.error.message)
    
    return response.full_text_annotation.text, response.full_text_annotation.pages[0].blocks

def parse_medical_data_from_blocks(text, blocks):
    """
    Parses structured text by analyzing blocks of text instead of just lines.
    This is much more robust for different layouts.
    """
    structured_data = {
        "doctor_name": "Not found",
        "patient_name": "Not found",
        "prescription_date": "Not found",
        "patient_age": "Not found",
        "patient_dob": "Not found",
        "medications": []
    }

    full_text_lines = text.split('\n')

    # Find Doctor, Patient, and Dates from the full text (this part is still effective)
    for i, line in enumerate(full_text_lines):
        if line.lower().startswith("docteur ") or line.lower().startswith("dr "):
            structured_data["doctor_name"] = line
        
        date_match = re.search(r'\d{2}/\d{2}/\d{4}', line)
        if date_match:
            structured_data["prescription_date"] = date_match.group(0)

        if "Nom & Prénom :" in line:
            structured_data["patient_name"] = line.split(":")[-1].strip()
        elif line.lower().startswith("nom prénom"): # For the new table format
            structured_data["patient_name"] = line.split(":")[-1].strip()
        
        if line.lower().startswith("age :"):
            structured_data["patient_age"] = line.split(":")[-1].strip()
        if "date de naissance" in line.lower():
             structured_data["patient_dob"] = line.split(":")[-1].strip()

    # Find medications by looking for a block that looks like a table
    for block in blocks:
        block_text = ""
        for paragraph in block.paragraphs:
            for word in paragraph.words:
                word_text = "".join([symbol.text for symbol in word.symbols])
                block_text += word_text + " "
        
        # Heuristic: A medication table block often contains words like "Dosage", "QSP", "Forme"
        if "Dosage" in block_text or "QSP" in block_text or "Nom Commercial" in block_text:
            # This is likely the medication table block. Let's parse its lines.
            table_lines = block_text.strip().split('\n')
            for line in table_lines:
                # Find lines that likely contain a medication, e.g., starting with an uppercase letter
                # and containing at least one number (for dosage, etc.)
                if re.search(r'[A-Z]', line) and re.search(r'\d', line) and "Date de Naissance" not in line:
                    # This is a very simplified parser for the table format.
                    # A more complex parser would analyze columns based on coordinates.
                    parts = line.split()
                    if len(parts) > 1:
                        # Assume the first word(s) are the name and the rest are details.
                        structured_data["medications"].append({
                            "name": parts[0], 
                            "dosage_and_frequency": " ".join(parts[1:])
                        })
            break # Stop after finding the first medication table

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
        
        # Use the new, more advanced function
        full_text, blocks = extract_structured_text_with_google_vision(image_content)
        
        # Parse the structured text
        structured_data = parse_medical_data_from_blocks(full_text, blocks)
        
        return jsonify(structured_data), 200
    except Exception as e:
        import traceback
        print("!!! A SERVER ERROR OCCURRED !!!")
        print(traceback.format_exc())
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
