import os
import traceback
import json
import re
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
from groq import Groq
from thefuzz import process, fuzz

# --- Flask & Clients ---
app = Flask(__name__)
CORS(app)
vision_client = vision.ImageAnnotatorClient()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- Load Medication Database ---
chifa_df = pd.read_excel("chifa_data.xlsx")

def normalize_string(s):
    if not isinstance(s, str):
        return ""
    s = s.lower().replace('bte', 'b').replace('boite', 'b')
    s = s.replace('comprimés', 'cp').replace('comprimé', 'cp')
    s = s.replace('grammes', 'g').replace('gramme', 'g').replace('milligrammes', 'mg')
    s = re.sub(r'(\d)[ ]+([a-z%]+)', r'\1\2', s)
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def build_reference_string(row):
    return f"{row['Nom Commercial']} {row['Dosage']} {row['Présentation']}"

def match_vignette_to_database(nom, dosage, conditionnement, db_df, threshold=85):
    input_string = f"{nom} {dosage} {conditionnement}"
    input_norm = normalize_string(input_string)

    db_df['ref'] = db_df.apply(build_reference_string, axis=1)
    db_df['ref_norm'] = db_df['ref'].apply(normalize_string)

    best_match, score = process.extractOne(input_norm, db_df['ref_norm'].tolist(), scorer=fuzz.token_set_ratio)

    if best_match and score >= threshold:
        matched_row = db_df[db_df['ref_norm'] == best_match].iloc[0]
        return {
            "nom": matched_row['Nom Commercial'],
            "dosage": matched_row['Dosage'],
            "conditionnement": matched_row['Présentation'],
            "score": score
        }
    else:
        return {
            "nom": nom,
            "dosage": dosage,
            "conditionnement": conditionnement,
            "score": score
        }

# --- OCR ---
def extract_text_with_google_vision(image_content):
    image = vision.Image(content=image_content)
    response = vision_client.text_detection(image=image)
    if response.error.message:
        raise Exception(response.error.message)
    return response.text_annotations[0].description if response.text_annotations else ""

# --- Groq Extraction ---
def extract_vignette_data_with_groq(text_content):
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert at reading French medication vignettes (the small price stickers). From the user's text, extract the information into a valid JSON object and nothing else.

                The JSON object must have these exact keys:
                - \"nom\": The commercial name ONLY (no dosage)
                - \"dosage\": e.g., \"200 mg\"
                - \"conditionnement\": format \"B/XX\"
                - \"ppa\": numeric value or empty string
                """
            },
            {
                "role": "user",
                "content": f"Here is the vignette text:\n\n---\n{text_content}\n---",
            }
        ],
        model="llama3-8b-8192",
        response_format={"type": "json_object"},
    )
    return chat_completion.choices[0].message.content

# --- API Endpoints ---
@app.route('/process_vignette', methods=['POST'])
def process_vignette_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        image_content = file.read()
        ocr_text = extract_text_with_google_vision(image_content)
        structured_data_json_string = extract_vignette_data_with_groq(ocr_text)
        structured_data = json.loads(structured_data_json_string)

        # Fuzzy match with DB
        verified = match_vignette_to_database(
            structured_data.get("nom", ""),
            structured_data.get("dosage", ""),
            structured_data.get("conditionnement", ""),
            chifa_df
        )

        # Update response with verified data
        structured_data["nom"] = verified["nom"]
        structured_data["dosage"] = verified["dosage"]
        structured_data["conditionnement"] = verified["conditionnement"]
        structured_data["match_score"] = verified["score"]

        return jsonify(structured_data)

    except Exception as e:
        print("!!! SERVER ERROR !!!")
        print(traceback.format_exc())
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

# --- Placeholder ---
@app.route('/process_prescription', methods=['POST'])
def process_prescription_endpoint():
    return jsonify({"message": "Prescription endpoint not implemented."})

if __name__ == '__main__':
    app.run(debug=True)
