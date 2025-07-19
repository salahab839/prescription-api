import os
import traceback
import json
import re
import pandas as pd
import uuid
import time
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google.cloud import vision
from groq import Groq
from thefuzz import process, fuzz

# --- Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Use the uploaded CSV file path
DB_PATH = os.path.join(BASE_DIR, "chifa_data.xlsx - Sheet1.csv")
app = Flask(__name__, template_folder='templates')
CORS(app)
SESSIONS = {}

# --- Initialize Clients ---
try:
    vision_client = vision.ImageAnnotatorClient()
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    print(f"CRITICAL ERROR during API client initialization: {e}")
    vision_client = None
    groq_client = None

# --- Helper Functions & DB Loading ---
def normalize_string(s):
    """Cleans and standardizes a string for matching."""
    if not isinstance(s, str): return ""
    s = s.lower()
    replacements = {'bte': 'b', 'boite': 'b', 'comprimés': 'cp', 'comprimé': 'cp', 'grammes': 'g', 'gramme': 'g', 'milligrammes': 'mg'}
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def build_reference_string(row, include_name=True, include_dci=True, include_details=True):
    """Builds a standardized string from a database row for matching."""
    parts = []
    if include_name:
        parts.append(row.get('Nom Commercial', ''))
    if include_dci:
        parts.append(row.get('DCI', ''))
    if include_details:
        parts.append(row.get('Dosage', ''))
        parts.append(row.get('Présentation', ''))
    return " ".join(filter(None, parts)).strip()

# --- Database Loading ---
DB_SIGNATURE_MAP = {}
DB_NAMES_MAP = {}
df = None
try:
    df = pd.read_csv(DB_PATH)
    df = df.astype(str).fillna('') 
    
    for index, row in df.iterrows():
        row_dict = row.to_dict()
        
        full_signature = normalize_string(build_reference_string(row_dict))
        if full_signature:
            DB_SIGNATURE_MAP[full_signature] = row_dict
        
        name_signature = normalize_string(row_dict.get('Nom Commercial', ''))
        if name_signature:
            if name_signature not in DB_NAMES_MAP:
                DB_NAMES_MAP[name_signature] = []
            DB_NAMES_MAP[name_signature].append(row_dict)
            
    print(f"Database loaded successfully with {len(df)} medications.")
except FileNotFoundError:
    print(f"CRITICAL ERROR: Database file not found at {DB_PATH}")
except Exception as e:
    print(f"CRITICAL ERROR loading database: {e}")

# --- Intelligent PPA Parser ---
def parse_ppa(text):
    """Extracts and cleans the PPA value from text."""
    if not isinstance(text, str): return ""
    if '=' in text:
        text = text.split('=')[-1]
    elif ':' in text:
        text = text.split(':')[-1]
    cleaned_text = re.sub(r'[^0-9,.]', '', text).strip()
    return cleaned_text.replace(',', '.')

# --- Image Processing Logic (REVISED) ---
def process_image_data(image_content):
    """
    Processes an image to extract medication data and match it against the database
    using a more flexible, multi-stage approach.
    """
    if not all([vision_client, groq_client, df is not None]):
        return {"status": "Échec: Service non initialisé"}

    # 1. OCR with Google Vision
    try:
        response = vision_client.text_detection(image=vision.Image(content=image_content))
        if not response.text_annotations:
            raise Exception("No text detected by Vision API")
        ocr_text = response.text_annotations[0].description
    except Exception as e:
        print(f"Vision API Error: {e}")
        return {"status": "Échec de la lecture (OCR)"}

    # 2. Structured Extraction with Groq LLM
    system_prompt = """
    You are an expert at reading French medication labels (vignettes). Your only task is to extract the information into a valid JSON object.
    Only return the JSON object, with no additional text, comments, or markdown formatting.
    Ensure all values are valid strings.
    The JSON must contain these keys: "nom", "dci", "dosage", "conditionnement", "ppa".
    Example of a valid JSON response:
    { "nom": "DOLIPRANE", "dci": "PARACETAMOL", "dosage": "1000 MG", "conditionnement": "BTE/8", "ppa": "1,95" }
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text to analyze:\n---\n{ocr_text}\n---"}
            ],
            model="llama3-8b-8192",
            response_format={"type": "json_object"}
        )
        ai_data = json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"Groq API Error: {e}")
        return {"status": "Échec de l'analyse (IA)"}

    # 3. Data Cleaning and Signature Creation
    ai_data['ppa'] = parse_ppa(ai_data.get('ppa', ''))
    ocr_name_sig = normalize_string(ai_data.get('nom', ''))
    ocr_details_sig = normalize_string(f"{ai_data.get('dci','')} {ai_data.get('dosage','')} {ai_data.get('conditionnement','')}")
    ocr_full_sig = normalize_string(f"{ocr_name_sig} {ocr_details_sig}")

    def get_verified_response(db_row, score, status="Vérifié"):
        """Formats the final response object."""
        return {
            "nom": db_row.get('Nom Commercial'), "dci": db_row.get('DCI'),
            "dosage": db_row.get('Dosage'), "conditionnement": db_row.get('Présentation'),
            "ppa": ai_data.get('ppa'), "match_score": score, "status": status,
            "posologie_qte_prise": "1", "posologie_unite": db_row.get('Forme', ''),
            "posologie_frequence": "3", "posologie_periode": "par jour"
        }

    # 4. Flexible Matching Logic
    
    # Attempt 1: High-confidence full signature match (for clear scans)
    best_full_match, score_full = process.extractOne(ocr_full_sig, DB_SIGNATURE_MAP.keys(), scorer=fuzz.token_set_ratio)
    if score_full >= 90:
        return get_verified_response(DB_SIGNATURE_MAP[best_full_match], score_full, status="Vérifié (Haute Confiance)")

    # Attempt 2: Name-first match (more forgiving)
    best_name_match, score_name = process.extractOne(ocr_name_sig, DB_NAMES_MAP.keys(), scorer=fuzz.token_set_ratio)
    
    # Lowered threshold for name matching
    if score_name >= 75:
        candidate_drugs = DB_NAMES_MAP[best_name_match]
        if len(candidate_drugs) == 1:
            return get_verified_response(candidate_drugs[0], score_name, status="Vérifié (Nom unique)")

        candidate_details = {
            normalize_string(build_reference_string(drug, include_name=False)): drug 
            for drug in candidate_drugs
        }
        
        best_candidate_details, score_candidate_details = process.extractOne(
            ocr_details_sig, candidate_details.keys(), scorer=fuzz.WRatio
        )
        
        # Lowered threshold for details matching
        if score_candidate_details >= 65:
            final_score = int((score_name * 0.6) + (score_candidate_details * 0.4))
            return get_verified_response(candidate_details[best_candidate_details], final_score, status="Auto-Corrigé")
        else:
            # Name was found, but details don't match well enough.
            return {"status": "Échec: Ambiguïté", "details": f"Nom '{best_name_match}' trouvé, mais DCI/Dosage incertain. Score détails: {score_candidate_details}"}

    # If all attempts fail, return a descriptive failure.
    return {"status": "Échec de la reconnaissance", "details": f"Aucun nom de médicament correspondant trouvé. Meilleur score: {score_name}"}


# --- API Routes (unchanged) ---
@app.route('/api/create-session', methods=['POST'])
def create_session():
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {"status": "pending", "medications": [], "timestamp": time.time()}
    return jsonify({"session_id": session_id})

@app.route('/phone-upload/<session_id>')
def phone_upload_page(session_id):
    return render_template('uploader.html') if session_id in SESSIONS else ("Session invalide.", 404)

@app.route('/api/upload-by-session/<session_id>', methods=['POST'])
def upload_by_session(session_id):
    if session_id not in SESSIONS or SESSIONS[session_id]['status'] != 'pending':
        return jsonify({"error": "Session invalide ou terminée"}), 404
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier"}), 400
    try:
        image_content = request.files['file'].read()
        processed_data = process_image_data(image_content)
        if "Échec" not in processed_data.get("status", ""):
            image_base64 = base64.b64encode(image_content).decode('utf-8')
            SESSIONS[session_id]['medications'].append({"data": processed_data, "image_base64": image_base64})
            return jsonify({"status": "success", "message": "Médicament ajouté."})
        else:
            return jsonify({"status": "failure", "message": processed_data.get("details", "Vignette illisible")})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Erreur de traitement: {e}"}), 500

@app.route('/api/finish-session/<session_id>', methods=['POST'])
def finish_session(session_id):
    if session_id in SESSIONS:
        SESSIONS[session_id]['status'] = 'finished'
        return jsonify({"status": "success"})
    return jsonify({"error": "Session invalide"}), 404

@app.route('/api/check-session/<session_id>')
def check_session(session_id):
    if session_id not in SESSIONS:
        return jsonify({"status": "invalid"}), 404
    session_info = SESSIONS[session_id]
    if time.time() - session_info.get("timestamp", 0) > 600: # 10 minutes
        if session_id in SESSIONS:
            del SESSIONS[session_id]
        return jsonify({"status": "expired"}), 410
    return jsonify({"status": session_info['status'], "medications": session_info['medications']})

@app.route('/process_vignette', methods=['POST'])
def process_vignette_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    try:
        image_content = request.files['file'].read()
        processed_data = process_image_data(image_content)
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        return jsonify({"data": processed_data, "image_base64": image_base64})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Une erreur interne est survenue: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
