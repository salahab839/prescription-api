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
# RESTORED: Reading the .xlsx file directly, as in the original code.
DB_PATH = os.path.join(BASE_DIR, "chifa_data.xlsx")

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
    if not isinstance(s, str): return ""
    s = s.lower()
    replacements = {'bte': 'b', 'boite': 'b', 'comprimés': 'cp', 'comprimé': 'cp', 'grammes': 'g', 'gramme': 'g', 'milligrammes': 'mg'}
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def build_reference_string(row, include_name=True, include_details=True):
    """Builds a standardized string from a database row for matching."""
    parts = []
    if include_name:
        parts.append(row.get('Nom Commercial', ''))
    if include_details:
        parts.append(row.get('Dosage', ''))
        parts.append(row.get('Présentation', ''))
    return " ".join(filter(None, parts)).strip()

# --- Database Loading (Restored Original Logic) ---
DB_SIGNATURE_MAP = {}
DB_DOSAGE_PRES_MAP = {}
DB_NAMES_MAP = {}
df = None
try:
    # RESTORED: Using pd.read_excel to correctly handle the .xlsx file.
    df = pd.read_excel(DB_PATH).astype(str).fillna('')
    
    for index, row in df.iterrows():
        row_dict = row.to_dict()
        # RESTORED: Rebuilding all three original maps for multi-tiered matching.
        full_signature = normalize_string(build_reference_string(row))
        DB_SIGNATURE_MAP[full_signature] = row_dict

        dosage_pres_signature = normalize_string(build_reference_string(row, include_name=False))
        if dosage_pres_signature not in DB_DOSAGE_PRES_MAP:
            DB_DOSAGE_PRES_MAP[dosage_pres_signature] = []
        DB_DOSAGE_PRES_MAP[dosage_pres_signature].append(row_dict)

        name_signature = normalize_string(build_reference_string(row, include_details=False))
        if name_signature not in DB_NAMES_MAP:
            DB_NAMES_MAP[name_signature] = []
        DB_NAMES_MAP[name_signature].append(row_dict)
            
    print(f"Database loaded successfully with {len(df)} medications.")
except FileNotFoundError:
    print(f"CRITICAL ERROR: Database file not found at {DB_PATH}")
except Exception as e:
    print(f"CRITICAL ERROR loading database: {e}")
    traceback.print_exc()

# --- Image Processing Logic (Restored & Enhanced) ---
def process_image_data(image_content):
    if not all([vision_client, groq_client, df is not None]):
        return {"status": "Échec: Service non initialisé"}

    # 1. OCR
    try:
        response = vision_client.text_detection(image=vision.Image(content=image_content))
        if not response.text_annotations:
            return {"status": "Échec OCR", "details": "Aucun texte détecté sur l'image."}
        ocr_text = response.text_annotations[0].description
        print(f"--- OCR Text ---\n{ocr_text}\n-----------------")
    except Exception as e:
        print(f"Google Vision API Error: {e}")
        return {"status": "Échec OCR", "details": str(e)}

    # 2. AI Extraction (Simplified Prompt)
    try:
        system_prompt = """
        You are an expert at reading French medication labels (vignettes). Your task is to extract information into a valid JSON object.
        The JSON must contain these keys: "nom", "dosage", "conditionnement".
        Only return the JSON object.
        Example: { "nom": "DOLIPRANE", "dosage": "1000 MG", "conditionnement": "BTE/8" }
        """
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Texte à analyser:\n---\n{ocr_text}\n---"}],
            model="llama3-8b-8192", response_format={"type": "json_object"}
        )
        ai_data = json.loads(chat_completion.choices[0].message.content)
        print(f"--- AI Data ---\n{ai_data}\n---------------")
    except Exception as e:
        print(f"Groq AI extraction failed: {e}. This scan will likely fail.")
        return {"status": "Échec de l'analyse IA", "details": str(e)}

    # 3. Response Formatting
    def get_response(db_row, score, status="Vérifié"):
        return {
            "nom": db_row.get('Nom Commercial'), "dci": db_row.get('DCI'),
            "dosage": db_row.get('Dosage'), "conditionnement": db_row.get('Présentation'),
            "ppa": db_row.get('PPA'),
            "match_score": score, "status": status,
            "posologie_qte_prise": "1", "posologie_unite": db_row.get('Forme', ''),
            "posologie_frequence": "3", "posologie_periode": "par jour"
        }

    # 4. Restored Multi-Tiered Matching Logic
    ocr_full_sig = normalize_string(f"{ai_data.get('nom','')} {ai_data.get('dosage','')} {ai_data.get('conditionnement','')}")
    ocr_name_sig = normalize_string(ai_data.get('nom',''))
    ocr_details_sig = normalize_string(f"{ai_data.get('dosage','')} {ai_data.get('conditionnement','')}")

    # Tier 1: High-confidence full signature match
    best_full_match, score_full = process.extractOne(ocr_full_sig, DB_SIGNATURE_MAP.keys(), scorer=fuzz.token_set_ratio)
    if score_full >= 85:
        print(f"Match found on Tier 1 (Full Signature): '{best_full_match}' with score {score_full}")
        return get_response(DB_SIGNATURE_MAP[best_full_match], score_full)

    # Tier 2: High-confidence name match, then disambiguate details
    best_name_match, score_name = process.extractOne(ocr_name_sig, DB_NAMES_MAP.keys(), scorer=fuzz.token_set_ratio)
    if score_name >= 90:
        candidate_drugs = DB_NAMES_MAP[best_name_match]
        if len(candidate_drugs) == 1:
            print(f"Match found on Tier 2 (Unique Name): '{best_name_match}' with score {score_name}")
            return get_response(candidate_drugs[0], int((score_name * 0.8) + 20), status="Auto-Corrigé")
        
        candidate_details_map = {normalize_string(build_reference_string(drug, include_name=False)): drug for drug in candidate_drugs}
        best_candidate_details, score_candidate_details = process.extractOne(ocr_details_sig, candidate_details_map.keys(), scorer=fuzz.WRatio)
        if score_candidate_details >= 75:
            print(f"Match found on Tier 2 (Name + Details): '{best_name_match}' + '{best_candidate_details}'")
            final_score = int((score_name * 0.6) + (score_candidate_details * 0.4))
            return get_response(candidate_details_map[best_candidate_details], final_score, status="Auto-Corrigé")

    # Tier 3: High-confidence details match, then guess name
    best_details_match, score_details = process.extractOne(ocr_details_sig, DB_DOSAGE_PRES_MAP.keys(), scorer=fuzz.token_set_ratio)
    if score_details >= 95:
        candidate_drugs = DB_DOSAGE_PRES_MAP[best_details_match]
        candidate_names_map = {normalize_string(build_reference_string(drug, include_details=False)): drug for drug in candidate_drugs}
        best_candidate_name, score_candidate_name = process.extractOne(ocr_name_sig, candidate_names_map.keys())
        if score_candidate_name >= 60:
            print(f"Match found on Tier 3 (Details + Name): '{best_details_match}' + '{best_candidate_name}'")
            final_score = int((score_details * 0.7) + (score_candidate_name * 0.3))
            return get_response(candidate_names_map[best_candidate_name], final_score, status="Auto-Corrigé")

    # If all tiers fail
    print("All matching tiers failed.")
    return {"status": "Échec de la reconnaissance", "details": "Aucune correspondance fiable trouvée."}


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
    if time.time() - session_info.get("timestamp", 0) > 600:
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
