import os
import traceback
import json
import re
import pandas as pd
import uuid
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google.cloud import vision
from groq import Groq
from thefuzz import process, fuzz

# --- Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chifa_data.xlsx")
app = Flask(__name__, template_folder='templates')
CORS(app)
SESSIONS = {}

# --- Initialize Clients ---
try:
    vision_client = vision.ImageAnnotatorClient()
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    print(f"ERREUR CRITIQUE lors de l'initialisation des clients API: {e}")

# --- Helper Functions & DB Loading ---
def normalize_string(s):
    if not isinstance(s, str): return ""
    s = s.lower().replace('bte', 'b').replace('boite', 'b').replace('comprimés', 'cp').replace('comprimé', 'cp').replace('grammes', 'g').replace('gramme', 'g').replace('milligrammes', 'mg')
    s = re.sub(r'[^a-z0-9]', ' ', s); return re.sub(r'\s+', ' ', s).strip()

def build_reference_string(row, include_name=True):
    name_part = row.get('Nom Commercial', '') if include_name else ''
    dosage_part = row.get('Dosage', '')
    presentation_part = row.get('Présentation', '')
    return f"{name_part} {dosage_part} {presentation_part}".strip()

DB_SIGNATURE_MAP = {}
DB_DOSAGE_PRES_MAP = {} # NEW: For the fallback search
try:
    df = pd.read_excel(DB_PATH) 
    for index, row in df.iterrows():
        # Full signature for Stage 1
        full_signature = normalize_string(build_reference_string(row))
        DB_SIGNATURE_MAP[full_signature] = row.to_dict()
        
        # Dosage+Presentation signature for Stage 2
        dosage_pres_signature = normalize_string(build_reference_string(row, include_name=False))
        if dosage_pres_signature not in DB_DOSAGE_PRES_MAP:
            DB_DOSAGE_PRES_MAP[dosage_pres_signature] = []
        DB_DOSAGE_PRES_MAP[dosage_pres_signature].append(row.to_dict())

    print(f"Base de données chargée avec {len(DB_SIGNATURE_MAP)} médicaments.")
except Exception as e:
    print(f"ERREUR critique lors du chargement de la base de données : {e}")


def process_image_data(image_content):
    """ Main processing logic with NEW Two-Stage Verification. """
    ocr_text = vision_client.text_detection(image=vision.Image(content=image_content)).text_annotations[0].description
    
    # --- THIS IS THE CORRECTED PROMPT ---
    # It is now more explicit and includes the required "JSON" keyword for the API.
    system_prompt = """
    Vous êtes un expert en lecture de vignettes de médicaments françaises. Votre unique tâche est de retourner un objet JSON valide.
    Ne retournez que l'objet JSON, sans aucun texte supplémentaire ni formatage markdown.

    Voici les clés que vous devez utiliser : "nom", "dosage", "conditionnement", "ppa".
    """
    
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Texte de la vignette à analyser:\n---\n{ocr_text}\n---"}],
        model="llama3-8b-8192", 
        response_format={"type": "json_object"}
    )
    ai_data = json.loads(chat_completion.choices[0].message.content)

    # --- Stage 1: Attempt a full match ---
    print("--- Stage 1: Attempting Full Match ---")
    full_vignette_sig = normalize_string(f"{ai_data.get('nom','')} {ai_data.get('dosage','')} {ai_data.get('conditionnement','')}")
    best_full_match, score_full = process.extractOne(full_vignette_sig, DB_SIGNATURE_MAP.keys(), scorer=fuzz.token_set_ratio)
    print(f"Full match result: '{best_full_match}' with score {score_full}%")

    if score_full >= 85: # High confidence match
        print("Success: High confidence match found in Stage 1.")
        verified_data = DB_SIGNATURE_MAP[best_full_match]
        return {"nom": verified_data.get('Nom Commercial'), "dosage": verified_data.get('Dosage'), "conditionnement": verified_data.get('Présentation'), "ppa": ai_data.get('ppa'), "match_score": score_full, "status": "Vérifié"}

    # --- Stage 2: Fallback to smart search if Stage 1 fails ---
    print("--- Stage 2: Fallback to Smart Search ---")
    dosage_pres_sig = normalize_string(f"{ai_data.get('dosage','')} {ai_data.get('conditionnement','')}")
    best_dosage_match, score_dosage = process.extractOne(dosage_pres_sig, DB_DOSAGE_PRES_MAP.keys())
    
    if score_dosage >= 95: # Very high confidence on dosage/presentation
        print(f"Found candidate pool with dosage/presentation match: '{best_dosage_match}'")
        candidate_drugs = DB_DOSAGE_PRES_MAP[best_dosage_match]
        ocr_name = normalize_string(ai_data.get('nom', ''))
        candidate_names = {normalize_string(drug.get('Nom Commercial')): drug for drug in candidate_drugs}
        best_name_match, score_name = process.extractOne(ocr_name, candidate_names.keys())
        
        print(f"Best name match in candidate pool: '{best_name_match}' with score {score_name}%")
        if score_name >= 70: # Confidence threshold for the name within the filtered list
            print("Success: High confidence match found in Stage 2.")
            verified_data = candidate_names[best_name_match]
            final_score = int((score_dosage * 0.6) + (score_name * 0.4)) # Weighted average
            return {"nom": verified_data.get('Nom Commercial'), "dosage": verified_data.get('Dosage'), "conditionnement": verified_data.get('Présentation'), "ppa": ai_data.get('ppa'), "match_score": final_score, "status": "Vérifié (Auto-Corrigé)"}

    # If both stages fail, return the raw data
    print("Failure: No confident match found in either stage.")
    return {"nom": ai_data.get('nom'), "dosage": ai_data.get('dosage'), "conditionnement": ai_data.get('conditionnement'), "ppa": ai_data.get('ppa'), "match_score": score_full, "status": "Non Vérifié"}

# --- All API Routes (No Changes) ---
@app.route('/api/create-session', methods=['POST'])
def create_session():
    session_id = str(uuid.uuid4()); SESSIONS[session_id] = {"status": "pending", "data": None, "timestamp": time.time()}; return jsonify({"session_id": session_id})
@app.route('/phone-upload/<session_id>')
def phone_upload_page(session_id):
    return render_template('uploader.html') if session_id in SESSIONS else ("Session invalide.", 404)
@app.route('/api/upload-by-session/<session_id>', methods=['POST'])
def upload_by_session(session_id):
    if session_id not in SESSIONS: return jsonify({"error": "Session invalide"}), 404
    if 'file' not in request.files: return jsonify({"error": "Aucun fichier"}), 400
    try:
        processed_data = process_image_data(request.files['file'].read())
        SESSIONS[session_id]['status'] = 'completed'; SESSIONS[session_id]['data'] = processed_data
        return jsonify({"status": "success"})
    except Exception as e:
        SESSIONS[session_id]['status'] = 'error'; SESSIONS[session_id]['data'] = str(e)
        return jsonify({"error": "Erreur de traitement"}), 500
@app.route('/api/check-session/<session_id>')
def check_session(session_id):
    if session_id not in SESSIONS: return jsonify({"status": "error"}), 404
    session_info = SESSIONS[session_id]
    if time.time() - session_info.get("timestamp", 0) > 600:
        del SESSIONS[session_id]; return jsonify({"status": "expired"}), 410
    if session_info['status'] == 'completed':
        data_to_return = session_info['data']; del SESSIONS[session_id]
        return jsonify({"status": "completed", "data": data_to_return})
    else:
        return jsonify({"status": session_info['status']})
@app.route('/process_vignette', methods=['POST'])
def process_vignette_endpoint():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    try:
        return jsonify(process_image_data(request.files['file'].read()))
    except Exception as e:
        return jsonify({"error": f"Une erreur interne est survenue: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
