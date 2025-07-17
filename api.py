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
DB_DOSAGE_PRES_MAP = {}
try:
    df = pd.read_excel(DB_PATH) 
    for index, row in df.iterrows():
        full_signature = normalize_string(build_reference_string(row))
        DB_SIGNATURE_MAP[full_signature] = row.to_dict()
        dosage_pres_signature = normalize_string(build_reference_string(row, include_name=False))
        if dosage_pres_signature not in DB_DOSAGE_PRES_MAP:
            DB_DOSAGE_PRES_MAP[dosage_pres_signature] = []
        DB_DOSAGE_PRES_MAP[dosage_pres_signature].append(row.to_dict())
    print(f"Base de données chargée avec {len(DB_SIGNATURE_MAP)} médicaments.")
except Exception as e:
    print(f"ERREUR critique lors du chargement de la base de données : {e}")


def process_image_data(image_content):
    """ Main processing logic with robust error handling and adjusted thresholds. """
    print("[CHECKPOINT A] Début du traitement de l'image.")
    
    response = vision_client.text_detection(image=vision.Image(content=image_content))
    if response.error.message: raise Exception(f"Erreur de l'API Google Vision: {response.error.message}")
    if not response.text_annotations: raise Exception("Aucun texte n'a été détecté dans l'image.")
    ocr_text = response.text_annotations[0].description
    print("[CHECKPOINT B] Succès. Texte extrait.")

    system_prompt = """
    Vous êtes un expert en lecture de vignettes de médicaments françaises. Votre unique tâche est de retourner un objet JSON valide.
    Ne retournez que l'objet JSON, sans aucun texte supplémentaire ni formatage markdown.
    Voici les clés que vous devez utiliser : "nom", "dosage", "conditionnement", "ppa".
    """
    chat_completion = groq_client.chat.completions.create(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Texte à analyser:\n---\n{ocr_text}\n---"}], model="llama3-8b-8192", response_format={"type": "json_object"})
    ai_data = json.loads(chat_completion.choices[0].message.content)
    print("[CHECKPOINT C] Succès. Données de l'IA reçues.")

    def get_verified_response(db_row, score, status="Vérifié"):
        return {"nom": db_row.get('Nom Commercial'), "dosage": db_row.get('Dosage'), "conditionnement": db_row.get('Présentation'), "ppa": ai_data.get('ppa'), "match_score": score, "status": status}

    # --- Stage 1: Attempt a full match ---
    full_vignette_sig = normalize_string(f"{ai_data.get('nom','')} {ai_data.get('dosage','')} {ai_data.get('conditionnement','')}")
    best_full_match, score_full = process.extractOne(full_vignette_sig, DB_SIGNATURE_MAP.keys(), scorer=fuzz.token_set_ratio)
    print(f"[DEBUG] Score Stage 1: {score_full}%")

    # --- ADJUSTED THRESHOLD ---
    if score_full >= 80:
        verified_data_row = DB_SIGNATURE_MAP[best_full_match]
        return get_verified_response(verified_data_row, score_full)

    # --- Stage 2: Fallback to smart search ---
    dosage_pres_sig = normalize_string(f"{ai_data.get('dosage','')} {ai_data.get('conditionnement','')}")
    best_dosage_match, score_dosage = process.extractOne(dosage_pres_sig, DB_DOSAGE_PRES_MAP.keys())
    print(f"[DEBUG] Score Stage 2 (Dosage/Pres): {score_dosage}%")
    
    if score_dosage >= 90: # Adjusted threshold
        candidate_drugs = DB_DOSAGE_PRES_MAP[best_dosage_match]
        ocr_name = normalize_string(ai_data.get('nom', ''))
        candidate_names = {normalize_string(drug.get('Nom Commercial')): drug for drug in candidate_drugs}
        best_name_match, score_name = process.extractOne(ocr_name, candidate_names.keys())
        print(f"[DEBUG] Score Stage 2 (Name): {score_name}%")
        
        # --- ADJUSTED THRESHOLD ---
        if score_name >= 65:
            verified_data_row = candidate_names[best_name_match]
            final_score = int((score_dosage * 0.6) + (score_name * 0.4))
            return get_verified_response(verified_data_row, final_score, status="Auto-Corrigé")

    print("[CHECKPOINT F] Aucune correspondance trouvée.")
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
        print(f"ERREUR DANS L'ENDPOINT /api/upload-by-session: {e}")
        traceback.print_exc()
        SESSIONS[session_id]['status'] = 'error'; SESSIONS[session_id]['data'] = str(e)
        return jsonify({"error": f"Erreur de traitement: {e}"}), 500
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
        print(f"ERREUR DANS L'ENDPOINT /process_vignette: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Une erreur interne est survenue: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
