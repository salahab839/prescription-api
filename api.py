import os
import traceback
import json
import re
import pandas as pd
import uuid
import time
import base64 # Nécessaire pour encoder les images
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

def build_reference_string(row, include_name=True, include_details=True):
    name_part = row.get('Nom Commercial', '') if include_name else ''; dosage_part = row.get('Dosage', '') if include_details else ''; presentation_part = row.get('Présentation', '') if include_details else ''; return f"{name_part} {dosage_part} {presentation_part}".strip()

DB_SIGNATURE_MAP = {}; DB_DOSAGE_PRES_MAP = {}; DB_NAMES_MAP = {}
try:
    df = pd.read_excel(DB_PATH) 
    for index, row in df.iterrows():
        row_dict = row.to_dict(); full_signature = normalize_string(build_reference_string(row)); DB_SIGNATURE_MAP[full_signature] = row_dict
        dosage_pres_signature = normalize_string(build_reference_string(row, include_name=False))
        if dosage_pres_signature not in DB_DOSAGE_PRES_MAP: DB_DOSAGE_PRES_MAP[dosage_pres_signature] = []
        DB_DOSAGE_PRES_MAP[dosage_pres_signature].append(row_dict)
        name_signature = normalize_string(build_reference_string(row, include_details=False))
        if name_signature not in DB_NAMES_MAP: DB_NAMES_MAP[name_signature] = []
        DB_NAMES_MAP[name_signature].append(row_dict)
    print(f"Base de données chargée avec {len(DB_SIGNATURE_MAP)} médicaments.")
except Exception as e:
    print(f"ERREUR critique lors du chargement de la base de données : {e}")

# --- Image Processing Logic ---
def process_image_data(image_content):
    response = vision_client.text_detection(image=vision.Image(content=image_content))
    if not response.text_annotations: raise Exception("Aucun texte détecté")
    ocr_text = response.text_annotations[0].description
    
    system_prompt = """
    Vous êtes un expert en lecture de vignettes de médicaments françaises. Votre unique tâche est de retourner un objet JSON valide.
    Ne retournez que l'objet JSON, sans aucun texte supplémentaire, commentaire, ou formatage markdown.
    Assurez-vous que toutes les valeurs sont des chaînes de caractères (strings) valides entre guillemets.
    Exemple de format JSON valide :
    { "nom": "DOLIPRANE", "dosage": "1000 MG", "conditionnement": "BTE/8", "ppa": "195.00" }
    Utilisez les clés suivantes : "nom", "dosage", "conditionnement", "ppa".
    """
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Texte à analyser:\n---\n{ocr_text}\n---"}],
        model="llama3-8b-8192",
        response_format={"type": "json_object"}
    )
    ai_data = json.loads(chat_completion.choices[0].message.content)
    
    ppa_raw = str(ai_data.get('ppa', '')); ppa_cleaned = re.sub(r'[^0-9.]', '', ppa_raw); ai_data['ppa'] = ppa_cleaned.replace(',', '.')

    def get_verified_response(db_row, score, status="Vérifié"):
        return {"nom": db_row.get('Nom Commercial'), "dosage": db_row.get('Dosage'), "conditionnement": db_row.get('Présentation'), "ppa": ai_data.get('ppa'), "match_score": score, "status": status}

    ocr_full_sig = normalize_string(f"{ai_data.get('nom','')} {ai_data.get('dosage','')} {ai_data.get('conditionnement','')}")
    ocr_name_sig = normalize_string(ai_data.get('nom',''))
    ocr_details_sig = normalize_string(f"{ai_data.get('dosage','')} {ai_data.get('conditionnement','')}")

    best_full_match, score_full = process.extractOne(ocr_full_sig, DB_SIGNATURE_MAP.keys(), scorer=fuzz.token_set_ratio)
    if score_full >= 85: return get_verified_response(DB_SIGNATURE_MAP[best_full_match], score_full)

    best_name_match, score_name = process.extractOne(ocr_name_sig, DB_NAMES_MAP.keys(), scorer=fuzz.token_set_ratio)
    if score_name >= 90:
        candidate_drugs = DB_NAMES_MAP[best_name_match]
        if len(candidate_drugs) == 1:
            return get_verified_response(candidate_drugs[0], int((score_name * 0.8) + 20), status="Auto-Corrigé")
        candidate_details = {normalize_string(build_reference_string(drug, include_name=False)): drug for drug in candidate_drugs}
        best_candidate_details, score_candidate_details = process.extractOne(ocr_details_sig, candidate_details.keys(), scorer=fuzz.WRatio)
        if score_candidate_details >= 75:
            return get_verified_response(candidate_details[best_candidate_details], int((score_name * 0.6) + (score_candidate_details * 0.4)), status="Auto-Corrigé")

    best_details_match, score_details = process.extractOne(ocr_details_sig, DB_DOSAGE_PRES_MAP.keys(), scorer=fuzz.token_set_ratio)
    if score_details >= 95:
        candidate_drugs = DB_DOSAGE_PRES_MAP[best_details_match]
        candidate_names = {normalize_string(build_reference_string(drug, include_details=False)): drug for drug in candidate_drugs}
        best_candidate_name, score_candidate_name = process.extractOne(ocr_name_sig, candidate_names.keys())
        if score_candidate_name >= 60:
            return get_verified_response(candidate_names[best_candidate_name], int((score_details * 0.7) + (score_candidate_name * 0.3)), status="Auto-Corrigé")

    return {"nom": ai_data.get('nom'), "dosage": ai_data.get('dosage'), "conditionnement": ai_data.get('conditionnement'), "ppa": ai_data.get('ppa'), "match_score": score_full, "status": "Non Vérifié"}

# --- API Routes ---
@app.route('/api/create-session', methods=['POST'])
def create_session():
    session_id = str(uuid.uuid4()); SESSIONS[session_id] = {"status": "pending", "medications": [], "timestamp": time.time()}; return jsonify({"session_id": session_id})

@app.route('/phone-upload/<session_id>')
def phone_upload_page(session_id):
    return render_template('uploader.html') if session_id in SESSIONS else ("Session invalide.", 404)

@app.route('/api/upload-by-session/<session_id>', methods=['POST'])
def upload_by_session(session_id):
    if session_id not in SESSIONS or SESSIONS[session_id]['status'] != 'pending':
        return jsonify({"error": "Session invalide ou terminée"}), 404
    if 'file' not in request.files: return jsonify({"error": "Aucun fichier"}), 400
    try:
        image_content = request.files['file'].read()
        processed_data = process_image_data(image_content)
        # Encode the image in Base64 to send it to the desktop app
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        # Store both data and image
        SESSIONS[session_id]['medications'].append({"data": processed_data, "image_base64": image_base64})
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": f"Erreur de traitement: {e}"}), 500

@app.route('/api/finish-session/<session_id>', methods=['POST'])
def finish_session(session_id):
    if session_id in SESSIONS: SESSIONS[session_id]['status'] = 'finished'; return jsonify({"status": "success"})
    return jsonify({"error": "Session invalide"}), 404

@app.route('/api/check-session/<session_id>')
def check_session(session_id):
    if session_id not in SESSIONS: return jsonify({"status": "invalid"}), 404
    session_info = SESSIONS[session_id]
    if time.time() - session_info.get("timestamp", 0) > 600:
        if session_id in SESSIONS: del SESSIONS[session_id]
        return jsonify({"status": "expired"}), 410
    return jsonify({"status": session_info['status']})

@app.route('/api/get-session-data/<session_id>')
def get_session_data(session_id):
    if session_id not in SESSIONS: return jsonify({"error": "Session invalide"}), 404
    session_data = SESSIONS.pop(session_id, {"medications": []})
    return jsonify({"medications": session_data['medications']})

@app.route('/process_vignette', methods=['POST'])
def process_vignette_endpoint():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    try:
        image_content = request.files['file'].read()
        processed_data = process_image_data(image_content)
        # Also return the image for the verification window
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        return jsonify({"data": processed_data, "image_base64": image_base64})
    except Exception as e:
        return jsonify({"error": f"Une erreur interne est survenue: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
