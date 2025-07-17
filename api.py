import os
import traceback
import json
import re
import pandas as pd
import uuid # For generating unique session IDs
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from google.cloud import vision
from groq import Groq
from thefuzz import process, fuzz

# --- Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chifa_data.xlsx")

# Tell Flask where to find the 'templates' folder
app = Flask(__name__, template_folder='templates')
CORS(app)

# --- In-memory dictionary to store session data ---
# This will link a session ID to an uploaded image file content
SESSIONS = {}

# --- Initialize Clients ---
try:
    vision_client = vision.ImageAnnotatorClient()
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    print(f"ERREUR CRITIQUE lors de l'initialisation des clients API: {e}")

# --- Helper Functions & DB Loading (No Changes) ---
def normalize_string(s):
    if not isinstance(s, str): return ""
    s = s.lower().replace('bte', 'b').replace('boite', 'b').replace('comprimés', 'cp').replace('comprimé', 'cp').replace('grammes', 'g').replace('gramme', 'g').replace('milligrammes', 'mg')
    s = re.sub(r'[^a-z0-9]', ' ', s); return re.sub(r'\s+', ' ', s).strip()
def build_reference_string(row):
    return f"{row.get('Nom Commercial', '')} {row.get('Dosage', '')} {row.get('Présentation', '')}"
DB_SIGNATURE_MAP = {}
try:
    df = pd.read_excel(DB_PATH) 
    df['reference_combined'] = df.apply(build_reference_string, axis=1)
    for index, row in df.iterrows():
        DB_SIGNATURE_MAP[normalize_string(row['reference_combined'])] = row.to_dict()
    print(f"Base de données chargée avec {len(DB_SIGNATURE_MAP)} médicaments.")
except Exception as e:
    print(f"ERREUR critique lors du chargement de la base de données : {e}")

def process_image_data(image_content):
    """ Main processing logic, refactored to be reusable. """
    ocr_text = vision_client.text_detection(image=vision.Image(content=image_content)).text_annotations[0].description
    system_prompt = "Vous êtes un expert en lecture de vignettes de médicaments françaises..."
    chat_completion = groq_client.chat.completions.create(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Texte: {ocr_text}"}], model="llama3-8b-8192", response_format={"type": "json_object"})
    ai_data = json.loads(chat_completion.choices[0].message.content)
    vignette_signature = normalize_string(f"{ai_data.get('nom','')} {ai_data.get('dosage','')} {ai_data.get('conditionnement','')}")
    best_match_signature, score = process.extractOne(vignette_signature, DB_SIGNATURE_MAP.keys(), scorer=fuzz.token_set_ratio)
    if score >= 75:
        verified_data = DB_SIGNATURE_MAP[best_match_signature]
        return {"nom": verified_data.get('Nom Commercial'), "dosage": verified_data.get('Dosage'), "conditionnement": verified_data.get('Présentation'), "ppa": ai_data.get('ppa'), "match_score": score, "status": "Vérifié"}
    else:
        return {"nom": ai_data.get('nom'), "dosage": ai_data.get('dosage'), "conditionnement": ai_data.get('conditionnement'), "ppa": ai_data.get('ppa'), "match_score": score, "status": "Non Vérifié"}

# --- NEW: API Routes for Phone-to-Desktop Workflow ---

@app.route('/api/create-session', methods=['POST'])
def create_session():
    session_id = str(uuid.uuid4()) # Generate a unique ID
    SESSIONS[session_id] = {"status": "pending", "data": None, "timestamp": time.time()}
    print(f"Session created: {session_id}")
    return jsonify({"session_id": session_id})

@app.route('/phone-upload/<session_id>')
def phone_upload_page(session_id):
    if session_id not in SESSIONS:
        return "Session invalide ou expirée.", 404
    # Serve the uploader.html page
    return render_template('uploader.html')

@app.route('/api/upload-by-session/<session_id>', methods=['POST'])
def upload_by_session(session_id):
    # --- THIS IS THE EDITED FUNCTION WITH BETTER LOGGING ---
    print(f"\n--- Requête d'upload reçue pour la session: {session_id} ---")
    if session_id not in SESSIONS:
        print("ERREUR: Session ID non trouvée.")
        return jsonify({"error": "Session invalide ou expirée"}), 404
    if 'file' not in request.files:
        print("ERREUR: 'file' non trouvé dans la requête d'upload.")
        return jsonify({"error": "Aucun fichier"}), 400
    
    file = request.files['file']
    
    try:
        print(f"[SESSION UPLOAD] Lecture du contenu de l'image...")
        image_content = file.read()
        print(f"[SESSION UPLOAD] Image lue, {len(image_content)} bytes. Traitement en cours...")
        
        processed_data = process_image_data(image_content)
        
        SESSIONS[session_id]['status'] = 'completed'
        SESSIONS[session_id]['data'] = processed_data
        print(f"Image reçue et traitée avec succès pour la session: {session_id}")
        return jsonify({"status": "success"})
    except Exception as e:
        # This will now print the full error to the logs
        print(f"!!!!!! ERREUR DANS L'UPLOAD PAR SESSION !!!!!!")
        print(f"Type de l'erreur: {type(e).__name__}")
        print(f"Message de l'erreur: {e}")
        print("--- Traceback complet ---")
        traceback.print_exc()
        print("--------------------------")
        
        SESSIONS[session_id]['status'] = 'error'
        SESSIONS[session_id]['data'] = str(e)
        return jsonify({"error": "Erreur lors du traitement de l'image"}), 500

@app.route('/api/check-session/<session_id>')
def check_session(session_id):
    if session_id not in SESSIONS:
        return jsonify({"status": "error", "message": "Session invalide"}), 404
    
    session_info = SESSIONS[session_id]
    # Clean up old sessions after some time
    if time.time() - session_info.get("timestamp", 0) > 600: # 10 minutes
        if session_id in SESSIONS:
            del SESSIONS[session_id]
        return jsonify({"status": "expired"}), 410

    if session_info['status'] == 'completed':
        # Return the data and clear the session
        data_to_return = session_info['data']
        if session_id in SESSIONS:
            del SESSIONS[session_id]
        return jsonify({"status": "completed", "data": data_to_return})
    else:
        return jsonify({"status": session_info['status']})

# --- Keep the direct upload endpoint for fallback ---
@app.route('/process_vignette', methods=['POST'])
def process_vignette_endpoint():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    try:
        image_content = file.read()
        processed_data = process_image_data(image_content)
        return jsonify(processed_data)
    except Exception as e:
        return jsonify({"error": f"Une erreur interne est survenue: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
