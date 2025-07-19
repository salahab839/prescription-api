# api.py

# --- FIX for eventlet crash ---
# These two lines MUST be the first lines of code to be executed.
import eventlet
eventlet.monkey_patch()
# -----------------------------

import os
import traceback
import pandas as pd
import uuid
import time
import base64
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, join_room
from functools import lru_cache
from google.cloud import vision
from groq import Groq
from thefuzz import process, fuzz

# --- Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chifa_data.xlsx")

app = Flask(__name__, template_folder='templates')
CORS(app)
# Note: async_mode is no longer explicitly set to 'eventlet' here
# because monkey_patch() and the gunicorn command handle it.
socketio = SocketIO(app, cors_allowed_origins="*")
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
    replacements = {'bte': 'b', 'boite': 'b', 'comprimés': 'cp', 'comprimé': 'cp', 'grammes': 'g', 'gramme': 'g', 'milligrammes': 'mg', 'injectable': 'inj', 'ampoule': 'amp'}
    s = re.sub(r'\W+', '', s)
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s

def load_database():
    """Loads and pre-processes the medication database."""
    try:
        if not os.path.exists(DB_PATH):
            print(f"CRITICAL ERROR: Database file not found at {DB_PATH}. Make sure it's in your Git repository.")
            return pd.DataFrame() # Return empty DataFrame
        
        df = pd.read_excel(DB_PATH)
        # Check if the essential column exists
        if 'NOM_MED' not in df.columns:
            print(f"CRITICAL ERROR: 'NOM_MED' column not found in {DB_PATH}. Check the Excel file.")
            print(f"Available columns are: {df.columns.tolist()}")
            return pd.DataFrame()

        df['normalized_name'] = df['NOM_MED'].apply(normalize_string)
        print("Database loaded and pre-processed successfully.")
        return df
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load or process database {DB_PATH}. Error: {e}")
        return pd.DataFrame()

MEDS_DF = load_database()

@lru_cache(maxsize=256)
def find_best_match_in_db(text_to_search):
    if MEDS_DF.empty:
        print("Skipping search: Medication database is empty or not loaded.")
        return None
    
    normalized_text = normalize_string(text_to_search)
    choices = MEDS_DF['normalized_name'].to_list()
    
    best_match_tuple = process.extractOne(normalized_text, choices, scorer=fuzz.token_sort_ratio)
    
    if not best_match_tuple: return None
    best_match_name, score = best_match_tuple

    if score > 85:
        match_row = MEDS_DF[MEDS_DF['normalized_name'] == best_match_name].iloc[0]
        return {
            "dci": match_row.get("DCI", "N/A"),
            "dosage": match_row.get("DOSAGE", "N/A"),
            "forme": match_row.get("FORME", "N/A"),
            "presentation": match_row.get("PRESENTATION", "N/A"),
            "nom_med": match_row.get("NOM_MED", "N/A"),
            "cip": match_row.get("CIP", "N/A"),
            "prix": match_row.get("PRIX", 0),
            "remboursable": match_row.get("REMBOURSABLE", "NON"),
            "score": score
        }
    return None

def process_image_data(image_content):
    if not vision_client or not groq_client:
        raise ConnectionError("API clients not initialized.")
    image = vision.Image(content=image_content)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    if not texts:
        raise ValueError("Aucun texte détecté par l'OCR.")
    ocr_text = texts[0].description
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Tu es un expert en pharmacie. Extrais uniquement le nom complet, le dosage, et la forme du médicament depuis le texte suivant. Ne donne rien d'autre. Format: Nom Dosage Forme."},
            {"role": "user", "content": ocr_text}
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# --- WebSocket Events ---
@socketio.on('join')
def on_join(data):
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)
        print(f"Client joined room: {session_id}")

# --- API Endpoints ---
@app.route('/')
def index():
    return render_template('uploader.html')

@app.route('/api/start-session', methods=['POST'])
def start_session():
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = { "status": "active", "medications": [], "timestamp": time.time() }
    return jsonify({"session_id": session_id}), 201

@app.route('/api/upload-by-session/<session_id>', methods=['POST'])
def upload_by_session(session_id):
    if session_id not in SESSIONS or 'file' not in request.files:
        return jsonify({"error": "Session invalide ou fichier manquant"}), 404
    try:
        image_content = request.files['file'].read()
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        processed_text = process_image_data(image_content)
        best_match_data = find_best_match_in_db(processed_text)
        if best_match_data:
            socketio.emit('new_medication', { "data": best_match_data, "image_base64": image_base64 }, room=session_id)
            SESSIONS[session_id]['medications'].append({"data": best_match_data})
            return jsonify({"status": "success", "message": "Traitement réussi"})
        else:
            return jsonify({"status": "error", "message": "Médicament non trouvé"}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Erreur de traitement: {e}"}), 500

@app.route('/api/finish-session/<session_id>', methods=['POST'])
def finish_session(session_id):
    if session_id in SESSIONS:
        SESSIONS[session_id]['status'] = 'finished'
        socketio.emit('session_finished', room=session_id)
    return jsonify({"status": "success"})

# This local run block is not used by gunicorn but is good for testing
if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
