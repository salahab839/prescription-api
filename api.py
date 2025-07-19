# api.py
import os
import traceback
import json
import re
import pandas as pd
import uuid
import time
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, join_room
from functools import lru_cache
from google.cloud import vision
from groq import Groq
from thefuzz import process, fuzz

# --- Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chifa_data.xlsx")

app = Flask(__name__)
CORS(app)
# Use 'eventlet' for production with gunicorn
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
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
    """Loads and pre-processes the medication database at startup."""
    try:
        df = pd.read_excel(DB_PATH)
        # Create a new column with normalized names for faster searching
        df['normalized_name'] = df['NOM_MED'].apply(normalize_string)
        print("Database loaded and pre-processed successfully.")
        return df
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load database {DB_PATH}. Error: {e}")
        return pd.DataFrame()

MEDS_DF = load_database()

@lru_cache(maxsize=256)
def find_best_match_in_db(text_to_search):
    """Finds the best matching medication using the pre-processed data."""
    if MEDS_DF.empty: return None
    
    normalized_text = normalize_string(text_to_search)
    choices = MEDS_DF['normalized_name'].to_list()
    
    # Use a robust scorer for better matching
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
    """Processes an image using Vision and Groq."""
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
    """Client joins a session room."""
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)
        print(f"Client joined room: {session_id}")

# --- API Endpoints ---
@app.route('/api/start-session', methods=['POST'])
def start_session():
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "status": "active",
        "medications": [],
        "timestamp": time.time()
    }
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
            # Emit data instantly to the desktop app via WebSocket
            socketio.emit('new_medication', {
                "data": best_match_data,
                "image_base64": image_base64
            }, room=session_id)
            
            # Optionally store it in the session
            SESSIONS[session_id]['medications'].append({"data": best_match_data})
            
            return jsonify({"status": "success", "message": "Traitement réussi"})
        else:
            return jsonify({"status": "error", "message": "Médicament non trouvé"}), 404

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Erreur de traitement: {e}"}), 500

@app.route('/api/finish-session/<session_id>', methods=['POST'])
def finish_session(session_id):
    # This can be triggered by navigator.sendBeacon
    if session_id in SESSIONS:
        SESSIONS[session_id]['status'] = 'finished'
        # Notify the desktop app that the mobile user has left
        socketio.emit('session_finished', room=session_id)
    return jsonify({"status": "success"})

# This endpoint is no longer needed by the app but is kept for compatibility/testing
@app.route('/api/check-session/<session_id>')
def check_session(session_id):
    if session_id in SESSIONS:
        return jsonify(SESSIONS[session_id])
    return jsonify({"status": "invalid"}), 404

if __name__ == '__main__':
    # For local testing, run with: python api.py
    socketio.run(app, debug=True, port=5000)
