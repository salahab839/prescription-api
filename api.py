# api.py
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
    replacements = {'bte': 'b', 'boite': 'b', 'comprimés': 'cp', 'comprimé': 'cp', 'grammes': 'g', 'gramme': 'g', 'milligrammes': 'mg', 'injectable': 'inj', 'ampoule': 'amp'}
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = re.sub(r'[^a-z0-9. ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def extract_numeric_dosage(dosage_str):
    if not isinstance(dosage_str, str): return None
    numbers = re.findall(r'(\d+\.?\d*)', dosage_str)
    if numbers:
        return float(numbers[0])
    return None

def extract_numbers_from_string(s):
    if not isinstance(s, str): return []
    return [int(num) for num in re.findall(r'\d+', s)]

def build_reference_string(row, include_name=True, include_details=True):
    parts = []
    if include_name: parts.append(row.get('Nom Commercial', ''))
    if include_details:
        parts.append(row.get('Dosage', ''))
        parts.append(row.get('Présentation', ''))
    return " ".join(filter(None, parts)).strip()

def extract_price(text):
    """
    MODIFIED: Extracts a price from a string, handling equations like '1+2=3' or '1+2'.
    It prioritizes the result after an equals sign.
    """
    if not isinstance(text, str): return None
    
    # Clean the string, keep essential math characters
    text = text.replace(',', '.').strip()
    
    # Priority 1: Look for an equation with an equals sign (e.g., ...=124.50)
    match_equals = re.search(r'=(\s*\d+\.?\d*)', text)
    if match_equals:
        try:
            return f"{float(match_equals.group(1)):.2f}"
        except (ValueError, IndexError):
            pass # Fallback to other methods

    # Priority 2: Try to evaluate a simple sum (e.g., 120+23+1)
    if '+' in text:
        try:
            # Remove anything that is not a digit, a dot, or a plus sign
            cleaned_sum_text = re.sub(r'[^0-9.+]', '', text)
            if cleaned_sum_text.count('+') > 0:
                result = sum(map(float, cleaned_sum_text.split('+')))
                return f"{result:.2f}"
        except (ValueError, IndexError, TypeError):
            pass # Fallback to simple number extraction

    # Priority 3: Fallback to finding any number
    match_num = re.search(r'(\d+\.?\d+)', text)
    if match_num:
        try:
            return f"{float(match_num.group(1)):.2f}"
        except (ValueError, IndexError):
            pass
            
    return "0.00"

# --- Database Loading ---
DB_NAMES_MAP = {}
df = None
try:
    try:
        df = pd.read_excel(DB_PATH).astype(str).fillna('')
    except Exception:
        df = pd.read_csv(DB_PATH).astype(str).fillna('')
    df['DosageNumeric'] = df['Dosage'].apply(extract_numeric_dosage)
    for index, row in df.iterrows():
        row_dict = row.to_dict()
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

# --- Image Processing Logic ---
def process_image_data(image_content):
    if not all([vision_client, groq_client, df is not None]):
        return {"status": "Échec: Service non initialisé"}
    try:
        response = vision_client.text_detection(image=vision.Image(content=image_content))
        if not response.text_annotations:
            return {"status": "Échec OCR", "details": "Aucun texte détecté."}
        ocr_text = response.text_annotations[0].description
    except Exception as e:
        return {"status": "Échec OCR", "details": str(e)}
    try:
        system_prompt = """
        You are an expert at reading French medication labels. Your task is to extract information into a valid JSON object.
        The JSON must contain these keys: "nom", "dosage", "conditionnement", "ppa".
        The "ppa" can be a simple number like "450.00" or an equation like "420.50+30.00=450.50".
        Only return the JSON object.
        Example: { "nom": "CLAMOXYL", "dosage": "1 G", "conditionnement": "B/14", "ppa": "450.50" }
        """
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Texte à analyser:\n---\n{ocr_text}\n---"}],
            model="llama3-8b-8192", response_format={"type": "json_object"}
        )
        ai_data = json.loads(chat_completion.choices[0].message.content)
        # MODIFIED: Use the new robust price extraction function
        ai_data['ppa'] = extract_price(ai_data.get('ppa', ''))
        print(f"--- AI Data (with parsed PPA) ---\n{ai_data}\n---------------")
    except Exception as e:
        return {"status": "Échec de l'analyse IA", "details": str(e)}

    def get_response(db_row, score, status="Vérifié"):
        ppa_value = ai_data.get('ppa', "0.00")
        if ppa_value == "0.00":
            # MODIFIED: Use the new robust price extraction function for the database value too
            ppa_value = extract_price(db_row.get('PPA', ''))
        
        return {
            "nom": db_row.get('Nom Commercial'), "dci": db_row.get('DCI'),
            "dosage": db_row.get('Dosage'), "conditionnement": db_row.get('Présentation'),
            "ppa": ppa_value,
            "match_score": score, "status": status,
            "posologie_qte_prise": "",

            "posologie_unite": db_row.get('Forme', ''),
            "posologie_frequence": "",
            "posologie_periode": ""
        }

    ocr_name_sig = normalize_string(ai_data.get('nom',''))
    ocr_numeric_dosage = extract_numeric_dosage(ai_data.get('dosage'))
    if not ocr_name_sig:
        return {"status": "Échec", "details": "Nom du médicament non trouvé par l'IA."}
    best_name_match, score_name = process.extractOne(ocr_name_sig, DB_NAMES_MAP.keys(), scorer=fuzz.token_set_ratio)
    if score_name < 80:
        return {"status": "Échec de la reconnaissance", "details": f"Nom non correspondant (Score: {score_name})."}
    candidate_drugs = DB_NAMES_MAP[best_name_match]
    if ocr_numeric_dosage is not None:
        exact_dosage_matches = [d for d in candidate_drugs if d.get('DosageNumeric') == ocr_numeric_dosage]
        if len(exact_dosage_matches) == 1:
            return get_response(exact_dosage_matches[0], 100, status="Vérifié (Dosage Exact)")
        if len(exact_dosage_matches) > 1:
            ocr_presentation_str = ai_data.get('conditionnement', '')
            ocr_numbers = extract_numbers_from_string(ocr_presentation_str)
            if ocr_numbers:
                possible_matches = []
                for drug in exact_dosage_matches:
                    db_numbers = extract_numbers_from_string(drug.get('Présentation', ''))
                    if db_numbers and sorted(ocr_numbers) == sorted(db_numbers):
                        possible_matches.append(drug)
                if len(possible_matches) == 1:
                    return get_response(possible_matches[0], 100, status="Vérifié (N° de conditionnement)")
            ocr_presentation_sig = normalize_string(ocr_presentation_str)
            if ocr_presentation_sig:
                candidate_presentations_map = {normalize_string(d.get('Présentation', '')): d for d in exact_dosage_matches}
                best_match, score = process.extractOne(ocr_presentation_sig, candidate_presentations_map.keys(), scorer=fuzz.partial_token_set_ratio)
                if score >= 85:
                    return get_response(candidate_presentations_map[best_match], score, status="Vérifié (Forme Exacte)")
    ocr_details_sig = normalize_string(f"{ai_data.get('dosage','')} {ai_data.get('conditionnement','')}")
    list_to_search = exact_dosage_matches if 'exact_dosage_matches' in locals() and exact_dosage_matches else candidate_drugs
    candidate_details_map = {normalize_string(build_reference_string(d, include_name=False)): d for d in list_to_search}
    if not candidate_details_map:
        return {"status": "Échec de la reconnaissance", "details": "Aucun candidat à la comparaison trouvé."}
    best_details_match, score_details = process.extractOne(ocr_details_sig, candidate_details_map.keys(), scorer=fuzz.WRatio)
    if score_details >= 75:
        final_score = int((score_name * 0.6) + (score_details * 0.4))
        return get_response(candidate_details_map[best_details_match], final_score, status="Auto-Corrigé")
    return {"status": "Échec de la reconnaissance", "details": "Aucune correspondance fiable trouvée."}

# --- API Routes ---
# MODIFIED: Renamed from /api/create-session to reflect it's for starting the whole app session
@app.route('/api/start-session', methods=['POST'])
def start_session():
    session_id = str(uuid.uuid4())
    # A session now represents a persistent link to a desktop app instance
    SESSIONS[session_id] = {"status": "active", "medications": [], "timestamp": time.time()}
    print(f"New persistent session created: {session_id}")
    return jsonify({"session_id": session_id}), 201

# MODIFIED: Renamed to /scan for clarity
@app.route('/scan/<session_id>')
def phone_upload_page(session_id):
    session = SESSIONS.get(session_id)
    if not session:
        return "Session invalide ou expirée.", 404
    # The page can be opened as long as the session is valid
    return render_template('uploader.html')

# MODIFIED: Renamed to /api/upload for clarity
@app.route('/api/upload/<session_id>', methods=['POST'])
def upload_by_session(session_id):
    session = SESSIONS.get(session_id)
    if not session:
        return jsonify({"error": "Session invalide ou expirée"}), 404
    # We no longer check for 'pending' status, an 'active' session can always receive uploads
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier"}), 400
    try:
        image_content = request.files['file'].read()
        processed_data = process_image_data(image_content)
        if "Échec" not in processed_data.get("status", ""):
            image_base64 = base64.b64encode(image_content).decode('utf-8')
            # Add medication to the persistent session
            SESSIONS[session_id]['medications'].append({"data": processed_data, "image_base64": image_base64})
            SESSIONS[session_id]['timestamp'] = time.time() # Update timestamp on activity
            return jsonify({"status": "success", "message": "Médicament ajouté."})
        else:
            return jsonify({"status": "failure", "message": processed_data.get("details", "Vignette illisible")})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Erreur de traitement: {e}"}), 500

# MODIFIED: This now only gets called by the phone when the page is closed
@app.route('/api/finish-session/<session_id>', methods=['POST'])
def finish_session(session_id):
    # This marks the session for deletion, it doesn't affect the desktop app's state directly
    if session_id in SESSIONS:
        SESSIONS[session_id]['status'] = 'finished'
        print(f"Phone disconnected from session: {session_id}")
        return jsonify({"status": "success"})
    return jsonify({"error": "Session invalide"}), 404

# MODIFIED: check-session logic adjusted for persistent sessions
@app.route('/api/check-session/<session_id>')
def check_session(session_id):
    # Clean up old, finished sessions first
    for s_id, s_info in list(SESSIONS.items()):
        # Expire sessions finished for > 2 mins or active for > 1 hour
        is_finished = s_info.get("status") == 'finished' and (time.time() - s_info.get("timestamp", 0) > 120)
        is_stale = s_info.get("status") != 'finished' and (time.time() - s_info.get("timestamp", 0) > 3600)
        if is_finished or is_stale:
            del SESSIONS[s_id]
            print(f"Expired session cleaned up: {s_id}")

    if session_id not in SESSIONS:
        return jsonify({"status": "expired"}), 410 # Use 410 Gone for expired sessions
    
    session_info = SESSIONS[session_id]
    return jsonify({"status": session_info['status'], "medications": session_info['medications']})

# This endpoint remains for local file processing
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
        return jsonify({"error": f"Erreur de traitement: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
