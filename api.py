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
DB_MAIN_PATH = os.path.join(BASE_DIR, "chifa_data.xlsx - Sheet1.csv")
DB_TARIF_PATH = os.path.join(BASE_DIR, "chifa_data 12121.xlsx - Sheet1.csv")

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

def build_reference_string(row, include_name=True, include_dci=True, include_details=True):
    parts = []
    if include_name: parts.append(row.get('Nom Commercial', ''))
    if include_dci: parts.append(row.get('DCI', ''))
    if include_details:
        parts.append(row.get('Dosage', ''))
        parts.append(row.get('Présentation', ''))
    return " ".join(filter(None, parts)).strip()

def parse_price(text):
    if not isinstance(text, str): return ""
    if '=' in text: text = text.split('=')[-1]
    elif ':' in text: text = text.split(':')[-1]
    cleaned_text = re.sub(r'[^0-9,.]', '', text).strip().replace(',', '.')
    try:
        return f"{float(cleaned_text):.2f}"
    except (ValueError, TypeError):
        return ""

# --- Database Loading and Merging ---
DB_NAMES_MAP = {}
DB_TARIF_MAP = {} # Map for Tarif de Référence lookups
df_merged = None
try:
    df_main = pd.read_csv(DB_MAIN_PATH).astype(str).fillna('')
    print(f"Main DB loaded with {len(df_main)} records.")

    df_tarif_source = pd.read_csv(DB_TARIF_PATH).astype(str).fillna('')
    # IMPORTANT: Assumes the reference price column is named 'Tarif'. If not, change it here.
    if 'Tarif' not in df_tarif_source.columns:
        if 'PPA' in df_tarif_source.columns:
            print("WARNING: 'Tarif' column not found in tariff DB, renaming 'PPA' to 'Tarif'.")
            df_tarif_source = df_tarif_source.rename(columns={'PPA': 'Tarif'})
        else:
            raise ValueError("'Tarif' or 'PPA' column not found in the tariff source file.")
    print(f"Tariff DB loaded with {len(df_tarif_source)} records.")

    merge_keys = ['Nom Commercial', 'Dosage', 'Présentation']
    for key in merge_keys:
        if key not in df_main.columns: raise ValueError(f"Merge key '{key}' not in main DB.")
        if key not in df_tarif_source.columns: raise ValueError(f"Merge key '{key}' not in tariff DB.")

    df_merged = pd.merge(df_main, df_tarif_source[['Tarif'] + merge_keys], on=merge_keys, how='left')
    print(f"Merged DB created with {len(df_merged)} records.")

    for index, row in df_merged.iterrows():
        row_dict = row.to_dict()
        
        name_signature = normalize_string(row_dict.get('Nom Commercial', ''))
        if name_signature:
            if name_signature not in DB_NAMES_MAP: DB_NAMES_MAP[name_signature] = []
            DB_NAMES_MAP[name_signature].append(row_dict)
        
        tarif_str = parse_price(row_dict.get('Tarif', ''))
        if tarif_str:
            if tarif_str not in DB_TARIF_MAP: DB_TARIF_MAP[tarif_str] = []
            DB_TARIF_MAP[tarif_str].append(row_dict)

    print(f"Database maps created. {len(DB_TARIF_MAP)} unique reference tariffs found.")

except Exception as e:
    print(f"CRITICAL ERROR loading/merging databases: {e}")
    traceback.print_exc()

# --- Image Processing Logic (REVISED with Tarif de Référence) ---
def process_image_data(image_content):
    if not all([vision_client, groq_client, df_merged is not None]):
        return {"status": "Échec: Service non initialisé"}

    # 1. OCR and AI Extraction
    try:
        response = vision_client.text_detection(image=vision.Image(content=image_content))
        if not response.text_annotations: raise Exception("No text detected")
        ocr_text = response.text_annotations[0].description
        
        # UPDATED PROMPT to specifically ask for "tarif de référence"
        system_prompt = """
        You are an expert at reading French medication labels (vignettes). Your task is to extract information into a valid JSON object.
        The JSON must contain these keys: "nom", "dci", "dosage", "conditionnement", "ppa", and "tarif_ref".
        - "ppa" is the total price.
        - "tarif_ref" is the "Tarif de Référence". Look for labels like "TR", "T.R", "Tarif", "Tarif de Réf.", etc. It is the reimbursement base price.
        
        Only return the JSON object, with no additional text or markdown.
        Example: { "nom": "DOLIPRANE", "dci": "PARACETAMOL", "dosage": "1000 MG", "conditionnement": "BTE/8", "ppa": "195.00", "tarif_ref": "150.00" }
        """
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Texte à analyser:\n---\n{ocr_text}\n---"}],
            model="llama3-8b-8192", response_format={"type": "json_object"}
        )
        ai_data = json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        return {"status": "Échec de la lecture/analyse", "details": str(e)}

    # 2. Signature Creation and Response Formatting
    ocr_name_sig = normalize_string(ai_data.get('nom', ''))
    extracted_ppa = parse_price(ai_data.get('ppa', ''))
    extracted_tarif = parse_price(ai_data.get('tarif_ref', '')) # NEW

    def get_verified_response(db_row, score, status="Vérifié"):
        final_ppa = extracted_ppa or parse_price(db_row.get('PPA')) # PPA is the price from the main file
        return {
            "nom": db_row.get('Nom Commercial'), "dci": db_row.get('DCI'),
            "dosage": db_row.get('Dosage'), "conditionnement": db_row.get('Présentation'),
            "ppa": final_ppa, "match_score": score, "status": status,
            "posologie_qte_prise": "1", "posologie_unite": db_row.get('Forme', ''),
            "posologie_frequence": "3", "posologie_periode": "par jour"
        }

    # 3. New Matching Logic: Tarif-First
    if extracted_tarif and extracted_tarif in DB_TARIF_MAP:
        tarif_candidates = DB_TARIF_MAP[extracted_tarif]
        if len(tarif_candidates) == 1:
            return get_verified_response(tarif_candidates[0], 100, "Vérifié (Tarif Unique)")
        else:
            candidate_names = {normalize_string(c.get('Nom Commercial')): c for c in tarif_candidates}
            best_name_match, score = process.extractOne(ocr_name_sig, candidate_names.keys())
            if score > 80:
                return get_verified_response(candidate_names[best_name_match], score, "Vérifié (Tarif + Nom)")
            else:
                return {"status": "Échec: Ambiguïté", "details": f"{len(tarif_candidates)} méds trouvés pour TR {extracted_tarif}, nom incertain."}

    # 4. Fallback to name-first matching if Tarif fails
    best_name_match, score_name = process.extractOne(ocr_name_sig, DB_NAMES_MAP.keys(), scorer=fuzz.token_set_ratio)
    if score_name >= 75:
        name_candidates = DB_NAMES_MAP[best_name_match]
        if len(name_candidates) == 1:
            return get_verified_response(name_candidates[0], score_name, "Vérifié (Nom unique)")

        ocr_details_sig = normalize_string(f"{ai_data.get('dci','')} {ai_data.get('dosage','')} {ai_data.get('conditionnement','')}")
        candidate_details = {normalize_string(build_reference_string(c, include_name=False)): c for c in name_candidates}
        best_details, score_details = process.extractOne(ocr_details_sig, candidate_details.keys(), scorer=fuzz.WRatio)
        if score_details >= 65:
            final_score = int((score_name * 0.6) + (score_details * 0.4))
            return get_verified_response(candidate_details[best_details], final_score, "Auto-Corrigé (Nom + Détails)")

    return {"status": "Échec de la reconnaissance", "details": f"Aucun médicament correspondant trouvé. TR: '{extracted_tarif}', Nom: '{ocr_name_sig}'"}


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
