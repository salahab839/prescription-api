# This is the simplified API for your Render server.
# It only needs to process one image at a time.

import os
import traceback
import json
import re
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
from groq import Groq
from thefuzz import process, fuzz

# --- Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chifa_data.xlsx")

app = Flask(__name__)
CORS(app)

# --- Initialize Clients ---
try:
    print("Initialisation des clients API...")
    vision_client = vision.ImageAnnotatorClient()
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("Clients API initialisés avec succès.")
except Exception as e:
    print(f"ERREUR CRITIQUE lors de l'initialisation des clients API: {e}")
    traceback.print_exc()


# --- Helper Functions (no changes) ---
def normalize_string(s):
    if not isinstance(s, str): return ""
    s = s.lower().replace('bte', 'b').replace('boite', 'b').replace('comprimés', 'cp').replace('comprimé', 'cp').replace('grammes', 'g').replace('gramme', 'g').replace('milligrammes', 'mg')
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def build_reference_string(row):
    nom = row.get('Nom Commercial', '')
    dosage = row.get('Dosage', '')
    presentation = row.get('Présentation', '')
    return f"{nom} {dosage} {presentation}"

# --- Load Database (no changes) ---
DB_SIGNATURE_MAP = {}
try:
    print(f"Chargement de la base de données depuis : {DB_PATH}")
    df = pd.read_excel(DB_PATH) 
    df['reference_combined'] = df.apply(build_reference_string, axis=1)
    for index, row in df.iterrows():
        normalized_signature = normalize_string(row['reference_combined'])
        DB_SIGNATURE_MAP[normalized_signature] = row.to_dict()
    print(f"Base de données chargée avec {len(DB_SIGNATURE_MAP)} médicaments.")
except Exception as e:
    print(f"ERREUR critique lors du chargement de la base de données : {e}")

# --- API Functions (no changes) ---
def extract_text_with_google_vision(image_content):
    image = vision.Image(content=image_content)
    response = vision_client.text_detection(image=image)
    if response.error.message: raise Exception(response.error.message)
    return response.text_annotations[0].description if response.text_annotations else ""

def extract_vignette_data_with_groq(text_content):
    # --- THIS IS THE CORRECTED PROMPT ---
    system_prompt = """
    Vous êtes un expert en lecture de vignettes de médicaments françaises. Votre unique tâche est de retourner un objet JSON valide.
    Ne retournez que l'objet JSON, sans aucun texte supplémentaire ni formatage markdown.

    Voici les clés que vous devez utiliser : "nom", "dosage", "conditionnement", "ppa".
    """
    
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Texte de la vignette à analyser:\n---\n{text_content}\n---"}],
        model="llama3-8b-8192", 
        response_format={"type": "json_object"},
    )
    return json.loads(chat_completion.choices[0].message.content)

# --- API Route ---
@app.route('/process_vignette', methods=['POST'])
def process_vignette_endpoint():
    print("\n--- Nouvelle requête reçue sur /process_vignette ---")
    if 'file' not in request.files: 
        print("ERREUR: 'file' non trouvé dans la requête.")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '': 
        print("ERREUR: Nom de fichier vide.")
        return jsonify({"error": "No selected file"}), 400
    
    try:
        print("[CHECKPOINT 1] Lecture du contenu de l'image.")
        image_content = file.read()
        print(f"[CHECKPOINT 1] Succès. {len(image_content)} bytes lus.")

        print("[CHECKPOINT 2] Appel de l'API Google Vision pour l'OCR.")
        ocr_text = extract_text_with_google_vision(image_content)
        print("[CHECKPOINT 2] Succès. Texte extrait par l'OCR.")

        print("[CHECKPOINT 3] Appel de l'API Groq pour l'analyse du texte.")
        ai_data = extract_vignette_data_with_groq(ocr_text)
        print("[CHECKPOINT 3] Succès. Données extraites par l'IA.")

        print("[CHECKPOINT 4] Normalisation et recherche dans la base de données.")
        vignette_signature = normalize_string(f"{ai_data.get('nom','')} {ai_data.get('dosage','')} {ai_data.get('conditionnement','')}")
        
        if not DB_SIGNATURE_MAP:
             print("ERREUR: La base de données de médicaments n'est pas chargée.")
             return jsonify({"error": "La base de données de médicaments n'est pas chargée."}), 500

        best_match_signature, score = process.extractOne(vignette_signature, DB_SIGNATURE_MAP.keys(), scorer=fuzz.token_set_ratio)
        print(f"[CHECKPOINT 4] Succès. Meilleure correspondance: '{best_match_signature}' avec score {score}%.")

        if score >= 75:
            verified_data = DB_SIGNATURE_MAP[best_match_signature]
            response_data = {"nom": verified_data.get('Nom Commercial'), "dosage": verified_data.get('Dosage'), "conditionnement": verified_data.get('Présentation'), "ppa": ai_data.get('ppa'), "match_score": score, "status": "Vérifié"}
        else:
            response_data = {"nom": ai_data.get('nom'), "dosage": ai_data.get('dosage'), "conditionnement": ai_data.get('conditionnement'), "ppa": ai_data.get('ppa'), "match_score": score, "status": "Non Vérifié"}
        
        print("--- Requête terminée avec succès. Envoi de la réponse. ---")
        return jsonify(response_data)
        
    except Exception as e:
        # This will now hopefully catch and print the detailed error to the logs
        print(f"!!!!!! ERREUR INATTENDUE DANS L'ENDPOINT !!!!!!")
        print(f"Type de l'erreur: {type(e).__name__}")
        print(f"Message de l'erreur: {e}")
        print("--- Traceback complet ---")
        traceback.print_exc()
        print("--------------------------")
        return jsonify({"error": f"Une erreur interne est survenue sur le serveur: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
