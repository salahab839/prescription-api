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

# --- Initialisation de l'application et des clients ---
app = Flask(__name__)
CORS(app)

vision_client = vision.ImageAnnotatorClient()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- Fonctions de normalisation et de signature ---
def normalize_string(s):
    if not isinstance(s, str): return ""
    s = s.lower()
    s = s.replace('bte', 'b').replace('boite', 'b')
    s = s.replace('comprimés', 'cp').replace('comprimé', 'cp')
    s = s.replace('grammes', 'g').replace('gramme', 'g')
    s = s.replace('milligrammes', 'mg')
    s = re.sub(r'[^a-z0-9]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def build_reference_string(row):
    # Construit la signature à partir des 3 colonnes
    nom = row.get('Nom Commercial', '')
    dosage = row.get('Dosage', '')
    presentation = row.get('Présentation', '')
    return f"{nom} {dosage} {presentation}"

# --- Chargement et préparation de la base de données au démarrage ---
DB_SIGNATURE_MAP = {}
try:
    print("Chargement de la base de données de médicaments au démarrage...")
    # Assurez-vous que le fichier est dans votre dépôt GitHub
    df = pd.read_excel("chifa_data.xlsx") 
    
    # Crée une colonne de référence combinée
    df['reference_combined'] = df.apply(build_reference_string, axis=1)
    
    # Crée le dictionnaire de signatures pour une recherche rapide
    for index, row in df.iterrows():
        normalized_signature = normalize_string(row['reference_combined'])
        # Stocke la ligne entière pour un accès facile plus tard
        DB_SIGNATURE_MAP[normalized_signature] = row.to_dict()
        
    print(f"Base de données chargée et préparée avec {len(DB_SIGNATURE_MAP)} médicaments.")
except Exception as e:
    print(f"ERREUR critique lors du chargement de la base de données : {e}")


# --- Fonctions de l'API ---
def extract_text_with_google_vision(image_content):
    image = vision.Image(content=image_content)
    response = vision_client.text_detection(image=image)
    if response.error.message: raise Exception(response.error.message)
    return response.text_annotations[0].description if response.text_annotations else ""

def extract_vignette_data_with_groq(text_content):
    # --- PROMPT AMÉLIORÉ AVEC EXEMPLES ---
    system_prompt = """
    Vous êtes un expert en lecture de vignettes de médicaments françaises. Votre tâche est d'extraire les informations dans un objet JSON valide. Ne retournez que l'objet JSON, sans aucun texte supplémentaire ni formatage markdown.

    Voici les clés que vous devez utiliser et les règles à suivre :
    - "nom": Le nom commercial SEUL. Ne pas inclure le dosage.
    - "dosage": Le dosage (ex: "250 mg", "100 MG/5 ML").
    - "conditionnement": Le conditionnement (ex: "Boite de 10 comprimés").
    - "ppa": Le prix (un nombre ou une chaîne vide).

    Exemple 1 :
    Texte en entrée : "FUCARE 250 mg Boite de 10 comprimés pelliculés PPA: 913.60"
    Votre sortie : {"nom": "FUCARE", "dosage": "250 mg", "conditionnement": "Boite de 10 comprimés pelliculés", "ppa": "913.60"}

    Exemple 2 :
    Texte en entrée : "DOLIPRANE 1000 MG B/8 COMP"
    Votre sortie : {"nom": "DOLIPRANE", "dosage": "1000 MG", "conditionnement": "B/8 COMP", "ppa": ""}
    """
    
    print("Envoi du texte à Groq pour analyse...")
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Texte de la vignette:\n---\n{text_content}\n---"}
        ],
        model="llama3-8b-8192",
        response_format={"type": "json_object"},
    )
    response_content = chat_completion.choices[0].message.content
    print(f"Réponse brute de l'IA : {response_content}") # Log pour le débogage
    return json.loads(response_content)

# --- Route de l'API ---
@app.route('/process_vignette', methods=['POST'])
def process_vignette_endpoint():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    
    try:
        image_content = file.read()
        ocr_text = extract_text_with_google_vision(image_content)
        ai_data = extract_vignette_data_with_groq(ocr_text)

        # Crée la signature de la vignette scannée à partir des données de l'IA
        vignette_signature = normalize_string(f"{ai_data.get('nom','')} {ai_data.get('dosage','')} {ai_data.get('conditionnement','')}")
        print(f"Signature de la vignette normalisée : '{vignette_signature}'")

        # Fuzzy match avec la base de données pré-chargée
        best_match_signature, score = process.extractOne(
            vignette_signature,
            DB_SIGNATURE_MAP.keys(),
            scorer=fuzz.token_set_ratio
        )
        
        print(f"Meilleure correspondance trouvée dans la DB : '{best_match_signature}' avec un score de {score}%")

        # --- SEUIL DE CONFIANCE AJUSTÉ ---
        if score >= 75:
            # Si une bonne correspondance est trouvée, retourne les données vérifiées de la DB
            verified_data = DB_SIGNATURE_MAP[best_match_signature]
            response_data = {
                "nom": verified_data.get('Nom Commercial'),
                "dosage": verified_data.get('Dosage'),
                "conditionnement": verified_data.get('Présentation'),
                "ppa": ai_data.get('ppa'), # Garde le PPA scanné car il peut changer
                "match_score": score,
                "status": "Vérifié"
            }
        else:
            # Sinon, retourne les données brutes de l'IA pour vérification manuelle
            response_data = {
                "nom": ai_data.get('nom'),
                "dosage": ai_data.get('dosage'),
                "conditionnement": ai_data.get('conditionnement'),
                "ppa": ai_data.get('ppa'),
                "match_score": score,
                "status": "Non Vérifié"
            }
            
        return jsonify(response_data)

    except Exception as e:
        print(f"!!! ERREUR SERVEUR !!!\n{traceback.format_exc()}")
        return jsonify({"error": f"Une erreur interne est survenue: {e}"}), 500

# --- Placeholder pour l'autre endpoint ---
@app.route('/process_prescription', methods=['POST'])
def process_prescription_endpoint():
    return jsonify({"message": "Endpoint pour les ordonnances non implémenté."})

if __name__ == '__main__':
    app.run(debug=True)
