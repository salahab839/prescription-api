import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
from groq import Groq
import json # Import the JSON library
import traceback

# --- Initialize the Flask App & API Clients ---
app = Flask(__name__)
CORS(app)

vision_client = vision.ImageAnnotatorClient()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- API Functions ---
def extract_text_with_google_vision(image_content):
    """Uses Google Cloud Vision API for superior OCR."""
    image = vision.Image(content=image_content)
    response = vision_client.text_detection(image=image)
    if response.error.message:
        raise Exception(response.error.message)
    return response.text_annotations[0].description if response.text_annotations else ""

def extract_data_with_groq_ai(text_content):
    """Uses a Groq language model to understand and structure the text."""
    print("Sending text to Groq for analysis...")
    
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are an expert medical data extraction assistant. From the user's text, which is from a French medical prescription, extract the information into a valid JSON object and nothing else. Do not add any extra explanations or markdown formatting like ```json.
                
                The JSON object must have these keys:
                - "doctor_name"
                - "patient_name"
                - "prescription_date"
                - "patient_age"
                - "patient_dob"
                - "medications": A list of objects, where each object has "name" and "dosage_and_frequency".
                """
            },
            {
                "role": "user",
                "content": f"Here is the prescription text:\n\n---\n{text_content}\n---",
            }
        ],
        model="llama3-8b-8192",
    )
    
    response_text = chat_completion.choices[0].message.content
    print("Received raw response from Groq:", response_text)

    # --- THIS IS THE NEW, ROBUST PARSING LOGIC ---
    try:
        # Find the start and end of the JSON object
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON object found in the AI response.")

        # Extract and parse the JSON string
        json_string = response_text[json_start:json_end]
        parsed_json = json.loads(json_string)
        print("Successfully parsed JSON from AI response.")
        return parsed_json
    except (json.JSONDecodeError, ValueError) as e:
        print(f"!!! Could not parse JSON from AI response: {e} !!!")
        return {"error": "AI failed to generate valid structured data."}
    # --- END OF NEW LOGIC ---

# --- API Endpoint ---
@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        image_content = file.read()
        ocr_text = extract_text_with_google_vision(image_content)
        structured_data = extract_data_with_groq_ai(ocr_text)
        
        return jsonify(structured_data) # Use Flask's jsonify for a proper response
    except Exception as e:
        print("!!! A SERVER ERROR OCCURRED !!!")
        print(traceback.format_exc())
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
