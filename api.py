import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
from groq import Groq # Import Groq

# --- Initialize the Flask App ---
app = Flask(__name__)
CORS(app)

# --- Initialize API Clients ---
# Google Vision client will use the secret file
vision_client = vision.ImageAnnotatorClient() 
# Groq client will use the environment variable
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
                "content": """You are an expert medical data extraction assistant. From the user's text, which is from a French medical prescription, extract the information into a valid JSON object and nothing else. Do not add any extra explanations or markdown formatting.

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
        # Llama 3 is a powerful model available on Groq
        model="llama3-8b-8192", 
    )
    
    response_text = chat_completion.choices[0].message.content
    print("Received structured data from Groq.")
    return response_text

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
        
        # Step 1: Use Google Vision to get perfect text
        ocr_text = extract_text_with_google_vision(image_content)
        
        # Step 2: Use Groq AI to understand the text and extract data
        structured_data_json = extract_data_with_groq_ai(ocr_text)
        
        # The AI returns a JSON string, so we return it directly
        return app.response_class(
            response=structured_data_json,
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        import traceback
        print("!!! A SERVER ERROR OCCURRED !!!")
        print(traceback.format_exc())
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
