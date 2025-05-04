import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
import requests
import base64

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Get the API token from environment variables
API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"

# Headers for the API request
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# In-memory storage for chat history
chat_history = []

@app.route('/')
def home():
    with open('index.html') as file:
        html_content = file.read()
    return html_content

@app.route('/style.css')
def css():
    return send_from_directory('.', 'style.css')

@app.route('/generate', methods=['POST'])
def generate():
    text = request.json.get('text', '').strip()
    if not text:
        return jsonify({"error": "Text input is required"})

    print(f"Received text: {text}")
    payload = {"inputs": text}

    print("Sending request to Hugging Face...")
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)  # 2-minute timeout
        print(f"API response status: {response.status_code}")
        print(f"API response content: {response.text[:200]}")  # Limit for readability

        if response.status_code == 200:
            image_bytes = response.content
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            image_url = f"data:image/png;base64,{image_base64}"
            print("Image generated successfully!")

            # Add to chat history
            chat_history.append({"text": text, "image": image_url})

            return jsonify({"image": image_url})
        else:
            print(f"Error from Hugging Face: {response.text}")
            return jsonify({"error": "Failed to generate image", "status": response.status_code})
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return jsonify({"error": "Request to Hugging Face failed", "details": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
