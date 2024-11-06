from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Get API key from environment variables
HF_API_KEY = os.getenv('HF_API_KEY')
client = InferenceClient(api_key=HF_API_KEY)

if not HF_API_KEY:
    print("WARNING: HF_API_KEY not found in environment variables!")

# Available models list
AVAILABLE_MODELS = {
    "gemma-2b": "google/gemma-2-2b-it",
    "llama-3": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "starchat": "HuggingFaceH4/starchat2-15b-v0.1",
    "mistral": "mistralai/Mistral-Nemo-Instruct-2407"
}

class ChatSession:
    def __init__(self, model_name, system_prompt=""):
        if not HF_API_KEY:
            raise ValueError("Hugging Face API key not configured!")
            
        self.client = InferenceClient(api_key=HF_API_KEY)
        self.model = AVAILABLE_MODELS.get(model_name)
        if not self.model:
            raise ValueError(f"Invalid model name: {model_name}")
            
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_response(self):
        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=self.messages,
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting response from model: {str(e)}")
            print(traceback.format_exc())
            return f"Error: {str(e)}"

# Store active chat sessions
chat_sessions = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify(list(AVAILABLE_MODELS.keys()))

@app.route('/api/create_session', methods=['POST'])
def create_session():
    try:
        if not HF_API_KEY:
            return jsonify({"error": "Hugging Face API key not configured. Please check your .env file."}), 500

        data = request.json
        session_id = data.get('session_id')
        model_name = data.get('model')
        system_prompt = data.get('system_prompt', '')

        if not all([session_id, model_name]):
            return jsonify({"error": "Missing required parameters"}), 400

        if model_name not in AVAILABLE_MODELS:
            return jsonify({"error": f"Invalid model name: {model_name}"}), 400

        try:
            chat_sessions[session_id] = ChatSession(model_name, system_prompt)
            return jsonify({"message": "Session created successfully"})
        except Exception as e:
            print(f"Error creating chat session: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": f"Error creating chat session: {str(e)}"}), 500

    except Exception as e:
        print(f"Unexpected error in create_session: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        session_id = data.get('session_id')
        message = data.get('message')

        if not all([session_id, message]):
            return jsonify({"error": "Missing required parameters"}), 400

        if session_id not in chat_sessions:
            return jsonify({"error": "Invalid session ID"}), 400

        session = chat_sessions[session_id]
        session.add_message("user", message)
        
        response = session.get_response()
        session.add_message("assistant", response)

        return jsonify({
            "response": response,
            "messages": session.messages
        })

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Error processing message: {str(e)}"}), 500

if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Print status of API key
    if HF_API_KEY:
        print("Hugging Face API key found!")
    else:
        print("WARNING: No Hugging Face API key found in .env file!")
        
    app.run(debug=True, port=5000)