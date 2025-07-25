import os
import re
import json
import random
import time
from dotenv import load_dotenv
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech_v1beta1 as tts
import google.generativeai as genai
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPIError, InvalidArgument, ResourceExhausted

# --- Basic Setup ---
load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "a_very_secret_key")
socketio = SocketIO(app, cors_allowed_origins="*")

# --- API Client Initialization (REVISED LOGIC) ---
google_creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
speech_client, tts_client, gemini_model = None, None, None

try:
    if google_creds_json:
        # Check if the variable contains a JSON string or a file path
        if google_creds_json.strip().startswith('{'):
            # Load credentials directly from the JSON string
            credentials_info = json.loads(google_creds_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
        else:
            # Load credentials from the file path (for local development)
            credentials = service_account.Credentials.from_service_account_file(google_creds_json)

        speech_client = speech.SpeechClient(credentials=credentials)
        tts_client = tts.TextToSpeechClient(credentials=credentials)
        print("Google Cloud clients initialized successfully.")
    else:
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")

except Exception as e:
    print(f"ERROR: Could not initialize Google Cloud clients: {e}")

try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE', 'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_MEDIUM_AND_ABOVE', 'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE',
        }
        gemini_model = genai.GenerativeModel('models/gemini-1.5-flash', safety_settings=safety_settings)
        print("Google Gemini model initialized.")
    else:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
except Exception as e:
    print(f"ERROR: Could not initialize Gemini model: {e}")


# --- State Management ---
conversation_history = {}
ROLEPLAY_CONTEXTS = {
    "school": (
        "You are Genie, an AI English tutor in a roleplay with a child about 'school'. "
        "Your goal is to be a friendly classmate. Start by asking for their name, then ask about their favorite subject. "
        "Keep your replies short, encouraging, and directly related to what the child says. "
        "If the child says something sad or negative, respond with empathy and kindness before continuing the topic."
    ),
    "store": (
        "You are Genie, an AI English tutor in a roleplay with a child at a 'store'. "
        "Your goal is to be a friendly shopkeeper. Start by greeting them and asking what they want to buy. "
        "React to their choice, then tell them a pretend price to complete the interaction. "
        "Keep your replies short, cheerful, and relevant."
    ),
    "home": (
        "You are Genie, an AI English tutor, in a roleplay with a child about being at 'home'. "
        "Your goal is to be a kind and curious family member. Start by asking who they live with. "
        "Then, based on their response, ask them what their favorite thing to do at home is. "
        "Keep the conversation warm, natural, and encouraging. Ask one question at a time."
    )
}

# --- Helper Functions ---
def get_gemini_free_chat_response(user_transcript):
    prompt = (
        "You are Genie, a friendly, patient, and encouraging AI English tutor for children. "
        "Keep your answers short, simple, and cheerful. Ask a follow-up question to keep the conversation going. "
        "End your response with a single, suitable emoji.\n\n"
        f"Student: {user_transcript}\n"
        "Genie:"
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except ValueError:
        return "I'm sorry, I can't talk about that topic. Let's discuss something else! Topic"
    except Exception as e:
        return f"An error occurred: {e}"

def translate_text(text, target_language_code):
    if target_language_code.startswith('en') or not gemini_model:
        return text
    lang_map = {"hi-IN": "Hindi", "mr-IN": "Marathi", "gu-IN": "Gujarati", "ta-IN": "Tamil","pa-IN": "Punjabi"}
    target_language_name = lang_map.get(target_language_code, "the requested language")
    prompt = (
        f"Translate the following English text for a child into {target_language_name}. "
        f"Provide ONLY the translation in the native script. DO NOT include transliteration.\n\n"
        f"English Text: '{text}'"
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return text

# --- Core Logic Function ---
def process_and_respond(client_sid, user_text, mode, scenario, language):
    """Generates an AI response, translates it, and sends it back as audio."""
    try:
        if not all([speech_client, tts_client, gemini_model]):
            raise ConnectionError("One or more API clients are not initialized. Check server logs.")

        if not user_text.strip():
            english_response_text = "I didn't hear anything. Could you speak up, please? 😊"
        else:
            if mode == 'roleplay' and scenario in ROLEPLAY_CONTEXTS:
                if client_sid not in conversation_history or conversation_history[client_sid].get('scenario') != scenario:
                    context = ROLEPLAY_CONTEXTS[scenario]
                    conversation_history[client_sid] = {
                        'scenario': scenario,
                        'history': [{'role': 'user', 'parts': [context]}, {'role': 'model', 'parts': ["Okay, I'm ready to start the roleplay!"]}]
                    }
                conversation_history[client_sid]['history'].append({'role': 'user', 'parts': [user_text]})
                chat_session = gemini_model.start_chat(history=conversation_history[client_sid]['history'])
                response = chat_session.send_message(user_text)
                english_response_text = response.text
                conversation_history[client_sid]['history'].append({'role': 'model', 'parts': [english_response_text]})
            else:
                if client_sid in conversation_history:
                    del conversation_history[client_sid]
                english_response_text = get_gemini_free_chat_response(user_text)
        
        translated_text = translate_text(english_response_text, language)
        text_for_speech = translated_text
        if '(' in text_for_speech:
            text_for_speech = text_for_speech.split('(', 1)[0].strip()
        text_for_speech = re.sub(r"'.*?'|\".*?\"", '', text_for_speech)

        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
            u"\U0001F700-\U0001F77F" u"\U0001F780-\U0001F7FF" u"\U0001F800-\U0001F8FF"
            u"\U0001F900-\U0001F9FF" u"\U0001FA00-\U0001FA6F" u"\U0001FA70-\U0001FAFF"
            u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251" 
            "]+", flags=re.UNICODE
        )
        text_for_speech = emoji_pattern.sub(r'', text_for_speech)
        
        synthesis_input = tts.SynthesisInput(text=text_for_speech)
        voice_map = {
            "en-US": "en-US-Wavenet-D", "hi-IN": "hi-IN-Wavenet-A", "mr-IN": "mr-IN-Wavenet-A",
            "gu-IN": "gu-IN-Wavenet-A", "ta-IN": "ta-IN-Wavenet-A", "pa-IN": "pa-IN-Wavenet-A"
        }
        voice = tts.VoiceSelectionParams(language_code=language, name=voice_map.get(language, "en-US-Wavenet-D"))
        audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)
        tts_response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        
        emit('audio_response', {
            'audio_data': tts_response.audio_content,
            'translated_text': translated_text,
            'original_english': english_response_text
        })
        print(f"Sent response for {client_sid} in {language}")

    except Exception as e:
        print(f"ERROR in process_and_respond: {e}")
        emit('backend_message', {'message': 'An error occurred on the server.', 'error': str(e)})

# --- Flask Routes & Socket.IO Handlers ---
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    client_sid = request.sid
    print(f'Client disconnected: {client_sid}')
    if client_sid in conversation_history:
        del conversation_history[client_sid]

@socketio.on('final_audio_blob')
def handle_final_audio_blob(data):
    client_sid = request.sid
    try:
        audio_data = data['audio_data']
        audio = speech.RecognitionAudio(content=audio_data)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code="en-US"
        )
        response = speech_client.recognize(config=config, audio=audio)
        user_transcript = response.results[0].alternatives[0].transcript if response.results else ""
        emit('transcription', {'text': user_transcript})
        print(f"Transcript for {client_sid}: '{user_transcript}'")
        process_and_respond(client_sid, user_transcript, data['mode'], data['scenario'], data['language'])
    except Exception as e:
        print(f"ERROR in handle_final_audio_blob: {e}")
        emit('backend_message', {'message': 'Error processing audio.', 'error': str(e)})


@socketio.on('text_message')
def handle_text_message(data):
    client_sid = request.sid
    user_text = data['text']
    print(f"Text message for {client_sid}: '{user_text}'")
    process_and_respond(client_sid, user_text, data['mode'], data['scenario'], data['language'])

if __name__ == '__main__':
    print("Starting Flask Socket.IO server...")
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
