import os
import pickle
import tempfile
import time
from typing import Dict
from urllib.parse import quote

import librosa
import numpy as np
import onnx
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from translate import Translator
from werkzeug.utils import secure_filename
import base64
from label.labels import DOG_LABEL_TYPE, CAT_LABEL_TYPE
from pet_type import PetType
import requests

app = Flask(__name__)
CORS(app)


def load_models(pet_type: PetType):
    ort_session = {}
    label_encoder = {}
    LABEL: Dict[str, str] = {}
    if pet_type == PetType.CAT:
        LABEL = CAT_LABEL_TYPE
        onnx_cat_model = onnx.load("models/cat_translator_model.onnx")
        ort_cat_session = ort.InferenceSession("models/cat_translator_model.onnx")
        with open("models/cat_label_encoder.pkl", "rb") as f:
            cat_label_encoder = pickle.load(f)
            ort_session = ort_cat_session
            label_encoder = cat_label_encoder

    elif pet_type == PetType.DOG:
        LABEL = DOG_LABEL_TYPE
        onnx_dog_model = onnx.load("models/dog_translator_model.onnx")
        ort_dog_session = ort.InferenceSession("models/dog_translator_model.onnx")
        with open("models/dog_label_encoder.pkl", "rb") as f:
            dog_label_encoder = pickle.load(f)
            ort_session = ort_dog_session
            label_encoder = dog_label_encoder

    return ort_session, label_encoder, LABEL


def extract_features(file_path, n_mfcc=40, max_length=200):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc = np.pad(mfcc, ((0, 0), (0, max(0, max_length - mfcc.shape[1]))), mode='constant')
        return mfcc[:, :max_length]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


@app.route("/")
def welcome():
    return jsonify({"message": "hello from the server"})


def translate_text(texts: list, language_code: str):
    translator = Translator(to_lang=language_code)
    try:
        results = [translator.translate(text) for text in texts]
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None


@app.route("/translate", methods=['POST'])
def translate():
    start_time = time.time()
    print(f"\n=== Starting translation request at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    try:
        if 'audio_file' not in request.files:
            return jsonify({"error": "no file uploaded"}), 404

        audio_file = request.files['audio_file']
        pet_type = request.args.get('pet_type')
        language_code = request.args.get('language_code')

        print(f"Request params - Pet type: {pet_type}, Language: {language_code}")
        print(f"File received - Name: {audio_file.filename}, Size: {len(audio_file.read()) / 1024:.2f}KB")
        audio_file.seek(0)

        if not pet_type or not language_code:
            return jsonify({"error": "Missing required parameters"}), 400

        file_size = len(audio_file.read())
        audio_file.seek(0)
        max_size = 10 * 1024 * 1024
        if file_size > max_size:
            return jsonify({"error": "File size exceeds 10 MB limit."}), 400

        # Feature extraction timing
        feature_start = time.time()
        try:
            pet_type = PetType[pet_type.upper()]
            filename = secure_filename(audio_file.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                audio_file.save(temp_file.name)
                file_path = temp_file.name

            feature = extract_features(file_path)
            os.unlink(file_path)
            print(f"Feature extraction took: {time.time() - feature_start:.2f} seconds")

            if feature is None:
                return jsonify({"error": "Error processing the file"}), 400

        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return jsonify({"error": "Error during feature extraction."}), 500

        # Model loading timing
        model_start = time.time()
        try:
            feature = np.expand_dims(feature, axis=0)
            feature = feature[..., np.newaxis]
            feature = feature.astype(np.float32)

            ort_session, label_encoder, LABEL = load_models(pet_type)
            print(f"Model loading took: {time.time() - model_start:.2f} seconds")

        except Exception as e:
            print(f"Model loading error: {str(e)}")
            return jsonify({"error": "Error loading models."}), 500

        # Prediction timing
        predict_start = time.time()
        try:
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            prediction = ort_session.run([output_name], {input_name: feature})[0]
            pred_label = label_encoder.inverse_transform([np.argmax(prediction)])
            print(f"Prediction took: {time.time() - predict_start:.2f} seconds")

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({"error": "Error during prediction."}), 500

        # Translation timing
        translation_start = time.time()
        try:
            text = pred_label[0]
            default_label = f'{pet_type.name} label'.capitalize()
            label = LABEL.get(text, default_label)
            if not label:
                label = default_label
            [translated_text, translated_label] = translate_text([text, label], language_code)
            print(f"Translation took: {time.time() - translation_start:.2f} seconds")

        except Exception as e:
            print(f"Translation error: {str(e)}")
            return jsonify({"error": "Error during translation."}), 500

        total_time = time.time() - start_time
        print(f"=== Total request processing time: {total_time:.2f} seconds ===\n")

        return jsonify({"text": translated_text, "label": translated_label})

    except Exception as e:
        print(f"General error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'aac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def find_closest_audio(input_text, pet_type: PetType):
    # Initialize models
    text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    audio_files, text_embeddings = None, None  # Initialize variables

    if pet_type == PetType.DOG:
        # Load stored embeddings for DOG
        with open("models/dog_text_embeddings.pkl", "rb") as f:
            data = pickle.load(f)
            audio_files = data["audio_files"]
            text_embeddings = data["text_embeddings"]

    elif pet_type == PetType.CAT:
        # Load stored embeddings for CAT
        with open("models/cat_text_embeddings.pkl", "rb") as f:
            data = pickle.load(f)
            audio_files = data["audio_files"]
            text_embeddings = data["text_embeddings"]

    # Check if text_embeddings was initialized
    if text_embeddings is None:
        return {
            "error": "Invalid pet type provided.",
            "status": "error"
        }

    input_embedding = text_encoder.encode([input_text])[0]
    distances = [cosine(input_embedding, text_embedding) for text_embedding in text_embeddings]
    best_match_idx = np.argmin(distances)
    return {
        'matched_audio': audio_files[best_match_idx],
        'confidence_score': 1 - distances[best_match_idx]  # Convert distance to confidence score
    }


@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        body = request.get_json()
        text = body['text']
        pet_type = body['petType']
        if not pet_type or not text:
            return jsonify({"error": "Missing required parameters pet_type, text"}), 400

        try:
            # Find the closest matching audio
            pet_type = PetType[pet_type.upper()]
            result = find_closest_audio(text, pet_type)

            matched_audio = result.get('matched_audio')
            encoded_audio = quote(matched_audio)
            return jsonify({
                'matched_audio_url': f'https://petspeak.mos.us-south-1.sufybkt.com/{encoded_audio}',
            })

        except Exception as e:
            # Clean up the temporary file in case of error
            return jsonify({
                'error': f'Transcription error: {str(e)}',
                'status': 'error'
            }), 500

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route("/transcribe", methods=['POST'])
def transcribe_audio():
    try:
        # Check if the audio file is present in the request
        if 'audio_file' not in request.files:
            return jsonify({"error": "no file uploaded"}), 400

        audio_file = request.files['audio_file']

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_file:
            audio_file.save(temp_file.name)
            file_path = temp_file.name

        # Read the audio file and encode it in base64
        with open(file_path, "rb") as f:
            audio_data = f.read()
            encoded_audio = base64.b64encode(audio_data).decode("utf-8")

        # Hugging Face API endpoint and headers
        api_url = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
        headers = {
            "Authorization": "Bearer hf_jwt_eyJhbGciOiJIUzI1NiJ9.eyJwZXJtaXNzaW9ucyI6eyJpbmZlcmVuY2Uuc2VydmVybGVzcy53cml0ZSI6dHJ1ZX0sIm9uQmVoYWxmT2YiOnsia2luZCI6InVzZXIiLCJfaWQiOiI2NTE1YmUyZWRhOGEzZDdjZGQ4OTQ2MmUiLCJ1c2VyIjoiamVkMDYifSwiaWF0IjoxNzQxMjIxMjQzLCJzdWIiOiJodHRwczovL3JvdXRlci5odWdnaW5nZmFjZS5jbyIsImV4cCI6MTc0MTIyNDg0MywiaXNzIjoiaHR0cHM6Ly9odWdnaW5nZmFjZS5jbyJ9.rbmBrARFviKVy21P7aJ7N8HQ38IBn1ZTQb3uCv6XcjE",  # Replace with your Hugging Face token
            "Content-Type": "audio/webm;codecs=opus"
        }

        # Prepare the payload
        payload = {
            "inputs": encoded_audio,
            # "parameters": {
            #     "task": "transcribe"  # Optional: Specify the task (transcribe or translate)
            # }
        }

        # Send the request to the Hugging Face API
        response = requests.post(api_url, headers=headers, json=payload)

        # Check if the request was successful
        if response.status_code != 200:
            return jsonify({"error": f"Hugging Face API error: {response.text}"}), 500

        # Extract the transcribed text from the response
        transcribed_text = response.json().get("text", "")

        # Clean up the temporary file
        os.unlink(file_path)

        return jsonify({"transcribed_text": transcribed_text})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=5000)
