import os
import pickle
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import quote

import librosa
import numpy as np
import onnx
import onnxruntime as ort
from flask import Flask, jsonify, request
from flask_cors import CORS
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from translate import Translator
from werkzeug.utils import secure_filename

from label.labels import DOG_LABEL_TYPE, CAT_LABEL_TYPE
from pet_type import PetType

# Configuration
UPLOAD_FOLDER = Path('temp_uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'aac'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SPEECHMATICS_API_URL = "https://asr.api.speechmatics.com/v2"
SPEECHMATICS_AUTH_TOKEN = "He1Vx16DbyA2IO8zuzgQFHyhcWCEfCTe"


@dataclass
class ModelConfig:
    ort_session: ort.InferenceSession
    label_encoder: Any
    label_types: Dict[str, str]


class AudioProcessor:
    def __init__(self, n_mfcc: int = 40, max_length: int = 200):
        self.n_mfcc = n_mfcc
        self.max_length = max_length

    def extract_features(self, file_path: str) -> Optional[np.ndarray]:
        try:
            audio, sample_rate = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.n_mfcc)
            mfcc = np.pad(
                mfcc,
                ((0, 0), (0, max(0, self.max_length - mfcc.shape[1]))),
                mode='constant'
            )
            return mfcc[:, :self.max_length]
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None


class ModelLoader:
    @staticmethod
    def load_models(pet_type: PetType) -> ModelConfig:
        if pet_type == PetType.CAT:
            return ModelLoader._load_pet_model("cat")
        elif pet_type == PetType.DOG:
            return ModelLoader._load_pet_model("dog")
        raise ValueError(f"Unsupported pet type: {pet_type}")

    @staticmethod
    def _load_pet_model(pet_name: str) -> ModelConfig:
        model_path = f"models/{pet_name}_translator_model.onnx"
        encoder_path = f"models/{pet_name}_label_encoder.pkl"

        onnx_model = onnx.load(model_path)
        ort_session = ort.InferenceSession(model_path)

        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)

        label_types = CAT_LABEL_TYPE if pet_name == "cat" else DOG_LABEL_TYPE

        return ModelConfig(ort_session, label_encoder, label_types)


class Translator:
    def __init__(self):
        self.text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def translate_text(self, texts: List[str], language_code: str) -> Optional[List[str]]:
        translator = Translator(to_lang=language_code)
        try:
            return [translator.translate(text) for text in texts]
        except Exception as e:
            print(f"Translation error: {e}")
            return None

    def find_closest_audio(self, input_text: str, pet_type: PetType) -> Dict:
        try:
            embeddings_file = f"models/{pet_type.name.lower()}_text_embeddings.pkl"

            with open(embeddings_file, "rb") as f:
                data = pickle.load(f)
                audio_files = data["audio_files"]
                text_embeddings = data["text_embeddings"]

            input_embedding = self.text_encoder.encode([input_text])[0]
            distances = [cosine(input_embedding, embed) for embed in text_embeddings]
            best_match_idx = np.argmin(distances)

            return {
                'matched_audio': audio_files[best_match_idx],
                'confidence_score': 1 - distances[best_match_idx]
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}


# class AudioTranscriber:
#     @staticmethod
#     def transcribe_audio(filepath: str, language_code: str = 'en') -> str:
#         settings = ConnectionSettings(
#             url=SPEECHMATICS_API_URL,
#             auth_token=SPEECHMATICS_AUTH_TOKEN,
#         )
#
#         conf = {
#             "type": "transcription",
#             "transcription_config": {"language": language_code}
#         }
#
#         with BatchClient(settings) as client:
#             job_id = client.submit_job(audio=filepath, transcription_config=conf)
#             return client.wait_for_completion(job_id, transcription_format='txt')


# Flask application setup
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Initialize services
audio_processor = AudioProcessor()
translator = Translator()


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def welcome():
    return jsonify({"message": "hello from the server"})


@app.route("/translate", methods=['POST'])
def translate_route():
    start_time = time.time()
    print(f"\n=== Starting translation request at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    try:
        # Validate request
        if 'audio_file' not in request.files:
            return jsonify({"error": "no file uploaded"}), 404

        audio_file = request.files['audio_file']
        pet_type = request.args.get('pet_type')
        language_code = request.args.get('language_code')

        if not all([pet_type, language_code]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Validate file size
        audio_file.seek(0, 2)
        file_size = audio_file.tell()
        audio_file.seek(0)

        if file_size > MAX_FILE_SIZE:
            return jsonify({"error": "File size exceeds 10 MB limit."}), 400

        # Process audio file
        filename = secure_filename(audio_file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
            audio_file.save(temp_file.name)
            feature = audio_processor.extract_features(temp_file.name)
            os.unlink(temp_file.name)

        if feature is None:
            return jsonify({"error": "Error processing the file"}), 400

        # Load model and make prediction
        pet_type_enum = PetType[pet_type.upper()]
        model_config = ModelLoader.load_models(pet_type_enum)

        feature = np.expand_dims(feature, axis=(0, -1)).astype(np.float32)
        input_name = model_config.ort_session.get_inputs()[0].name
        output_name = model_config.ort_session.get_outputs()[0].name
        prediction = model_config.ort_session.run([output_name], {input_name: feature})[0]
        pred_label = model_config.label_encoder.inverse_transform([np.argmax(prediction)])[0]

        # Translate results
        default_label = f'{pet_type_enum.name} label'.capitalize()
        label = model_config.label_types.get(pred_label, default_label)
        translated_results = translator.translate_text([pred_label, label], language_code)

        if not translated_results:
            return jsonify({"error": "Translation failed"}), 500

        print(f"=== Total request processing time: {time.time() - start_time:.2f} seconds ===\n")
        return jsonify({
            "text": translated_results[0],
            "label": translated_results[1]
        })

    except Exception as e:
        print(f"Error in translation route: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        body = request.get_json()
        pet_type = body.get('pet_type')
        language_code = body.get('language_code')
        text = body.get('text')

        if not all([pet_type, language_code, text]):
            return jsonify({"error": "Missing required parameters"}), 400

        pet_type_enum = PetType[pet_type.upper()]
        result = translator.find_closest_audio(text, pet_type_enum)

        if "error" in result:
            return jsonify(result), 500

        matched_audio = result['matched_audio']
        encoded_audio = quote(matched_audio)
        return jsonify({
            'matched_audio_url': f'https://lingopet.mos.us-south-1.sufybkt.com/{encoded_audio}',
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=5000)
