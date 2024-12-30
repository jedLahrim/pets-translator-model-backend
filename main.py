import os
import pickle
import tempfile
from typing import Dict

import librosa
import numpy as np
import onnx
import onnxruntime as ort
from flask import Flask, request, jsonify
from translate import Translator
from werkzeug.utils import secure_filename

from label.labels import DOG_LABEL_TYPE, CAT_LABEL_TYPE
from pet_type import PetType

app = Flask(__name__)


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
    if 'audio_file' not in request.files:
        return jsonify({"error": "no file uploaded"}), 404

    audio_file = request.files['audio_file']
    pet_type = request.args.get('pet_type')
    language_code = request.args.get('language_code')

    if not pet_type:
        return jsonify({"error": "PetType is required CAT OR DOG"}), 400

    if not language_code:
        return jsonify({"error": "language_code is required"}), 400

    file_size = len(audio_file.read())
    audio_file.seek(0)  # Reset file pointer after reading
    max_size = 10 * 1024 * 1024
    if file_size > max_size:
        return jsonify({"error": "File size exceeds 10 MB limit."}), 400

    try:
        pet_type = PetType[pet_type.upper()]
        filename = secure_filename(audio_file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            audio_file.save(temp_file.name)
            file_path = temp_file.name

        feature = extract_features(file_path)
        os.unlink(file_path)

        if feature is None:
            return jsonify({"error": "Error processing the file"}), 400

        feature = np.expand_dims(feature, axis=0)
        feature = feature[..., np.newaxis]
        feature = feature.astype(np.float32)

        ort_session, label_encoder, LABEL = load_models(pet_type)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        prediction = ort_session.run([output_name], {input_name: feature})[0]
        pred_label = label_encoder.inverse_transform([np.argmax(prediction)])
        text = pred_label[0]
        default_label = f'{pet_type.name} label'.capitalize()
        label = LABEL.get(text, default_label)
        if not label:
            label = default_label
        [translated_text, translated_label] = translate_text([text, label], language_code)

        return jsonify({"text": translated_text, "label": translated_label})

    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='localhost', port=5002)