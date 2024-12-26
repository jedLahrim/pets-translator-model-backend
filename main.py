import os
import pickle
import tempfile
from typing import Dict

import librosa
import numpy as np
import onnx
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from translate import Translator

from label.labels import DOG_LABEL_TYPE, CAT_LABEL_TYPE
from pet_type import PetType

#
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def load_models(pet_type: PetType):
    ort_session = {}
    label_encoder = {}
    LABEL: Dict[str, str] = {}
    if pet_type == PetType.CAT:
        LABEL = CAT_LABEL_TYPE
        # Load the ONNX model
        onnx_cat_model = onnx.load("models/cat_translator_model.onnx")
        ort_cat_session = ort.InferenceSession("models/cat_translator_model.onnx")
        # Load the saved label encoder
        with open("models/cat_label_encoder.pkl", "rb") as f:
            cat_label_encoder = pickle.load(f)
            ort_session = ort_cat_session
            label_encoder = cat_label_encoder


    elif pet_type == PetType.DOG:
        LABEL = DOG_LABEL_TYPE
        # Load the ONNX model
        onnx_dog_model = onnx.load("models/dog_translator_model.onnx")
        ort_dog_session = ort.InferenceSession("models/dog_translator_model.onnx")

        # Load the saved label encoder
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


@app.get("/")
async def welcome():
    return {"message": "hello from the server"}


def translate_text(texts: list, language_code: str):
    translator = Translator(to_lang=language_code)
    try:
        results = [translator.translate(text) for text in texts]
        return results
    except Exception as e:
        print(f"Error: {e}")
        return None


@app.post("/translate")
async def translate(pet_type: PetType, language_code: str, audio_file: UploadFile = File(...)):
    file_size = audio_file.size

    if file_size > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 5 MB limit.")

    if not pet_type:
        raise HTTPException(400, detail="PetType is required CAT OR DOG")

    if not language_code:
        raise HTTPException(400, detail="language_code is required")

    if not audio_file:
        raise HTTPException(404, detail="no file uploaded")
    try:
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_file:
            contents = await audio_file.read()
            temp_file.write(contents)
            file_path = temp_file.name

        # Extract features from the audio
        feature = extract_features(file_path)

        # Clean up the temporary file
        os.unlink(file_path)

        if feature is None:
            raise HTTPException(status_code=400, detail="Error processing the file")

        feature = np.expand_dims(feature, axis=0)  # Add batch dimension
        feature = feature[..., np.newaxis]  # Add channel dimension

        # Convert to float32
        feature = feature.astype(np.float32)
        ort_session, label_encoder, LABEL = load_models(pet_type)
        # Get input and output names
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        # Run inference
        prediction = ort_session.run([output_name], {input_name: feature})[0]

        # Get the predicted label
        pred_label = label_encoder.inverse_transform([np.argmax(prediction)])
        text = pred_label[0]
        default_label = f'{pet_type.name} label'.capitalize()
        label = LABEL.get(text, default_label)
        if not label:
            label = default_label
        [translated_text, translated_label] = translate_text([text, label], language_code)
        return {"text": translated_text, "label": translated_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="localhost", port=5002)
