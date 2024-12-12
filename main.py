import os
import pickle
import tempfile

import librosa
import numpy as np
import onnx
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

# Load the ONNX model
onnx_model = onnx.load("cat_translator_model.onnx")
ort_session = ort.InferenceSession("cat_translator_model.onnx")

# Load the saved label encoder
with open("cat_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


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


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            contents = await file.read()
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

        # Get input and output names
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        # Run inference
        prediction = ort_session.run([output_name], {input_name: feature})[0]

        # Get the predicted label
        pred_label = label_encoder.inverse_transform([np.argmax(prediction)])
        return {"label": pred_label[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="localhost", port=5002)
