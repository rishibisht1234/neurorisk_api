import os
import joblib
import numpy as np

from features.audio_features import extract_audio_features
from features.handwriting_features import extract_handwriting_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "multimodel_pd_model.joblib")

model = joblib.load(MODEL_PATH)


def predict_with_features(audio_path, image_path):
    audio_feats = extract_audio_features(audio_path)
    handwriting_feats = extract_handwriting_features(image_path)

    X = np.hstack([audio_feats, handwriting_feats]).reshape(1, -1)

    risk = model.predict_proba(X)[0][1]
    print("Extracted features:", X.shape)
    print("Model expects:", model.n_features_in_)

    return float(risk), X
