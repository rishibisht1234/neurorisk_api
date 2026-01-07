import os
import joblib
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "multimodel_pd_model.joblib")

model = joblib.load(MODEL_PATH)

explainer = shap.TreeExplainer(model)

def explain_prediction(X):
    return explainer.shap_values(X)
