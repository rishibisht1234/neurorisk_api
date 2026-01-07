from fastapi import FastAPI, UploadFile, File, HTTPException,Body
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid

from inference.score import predict_with_features
from explainability.explain import explain_prediction
from explainability.feature_names import FEATURE_NAMES
from explainability.modality_shap import modality_contributions
from explainability.clinical_text import clinical_explanation
from utils.risk_bands import risk_band
import traceback
from agent.explainer import NeuroRiskAgent
agent = NeuroRiskAgent()

app = FastAPI(
    title="NeuroRisk AI",
    description="Multimodal early Parkinson’s risk screening using speech and handwriting biomarkers. Not a medical diagnosis.",
    version="1.0.0"
)



# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

ENABLE_EXPLAINABILITY = os.environ.get(
    "ENABLE_EXPLAINABILITY", "false"
).lower() == "true"




@app.get("/")
def root():
    return {
        "status": "NeuroRisk AI API running",
        "message": "Go to /docs for Swagger UI"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "service": "NeuroRisk AI"
    }
    
@app.post("/predict")
async def predict(
    audio: UploadFile = File(...),
    image: UploadFile = File(...)
):
    uid = str(uuid.uuid4())
    audio_path = f"{TEMP_DIR}/{uid}_audio.wav"
    image_path = f"{TEMP_DIR}/{uid}_image.png"

    try:
        # Save files
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # -----------------------------
        # DEBUG PRINTS (IMPORTANT)
        # -----------------------------
        print("Audio saved to:", audio_path)
        print("Image saved to:", image_path)

        # -----------------------------
        # Prediction
        # -----------------------------
        risk, X = predict_with_features(audio_path, image_path)
        print("Risk:", risk)
        print("Feature shape:", X.shape)

        band, color = risk_band(risk)

        # -----------------------------
        # SHAP (temporarily optional)
        # -----------------------------
        if ENABLE_EXPLAINABILITY:
            shap_values = explain_prediction(X)
            modality = modality_contributions(shap_values)
            clinical_text = clinical_explanation(risk, modality)
        else:
            modality = None
            clinical_text = "Explainability disabled for production deployment."

        return {
            "risk_score": float(risk),
            "risk_band": band,
            "modality_contribution": modality,
            "clinical_interpretation": clinical_text
        }

    except Exception as e:
        print("🔥 INTERNAL ERROR 🔥")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for p in [audio_path, image_path]:
            if os.path.exists(p):
                os.remove(p)


@app.post("/agent/explain")
def explain_agent(prediction: dict = Body(...)):
    """
    Takes output from /predict and returns
    human-friendly explanation.
    """
    return agent.explain(prediction)




PORT = int(os.environ.get("PORT", 8000))
