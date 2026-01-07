import numpy as np

def modality_contributions(shap_values):
    shap_vals = shap_values[0][:, 1]

    speech = np.sum(np.abs(shap_vals[:15]))
    handwriting = np.sum(np.abs(shap_vals[15:]))

    total = speech + handwriting + 1e-8

    return {
        "speech": speech / total,
        "handwriting": handwriting / total
    }
