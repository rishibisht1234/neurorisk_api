def risk_band(risk):
    if risk < 0.3:
        return "Low Risk", "green"
    elif risk < 0.6:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"
