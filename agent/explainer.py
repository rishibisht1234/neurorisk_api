from agent.azure_llm import call_llm

class NeuroRiskAgent:

    def explain(self, prediction: dict):
        risk = prediction["risk_score"]
        band = prediction["risk_band"]

        response = prediction.copy()

        # 1️⃣ Summary
        if risk < 0.3:
            response["summary"] = "Low neurological risk indicators were detected."
            response["tone"] = "reassuring"
        elif risk < 0.6:
            response["summary"] = "Moderate neurological risk indicators were detected."
            response["tone"] = "monitoring"
        else:
            response["summary"] = "Elevated neurological risk indicators were detected."
            response["tone"] = "cautionary"

        # 2️⃣ Interpretation
        response["interpretation"] = (
            "This assessment is based on patterns observed in speech "
            "and handwriting biomarkers."
        )

        # 3️⃣ Next steps
        if band == "Low Risk":
            response["recommended_next_steps"] = [
                "No immediate action required",
                "Optional re-screening after a few months"
            ]
        elif band == "Moderate Risk":
            response["recommended_next_steps"] = [
                "Monitor changes over time",
                "Consider a follow-up screening"
            ]
        else:
            response["recommended_next_steps"] = [
                "Consider consulting a healthcare professional",
                "Use this result as supportive information only"
            ]

        # 4️⃣ Disclaimer
        response["disclaimer"] = (
            "This is an AI-based early risk screening tool "
            "and not a medical diagnosis."
        )

        # ✅ SAFE LLM CALL
        try:
            llm_text = call_llm(response)
            return {
                "explanation": llm_text,
                "source": "llm"
            }

        except Exception as e:
            # 🔒 FAIL SAFE (VERY IMPORTANT FOR AZURE)
            return {
                "explanation": response["summary"],
                "fallback_reason": str(e),
                "source": "rule_based"
            }
