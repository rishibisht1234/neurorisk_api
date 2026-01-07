import os
from openai import AzureOpenAI


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


def call_llm(facts: dict) -> str:
    system_prompt = """
    You are a medical AI explanation assistant.

    STRICT RULES:
    - Do NOT diagnose.
    - Do NOT give medical advice.
    - Do NOT suggest treatment.
    - Do NOT introduce new facts.
    - Only rewrite the explanation clearly.
    - Always include a disclaimer.
    """

    user_prompt = f"""
    Risk score: {facts['risk_score']}
    Risk band: {facts['risk_band']}
    Tone: {facts['tone']}
    Summary: {facts['summary']}
    Interpretation: {facts['interpretation']}

    Explain this result clearly and cautiously for a general user.
    """

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=250
    )

    return response.choices[0].message.content.strip()
