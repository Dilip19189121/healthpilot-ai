"""
Chat + retrieval helpers for HealthPilot AI.
Uses Groq's OpenAI-compatible chat API for the conversational agent
and report generation. Falls back gracefully with no API key set.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

SYSTEM_PROMPT = (
    "You are HealthPilot, a calm and knowledgeable health information assistant "
    "embedded in a clinical screening tool. Explain concepts in plain language, "
    "reference the metrics the user gives you when relevant, and always be clear "
    "that you provide general health information, not medical diagnosis or treatment. "
    "Encourage the person to consult a licensed clinician for anything urgent or "
    "personal to their care. Keep answers concise and structured."
)


def get_client(api_key: str):
    from groq import Groq
    return Groq(api_key=api_key)


def chat(api_key: str, model: str, history: list[dict], user_message: str) -> str:
    client = get_client(api_key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
        max_tokens=700,
    )
    return resp.choices[0].message.content


def generate_report(api_key: str, model: str, metrics: dict, risk_probability: float) -> str:
    client = get_client(api_key)
    prompt = (
        "Write a short, plain-language health screening summary (150-220 words) "
        "for the patient based on these values and the model's estimated risk. "
        "Structure it as: a one-line headline, 2-3 bullet observations about the "
        "readings, and a closing note encouraging follow-up with a clinician. "
        f"\n\nReadings: {metrics}\nEstimated risk probability: {risk_probability:.0%}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=500,
    )
    return resp.choices[0].message.content


# ---------------- lightweight TF-IDF retrieval (RAG) ----------------

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def build_index(chunks: list[str]):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix


def retrieve(query: str, chunks: list[str], vectorizer, matrix, top_k: int = 3) -> list[str]:
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, matrix)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in top_idx if sims[i] > 0]
