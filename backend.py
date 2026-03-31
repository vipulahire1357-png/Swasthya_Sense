"""
backend.py — Swasthya Sense LLM Backend (Groq LLaMA 3)
"""

import os
from groq import Groq
from dotenv import load_dotenv
from medicine_db import get_db_connection

load_dotenv()


def get_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return None
    return Groq(api_key=api_key)


def query_llm(prompt: str) -> str:
    client = get_client()
    if not client:
        return "❌ Error: Set a valid GROQ_API_KEY in the .env file."
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Groq API Error: {e}"


def check_interaction(med1: str, med2: str, *additional_meds) -> dict:
    meds     = [med1, med2] + list(additional_meds)
    meds_str = ", ".join([m.strip() for m in meds if m.strip()])

    prompt = (
        f"Check for drug interactions between: {meds_str}. "
        "First line must be exactly 'Severity: Safe', 'Severity: Moderate', or 'Severity: Dangerous'. "
        "Then provide a brief clinical explanation."
    )
    response = query_llm(prompt)

    severity = "Unknown"
    lower    = response.lower()
    if "severity: dangerous" in lower or "dangerous" in lower:
        severity = "Dangerous"
    elif "severity: moderate" in lower or "moderate" in lower:
        severity = "Moderate"
    elif "severity: safe" in lower or "safe" in lower:
        severity = "Safe"

    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "INSERT INTO interactions (medications, severity, details) VALUES (?, ?, ?)",
            (meds_str, severity, response)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("DB error:", e)

    return {"severity": severity, "details": response}
