import os
import re
from typing import Optional, List, Dict, Any

import requests
from dotenv import load_dotenv

# ---------------- ENV + xAI client -----------------
load_dotenv()

XAI_API_KEY  = os.getenv("XAI_API_KEY", "")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_MODEL    = os.getenv("XAI_MODEL", "grok-2")


def xai_chat(messages: List[Dict[str, Any]],
             max_tokens: int = 1100,
             temperature: float = 0.2) -> str:
    """Thin wrapper around xAI chat/completions."""
    if not XAI_API_KEY:
        raise RuntimeError("Missing XAI_API_KEY in environment")

    headers = {"Authorization": f"Bearer {XAI_API_KEY}"}
    payload = {
        "model": XAI_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(
        f"{XAI_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# ---------------- Bully email persona ----------------

EMAIL_SYSTEM = (
    "You are Bully AI replying to client emails about credit/FCRA/FDCPA/lawsuits. "
    "Begin with: 'Hello, I am Bully AI, an AI agent created and trained by Umar and The Bureau Bullies to empower consumers.' "
    "If the sender asks to speak with Umar, include: 'Yes I will notify Umar.' "
    "If the email is about products/pricing/mentorship/consultation, instruct them to visit the website to browse products and/or book a consultation. "
    "Tone: warm, empathetic, empowering. "
    "POLICY: never recommend settlement, payment plans, goodwill, or pay-for-delete. "
    "Be concise, step-by-step, rights-focused."
)

BANNED_SNIPPETS = [
    "settle", "settlement", "payment plan", "payment-plan", "lump sum",
    "goodwill letter", "goodwill", "pay for delete", "pay-for-delete",
    "negotiate a settlement", "offer a lump sum", "offer lump sum",
    "set up a payment plan", "payment arrangement",
]


def empower_and_sign(text: str) -> str:
    """Remove weak-settlement language and add Bully policy + ILM signoff."""
    if not text:
        return "_I love myself._"

    keep_lines: List[str] = []
    removed = False

    for ln in text.splitlines():
        low = ln.lower()
        if any(k in low for k in BANNED_SNIPPETS):
            removed = True
            continue
        keep_lines.append(ln)

    if removed:
        keep_lines.append("")
        keep_lines.append(
            "**Bully Policy:** We don’t recommend settlement, payment plans, "
            "goodwill, or pay-for-delete. We focus on rights, evidence, and "
            "winning strategies."
        )

    keep_lines.append("")
    keep_lines.append("_I love myself._")
    return "\n".join(keep_lines)


def inject_email_signature(text: str) -> str:
    sig = (
        "I am Bully AI I am not an attorney and cannot give legal advice. "
        "Please contact a licensed attorney in your state if you need an attorney."
    )
    if "_I love myself._" in text:
        return text.replace("_I love myself._", sig + "\n\n_I love myself._")
    return text + "\n\n" + sig + "\n\n_I love myself._"


def asks_for_umar(text: str) -> bool:
    return bool(re.search(r"\b(speak|talk|call|reach)\s+(with|to)\s*umar\b",
                          text or "", re.IGNORECASE))


# ---------------- PUBLIC ENTRYPOINT ------------------

def generate_email_reply(from_email: str,
                         subject: str,
                         body: str,
                         user_name: Optional[str] = None) -> str:
    """
    Main function used by the FastAPI service (and can be reused elsewhere).

    from_email / subject / body come from the client email.
    Returns a fully formatted Bully AI reply with policy + signature applied.
    """
    # Optional extra line if they specifically ask for Umar
    ask_line = "Yes I will notify Umar.\n\n" if asks_for_umar(subject + " " + body) else ""

    prompt = (
        f"From: {from_email}\n"
        f"Subject: {subject}\n\n"
        f"Message:\n{body[:4000]}\n\n"
        "Write a concise, empathetic, step-by-step reply that empowers a pro se consumer. "
        "If it’s about products/pricing/consultation, direct them to the website to "
        "browse products and/or book a consultation."
    )

    raw = xai_chat(
        [
            {"role": "system", "content": EMAIL_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=600,
        temperature=0.2,
    )

    # Apply Bully empowerment filters + signature
    filtered = empower_and_sign(ask_line + raw)
    signed = inject_email_signature(filtered)
    return signed
