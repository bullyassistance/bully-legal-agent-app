# bully_core.py
"""
Core logic for Bully Legal Agent that can be reused by Streamlit and the email API.
Right now this is a simple placeholder. Later you'll paste in your real logic.
"""

from typing import Optional

def generate_email_reply(email_text: str, user_name: Optional[str] = None) -> str:
    """
    Take the raw email text and return Bully Legal Agent's advice as plain text.
    """
    intro = f"Hi {user_name or 'Friend'},\n\n"

    body = (
        "Thanks for reaching out about your credit situation.\n\n"
        "This is a placeholder response from Bully Legal Agent's email core. "
        "Once everything is wired up, you'll replace this function body with "
        "the real analysis / LLM call that you already use in your app.\n\n"
    )

    closing = (
        "---\nThis is Bully Legal Agent, an automated educational assistant. "
        "This is NOT legal advice and does not create an attorney–client "
        "relationship. Always consult a licensed attorney in your state."
    )

    return intro + body + closing
