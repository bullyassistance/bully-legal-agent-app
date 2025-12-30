# email_api.py
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from bully_core import generate_email_reply  # 👈 import shared logic

app = FastAPI()

class EmailRequest(BaseModel):
    from_email: str
    subject: str
    body: str
    user_name: Optional[str] = None

class EmailResponse(BaseModel):
    reply: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Bully Legal Agent email API is running"}

@app.post("/generate-reply", response_model=EmailResponse)
def generate_reply(req: EmailRequest):
    """
    This endpoint will be called (later) by Google Apps Script or any client.
    """
    email_text = f"From: {req.from_email}\nSubject: {req.subject}\n\n{req.body}"
    reply_text = generate_email_reply(email_text, req.user_name)
    return EmailResponse(reply=reply_text)
