# email_api.py

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

# Import your REAL reply engine
from bully_core import generate_email_reply   # <-- THIS MUST MATCH YOUR FILE NAME

app = FastAPI()

class EmailRequest(BaseModel):
    from_email: str
    subject: str
    body: str
    user_name: Optional[str] = None

class EmailResponse(BaseModel):
    reply: str


@app.get("/")
def health():
    return {"status": "ok", "message": "Bully Email API running"}


@app.post("/generate-reply", response_model=EmailResponse)
def generate_reply(req: EmailRequest):
    # CALL YOUR REAL CREDIT AI LOGIC
    reply_text = generate_email_reply(
        orig_from=req.from_email,
        subject=req.subject,
        body=req.body,
        user_name=req.user_name
    )
    
    return EmailResponse(reply=reply_text)
