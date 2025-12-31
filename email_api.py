from fastapi import FastAPI
from pydantic import BaseModel
from bully_core import generate_email_reply  # or whatever path you used

app = FastAPI()

class EmailRequest(BaseModel):
    from_email: str
    subject: str
    body: str
    user_name: str | None = None

class EmailResponse(BaseModel):
    reply: str

@app.post("/generate-reply", response_model=EmailResponse)
def generate_reply(req: EmailRequest) -> EmailResponse:
    reply_text = generate_email_reply(
        orig_from=req.from_email,
        subject=req.subject,
        body=req.body,
        user_name=req.user_name,
    )
    return EmailResponse(reply=reply_text)
