# =======================================
# Bully AI — Litigation + Email Assistant
# =======================================
# Features
# - Accounts + Admin (SQLite + bcrypt)
# - Freeform Q&A (voice), “Hey Bully …” wake phrase
# - Complaint Builder (contract law options, damages calc, PDFs)
# - My Cases (saved PDFs, uploads, ask about the case)
# - Upload & Respond (quick drafting)
# - Email Assistant: IMAP fetch + SMTP send (Gmail/Outlook)
#
# Required installs (run once):
#   python3 -m pip install --upgrade streamlit python-dotenv requests ddgs PyPDF2 pymupdf \
#       reportlab pillow pytesseract SpeechRecognition pydub gTTS streamlit-mic-recorder imageio-ffmpeg \
#       pandas bcrypt
#
# Optional for OCR on macOS:
#   brew install tesseract
#
# Run:
#   python3 -m streamlit run legal_agent.py --server.port 8510
# ---------------------------------------

import os, io, re, json, csv, sqlite3, tempfile, textwrap, subprocess, imaplib, smtplib, ssl, email
import datetime as dt
from email.mime.multipart import MIMEMultipart
from email.mime.text     import MIMEText
from email.header        import decode_header, make_header
from typing import List, Dict, Tuple, Optional

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import requests
from ddgs import DDGS

import PyPDF2
try:
    import fitz  # pymupdf
except Exception:
    fitz = None

from PIL import Image
try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from streamlit_mic_recorder import mic_recorder
except Exception:
    mic_recorder = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

import imageio_ffmpeg
try:
    if AudioSegment:
        AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

try:
    from gtts import gTTS
except Exception:
    gTTS = None

import bcrypt
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# ---------- small helper to support both old/new Streamlit rerun names ----------
def re_run():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()  # fallback for older Streamlit
        except Exception:
            pass

# ----------------- ENV -------------------
load_dotenv()
XAI_API_KEY  = os.getenv("XAI_API_KEY","")
XAI_BASE_URL = os.getenv("XAI_BASE_URL","https://api.x.ai/v1")
XAI_MODEL    = os.getenv("XAI_MODEL","grok-2")

# Email (fetch)
EMAIL_MODE   = os.getenv("EMAIL_MODE","draft").lower()   # draft | imap
IMAP_HOST    = os.getenv("IMAP_HOST","")
IMAP_USER    = os.getenv("IMAP_USER","")
IMAP_PASS    = os.getenv("IMAP_PASS","")

# Email (send)
SMTP_HOST    = os.getenv("SMTP_HOST","")
SMTP_PORT    = int(os.getenv("SMTP_PORT","587"))
SMTP_USER    = os.getenv("SMTP_USER","")
SMTP_PASS    = os.getenv("SMTP_PASS","")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", SMTP_USER or IMAP_USER)
SENDER_NAME  = os.getenv("SENDER_NAME","Bully AI")

ADMIN_INVITE = os.getenv("ADMIN_INVITE_CODE","").strip()

APP_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(APP_DIR, "bully.db")
STORE_DIR = os.path.join(APP_DIR, "storage")
os.makedirs(STORE_DIR, exist_ok=True)

US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID","IL","IN","KS","KY","LA",
    "MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR",
    "PA","PR","RI","SC","SD","TN","TX","UT","VA","VT","WA","WI","WV","WY"
]
DATE_FORMATS = ["%Y-%m-%d","%m/%d/%Y","%m-%d-%Y","%b %d, %Y","%B %d, %Y"]

# -------------- helpers -------------------
def ddg_snippets(q: str, k: int=3) -> List[str]:
    try:
        with DDGS() as ddgs:
            out = ddgs.text(q, max_results=k)
            return [r.get("body","") for r in out if r.get("body")]
    except Exception:
        return []

def to_iso_date(s: str) -> str:
    if not s or not s.strip():
        return ""
    for fmt in DATE_FORMATS:
        try:  return dt.datetime.strptime(s.strip(), fmt).date().isoformat()
        except Exception:  continue
    return ""

def xai_chat(messages, max_tokens=1100, temperature=0.2) -> str:
    if not XAI_API_KEY:
        raise RuntimeError("Missing XAI_API_KEY")
    headers = {"Authorization": f"Bearer {XAI_API_KEY}"}
    payload = {"model": XAI_MODEL, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    r = requests.post(f"{XAI_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# Redaction
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_RE   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
ACCT_RE  = re.compile(r"\b\d{8,}\b")

def redact_text(s: str) -> str:
    s = EMAIL_RE.sub("[redacted-email]", s)
    s = PHONE_RE.sub("[redacted-phone]", s)
    s = SSN_RE.sub("[redacted-ssn]", s)
    s = ACCT_RE.sub(lambda m: "[acct-****" + m.group(0)[-4:] + "]", s)
    return s

# PDF / OCR
def extract_text_from_pdf(file) -> str:
    try:
        if fitz is not None:
            file.seek(0)
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return "\n".join([p.get_text() for p in doc])
    except Exception:
        pass
    try:
        file.seek(0)
        rd = PyPDF2.PdfReader(file)
        return "\n".join([(p.extract_text() or "") for p in rd.pages])
    except Exception as e:
        return f"[PDF error: {e}]"

def extract_text_from_image(file) -> str:
    if pytesseract is None:
        return "[OCR not available]"
    try:
        return pytesseract.image_to_string(Image.open(file))
    except Exception as e:
        return f"[Image error: {e}]"

# Voice
def _sniff_audio_format(b: bytes) -> str:
    if len(b)>=12 and b[:4]==b"RIFF" and b[8:12]==b"WAVE": return "wav"
    if b[:4]==b"OggS": return "ogg"
    if b[:4]==b"\x1aE\xdf\xa3": return "webm"
    return "unknown"

def _to_wav_bytes(audio_bytes: bytes) -> Optional[bytes]:
    if _sniff_audio_format(audio_bytes)=="wav":
        return audio_bytes
    if AudioSegment:
        try:
            fmt = _sniff_audio_format(audio_bytes)
            seg = (AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
                   if fmt in ("webm","ogg") else AudioSegment.from_file(io.BytesIO(audio_bytes)))
            out = io.BytesIO(); seg.export(out, format="wav")
            return out.getvalue()
        except Exception:
            pass
    try:
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in.bin")
            dst = os.path.join(td, "out.wav")
            open(src,"wb").write(audio_bytes)
            subprocess.run([ff,"-y","-i",src,"-ar","16000","-ac","1",dst],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return open(dst,"rb").read()
    except Exception:
        return None

def transcribe_audio_bytes(b: bytes) -> Optional[str]:
    if sr is None or not b or len(b)<500:
        return None
    wb = _to_wav_bytes(b)
    if not wb: return None
    try:
        r = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(wb)) as src:
            audio = r.record(src)
        return r.recognize_google(audio)
    except Exception:
        return None

VOICE_PRESETS = {
    "UK (Jarvis/Alfred-like)": {"lang":"en","tld":"co.uk"},
    "US (default)": {"lang":"en","tld":"com"},
}

def speak_to_mp3(text: str, preset="UK (Jarvis/Alfred-like)", slow=False) -> Optional[bytes]:
    if not text or gTTS is None:
        return None
    cfg = VOICE_PRESETS.get(preset, VOICE_PRESETS["UK (Jarvis/Alfred-like)"])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        gTTS(text, lang=cfg["lang"], tld=cfg["tld"], slow=slow).save(tmp.name)
        data = open(tmp.name,"rb").read()
    try: os.remove(tmp.name)
    except Exception: pass
    return data

# Wake phrase
WAKE_RE = re.compile(r'^\s*(hey\s*[,\s-]*bully)\b[:,\s-]*', re.IGNORECASE)
def strip_wake(txt: str) -> Tuple[str,bool]:
    if not txt: return "", False
    m = WAKE_RE.match(txt)
    return (txt[m.end():].strip(), True) if m else (txt.strip(), False)

# Empowerment filter
BANNED_SNIPPETS = [
    "settle", "settlement", "payment plan", "payment-plan", "lump sum", "goodwill letter",
    "goodwill", "pay for delete", "pay-for-delete", "negotiate a settlement", "offer a lump sum",
    "offer lump sum", "set up a payment plan", "payment arrangement"
]
def empower_and_sign(text: str) -> str:
    if not text:
        return "_I love myself._"
    keep, removed = [], False
    for ln in text.splitlines():
        low = ln.lower()
        if any(k in low for k in BANNED_SNIPPETS):
            removed = True
            continue
        keep.append(ln)
    if removed:
        keep.append("")
        keep.append("**Bully Policy:** We don’t recommend settlement, payment plans, goodwill, or pay-for-delete. "
                    "We focus on rights, evidence, and winning strategies.")
    keep.append("")
    keep.append("_I love myself._")
    return "\n".join(keep)

def inject_email_signature(text: str) -> str:
    sig = ("I am Bully AI I am not an attorney and cannot give legal advice. "
           "Please contact a licensed attorney in your state if you need an attorney.")
    if "_I love myself._" in text:
        return text.replace("_I love myself._", sig + "\n\n_I love myself._")
    return text + "\n\n" + sig + "\n\n_I love myself._"

# -------------- DB ------------------------
def db_conn():
    con = sqlite3.connect(DB_PATH); con.row_factory = sqlite3.Row
    return con

def init_db():
    con = db_conn(); cur = con.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        email TEXT UNIQUE,
        password_hash TEXT,
        name TEXT,
        phone TEXT,
        city TEXT,
        state TEXT,
        role TEXT DEFAULT 'user',
        status TEXT DEFAULT 'active'
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        created_at TEXT,
        title TEXT,
        state TEXT,
        furnisher TEXT,
        cras_json TEXT,
        reason TEXT,
        facts_count INTEGER,
        damages_total REAL,
        complaint_text TEXT,
        complaint_pdf_path TEXT,
        order_pdf_path TEXT,
        discovery_pdf_path TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id INTEGER,
        created_at TEXT,
        doc_type TEXT,
        filename TEXT,
        mimetype TEXT,
        path TEXT,
        text_excerpt TEXT,
        FOREIGN KEY(case_id) REFERENCES cases(id)
      )
    """)
    con.commit(); con.close()

def get_user_by_email(email_addr: str):
    con = db_conn()
    r = con.execute("SELECT * FROM users WHERE email=?", (email_addr,)).fetchone()
    con.close(); return r

def create_user(email_addr, password, name, phone, city, state, role):
    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    con = db_conn(); cur = con.cursor()
    cur.execute("""INSERT INTO users (created_at,email,password_hash,name,phone,city,state,role)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (dt.datetime.utcnow().isoformat(), email_addr.strip(), pw_hash, name.strip(),
                 phone.strip(), city.strip(), state.strip(), role))
    con.commit(); uid = cur.lastrowid; con.close(); return uid

def verify_password(pw: str, pw_hash: str) -> bool:
    try:    return bcrypt.checkpw(pw.encode("utf-8"), pw_hash.encode("utf-8"))
    except: return False

def list_users() -> pd.DataFrame:
    con = db_conn()
    df = pd.read_sql_query("SELECT id, created_at, email, name, phone, city, state, role, status FROM users", con)
    con.close(); return df

def set_user_role(user_id, role):
    con = db_conn(); con.execute("UPDATE users SET role=? WHERE id=?", (role, user_id)); con.commit(); con.close()

def set_user_status(user_id, status):
    con = db_conn(); con.execute("UPDATE users SET status=? WHERE id=?", (status, user_id)); con.commit(); con.close()

def create_case(user_id, title, state, furnisher, cras, reason, facts_count, damages_total,
                complaint_text, comp_pdf, order_pdf, disc_pdf) -> int:
    user_dir = os.path.join(STORE_DIR, str(user_id)); os.makedirs(user_dir, exist_ok=True)
    con = db_conn(); cur = con.cursor()
    cur.execute("""INSERT INTO cases (user_id, created_at, title, state, furnisher, cras_json, reason,
                                      facts_count, damages_total, complaint_text,
                                      complaint_pdf_path, order_pdf_path, discovery_pdf_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (user_id, dt.datetime.utcnow().isoformat(), title, state, furnisher, json.dumps(cras), reason,
                 facts_count, damages_total, complaint_text, "", "", ""))
    con.commit(); cid = cur.lastrowid
    case_dir = os.path.join(user_dir, str(cid)); os.makedirs(case_dir, exist_ok=True)
    comp_path = os.path.join(case_dir, "complaint.pdf"); open(comp_path,"wb").write(comp_pdf)
    order_path= os.path.join(case_dir, "proposed_order.pdf"); open(order_path,"wb").write(order_pdf)
    disc_path = os.path.join(case_dir, "discovery_pack.pdf"); open(disc_path,"wb").write(disc_pdf)
    cur.execute("""UPDATE cases SET complaint_pdf_path=?, order_pdf_path=?, discovery_pdf_path=? WHERE id=?""",
                (comp_path, order_path, disc_path, cid))
    con.commit(); con.close(); return cid

def list_cases(user_id) -> pd.DataFrame:
    con = db_conn()
    df = pd.read_sql_query(
        "SELECT id, created_at, title, state, furnisher, cras_json, reason, facts_count, damages_total FROM cases WHERE user_id=? ORDER BY id DESC",
        con, params=(user_id,))
    con.close(); return df

def get_case(case_id, user_id=None):
    con = db_conn()
    if user_id:
        row = con.execute("SELECT * FROM cases WHERE id=? AND user_id=?", (case_id, user_id)).fetchone()
    else:
        row = con.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    con.close(); return row

def add_document(case_id, doc_type, filename, mimetype, content, text_excerpt) -> int:
    case = get_case(case_id); user_id = case["user_id"]
    up_dir = os.path.join(STORE_DIR, str(user_id), str(case_id), "uploads"); os.makedirs(up_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename); path = os.path.join(up_dir, safe)
    open(path,"wb").write(content)
    con = db_conn(); cur = con.cursor()
    cur.execute("""INSERT INTO documents (case_id, created_at, doc_type, filename, mimetype, path, text_excerpt)
                   VALUES (?,?,?,?,?,?,?)""",
                (case_id, dt.datetime.utcnow().isoformat(), doc_type, safe, mimetype, path, (text_excerpt or "")[:4000]))
    con.commit(); did = cur.lastrowid; con.close(); return did

def list_documents(case_id) -> pd.DataFrame:
    con = db_conn()
    df = pd.read_sql_query("SELECT id, created_at, doc_type, filename, mimetype, path FROM documents WHERE case_id=? ORDER BY id DESC",
                           con, params=(case_id,))
    con.close(); return df

# -------------- Drafting systems --------------
FREEFORM_SYSTEM = (
    "You are Bully AI Legal Explainer. Your mission is pro se empowerment. "
    "Answer about U.S. consumer law, CONTRACT LAW (common law & UCC Article 2), and the FRCP. "
    "Tone: warm, validating, action-oriented. Never shame the user. "
    "POLICY: Do NOT recommend settlement, payment plans, goodwill letters, or pay-for-delete. "
    "Start with the plan, then the why."
)
COMPLAINT_SYSTEM = (
    "You are Bully AI Litigation Drafter. Draft a federal complaint that satisfies Twombly/Iqbal. "
    "Use ONLY the numbered facts provided; do not invent. "
    "Include Jurisdiction & Venue; Parties; FACTS; Counts; Prayer (Actual, Statutory, Punitive, Injunctive if applicable). "
    "If contract counts toggled, include breach elements; include UCC Art. 2 when goods are involved. "
    "Assertive tone that pressures settlement indirectly but do not suggest settlement."
)
RESPONDER_SYSTEM = (
    "You are Bully AI Litigation Strategist. Pro se empowerment only. "
    "POLICY: No settlement/payment plans/goodwill/pay-for-delete. "
    "Produce (A) next-steps plan, (B) a response draft with headings and authority, (C) exhibit checklist."
)
EMAIL_SYSTEM = (
    "You are Bully AI replying to client emails about credit/FCRA/FDCPA/lawsuits. "
    "Begin with: 'Hello, I am Bully AI, an AI agent created and trained by Umar and The Bureau Bullies to empower consumers.' "
    "If the sender asks to speak with Umar, include: 'Yes I will notify Umar.' "
    "If the email is about products/pricing/mentorship/consultation, instruct them to visit the website to browse products and/or book a consultation. "
    "Tone: warm, empathetic, empowering. "
    "POLICY: never recommend settlement, payment plans, goodwill, or pay-for-delete. "
    "Be concise, step-by-step, rights-focused."
)

# Numbered facts
def build_numbered_facts(intake: Dict, target_n: int) -> List[str]:
    facts = []
    V = intake.get("venue",{}); D = intake.get("defendants",{}); CRA = intake.get("cra_timeline",{})
    ACC = intake.get("accounts",[]); LET = intake.get("furnisher_letters",[])
    dmg = intake.get("damages",{}); C = intake.get("contract",{}) or {}

    if V.get("city") and V.get("state"):
        facts.append(f"Plaintiff resides in {V['city']}, {V['state']} and a substantial part of events occurred here.")
    if D.get("furnisher"):
        facts.append(f"{D['furnisher']} is a furnisher within 15 U.S.C. § 1681s-2.")
    for c in D.get("cras",[]):
        facts.append(f"{c} is a consumer reporting agency within 15 U.S.C. § 1681a(f).")

    for ln in ACC:
        p=[x.strip() for x in ln.split("|")]
        if len(p)>=2:
            cred, acct = p[0], p[1]
            issue = p[2] if len(p)>=3 else "inaccuracy"
            facts.append(f"Tradeline: {cred} account ending {acct[-3:] if acct else ''} shows alleged {issue}.")

    if CRA.get("first_dispute_date"):
        facts.append(f"On {CRA['first_dispute_date']}, Plaintiff disputed with the CRA(s) (tracking: {CRA.get('first_dispute_tracking') or 'n/a'}).")
    if CRA.get("mov_date"):
        facts.append(f"On {CRA['mov_date']}, Plaintiff requested Method of Verification (tracking: {CRA.get('mov_tracking') or 'n/a'}).")
    if CRA.get("acdv_date"):
        facts.append(f"On {CRA['acdv_date']}, Plaintiff requested the ACDV record (tracking: {CRA.get('acdv_tracking') or 'n/a'}).")
    if CRA.get("response_text") or CRA.get("response_date"):
        facts.append(f"On {CRA.get('response_date') or 'a later date'}, CRA response: {CRA.get('response_text') or 'no details'}.")

    for ln in LET:
        p=[x.strip() for x in ln.split("|")]
        if len(p)>=3:
            furn, d, trk = p[0], p[1], p[2]
            ask = p[3] if len(p)>=4 else "investigate and correct"
            resp= p[4] if len(p)>=5 else ""
            resp_d = p[5] if len(p)>=6 else ""
            facts.append(f"On {d or 'unknown date'}, Plaintiff sent demand to {furn} (tracking {trk}), requesting {ask}.")
            if resp or resp_d:
                facts.append(f"On {resp_d or 'later'}, {furn} responded: {resp or 'no details'}.")

    if C:
        if C.get("date"):
            facts.append(f"Parties formed a {C.get('type','contract')} on {C['date']}, governed by {C.get('governing_law','state')} law.")
        else:
            facts.append(f"Parties formed an enforceable {C.get('type','contract')} with offer, acceptance, and consideration.")
        if C.get("plaintiff_performed"): facts.append("Plaintiff performed or was ready, willing, and able to perform.")
        for t in C.get("terms",[]): facts.append(f"Key term: {t}.")
        for b in C.get("breaches",[]): facts.append(f"Defendant breached by: {b}.")
        if C.get("notice_date"): facts.append(f"Notice of breach sent on {C['notice_date']}.")
        if C.get("arbitration"): facts.append("Agreement contains an arbitration clause.")
        if C.get("attorney_fee"): facts.append("Agreement contains an attorney-fee clause.")

    facts.append("Plaintiff suffered concrete harms including time loss, emotional distress, and credit confusion.")
    if dmg.get("actual_total"):
        facts.append(f"Actual damages currently calculated at ${dmg['actual_total']:.2f} (may be updated).")

    stock = [
        "Defendant(s) received dispute notice via CRA reinvestigation process.",
        "Defendant(s) failed to reasonably investigate all pertinent information.",
        "Inaccurate information remained despite disputes and follow-up.",
        "Plaintiff mitigated harm by sending additional disputes and requests.",
    ]
    for s in stock:
        if len(facts)>=target_n: break
        facts.append(s)
    while len(facts)<target_n:
        facts.append("Reserved factual allegation pending initial disclosures and discovery.")
    return facts[:target_n]

def facts_section_text(facts: List[str]) -> str:
    return "\n".join([f"{i+1}. {f}" for i,f in enumerate(facts)])

# PDF helpers
def _wrap(text, width=92):
    out=[]
    for p in (text or "").split("\n"):
        out.extend(textwrap.wrap(p, width=width) if p.strip() else [""])
    return out

def _pdf(title: str, sections: List[Tuple[str,str]]) -> bytes:
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=LETTER)
    W,H = LETTER; m=0.85*inch; y=H-m
    def head(t):
        nonlocal y
        if y<1.2*inch: c.showPage(); y=H-m
        c.setFont("Times-Bold", 13); c.drawString(m,y,t); y-=0.22*inch
    def para(t):
        nonlocal y; c.setFont("Times-Roman", 11)
        for ln in _wrap(t, 95):
            if y<1.0*inch: c.showPage(); y=H-m
            c.drawString(m,y,ln); y-=0.18*inch
        y-=0.06*inch
    c.setTitle(title); c.setFont("Times-Bold",15); c.drawCentredString(W/2,y,title); y-=0.35*inch
    for h,b in sections: head(h); para(b)
    c.showPage(); c.save(); return buf.getvalue()

def make_complaint_pdf(intake: Dict, complaint_text: str, toa: Dict, damages: Dict) -> bytes:
    dsum = f"Actual base ${damages['actual_base']:.2f}; time value ${damages['time_value']:.2f}; costs ${damages['costs_total']:.2f}; interest ${damages['interest']:.2f}; total ${damages['actual_total']:.2f}."
    toa_text = ""
    for k in ["Cases","Statutes","Rules"]:
        vals = toa.get(k,[])
        if vals: toa_text += f"\n{k}:\n" + "\n".join(f"- {x}" for x in vals)
    sections=[("Caption", f"{intake['plaintiff']['name']} v. {intake['defendants'].get('furnisher') or ''} "
                          f"{', '.join(intake['defendants'].get('cras',[]))}\nCourt: {intake['venue']['court']} "
                          f"({intake['venue']['city']}, {intake['venue']['state']})"),
              ("Complaint", complaint_text),
              ("Damages Summary", dsum),
              ("Table of Authorities", toa_text or "None detected.")]
    return _pdf("Complaint", sections)

def make_order_pdf(intake: Dict) -> bytes:
    inj = intake["damages"].get("injunctive")
    inj_text = intake["damages"].get("inj_text","")
    body = "Upon consideration of the Complaint and applicable law, it is ORDERED:\n\n"
    if inj: body += f"1. Defendant(s) shall implement the following injunctive relief: {inj_text}.\n"
    body += "2. Defendant(s) shall correct all inaccurate credit reporting identified in the Complaint.\n"
    body += "3. The Court retains jurisdiction to enforce this Order."
    return _pdf("Proposed Order", [("Proposed Order", body)])

def make_discovery_pdf(intake: Dict) -> bytes:
    rogs = ["Identify each person who handled Plaintiff’s dispute(s) and describe their role.",
            "Describe policies and procedures used to investigate FCRA disputes.",
            "Explain each step taken in response to Plaintiff’s dispute(s), with dates."]
    rfp  = ["All ACDV/e-OSCAR records and dispute results for Plaintiff.",
            "All communications with any CRA concerning Plaintiff.",
            "All policies, manuals, and training for FCRA investigations."]
    rfa  = ["Admit the reported information was inaccurate.",
            "Admit no reasonable investigation was conducted."]
    body = "INTERROGATORIES:\n" + "\n".join([f"{i+1}. {q}" for i,q in enumerate(rogs)]) + \
           "\n\nREQUESTS FOR PRODUCTION:\n" + "\n".join([f"{i+1}. {q}" for i,q in enumerate(rfp)]) + \
           "\n\nREQUESTS FOR ADMISSION:\n" + "\n".join([f"{i+1}. {q}" for i,q in enumerate(rfa)])
    return _pdf("Discovery Pack", [("Discovery Requests", body)])

CASE_RE = re.compile(r"\b([A-Z][A-Za-z0-9.&'\- ]+ v\. [A-Z][A-Za-z0-9.&'\- ]+),?\s+\d{1,4}\s+[A-Z][A-Za-z.\d]*\s+\d{1,5}(?:\s*\(\d{4}\))?")
USC_RE  = re.compile(r"\b\d+\s+U\.S\.C\.?\s*§+\s*[\d\w\-]+")
FRCP_RE = re.compile(r"\b(?:Fed\.?|Federal)\s*R\.?\s*Civ\.?\s*P\.?\s*[\d\w().\-]+")

def table_of_authorities(text: str) -> Dict[str, List[str]]:
    return {
        "Cases": sorted(set(m.group(0) for m in CASE_RE.finditer(text))),
        "Statutes": sorted(set(m.group(0) for m in USC_RE.finditer(text))),
        "Rules": sorted(set(m.group(0) for m in FRCP_RE.finditer(text)))
    }

# -------------- Email Assistant helpers --------------
IGNORED_SENDERS = [
    "trello","mailform","wix","pnc","no-reply","noreply","do-not-reply","donotreply",
    "notifications","mailer-daemon","github","stripe","slack","calendar","noreplies"
]
CREDIT_KEYWORDS = [
    "credit","report","dispute","bureau","experian","equifax","transunion","fcra","fdcpa",
    "collector","collection","debt","charge off","charge-off","acdv","method of verification",
    "lawsuit","complaint","answer","motion","summons","service","validation","verification","tradeline"
]

def decode_hdr(s: str) -> str:
    try: return str(make_header(decode_header(s or "")))
    except Exception: return s or ""

def is_automated_sender(from_addr: str) -> bool:
    low = (from_addr or "").lower()
    return any(bad in low for bad in IGNORED_SENDERS)

def looks_client_credit(text: str) -> bool:
    low = (text or "").lower()
    return any(k in low for k in CREDIT_KEYWORDS)

def asks_for_umar(text: str) -> bool:
    return bool(re.search(r"\b(speak|talk|call|reach)\s+(with|to)\s*umar\b", (text or ""), re.I))

def imap_fetch_latest(max_n=10) -> List[Dict]:
    if EMAIL_MODE != "imap": return []
    out=[]
    try:
        M = imaplib.IMAP4_SSL(IMAP_HOST)
        M.login(IMAP_USER, IMAP_PASS)
        M.select("INBOX")
        typ, data = M.search(None, 'UNSEEN')
        ids = data[0].split()[-max_n:]
        for uid in reversed(ids):
            typ, msg_data = M.fetch(uid, '(RFC822)')
            if typ != 'OK' or not msg_data: continue
            msg = email.message_from_bytes(msg_data[0][1])
            from_addr = decode_hdr(msg.get("From"))
            subject   = decode_hdr(msg.get("Subject"))
            message_id= msg.get("Message-ID")
            refs      = msg.get("References")
            in_reply  = msg.get("Message-ID")

            body_txt = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    disp  = part.get("Content-Disposition","")
                    if ctype=="text/plain" and "attachment" not in (disp or "").lower():
                        try: body_txt = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                        except Exception: pass
                        break
            else:
                try: body_txt = msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="replace")
                except Exception: body_txt = msg.get_payload()

            snippet = (body_txt or "").strip().replace("\r"," ").replace("\n"," ")
            if len(snippet)>280: snippet = snippet[:280]+"…"
            out.append({
                "uid": uid.decode() if isinstance(uid,bytes) else str(uid),
                "from": from_addr, "subject": subject, "snippet": snippet, "body": body_txt,
                "message_id": message_id, "references": refs, "in_reply_to": in_reply
            })
        try: M.close()
        except Exception: pass
        M.logout()
    except Exception:
        return []
    return out

def draft_email_reply(orig_from: str, subject: str, body: str) -> str:
    ask_line = "Yes I will notify Umar.\n\n" if asks_for_umar(subject + " " + body) else ""
    prompt = (f"From: {orig_from}\nSubject: {subject}\n\nMessage:\n{body[:4000]}\n\n"
              "Write a concise, empathetic, step-by-step reply that empowers a pro se consumer. "
              "If it’s about products/pricing/consultation, direct them to the website.")
    raw = xai_chat(
        [{"role":"system","content": EMAIL_SYSTEM},
         {"role":"user","content": prompt}],
        max_tokens=600, temperature=0.2
    )
    filtered = empower_and_sign(ask_line + raw)
    signed   = inject_email_signature(filtered)
    return signed

def send_email(to_addr: str, subject: str, text_body: str, html_body: Optional[str]=None,
               in_reply_to: Optional[str]=None, references: Optional[str]=None) -> Tuple[bool,str]:
    """SMTP send (Gmail/Outlook). Returns (ok, message)."""
    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS and SENDER_EMAIL):
        return False, "SMTP not configured (.env)"
    msg = MIMEMultipart("alternative")
    msg["From"] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
    msg["To"]   = to_addr
    msg["Subject"] = subject
    if in_reply_to: msg["In-Reply-To"] = in_reply_to
    if references:  msg["References"]  = references
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    if html_body:
        msg.attach(MIMEText(html_body, "html", "utf-8"))
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=ctx)
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SENDER_EMAIL, [to_addr], msg.as_string())
        return True, "Sent."
    except Exception as e:
        return False, f"Send error: {e}"

# -------------- Streamlit UI -----------------
st.set_page_config(page_title="Bully AI — Litigation + Email", page_icon="⚖️", layout="wide")
init_db()

voice_preset = st.sidebar.selectbox("Voice preset", list(VOICE_PRESETS.keys()), index=0)
st.sidebar.caption("Info only. Not legal advice. No attorney-client relationship.")

# Auth
def login_screen():
    st.title("⚖️ Bully AI — Sign in")
    with st.form("login"):
        email_i = st.text_input("Email")
        pw      = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in")
    if ok:
        u = get_user_by_email(email_i.strip())
        if not u or u["status"]!="active" or not verify_password(pw, u["password_hash"]):
            st.error("Invalid credentials or inactive account.")
        else:
            st.session_state.user = dict(u); re_run()
    st.markdown("---")
    if st.button("Create an account"):
        st.session_state.view = "signup"; re_run()

def signup_screen():
    st.title("⚖️ Bully AI — Create account")
    with st.form("signup"):
        name  = st.text_input("Full name")
        email_i = st.text_input("Email")
        pw1   = st.text_input("Password", type="password")
        pw2   = st.text_input("Confirm password", type="password")
        phone = st.text_input("Mobile (optional)")
        colA,colB = st.columns(2)
        city  = colA.text_input("City")
        state = colB.selectbox("State", US_STATES, index=US_STATES.index("GA"))
        invite= st.text_input("Admin invite (optional)")
        ok = st.form_submit_button("Create account")
    if ok:
        if not (name.strip() and re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email_i.strip()) and pw1 and pw1==pw2 and city.strip() and state.strip()):
            st.error("Please fill name, valid email, matching passwords, city, and state.")
        elif get_user_by_email(email_i.strip()):
            st.error("This email is already registered.")
        else:
            con = db_conn(); n = con.execute("SELECT COUNT(*) n FROM users").fetchone()["n"]; con.close()
            role = "admin" if n==0 or (ADMIN_INVITE and invite.strip()==ADMIN_INVITE) else "user"
            create_user(email_i.strip(), pw1, name.strip(), phone.strip(), city.strip(), state.strip(), role)
            st.success(f"Account created. You are {role}. Please sign in.")
            st.session_state.view="login"; re_run()
    st.markdown("---")
    if st.button("Back to sign in"):
        st.session_state.view="login"; re_run()

def app_main(user: dict):
    st.sidebar.write(f"Signed in as **{user.get('name','')}** ({user.get('email','')}) — *{user.get('role','user')}*")
    if st.sidebar.button("Sign out"):
        st.session_state.pop("user", None); re_run()

    labels = ["Freeform Q&A", "New Complaint", "My Cases", "Upload & Respond", "Email Assistant (Send)"]
    if user.get("role")=="admin":
        labels.append("Admin")
    tabs = st.tabs(labels)

    # ----- Freeform Q&A -----
    with tabs[0]:
        st.header("Freeform Q&A — say 'Hey Bully, ...'")
        q = st.text_input("Ask a question", placeholder="Hey Bully, explain Federal Rule 3")
        mic_audio=None
        if mic_recorder:
            st.write("🎙️ Voice:")
            mic_audio = mic_recorder(start_prompt="Start", stop_prompt="Stop", just_once=True, use_container_width=True)
            if isinstance(mic_audio, dict) and mic_audio.get("bytes"):
                heard = transcribe_audio_bytes(mic_audio["bytes"])
                if heard:
                    st.info(f"Heard: {heard}")
                    if not q: q = heard
        if st.button("Answer", key="qa_btn"):
            q2,_ = strip_wake(q)
            if not q2: st.warning("Please enter a question.")
            else:
                web = ddg_snippets(q2, k=2)
                ctx = ("\n\nWeb:\n- " + "\n- ".join(web)) if web else ""
                ans = xai_chat([{"role":"system","content":FREEFORM_SYSTEM},
                                {"role":"user","content": q2 + ctx}], max_tokens=700)
                processed = empower_and_sign(ans)
                st.markdown("### Answer")
                st.write(processed)
                mp3 = speak_to_mp3(processed, preset=voice_preset)
                if mp3: st.audio(mp3, format="audio/mp3")

    # ----- New Complaint -----
    with tabs[1]:
        st.header("New Complaint (saved to your account)")
        with st.form("complaint_form"):
            st.subheader("Plaintiff")
            p_name  = st.text_input("Full Name", value=user.get("name",""))
            p_addr  = st.text_area("Address")
            p_email = st.text_input("Email", value=user.get("email",""))
            p_phone = st.text_input("Mobile (optional)", value=user.get("phone",""))
            st.subheader("Venue")
            colA,colB,colC = st.columns([1.2,0.8,1.2])
            city  = colA.text_input("City", value=user.get("city",""))
            state = colB.selectbox("State", US_STATES, index=US_STATES.index(user.get("state","GA") or "GA"))
            court = colC.text_input("Court (e.g., N.D. Ga.)")

            st.subheader("Defendants")
            sue_furnisher = st.checkbox("Include a Furnisher", True)
            furnisher_name= st.text_input("Furnisher name", "American Express")
            cras = st.multiselect("CRAs to sue", ["Experian","Equifax","TransUnion"])

            st.markdown("---")
            st.subheader("Accounts (one per line: Creditor | Account # | Issue)")
            furnisher_accounts = st.text_area("Furnisher Accounts", height=80)

            st.subheader("CRA Timeline")
            colD,colE = st.columns(2)
            first_dispute_date = colD.text_input("First dispute date (MM/DD/YYYY)")
            first_dispute_track= colE.text_input("First dispute tracking #")
            mov_date           = colD.text_input("MOV request date (optional)")
            mov_track          = colE.text_input("MOV tracking # (optional)")
            acdv_date          = colD.text_input("ACDV request date (optional)")
            acdv_track         = colE.text_input("ACDV tracking # (optional)")
            cra_response_text  = st.text_area("CRA response text")
            cra_response_date  = st.text_input("CRA response date (MM/DD/YYYY)")

            st.subheader("Demand letters to Furnisher (optional)")
            st.caption("One per line: Furnisher | Date | Tracking # | What you asked | Response | Response date")
            furnisher_letters = st.text_area("Furnisher Letters", height=90)

            st.subheader("Contract Law (optional)")
            c1,c2,c3 = st.columns(3)
            contract_type = c1.selectbox("Contract type", ["Credit Card","Loan","Services","Goods (UCC Article 2)","Other"], 0)
            contract_law  = c2.text_input("Governing law (state)", value=state)
            contract_date = c3.text_input("Date executed (MM/DD/YYYY)", value="")
            is_written    = st.checkbox("Written contract (Statute of Frauds satisfied)?", True)
            arbitration   = st.checkbox("Arbitration clause present")
            attorney_fee  = st.checkbox("Attorney-fee clause")
            plaintiff_performed = st.checkbox("Plaintiff performed or was ready/willing/able to perform", True)
            st.caption("Key terms (one per line)")
            contract_terms = st.text_area("Key terms", height=70)
            st.caption("Breaches (one per line)")
            contract_breaches = st.text_area("Breaches", height=70)
            n1,n2,n3 = st.columns(3)
            notice_date = n1.text_input("Notice of breach sent (MM/DD/YYYY)")
            cure_days   = n2.number_input("Cure period days", 0, 120, 0)
            cure_result = n3.selectbox("Cure result", ["No cure","Partial cure","Full cure","No cure period"], 0)

            st.subheader("Narrative & Counts")
            narrative = st.text_area("What happened?")
            target_facts = st.number_input("Target number of factual allegations", 10, 120, 33)
            colC1,colC2,colC3 = st.columns(3)
            count_1681i   = colC1.checkbox("§1681i (CRA reinvestigation)", True if cras else False)
            count_1681eb  = colC2.checkbox("§1681e(b) (CRA procedures)", False)
            count_contract = colC3.checkbox("Breach of Contract", False)

            st.subheader("Damages")
            colD1,colD2,colD3,colD4 = st.columns(4)
            actual_base = colD1.number_input("Actual damages (base $)", 0, 1_000_000, 0)
            hours       = colD2.number_input("Hours spent", 0, 10_000, 0)
            hourly      = colD3.number_input("Hourly rate ($/hr)", 0, 1_000, 0)
            int_rate    = colD4.number_input("Interest %", 0.0, 50.0, 0.0, step=0.1)
            harm_start  = st.text_input("Harm start (MM/DD/YYYY)")
            costs_text  = st.text_area("Costs (Item | Amount)", height=60)
            colE1,colE2,colE3 = st.columns(3)
            want_stat = colE1.checkbox("Statutory", True)
            want_pun  = colE2.checkbox("Punitive", True)
            want_inj  = colE3.checkbox("Injunctive", False)
            inj_text  = st.text_input("If injunctive, describe")

            st.subheader("Exhibits (optional)")
            exhibit_list = st.text_area("Exhibits list (Title | why it matters)")
            uploads = st.file_uploader("Upload exhibits (PDF/JPG/PNG)", type=["pdf","jpg","jpeg","png"], accept_multiple_files=True)
            auto_redact = st.checkbox("Auto-redact PII", True)

            ok = st.form_submit_button("📝 Draft & Save Case")

        if ok:
            if not (p_name.strip() and re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", p_email.strip()) and city.strip() and state.strip()):
                st.error("Plaintiff name + valid email, and venue city/state are required.")
            else:
                def parse_costs(s: str):
                    out=[]
                    for ln in (s or "").splitlines():
                        if "|" in ln:
                            k,a=[x.strip() for x in ln.split("|",1)]
                            try: out.append((k,float(a)))
                            except: pass
                    return out
                costs = parse_costs(costs_text)
                time_value = hours*hourly
                costs_total = sum(a for _,a in costs)
                principal = actual_base + time_value + costs_total
                def interest_amount(principal, rate, start_iso):
                    try:
                        if not principal or not rate or not start_iso: return 0.0
                        days = (dt.date.today() - dt.date.fromisoformat(start_iso)).days
                        return principal*(rate/100.0)*(max(0,days)/365.0)
                    except: return 0.0
                harm_iso = to_iso_date(harm_start)
                interest = round(interest_amount(principal, int_rate, harm_iso), 2)
                actual_total = round(principal + interest, 2)

                CRA = {
                    "first_dispute_date": to_iso_date(first_dispute_date),
                    "first_dispute_tracking": first_dispute_track.strip(),
                    "mov_date": to_iso_date(mov_date),
                    "mov_tracking": mov_track.strip(),
                    "acdv_date": to_iso_date(acdv_date),
                    "acdv_tracking": acdv_track.strip(),
                    "response_text": cra_response_text.strip(),
                    "response_date": to_iso_date(cra_response_date)
                }
                contract = {
                    "type": contract_type, "governing_law": (contract_law or "").strip(), "date": to_iso_date(contract_date),
                    "written": bool(is_written), "arbitration": bool(arbitration), "attorney_fee": bool(attorney_fee),
                    "plaintiff_performed": bool(plaintiff_performed),
                    "terms": [ln.strip() for ln in (contract_terms or "").splitlines() if ln.strip()],
                    "breaches": [ln.strip() for ln in (contract_breaches or "").splitlines() if ln.strip()],
                    "notice_date": to_iso_date(notice_date), "cure_days": int(cure_days), "cure_result": cure_result
                }
                intake = {
                    "plaintiff": {"name":p_name.strip(), "address":p_addr.strip(), "email":p_email.strip(), "phone": p_phone.strip()},
                    "venue": {"city":city.strip(), "state":state.strip(), "court":court.strip()},
                    "defendants": {"furnisher": furnisher_name.strip() if sue_furnisher else "", "cras": cras},
                    "accounts": [ln.strip() for ln in (furnisher_accounts or "").splitlines() if ln.strip()],
                    "cra_timeline": CRA,
                    "contract": contract,
                    "furnisher_letters": [ln.strip() for ln in (furnisher_letters or "").splitlines() if ln.strip()],
                    "facts": {"n_target": int(target_facts), "narrative": narrative.strip()},
                    "counts": {"1681i": bool(count_1681i),"1681e_b": bool(count_1681eb),"contract_breach": bool(count_contract)},
                    "damages": {"actual_base": actual_base, "time_value": time_value, "costs_list": costs,
                                "costs_total": costs_total, "interest": interest, "principal": round(principal,2),
                                "actual_total": actual_total, "statutory": bool(want_stat), "punitive": bool(want_pun),
                                "injunctive": bool(want_inj), "inj_text": inj_text.strip()},
                    "exhibits_list": exhibit_list.strip()
                }
                facts = build_numbered_facts(intake, intake["facts"]["n_target"])
                facts_text = facts_section_text(facts)

                with st.spinner("Drafting complaint…"):
                    complaint_text = xai_chat(
                        [{"role":"system","content": COMPLAINT_SYSTEM},
                         {"role":"user","content": "Use this immutable FACTS section.\n"
                                                   f"FACTS:\n{facts_text}\n\nINTAKE:\n{json.dumps(intake)}"}],
                        max_tokens=1800, temperature=0.15
                    )
                    toa = table_of_authorities(complaint_text)
                    comp_pdf = make_complaint_pdf(intake, complaint_text, toa, intake["damages"])
                    order_pdf= make_order_pdf(intake)
                    disc_pdf = make_discovery_pdf(intake)

                    title = f"{p_name} v. {(furnisher_name if sue_furnisher else '')} {' '.join(cras)}".strip()
                    cid = create_case(user["id"], title, state, furnisher_name if sue_furnisher else "", cras,
                                      intake["facts"]["narrative"][:120], len(facts), intake["damages"]["actual_total"],
                                      complaint_text, comp_pdf, order_pdf, disc_pdf)
                    for f in (uploads or []):
                        raw = f.getbuffer()
                        txt = extract_text_from_pdf(f) if f.name.lower().endswith(".pdf") else extract_text_from_image(f)
                        if auto_redact: txt = redact_text(txt)
                        add_document(cid, "exhibit", f.name, f.type, raw, txt)

                    st.success(f"Case saved (ID #{cid}). See 'My Cases'.")
                    st.session_state["last_results"]={"comp_pdf":comp_pdf,"order_pdf":order_pdf,"disc_pdf":disc_pdf}
        res = st.session_state.get("last_results")
        if res:
            st.markdown("### Quick Download")
            st.download_button("⬇️ Complaint (PDF)", res["comp_pdf"], "complaint.pdf", mime="application/pdf")
            st.download_button("⬇️ Proposed Order (PDF)", res["order_pdf"], "proposed_order.pdf", mime="application/pdf")
            st.download_button("⬇️ Discovery Pack (PDF)", res["disc_pdf"], "discovery_pack.pdf", mime="application/pdf")

    # ----- My Cases -----
    with tabs[2]:
        st.header("My Cases")
        df = list_cases(user["id"])
        if df.empty:
            st.info("No cases yet.")
        else:
            st.dataframe(df)
            case_id = st.number_input("Open case ID", min_value=int(df["id"].min()), max_value=int(df["id"].max()))
            if st.button("Open"):
                case = get_case(int(case_id), user_id=user["id"])
                if not case: st.error("Case not found.")
                else:
                    st.subheader(f"Case #{case['id']}: {case['title']}")
                    col1,col2,col3 = st.columns(3)
                    if os.path.exists(case["complaint_pdf_path"]):
                        col1.download_button("⬇️ Complaint", open(case["complaint_pdf_path"],"rb").read(), "complaint.pdf")
                    if os.path.exists(case["order_pdf_path"]):
                        col2.download_button("⬇️ Proposed Order", open(case["order_pdf_path"],"rb").read(), "proposed_order.pdf")
                    if os.path.exists(case["discovery_pdf_path"]):
                        col3.download_button("⬇️ Discovery Pack", open(case["discovery_pdf_path"],"rb").read(), "discovery_pack.pdf")
                    st.markdown("---")
                    st.subheader("Upload answers/motions")
                    up = st.file_uploader("Upload (PDF/JPG/PNG)", type=["pdf","jpg","jpeg","png"], accept_multiple_files=True, key="caseup")
                    dtype = st.selectbox("Type", ["answer","motion","letter","exhibit","other"])
                    if st.button("Save uploads to case"):
                        if not up: st.warning("No files.")
                        else:
                            n=0
                            for f in up:
                                raw = f.getbuffer()
                                txt = extract_text_from_pdf(f) if f.name.lower().endswith(".pdf") else extract_text_from_image(f)
                                add_document(case["id"], dtype, f.name, f.type, raw, txt); n+=1
                            st.success(f"Saved {n} doc(s).")
                    st.markdown("---")
                    st.subheader("Ask Bully about THIS case")
                    q_case = st.text_area("Question")
                    if st.button("Answer using case context"):
                        ctx = (case["complaint_text"] or "")[:6000]
                        docs_df = list_documents(case["id"])
                        if not docs_df.empty:
                            con = db_conn(); rows = con.execute("SELECT text_excerpt FROM documents WHERE case_id=? ORDER BY id DESC LIMIT 6",(case["id"],)).fetchall(); con.close()
                            for r in rows: ctx += "\n\n[Doc excerpt]\n" + (r["text_excerpt"] or "")[:800]
                        ans = xai_chat(
                            [{"role":"system","content":"You are Bully AI. Use provided case context; no fabricated facts. POLICY: no settlement/payment plans/goodwill/pay-for-delete."},
                             {"role":"user","content": f"Case context (excerpted):\n{ctx}\n\nQuestion:\n{q_case}"}],
                            max_tokens=900, temperature=0.15
                        )
                        st.write(empower_and_sign(ans))

    # ----- Upload & Respond -----
    with tabs[3]:
        st.header("Upload & Respond (quick)")
        opp = st.file_uploader("Upload opposing filing (PDF/JPG/PNG)", type=["pdf","jpg","jpeg","png"], accept_multiple_files=True)
        ctx = st.text_area("Goal / posture")
        if st.button("Draft Response"):
            if not opp: st.warning("Please upload at least one file.")
            else:
                body=""
                for f in opp:
                    body += "\n\n" + (extract_text_from_pdf(f) if f.name.lower().endswith(".pdf") else extract_text_from_image(f))
                plan = xai_chat([{"role":"system","content": RESPONDER_SYSTEM},
                                 {"role":"user","content": f"Opponent text:\n{body[:9000]}\n\nContext:\n{ctx}"}],
                                max_tokens=1200)
                processed = empower_and_sign(plan)
                st.write(processed)
                resp_pdf = _pdf("Response Draft", [("Draft", processed)])
                st.download_button("⬇️ Response (PDF)", resp_pdf, "response_draft.pdf", mime="application/pdf")

    # ----- Email Assistant (Send) -----
    with tabs[4]:
        st.header("Email Assistant (IMAP fetch + SMTP send)")
        mode = st.radio("Mode", ["Draft-only (paste email)","IMAP (fetch unseen)"], index=1 if EMAIL_MODE=="imap" else 0)

        if mode.startswith("IMAP"):
            st.write(f"IMAP host: `{IMAP_HOST or '—'}` | user: `{IMAP_USER or '—'}`")
            st.write(f"SMTP host: `{SMTP_HOST or '—'}` | user: `{SMTP_USER or '—'}` | sender: `{SENDER_EMAIL or '—'}`")
            if st.button("Fetch latest (unseen)"):
                msgs = imap_fetch_latest(12)
                if not msgs: st.info("No unseen or IMAP not configured.")
                for m in msgs:
                    if is_automated_sender(m["from"]) or not looks_client_credit(m["subject"]+" "+m["body"]):
                        continue
                    with st.expander(f"From: {m['from']} | Subject: {m['subject']}"):
                        st.write(m["snippet"] or "")
                        if st.button("Draft reply", key=f"draft_{m['uid']}"):
                            draft = draft_email_reply(m["from"], m["subject"], m["body"])
                            st.code(draft)
                            to_addr = re.findall(r"<([^>]+)>", m["from"]) or [m["from"]]
                            to_addr = to_addr[0]
                            if st.button("Send reply", key=f"send_{m['uid']}"):
                                ok,msg = send_email(
                                    to_addr=to_addr,
                                    subject="Re: " + (m["subject"] or ""),
                                    text_body=draft,
                                    html_body=None,
                                    in_reply_to=m.get("message_id"),
                                    references=m.get("references")
                                )
                                st.success("Sent.") if ok else st.error(msg)
        else:
            st.write("Paste any client email to draft + send a reply.")
            p_from = st.text_input("From (name <email>)", placeholder="Jane Doe <jane@example.com>")
            p_to   = st.text_input("Send reply to (email)", placeholder="jane@example.com")
            p_subj = st.text_input("Subject", placeholder="Question about a dispute result")
            p_body = st.text_area("Body", height=220, placeholder="Paste their message here")
            if st.button("Draft reply (local)"):
                if is_automated_sender(p_from) or not looks_client_credit(p_subj+" "+p_body):
                    st.warning("This doesn’t look like a client credit/FDCPA/FCRA email, or it’s automated.")
                else:
                    draft = draft_email_reply(p_from, p_subj, p_body)
                    st.code(draft)
                    if st.button("Send now"):
                        ok,msg = send_email(to_addr=p_to or re.findall(r"<([^>]+)>",p_from)[0],
                                            subject="Re: "+(p_subj or ""),
                                            text_body=draft, html_body=None)
                        st.success("Sent.") if ok else st.error(msg)

    # ----- Admin -----
    if user.get("role")=="admin":
        with tabs[-1]:
            st.header("Admin")
            st.subheader("Users")
            st.dataframe(list_users())
            colA,colB,colC = st.columns(3)
            uid_role = colA.number_input("User ID (role)", min_value=0, step=1)
            new_role = colB.selectbox("New role", ["user","admin"])
            if colC.button("Update Role"): set_user_role(int(uid_role), new_role); st.success("Role updated.")
            colD,colE,colF = st.columns(3)
            uid_stat = colD.number_input("User ID (status)", min_value=0, step=1, key="uidstat")
            new_stat= colE.selectbox("New status", ["active","disabled"])
            if colF.button("Update Status"): set_user_status(int(uid_stat), new_stat); st.success("Status updated.")
            st.markdown("---")
            st.subheader("Monthly Analytics")
            con = db_conn()
            dfc = pd.read_sql_query("SELECT id, created_at, state, furnisher, cras_json, reason FROM cases", con); con.close()
            if dfc.empty: st.info("No cases yet.")
            else:
                dfc["month"]=dfc["created_at"].str.slice(0,7)
                month = st.selectbox("Month", sorted(dfc["month"].unique()))
                filt = dfc[dfc["month"]==month]
                cr = filt.copy(); cr["cras_json"]=cr["cras_json"].fillna("[]").apply(json.loads)
                cr = cr.explode("cras_json").rename(columns={"cras_json":"defendant"})
                furn=filt["furnisher"].dropna()
                defendants = pd.concat([furn.rename("defendant"), cr["defendant"].dropna()])
                top_def = defendants.value_counts().reset_index(); top_def.columns=["defendant","cases"]
                st.write("**Top defendants**"); st.dataframe(top_def)
                top_states = filt["state"].value_counts().reset_index(); top_states.columns=["state","cases"]
                st.write("**Top states**"); st.dataframe(top_states)
                top_reasons = filt["reason"].value_counts().reset_index(); top_reasons.columns=["reason","cases"]
                st.write("**Top reasons**"); st.dataframe(top_reasons)
                csv_buf = io.StringIO(); w=csv.writer(csv_buf); w.writerow(["table","key","cases"])
                for _,r in top_def.iterrows(): w.writerow(["defendant", r["defendant"], r["cases"]])
                for _,r in top_states.iterrows(): w.writerow(["state", r["state"], r["cases"]])
                for _,r in top_reasons.iterrows(): w.writerow(["reason", r["reason"], r["cases"]])
                st.download_button("⬇️ Download CSV", csv_buf.getvalue().encode("utf-8"),
                                   file_name=f"analytics-{month}.csv", mime="text/csv")

# ---- entry ----
if "view" not in st.session_state:
    st.session_state.view="login"
if "user" in st.session_state and st.session_state.user:
    app_main(st.session_state.user)
else:
    if st.session_state.view=="signup":
        signup_screen()
    else:
        login_screen()
