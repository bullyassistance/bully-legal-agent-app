# bully_mailbot.py — Bully AI email agent (Gmail IMAP/SMTP)
# - Auto-reconnects on IMAP SSL/EOF drops
# - Skips Zapier (any @zapier.com), Trello, Jotform, DocuSign, etc.
# - Routes only real customer emails
# - Never forwards/answers charge-off or collections to Cat
# - Boss routing for product or “speak to Umar” requests
# - Pro-consumer federal playbook is enforced in policy_guard.apply_bully_policy()

from __future__ import annotations
import os, re, json, time, imaplib, email, smtplib, ssl
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Optional, List, Dict

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None

from policy_guard import apply_bully_policy
import requests

# =========================
# ENV / CONFIG
# =========================
load_dotenv()

IMAP_HOST   = os.getenv("IMAP_HOST", "imap.gmail.com")
IMAP_USER   = os.getenv("IMAP_USER")
IMAP_PASS   = os.getenv("IMAP_PASS")

SMTP_HOST   = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT   = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER   = os.getenv("SMTP_USER", IMAP_USER)
SMTP_PASS   = os.getenv("SMTP_PASS", IMAP_PASS)

SENDER_EMAIL = os.getenv("SENDER_EMAIL", IMAP_USER or "")
SENDER_NAME  = os.getenv("SENDER_NAME", "Bully AI")

DRY_RUN      = os.getenv("DRY_RUN", "0") != "0"
START_FROM   = os.getenv("START_FROM", "now").strip()  # "now", "yesterday", or integer days
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
MAX_PER_RUN  = int(os.getenv("MAX_PER_RUN", "20"))
MARKER_FILE  = os.getenv("MARKER_FILE", "mailbot_marker.json")

# LLM (xAI / Grok)
XAI_API_KEY  = os.getenv("XAI_API_KEY")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_MODEL    = os.getenv("XAI_MODEL", "grok-2")

# Forwarding
CAT_EMAIL    = os.getenv("CAT_EMAIL", "catrinabasanes@gmail.com")
BOSS_EMAIL   = os.getenv("BOSS_EMAIL", "info@clarktextile.com")

# Reconnect/backoff
IMAP_MAX_BACKOFF = int(os.getenv("IMAP_MAX_BACKOFF", "300"))  # 5 min cap

# =========================
# CONSTANTS / PATTERNS
# =========================

NOW_UTC = lambda: datetime.now(timezone.utc)

# Automated senders to ignore completely
SKIP_SENDER_PATTERNS = [
    r"(?i)@zapier\.com$",             # <— broadened: any address at zapier.com
    r"(?i)alerts\+noreply@mail\.zapier\.com",
    r"(?i)noreply@jotform(sign|\.com|\.us)?",
    r"(?i)onboarding-noreply@jotform\.us",
    r"(?i)support@jotform\.com",
    r"(?i)dse_.*@docusign\.net",
    r"(?i)orders@docusign\.net",
    r"(?i)do-not-reply@trello\.com",
    r"(?i)notifications@slack\.com",
    r"(?i)no-?reply@",
    r"(?i)mailer-daemon@",
    r"(?i)postmaster@",
    r"(?i)bounce@",
    r"(?i)mg\.msgsndr\.net",
    r"(?i)mailform\.io",
    r"(?i)stripe\.com",
]

SKIP_SUBJECT_PATTERNS = [
    r"(?i)security alert",
    r"(?i)verification code",
    r"(?i)order confirmation",
    r"(?i)receipt",
    r"(?i)welcome to",
    r"(?i)invoice",
    r"(?i)subscription",
    r"(?i)activation",
    r"(?i)overage|account review",  # vendor housekeeping (e.g., Zapier overages)
]

# Billing/CS keywords for Cat (do NOT include “charge” alone)
CAT_KEYWORDS = [
    r"(?i)\bcancel(lation)?\b",
    r"(?i)\brefund(s)?\b",
    r"(?i)\bsubscription(s)?\b|\bunsubscribe\b|\bsubscribe\b",
    r"(?i)\bbilling\b",
    r"(?i)\binvoice(s)?\b",
    r"(?i)\bchargeback(s)?\b",
    r"(?i)\bcharged?\s+(my|me|card|account)\b",
    r"(?i)\bcredit\s*card\s*charge(s)?\b",
    r"(?i)\bpayment(s)?\b",
]

# Guard: never send charge-off/collection to Cat
CHARGEOFF_GUARD = re.compile(r"(?i)\bcharge[- ]?off(s)?\b|chargeoff(s)?|collections?\b")

# Boss routing triggers (product, missed consults, or “speak to Umar”)
BOSS_KEYWORDS = [
    r"(?i)\bwhich product\b", r"(?i)\bwhat product\b", r"(?i)\bproduct to buy\b",
    r"(?i)\bmissed consult|\bmissed consultation\b",
    r"(?i)\bspeak (to|with) (umar|brother umar)\b", r"(?i)\bcan i (talk|speak) to (umar|brother umar)\b",
    r"(?i)\bask (umar|brother umar)\b",
    r"(?i)\bdoes the bully bundle\b", r"(?i)\bbully bundle\b",
]

OUR_ADDRESSES = [a.lower() for a in [SENDER_EMAIL, IMAP_USER] if a]

# =========================
# UTIL
# =========================
def log(*a): print(*a, flush=True)

def load_marker() -> Dict[str, int]:
    if os.path.exists(MARKER_FILE):
        try: return json.load(open(MARKER_FILE, "r"))
        except Exception: pass
    return {"min_uid": 0, "bootstrapped": False}

def save_marker(min_uid: int, bootstrapped: bool = True):
    json.dump({"min_uid": min_uid, "bootstrapped": bootstrapped}, open(MARKER_FILE, "w"))

def since_clause() -> Optional[str]:
    s = START_FROM.lower()
    if s == "now": return None
    if s == "yesterday":
        dt = NOW_UTC() - timedelta(days=1)
    else:
        try:
            days = int(s); dt = NOW_UTC() - timedelta(days=days)
        except Exception:
            return None
    return dt.strftime("%d-%b-%Y")

def is_automated(from_addr: str, subject: str) -> bool:
    for pat in SKIP_SENDER_PATTERNS:
        if re.search(pat, from_addr or ""): return True
    for pat in SKIP_SUBJECT_PATTERNS:
        if re.search(pat, subject or ""): return True
    return False

def needs_cat(body: str, subject: str) -> bool:
    text = f"{subject}\n{body}"
    if CHARGEOFF_GUARD.search(text):  # never Cat for charge-offs/collections
        return False
    for pat in CAT_KEYWORDS:
        if re.search(pat, text): return True
    return False

def needs_boss(body: str, subject: str) -> bool:
    text = f"{subject}\n{body}"
    for pat in BOSS_KEYWORDS:
        if re.search(pat, text): return True
    return False

def sanitize_subject(s: str) -> str:
    s = (s or "").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip() or "Re:"

def extract_name(addr: str, fallback="Friend") -> str:
    m = re.match(r'"?([^"]+)"?\s*<', addr or "")
    if m: return m.group(1).strip()
    m = re.match(r'([^@]+)@', addr or "")
    return (m.group(1).replace(".", " ").title() if m else fallback)

# =========================
# IMAP / SMTP
# =========================
def open_imap() -> imaplib.IMAP4_SSL:
    if not IMAP_USER or not IMAP_PASS:
        raise RuntimeError("IMAP_USER or IMAP_PASS missing in .env")
    context = ssl.create_default_context()
    M = imaplib.IMAP4_SSL(IMAP_HOST, ssl_context=context)
    M.login(IMAP_USER, IMAP_PASS)
    M.select("INBOX")
    return M

def reopen_imap_with_backoff(prev_backoff: int) -> tuple[imaplib.IMAP4_SSL, int]:
    backoff = min(max(5, prev_backoff * 2 if prev_backoff else 5), IMAP_MAX_BACKOFF)
    log(f"[IMAP] reconnecting in {backoff}s…")
    time.sleep(backoff)
    M = open_imap()
    log("[IMAP] reconnected.")
    return M, backoff

def open_smtp() -> smtplib.SMTP:
    s = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
    s.ehlo(); s.starttls(context=ssl.create_default_context())
    s.login(SMTP_USER, SMTP_PASS)
    return s

def send_email(to_addr: str, subject: str, body: str, reply_to: Optional[str] = None):
    if DRY_RUN:
        log("[DRY_RUN] Would send ->", to_addr, "|", subject); return
    msg = EmailMessage()
    msg["From"] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
    msg["To"] = to_addr
    msg["Subject"] = sanitize_subject(subject)
    if reply_to: msg["Reply-To"] = reply_to
    msg.set_content(body)
    with open_smtp() as s:
        s.send_message(msg)

def forward_email(forward_to: str, original_from: str, subject: str, body: str, note: str):
    fsub = f"[FORWARD] {sanitize_subject(subject)}"
    fbody = (
        f"{note}\n\nFrom: {original_from}\nSubject: {sanitize_subject(subject)}\n\n"
        "----- Message Start -----\n"
        f"{body}\n"
        "----- Message End -----\n"
    )
    send_email(forward_to, fsub, fbody, reply_to=SENDER_EMAIL)

# =========================
# LLM (Grok-2)
# =========================
SYSTEM_RULES = """
You are Bully AI, created and trained by Umar and The Bureau Bullies to empower consumers.
STRICT RULES:
- Never recommend pay-for-delete, goodwill letters, settlements, payment plans, or small-claims court.
- Always favor the federal path: FCRA/FDCPA, CRA Method of Verification (MOV), ACDV/e-OSCAR records,
  §623 direct disputes, tracking numbers/dates, and federal district court if needed.
- Be assertive, pro-consumer, and specific. No weak language like “maybe/you could”.
- Number the steps. Plain English. No markdown asterisks.
- Do not include any disclaimer or “I love myself”—the system appends it.
"""

def llm_answer(question: str, email_context: str) -> str:
    content = (
        f"User message:\n{question}\n\n"
        f"Email context:\n{email_context}\n"
        f"Respond with precise, assertive steps that follow the STRICT RULES.\n"
    )
    if not XAI_API_KEY:
        return ("Here’s the federal path: dispute under FCRA §611 (detailed items), request CRA Method of Verification "
                "(MOV) and the ACDV/e-OSCAR data, send a §623 direct dispute with exhibits to the furnisher, track all "
                "dates/tracking numbers, and escalate to federal court if reinvestigation is unreasonable.")
    try:
        url = f"{XAI_BASE_URL.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": XAI_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_RULES.strip()},
                {"role": "user", "content": content.strip()},
            ],
            "temperature": 0.2,
            "max_tokens": 900,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return (f"Follow the federal path: §611 dispute, MOV + ACDV/e-OSCAR request, §623 direct dispute to furnisher, "
                f"log dates/tracking, and file in federal court if necessary. (Model error: {e})")

# =========================
# FETCH & PROCESS
# =========================
def fetch_new_uids(M: imaplib.IMAP4_SSL, min_uid: int) -> List[int]:
    since = since_clause()
    crit = '(ALL)'
    if since:
        crit = f'(SINCE "{since}")'
    typ, data = M.uid("SEARCH", None, crit)
    if typ != "OK": return []
    all_uids = [int(u) for u in (data[0] or b"").split()]
    new_uids = [u for u in all_uids if u > min_uid]
    return new_uids[-MAX_PER_RUN:]

def parse_message(M: imaplib.IMAP4_SSL, uid: int):
    typ, data = M.uid("FETCH", str(uid), "(RFC822)")
    raw = data[0][1]
    msg = email.message_from_bytes(raw)
    from_addr = email.utils.parseaddr(msg.get("From", ""))[1]
    subject = msg.get("Subject", "") or "(no subject)"
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body += part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                except Exception:
                    pass
    else:
        try:
            body = msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="replace")
        except Exception:
            body = msg.get_payload()
    return from_addr or "", subject, body or ""

# =========================
# MAIN LOOP (with reconnect)
# =========================
def main():
    print("Bully MailBot starting…")
    print(f"[ENV] Using: | IMAP_USER={IMAP_USER or ''} | SENDER_EMAIL={SENDER_EMAIL or ''}")

    backoff = 0
    processed_this_run: set[int] = set()

    # Bootstrap marker
    min_uid = load_marker().get("min_uid", 0)
    bootstrapped = load_marker().get("bootstrapped", False)

    M = None
    while True:
        try:
            # Open or re-open IMAP
            if M is None:
                M = open_imap()
                # On first successful connect, bootstrap if needed
                if not bootstrapped:
                    typ, data = M.uid("SEARCH", None, "(ALL)")
                    if typ == "OK":
                        uids = [int(u) for u in (data[0] or b"").split()]
                        min_uid = max(uids) if uids else 0
                        save_marker(min_uid, True)
                        bootstrapped = True
                        print(f"[Marker] Bootstrapped min_uid={min_uid} (START_FROM={START_FROM}).")
                print(f"[LLM] Ready (model={XAI_MODEL})")
                print(f"Bully MailBot started. DRY_RUN = {DRY_RUN} | polling every {POLL_SECONDS} sec")

            # Select inbox (may throw on dropped session)
            M.select("INBOX")

            new_uids = fetch_new_uids(M, min_uid)
            highest_seen = min_uid

            for uid in new_uids:
                if uid in processed_this_run:
                    continue
                processed_this_run.add(uid)

                from_addr, subject, body = parse_message(M, uid)
                subject_clean = sanitize_subject(subject)

                # Skip ourselves and bots
                if (from_addr.lower() in OUR_ADDRESSES) or is_automated(from_addr, subject_clean):
                    continue

                # CAT?
                if needs_cat(body, subject_clean):
                    note = "Hello, Cat — this is Bully AI forwarding an email for you to review. Please respond to the customer directly from the company email."
                    forward_email(CAT_EMAIL, from_addr, subject_clean, body, note)
                    print(f"[MailBot] FORWARDED to Cat | {from_addr} | {subject_clean}")
                    highest_seen = max(highest_seen, uid)
                    continue

                # BOSS?
                if needs_boss(body, subject_clean):
                    note = "Hey Boss — Bully AI here. Please reach out to this customer directly."
                    forward_email(BOSS_EMAIL, from_addr, subject_clean, body, note)
                    ack = (
                        "Hello, I am Bully AI, an AI agent created and trained by Umar and The Bureau Bullies to empower consumers.\n\n"
                        f"{extract_name(from_addr)}, I forwarded your request to the boss so he can reach out directly.\n\n"
                        "I am Bully AI. I am not an attorney and cannot give legal advice. Please contact a licensed attorney in your state if you need an attorney.\n\n"
                        "I love myself."
                    )
                    send_email(from_addr, f"Re: {subject_clean}", ack, reply_to=SENDER_EMAIL)
                    print(f"[MailBot] FORWARDED to Boss | {from_addr} | {subject_clean}")
                    highest_seen = max(highest_seen, uid)
                    continue

                # Answer
                greeting = ("Hello, I am Bully AI, an AI agent created and trained by Umar and The Bureau Bullies "
                            "to empower consumers.")
                name = extract_name(from_addr, "Friend")
                question_text = f"Subject: {subject_clean}\n\n{body}"

                raw = llm_answer(question_text, email_context="")
                final = apply_bully_policy(user_name=name, raw_answer=f"{greeting}\n\n{raw}")

                send_email(from_addr, f"Re: {subject_clean}", final, reply_to=SENDER_EMAIL)
                print(f"[MailBot] SENT | {from_addr} | {subject_clean}")

                highest_seen = max(highest_seen, uid)

            if highest_seen > min_uid:
                min_uid = highest_seen
                save_marker(min_uid, True)

            backoff = 0  # healthy loop
            time.sleep(POLL_SECONDS)

        except (imaplib.IMAP4.abort, imaplib.IMAP4.error, ssl.SSLError, OSError) as e:
            print("[Loop error]", e)
            # force reconnect with backoff
            try:
                if M is not None:
                    try:
                        M.close()
                    except Exception:
                        pass
                    try:
                        M.logout()
                    except Exception:
                        pass
            finally:
                M = None
            M, backoff = reopen_imap_with_backoff(backoff)
            continue

        except KeyboardInterrupt:
            print("Stopping (Ctrl+C).")
            try:
                if M is not None:
                    M.close(); M.logout()
            except Exception:
                pass
            break

        except Exception as e:
            print("[Loop error]", e)
            time.sleep(min(POLL_SECONDS * 2, 120))
            continue

if __name__ == "__main__":
    main()

