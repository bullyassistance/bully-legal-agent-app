# policy_guard.py  — hard filters to prevent PFD/goodwill/settlement/small-claims
# Full drop-in file with a compatibility append_footer() wrapper.

from __future__ import annotations
import re
from textwrap import dedent

# ---- Footer content (shown once, de-duplicated) -----------------------------

DISCLAIMER_TEXT = (
    "I am Bully AI. I am not an attorney and cannot give legal advice. "
    "Please contact a licensed attorney in your state if you need an attorney."
)

SIGNOFF_TEXT = "I love myself."

# ---- Pro-consumer replacement block (used when bad advice is detected) ------

ANTI_PFD_BLOCK = dedent("""
    We don’t use “pay-for-delete,” goodwill letters, settlements, or payment plans. 
    Those weaken your position and are not guaranteed. Do this instead:

    1) Pull all three reports, mark each inaccurate or unverifiable item.
    2) Dispute with the CRA publishing it under FCRA § 611. Be specific about what’s wrong.
    3) If a CRA says “verified,” immediately request the CRA’s Method of Verification and the furnisher’s ACDV/e-OSCAR data
       (dispute code, fields furnished, documents reviewed, and investigation steps).
    4) Send a targeted direct dispute to the furnisher (FCRA § 623(a)(8)/(b)) with your facts and exhibits. 
       Ask for internal investigation notes/policies and the data they used to report.
    5) Track every date and USPS tracking number. Keep all letters, responses, and reports. 
       If reinvestigations are unreasonable or reporting remains inaccurate, you can sue in federal district court.
""").strip()

# ---- Helpers ----------------------------------------------------------------

ASTERS = re.compile(r"\*{1,3}")                    # strips ** and ***
WS_MULTI = re.compile(r"[ \t]+\n")                 # trim trailing spaces
BLANKS = re.compile(r"\n{3,}")                     # collapse blank lines
NEWLINES_AROUND = re.compile(r"[ \t]*\n[ \t]*")    # normalize linebreak spacing

BANNED_PATTERNS = re.compile(
    r"""(?ix)
        (pay\s*[- ]?for\s*[- ]?delete|
         pay\s*[- ]?to\s*[- ]?delete|
         pfd\b|
         good\s*[- ]?will(\s*letter|\s*adjustment)?|
         settle(?:ment)?\b|
         payment\s*plan\b|
         negotiate|negotiation|
         small\s*claims?\b|magistrate\s*court\b)
    """
)

WEAK_PHRASES = [
    (re.compile(r"(?i)\b(maybe|perhaps|you could|you might|try to|consider)\b"), " "),
    (re.compile(r"(?i)\b(negotiate|settle|payment plan)\b"), ""),
]

SMALL_CLAIMS = re.compile(r"(?i)\b(small\s*claims?|magistrate\s*court)\b")
FEDERALIZE_TO = "federal district court (FCRA/FDCPA as applicable)"

# ---- Transformations --------------------------------------------------------

def _strip_md(s: str) -> str:
    s = ASTERS.sub("", s)
    s = WS_MULTI.sub("\n", s)
    s = BLANKS.sub("\n\n", s)
    return s.strip()

def _remove_banned_and_rewrite(text: str) -> str:
    if BANNED_PATTERNS.search(text):
        text = BANNED_PATTERNS.sub("", text)
        text = f"{ANTI_PFD_BLOCK}\n\n{text}".strip()
    return text

def _strengthen(text: str) -> str:
    for pat, repl in WEAK_PHRASES:
        text = pat.sub(repl, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text

def _federalize(text: str) -> str:
    if SMALL_CLAIMS.search(text):
        text = SMALL_CLAIMS.sub(FEDERALIZE_TO, text)
    return text

def _dedupe_footer(text: str) -> str:
    text = re.sub(re.escape(DISCLAIMER_TEXT), "", text, flags=re.I)
    text = re.sub(r"\bI love myself\.?\b", "", text, flags=re.I)
    text = text.rstrip()
    footer = f"{DISCLAIMER_TEXT}\n\n{SIGNOFF_TEXT}"
    return f"{text}\n\n{footer}".strip()

def _add_empathy(name: str) -> str:
    return (f"{name}, I hear you and I know this is stressful. "
            f"I’m here to help you stay in control and win this the right way.")

# ---- Public API -------------------------------------------------------------

def apply_bully_policy(*, user_name: str, raw_answer: str) -> str:
    text = _strip_md(raw_answer or "")
    text = _remove_banned_and_rewrite(text)
    text = _strengthen(text)
    text = _federalize(text)

    empath = _add_empathy(user_name or "Friend")
    if empath.lower() not in text.lower():
        text = f"{empath}\n\n{text}".strip()

    text = _dedupe_footer(text)
    text = NEWLINES_AROUND.sub("\n", text).strip()
    return text

# ---- Backwards-compat shim (for older imports) ------------------------------

def append_footer(text: str) -> str:
    """Compatibility wrapper so older code doesn't crash."""
    text = _strip_md(text or "")
    return _dedupe_footer(text)
