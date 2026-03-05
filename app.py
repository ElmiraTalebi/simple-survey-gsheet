import hashlib
import json
from datetime import datetime
from typing import Dict, List, Set, Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI

st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ── Secrets ────────────────────────────────────────────────
def _secret(*keys, default=None):
    for k in keys:
        if k in st.secrets: return st.secrets[k]
    return default

def _require_secret(*keys):
    v = _secret(*keys)
    if v is None: raise KeyError(f"Missing secret. Tried: {', '.join(keys)}")
    return v

# ── OpenAI ─────────────────────────────────────────────────
OPENAI_API_KEY = _secret("openai_api_key", "OPENAI_API_KEY", "openai_key")
openai_client: Optional[OpenAI] = None
openai_init_error: Optional[str] = None
if OPENAI_API_KEY:
    try: openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e: openai_init_error = str(e)
else: openai_init_error = "OpenAI API key not found."

# ── Google Sheets ───────────────────────────────────────────
sheet = None
sheets_init_error: Optional[str] = None

def _init_sheets():
    global sheet, sheets_init_error
    if sheet is not None or sheets_init_error is not None: return
    try:
        creds = Credentials.from_service_account_info(
            _require_secret("gcp_service_account"),
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        book = gspread.authorize(creds).open_by_key(_require_secret("gsheet_id"))
        try: sheet_local = book.worksheet("Form")
        except Exception:
            sheet_local = book.add_worksheet(title="Form", rows=2000, cols=20)
            sheet_local.append_row(["timestamp", "name", "json"])
        sheet = sheet_local
    except Exception as e: sheets_init_error = str(e)

def load_past_checkins(name: str) -> List[Dict]:
    _init_sheets()
    if sheet is None: return []
    try:
        past = []
        for row in sheet.get_all_values()[1:]:
            if len(row) >= 3 and row[1].strip().lower() == name.strip().lower():
                try:
                    d = json.loads(row[2]); d["timestamp"] = row[0]; past.append(d)
                except: continue
        return past[-5:]
    except: return []

def save_to_sheet():
    _init_sheets()
    if sheet is None: raise RuntimeError(f"Sheets unavailable: {sheets_init_error}")
    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        st.session_state.get("patient_name", "Unknown"),
        json.dumps({
            "feeling_level":    st.session_state.feeling_level,
            "pain":             st.session_state.pain_yesno,
            "pain_locations":   sorted(list(st.session_state.selected_parts)),
            "pain_severities":  st.session_state.get("pain_severities", {}),
            "pain_timing":      st.session_state.get("pain_timing"),
            "eating":           st.session_state.get("eating_status"),
            "food_type":        st.session_state.get("food_type"),
            "shakes_per_day":   st.session_state.get("shakes_per_day"),
            "hydration":        st.session_state.get("hydration"),
            "symptoms":         st.session_state.symptoms,
            "conversation":     st.session_state.messages,
            "fast_path":        st.session_state.get("fast_path", False),
        })
    ])

# ── Helpers ─────────────────────────────────────────────────
def _openai_ready():
    return openai_client is not None and openai_init_error is None

def get_opening_message(last: Dict, name: str) -> str:
    """
    Greeting that references the SPECIFIC most concerning item from last visit,
    then asks ONE targeted yes/no or scale question about that exact thing.
    This replaces the generic 'Compared to your last visit, how are things?' panel title.
    """
    if not _openai_ready():
        # Fallback: build a specific question from data without GPT
        syms  = last.get("symptoms", [])
        locs  = last.get("pain_locations", [])
        sevs  = last.get("pain_severities", {})
        if syms:
            top = syms[0]
            return f"Hi {name}! Last time you reported {top.lower()}. Is that still bothering you, or has it improved?"
        if locs:
            top = locs[0]
            sev = sevs.get(top)
            sev_str = f" (severity {sev}/10)" if sev else ""
            return f"Hi {name}! Last time you had pain in your {top.lower()}{sev_str}. How is that today — better, worse, or about the same?"
        return f"Hi {name}! How have things been since your last visit — better, worse, or about the same?"

    fl   = last.get("feeling_level", "?")
    pn   = "yes" if last.get("pain") else "no"
    ploc = ", ".join(last.get("pain_locations", [])) or "none"
    sym  = ", ".join(last.get("symptoms", [])) or "none"
    sevs = last.get("pain_severities", {})
    sev_str = ", ".join(f"{k}: {v}/10" for k, v in sevs.items()) if sevs else "not recorded"
    ts   = last.get("timestamp", "your last visit")

    system = (
        "You are a concise symptom-intake assistant for a cancer care clinic treating head/neck cancer patients. "
        "Write exactly 2 sentences:\n"
        "Sentence 1: Greet the patient by first name only. Reference the MOST SPECIFIC and MOST CONCERNING "
        "item from their last visit — name the actual symptom, pain location, or severity score. "
        "Do NOT be generic (e.g. do NOT say 'some concerns' or 'your symptoms'). "
        "Sentence 2: Ask ONE short, direct follow-up question about that specific item — "
        "e.g. 'Is your throat pain better, worse, or about the same?' or "
        "'Last time your nausea was quite bad — are you still experiencing it?' "
        "The question must be answerable with the three buttons: Better / About the same / Worse. "
        "No filler. No 'Let's get started'. No 'How are you today?'. Just greet + targeted question."
    )
    try:
        r = openai_client.chat.completions.create(
            model=_secret("openai_model", default="gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": (
                    f"Patient first name: {name}. Last visit: {ts}. "
                    f"Feeling: {fl}. Pain: {pn}. Pain locations: {ploc}. "
                    f"Pain severities: {sev_str}. Symptoms: {sym}."
                )}
            ],
            max_tokens=100, temperature=0.4,
        )
        return (r.choices[0].message.content or "").strip()
    except:
        syms = last.get("symptoms", [])
        locs = last.get("pain_locations", [])
        if syms:
            return f"Hi {name}! Last time you reported {syms[0].lower()}. Is that still bothering you, or has it improved?"
        if locs:
            return f"Hi {name}! Last time you had pain in your {locs[0].lower()}. How is that today — better, worse, or about the same?"
        return f"Hi {name}! How have things been since your last visit — better, worse, or about the same?"

def transcribe_audio(audio_bytes: bytes) -> str:
    if not _openai_ready(): return "(Transcription unavailable.)"
    try:
        import io
        f = io.BytesIO(audio_bytes); f.name = "recording.wav"
        return (openai_client.audio.transcriptions.create(
            model=_secret("whisper_model", default="whisper-1"), file=f, language="en"
        ).text or "").strip()
    except Exception as e: return f"(Transcription failed: {e})"

# ── Curated Follow-Up Questions ────────────────────────────
# Derived from actual clinician-patient transcripts (Phase I Patients 1-5).
# Follow-ups fire ONLY when concerning — "minimize time, only drill when needed."

FEELING_FOLLOWUPS = {
    "fair": "What's been the hardest part for you lately?",
    "poor": "What is bothering you the most right now?",
}

PAIN_DRILL = {
    "severity_high": "Is this pain constant, or mainly when you swallow or eat?",
    "new_location":  "When did you first notice pain in this new area?",
    "worsening":     "When did it start getting worse?",
}

EATING_DRILL = {
    "worse":    "What's making it harder to eat — pain, nausea, or no appetite?",
    "cant_eat": "Are you able to get any fluids or shakes down at all?",
}

SYMPTOM_FOLLOWUPS = {
    "Nausea":                   "Are you able to keep food and fluids down?",
    "Vomiting":                 "How many times in the last 24 hours?",
    "Mouth sores":              "Are the sores making it hard to eat or drink?",
    "Trouble swallowing":       "Are you coughing or choking when you swallow?",
    "Fever / chills":           "Have you taken your temperature? What was it?",
    "Constipation":             "How many days since your last bowel movement?",
    "Diarrhea":                 "How many episodes today?",
    "Dizziness / unsteadiness": "Does it happen when you change position (sitting to standing)?",
    "Shortness of breath":      "Does it happen at rest or only with activity?",
}

def get_curated_followup(stage_id: int) -> Optional[str]:
    """Return ONE curated follow-up ONLY if the answer is concerning."""
    if stage_id == 1:
        feeling = st.session_state.get("feeling_level")
        return FEELING_FOLLOWUPS.get(feeling)  # None for excellent/good/very good

    if stage_id == 2:
        if not st.session_state.pain_yesno:
            return None
        # Check for high severity or new locations
        sevs = st.session_state.get("pain_severities", {})
        past = st.session_state.get("past_checkins", [])
        prev_locs = set(past[-1].get("pain_locations", [])) if past else set()
        curr_locs = st.session_state.get("selected_parts", set())
        new_locs = curr_locs - prev_locs

        high = [l for l, s in sevs.items() if s >= 5]
        if high:
            return PAIN_DRILL["severity_high"]
        if new_locs:
            return PAIN_DRILL["new_location"]
        return None

    if stage_id == 3:  # eating
        status = st.session_state.get("eating_status")
        if status in ("worse", "cant_eat"):
            return EATING_DRILL.get(status)
        return None

    if stage_id == 4:  # symptoms
        symptoms = st.session_state.get("symptoms", [])
        # Priority: concerning symptoms that need immediate drill-down
        priority = ["Fever / chills", "Vomiting", "Trouble swallowing",
                     "Shortness of breath", "Dizziness / unsteadiness"]
        for s in priority:
            if s in symptoms and s in SYMPTOM_FOLLOWUPS:
                return SYMPTOM_FOLLOWUPS[s]
        # Check for new symptoms vs last visit
        past = st.session_state.get("past_checkins", [])
        prev = set(past[-1].get("symptoms", [])) if past else set()
        new_syms = set(symptoms) - prev
        for s in new_syms:
            if s in SYMPTOM_FOLLOWUPS:
                return SYMPTOM_FOLLOWUPS[s]
        return None

    return None

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

:root {
    --bg:        #f5f3ef;
    --surface:   #ffffff;
    --border:    rgba(180,170,155,0.30);
    --accent:    #2a9d8f;
    --accent-lt: rgba(42,157,143,0.09);
    --accent-md: rgba(42,157,143,0.22);
    --patient:   #264653;
    --text:      #2c2c2c;
    --muted:     #8a8075;
    --shadow-sm: 0 2px 10px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 24px rgba(0,0,0,0.09);
    --r-sm: 12px; --r-md: 18px; --r-lg: 24px;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Nunito', sans-serif !important;
    background: var(--bg) !important;
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; z-index: -1;
    background:
        radial-gradient(ellipse 70% 50% at 88% 8%, rgba(42,157,143,0.12) 0%, transparent 65%),
        radial-gradient(ellipse 55% 40% at 8% 92%, rgba(38,70,83,0.08) 0%, transparent 65%),
        #f5f3ef;
}
[data-testid="stHeader"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stToolbar"] {
    display: none !important; height: 0 !important;
}
[data-testid="stMainBlockContainer"] { padding-top: 1rem !important; }
.block-container { max-width: 680px !important; padding: 0 1.2rem 3rem !important; }
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stAppDeployButton"], .stDeployButton,
.stApp > header, iframe[title="streamlitApp"],
[data-testid="manage-app-button"], [data-testid="stRunningManWidget"],
[data-testid="collapsedControl"], [data-testid="stAppIframe"],
div:has(> iframe[title="streamlitApp"]) {
    display: none !important; height: 0 !important;
    max-height: 0 !important; overflow: hidden !important;
}
.stApp > div:first-child:not(:has([data-testid="stMainBlockContainer"])) {
    display: none !important; height: 0 !important;
}

.app-header {
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 26px; padding-bottom: 18px;
    border-bottom: 1.5px solid var(--border);
}
.app-header-icon {
    width: 46px; height: 46px; border-radius: 13px;
    background: linear-gradient(135deg, var(--accent) 0%, #1d7a6e 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 21px; box-shadow: 0 3px 12px rgba(42,157,143,0.30);
    flex-shrink: 0;
}
.app-header-title {
    font-family: 'Lora', serif; font-size: 21px; font-weight: 600;
    color: var(--text); line-height: 1.2;
}
.app-header-sub {
    font-size: 11px; color: var(--muted); font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; margin-top: 3px;
}

.chat-window {
    max-height: 38vh; overflow-y: auto;
    padding: 14px; border-radius: var(--r-lg);
    background: var(--surface); border: 1.5px solid var(--border);
    box-shadow: var(--shadow-sm); margin-bottom: 14px;
    scrollbar-width: thin; scrollbar-color: rgba(0,0,0,0.10) transparent;
}
.row-left  { display:flex; justify-content:flex-start; align-items:flex-end; margin:7px 0; gap:9px; }
.row-right { display:flex; justify-content:flex-end;   align-items:flex-end; margin:7px 0; gap:9px; }
.avatar {
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
    background: var(--accent-lt); border: 1.5px solid var(--accent-md);
}
.bubble-doc {
    background: var(--surface); border: 1.5px solid var(--border);
    border-radius: var(--r-md); border-bottom-left-radius: 4px;
    padding: 10px 14px; max-width: 78%; box-shadow: var(--shadow-sm);
    font-size: 14px; line-height: 1.6; color: var(--text);
    white-space: pre-wrap; animation: fadeUp 0.22s ease both;
}
.bubble-pat {
    background: var(--patient); color: #fff;
    border-radius: var(--r-md); border-bottom-right-radius: 4px;
    padding: 10px 14px; max-width: 78%; box-shadow: var(--shadow-sm);
    font-size: 14px; line-height: 1.6;
    white-space: pre-wrap; animation: fadeUp 0.22s ease both;
}

.panel {
    background: transparent; border: none; border-radius: 0;
    padding: 0 0 10px; box-shadow: none;
    animation: fadeUp 0.28s ease both;
}
.panel-title {
    display: flex; align-items: flex-end; gap: 9px;
    margin-bottom: 14px;
    font-family: 'Nunito', sans-serif !important;
    font-size: 14px; font-weight: 400;
}
.panel-title-avatar {
    width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    background: var(--accent-lt); border: 1.5px solid var(--accent-md);
}
.panel-title-bubble {
    background: var(--surface); border: 1.5px solid var(--border);
    border-radius: var(--r-md); border-bottom-left-radius: 4px;
    padding: 10px 15px; box-shadow: var(--shadow-sm);
    font-family: 'Nunito', sans-serif !important;
    font-size: 15px; font-weight: 600; color: var(--text);
    line-height: 1.5; max-width: 82%;
    animation: fadeUp 0.22s ease both;
}
.small-note {
    font-size: 12px; color: var(--muted); font-weight: 500;
    margin: 0 0 10px;
}
.divider { border: none; border-top: 1.5px solid var(--border); margin: 14px 0 12px; }
.inline-followup {
    background: var(--accent-lt); border-left: 3px solid var(--accent);
    border-radius: 0 var(--r-sm) var(--r-sm) 0;
    padding: 11px 14px; margin: 12px 0 8px;
    font-size: 14px; line-height: 1.6; color: var(--text);
    animation: fadeUp 0.22s ease both;
}
.inline-patient {
    background: rgba(38,70,83,0.07);
    border-radius: var(--r-sm) var(--r-sm) 0 var(--r-sm);
    padding: 9px 13px; margin: 6px 0; text-align: right;
    font-size: 14px; line-height: 1.6; color: var(--text);
    animation: fadeUp 0.18s ease both;
}

.stButton > button {
    font-family: 'Nunito', sans-serif !important;
    border-radius: var(--r-sm) !important;
    padding: 0.45rem 1rem !important;
    font-size: 14px !important; font-weight: 600 !important;
    border: 1.5px solid var(--border) !important;
    background: var(--surface) !important; color: var(--text) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.14s ease !important; white-space: nowrap !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important; color: var(--accent) !important;
    background: var(--accent-lt) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent) 0%, #1d7a6e 100%) !important;
    color: #fff !important; border-color: transparent !important;
    box-shadow: 0 3px 14px rgba(42,157,143,0.32) !important;
    font-size: 15px !important; padding: 0.55rem 1rem !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #30ada0 0%, #1a6e64 100%) !important;
    color: #fff !important; transform: translateY(-1px) !important;
}

[data-testid="stAudioInput"] { margin: 0 !important; padding: 0 !important; }
[data-testid="stAudioInput"] > label { display: none !important; }
[data-testid="stAudioInput"] > div {
    height: 38px !important; min-height: 38px !important;
    border-radius: var(--r-sm) !important;
    border: 1.5px solid var(--border) !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    background: var(--surface) !important; box-shadow: var(--shadow-sm) !important;
    width: 100% !important; overflow: hidden !important;
}
[data-testid="stAudioInput"] > div > * { transform: scale(0.88); transform-origin: center; }

[data-testid="stTextInput"] > div > div > input {
    font-family: 'Nunito', sans-serif !important;
    border-radius: var(--r-sm) !important;
    border: 1.5px solid var(--border) !important;
    padding: 7px 14px !important; font-size: 14px !important;
    background: var(--surface) !important; height: 38px !important;
    box-shadow: var(--shadow-sm) !important; color: var(--text) !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-lt) !important;
    outline: none !important;
}
[data-testid="stTextInput"] > label { display: none !important; }

[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton) {
    position: relative !important;
}
[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton)
  [data-testid="stElementContainer"]:has(.stButton) {
    position: absolute !important; right: 8px !important; top: 4px !important;
    z-index: 2 !important; width: auto !important;
}
[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton)
  .stButton { display: flex !important; justify-content: flex-end !important; }
[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton)
  .stButton button {
    width: 30px !important; height: 30px !important; min-height: 30px !important;
    padding: 0 !important; border-radius: 50% !important;
    background: linear-gradient(135deg, var(--accent) 0%, #1d7a6e 100%) !important;
    color: #fff !important; border: none !important;
    font-size: 15px !important; font-weight: 700 !important;
}
[data-testid="stColumn"]:has(.stButton) [data-testid="stTextInput"] > div > div > input {
    padding-right: 42px !important;
}

.summary-wrap {
    background: var(--surface); border: 1.5px solid var(--border);
    border-radius: var(--r-lg); padding: 26px 22px 20px;
    box-shadow: var(--shadow-md); animation: fadeUp 0.35s ease both;
}
.summary-title {
    font-family: 'Lora', serif; font-size: 20px; font-weight: 600;
    color: var(--text); margin-bottom: 4px;
}
.summary-sub {
    font-size: 11px; color: var(--muted); font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 18px;
}
.summary-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.summary-table tr { border-bottom: 1.5px solid var(--border); }
.summary-table tr:last-child { border-bottom: none; }
.summary-table td { padding: 11px 8px; vertical-align: top; line-height: 1.5; }
.summary-table td:first-child {
    font-weight: 700; color: var(--muted); width: 36%;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; padding-top: 14px;
}
.tag {
    display: inline-block; background: var(--accent-lt); color: var(--accent);
    border: 1px solid var(--accent-md); border-radius: 8px;
    padding: 2px 10px; font-size: 13px; font-weight: 600; margin: 2px 3px 2px 0;
}
.submitted-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #d1fae5; color: #065f46;
    border: 1.5px solid #6ee7b7; border-radius: 10px;
    padding: 5px 14px; font-size: 13px; font-weight: 700; margin-bottom: 16px;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────
# STAGES (redesigned from clinical transcripts):
#  -1 = name entry
#   0 = quick status (returning) — fast path if nothing changed
#   1 = overall feeling (PROMIS) — follow-up ONLY if fair/poor
#   2 = pain: yes/no → location → severity → timing (follow-up if ≥5 or new)
#   3 = eating & nutrition (from transcripts — asked in 100% of visits)
#   4 = symptom checklist (clinically-derived, "Other" hidden)
#   5 = submit + optional note
defaults = {
    "messages": [], "stage": -1, "patient_name": "",
    "selected_parts": set(), "pain_yesno": None, "feeling_level": None,
    "symptoms": [], "submitted": False, "past_checkins": [],
    "last_audio_hash": None, "mic_key_counter": 0,
    "stage_answered": {}, "followup_fired": {},
    "pain_severities": {}, "pain_timing": None,
    "eating_status": None, "food_type": None,
    "shakes_per_day": None, "hydration": None,
    "show_other": {},  # {stage_id: bool}
    "fast_path": False,
    "pain_sub": "ask",   # sub-stages within pain: ask → map → timing → done
    "eat_sub": "status", # sub-stages within eating: status → type → shakes → hydration → done
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ── Core helpers ────────────────────────────────────────────
def add_doctor(text, stage=None):
    s = stage if stage is not None else st.session_state.get("stage", -1)
    st.session_state.messages.append({"role":"doctor","content":text,"stage":s})

def add_patient(text, stage=None):
    s = stage if stage is not None else st.session_state.get("stage", -1)
    st.session_state.messages.append({"role":"patient","content":text,"stage":s})

def is_answered(sid): return st.session_state.stage_answered.get(sid, False)
def mark_answered(sid): st.session_state.stage_answered[sid] = True
def followup_fired(sid): return st.session_state.followup_fired.get(sid, False)
def mark_followup_fired(sid): st.session_state.followup_fired[sid] = True

def toggle_body_part(part):
    if part in st.session_state.selected_parts: st.session_state.selected_parts.remove(part)
    else: st.session_state.selected_parts.add(part)

def advance_stage():
    s = st.session_state.stage
    if st.session_state.fast_path:
        st.session_state.stage = 5; return
    if   s == 0: st.session_state.stage = 1
    elif s == 1: st.session_state.stage = 2
    elif s == 2: st.session_state.stage = 3  # eating is always asked
    elif s == 3: st.session_state.stage = 4
    elif s == 4: st.session_state.stage = 5

def on_answer(text: str, stage_id: int):
    """Record answer. Fire ONE curated follow-up only if concerning."""
    add_patient(text, stage=stage_id)
    mark_answered(stage_id)
    if not followup_fired(stage_id):
        fu = get_curated_followup(stage_id)
        if fu:
            mark_followup_fired(stage_id)
            add_doctor(fu, stage=stage_id)

def on_followup_reply(text: str, stage_id: int):
    """Patient replied to curated follow-up. No further questions."""
    add_patient(text, stage=stage_id)

def handle_voice(audio_value, stage_id: int, is_followup: bool = False) -> bool:
    if audio_value is None: return False
    try:
        ab = audio_value.getvalue()
        ah = hashlib.sha1(ab).hexdigest()
    except: return False
    if not ab or not ah or ah == st.session_state.last_audio_hash: return False
    st.session_state.last_audio_hash = ah
    st.session_state.mic_key_counter += 1
    with st.spinner("Transcribing…"):
        t = transcribe_audio(ab)
    if t and not t.startswith("(Transcription failed"):
        st.info(f'Heard: "{t}"')
        if is_followup: on_followup_reply(t, stage_id)
        else: on_answer(t, stage_id)
        return True
    return False

def render_chat_window():
    current = st.session_state.stage
    past_msgs = [m for m in st.session_state.messages if m.get("stage", -99) < current]
    if not past_msgs: return
    st.markdown('<div class="chat-window">', unsafe_allow_html=True)
    for msg in past_msgs:
        if msg.get("role") == "doctor":
            st.markdown(f'<div class="row-left"><div class="avatar">🩺</div>'
                        f'<div class="bubble-doc">{msg.get("content","")}</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="row-right"><div class="bubble-pat">{msg.get("content","")}</div>'
                        f'<div class="avatar">🙂</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_inline(stage_id: int):
    for msg in [m for m in st.session_state.messages if m.get("stage") == stage_id]:
        if msg["role"] == "doctor":
            st.markdown(f'<div class="inline-followup">🩺 {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="inline-patient">{msg["content"]} 🙂</div>',
                        unsafe_allow_html=True)

def render_followup_input(stage_id: int):
    """Text + mic for replying to a curated follow-up."""
    # Include mic_key_counter in widget keys to avoid DuplicateWidgetID on reruns
    ctr = st.session_state.mic_key_counter
    c_main, c_mic = st.columns([7, 2.5], gap="small")
    with c_main:
        typed = st.text_input("", placeholder="Reply…",
                              key=f"fu_txt_{stage_id}_{ctr}", label_visibility="collapsed")
        send = st.button("↑", key=f"fu_send_{stage_id}_{ctr}")
    with c_mic:
        audio = None
        if hasattr(st, "audio_input"):
            audio = st.audio_input("", label_visibility="collapsed",
                                   key=f"fu_mic_{stage_id}_{ctr}")
    if send and typed and typed.strip():
        on_followup_reply(typed.strip(), stage_id); st.rerun()
    if handle_voice(audio, stage_id, is_followup=True): st.rerun()

def render_other_input(stage_id: int, placeholder: str = "Describe…"):
    """Text input shown only after clicking 'Other'."""
    c_main, c_mic = st.columns([7, 2.5], gap="small")
    with c_main:
        typed = st.text_input("", placeholder=placeholder,
                              key=f"other_txt_{stage_id}", label_visibility="collapsed")
        send = st.button("↑", key=f"other_send_{stage_id}")
    with c_mic:
        audio = None
        if hasattr(st, "audio_input"):
            audio = st.audio_input("", key=f"other_mic_{stage_id}_{st.session_state.mic_key_counter}",
                                   label_visibility="collapsed")
    if send and typed and typed.strip():
        on_answer(typed.strip(), stage_id); st.rerun()
    if handle_voice(audio, stage_id): st.rerun()

def body_svg(selected: Set[str], prev_locs: Optional[Set[str]] = None) -> str:
    if prev_locs is None:
        prev_locs = set()
    def fill(p):
        if p in selected: return "#1f7aff"
        if p in prev_locs: return "#f4a261"
        return "#cfd8e6"
    s = "#6b7a90"
    return f"""<svg width="200" height="325" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">
  <defs><filter id="sh"><feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="rgba(0,0,0,0.12)"/></filter></defs>
  <g filter="url(#sh)"><circle cx="160" cy="70" r="38" fill="{fill('Head')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><rect x="110" y="120" width="100" height="70" rx="24" fill="{fill('Chest')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><rect x="115" y="195" width="90" height="70" rx="22" fill="{fill('Abdomen')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M110 132 C80 145,72 180,78 220 C82 250,92 270,100 290 C108 310,115 320,120 320 L120 130Z" fill="{fill('Left Arm')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M210 132 C240 145,248 180,242 220 C238 250,228 270,220 290 C212 310,205 320,200 320 L200 130Z" fill="{fill('Right Arm')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M135 265 C120 310,118 360,126 410 C132 445,132 475,128 500 L155 500 C158 470,160 435,156 405 C150 355,152 312,165 265Z" fill="{fill('Left Leg')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M185 265 C200 310,202 360,194 410 C188 445,188 475,192 500 L165 500 C162 470,160 435,164 405 C170 355,168 312,155 265Z" fill="{fill('Right Leg')}" stroke="{s}" stroke-width="2"/></g>
</svg>""".strip()

def panel_q(text):
    """Render a panel-title question bubble."""
    st.markdown(f'<div class="panel-title"><div class="panel-title-avatar">🩺</div>'
                f'<div class="panel-title-bubble">{text}</div></div>', unsafe_allow_html=True)

# ── Warnings ────────────────────────────────────────────────
if openai_init_error: st.warning(f"LLM not ready: {openai_init_error}")
_init_sheets()
if sheets_init_error: st.warning(f"Sheets not ready: {sheets_init_error}")

st.markdown('''
<div class="app-header">
  <div class="app-header-icon">🩺</div>
  <div>
    <div class="app-header-title">Symptom Check-In</div>
    <div class="app-header-sub">Cancer Care · Daily Check-In</div>
  </div>
</div>''', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE -1 — Name entry
# ════════════════════════════════════════════════════════════
if st.session_state.stage == -1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_q("Welcome · Please enter your name")
    name_input = st.text_input("Your name:", value=st.session_state.patient_name)
    if st.button("Start Check-In"):
        if name_input.strip():
            st.session_state.patient_name = name_input.strip()
            with st.spinner("Loading your history…"):
                st.session_state.past_checkins = load_past_checkins(name_input.strip())
            past = st.session_state.past_checkins
            if past:
                last = past[-1]
                # Pre-populate from last visit
                st.session_state.selected_parts = set(last.get("pain_locations", []))
                st.session_state.symptoms = list(last.get("symptoms", []))
                st.session_state.pain_severities = dict(last.get("pain_severities", {}))
                with st.spinner("Getting ready…"):
                    opening = get_opening_message(last, name_input.strip())
                add_doctor(opening, stage=0)
                st.session_state.stage = 0
            else:
                add_doctor(f"Hi {name_input.strip()}! Let's go through a few quick questions.", stage=1)
                st.session_state.stage = 1
            st.rerun()
        else:
            st.warning("Please enter your name.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

stage = st.session_state.stage
render_chat_window()

# ════════════════════════════════════════════════════════════
# STAGE 0 — Quick status + fast path
# ════════════════════════════════════════════════════════════
if stage == 0:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    # No panel_q() here — the GPT-generated opening message IS the question.
    # It greets by name and asks something specific about the most concerning
    # item from last visit, so the three buttons below answer that directly.
    render_inline(0)

    if not is_answered(0):
        st.markdown('<div class="small-note">Choose the option that best describes how that has changed:</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            if st.button("👍 Better", use_container_width=True, key="s0_better"):
                on_answer("Better since last visit.", 0)
                advance_stage(); st.rerun()
        with c2:
            if st.button("➡️ About the same", use_container_width=True, key="s0_same"):
                st.session_state.fast_path = True
                on_answer("About the same as last visit.", 0)
                advance_stage(); st.rerun()
        with c3:
            if st.button("⚠️ Worse", use_container_width=True, key="s0_worse"):
                on_answer("Worse than last visit.", 0)
                advance_stage(); st.rerun()
    else:
        # Fallback: if somehow landed here already answered, move on
        stage_msgs = [m for m in st.session_state.messages if m.get("stage") == 0]
        if stage_msgs and stage_msgs[-1]["role"] == "patient":
            advance_stage(); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 1 — Overall feeling (follow-up ONLY if fair/poor)
# ════════════════════════════════════════════════════════════
elif stage == 1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_q("How are you feeling overall today?")
    render_inline(1)

    if not is_answered(1):
        cols = st.columns(5, gap="small")
        for idx, (label, val) in enumerate([
            ("Excellent","excellent"),("Very Good","very good"),
            ("Good","good"),("Fair","fair"),("Poor","poor"),
        ]):
            with cols[idx]:
                if st.button(label, key=f"feel_{idx}", use_container_width=True):
                    st.session_state.feeling_level = val
                    on_answer(f"Feeling {val}.", 1); st.rerun()
    else:
        stage_msgs = [m for m in st.session_state.messages if m.get("stage") == 1]
        last_is_doc = stage_msgs and stage_msgs[-1]["role"] == "doctor"
        if last_is_doc and followup_fired(1):
            # Show quick reply buttons for the curated follow-up
            feeling = st.session_state.feeling_level
            if feeling == "fair":
                replies = ["Pain is worse", "Very tired", "Hard to eat"]
            elif feeling == "poor":
                replies = ["Pain is bad", "Can't eat", "Very weak"]
            else:
                replies = []
            if replies:
                rc = st.columns(len(replies), gap="small")
                for i, r in enumerate(replies):
                    with rc[i]:
                        if st.button(r, key=f"f1_r_{i}", use_container_width=True):
                            on_followup_reply(r, 1); st.rerun()
            render_followup_input(1)
        else:
            advance_stage(); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 2 — Pain (location + severity + timing in one stage)
# ════════════════════════════════════════════════════════════
elif stage == 2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    # Sub-stages within pain: 2a = yes/no, 2b = body map + severity, 2c = timing
    pain_sub = st.session_state.get("pain_sub", "ask")

    if pain_sub == "ask":
        panel_q("Do you have any pain today?")
        render_inline(2)
        if not is_answered(2):
            c1, c2 = st.columns(2, gap="small")
            with c1:
                if st.button("✅ Yes", use_container_width=True, key="p_yes"):
                    st.session_state.pain_yesno = True
                    add_patient("Yes, I have pain.", stage=2)
                    mark_answered(2)  # must mark so is_answered(2) is True in done sub-stage
                    st.session_state.pain_sub = "map"; st.rerun()
            with c2:
                if st.button("🙂 No pain", use_container_width=True, key="p_no"):
                    st.session_state.pain_yesno = False
                    on_answer("No pain today.", 2)
                    advance_stage(); st.rerun()
        else:
            # Answered but still in "ask" sub → must have said No → advance
            advance_stage(); st.rerun()

    elif pain_sub == "map":
        panel_q("Where do you feel pain?")
        past = st.session_state.get("past_checkins", [])
        prev_locs = set(past[-1].get("pain_locations", [])) if past else set()
        prev_sevs = dict(past[-1].get("pain_severities", {})) if past else {}

        if prev_locs:
            st.markdown('<div class="small-note">🟠 Orange = last visit. Tap a location to select it — a severity slider will appear inline.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="small-note">Tap a location to select it — a severity slider will appear inline.</div>',
                        unsafe_allow_html=True)

        col_svg, col_btns = st.columns([1, 1.4], gap="medium")
        with col_svg:
            st.markdown(body_svg(st.session_state.selected_parts, prev_locs),
                        unsafe_allow_html=True)

        with col_btns:
            BODY_PARTS = ["Head", "Throat/Neck", "Chest", "Abdomen",
                          "Left Arm", "Right Arm", "Left Leg", "Right Leg"]
            for part in BODY_PARTS:
                selected = part in st.session_state.selected_parts
                lbl = f"✓ {part}" if selected else part
                if st.button(lbl, key=f"bp_{part}", use_container_width=True):
                    toggle_body_part(part)
                    # If deselecting, remove its severity too
                    if part in st.session_state.pain_severities and part not in st.session_state.selected_parts:
                        del st.session_state.pain_severities[part]
                    st.rerun()

                # Inline severity slider — appears immediately under this button
                # when the location is selected
                if part in st.session_state.selected_parts:
                    default_sev = prev_sevs.get(part, 3)
                    current_sev = st.session_state.pain_severities.get(part, default_sev)
                    sev_val = st.slider(
                        f"_{part}_sev",          # label hidden via CSS
                        0, 10, current_sev,
                        key=f"sev_{part}",
                        label_visibility="collapsed",
                        help=f"{part} pain: 0 = none, 10 = worst"
                    )
                    st.session_state.pain_severities[part] = sev_val
                    # Show the numeric value as a small badge
                    color = "#e63946" if sev_val >= 7 else "#f4a261" if sev_val >= 4 else "#2a9d8f"
                    st.markdown(
                        f'<div style="text-align:right;font-size:12px;font-weight:700;'
                        f'color:{color};margin:-8px 0 6px;">'
                        f'{sev_val}/10</div>',
                        unsafe_allow_html=True
                    )

            # Other location option
            if st.button("➕ Other location", key="bp_other", use_container_width=True):
                st.session_state.show_other[2] = True; st.rerun()

        # Other location free-text input (outside columns so it spans full width)
        if st.session_state.show_other.get(2):
            c_m, c_mic = st.columns([7, 2.5], gap="small")
            with c_m:
                other_loc = st.text_input("", placeholder="Describe other location…",
                                          key="bp_other_txt", label_visibility="collapsed")
                if st.button("↑", key="bp_other_send"):
                    if other_loc.strip():
                        st.session_state.selected_parts.add(other_loc.strip())
                        st.session_state.show_other[2] = False
                        st.rerun()

        if st.button("Confirm ➜", key="pain_confirm", use_container_width=True, type="primary"):
            st.session_state.pain_sub = "timing"; st.rerun()

    elif pain_sub == "timing":
        panel_q("Is your pain constant, or mainly at certain times?")
        for idx, (lbl, val) in enumerate([
            ("Constant", "constant"),
            ("When eating/swallowing", "eating"),
            ("When moving/standing", "movement"),
            ("Comes and goes", "intermittent"),
        ]):
            if st.button(lbl, key=f"pt_{idx}", use_container_width=True):
                st.session_state.pain_timing = val
                locs = sorted(st.session_state.selected_parts)
                sevs = st.session_state.pain_severities
                summary = ", ".join(f"{l} ({sevs.get(l,'?')}/10)" for l in locs)
                on_answer(f"Pain: {summary}. Timing: {val}.", 2)
                st.session_state.pain_sub = "done"; st.rerun()

    elif pain_sub == "done":
        render_inline(2)
        stage_msgs = [m for m in st.session_state.messages if m.get("stage") == 2]
        last_is_doc = stage_msgs and stage_msgs[-1]["role"] == "doctor"
        if last_is_doc and followup_fired(2):
            # Curated follow-up for high severity / new pain
            replies = ["Yes, a lot", "Somewhat", "Not really"]
            rc = st.columns(len(replies), gap="small")
            for i, r in enumerate(replies):
                with rc[i]:
                    if st.button(r, key=f"f2_r_{i}", use_container_width=True):
                        on_followup_reply(r, 2); st.rerun()
            render_followup_input(2)
        else:
            advance_stage(); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 3 — Eating & Nutrition (from transcripts: 100% of visits)
# ════════════════════════════════════════════════════════════
elif stage == 3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    eat_sub = st.session_state.get("eat_sub", "status")

    if eat_sub == "status":
        panel_q("How has eating been?")
        render_inline(3)
        for idx, (lbl, val) in enumerate([
            ("👍 Better than before", "better"),
            ("➡️ About the same", "same"),
            ("👎 Harder to eat", "worse"),
            ("🚫 Can't eat", "cant_eat"),
        ]):
            if st.button(lbl, key=f"eat_{idx}", use_container_width=True):
                st.session_state.eating_status = val
                add_patient(f"Eating: {lbl.split(' ', 1)[1]}", stage=3)
                st.session_state.eat_sub = "type"; st.rerun()

    elif eat_sub == "type":
        panel_q("What are you able to eat?")
        for idx, (lbl, val) in enumerate([
            ("Normal food", "normal"),
            ("Soft food / purees", "soft"),
            ("Liquids only", "liquids"),
            ("Tube feeding", "tube"),
        ]):
            if st.button(lbl, key=f"food_{idx}", use_container_width=True):
                st.session_state.food_type = val
                add_patient(f"Eating: {lbl}", stage=3)
                st.session_state.eat_sub = "shakes"; st.rerun()

    elif eat_sub == "shakes":
        panel_q("How many supplement shakes/Boosts today?")
        cols = st.columns(4, gap="small")
        for idx, (lbl, val) in enumerate([("None","0"),("1–2","1-2"),("3–4","3-4"),("5+","5+")]):
            with cols[idx]:
                if st.button(lbl, key=f"shake_{idx}", use_container_width=True):
                    st.session_state.shakes_per_day = val
                    add_patient(f"Shakes: {lbl}", stage=3)
                    st.session_state.eat_sub = "hydration"; st.rerun()

    elif eat_sub == "hydration":
        panel_q("Are you staying hydrated?")
        cols = st.columns(3, gap="small")
        for idx, (lbl, val) in enumerate([("Yes, well","yes"),("Trying","trying"),("Not enough","no")]):
            with cols[idx]:
                if st.button(lbl, key=f"hyd_{idx}", use_container_width=True):
                    st.session_state.hydration = val
                    on_answer(f"Hydration: {lbl}", 3)
                    st.session_state.eat_sub = "done"; st.rerun()

    elif eat_sub == "done":
        render_inline(3)
        stage_msgs = [m for m in st.session_state.messages if m.get("stage") == 3]
        last_is_doc = stage_msgs and stage_msgs[-1]["role"] == "doctor"
        if last_is_doc and followup_fired(3):
            replies = ["Pain when eating", "No appetite", "Nausea"]
            rc = st.columns(len(replies), gap="small")
            for i, r in enumerate(replies):
                with rc[i]:
                    if st.button(r, key=f"f3_r_{i}", use_container_width=True):
                        on_followup_reply(r, 3); st.rerun()
            render_followup_input(3)
        else:
            advance_stage(); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 4 — Symptom checklist (clinically-derived from transcripts)
# ════════════════════════════════════════════════════════════
elif stage == 4:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_q("Any of these symptoms today?")

    # Derived from what clinicians screen in EVERY transcript
    SYMPTOM_LIST = [
        "Fatigue / low energy",
        "Nausea",
        "Vomiting",
        "Mouth sores",
        "Trouble swallowing",
        "Constipation",
        "Diarrhea",
        "Fever / chills",
        "Dizziness / unsteadiness",
        "Numbness / tingling (hands or feet)",
        "Hearing changes / ringing in ears",
        "Coughing / choking",
        "Anxiety / low mood",
    ]

    past = st.session_state.get("past_checkins", [])
    prev_syms = set(past[-1].get("symptoms", [])) if past else set()

    if not is_answered(4):
        render_inline(4)
        if prev_syms:
            st.markdown('<div class="small-note">✓ = from last visit. Tap to remove if resolved.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="small-note">Tap all that apply:</div>',
                        unsafe_allow_html=True)

        sc = st.columns(2, gap="small")
        for idx, sym in enumerate(SYMPTOM_LIST):
            with sc[idx % 2]:
                lbl = f"✓ {sym}" if sym in st.session_state.symptoms else sym
                if st.button(lbl, key=f"sym_{idx}", use_container_width=True):
                    if sym in st.session_state.symptoms:
                        st.session_state.symptoms.remove(sym)
                    else:
                        st.session_state.symptoms.append(sym)
                    st.rerun()

        if st.button("➕ Other symptom", key="sym_other_btn", use_container_width=True):
            st.session_state.show_other[4] = True; st.rerun()

        if st.session_state.show_other.get(4):
            # Inline handler: adds to symptoms list only, does NOT trigger follow-up flow
            ctr = st.session_state.mic_key_counter
            c_m4, c_mic4 = st.columns([7, 2.5], gap="small")
            with c_m4:
                other_sym = st.text_input("", placeholder="Describe other symptom…",
                                          key=f"other_sym_txt_{ctr}", label_visibility="collapsed")
                other_sym_send = st.button("↑", key=f"other_sym_send_{ctr}")
            with c_mic4:
                if hasattr(st, "audio_input"):
                    other_sym_audio = st.audio_input("", key=f"other_sym_mic_{ctr}",
                                                     label_visibility="collapsed")
                    if handle_voice(other_sym_audio, 4): st.rerun()
            if other_sym_send and other_sym and other_sym.strip():
                sym_val = other_sym.strip()
                if sym_val not in st.session_state.symptoms:
                    st.session_state.symptoms.append(sym_val)
                st.session_state.show_other[4] = False
                st.rerun()

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        if st.button("Confirm symptoms ➜", key="sym_confirm", use_container_width=True, type="primary"):
            sym_txt = "; ".join(st.session_state.symptoms) if st.session_state.symptoms else "none"
            on_answer(f"Symptoms: {sym_txt}", 4); st.rerun()
    else:
        render_inline(4)
        stage_msgs = [m for m in st.session_state.messages if m.get("stage") == 4]
        last_is_doc = stage_msgs and stage_msgs[-1]["role"] == "doctor"
        if last_is_doc and followup_fired(4):
            render_followup_input(4)
        else:
            # Give patient a chance to review before advancing
            if st.button("Continue →", key="sym_next", use_container_width=True, type="primary"):
                advance_stage(); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 5 — Submit
# ════════════════════════════════════════════════════════════
elif stage == 5:
    if st.session_state.submitted:
        name      = st.session_state.get("patient_name","—")
        feeling   = st.session_state.get("feeling_level") or "—"
        pain      = st.session_state.get("pain_yesno")
        locations = sorted(list(st.session_state.get("selected_parts", set())))
        severities = st.session_state.get("pain_severities", {})
        symptoms  = st.session_state.get("symptoms", [])
        timing    = st.session_state.get("pain_timing") or "—"
        eating    = st.session_state.get("eating_status") or "—"
        food      = st.session_state.get("food_type") or "—"
        shakes    = st.session_state.get("shakes_per_day") or "—"
        hydration = st.session_state.get("hydration") or "—"
        is_fast   = st.session_state.get("fast_path", False)

        # On fast path, fill nutrition fields from last visit if not collected today
        if is_fast:
            past = st.session_state.get("past_checkins", [])
            if past:
                last = past[-1]
                if eating == "—": eating = last.get("eating") or "—"
                if food   == "—": food   = last.get("food_type") or "—"
                if shakes == "—": shakes = last.get("shakes_per_day") or "—"
                if hydration == "—": hydration = last.get("hydration") or "—"
                if feeling == "—": feeling = last.get("feeling_level") or "—"

        pain_str = "Yes" if pain else ("No" if pain is False else "—")
        carried  = ' <span style="font-size:11px;opacity:.55;">(from last visit)</span>'

        # Pain locations: on fast path these are pre-populated from last visit
        if locations:
            loc_html = "".join(
                f'<span class="tag">{l} ({severities.get(l,"?")}/10)</span>'
                for l in locations
            )
            if is_fast: loc_html += carried
        else:
            loc_html = "N/A"

        sym_html = "".join(f'<span class="tag">{s}</span>' for s in symptoms) or "None"
        if is_fast and symptoms: sym_html += carried

        fast_note = ""
        if is_fast:
            fast_note = '<tr><td>Status</td><td><span class="tag">No changes since last visit</span></td></tr>'

        # Nutrition rows: hide entirely if all are still "—" (truly no data)
        show_nutrition = not (eating == "—" and food == "—" and shakes == "—" and hydration == "—")
        nutrition_rows = ""
        if show_nutrition:
            eat_disp    = f"{eating}{carried}"  if is_fast and eating    != "—" else eating
            food_disp   = f"{food}{carried}"    if is_fast and food      != "—" else food
            shakes_disp = f"{shakes}{carried}"  if is_fast and shakes    != "—" else shakes
            hyd_disp    = f"{hydration}{carried}" if is_fast and hydration != "—" else hydration
            nutrition_rows = f"""
    <tr><td>Eating</td><td>{eat_disp}</td></tr>
    <tr><td>Diet type</td><td>{food_disp}</td></tr>
    <tr><td>Shakes/day</td><td>{shakes_disp}</td></tr>
    <tr><td>Hydration</td><td>{hyd_disp}</td></tr>"""

        # Timing row: only show if pain was reported
        timing_row = f"<tr><td>Pain timing</td><td>{timing}</td></tr>" if pain else ""

        # Extract notes from free-text replies
        skip = {"About the same.", "Things are better.", "Things are worse.",
                "No pain today.", "Yes, I have pain."}
        patient_lines = [m["content"] for m in st.session_state.messages
                         if m["role"]=="patient" and m["content"] not in skip
                         and not m["content"].startswith("Feeling ")
                         and not m["content"].startswith("Pain:")
                         and not m["content"].startswith("Symptoms:")
                         and not m["content"].startswith("Eating:")
                         and not m["content"].startswith("Shakes:")
                         and not m["content"].startswith("Hydration:")]

        conv_cell = "<span style='opacity:.4'>—</span>"
        if patient_lines and _openai_ready():
            try:
                sr = openai_client.chat.completions.create(
                    model=_secret("openai_model", default="gpt-4o-mini"),
                    messages=[
                        {"role":"system","content":"Extract medically relevant facts. One bullet per fact. If nothing: None"},
                        {"role":"user","content":"\n".join(f"- {l}" for l in patient_lines)}
                    ], max_tokens=300, temperature=0.2,
                )
                txt = (sr.choices[0].message.content or "").strip()
                if txt and txt != "None":
                    items = [l.lstrip("•-– ").strip() for l in txt.split("\n") if l.strip() and l.strip()!="None"]
                    if items:
                        conv_cell = "<ul style='margin:0;padding-left:18px;'>"+"".join(
                            f"<li style='font-size:14px;'>{l}</li>" for l in items)+"</ul>"
            except: pass

        st.markdown(f"""
<div class="summary-wrap">
  <div class="submitted-badge">✅ Submitted</div>
  <div class="summary-title">Check-In Summary — {name}</div>
  <div class="summary-sub">Your care team will review this.</div>
  <table class="summary-table">
    <tr><td>Patient</td><td>{name}</td></tr>
    {fast_note}
    <tr><td>Feeling</td><td>{feeling}</td></tr>
    <tr><td>Pain</td><td>{pain_str}</td></tr>
    <tr><td>Pain locations</td><td>{loc_html}</td></tr>
    {timing_row}
    {nutrition_rows}
    <tr><td>Symptoms</td><td>{sym_html}</td></tr>
    <tr><td>Notes</td><td>{conv_cell}</td></tr>
  </table>
</div>""", unsafe_allow_html=True)

    else:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_q("Ready to submit — anything else?")

        note = st.text_input("", placeholder="Add a note for your care team (optional)…",
                             key="final_note", label_visibility="collapsed")

        if st.button("✅ Submit Check-In", use_container_width=True, type="primary"):
            if note and note.strip():
                add_patient(note.strip(), stage=5)
            try:
                save_to_sheet(); st.session_state.submitted = True; st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
