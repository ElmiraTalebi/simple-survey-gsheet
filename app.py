import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional

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
    """Brief greeting referencing last visit — NO question, just a warm hello."""
    if not _openai_ready():
        return f"Hi {name}! Good to see you again. Let's see how things are today."
    fl   = last.get("feeling_level", "?")
    pn   = "yes" if last.get("pain") else "no"
    ploc = ", ".join(last.get("pain_locations", [])) or "none"
    sym  = ", ".join(last.get("symptoms", [])) or "none"
    ts   = last.get("timestamp", "your last visit")
    system = (
        "You are a warm symptom-intake assistant for a head and neck cancer care clinic. "
        "Write ONE sentence greeting the patient by name, briefly mention their "
        "specific data from last time (e.g. throat pain, eating difficulty, fatigue, or symptoms). "
        "End with: 'Let\\'s see how things are today.' "
        "Do NOT ask a question. Max 2 sentences. No filler."
    )
    try:
        r = openai_client.chat.completions.create(
            model=_secret("openai_model", default="gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Patient: {name}. Last: {ts}. "
                                               f"Feeling: {fl}. Pain: {pn}. "
                                               f"Locations: {ploc}. Symptoms: {sym}."}
            ],
            max_tokens=80, temperature=0.5,
        )
        return (r.choices[0].message.content or "").strip()
    except:
        return f"Hi {name}! Let's see how things are today."

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
    "fair":       "What's been the hardest part — throat pain, eating, or fatigue?",
    "unwell":     "What is bothering you the most right now?",
    "very unwell":"What is bothering you the most right now?",
}

PAIN_DRILL = {
    "severity_high": "Is this pain constant, or mainly when you swallow or eat?",
    "new_location":  "When did you first notice pain in this new area?",
    "worsening":     "When did it start getting worse?",
}

EATING_DRILL = {
    "worse":    "What's making it harder — throat pain, nausea, or no appetite?",
    "cant_eat": "Are you able to get any fluids or shakes down at all, even small sips?",
}

SYMPTOM_FOLLOWUPS = {
    "Nausea":                              "Are you able to keep food and fluids down?",
    "Vomiting":                            "How many times in the last 24 hours?",
    "Mouth sores / ulcers":               "Are the sores making it hard to eat or drink?",
    "Trouble swallowing":                  "Are you coughing or choking when you swallow?",
    "Fever / chills":                      "Have you taken your temperature? What was it?",
    "Constipation":                        "How many days since your last bowel movement?",
    "Diarrhea":                            "How many episodes today?",
    "Dizziness / unsteadiness":           "Does it happen when you change position (sitting to standing)?",
    "Ringing in ears / hearing changes":  "Is it affecting your ability to hear conversations?",
    "Mucus / thick phlegm":               "Is the mucus making it hard to swallow or sleep?",
    "Skin reaction / radiation burn":     "Is the skin open or just red and irritated?",
    "Swelling (legs or ankles)":          "Is the swelling new, or has it been there a while?",
}

def get_curated_followup(stage_id: int) -> Optional[str]:
    """Return ONE curated follow-up ONLY if the answer is concerning."""
    if stage_id == 1:
        feeling = st.session_state.get("feeling_level")
        return FEELING_FOLLOWUPS.get(feeling)  # None for very well/well

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
                     "Dizziness / unsteadiness", "Mouth sores / ulcers", "Skin reaction / radiation burn"]
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
    c_main, c_mic = st.columns([7, 2.5], gap="small")
    with c_main:
        typed = st.text_input("", placeholder="Reply…",
                              key=f"fu_txt_{stage_id}", label_visibility="collapsed")
        send = st.button("↑", key=f"fu_send_{stage_id}")
    with c_mic:
        audio = None
        if hasattr(st, "audio_input"):
            audio = st.audio_input("", label_visibility="collapsed",
                                   key=f"fu_mic_{stage_id}_{st.session_state.mic_key_counter}")
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


def face_neck_svg(selected: set, prev_locs: set = set()) -> str:
    """
    SVG diagram of a face + neck with 6 labelled pain regions.
    Selected regions → teal (#2a9d8f). Prev-visit-only → orange (#f4a261). Default → light grey.
    Pure display — buttons in the adjacent column do the toggling.
    """
    def fill(p):
        if p in selected:  return "#2a9d8f"   # teal  – currently selected
        if p in prev_locs: return "#f4a261"   # orange – from last visit
        return "#dde3ec"                       # grey   – unselected

    def text_col(p):
        return "#ffffff" if (p in selected or p in prev_locs) else "#4a5568"

    # colours
    s  = "#8a97aa"   # stroke
    sw = "1.5"       # stroke-width

    return f"""
<svg width="190" height="340" viewBox="0 0 190 340"
     xmlns="http://www.w3.org/2000/svg"
     style="display:block;margin:0 auto;filter:drop-shadow(0 2px 6px rgba(0,0,0,0.10))">
  <defs>
    <filter id="sh2"><feDropShadow dx="0" dy="1" stdDeviation="1.5"
      flood-color="rgba(0,0,0,0.13)"/></filter>
  </defs>

  <!-- ── Neck region ── -->
  <rect x="65" y="218" width="60" height="68" rx="14"
        fill="{fill('Neck')}" stroke="{s}" stroke-width="{sw}" filter="url(#sh2)"/>
  <text x="95" y="257" text-anchor="middle" font-family="Nunito,sans-serif"
        font-size="11" font-weight="700" fill="{text_col('Neck')}">Neck</text>

  <!-- ── Throat region (lower neck / collar) ── -->
  <rect x="60" y="285" width="70" height="46" rx="14"
        fill="{fill('Throat')}" stroke="{s}" stroke-width="{sw}" filter="url(#sh2)"/>
  <text x="95" y="313" text-anchor="middle" font-family="Nunito,sans-serif"
        font-size="11" font-weight="700" fill="{text_col('Throat')}">Throat</text>

  <!-- ── Face oval ── -->
  <ellipse cx="95" cy="115" rx="68" ry="88"
           fill="#f5f0ea" stroke="{s}" stroke-width="{sw}" filter="url(#sh2)"/>

  <!-- ── Ear Left ── -->
  <ellipse cx="22" cy="118" rx="14" ry="20"
           fill="{fill('Ear(s)')}" stroke="{s}" stroke-width="{sw}" filter="url(#sh2)"/>
  <!-- ── Ear Right ── -->
  <ellipse cx="168" cy="118" rx="14" ry="20"
           fill="{fill('Ear(s)')}" stroke="{s}" stroke-width="{sw}" filter="url(#sh2)"/>
  <!-- Ear label centred between both ears, above face -->
  <text x="95" y="44" text-anchor="middle" font-family="Nunito,sans-serif"
        font-size="10" font-weight="700" fill="{text_col('Ear(s)')}">Ear(s)</text>
  <!-- small arrows pointing to each ear -->
  <line x1="64" y1="46" x2="36" y2="100" stroke="{text_col('Ear(s)')}" stroke-width="1" opacity="0.5"/>
  <line x1="126" y1="46" x2="154" y2="100" stroke="{text_col('Ear(s)')}" stroke-width="1" opacity="0.5"/>

  <!-- ── Jaw region (lower face arc) ── -->
  <path d="M42 165 Q95 215 148 165 Q148 195 95 205 Q42 195 42 165Z"
        fill="{fill('Jaw')}" stroke="{s}" stroke-width="{sw}" filter="url(#sh2)"/>
  <text x="95" y="195" text-anchor="middle" font-family="Nunito,sans-serif"
        font-size="11" font-weight="700" fill="{text_col('Jaw')}">Jaw</text>

  <!-- ── Lips / Gums strip ── -->
  <rect x="65" y="155" width="60" height="22" rx="11"
        fill="{fill('Lips / Gums')}" stroke="{s}" stroke-width="{sw}" filter="url(#sh2)"/>
  <text x="95" y="170" text-anchor="middle" font-family="Nunito,sans-serif"
        font-size="9.5" font-weight="700" fill="{text_col('Lips / Gums')}">Lips / Gums</text>

  <!-- ── Mouth / Tongue oval (mid face) ── -->
  <ellipse cx="95" cy="133" rx="30" ry="16"
           fill="{fill('Mouth / Tongue')}" stroke="{s}" stroke-width="{sw}" filter="url(#sh2)"/>
  <text x="95" y="137" text-anchor="middle" font-family="Nunito,sans-serif"
        font-size="9" font-weight="700" fill="{text_col('Mouth / Tongue')}">Mouth/Tongue</text>

  <!-- ── Eyes (decorative, not clickable) ── -->
  <ellipse cx="73" cy="100" rx="12" ry="8" fill="#c8d0dc" stroke="{s}" stroke-width="1"/>
  <ellipse cx="117" cy="100" rx="12" ry="8" fill="#c8d0dc" stroke="{s}" stroke-width="1"/>
  <circle  cx="73"  cy="100" r="4"  fill="#6b7a90"/>
  <circle  cx="117" cy="100" r="4"  fill="#6b7a90"/>

  <!-- ── Nose (decorative) ── -->
  <path d="M90 108 Q95 122 100 108" fill="none" stroke="{s}" stroke-width="1.2"/>

  <!-- ── Hair / head top (decorative) ── -->
  <ellipse cx="95" cy="33" rx="62" ry="22" fill="#b0bac8" opacity="0.35"/>
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
    <div class="app-header-sub">Head &amp; Neck Cancer Care · Weekly Check-In</div>
  </div>
</div>''', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE -1 — Name entry
# ════════════════════════════════════════════════════════════
if st.session_state.stage == -1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_q("Welcome · Please enter your name to begin your check-in")
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
                add_doctor(f"Hi {name_input.strip()}! I'm going to ask you a few questions about how you've been feeling since your last visit. This helps your care team prepare for today.", stage=1)
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
    panel_q("Compared to your last visit, how are things going overall?")
    render_inline(0)

    if not is_answered(0):
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            if st.button("👍 Better", use_container_width=True, key="s0_better"):
                on_answer("Things are better.", 0); st.rerun()
        with c2:
            if st.button("➡️ About the same", use_container_width=True, key="s0_same"):
                st.session_state.fast_path = True
                on_answer("About the same.", 0); advance_stage(); st.rerun()
        with c3:
            if st.button("⚠️ Worse", use_container_width=True, key="s0_worse"):
                on_answer("Things are worse.", 0); st.rerun()
    else:
        # If answered and no follow-up pending, advance
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
            ("Very Well","very well"),("Well","well"),
            ("Fair","fair"),("Unwell","unwell"),("Very Unwell","very unwell"),
        ]):
            with cols[idx]:
                if st.button(label, key=f"feel_{idx}", use_container_width=True):
                    st.session_state.feeling_level = val
                    on_answer(f"Feeling {val}.", 1); st.rerun()
    else:
        stage_msgs = [m for m in st.session_state.messages if m.get("stage") == 1]
        last_is_doc = stage_msgs and stage_msgs[-1]["role"] == "doctor"
        if last_is_doc:
            # Show quick reply buttons for the curated follow-up
            feeling = st.session_state.feeling_level
            if feeling == "fair":
                replies = ["Throat pain", "Very fatigued", "Hard to eat or drink"]
            elif feeling in ("unwell", "very unwell"):
                replies = ["Throat/mouth pain", "Can't eat or drink", "Very weak or dizzy"]
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
        panel_q("Where do you feel pain? (Select all that apply)")
        past = st.session_state.get("past_checkins", [])
        prev_locs = set(past[-1].get("pain_locations", [])) if past else set()

        if prev_locs:
            st.markdown('<div class="small-note">🟠 Orange = last visit. Tap the diagram or buttons to select.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="small-note">Tap the diagram or buttons to select all areas with pain.</div>',
                        unsafe_allow_html=True)

        # Head & neck cancer specific locations
        HN_LOCATIONS = [
            "Throat",
            "Mouth / Tongue",
            "Neck",
            "Ear(s)",
            "Jaw",
            "Lips / Gums",
        ]

        col_svg, col_btns = st.columns([1, 1], gap="medium")
        with col_svg:
            st.markdown(face_neck_svg(st.session_state.selected_parts, prev_locs),
                        unsafe_allow_html=True)
        with col_btns:
            for part in HN_LOCATIONS:
                in_curr = part in st.session_state.selected_parts
                lbl = f"✓ {part}" if in_curr else part
                if st.button(lbl, key=f"bp_{part}", use_container_width=True):
                    toggle_body_part(part); st.rerun()
            if st.button("➕ Other", key="bp_other", use_container_width=True):
                st.session_state.show_other[2] = True; st.rerun()

        if st.session_state.show_other.get(2):
            c_m, c_mic = st.columns([7, 2.5], gap="small")
            with c_m:
                other_loc = st.text_input("", placeholder="Other location…",
                                          key="bp_other_txt", label_visibility="collapsed")
                if st.button("↑", key="bp_other_send"):
                    if other_loc.strip():
                        st.session_state.selected_parts.add(other_loc.strip())
                        st.rerun()

        if st.session_state.selected_parts:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown('<div class="small-note">Rate severity (0 = none, 10 = worst):</div>',
                        unsafe_allow_html=True)
            prev_sevs = dict(past[-1].get("pain_severities", {})) if past else {}
            for part in sorted(st.session_state.selected_parts):
                default = prev_sevs.get(part, 3)
                val = st.slider(part, 0, 10, default, key=f"sev_{part}")
                st.session_state.pain_severities[part] = val

        if st.button("Confirm ➜", key="pain_confirm", use_container_width=True, type="primary"):
            st.session_state.pain_sub = "timing"; st.rerun()

    elif pain_sub == "timing":
        panel_q("When does the pain occur?")
        for idx, (lbl, val) in enumerate([
            ("When swallowing", "swallowing"),
            ("When talking", "talking"),
            ("All the time / constant", "constant"),
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
        if last_is_doc:
            # Curated follow-up for high severity / new pain
            replies = ["Yes, constant", "Only when swallowing", "Manageable with meds"]
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
        panel_q("How has eating and drinking been since your last visit?")
        render_inline(3)
        for idx, (lbl, val) in enumerate([
            ("👍 Better than before", "better"),
            ("➡️ About the same", "same"),
            ("👎 Harder to eat or drink", "worse"),
            ("🚫 Unable to eat or drink", "cant_eat"),
        ]):
            if st.button(lbl, key=f"eat_{idx}", use_container_width=True):
                st.session_state.eating_status = val
                add_patient(f"Eating: {lbl.split(' ', 1)[1]}", stage=3)
                st.session_state.eat_sub = "type"; st.rerun()

    elif eat_sub == "type":
        panel_q("What are you currently able to eat?")
        for idx, (lbl, val) in enumerate([
            ("Regular / near-normal food", "normal"),
            ("Soft foods (purees, soups, pudding)", "soft"),
            ("Liquids only (shakes, broth)", "liquids"),
            ("Tube feeding only", "tube"),
        ]):
            if st.button(lbl, key=f"food_{idx}", use_container_width=True):
                st.session_state.food_type = val
                add_patient(f"Eating: {lbl}", stage=3)
                st.session_state.eat_sub = "shakes"; st.rerun()

    elif eat_sub == "shakes":
        panel_q("How many protein shakes or supplement drinks (Boost / Ensure) per day?")
        cols = st.columns(4, gap="small")
        for idx, (lbl, val) in enumerate([("None","0"),("1–2","1-2"),("3–4","3-4"),("5+","5+")]):
            with cols[idx]:
                if st.button(lbl, key=f"shake_{idx}", use_container_width=True):
                    st.session_state.shakes_per_day = val
                    add_patient(f"Shakes: {lbl}", stage=3)
                    st.session_state.eat_sub = "hydration"; st.rerun()

    elif eat_sub == "hydration":
        panel_q("How has your fluid intake been?")
        cols = st.columns(3, gap="small")
        for idx, (lbl, val) in enumerate([("Drinking well","yes"),("Getting some fluids","trying"),("Very little / not enough","no")]):
            with cols[idx]:
                if st.button(lbl, key=f"hyd_{idx}", use_container_width=True):
                    st.session_state.hydration = val
                    on_answer(f"Hydration: {lbl}", 3)
                    st.session_state.eat_sub = "done"; st.rerun()

    elif eat_sub == "done":
        render_inline(3)
        stage_msgs = [m for m in st.session_state.messages if m.get("stage") == 3]
        last_is_doc = stage_msgs and stage_msgs[-1]["role"] == "doctor"
        if last_is_doc:
            replies = ["Throat pain when swallowing", "No appetite", "Nausea / vomiting"]
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
    panel_q("Are you experiencing any of the following? (Select all that apply)")

    # Head & neck oncology symptom list — derived from all 5 clinical transcripts
    SYMPTOM_LIST = [
        "Fatigue / low energy",
        "Nausea",
        "Vomiting",
        "Mouth sores / ulcers",
        "Dry mouth",
        "Trouble swallowing",
        "Coughing / choking when swallowing",
        "Mucus / thick phlegm",
        "Constipation",
        "Diarrhea",
        "Fever / chills",
        "Dizziness / unsteadiness",
        "Ringing in ears / hearing changes",
        "Numbness / tingling (hands or feet)",
        "Skin reaction / radiation burn",
        "Swelling (legs or ankles)",
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
            render_other_input(4, "Describe other symptom…")

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        if st.button("Confirm symptoms ➜", key="sym_confirm", use_container_width=True, type="primary"):
            sym_txt = "; ".join(st.session_state.symptoms) if st.session_state.symptoms else "none"
            on_answer(f"Symptoms: {sym_txt}", 4); st.rerun()
    else:
        render_inline(4)
        stage_msgs = [m for m in st.session_state.messages if m.get("stage") == 4]
        last_is_doc = stage_msgs and stage_msgs[-1]["role"] == "doctor"
        if last_is_doc:
            render_followup_input(4)
        else:
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

        pain_str = "Yes" if pain else ("No" if pain is False else "—")
        loc_html = "".join(f'<span class="tag">{l} ({severities.get(l,"?")}/10)</span>' for l in locations) or "N/A"
        sym_html = "".join(f'<span class="tag">{s}</span>' for s in symptoms) or "None"

        fast_note = ""
        if st.session_state.fast_path:
            fast_note = '<tr><td>Status</td><td><span class="tag">No changes from last visit</span></td></tr>'

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
  <div class="summary-sub">Your care team will review this before your visit.</div>
  <table class="summary-table">
    <tr><td>Patient</td><td>{name}</td></tr>
    {fast_note}
    <tr><td>Overall Feeling</td><td>{feeling}</td></tr>
    <tr><td>Pain Today</td><td>{pain_str}</td></tr>
    <tr><td>Pain Location(s)</td><td>{loc_html}</td></tr>
    <tr><td>Pain Timing</td><td>{timing}</td></tr>
    <tr><td>Eating & Drinking</td><td>{eating}</td></tr>
    <tr><td>Diet Type</td><td>{food}</td></tr>
    <tr><td>Supplement Shakes/day</td><td>{shakes}</td></tr>
    <tr><td>Fluid Intake</td><td>{hydration}</td></tr>
    <tr><td>Symptoms</td><td>{sym_html}</td></tr>
    <tr><td>Additional Notes</td><td>{conv_cell}</td></tr>
  </table>
</div>""", unsafe_allow_html=True)

    else:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_q("Almost done — is there anything else you'd like your care team to know?")

        note = st.text_input("", placeholder="E.g. a new symptom, question for your doctor, medication concern… (optional)",
                             key="final_note", label_visibility="collapsed")

        if st.button("✅ Submit Check-In", use_container_width=True, type="primary"):
            if note and note.strip():
                add_patient(note.strip(), stage=5)
            try:
                save_to_sheet(); st.session_state.submitted = True; st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
