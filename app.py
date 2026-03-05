import hashlib
import json
from datetime import datetime
from typing import Dict, List, Set, Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI

st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ── Query-param handler (HTML button interactions) ───────────
_qp = st.query_params
if "bp_toggle" in _qp:
    _part = _qp["bp_toggle"]
    if "selected_parts" not in st.session_state:
        st.session_state.selected_parts = set()
    if _part in st.session_state.selected_parts:
        st.session_state.selected_parts.discard(_part)
        for _d in ("pain_severities", "part_followup_q", "part_followup_a"):
            if _d in st.session_state: st.session_state[_d].pop(_part, None)
    else:
        st.session_state.selected_parts.add(_part)
    st.query_params.clear(); st.rerun()
if "pfu_part" in _qp and "pfu_ans" in _qp:
    if "part_followup_a" not in st.session_state:
        st.session_state.part_followup_a = {}
    st.session_state.part_followup_a[_qp["pfu_part"]] = _qp["pfu_ans"]
    st.query_params.clear(); st.rerun()

# ── Secrets ────────────────────────────────────────────────
def _secret(*keys, default=None):
    for k in keys:
        if k in st.secrets:
            val = st.secrets[k]
            # Return nested sections as dict (for gcp_service_account etc.)
            try:
                return dict(val) if hasattr(val, "keys") else val
            except Exception:
                return val
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
            "feeling_level":    st.session_state.get("feeling_level"),
            "pain":             st.session_state.get("pain_yesno"),
            "pain_locations":   sorted(list(st.session_state.get("selected_parts", set()))),
            "pain_severities":  st.session_state.get("pain_severities", {}),
            "pain_timing":      st.session_state.get("pain_timing"),
            "eating":           st.session_state.get("eating_status"),
            "food_type":        st.session_state.get("food_type"),
            "shakes_per_day":   st.session_state.get("shakes_per_day"),
            "hydration":        st.session_state.get("hydration"),
            "symptoms":         st.session_state.get("symptoms", []),
            "conversation":     st.session_state.get("messages", []),
            "fast_path":        st.session_state.get("fast_path", False),
        })
    ])

# ── Helpers ─────────────────────────────────────────────────
def _openai_ready():
    return openai_client is not None and openai_init_error is None

def get_opening_message(last: Dict, name: str) -> str:
    """
    Fully deterministic greeting — references the most concerning item from
    last visit. No GPT: consistent, fast, and safe to demo to stakeholders.
    """
    sevs = last.get("pain_severities", {})
    locs = last.get("pain_locations", [])
    syms = last.get("symptoms", [])

    # Pick the highest-severity pain location if present
    if sevs and locs:
        worst_loc = max(sevs, key=lambda l: sevs[l])
        worst_sev = sevs[worst_loc]
        return (f"Hi {name}! Last time you had pain in your {worst_loc.lower()} "
                f"(severity {worst_sev}/10). How is that today — better, worse, or about the same?")
    if locs:
        return (f"Hi {name}! Last time you reported pain in your {locs[0].lower()}. "
                f"How is that today — better, worse, or about the same?")
    # Fall back to most concerning symptom
    priority_syms = ["Fever / chills", "Vomiting", "Trouble swallowing",
                     "Shortness of breath", "Dizziness / unsteadiness"]
    for s in priority_syms:
        if s in syms:
            return (f"Hi {name}! Last time you reported {s.lower()}. "
                    f"Is that still bothering you — better, worse, or about the same?")
    if syms:
        return (f"Hi {name}! Last time you reported {syms[0].lower()}. "
                f"How are you feeling today — better, worse, or about the same?")
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

# ── Symptom follow-up — fires ONLY for urgent NEW symptoms ──
# Rule: ONE question max, ONLY if symptom is clinically urgent AND new vs last visit.
# "New" = not reported last visit. Always-urgent = Fever, SOB, Vomiting (ask even if recurring).

_URGENT_NEW = [
    ("Fever / chills",           "Have you taken your temperature? What was it?"),
    ("Shortness of breath",      "Is it happening at rest, or only with activity?"),
    ("Vomiting",                 "How many times in the last 24 hours?"),
    ("Trouble swallowing",       "Are you coughing or choking when you swallow?"),
    ("Dizziness / unsteadiness", "Does it get worse when you stand up?"),
    ("Diarrhea",                 "How many episodes today?"),
    ("Constipation",             "How many days since your last bowel movement?"),
    ("Mouth sores",              "Are the sores making it hard to eat or drink?"),
]
_ALWAYS_URGENT = {"Fever / chills", "Shortness of breath", "Vomiting"}

def get_symptom_followup() -> Optional[str]:
    """Return ONE question for the highest-priority urgent/new symptom, or None."""
    symptoms = set(st.session_state.get("symptoms", []))
    if not symptoms:
        return None
    past = st.session_state.get("past_checkins", [])
    prev = set(past[-1].get("symptoms", [])) if past else set()
    for sym, q in _URGENT_NEW:
        if sym in symptoms and (sym in _ALWAYS_URGENT or sym not in prev):
            return q
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

/* ── Followup reply pill buttons ── */
/* Streamlit renders widgets AFTER markdown divs as siblings in the same container.
   We use a sentinel div with a data attribute and the adjacent sibling combinator. */
.stColumn:has(.reply-sentinel) .stButton > button,
.reply-btn-wrap + div .stButton > button,
div[data-reply-pills] ~ div .stButton > button {
    background: rgba(42,157,143,0.12) !important;
    color: var(--accent) !important;
    border: 1.5px solid var(--accent-md) !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    padding: 0.2rem 0.75rem !important;
    min-height: 28px !important;
    height: 28px !important;
    box-shadow: none !important;
    margin-bottom: 3px !important;
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

# ── Session state ────────────────────────────────────────────
# Stages:  -1=name  0=returning status  1=new patient greeting
#           2=pain map+timing  3=eating  4=symptoms  5=submit
defaults = {
    "messages": [], "stage": -1, "patient_name": "",
    "selected_parts": set(), "pain_yesno": None,
    "symptoms": [], "submitted": False, "past_checkins": [],
    "last_audio_hash": None, "mic_key_counter": 0,
    "pain_severities": {}, "pain_timing": None,
    "eating_status": None, "food_type": None,
    "shakes_per_day": None, "hydration": None,
    "show_other": {},
    "fast_path": False,
    "pain_sub": "map",      # map → timing → done
    "eat_sub": "status",    # status → type → shakes → hydration → done
    "part_followup_q": {},  # {body_part: question_str}
    "part_followup_a": {},  # {body_part: answer_str}
    "part_fu_logged": False,
    "sym_fu_q": None,       # the ONE symptom followup question (or None)
    "sym_fu_a": None,       # patient's answer to it (or None)
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Core helpers ──────────────────────────────────────────────
def add_doctor(text, stage=None):
    s = stage if stage is not None else st.session_state.stage
    st.session_state.messages.append({"role": "doctor", "content": text, "stage": s})

def add_patient(text, stage=None):
    s = stage if stage is not None else st.session_state.stage
    st.session_state.messages.append({"role": "patient", "content": text, "stage": s})

def toggle_body_part(part):
    if part in st.session_state.selected_parts:
        st.session_state.selected_parts.discard(part)
        st.session_state.pain_severities.pop(part, None)
        st.session_state.part_followup_q.pop(part, None)
        st.session_state.part_followup_a.pop(part, None)
    else:
        st.session_state.selected_parts.add(part)

def advance_stage():
    s = st.session_state.stage
    if st.session_state.fast_path and s == 0:
        st.session_state.stage = 5; return
    if   s == 0: st.session_state.stage = 2
    elif s == 1: st.session_state.stage = 2
    elif s == 2: st.session_state.stage = 3
    elif s == 3: st.session_state.stage = 4
    elif s == 4: st.session_state.stage = 5

def handle_voice(audio_value) -> Optional[str]:
    if audio_value is None: return None
    try:
        ab = audio_value.getvalue()
        ah = hashlib.sha1(ab).hexdigest()
    except: return None
    if not ab or ah == st.session_state.last_audio_hash: return None
    st.session_state.last_audio_hash = ah
    st.session_state.mic_key_counter += 1
    with st.spinner("Transcribing…"):
        t = transcribe_audio(ab)
    return t if t and not t.startswith("(Transcription") else None

def render_chat_window():
    current = st.session_state.stage
    past = [m for m in st.session_state.messages if m.get("stage", -99) < current]
    if not past: return
    st.markdown('<div class="chat-window">', unsafe_allow_html=True)
    for m in past:
        if m["role"] == "doctor":
            st.markdown(f'<div class="row-left"><div class="avatar">🩺</div>'
                        f'<div class="bubble-doc">{m["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="row-right"><div class="bubble-pat">{m["content"]}</div>'
                        f'<div class="avatar">🙂</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_inline(stage_id: int):
    for m in [m for m in st.session_state.messages if m.get("stage") == stage_id]:
        if m["role"] == "doctor":
            st.markdown(f'<div class="inline-followup">🩺 {m["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="inline-patient">{m["content"]} 🙂</div>', unsafe_allow_html=True)

def render_text_mic(key_prefix: str, placeholder: str = "Type your answer…") -> Optional[str]:
    """Text input + optional mic. Returns submitted text or None."""
    ctr = st.session_state.mic_key_counter
    c_txt, c_mic = st.columns([7, 2.5], gap="small")
    with c_txt:
        typed = st.text_input("", placeholder=placeholder,
                              key=f"{key_prefix}_t{ctr}", label_visibility="collapsed")
        if st.button("↑", key=f"{key_prefix}_s{ctr}") and typed.strip():
            return typed.strip()
    with c_mic:
        if hasattr(st, "audio_input"):
            audio = st.audio_input("", label_visibility="collapsed", key=f"{key_prefix}_m{ctr}")
            t = handle_voice(audio)
            if t: return t
    return None

def panel_q(text):
    st.markdown(f'<div class="panel-title"><div class="panel-title-avatar">🩺</div>'
                f'<div class="panel-title-bubble">{text}</div></div>', unsafe_allow_html=True)


# ── Body SVG ─────────────────────────────────────────────────
def body_svg(selected: Set[str], prev_locs: Optional[Set[str]] = None) -> str:
    if prev_locs is None: prev_locs = set()
    def fill(p):
        if p in selected: return "#e63946"
        if p in prev_locs: return "#f4a261"
        return "#a8d5b5"
    def stroke(p):
        if p in selected: return "#b52535"
        if p in prev_locs: return "#c47a3a"
        return "#6daa82"
    return f"""<svg width="200" height="325" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">
  <defs><filter id="sh"><feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="rgba(0,0,0,0.12)"/></filter></defs>
  <g filter="url(#sh)"><circle cx="160" cy="70" r="38" fill="{fill('Head')}" stroke="{stroke('Head')}" stroke-width="2"/></g>
  <g filter="url(#sh)"><rect x="145" y="108" width="30" height="22" rx="8" fill="{fill('Throat/Neck')}" stroke="{stroke('Throat/Neck')}" stroke-width="2"/></g>
  <g filter="url(#sh)"><rect x="110" y="120" width="100" height="70" rx="24" fill="{fill('Chest')}" stroke="{stroke('Chest')}" stroke-width="2"/></g>
  <g filter="url(#sh)"><rect x="115" y="195" width="90" height="70" rx="22" fill="{fill('Abdomen')}" stroke="{stroke('Abdomen')}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M110 132 C80 145,72 180,78 220 C82 250,92 270,100 290 C108 310,115 320,120 320 L120 130Z" fill="{fill('Left Arm')}" stroke="{stroke('Left Arm')}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M210 132 C240 145,248 180,242 220 C238 250,228 270,220 290 C212 310,205 320,200 320 L200 130Z" fill="{fill('Right Arm')}" stroke="{stroke('Right Arm')}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M135 265 C120 310,118 360,126 410 C132 445,132 475,128 500 L155 500 C158 470,160 435,156 405 C150 355,152 312,165 265Z" fill="{fill('Left Leg')}" stroke="{stroke('Left Leg')}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M185 265 C200 310,202 360,194 410 C188 445,188 475,192 500 L165 500 C162 470,160 435,164 405 C170 355,168 312,155 265Z" fill="{fill('Right Leg')}" stroke="{stroke('Right Leg')}" stroke-width="2"/></g>
</svg>""".strip()

# ── Warnings + header ────────────────────────────────────────
if openai_init_error: st.warning(f"LLM not ready: {openai_init_error}")
_init_sheets()
if sheets_init_error: st.warning(f"Sheets not ready: {sheets_init_error}")

st.markdown('''<div class="app-header">
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
    name_input = st.text_input("", placeholder="Your name…",
                               value=st.session_state.patient_name,
                               label_visibility="collapsed")
    if st.button("Start Check-In", use_container_width=True, type="primary"):
        name = name_input.strip()
        if name:
            st.session_state.patient_name = name
            # Reset all per-session flow state
            for k in ("part_followup_q","part_followup_a","part_fu_logged",
                       "sym_fu_q","sym_fu_a","pain_sub","eat_sub","messages",
                       "pain_yesno","pain_severities","pain_timing",
                       "eating_status","food_type","shakes_per_day","hydration",
                       "selected_parts","symptoms","fast_path","submitted",
                       "stage_answered","followup_fired","show_other"):
                st.session_state[k] = defaults.get(k, None) if k not in ("selected_parts","symptoms","messages","show_other","part_followup_q","part_followup_a","pain_severities") else type(defaults[k])()
            st.session_state.part_fu_logged = False
            st.session_state.fast_path = False
            st.session_state.submitted = False
            st.session_state.pain_sub = "map"
            st.session_state.eat_sub = "status"
            st.session_state.sym_fu_q = None
            st.session_state.sym_fu_a = None
            st.session_state.show_other = {}
            with st.spinner("Loading your history…"):
                st.session_state.past_checkins = load_past_checkins(name)
            past = st.session_state.past_checkins
            if past:
                last = past[-1]
                st.session_state.selected_parts = set(last.get("pain_locations", []))
                st.session_state.symptoms = list(last.get("symptoms", []))
                st.session_state.pain_severities = dict(last.get("pain_severities", {}))
                add_doctor(get_opening_message(last, name), stage=0)
                st.session_state.stage = 0
            else:
                add_doctor(f"Hi {name}! Let's do a quick check-in.", stage=1)
                st.session_state.stage = 1
            st.rerun()
        else:
            st.warning("Please enter your name.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

stage = st.session_state.stage
render_chat_window()

# ════════════════════════════════════════════════════════════
# STAGE 0 — Returning patient quick status
# ════════════════════════════════════════════════════════════
if stage == 0:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    render_inline(0)
    s0_done = any(m["role"] == "patient" for m in st.session_state.messages if m.get("stage") == 0)
    if not s0_done:
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            if st.button("👍 Better", use_container_width=True, key="s0_b"):
                st.session_state.fast_path = True
                add_patient("Better since last visit.", stage=0)
                advance_stage(); st.rerun()
        with c2:
            if st.button("➡️ About the same", use_container_width=True, key="s0_s"):
                st.session_state.fast_path = True
                add_patient("About the same.", stage=0)
                advance_stage(); st.rerun()
        with c3:
            if st.button("⚠️ Worse", use_container_width=True, key="s0_w"):
                st.session_state.fast_path = False
                add_patient("Things are worse.", stage=0)
                advance_stage(); st.rerun()
    else:
        advance_stage(); st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 1 — New patient (no history) — goes straight to pain
# ════════════════════════════════════════════════════════════
elif stage == 1:
    # No UI needed — immediately advance to pain stage
    advance_stage(); st.rerun()

# ════════════════════════════════════════════════════════════
# STAGE 2 — Pain: body map → timing → done
# ════════════════════════════════════════════════════════════
elif stage == 2:
    import urllib.parse
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    pain_sub = st.session_state.pain_sub
    past = st.session_state.get("past_checkins", [])
    prev_locs = set(past[-1].get("pain_locations", [])) if past else set()
    prev_sevs = dict(past[-1].get("pain_severities", {})) if past else {}

    # ── MAP ──────────────────────────────────────────────────
    if pain_sub == "map":
        panel_q("Where do you feel pain? Tap to mark, then rate severity.")
        if prev_locs:
            st.markdown('<div class="small-note">🔴 = selected · 🟠 = pain last visit · 🟢 = none</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="small-note">Tap a body part to mark it painful, then rate severity.</div>',
                        unsafe_allow_html=True)

        col_svg, col_btns = st.columns([1, 1.4], gap="medium")
        with col_svg:
            st.markdown(body_svg(st.session_state.selected_parts, prev_locs), unsafe_allow_html=True)

        def _part_fu_q(part: str, sev: int) -> Optional[str]:
            """ONE inline question per body part — strict rules:
            - New location this visit: ask when it started
            - Severity ≥ 7: ask functional impact relevant to region
            - Severity jumped >2 vs last visit: ask what changed
            - Otherwise: None"""
            loc = part.lower()
            is_new = part not in prev_locs
            prev_sev = prev_sevs.get(part, 0)
            if is_new:
                return "When did this pain start?"
            if sev >= 7:
                if any(r in loc for r in ("head", "throat", "neck", "chest")):
                    return "Is it making it hard to swallow or eat?"
                if "abdomen" in loc:
                    return "Is it constant or does it come and go?"
                return "Is it limiting your movement?"
            if sev > prev_sev + 2:
                return f"Up from {prev_sev}→{sev}/10 — what do you think made it worse?"
            return None

        def _quick_replies_for(q: str) -> list:
            ql = q.lower()
            if "start" in ql or "when did" in ql:
                return ["Today", "2–3 days ago", "~1 week ago", "Over a week ago"]
            if "swallow" in ql or "eat" in ql:
                return ["Yes, hard to eat/swallow", "A little", "Not really"]
            if "constant" in ql or "come and go" in ql:
                return ["Constant", "Comes and goes"]
            if "movement" in ql or "limiting" in ql:
                return ["Yes, limits me", "A little", "Not really"]
            if "worse" in ql or "made it" in ql:
                return ["More activity", "Stress", "No idea"]
            return ["Yes", "Somewhat", "No"]

        BODY_PARTS = ["Head", "Throat/Neck", "Chest", "Abdomen",
                      "Left Arm", "Right Arm", "Left Leg", "Right Leg"]

        with col_btns:
            for part in BODY_PARTS:
                selected = part in st.session_state.selected_parts
                lbl = f"🔴 {part} ✓" if selected else (f"🟠 {part}" if part in prev_locs else f"🟢 {part}")
                if st.button(lbl, key=f"bp_{part}", use_container_width=True):
                    toggle_body_part(part)
                    st.rerun()

                if not selected:
                    continue

                # Severity slider
                default_sev = prev_sevs.get(part, 5)
                cur_sev = st.session_state.pain_severities.get(part, default_sev)
                sev_val = st.slider(f"Severity — {part}", 0, 10, cur_sev,
                                    key=f"sev_{part}", label_visibility="collapsed")
                st.session_state.pain_severities[part] = sev_val

                color = "#e63946" if sev_val >= 7 else "#f4a261" if sev_val >= 4 else "#2a9d8f"
                prev_v = prev_sevs.get(part)
                d = sev_val - prev_v if prev_v is not None else None
                delta = (f" ▲{d}" if d and d > 0 else f" ▼{abs(d)}" if d and d < 0 else " =same") if d is not None else ""

                # Compute and cache per-part followup
                q = _part_fu_q(part, sev_val)
                if q and part not in st.session_state.part_followup_q:
                    st.session_state.part_followup_q[part] = q
                elif not q and part in st.session_state.part_followup_q:
                    # Slider moved out of trigger zone — clear it unless already answered
                    if part not in st.session_state.part_followup_a:
                        st.session_state.part_followup_q.pop(part, None)

                stored_q = st.session_state.part_followup_q.get(part)
                stored_a = st.session_state.part_followup_a.get(part)

                if stored_q and not stored_a:
                    pills = "".join(
                        f'<a href="?{urllib.parse.urlencode({"pfu_part": part, "pfu_ans": r})}" '
                        f'target="_self" style="display:inline-block;padding:3px 9px;margin:2px 3px 0 0;'
                        f'background:#fff4ec;color:#b85c00;border:1.5px solid #f4a261;border-radius:10px;'
                        f'font-size:11px;font-weight:700;font-family:Nunito,sans-serif;text-decoration:none;">'
                        f'{r}</a>'
                        for r in _quick_replies_for(stored_q))
                    st.markdown(
                        f'<div style="margin-top:-10px;margin-bottom:4px;">'
                        f'<div style="text-align:right;font-size:11px;font-weight:800;color:{color};">'
                        f'{sev_val}/10{delta}</div>'
                        f'<div style="padding:6px 10px 8px;background:#fff8f2;'
                        f'border:1.5px solid #f4a261;border-radius:10px;">'
                        f'<div style="font-size:11px;color:#9a4e10;font-weight:600;margin-bottom:4px;">'
                        f'🩺 {stored_q}</div><div>{pills}</div></div></div>',
                        unsafe_allow_html=True)
                elif stored_q and stored_a:
                    st.markdown(
                        f'<div style="margin-top:-10px;margin-bottom:4px;">'
                        f'<div style="text-align:right;font-size:11px;font-weight:800;color:{color};">'
                        f'{sev_val}/10{delta}</div>'
                        f'<div style="padding:4px 10px;background:#fff8f2;'
                        f'border:1px solid rgba(244,162,97,0.35);border-radius:8px;'
                        f'font-size:11px;color:#b85c00;font-weight:700;">'
                        f'🩺 {stored_q} · ✓ {stored_a}</div></div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div style="margin-top:-10px;margin-bottom:4px;text-align:right;'
                        f'font-size:11px;font-weight:800;color:{color};">{sev_val}/10{delta}</div>',
                        unsafe_allow_html=True)

            if st.button("➕ Other location", key="bp_other", use_container_width=True):
                st.session_state.show_other[2] = True; st.rerun()

        if st.session_state.show_other.get(2):
            t = render_text_mic("bp_other_txt", "Describe location…")
            if t:
                st.session_state.selected_parts.add(t)
                st.session_state.show_other[2] = False; st.rerun()

        c_no, c_ok = st.columns([1, 2], gap="small")
        with c_no:
            if st.button("🙂 No pain", key="pain_none", use_container_width=True):
                st.session_state.pain_yesno = False
                st.session_state.selected_parts = set()
                st.session_state.pain_severities = {}
                st.session_state.part_followup_q = {}
                st.session_state.part_followup_a = {}
                add_patient("No pain today.", stage=2)
                advance_stage(); st.rerun()
        with c_ok:
            if st.button("Confirm locations ➜", key="pain_confirm",
                         use_container_width=True, type="primary"):
                if st.session_state.selected_parts:
                    st.session_state.pain_yesno = True
                    st.session_state.pain_sub = "timing"; st.rerun()
                else:
                    st.warning("Select at least one location, or click 'No pain'.")

    # ── TIMING ───────────────────────────────────────────────
    elif pain_sub == "timing":
        panel_q("Is your pain constant, or mainly at certain times?")
        for idx, (lbl, val) in enumerate([
            ("Constant", "constant"),
            ("When eating / swallowing", "eating"),
            ("When moving / standing", "movement"),
            ("Comes and goes", "intermittent"),
        ]):
            if st.button(lbl, key=f"pt_{idx}", use_container_width=True):
                st.session_state.pain_timing = val
                locs = sorted(st.session_state.selected_parts)
                sevs = st.session_state.pain_severities
                summary = ", ".join(f"{l} ({sevs.get(l,'?')}/10)" for l in locs)
                add_patient(f"Pain: {summary}. Timing: {val}.", stage=2)
                st.session_state.pain_sub = "done"; st.rerun()

    # ── DONE ─────────────────────────────────────────────────
    elif pain_sub == "done":
        # Log inline Q&As into message history exactly once
        if not st.session_state.part_fu_logged:
            for part, ans in st.session_state.part_followup_a.items():
                q = st.session_state.part_followup_q.get(part, "")
                if q and ans:
                    add_doctor(q, stage=2)
                    add_patient(ans, stage=2)
            st.session_state.part_fu_logged = True
        render_inline(2)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        if st.button("Next →", key="pain_next", use_container_width=True, type="primary"):
            advance_stage(); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 3 — Eating & Nutrition
# No GPT followup — the sub-stages already collect everything needed.
# ════════════════════════════════════════════════════════════
elif stage == 3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    eat_sub = st.session_state.eat_sub

    if eat_sub == "status":
        panel_q("How has eating been?")
        for idx, (lbl, val) in enumerate([
            ("👍 Better than before", "better"),
            ("➡️ About the same",     "same"),
            ("👎 Harder to eat",      "worse"),
            ("🚫 Can't eat",          "cant_eat"),
        ]):
            if st.button(lbl, key=f"eat_{idx}", use_container_width=True):
                st.session_state.eating_status = val
                add_patient(lbl.split(" ", 1)[1], stage=3)  # strip emoji
                st.session_state.eat_sub = "type" if val in ("worse", "cant_eat") else "done"
                st.rerun()

    elif eat_sub == "type":
        panel_q("What are you able to eat?")
        for idx, (lbl, val) in enumerate([
            ("Normal food", "normal"), ("Soft food / purees", "soft"),
            ("Liquids only", "liquids"), ("Tube feeding", "tube"),
        ]):
            if st.button(lbl, key=f"food_{idx}", use_container_width=True):
                st.session_state.food_type = val
                add_patient(lbl, stage=3)
                st.session_state.eat_sub = "shakes"; st.rerun()

    elif eat_sub == "shakes":
        panel_q("How many supplement shakes / Boosts today?")
        cols = st.columns(4, gap="small")
        for idx, (lbl, val) in enumerate([("None","0"),("1–2","1-2"),("3–4","3-4"),("5+","5+")]):
            with cols[idx]:
                if st.button(lbl, key=f"shk_{idx}", use_container_width=True):
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
                    add_patient(f"Hydration: {lbl}", stage=3)
                    st.session_state.eat_sub = "done"; st.rerun()

    elif eat_sub == "done":
        render_inline(3)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        if st.button("Next →", key="eat_next", use_container_width=True, type="primary"):
            advance_stage(); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 4 — Symptoms + ONE conditional followup
# GPT followup fires ONLY when:
#   - symptom is clinically urgent AND (new this visit OR always-urgent)
#   - exactly ONE question, no chaining
# ════════════════════════════════════════════════════════════
elif stage == 4:
    SYMPTOM_LIST = [
        "Fatigue / low energy", "Nausea", "Vomiting", "Mouth sores",
        "Trouble swallowing", "Constipation", "Diarrhea", "Fever / chills",
        "Dizziness / unsteadiness", "Numbness / tingling (hands or feet)",
        "Hearing changes / ringing in ears", "Coughing / choking", "Anxiety / low mood",
    ]
    past = st.session_state.get("past_checkins", [])
    prev_syms = set(past[-1].get("symptoms", [])) if past else set()

    # Has the patient confirmed symptoms yet?
    sym_confirmed = any(
        m["role"] == "patient" and m["content"].startswith("Symptoms:")
        for m in st.session_state.messages if m.get("stage") == 4
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    if not sym_confirmed:
        panel_q("Any of these symptoms today?")
        hint = "✓ = reported last visit. Tap to remove if resolved." if prev_syms else "Tap all that apply."
        st.markdown(f'<div class="small-note">{hint}</div>', unsafe_allow_html=True)

        sc = st.columns(2, gap="small")
        for idx, sym in enumerate(SYMPTOM_LIST):
            with sc[idx % 2]:
                active = sym in st.session_state.symptoms
                if st.button(f"✓ {sym}" if active else sym, key=f"sym_{idx}", use_container_width=True):
                    if active: st.session_state.symptoms.remove(sym)
                    else: st.session_state.symptoms.append(sym)
                    st.rerun()

        if st.button("➕ Other symptom", key="sym_other_btn", use_container_width=True):
            st.session_state.show_other[4] = True; st.rerun()

        if st.session_state.show_other.get(4):
            t = render_text_mic("sym_other", "Describe other symptom…")
            if t:
                if t not in st.session_state.symptoms:
                    st.session_state.symptoms.append(t)
                st.session_state.show_other[4] = False; st.rerun()

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        if st.button("Confirm symptoms ➜", key="sym_confirm",
                     use_container_width=True, type="primary"):
            sym_txt = "; ".join(st.session_state.symptoms) or "none"
            add_patient(f"Symptoms: {sym_txt}", stage=4)
            # Decide RIGHT NOW whether a followup is needed — store in slot
            st.session_state.sym_fu_q = get_symptom_followup()
            st.session_state.sym_fu_a = None
            st.rerun()

    else:
        render_inline(4)
        fq = st.session_state.sym_fu_q
        fa = st.session_state.sym_fu_a

        if fq and fa is None:
            # Show the ONE followup question + quick replies
            st.markdown(f'<div class="inline-followup">🩺 {fq}</div>', unsafe_allow_html=True)

            ql = fq.lower()
            if "temperature" in ql:
                replies = ["It was elevated", "Normal / not sure", "Haven't checked"]
            elif "times" in ql or "episodes" in ql or "how many" in ql:
                replies = ["1–2", "3–5", "More than 5"]
            elif "rest" in ql or "activity" in ql:
                replies = ["At rest", "Only with activity", "Both"]
            elif "stand" in ql or "worse when" in ql:
                replies = ["Yes, worse standing", "No difference"]
            elif "coughing" in ql or "choking" in ql:
                replies = ["Yes, often", "Sometimes", "No"]
            elif "bowel" in ql or "days since" in ql:
                replies = ["1–2 days", "3–4 days", "4+ days"]
            elif "eat or drink" in ql or "hard to eat" in ql or "sores" in ql:
                replies = ["Yes, makes it hard", "A little", "Not really"]
            elif "swallow" in ql or "choking" in ql:
                replies = ["Yes, coughing/choking", "Some difficulty", "No"]
            else:
                replies = ["Yes", "Somewhat", "No"]

            rc = st.columns(len(replies), gap="small")
            for i, r in enumerate(replies):
                with rc[i]:
                    if st.button(r, key=f"sym_fu_r{i}", use_container_width=True):
                        st.session_state.sym_fu_a = r
                        add_patient(r, stage=4)
                        st.rerun()
            st.markdown('<div class="small-note" style="margin-top:6px;">Or type your answer:</div>',
                        unsafe_allow_html=True)
            t = render_text_mic("sym_fu", "Type your answer…")
            if t:
                st.session_state.sym_fu_a = t
                add_patient(t, stage=4)
                st.rerun()

        else:
            # No followup needed, or already answered — show Next
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            if st.button("Continue →", key="sym_next", use_container_width=True, type="primary"):
                advance_stage(); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 5 — Submit + Summary
# GPT used ONLY here: extract free-text notes for care team.
# Strict prompt: facts only, no hallucination.
# ════════════════════════════════════════════════════════════
elif stage == 5:
    if st.session_state.submitted:
        name      = st.session_state.patient_name or "—"
        pain      = st.session_state.pain_yesno
        locations = sorted(st.session_state.selected_parts)
        sevs      = st.session_state.pain_severities
        timing    = st.session_state.pain_timing or "—"
        eating    = st.session_state.eating_status or "—"
        food      = st.session_state.food_type or "—"
        shakes    = st.session_state.shakes_per_day or "—"
        hydration = st.session_state.hydration or "—"
        symptoms  = st.session_state.symptoms
        is_fast   = st.session_state.fast_path
        carried   = ' <span style="font-size:11px;opacity:.55;">(from last visit)</span>'

        # On fast path, pull missing fields from last visit
        if is_fast:
            last = (st.session_state.past_checkins or [{}])[-1]
            if eating    == "—": eating    = last.get("eating") or "—"
            if food      == "—": food      = last.get("food_type") or "—"
            if shakes    == "—": shakes    = last.get("shakes_per_day") or "—"
            if hydration == "—": hydration = last.get("hydration") or "—"

        pain_str  = "Yes" if pain else ("No" if pain is False else "—")
        fast_row  = '<tr><td>Status</td><td><span class="tag">No changes</span></td></tr>' if is_fast else ""

        if locations:
            loc_html = "".join(f'<span class="tag">{l} ({sevs.get(l,"?")}/10)</span>' for l in locations)
            if is_fast: loc_html += carried
        else:
            loc_html = "None"

        sym_html = "".join(f'<span class="tag">{s}</span>' for s in symptoms) or "None"
        if is_fast and symptoms: sym_html += carried

        timing_row = f"<tr><td>Pain timing</td><td>{timing}</td></tr>" if pain else ""
        show_nut   = not (eating == "—" and food == "—" and shakes == "—" and hydration == "—")
        nut_rows   = ""
        if show_nut:
            def _d(v): return f"{v}{carried}" if is_fast and v != "—" else v
            nut_rows = (f"<tr><td>Eating</td><td>{_d(eating)}</td></tr>"
                        f"<tr><td>Diet type</td><td>{_d(food)}</td></tr>"
                        f"<tr><td>Shakes/day</td><td>{_d(shakes)}</td></tr>"
                        f"<tr><td>Hydration</td><td>{_d(hydration)}</td></tr>")

        # GPT: extract free-text patient notes ONLY (skip structured answers)
        STRUCTURED = {"No pain today.", "About the same.", "Things are worse.", "Better since last visit."}
        STRUCTURED_PREFIXES = ("Pain:", "Symptoms:", "Hydration:", "Shakes:", "Food:", "Eating:")
        free_lines = [
            m["content"] for m in st.session_state.messages
            if m["role"] == "patient"
            and m["content"] not in STRUCTURED
            and not any(m["content"].startswith(p) for p in STRUCTURED_PREFIXES)
        ]
        notes_cell = "<span style='opacity:.4'>—</span>"
        if free_lines and _openai_ready():
            try:
                resp = openai_client.chat.completions.create(
                    model=_secret("openai_model", default="gpt-4o-mini"),
                    temperature=0.0,
                    max_tokens=200,
                    messages=[
                        {"role": "system", "content": (
                            "You extract clinical notes from cancer patient check-in responses.\n"
                            "OUTPUT FORMAT: bullet points only, one per line, starting with •\n"
                            "INCLUDE ONLY: symptoms, durations, triggers, functional impacts, medication mentions.\n"
                            "DO NOT include: vague feelings, structured answers already captured in the form, "
                            "interpretations, diagnoses, or anything not explicitly stated.\n"
                            "If nothing clinically relevant: output exactly 'None'.\n"
                            "Maximum 5 bullets."
                        )},
                        {"role": "user", "content": "\n".join(f"- {l}" for l in free_lines)},
                    ],
                )
                txt = (resp.choices[0].message.content or "").strip()
                if txt.lower() != "none":
                    items = [l.lstrip("• -").strip() for l in txt.splitlines()
                             if l.strip() and l.strip().lower() != "none"]
                    if items:
                        notes_cell = ("<ul style='margin:0;padding-left:18px;'>" +
                                      "".join(f"<li style='font-size:14px;'>{i}</li>" for i in items) +
                                      "</ul>")
            except: pass

        st.markdown(f"""
<div class="summary-wrap">
  <div class="submitted-badge">✅ Submitted</div>
  <div class="summary-title">Check-In Summary — {name}</div>
  <div class="summary-sub">Your care team will review this.</div>
  <table class="summary-table">
    <tr><td>Patient</td><td>{name}</td></tr>
    {fast_row}
    <tr><td>Pain</td><td>{pain_str}</td></tr>
    <tr><td>Pain locations</td><td>{loc_html}</td></tr>
    {timing_row}
    {nut_rows}
    <tr><td>Symptoms</td><td>{sym_html}</td></tr>
    <tr><td>Notes</td><td>{notes_cell}</td></tr>
  </table>
</div>""", unsafe_allow_html=True)

    else:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        worst_sev = max(st.session_state.pain_severities.values(), default=0)
        s0_ans = next((m["content"] for m in st.session_state.messages
                       if m.get("stage") == 0 and m["role"] == "patient"), "")
        if "worse" in s0_ans.lower() or worst_sev >= 7:
            st.markdown('<div class="inline-followup" style="margin-bottom:10px;">🩺 '
                        "We're sorry you're having a hard time. "
                        "Your care team will review this right away.</div>", unsafe_allow_html=True)

        panel_q("Ready to submit — anything else to add?")
        st.markdown('<div class="small-note">Optional note for your care team:</div>',
                    unsafe_allow_html=True)
        note = st.text_input("", placeholder="e.g. Started a new medication yesterday…",
                             key="final_note", label_visibility="collapsed")
        if st.button("✅ Submit Check-In", use_container_width=True, type="primary"):
            if note and note.strip():
                add_patient(note.strip(), stage=5)
            try:
                save_to_sheet()
                st.session_state.submitted = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
