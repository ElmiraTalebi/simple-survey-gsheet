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
        if k in st.secrets:
            val = st.secrets[k]
            try: return dict(val) if hasattr(val, "keys") else val
            except: return val
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
            "pain_severity":    st.session_state.get("pain_severity"),
            "pain_timing":      st.session_state.get("pain_timing"),
            "eating":           st.session_state.get("eating_status"),
            "symptoms":         st.session_state.get("symptoms", []),
            "conversation":     st.session_state.get("messages", []),
            "fast_path":        st.session_state.get("fast_path", False),
        })
    ])

# ── Helpers ─────────────────────────────────────────────────
def _openai_ready():
    return openai_client is not None and openai_init_error is None

def get_opening_message(last: Dict, name: str) -> str:
    """Deterministic greeting — references most concerning item from last visit."""
    sevs = last.get("pain_severities", {})
    locs = last.get("pain_locations", [])
    syms = last.get("symptoms", [])
    sev  = last.get("pain_severity")  # overall severity

    if sevs and locs:
        worst_loc = max(sevs, key=lambda l: sevs[l])
        worst_sev = sevs[worst_loc]
        return (f"Hi {name}! Last time you had pain in your {worst_loc.lower()} "
                f"({worst_sev}/10). How is that — better, worse, or the same?")
    if sev and locs:
        return (f"Hi {name}! Last time you had pain ({sev}/10) in your "
                f"{', '.join(l.lower() for l in locs[:2])}. Better, worse, or the same?")
    if locs:
        return (f"Hi {name}! Last time you reported pain in your {locs[0].lower()}. "
                f"Better, worse, or the same?")
    priority = ["Fever / chills", "Vomiting", "Trouble swallowing"]
    for s in priority:
        if s in syms:
            return f"Hi {name}! Last time you reported {s.lower()}. Is that still bothering you?"
    if syms:
        return f"Hi {name}! Last time you reported {syms[0].lower()}. How are things today?"
    return f"Hi {name}! How have things been — better, worse, or the same?"

def transcribe_audio(audio_bytes: bytes) -> str:
    if not _openai_ready(): return "(Transcription unavailable.)"
    try:
        import io
        f = io.BytesIO(audio_bytes); f.name = "recording.wav"
        return (openai_client.audio.transcriptions.create(
            model=_secret("whisper_model", default="whisper-1"), file=f, language="en"
        ).text or "").strip()
    except Exception as e: return f"(Transcription failed: {e})"

# ── Curated Follow-Ups (only when concerning) ──────────────
SYMPTOM_FOLLOWUPS = {
    "Vomiting":                 "How many times in the last 24 hours?",
    "Trouble swallowing":       "Are you coughing or choking when you swallow?",
    "Fever / chills":           "Have you taken your temperature?",
    "Constipation":             "How many days since your last bowel movement?",
    "Diarrhea":                 "How many episodes today?",
    "Dizziness / unsteadiness": "Does it happen when you stand up?",
    "Shortness of breath":      "At rest or only with activity?",
}

def get_curated_followup(stage_id: int) -> Optional[str]:
    """ONE follow-up ONLY if concerning."""
    if stage_id == 2:  # pain
        sev = st.session_state.get("pain_severity", 0)
        if sev and sev >= 6:
            return "Is this pain making it hard to eat or do daily activities?"
        past = st.session_state.get("past_checkins", [])
        prev_locs = set(past[-1].get("pain_locations", [])) if past else set()
        new_locs = st.session_state.get("selected_parts", set()) - prev_locs
        if new_locs:
            return f"When did you first notice the pain in your {next(iter(new_locs)).lower()}?"
        return None

    if stage_id == 3:  # eating
        status = st.session_state.get("eating_status")
        if status == "worse":
            return "What's making it harder — pain, nausea, or no appetite?"
        if status == "cant_eat":
            return "Are you able to get any fluids or shakes down?"
        return None

    if stage_id == 4:  # symptoms
        symptoms = st.session_state.get("symptoms", [])
        priority = ["Fever / chills", "Vomiting", "Trouble swallowing",
                     "Shortness of breath", "Dizziness / unsteadiness"]
        for s in priority:
            if s in symptoms and s in SYMPTOM_FOLLOWUPS:
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
    font-size: 21px; box-shadow: 0 3px 12px rgba(42,157,143,0.30); flex-shrink: 0;
}
.app-header-title { font-family: 'Lora', serif; font-size: 21px; font-weight: 600; color: var(--text); }
.app-header-sub { font-size: 11px; color: var(--muted); font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; margin-top: 3px; }

.chat-window {
    max-height: 38vh; overflow-y: auto; padding: 14px; border-radius: var(--r-lg);
    background: var(--surface); border: 1.5px solid var(--border);
    box-shadow: var(--shadow-sm); margin-bottom: 14px;
}
.row-left  { display:flex; justify-content:flex-start; align-items:flex-end; margin:7px 0; gap:9px; }
.row-right { display:flex; justify-content:flex-end;   align-items:flex-end; margin:7px 0; gap:9px; }
.avatar {
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0;
    background: var(--accent-lt); border: 1.5px solid var(--accent-md);
}
.bubble-doc {
    background: var(--surface); border: 1.5px solid var(--border);
    border-radius: var(--r-md); border-bottom-left-radius: 4px;
    padding: 10px 14px; max-width: 78%; box-shadow: var(--shadow-sm);
    font-size: 14px; line-height: 1.6; color: var(--text); white-space: pre-wrap;
}
.bubble-pat {
    background: var(--patient); color: #fff;
    border-radius: var(--r-md); border-bottom-right-radius: 4px;
    padding: 10px 14px; max-width: 78%; box-shadow: var(--shadow-sm);
    font-size: 14px; line-height: 1.6; white-space: pre-wrap;
}

.panel { background: transparent; border: none; padding: 0 0 10px; box-shadow: none; }
.panel-title {
    display: flex; align-items: flex-end; gap: 9px; margin-bottom: 14px;
    font-family: 'Nunito', sans-serif !important; font-size: 14px;
}
.panel-title-avatar {
    width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 14px;
    background: var(--accent-lt); border: 1.5px solid var(--accent-md);
}
.panel-title-bubble {
    background: var(--surface); border: 1.5px solid var(--border);
    border-radius: var(--r-md); border-bottom-left-radius: 4px;
    padding: 10px 15px; box-shadow: var(--shadow-sm);
    font-family: 'Nunito', sans-serif !important;
    font-size: 15px; font-weight: 600; color: var(--text); line-height: 1.5; max-width: 82%;
}
.small-note { font-size: 12px; color: var(--muted); font-weight: 500; margin: 0 0 10px; }
.divider { border: none; border-top: 1.5px solid var(--border); margin: 14px 0 12px; }
.inline-followup {
    background: var(--accent-lt); border-left: 3px solid var(--accent);
    border-radius: 0 var(--r-sm) var(--r-sm) 0;
    padding: 11px 14px; margin: 12px 0 8px; font-size: 14px; line-height: 1.6; color: var(--text);
}
.inline-patient {
    background: rgba(38,70,83,0.07);
    border-radius: var(--r-sm) var(--r-sm) 0 var(--r-sm);
    padding: 9px 13px; margin: 6px 0; text-align: right; font-size: 14px; color: var(--text);
}

.stButton > button {
    font-family: 'Nunito', sans-serif !important; border-radius: var(--r-sm) !important;
    padding: 0.45rem 1rem !important; font-size: 14px !important; font-weight: 600 !important;
    border: 1.5px solid var(--border) !important; background: var(--surface) !important;
    color: var(--text) !important; box-shadow: var(--shadow-sm) !important;
    transition: all 0.14s ease !important; white-space: nowrap !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important; color: var(--accent) !important;
    background: var(--accent-lt) !important; transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent) 0%, #1d7a6e 100%) !important;
    color: #fff !important; border-color: transparent !important;
    box-shadow: 0 3px 14px rgba(42,157,143,0.32) !important;
    font-size: 15px !important; padding: 0.55rem 1rem !important;
}

[data-testid="stAudioInput"] { margin: 0 !important; padding: 0 !important; }
[data-testid="stAudioInput"] > label { display: none !important; }
[data-testid="stAudioInput"] > div {
    height: 38px !important; min-height: 38px !important; border-radius: var(--r-sm) !important;
    border: 1.5px solid var(--border) !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    background: var(--surface) !important; width: 100% !important; overflow: hidden !important;
}
[data-testid="stAudioInput"] > div > * { transform: scale(0.88); transform-origin: center; }
[data-testid="stTextInput"] > div > div > input {
    font-family: 'Nunito', sans-serif !important; border-radius: var(--r-sm) !important;
    border: 1.5px solid var(--border) !important; padding: 7px 14px !important;
    font-size: 14px !important; background: var(--surface) !important; height: 38px !important;
    box-shadow: var(--shadow-sm) !important; color: var(--text) !important;
}
[data-testid="stTextInput"] > label { display: none !important; }
[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton) { position: relative !important; }
[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton)
  [data-testid="stElementContainer"]:has(.stButton) {
    position: absolute !important; right: 8px !important; top: 4px !important;
    z-index: 2 !important; width: auto !important;
}
[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton) .stButton button {
    width: 30px !important; height: 30px !important; min-height: 30px !important;
    padding: 0 !important; border-radius: 50% !important;
    background: linear-gradient(135deg, var(--accent) 0%, #1d7a6e 100%) !important;
    color: #fff !important; border: none !important; font-size: 15px !important; font-weight: 700 !important;
}
[data-testid="stColumn"]:has(.stButton) [data-testid="stTextInput"] > div > div > input {
    padding-right: 42px !important;
}

.summary-wrap {
    background: var(--surface); border: 1.5px solid var(--border);
    border-radius: var(--r-lg); padding: 26px 22px 20px;
    box-shadow: var(--shadow-md);
}
.summary-title { font-family: 'Lora', serif; font-size: 20px; font-weight: 600; color: var(--text); margin-bottom: 4px; }
.summary-sub { font-size: 11px; color: var(--muted); font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 18px; }
.summary-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.summary-table tr { border-bottom: 1.5px solid var(--border); }
.summary-table tr:last-child { border-bottom: none; }
.summary-table td { padding: 11px 8px; vertical-align: top; line-height: 1.5; }
.summary-table td:first-child { font-weight: 700; color: var(--muted); width: 36%;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; padding-top: 14px; }
.tag { display: inline-block; background: var(--accent-lt); color: var(--accent);
    border: 1px solid var(--accent-md); border-radius: 8px;
    padding: 2px 10px; font-size: 13px; font-weight: 600; margin: 2px 3px 2px 0; }
.submitted-badge { display: inline-flex; align-items: center; gap: 6px;
    background: #d1fae5; color: #065f46; border: 1.5px solid #6ee7b7; border-radius: 10px;
    padding: 5px 14px; font-size: 13px; font-weight: 700; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────
# SIMPLIFIED STAGES:
#   -1 = name entry
#    0 = "better / same / worse?" (returning patients)
#    2 = pain: body map → severity → timing (one flow, no per-part follow-ups)
#    3 = eating (one question, drill only if concerning)
#    4 = symptom checklist
#    5 = submit
defaults = {
    "messages": [], "stage": -1, "patient_name": "",
    "selected_parts": set(), "pain_yesno": None, "feeling_level": None,
    "symptoms": [], "submitted": False, "past_checkins": [],
    "last_audio_hash": None, "mic_key_counter": 0,
    "stage_answered": {}, "followup_fired": {},
    "pain_severity": None, "pain_timing": None,
    "eating_status": None,
    "show_other": {},
    "fast_path": False,
    "pain_sub": "map",
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
    if st.session_state.fast_path and s == 0:
        st.session_state.stage = 5; return
    if   s == 0: st.session_state.stage = 2
    elif s == 1: st.session_state.stage = 2
    elif s == 2: st.session_state.stage = 3
    elif s == 3: st.session_state.stage = 4
    elif s == 4: st.session_state.stage = 5

def on_answer(text: str, stage_id: int):
    add_patient(text, stage=stage_id)
    mark_answered(stage_id)
    if not followup_fired(stage_id):
        fu = get_curated_followup(stage_id)
        if fu:
            mark_followup_fired(stage_id)
            add_doctor(fu, stage=stage_id)

def on_followup_reply(text: str, stage_id: int):
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
        if msg["role"] == "doctor":
            st.markdown(f'<div class="row-left"><div class="avatar">🩺</div>'
                        f'<div class="bubble-doc">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="row-right"><div class="bubble-pat">{msg["content"]}</div>'
                        f'<div class="avatar">🙂</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_inline(stage_id: int):
    for msg in [m for m in st.session_state.messages if m.get("stage") == stage_id]:
        if msg["role"] == "doctor":
            st.markdown(f'<div class="inline-followup">🩺 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="inline-patient">{msg["content"]} 🙂</div>', unsafe_allow_html=True)

def render_followup_input(stage_id: int):
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

def body_svg(selected: Set[str], prev_locs: Set[str] = set()) -> str:
    def fill(p):
        if p in selected: return "#e63946"
        if p in prev_locs: return "#f4a261"
        return "#a8d5b5"
    def stroke(p):
        if p in selected: return "#b52535"
        if p in prev_locs: return "#c47a3a"
        return "#6daa82"
    s = "#6b7a90"
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

def panel_q(text):
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
            with st.spinner("Loading…"):
                st.session_state.past_checkins = load_past_checkins(name_input.strip())
            past = st.session_state.past_checkins
            if past:
                last = past[-1]
                st.session_state.selected_parts = set(last.get("pain_locations", []))
                st.session_state.symptoms = list(last.get("symptoms", []))
                opening = get_opening_message(last, name_input.strip())
                add_doctor(opening, stage=0)
                st.session_state.stage = 0
            else:
                add_doctor(f"Hi {name_input.strip()}! Let's go through a few quick questions.", stage=2)
                st.session_state.stage = 2
            st.rerun()
        else:
            st.warning("Please enter your name.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

stage = st.session_state.stage
render_chat_window()

# ════════════════════════════════════════════════════════════
# STAGE 0 — Quick status (better / same / worse)
# ════════════════════════════════════════════════════════════
if stage == 0:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    render_inline(0)

    if not is_answered(0):
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            if st.button("👍 Better", use_container_width=True, key="s0_better"):
                st.session_state.fast_path = True
                on_answer("Better.", 0); advance_stage(); st.rerun()
        with c2:
            if st.button("➡️ Same", use_container_width=True, key="s0_same"):
                st.session_state.fast_path = True
                on_answer("About the same.", 0); advance_stage(); st.rerun()
        with c3:
            if st.button("⚠️ Worse", use_container_width=True, key="s0_worse"):
                st.session_state.fast_path = False
                on_answer("Worse.", 0); advance_stage(); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 2 — Pain (simplified: map → one severity → timing → auto-advance)
# ════════════════════════════════════════════════════════════
elif stage == 2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    pain_sub = st.session_state.pain_sub

    if pain_sub == "map":
        panel_q("Where do you feel pain? Tap to select.")
        past = st.session_state.get("past_checkins", [])
        prev_locs = set(past[-1].get("pain_locations", [])) if past else set()

        if prev_locs:
            st.markdown('<div class="small-note">🟠 = last visit · 🔴 = selected · 🟢 = no pain</div>',
                        unsafe_allow_html=True)

        col_svg, col_btns = st.columns([1, 1], gap="medium")
        with col_svg:
            st.markdown(body_svg(st.session_state.selected_parts, prev_locs), unsafe_allow_html=True)
        with col_btns:
            for part in ["Head","Throat/Neck","Chest","Abdomen","Left Arm","Right Arm","Left Leg","Right Leg"]:
                sel = part in st.session_state.selected_parts
                lbl = f"🔴 {part}" if sel else (f"🟠 {part}" if part in prev_locs else f"🟢 {part}")
                if st.button(lbl, key=f"bp_{part}", use_container_width=True):
                    toggle_body_part(part); st.rerun()

        c_no, c_confirm = st.columns([1, 2], gap="small")
        with c_no:
            if st.button("🙂 No pain", key="pain_none", use_container_width=True):
                st.session_state.pain_yesno = False
                st.session_state.selected_parts = set()
                on_answer("No pain today.", 2); advance_stage(); st.rerun()
        with c_confirm:
            if st.button("Confirm ➜", key="pain_confirm", use_container_width=True, type="primary"):
                if st.session_state.selected_parts:
                    st.session_state.pain_yesno = True
                    st.session_state.pain_sub = "severity"; st.rerun()
                else:
                    st.warning("Select at least one location, or tap 'No pain'.")

    elif pain_sub == "severity":
        locs = sorted(st.session_state.selected_parts)
        panel_q(f"How bad is your pain overall? ({', '.join(locs)})")
        sev = st.slider("Overall pain severity", 0, 10, 5, key="pain_sev_slider",
                         label_visibility="collapsed")
        st.session_state.pain_severity = sev
        color = "#e63946" if sev >= 7 else "#f4a261" if sev >= 4 else "#2a9d8f"
        st.markdown(f'<div style="text-align:center;font-size:24px;font-weight:800;color:{color};">'
                    f'{sev}/10</div>', unsafe_allow_html=True)

        for idx, (lbl, val) in enumerate([
            ("Constant", "constant"),
            ("When eating/swallowing", "eating"),
            ("When moving", "movement"),
            ("Comes and goes", "intermittent"),
        ]):
            if st.button(lbl, key=f"pt_{idx}", use_container_width=True):
                st.session_state.pain_timing = val
                summary = ", ".join(locs)
                on_answer(f"Pain in {summary}, severity {sev}/10, timing: {val}.", 2)
                st.session_state.pain_sub = "done"; st.rerun()

    elif pain_sub == "done":
        render_inline(2)
        stage_msgs = [m for m in st.session_state.messages if m.get("stage") == 2]
        last_is_doc = stage_msgs and stage_msgs[-1]["role"] == "doctor"
        if last_is_doc:
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
# STAGE 3 — Eating (ONE question, drill only if concerning)
# ════════════════════════════════════════════════════════════
elif stage == 3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_q("How has eating been?")

    if not is_answered(3):
        render_inline(3)
        for idx, (lbl, val) in enumerate([
            ("👍 Good / better", "better"),
            ("➡️ About the same", "same"),
            ("👎 Harder to eat", "worse"),
            ("🚫 Can barely eat", "cant_eat"),
        ]):
            if st.button(lbl, key=f"eat_{idx}", use_container_width=True):
                st.session_state.eating_status = val
                on_answer(f"Eating: {lbl.split(' ', 1)[1]}", 3)
                st.rerun()
    else:
        render_inline(3)
        stage_msgs = [m for m in st.session_state.messages if m.get("stage") == 3]
        last_is_doc = stage_msgs and stage_msgs[-1]["role"] == "doctor"
        if last_is_doc:
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
# STAGE 4 — Symptom checklist
# ════════════════════════════════════════════════════════════
elif stage == 4:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    panel_q("Any of these symptoms today?")

    SYMPTOM_LIST = [
        "Fatigue / low energy", "Nausea", "Vomiting", "Mouth sores",
        "Trouble swallowing", "Constipation", "Diarrhea", "Fever / chills",
        "Dizziness / unsteadiness", "Numbness / tingling",
        "Hearing changes / tinnitus", "Coughing / choking", "Anxiety / low mood",
    ]

    past = st.session_state.get("past_checkins", [])
    prev_syms = set(past[-1].get("symptoms", [])) if past else set()

    if not is_answered(4):
        render_inline(4)
        if prev_syms:
            st.markdown('<div class="small-note">✓ = from last visit. Tap to remove if resolved.</div>',
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

        if st.button("➕ Other", key="sym_other", use_container_width=True):
            st.session_state.show_other[4] = True; st.rerun()
        if st.session_state.show_other.get(4):
            ctr = st.session_state.mic_key_counter
            other_sym = st.text_input("", placeholder="Describe…",
                                      key=f"other_sym_{ctr}", label_visibility="collapsed")
            if st.button("Add", key=f"other_sym_add_{ctr}"):
                if other_sym and other_sym.strip():
                    st.session_state.symptoms.append(other_sym.strip())
                    st.session_state.show_other[4] = False; st.rerun()

        if st.button("Confirm ➜", key="sym_confirm", use_container_width=True, type="primary"):
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
        sev       = st.session_state.get("pain_severity")
        symptoms  = st.session_state.get("symptoms", [])
        timing    = st.session_state.get("pain_timing") or "—"
        eating    = st.session_state.get("eating_status") or "—"
        is_fast   = st.session_state.get("fast_path", False)

        pain_str = "Yes" if pain else ("No" if pain is False else "—")
        loc_html = ", ".join(locations) if locations else "N/A"
        sev_str = f"{sev}/10" if sev is not None else "—"
        sym_html = "".join(f'<span class="tag">{s}</span>' for s in symptoms) or "None"
        carried = ' <span style="font-size:11px;opacity:.5;">(from last visit)</span>'

        fast_note = ""
        if is_fast:
            fast_note = '<tr><td>Status</td><td><span class="tag">No changes</span></td></tr>'
            if loc_html != "N/A": loc_html += carried
            if sym_html != "None": sym_html += carried

        st.markdown(f"""
<div class="summary-wrap">
  <div class="submitted-badge">✅ Submitted</div>
  <div class="summary-title">Check-In Summary — {name}</div>
  <div class="summary-sub">Your care team will review this.</div>
  <table class="summary-table">
    <tr><td>Patient</td><td>{name}</td></tr>
    {fast_note}
    <tr><td>Pain</td><td>{pain_str}</td></tr>
    <tr><td>Locations</td><td>{loc_html}</td></tr>
    <tr><td>Severity</td><td>{sev_str}</td></tr>
    <tr><td>Timing</td><td>{timing}</td></tr>
    <tr><td>Eating</td><td>{eating}</td></tr>
    <tr><td>Symptoms</td><td>{sym_html}</td></tr>
  </table>
</div>""", unsafe_allow_html=True)

    else:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_q("Ready to submit — anything else?")
        note = st.text_input("", placeholder="Optional note for your care team…",
                             key="final_note", label_visibility="collapsed")
        if st.button("✅ Submit Check-In", use_container_width=True, type="primary"):
            if note and note.strip():
                add_patient(note.strip(), stage=5)
            try:
                save_to_sheet(); st.session_state.submitted = True; st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
