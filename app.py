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
            "feeling_level":  st.session_state.feeling_level,
            "pain":           st.session_state.pain_yesno,
            "pain_locations": sorted(list(st.session_state.selected_parts)),
            "symptoms":       st.session_state.symptoms,
            "conversation":   st.session_state.messages,
        })
    ])

# ── Prompting ───────────────────────────────────────────────
def build_system_prompt(extra_context: str = "") -> str:
    name     = st.session_state.get("patient_name", "the patient")
    feeling  = st.session_state.get("feeling_level", None)
    pain     = st.session_state.get("pain_yesno", None)
    locs     = sorted(list(st.session_state.get("selected_parts", set())))
    symptoms = st.session_state.get("symptoms", [])

    lines = []
    if feeling  is not None: lines.append(f"- Feeling: {feeling}")
    if pain     is not None: lines.append(f"- Pain: {'yes' if pain else 'no'}")
    if locs:                 lines.append(f"- Pain locations: {', '.join(locs)}")
    if symptoms:             lines.append(f"- Symptoms: {', '.join(symptoms)}")
    session_str = "\n".join(lines) if lines else "Nothing collected yet."

    past = st.session_state.get("past_checkins", [])
    if past:
        mem = [f"  [{p.get('timestamp','?')}] Feeling:{p.get('feeling_level','?')}/10 | "
               f"Pain:{'yes' if p.get('pain') else 'no'} | "
               f"Locations:{', '.join(p.get('pain_locations',[])) or 'none'} | "
               f"Symptoms:{', '.join(p.get('symptoms',[])) or 'none'}" for p in past]
        memory_str = "\n".join(mem)
    else:
        memory_str = "No previous check-ins."

    ctx = f"\nCURRENT TASK:\n{extra_context}\n" if extra_context else ""

    return f"""You are a virtual symptom-intake assistant for a cancer care clinic.
Daily check-in with: {name}.

TODAY'S DATA: {session_str}
PATIENT HISTORY: {memory_str}
{ctx}
STRICT OUTPUT RULES:
1. Output ONLY a single short question. Nothing else.
2. NEVER start with "Thank you", "I'm sorry", "Great", "I see", "I understand",
   "Of course", "That's", "I'm glad", or ANY acknowledgement whatsoever.
3. No preamble. No explanation. Just the question.
4. A very brief empathetic lead-in (max 6 words) is allowed only when the patient
   shares something emotionally significant — otherwise skip it entirely and go
   straight to the question.
5. "Let's stay focused" is ONLY for genuinely off-topic messages (e.g. jokes, news,
   questions about you). NEVER use it for normal health-related replies.
6. Never give medical advice. If asked: "Your care team will follow up."
"""

def _openai_ready():
    return openai_client is not None and openai_init_error is None

def get_opening_message(last: Dict, name: str) -> str:
    """
    Generate the opening history-recap message with a lighter, non-restrictive prompt.
    Explicitly names the patient's last-visit data and asks a targeted question about it.
    """
    if not _openai_ready():
        return f"Hi {name}! Good to see you again. How have you been since your last visit?"

    fl   = last.get("feeling_level", "?")
    pn   = "yes" if last.get("pain") else "no"
    ploc = ", ".join(last.get("pain_locations", [])) or "none"
    sym  = ", ".join(last.get("symptoms", [])) or "none"
    ts   = last.get("timestamp", "your last visit")

    # Pick the most specific thing to follow up on
    if last.get("symptoms"):
        focus = f"ask specifically how their {sym} have been — better, worse, or the same?"
    elif last.get("pain") and last.get("pain_locations"):
        focus = f"ask specifically about their {ploc} pain — is it still there, how severe?"
    elif fl not in ("?", None) and str(fl).isdigit() and int(fl) <= 4:
        focus = f"ask what was contributing to their low feeling score of {fl}/10 and whether it has changed"
    else:
        focus = f"ask how they have been feeling since then"

    system = (
        "You are a warm, empathetic virtual symptom-intake assistant for a cancer care clinic. "
        "Write a short opening message (2–3 sentences max). "
        "Sentence 1: greet the patient by name and briefly mention their SPECIFIC data from last time "
        "(name the actual symptoms, pain locations, or feeling score — do NOT be generic). "
        "Sentence 2-3: " + focus + ". "
        "Do NOT say 'Thank you', 'Great', 'I see', or any filler. Be warm but direct."
    )
    try:
        r = openai_client.chat.completions.create(
            model=_secret("openai_model", default="gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Patient name: {name}. Last visit: {ts}. "
                                               f"Feeling: {fl}/10. Pain: {pn}. "
                                               f"Locations: {ploc}. Symptoms: {sym}."}
            ],
            max_tokens=120, temperature=0.5,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Hi {name}! Last time you reported {sym if sym != 'none' else 'some symptoms'}. How have things been since then?"

def get_gpt_reply(extra_context: str = "") -> str:
    if not _openai_ready():
        return "(Assistant unavailable — check OpenAI API key.)"
    msgs = [{"role": "system", "content": build_system_prompt(extra_context)}]
    for p in st.session_state.get("past_checkins", []):
        ts = p.get("timestamp","?"); fl = p.get("feeling_level","?")
        pn = "yes" if p.get("pain") else "no"
        locs = ", ".join(p.get("pain_locations",[])) or "none"
        syms = ", ".join(p.get("symptoms",[])) or "none"
        msgs.append({"role":"user",      "content": f"[Past visit {ts}] Feeling:{fl}/10. Pain:{pn}. Locations:{locs}. Symptoms:{syms}."})
        msgs.append({"role":"assistant", "content": f"Noted your check-in from {ts}."})
    for m in st.session_state.messages[-20:]:
        msgs.append({"role": "assistant" if m.get("role")=="doctor" else "user",
                     "content": m.get("content","")})
    try:
        r = openai_client.chat.completions.create(
            model=_secret("openai_model", default="gpt-4o-mini"),
            messages=msgs, max_tokens=120, temperature=0.5,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(Error: {e})"

def transcribe_audio(audio_bytes: bytes) -> str:
    if not _openai_ready(): return "(Transcription unavailable.)"
    try:
        import io
        f = io.BytesIO(audio_bytes); f.name = "recording.wav"
        return (openai_client.audio.transcriptions.create(
            model=_secret("whisper_model", default="whisper-1"), file=f, language="en"
        ).text or "").strip()
    except Exception as e: return f"(Transcription failed: {e})"

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Lora:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
<style>
:root {
    --bg:        #f5f3ef;
    --surface:   #ffffff;
    --border:    rgba(180,170,155,0.35);
    --accent:    #2a9d8f;
    --accent-lt: rgba(42,157,143,0.10);
    --accent-md: rgba(42,157,143,0.22);
    --patient:   #264653;
    --text:      #2c2c2c;
    --text-muted:#888079;
    --shadow-sm: 0 2px 10px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 24px rgba(0,0,0,0.09);
    --r-sm: 14px;
    --r-md: 20px;
    --r-lg: 26px;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Nunito', sans-serif !important;
    background: var(--bg) !important;
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; z-index: -1;
    background:
        radial-gradient(ellipse 70% 50% at 90% 10%, rgba(42,157,143,0.10) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 10% 90%, rgba(38,70,83,0.07) 0%, transparent 70%),
        #f5f3ef;
}
[data-testid="stHeader"]{ background:transparent !important; }
[data-testid="stDecoration"]{ display:none !important; }
[data-testid="stMainBlockContainer"]{ padding-top: 2rem !important; }
.block-container{ max-width: 680px !important; padding: 0 1.2rem 3rem !important; }

/* ── Page header ── */
.app-header {
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 28px; padding-bottom: 20px;
    border-bottom: 1.5px solid var(--border);
}
.app-header-icon {
    width: 48px; height: 48px; border-radius: 14px;
    background: linear-gradient(135deg, var(--accent), #21867a);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; box-shadow: var(--shadow-sm);
    flex-shrink: 0;
}
.app-header-title {
    font-family: 'Lora', serif; font-size: 22px; font-weight: 600;
    color: var(--text); line-height: 1.2; letter-spacing: -0.3px;
}
.app-header-sub {
    font-size: 12px; color: var(--text-muted); font-weight: 500;
    letter-spacing: 0.04em; text-transform: uppercase; margin-top: 2px;
}

/* ── Chat history window ── */
.chat-window {
    max-height: 38vh; overflow-y: auto;
    padding: 16px 14px; border-radius: var(--r-lg);
    background: var(--surface);
    border: 1.5px solid var(--border);
    box-shadow: var(--shadow-sm);
    margin-bottom: 16px;
    scrollbar-width: thin;
    scrollbar-color: rgba(0,0,0,0.12) transparent;
}
.chat-window::-webkit-scrollbar { width: 4px; }
.chat-window::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.12); border-radius: 4px; }

.row-left  { display:flex; justify-content:flex-start; align-items:flex-end; margin:8px 0; gap:10px; }
.row-right { display:flex; justify-content:flex-end;   align-items:flex-end; margin:8px 0; gap:10px; }

.avatar {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0;
    background: var(--accent-lt);
    border: 1.5px solid var(--accent-md);
}

.bubble-doc {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: var(--r-md); border-bottom-left-radius: 5px;
    padding: 10px 14px; max-width: 78%;
    box-shadow: var(--shadow-sm);
    font-size: 14px; line-height: 1.6; color: var(--text);
    white-space: pre-wrap;
    animation: fadeUp 0.25s ease both;
}
.bubble-pat {
    background: var(--patient);
    border-radius: var(--r-md); border-bottom-right-radius: 5px;
    padding: 10px 14px; max-width: 78%;
    box-shadow: var(--shadow-sm);
    font-size: 14px; line-height: 1.6; color: #fff;
    white-space: pre-wrap;
    animation: fadeUp 0.25s ease both;
}

/* ── Active stage panel ── */
.panel {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: var(--r-lg);
    padding: 22px 20px 18px;
    box-shadow: var(--shadow-md);
    margin-top: 0;
    animation: fadeUp 0.3s ease both;
}
.panel-title {
    font-family: 'Lora', serif;
    font-size: 17px; font-weight: 600;
    color: var(--text); margin-bottom: 14px;
    line-height: 1.3; letter-spacing: -0.2px;
}
.small-note {
    font-size: 12px; color: var(--text-muted);
    font-weight: 500; margin: 0 0 10px 0;
    letter-spacing: 0.02em;
}
.divider {
    border: none;
    border-top: 1.5px solid var(--border);
    margin: 16px 0 12px;
}

/* ── Inline follow-up messages inside panel ── */
.inline-followup {
    background: var(--accent-lt);
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--r-sm) var(--r-sm) 0;
    padding: 11px 14px; margin: 12px 0 8px;
    font-size: 14px; line-height: 1.6; color: var(--text);
    animation: fadeUp 0.25s ease both;
}
.inline-patient {
    background: rgba(38,70,83,0.07);
    border-radius: var(--r-sm) var(--r-sm) 0 var(--r-sm);
    padding: 9px 13px; margin: 6px 0;
    font-size: 14px; line-height: 1.6; color: var(--text);
    text-align: right;
    animation: fadeUp 0.2s ease both;
}

/* ── Buttons — all uniform, pill-shaped ── */
.stButton > button {
    font-family: 'Nunito', sans-serif !important;
    border-radius: 12px !important;
    padding: 0.45rem 1rem !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    border: 1.5px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.15s ease !important;
    white-space: nowrap !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-lt) !important;
    box-shadow: 0 2px 12px rgba(42,157,143,0.18) !important;
    transform: translateY(-1px) !important;
}
/* Primary (Next) button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), #21867a) !important;
    color: white !important;
    border-color: transparent !important;
    box-shadow: 0 3px 14px rgba(42,157,143,0.35) !important;
    font-size: 15px !important;
    padding: 0.55rem 1rem !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #33b09f, #1d7269) !important;
    color: white !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px rgba(42,157,143,0.45) !important;
}

/* ── Mic widget ── */
[data-testid="stAudioInput"] { margin: 0 !important; padding: 0 !important; }
[data-testid="stAudioInput"] > label { display: none !important; }
[data-testid="stAudioInput"] > div {
    height: 38px !important; min-height: 38px !important;
    border-radius: 12px !important;
    border: 1.5px solid var(--border) !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color 0.15s !important;
}
[data-testid="stAudioInput"] > div:hover {
    border-color: var(--accent) !important;
}

/* ── Text inputs ── */
[data-testid="stTextInput"] > div > div > input {
    font-family: 'Nunito', sans-serif !important;
    border-radius: 12px !important;
    border: 1.5px solid var(--border) !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    background: var(--surface) !important;
    height: 38px !important;
    box-shadow: var(--shadow-sm) !important;
    color: var(--text) !important;
    transition: border-color 0.15s !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-lt) !important;
}
[data-testid="stTextInput"] > label { display: none !important; }

/* ── Chat input (stage 5) ── */
[data-testid="stChatInput"] textarea {
    font-family: 'Nunito', sans-serif !important;
    border-radius: 12px !important;
    border: 1.5px solid var(--border) !important;
    background: var(--surface) !important;
}

/* ── Summary card ── */
.summary-wrap {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: var(--r-lg);
    padding: 28px 24px 22px;
    box-shadow: var(--shadow-md);
    animation: fadeUp 0.4s ease both;
}
.summary-title {
    font-family: 'Lora', serif;
    font-size: 20px; font-weight: 600;
    color: var(--text); margin-bottom: 4px;
    letter-spacing: -0.3px;
}
.summary-sub {
    font-size: 12px; color: var(--text-muted);
    font-weight: 500; margin-bottom: 20px;
    letter-spacing: 0.04em; text-transform: uppercase;
}
.summary-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.summary-table tr { border-bottom: 1.5px solid var(--border); }
.summary-table tr:last-child { border-bottom: none; }
.summary-table td { padding: 11px 8px; vertical-align: top; line-height: 1.5; }
.summary-table td:first-child {
    font-weight: 700; color: var(--text-muted);
    width: 36%; font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.06em; padding-top: 14px;
}
.tag {
    display: inline-block;
    background: var(--accent-lt);
    color: var(--accent);
    border: 1px solid var(--accent-md);
    border-radius: 8px;
    padding: 2px 10px; font-size: 13px; font-weight: 600;
    margin: 2px 3px 2px 0;
}
.submitted-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #d1fae5; color: #065f46;
    border: 1.5px solid #6ee7b7;
    border-radius: 10px; padding: 5px 14px;
    font-size: 13px; font-weight: 700;
    margin-bottom: 16px; letter-spacing: 0.02em;
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Streamlit chrome cleanup ── */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebarNav"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────
# Stages:
#  -1 = name entry
#   0 = history recap (skip if no history)
#   1 = feeling scale
#   2 = pain yes/no
#   3 = body pain map (skip if no pain)
#   4 = symptom checklist
#   5 = free chat + submit
#
# KEY DESIGN: every message is tagged with the stage it belongs to.
#   {"role": "doctor"|"patient", "content": "...", "stage": 1}
# The top chat window shows messages from COMPLETED stages only.
# Messages from the CURRENT stage are rendered inline inside the panel.
defaults = {
    "messages": [], "stage": -1, "patient_name": "",
    "selected_parts": set(), "pain_yesno": None, "feeling_level": None,
    "symptoms": [], "submitted": False, "past_checkins": [],
    "last_audio_hash": None, "mic_key_counter": 0,
    "followup_counts": {}, "stage_answered": {},
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

def toggle_body_part(part):
    if part in st.session_state.selected_parts: st.session_state.selected_parts.remove(part)
    else: st.session_state.selected_parts.add(part)

MAX_FOLLOWUPS = {0:1, 1:3, 2:3, 3:3, 4:3}

def followup_count(sid): return st.session_state.followup_counts.get(sid, 0)
def can_followup(sid):   return followup_count(sid) < MAX_FOLLOWUPS.get(sid, 0)
def record_followup(sid): st.session_state.followup_counts[sid] = followup_count(sid)+1
def is_answered(sid):    return st.session_state.stage_answered.get(sid, False)
def mark_answered(sid):  st.session_state.stage_answered[sid] = True

def advance_stage():
    s = st.session_state.stage
    if   s == 0: st.session_state.stage = 1
    elif s == 1: st.session_state.stage = 2
    elif s == 2: st.session_state.stage = 4 if st.session_state.pain_yesno is False else 3
    elif s == 3: st.session_state.stage = 4
    elif s == 4: st.session_state.stage = 5

def on_patient_answer(text: str, stage_id: int, extra_context: str = ""):
    """
    Record patient answer, then fire GPT follow-up if budget remains.
    Messages are tagged with stage_id so they render inline in the panel,
    NOT in the top chat window.
    """
    add_patient(text, stage=stage_id)
    mark_answered(stage_id)
    if can_followup(stage_id):
        record_followup(stage_id)
        with st.spinner("Assistant is thinking…"):
            reply = get_gpt_reply(extra_context=extra_context)
        add_doctor(reply, stage=stage_id)
    # NOTE: we do NOT auto-advance — patient clicks Next when ready

def on_followup_reply(text: str, stage_id: int, extra_context: str = ""):
    """
    Patient replying to a follow-up question inside the panel.
    Same budget logic, but we distinguish from the initial answer.
    """
    add_patient(text, stage=stage_id)
    if can_followup(stage_id):
        record_followup(stage_id)
        with st.spinner("Assistant is thinking…"):
            reply = get_gpt_reply(extra_context=extra_context)
        add_doctor(reply, stage=stage_id)

def handle_voice(audio_value, stage_id: int, extra_context: str = "",
                 is_followup: bool = False) -> bool:
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
        if is_followup:
            on_followup_reply(t, stage_id, extra_context)
        else:
            on_patient_answer(t, stage_id, extra_context)
        return True
    st.warning("Could not transcribe. Please try again.")
    return False

def render_chat_window():
    """Only show messages from COMPLETED stages (stage < current stage)."""
    current = st.session_state.stage
    past_msgs = [m for m in st.session_state.messages
                 if m.get("stage", -99) < current]
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

def render_inline_stage_messages(stage_id: int):
    """
    Render messages from the current stage inline inside the panel.
    Doctor messages = left-aligned card. Patient messages = right-aligned.
    """
    stage_msgs = [m for m in st.session_state.messages if m.get("stage") == stage_id]
    for msg in stage_msgs:
        if msg.get("role") == "doctor":
            st.markdown(f'<div class="inline-followup">🩺 {msg.get("content","")}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="inline-patient">{msg.get("content","")} 🙂</div>',
                        unsafe_allow_html=True)

def render_followup_input(stage_id: int, extra_context: str = ""):
    """
    Compact text + mic row for answering follow-up questions inline in the panel.
    Only shown when the stage has been answered AND GPT still has follow-up budget.
    """
    c_txt, c_send, c_mic = st.columns([6, 1, 1], gap="small")
    with c_txt:
        typed = st.text_input("", placeholder="Reply…",
                              key=f"fu_txt_{stage_id}_{followup_count(stage_id)}",
                              label_visibility="collapsed")
    with c_send:
        send_clicked = st.button("↑", key=f"fu_send_{stage_id}_{followup_count(stage_id)}",
                                 use_container_width=True)
    with c_mic:
        audio_val = None
        if hasattr(st, "audio_input"):
            audio_val = st.audio_input("", label_visibility="collapsed",
                                       key=f"fu_mic_{stage_id}_{st.session_state.mic_key_counter}")
    if send_clicked and typed and typed.strip():
        on_followup_reply(typed.strip(), stage_id, extra_context)
        st.rerun()
    if handle_voice(audio_val, stage_id, extra_context, is_followup=True):
        st.rerun()

def render_next_button(label="Next →"):
    """Next button — always available once a stage is answered."""
    if st.button(label, use_container_width=True,
                 key=f"next_{st.session_state.stage}", type="primary"):
        advance_stage(); st.rerun()

def render_text_mic_row(stage_id: int, extra_context: str = "",
                        placeholder: str = "Or type your answer…"):
    """
    Primary answer row: [text input ────] [↑ Send] [🎤]
    Used as the free-text alternative alongside widget buttons.
    """
    c_txt, c_send, c_mic = st.columns([6, 1, 1], gap="small")
    with c_txt:
        typed = st.text_input("", placeholder=placeholder,
                              key=f"txt_{stage_id}", label_visibility="collapsed")
    with c_send:
        send_clicked = st.button("↑", key=f"txtsend_{stage_id}", use_container_width=True)
    with c_mic:
        audio_val = None
        if hasattr(st, "audio_input"):
            audio_val = st.audio_input("", key=f"mic_{stage_id}_{st.session_state.mic_key_counter}",
                                       label_visibility="collapsed")
    if send_clicked and typed and typed.strip():
        on_patient_answer(typed.strip(), stage_id, extra_context)
        st.rerun()
    if handle_voice(audio_val, stage_id, extra_context):
        st.rerun()

def body_svg(selected: Set[str]) -> str:
    def fill(p): return "#1f7aff" if p in selected else "#cfd8e6"
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

# ── Warnings ─────────────────────────────────────────────────
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
    st.markdown('<div class="panel"><div class="panel-title">Welcome · Please enter your name</div>', unsafe_allow_html=True)
    name_input = st.text_input("Your name:", value=st.session_state.patient_name)
    if st.button("Start Check-In"):
        if name_input.strip():
            st.session_state.patient_name = name_input.strip()
            with st.spinner("Loading your history…"):
                st.session_state.past_checkins = load_past_checkins(name_input.strip())
            past = st.session_state.past_checkins
            if past:
                last = past[-1]
                with st.spinner("Getting your assistant ready…"):
                    opening = get_opening_message(last, name_input.strip())
                add_doctor(opening, stage=0)
                st.session_state.stage = 0
            else:
                context = (
                    f"First check-in for {name_input.strip()}. "
                    f"Greet by name, one sentence intro, say you'll start with a few quick questions."
                )
                with st.spinner("Getting your assistant ready…"):
                    opening = get_gpt_reply(extra_context=context)
                if not opening or opening.startswith("("):
                    opening = f"Hi {name_input.strip()}! I'm your check-in assistant. Let's go through a few quick questions."
                add_doctor(opening, stage=1)
                st.session_state.stage = 1
            st.rerun()
        else:
            st.warning("Please enter your name to continue.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

stage = st.session_state.stage

# ── Chat window: only past-stage messages ───────────────────
render_chat_window()

# ════════════════════════════════════════════════════════════
# STAGE 0 — History recap
# ════════════════════════════════════════════════════════════
if stage == 0:
    history_ctx = (
        "Patient is replying about how they have been since last visit. "
        "Ask ONE specific clinical follow-up question if warranted — reference their actual data. "
        "Do NOT start structured check-in questions yet. No filler."
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">💬 Catching up from your last visit</div>', unsafe_allow_html=True)

    # Always render inline messages (GPT question + any patient replies so far)
    render_inline_stage_messages(stage_id=0)

    # Determine if the last message in this stage is from the doctor (needs a reply)
    stage0_msgs = [m for m in st.session_state.messages if m.get("stage") == 0]
    last_is_doctor = stage0_msgs and stage0_msgs[-1].get("role") == "doctor"

    if last_is_doctor:
        # Always show input when doctor just asked something
        if not is_answered(0):
            render_text_mic_row(stage_id=0, extra_context=history_ctx,
                                placeholder="Type your reply…")
        else:
            render_followup_input(stage_id=0, extra_context=history_ctx)

    if is_answered(0):
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        render_next_button("Start today's check-in →")

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 1 — Feeling scale
# ════════════════════════════════════════════════════════════
elif stage == 1:
    feeling_ctx = (
        "Patient answered how they are feeling today using the PROMIS 5-point scale (excellent/very good/good/fair/poor). "
        "Ask ONE specific follow-up about what is driving that feeling. No filler. Just the question."
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">How are you feeling today?</div>', unsafe_allow_html=True)

    # PROMIS Global Health 5-point scale — standard in oncology patient-reported outcomes
    FEELING_OPTIONS = [
        ("Excellent", "excellent"),
        ("Very Good",  "very good"),
        ("Good",      "good"),
        ("Fair",      "fair"),
        ("Poor",      "poor"),
    ]

    if not is_answered(1):
        # Show opening GPT message (tagged stage=1) inline
        render_inline_stage_messages(stage_id=1)
        st.markdown('<div class="small-note">Choose how you feel, or describe in your own words below</div>',
                    unsafe_allow_html=True)

        # 5 buttons in a single row
        opt_cols = st.columns(5, gap="small")
        for idx, (label, value) in enumerate(FEELING_OPTIONS):
            with opt_cols[idx]:
                btn_label = f"✓ {label}" if st.session_state.feeling_level == value else label
                if st.button(btn_label, key=f"feel_{idx}", use_container_width=True):
                    st.session_state.feeling_level = value
                    st.rerun()

        # Selected indicator + Send
        if st.session_state.feeling_level is not None:
            c_sel, c_send = st.columns([5, 2], gap="small")
            with c_sel:
                st.markdown(f"<div style='padding:5px 2px;font-size:14px;'>"
                            f"Selected: <b>{st.session_state.feeling_level}</b></div>",
                            unsafe_allow_html=True)
            with c_send:
                if st.button("Send ➜", key="send_feeling", use_container_width=True):
                    on_patient_answer(
                        f"I'm feeling {st.session_state.feeling_level} today.",
                        1, feeling_ctx)
                    st.rerun()

        # Free text + mic row
        st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
        render_text_mic_row(stage_id=1, extra_context=feeling_ctx,
                            placeholder="Or describe how you feel in your own words…")

    else:
        render_inline_stage_messages(stage_id=1)
        stage1_msgs = [m for m in st.session_state.messages if m.get("stage") == 1]
        if stage1_msgs and stage1_msgs[-1].get("role") == "doctor":
            render_followup_input(stage_id=1, extra_context=feeling_ctx)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        render_next_button("Next question →")

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 2 — Pain yes/no
# ════════════════════════════════════════════════════════════
elif stage == 2:
    pain_ctx = (
        "Patient answered whether they have pain. "
        "Ask ONE follow-up about the nature or severity of their pain — "
        "e.g. how intense it is, how long they've had it, or what it feels like. "
        "IMPORTANT: Do NOT ask where the pain is located — location is collected separately in the next step. "
        "No filler."
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Do you have any pain today?</div>', unsafe_allow_html=True)

    if not is_answered(2):
        render_inline_stage_messages(stage_id=2)
        st.markdown('<div class="small-note">Choose an option or describe in your own words</div>',
                    unsafe_allow_html=True)

        # [Yes] [No] [text ────] [↑] [🎤] all on one row
        c1, c2, c_txt, c_send, c_mic = st.columns([1.7, 1.7, 4, 1, 1], gap="small")
        with c1:
            if st.button("✅ Yes, pain", use_container_width=True, key="pain_yes"):
                st.session_state.pain_yesno = True
                on_patient_answer("Yes, I have pain today.", 2, pain_ctx); st.rerun()
        with c2:
            if st.button("🙂 No pain", use_container_width=True, key="pain_no"):
                st.session_state.pain_yesno = False
                on_patient_answer("No, I don't have any pain today.", 2, pain_ctx); st.rerun()
        with c_txt:
            typed_pain = st.text_input("", placeholder="Or describe…",
                                       key="txt_2", label_visibility="collapsed")
        with c_send:
            send_pain = st.button("↑", key="txtsend_2", use_container_width=True)
        with c_mic:
            audio_pain = None
            if hasattr(st, "audio_input"):
                audio_pain = st.audio_input("", key=f"mic_2_{st.session_state.mic_key_counter}",
                                            label_visibility="collapsed")
        if send_pain and typed_pain and typed_pain.strip():
            on_patient_answer(typed_pain.strip(), 2, pain_ctx); st.rerun()
        if handle_voice(audio_pain, 2, pain_ctx): st.rerun()

    else:
        render_inline_stage_messages(stage_id=2)
        stage2_msgs = [m for m in st.session_state.messages if m.get("stage") == 2]
        if stage2_msgs and stage2_msgs[-1].get("role") == "doctor":
            render_followup_input(stage_id=2, extra_context=pain_ctx)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        render_next_button("Next question →")

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 3 — Body pain map
# ════════════════════════════════════════════════════════════
elif stage == 3:
    location_ctx = (
        "Patient marked pain locations on a body map. "
        "Ask ONE follow-up about those locations (severity, duration, character). No filler."
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Where do you feel pain?</div>', unsafe_allow_html=True)

    if not is_answered(3):
        render_inline_stage_messages(stage_id=3)
        st.markdown('<div class="small-note">Select areas on the map, or describe below</div>',
                    unsafe_allow_html=True)

        col_svg, col_btns = st.columns([1, 1], gap="medium")
        with col_svg:
            st.markdown(body_svg(st.session_state.selected_parts), unsafe_allow_html=True)
        with col_btns:
            st.markdown("**Toggle regions:**")
            for part in ["Head","Chest","Abdomen","Left Arm","Right Arm","Left Leg","Right Leg"]:
                label = f"✓ {part}" if part in st.session_state.selected_parts else part
                if st.button(label, key=f"toggle_{part}", use_container_width=True):
                    toggle_body_part(part); st.rerun()
            st.markdown(
                '<div class="small-note">Selected: '
                + (", ".join(sorted(st.session_state.selected_parts)) or "None") + "</div>",
                unsafe_allow_html=True)

        c_txt3, c_send3a, c_mic3, c_send3b = st.columns([4, 1, 1, 2], gap="small")
        with c_txt3:
            typed_loc = st.text_input("", placeholder="Or describe where you feel pain…",
                                      key="txt_3", label_visibility="collapsed")
        with c_send3a:
            send_txt3 = st.button("↑", key="txtsend_3", use_container_width=True)
        with c_mic3:
            audio_loc = None
            if hasattr(st, "audio_input"):
                audio_loc = st.audio_input("", key=f"mic_3_{st.session_state.mic_key_counter}",
                                           label_visibility="collapsed")
        with c_send3b:
            send_locs = st.button("Send locations ➜", key="send_locs", use_container_width=True)

        if send_txt3 and typed_loc and typed_loc.strip():
            on_patient_answer(typed_loc.strip(), 3, location_ctx); st.rerun()
        if handle_voice(audio_loc, 3, location_ctx): st.rerun()
        if send_locs:
            loc_txt = ", ".join(sorted(st.session_state.selected_parts)) if st.session_state.selected_parts else "not sure of location"
            on_patient_answer(f"Pain locations: {loc_txt}.", 3, location_ctx); st.rerun()

    else:
        render_inline_stage_messages(stage_id=3)
        stage3_msgs = [m for m in st.session_state.messages if m.get("stage") == 3]
        if stage3_msgs and stage3_msgs[-1].get("role") == "doctor":
            render_followup_input(stage_id=3, extra_context=location_ctx)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        render_next_button("Next question →")

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 4 — Symptom checklist
# ════════════════════════════════════════════════════════════
elif stage == 4:
    symptom_ctx = (
        "Patient submitted their symptom checklist. "
        "Ask ONE follow-up about their most notable symptom. No filler."
    )
    symptom_options = [
        "Fatigue / low energy","Nausea","Vomiting","Poor appetite",
        "Mouth sores","Trouble swallowing","Shortness of breath",
        "Fever / chills","Constipation","Diarrhea","Sleep problems","Anxiety / low mood",
    ]
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Any of these symptoms today?</div>', unsafe_allow_html=True)

    if not is_answered(4):
        render_inline_stage_messages(stage_id=4)
        st.markdown('<div class="small-note">Tap to select all that apply, then click Send — or describe below</div>',
                    unsafe_allow_html=True)

        sc = st.columns(2, gap="small")
        for idx, symptom in enumerate(symptom_options):
            with sc[idx % 2]:
                label = f"✓ {symptom}" if symptom in st.session_state.symptoms else symptom
                if st.button(label, key=f"sym_{idx}", use_container_width=True):
                    if symptom in st.session_state.symptoms: st.session_state.symptoms.remove(symptom)
                    else: st.session_state.symptoms.append(symptom)
                    st.rerun()

        c_txt4, c_send4a, c_mic4, c_send4b = st.columns([4, 1, 1, 2], gap="small")
        with c_txt4:
            typed_sym = st.text_input("", placeholder="Or describe your symptoms…",
                                      key="txt_4", label_visibility="collapsed")
        with c_send4a:
            send_txt4 = st.button("↑", key="txtsend_4", use_container_width=True)
        with c_mic4:
            audio_sym = None
            if hasattr(st, "audio_input"):
                audio_sym = st.audio_input("", key=f"mic_4_{st.session_state.mic_key_counter}",
                                           label_visibility="collapsed")
        with c_send4b:
            send_syms = st.button("Send symptoms ➜", key="send_syms", use_container_width=True)

        if send_txt4 and typed_sym and typed_sym.strip():
            on_patient_answer(typed_sym.strip(), 4, symptom_ctx); st.rerun()
        if handle_voice(audio_sym, 4, symptom_ctx): st.rerun()
        if send_syms:
            sym_txt = "; ".join(st.session_state.symptoms) if st.session_state.symptoms else "no symptoms from checklist"
            on_patient_answer(f"Symptoms today: {sym_txt}.", 4, symptom_ctx); st.rerun()

    else:
        render_inline_stage_messages(stage_id=4)
        stage4_msgs = [m for m in st.session_state.messages if m.get("stage") == 4]
        if stage4_msgs and stage4_msgs[-1].get("role") == "doctor":
            render_followup_input(stage_id=4, extra_context=symptom_ctx)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        render_next_button("Finish check-in →")

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 5 — Free chat + submit → summary
# ════════════════════════════════════════════════════════════
elif stage == 5:
    if st.session_state.submitted:
        name      = st.session_state.get("patient_name","—")
        feeling   = st.session_state.get("feeling_level",None)
        pain      = st.session_state.get("pain_yesno",None)
        locations = sorted(list(st.session_state.get("selected_parts",set())))
        symptoms  = st.session_state.get("symptoms",[])

        widget_msgs = {
            f"My feeling level today is {feeling}/10.",
            "Yes, I have pain today.", "No, I don't have any pain today.",
        }
        if locations: widget_msgs.add(f"Pain locations: {', '.join(locations)}.")
        if symptoms:  widget_msgs.add(f"Symptoms today: {'; '.join(symptoms)}.")

        feeling_display = feeling if feeling is not None else "—"
        pain_str  = "Yes" if pain is True else ("No" if pain is False else "—")
        sym_html  = "".join(f'<span class="tag">{s}</span>' for s in symptoms) or "<span style='opacity:.4'>None</span>"
        loc_html  = "".join(f'<span class="tag">{l}</span>' for l in locations) or "<span style='opacity:.4'>N/A</span>"

        patient_lines = [m["content"] for m in st.session_state.messages
                         if m.get("role")=="patient" and m.get("content","") not in widget_msgs]

        summary_text = "None"
        if patient_lines and _openai_ready():
            try:
                sr = openai_client.chat.completions.create(
                    model=_secret("openai_model", default="gpt-4o-mini"),
                    messages=[
                        {"role":"system","content":(
                            "Clinical notes assistant. Extract ONLY medically relevant facts from patient's "
                            "free-text messages: pain details, severity, duration, triggers, mood, appetite, sleep, energy. "
                            "One bullet per fact. No greetings or filler. If nothing relevant: None"
                        )},
                        {"role":"user","content":"\n".join(f"- {l}" for l in patient_lines)}
                    ], max_tokens=300, temperature=0.2,
                )
                summary_text = (sr.choices[0].message.content or "").strip()
            except: pass

        if summary_text and summary_text != "None":
            items = [l.lstrip("•-– ").strip() for l in summary_text.split("\n")
                     if l.strip() and l.strip()!="None"]
            conv_cell = "<ul style='margin:0;padding-left:18px;'>"+"".join(
                f"<li style='margin-bottom:4px;font-size:14px;color:#1a2540'>{l}</li>" for l in items)+"</ul>"
        else:
            conv_cell = "<span style='opacity:.4'>No additional details shared</span>"

        # Summary styles already defined in main CSS block

        st.markdown(f"""
<div class="summary-wrap">
  <div class="submitted-badge">✅ Submitted</div>
  <div class="summary-title">Check-In Summary — {name}</div>
  <div class="summary-sub">Your care team will review this shortly.</div>
  <table class="summary-table">
    <tr><td>Patient name</td><td>{name}</td></tr>
    <tr><td>Feeling level</td><td>{feeling_display}</td></tr>
    <tr><td>Pain today</td><td>{pain_str}</td></tr>
    <tr><td>Pain locations</td><td>{loc_html}</td></tr>
    <tr><td>Symptoms</td><td>{sym_html}</td></tr>
    <tr><td>Conversation notes</td><td>{conv_cell}</td></tr>
  </table>
</div>""", unsafe_allow_html=True)

    else:
        # Show all conversation history in the chat window for stage 5
        st.markdown('<div class="chat-window">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg.get("role") == "doctor":
                st.markdown(f'<div class="row-left"><div class="avatar">🩺</div>'
                            f'<div class="bubble-doc">{msg.get("content","")}</div></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="row-right"><div class="bubble-pat">{msg.get("content","")}</div>'
                            f'<div class="avatar">🙂</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="panel-title">💬 Anything else to share?</div>'
                    '<div class="small-note">Chat with the assistant, or submit when ready.</div></div>',
                    unsafe_allow_html=True)

        cols = st.columns([5,1], vertical_alignment="bottom")
        with cols[0]:
            user_text_5 = st.chat_input("Anything else to mention…", key="chat_input_5")
        with cols[1]:
            audio_5 = None
            if hasattr(st,"audio_input"):
                audio_5 = st.audio_input("", key=f"mic_5_{st.session_state.mic_key_counter}",
                                         label_visibility="collapsed")
        if user_text_5:
            add_patient(user_text_5, stage=5)
            with st.spinner("Assistant is thinking…"):
                reply = get_gpt_reply()
            add_doctor(reply, stage=5); st.rerun()
        if audio_5 is not None:
            try: ab=audio_5.getvalue(); ah=hashlib.sha1(ab).hexdigest()
            except: ab=ah=None
            if ab and ah and ah != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash=ah; st.session_state.mic_key_counter+=1
                with st.spinner("Transcribing…"): t=transcribe_audio(ab)
                if t and not t.startswith("(Transcription failed"):
                    st.info(f'Heard: "{t}"')
                    add_patient(t, stage=5)
                    with st.spinner("Assistant is thinking…"): reply=get_gpt_reply()
                    add_doctor(reply, stage=5); st.rerun()
        if st.button("✅ Submit Check-In", use_container_width=True):
            try: save_to_sheet(); st.session_state.submitted=True; st.rerun()
            except Exception as e: st.error(f"Failed to save: {e}")
