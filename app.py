import hashlib
import json
from datetime import datetime
from typing import Dict, List, Set, Optional

import streamlit as st

# Third-party
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI

# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="wide")

# ============================================================
# UTIL: Secrets helpers (robust across local + Streamlit Cloud)
# ============================================================
def _secret(*keys: str, default=None):
    """Return the first matching secret value among keys."""
    for k in keys:
        if k in st.secrets:
            return st.secrets[k]
    return default

def _require_secret(*keys: str) -> str:
    v = _secret(*keys)
    if v is None:
        raise KeyError(f"Missing required secret. Tried: {', '.join(keys)}")
    return v

# ============================================================
# OPENAI CLIENT
# ============================================================
OPENAI_API_KEY = _secret("openai_api_key", "OPENAI_API_KEY", "openai_key")
openai_client: Optional[OpenAI] = None
openai_init_error: Optional[str] = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        openai_init_error = str(e)
else:
    openai_init_error = "OpenAI API key not found in Streamlit secrets."

# ============================================================
# GOOGLE SHEETS SETUP
# ============================================================
sheet = None
sheets_init_error: Optional[str] = None

def _init_sheets():
    global sheet, sheets_init_error
    if sheet is not None or sheets_init_error is not None:
        return

    try:
        gcp_sa = _require_secret("gcp_service_account")
        gsheet_id = _require_secret("gsheet_id")
        scope = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(gcp_sa, scopes=scope)
        gs_client = gspread.authorize(creds)

        book = gs_client.open_by_key(gsheet_id)
        try:
            sheet_local = book.worksheet("Form")
        except Exception:
            sheet_local = book.add_worksheet(title="Form", rows=2000, cols=20)
            sheet_local.append_row(["timestamp", "name", "json"])  # header

        sheet = sheet_local
    except Exception as e:
        sheets_init_error = str(e)

# ============================================================
# PERSISTENCE: load/save check-ins
# ============================================================
def load_past_checkins(name: str) -> List[Dict]:
    """
    Load previous check-ins for this patient from Google Sheets.
    Returns up to the last 5 sessions — injected into GPT system prompt as memory.
    """
    _init_sheets()
    if sheet is None:
        return []

    try:
        all_rows = sheet.get_all_values()
        past: List[Dict] = []
        for row in all_rows[1:]:
            if len(row) >= 3 and row[1].strip().lower() == name.strip().lower():
                try:
                    data = json.loads(row[2])
                    data["timestamp"] = row[0]
                    past.append(data)
                except Exception:
                    continue
        return past[-5:]
    except Exception:
        return []

def save_to_sheet() -> None:
    """Save patient name, timestamp, and full chat as a dict to Google Sheets."""
    _init_sheets()
    if sheet is None:
        raise RuntimeError(f"Google Sheets not available: {sheets_init_error}")

    chat_dict = {
        "feeling_level": st.session_state.feeling_level,
        "pain": st.session_state.pain_yesno,
        "pain_locations": sorted(list(st.session_state.selected_parts)),
        "symptoms": st.session_state.symptoms,
        "conversation": st.session_state.messages,
    }
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = st.session_state.get("patient_name", "Unknown")
    sheet.append_row([timestamp, name, json.dumps(chat_dict)])

# ============================================================
# STAGE HELPERS
# ============================================================
STAGE_TITLES = {
    -1: "Welcome",
    0: "Overall feeling",
    1: "Pain today",
    2: "Pain location",
    3: "Symptoms",
    4: "Final review",
}

def stage_key(stage: int) -> str:
    return f"stage{stage}"

def stage_focus_text(stage: int) -> str:
    if stage == 0:
        return "Focus on how the patient is feeling overall today."
    if stage == 1:
        return "Focus only on whether the patient has pain today and any brief clarification."
    if stage == 2:
        return "Focus only on where the pain is located and any brief clarification about that pain."
    if stage == 3:
        return "Focus only on the selected symptoms and one gentle follow-up if needed."
    if stage == 4:
        return "Wrap up, allow any final symptom-related detail, and help the patient finish the check-in."
    return "Start the check-in warmly."

def next_stage_after(current_stage: int) -> int:
    if current_stage == 0:
        return 1
    if current_stage == 1:
        return 2 if st.session_state.pain_yesno else 3
    if current_stage == 2:
        return 3
    if current_stage == 3:
        return 4
    return 4

def stage_answered(stage: int) -> bool:
    return stage_key(stage) in st.session_state.answered_stages

# ============================================================
# PROMPTING
# ============================================================
def build_system_prompt() -> str:
    """
    Build the GPT system prompt.
    The left side collects the structured answer for the current stage.
    The right side is a brief follow-up conversation for the SAME stage.
    """
    name = st.session_state.get("patient_name", "the patient")
    current_stage = st.session_state.get("stage", -1)

    feeling = st.session_state.get("feeling_level", None)
    pain = st.session_state.get("pain_yesno", None)
    locations = sorted(list(st.session_state.get("selected_parts", set())))
    symptoms = st.session_state.get("symptoms", [])

    session_lines = []
    if feeling is not None:
        session_lines.append(f"- Feeling level: {feeling}/10")
    if pain is not None:
        session_lines.append(f"- Pain today: {'yes' if pain else 'no'}")
    if locations:
        session_lines.append(f"- Pain locations: {', '.join(locations)}")
    if symptoms:
        session_lines.append(f"- Symptoms from checklist: {', '.join(symptoms)}")
    session_str = "\n".join(session_lines) if session_lines else "Check-in just started — nothing collected yet."

    past = st.session_state.get("past_checkins", [])
    if past:
        mem_lines = []
        for p in past:
            ts = p.get("timestamp", "unknown date")
            fl = p.get("feeling_level", "?")
            pn = "yes" if p.get("pain") else "no"
            locs = ", ".join(p.get("pain_locations", [])) or "none"
            syms = ", ".join(p.get("symptoms", [])) or "none"
            mem_lines.append(f"  [{ts}] Feeling: {fl}/10 | Pain: {pn} | Locations: {locs} | Symptoms: {syms}")
        memory_str = "\n".join(mem_lines)
    else:
        memory_str = "No previous check-ins on record — this may be their first session."

    return f"""You are a warm, empathetic virtual symptom-intake assistant for a cancer care clinic.
Your role is to conduct a brief, natural daily check-in with the patient: {name}.

IMPORTANT UI CONTEXT:
- The page is split into two sides.
- The LEFT side contains the current structured questionnaire stage.
- The RIGHT side is for your brief conversational follow-up.
- Stay focused on the CURRENT stage only unless the patient clearly brings up another urgent symptom.

CURRENT STAGE:
- Stage number: {current_stage}
- Stage name: {STAGE_TITLES.get(current_stage, "Unknown")}
- Stage focus: {stage_focus_text(current_stage)}

This is a light daily check-in, not a deep formal survey.

=== TODAY'S STRUCTURED DATA (collected so far) ===
{session_str}

=== PATIENT HISTORY / MEMORY (past sessions) ===
{memory_str}

=== YOUR RULES ===
1. Be warm, natural, and conversational. Short sentences.
2. After a structured answer is submitted, ask ONE focused follow-up question for the current stage.
3. Keep follow-up brief so the patient can move to the next page.
4. Use memory when helpful, especially for recurring issues.
5. If the patient goes off-topic, redirect gently back to symptoms.
6. NEVER give medical advice, diagnoses, medication suggestions, or treatment guidance.
   If asked, say: "I'm not able to give medical advice — I'm here so your care team can follow up with you."
7. Ask ONE question at a time. Do not list multiple questions.
8. When the patient seems done for this stage, you may say they can continue to the next question.
9. In stage 4, once complete, say: "It sounds like we have a good picture of how you're doing today. Feel free to hit Submit when you're ready."
"""

def _openai_ready() -> bool:
    return (openai_client is not None) and (openai_init_error is None)

def get_gpt_reply() -> str:
    """Call the GPT API with the conversation history."""
    if not _openai_ready():
        return "(LLM is not configured right now. Please check app secrets for an OpenAI API key.)"

    openai_messages = [{"role": "system", "content": build_system_prompt()}]

    history = st.session_state.messages[-20:]
    for msg in history:
        if msg.get("role") == "doctor":
            openai_messages.append({"role": "assistant", "content": msg.get("content", "")})
        elif msg.get("role") == "patient":
            openai_messages.append({"role": "user", "content": msg.get("content", "")})

    try:
        response = openai_client.chat.completions.create(
            model=_secret("openai_model", default="gpt-4o-mini"),
            messages=openai_messages,
            max_tokens=350,
            temperature=0.6,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(Sorry, I couldn't connect right now. Please try again. Error: {e})"

def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio using Whisper API, if available."""
    if not _openai_ready():
        return "(Transcription unavailable: OpenAI client not configured.)"
    try:
        import io
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.wav"
        result = openai_client.audio.transcriptions.create(
            model=_secret("whisper_model", default="whisper-1"),
            file=audio_file,
            language="en",
        )
        return (result.text or "").strip()
    except Exception as e:
        return f"(Transcription failed: {e})"

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#eef4ff,#f6fbff);
}
.block-container{
    padding-top: 1.2rem;
    padding-bottom: 1rem;
    max-width: 1500px;
}
.header{
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 12px 0;
}
.subheader{
    color: rgba(0,0,0,0.6);
    margin-bottom: 18px;
    font-size: 15px;
}
.top-shell{
    max-width: 1400px;
    margin: 0 auto;
}
.stage-badge{
    display: inline-block;
    background: #eaf2ff;
    color: #1f4fa3;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 10px;
}
.panel{
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(200,210,230,0.65);
    border-radius: 20px;
    padding: 20px;
    min-height: 72vh;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    backdrop-filter: blur(8px);
}
.panel-title{
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 8px;
}
.panel-note{
    font-size: 14px;
    color: rgba(0,0,0,0.6);
    margin-bottom: 16px;
}
.chat-window{
    height: 52vh;
    overflow-y: auto;
    padding: 10px 6px 10px 2px;
    margin-bottom: 12px;
}
.row-left{
    display:flex;
    justify-content:flex-start;
    align-items:flex-end;
    margin:10px 0;
    gap:10px;
}
.row-right{
    display:flex;
    justify-content:flex-end;
    align-items:flex-end;
    margin:10px 0;
    gap:10px;
}
.avatar{
    width:36px;
    height:36px;
    border-radius:50%;
    display:flex;
    justify-content:center;
    align-items:center;
    background:rgba(255,255,255,0.96);
    border:1px solid rgba(210,220,240,0.9);
    box-shadow:0 2px 8px rgba(0,0,0,0.08);
    font-size:18px;
    flex:0 0 auto;
}
.bubble-doc{
    background:#ffffff;
    border:1px solid rgba(220,225,235,0.95);
    border-radius:18px;
    padding:12px 14px;
    max-width:76%;
    box-shadow:0 2px 10px rgba(0,0,0,0.05);
    white-space:pre-wrap;
}
.bubble-pat{
    background:#1f7aff;
    color:white;
    border-radius:18px;
    padding:12px 14px;
    max-width:76%;
    box-shadow:0 2px 10px rgba(0,0,0,0.08);
    white-space:pre-wrap;
}
.small-note{
    color: rgba(0,0,0,0.58);
    font-size: 13px;
    margin-top: 8px;
}
.answer-box{
    background: #f7faff;
    border: 1px solid #dfeaff;
    border-radius: 16px;
    padding: 14px;
    margin: 14px 0;
}
.progress-wrap{
    display:flex;
    gap:8px;
    flex-wrap:wrap;
    margin-bottom: 14px;
}
.progress-pill{
    padding:7px 12px;
    border-radius:999px;
    background:#eef3fb;
    color:#58708f;
    font-size:13px;
    font-weight:600;
}
.progress-pill.active{
    background:#1f7aff;
    color:#fff;
}
.progress-pill.done{
    background:#dff3e8;
    color:#16663d;
}
.stButton > button{
    font-size: 18px !important;
    padding: 16px 12px !important;
    border-radius: 16px !important;
    font-weight: 600 !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button{
    min-height: 58px !important;
}
[data-testid="stTextInput"] input{
    font-size: 17px !important;
    padding: 0.8rem 0.9rem !important;
}
[data-testid="stAudioInput"]{
    margin-top: 6px;
}
hr.soft{
    border: none;
    border-top: 1px solid rgba(210,220,235,0.8);
    margin: 14px 0 16px 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INIT
# ============================================================
defaults = {
    "messages": [],
    "stage": -1,
    "patient_name": "",
    "selected_parts": set(),
    "pain_yesno": None,
    "feeling_level": None,
    "symptoms": [],
    "submitted": False,
    "past_checkins": [],
    "gpt_followup_done": set(),
    "answered_stages": set(),
    "last_audio_hash": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# HELPERS
# ============================================================
def add_doctor(text: str) -> None:
    st.session_state.messages.append({"role": "doctor", "content": text})

def add_patient(text: str) -> None:
    st.session_state.messages.append({"role": "patient", "content": text})

def gpt_followup(stage_name: str) -> None:
    """Fire a GPT follow-up after a structured widget is submitted (once per stage)."""
    if stage_name in st.session_state.gpt_followup_done:
        return
    st.session_state.gpt_followup_done.add(stage_name)
    reply = get_gpt_reply()
    add_doctor(reply)

def toggle_body_part(part: str) -> None:
    if part in st.session_state.selected_parts:
        st.session_state.selected_parts.remove(part)
    else:
        st.session_state.selected_parts.add(part)

def body_svg(selected: Set[str]) -> str:
    def fill(part: str) -> str:
        return "#1f7aff" if part in selected else "#cfd8e6"
    stroke = "#6b7a90"
    return f"""
<svg width="320" height="520" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="rgba(0,0,0,0.15)"/>
    </filter>
  </defs>
  <g filter="url(#shadow)">
    <circle cx="160" cy="70" r="38" fill="{fill('Head')}" stroke="{stroke}" stroke-width="2"/>
  </g>
  <g filter="url(#shadow)">
    <rect x="110" y="120" width="100" height="70" rx="24" fill="{fill('Chest')}" stroke="{stroke}" stroke-width="2"/>
  </g>
  <g filter="url(#shadow)">
    <rect x="115" y="195" width="90" height="70" rx="22" fill="{fill('Abdomen')}" stroke="{stroke}" stroke-width="2"/>
  </g>
  <g filter="url(#shadow)">
    <path d="M110 132 C 80 145, 72 180, 78 220 C 82 250, 92 270, 100 290 C 108 310, 115 320, 120 320 L 120 130 Z"
          fill="{fill('Left Arm')}" stroke="{stroke}" stroke-width="2"/>
  </g>
  <g filter="url(#shadow)">
    <path d="M210 132 C 240 145, 248 180, 242 220 C 238 250, 228 270, 220 290 C 212 310, 205 320, 200 320 L 200 130 Z"
          fill="{fill('Right Arm')}" stroke="{stroke}" stroke-width="2"/>
  </g>
  <g filter="url(#shadow)">
    <path d="M135 265 C 120 310, 118 360, 126 410 C 132 445, 132 475, 128 500 L 155 500 C 158 470, 160 435, 156 405 C 150 355, 152 312, 165 265 Z"
          fill="{fill('Left Leg')}" stroke="{stroke}" stroke-width="2"/>
  </g>
  <g filter="url(#shadow)">
    <path d="M185 265 C 200 310, 202 360, 194 410 C 188 445, 188 475, 192 500 L 165 500 C 162 470, 160 435, 164 405 C 170 355, 168 312, 155 265 Z"
          fill="{fill('Right Leg')}" stroke="{stroke}" stroke-width="2"/>
  </g>
  <text x="160" y="518" text-anchor="middle" font-size="12" fill="rgba(0,0,0,0.45)">
    Click buttons to toggle regions
  </text>
</svg>""".strip()

def render_progress(current_stage: int) -> None:
    labels = {
        0: "Feeling",
        1: "Pain",
        2: "Location",
        3: "Symptoms",
        4: "Finish",
    }
    html = ['<div class="progress-wrap">']
    for idx in [0, 1, 2, 3, 4]:
        classes = ["progress-pill"]
        if idx == current_stage:
            classes.append("active")
        elif stage_key(idx) in st.session_state.answered_stages or idx < current_stage:
            classes.append("done")
        html.append(f'<div class="{" ".join(classes)}">{idx + 1}. {labels[idx]}</div>')
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)

def send_followup_message_from_text(text: str) -> None:
    text = text.strip()
    if not text:
        return
    add_patient(text)
    with st.spinner("Assistant is thinking…"):
        reply = get_gpt_reply()
    add_doctor(reply)

def handle_audio_followup(stage: int) -> None:
    audio_value = None
    if hasattr(st, "audio_input"):
        audio_value = st.audio_input("🎙️ Speak instead of typing", key=f"mic_input_stage_{stage}")
    else:
        st.caption("🎙️ Upgrade Streamlit for voice")

    if audio_value is not None:
        try:
            audio_bytes = audio_value.getvalue()
            audio_hash = hashlib.sha1(audio_bytes).hexdigest()
        except Exception:
            audio_bytes = None
            audio_hash = None

        if audio_bytes and audio_hash and audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash
            with st.spinner("Transcribing your voice…"):
                transcribed = transcribe_audio(audio_bytes)

            if transcribed and not transcribed.startswith("(Transcription failed"):
                st.info(f'🎙️ Heard: "{transcribed}"')
                send_followup_message_from_text(transcribed)
                st.rerun()
            else:
                st.warning(f"Could not transcribe audio. {transcribed} Please try again or type your message.")

def render_right_panel(stage: int) -> None:
    st.markdown('<div class="panel-title">Assistant follow-up</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.info("The assistant will appear here.")
    else:
        st.markdown('<div class="chat-window">', unsafe_allow_html=True)
        for msg in st.session_state.messages[-12:]:
            if msg.get("role") == "doctor":
                st.markdown(
                    f"""
                    <div class="row-left">
                      <div class="avatar">🩺</div>
                      <div class="bubble-doc">{msg.get("content","")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="row-right">
                      <div class="bubble-pat">{msg.get("content","")}</div>
                      <div class="avatar">🙂</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    if stage == 4 or stage_answered(stage):
        st.markdown('<hr class="soft" />', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-note">You can reply here if you want to answer the assistant before moving on.</div>',
            unsafe_allow_html=True,
        )
        input_key = f"followup_text_stage_{stage}"
        user_text = st.text_input("Type your reply", key=input_key, placeholder="Type a short reply for the assistant…")
        send_col, spacer_col = st.columns([1, 2])
        with send_col:
            if st.button("Send reply", key=f"send_reply_{stage}", use_container_width=True):
                if user_text.strip():
                    send_followup_message_from_text(user_text)
                    st.session_state[input_key] = ""
                    st.rerun()
                else:
                    st.warning("Please type a message first.")
        handle_audio_followup(stage)
    else:
        st.markdown(
            '<div class="panel-note">Answer the question on the left first. Then the assistant will ask a short follow-up here.</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# TOP WARNINGS
# ============================================================
if openai_init_error:
    st.warning(f"LLM not ready: {openai_init_error}")

_init_sheets()
if sheets_init_error:
    st.warning(f"Google Sheets not ready: {sheets_init_error}")

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="top-shell">', unsafe_allow_html=True)
st.markdown('<div class="header">🩺 Cancer Symptom Check-In</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subheader">One question per page. Answer on the left, then review the assistant follow-up on the right.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# STAGE -1 — Name entry
# ============================================================
if st.session_state.stage == -1:
    st.markdown('<div class="panel" style="min-height:auto;"><div class="panel-title">Welcome</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-note">Please enter your name to start today’s check-in.</div>', unsafe_allow_html=True)
    name_input = st.text_input("Your name:", value=st.session_state.patient_name)

    if st.button("Start Check-In", use_container_width=False):
        if name_input.strip():
            st.session_state.patient_name = name_input.strip()

            with st.spinner("Loading your history…"):
                st.session_state.past_checkins = load_past_checkins(name_input.strip())

            with st.spinner("Getting your assistant ready…"):
                opening = get_gpt_reply()

            if not opening or opening.startswith("(LLM is not configured"):
                opening = (
                    f"Hi {st.session_state.patient_name}! I'm your virtual check-in assistant. "
                    "Let's do a quick check-in. How have you been feeling today?"
                )

            add_doctor(opening)
            st.session_state.stage = 0
            st.rerun()
        else:
            st.warning("Please enter your name to continue.")

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()

# ============================================================
# MAIN TWO-PANEL LAYOUT
# ============================================================
render_progress(st.session_state.stage)
left_col, right_col = st.columns([1.05, 1.0], gap="large")
stage = st.session_state.stage

with left_col:
    st.markdown(f'<div class="panel"><div class="stage-badge">Step {stage + 1} of 5</div>', unsafe_allow_html=True)

    # Stage 0 — Feeling scale
    if stage == 0:
        st.markdown('<div class="panel-title">How are you feeling today?</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-note">Tap a number from 0 (worst) to 10 (best).</div>', unsafe_allow_html=True)

        scale_cols = st.columns(11)
        for i in range(11):
            with scale_cols[i]:
                if st.button(str(i), key=f"feeling_{i}", use_container_width=True):
                    st.session_state.feeling_level = i

        if st.session_state.feeling_level is None:
            st.markdown('<div class="small-note">No selection yet</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="answer-box"><b>Selected feeling level:</b> {st.session_state.feeling_level}/10</div>',
                unsafe_allow_html=True,
            )

        if not stage_answered(0):
            if st.button("Save answer and ask assistant ➜", key="submit_stage0", use_container_width=True):
                if st.session_state.feeling_level is None:
                    st.warning("Please choose a number before continuing.")
                else:
                    add_patient(f"My feeling level today is {st.session_state.feeling_level}/10.")
                    st.session_state.answered_stages.add(stage_key(0))
                    with st.spinner("Assistant is thinking…"):
                        gpt_followup(stage_key(0))
                    st.rerun()
        else:
            st.success("Saved. The assistant is ready on the right.")
            if st.button("Continue to next question ➜", key="next_from_0", use_container_width=True):
                st.session_state.stage = next_stage_after(0)
                st.rerun()

    # Stage 1 — Pain yes/no
    elif stage == 1:
        st.markdown('<div class="panel-title">Do you have any pain today?</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-note">Choose the option that fits best.</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Yes, I have pain", key="pain_yes", use_container_width=True):
                st.session_state.pain_yesno = True
        with c2:
            if st.button("🙂 No pain today", key="pain_no", use_container_width=True):
                st.session_state.pain_yesno = False

        if st.session_state.pain_yesno is None:
            st.markdown('<div class="small-note">No selection yet</div>', unsafe_allow_html=True)
        else:
            selected_text = "Yes, I have pain today." if st.session_state.pain_yesno else "No, I do not have pain today."
            st.markdown(
                f'<div class="answer-box"><b>Selected answer:</b> {selected_text}</div>',
                unsafe_allow_html=True,
            )

        if not stage_answered(1):
            if st.button("Save answer and ask assistant ➜", key="submit_stage1", use_container_width=True):
                if st.session_state.pain_yesno is None:
                    st.warning("Please choose Yes or No before continuing.")
                else:
                    if st.session_state.pain_yesno:
                        add_patient("Yes, I have pain today.")
                    else:
                        add_patient("No, I don't have any pain today.")
                    st.session_state.answered_stages.add(stage_key(1))
                    with st.spinner("Assistant is thinking…"):
                        gpt_followup(stage_key(1))
                    st.rerun()
        else:
            st.success("Saved. The assistant is ready on the right.")
            button_label = "Continue to pain location ➜" if st.session_state.pain_yesno else "Continue to symptoms ➜"
            if st.button(button_label, key="next_from_1", use_container_width=True):
                st.session_state.stage = next_stage_after(1)
                st.rerun()

    # Stage 2 — Body pain map
    elif stage == 2:
        st.markdown('<div class="panel-title">Where do you feel pain?</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-note">Click the body regions that apply.</div>', unsafe_allow_html=True)

        map_left, map_right = st.columns([1.15, 1.0])
        with map_left:
            st.markdown(body_svg(st.session_state.selected_parts), unsafe_allow_html=True)
        with map_right:
            st.markdown("**Toggle regions:**")
            for part in ["Head", "Chest", "Abdomen", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]:
                label = f"✓ {part}" if part in st.session_state.selected_parts else part
                if st.button(label, key=f"toggle_{part}", use_container_width=True):
                    toggle_body_part(part)
                    st.rerun()
            if st.button("Clear all", key="clear_parts", use_container_width=True):
                st.session_state.selected_parts = set()
                st.rerun()

        st.markdown(
            '<div class="answer-box"><b>Selected regions:</b> '
            + (", ".join(sorted(st.session_state.selected_parts)) or "None")
            + "</div>",
            unsafe_allow_html=True,
        )

        if not stage_answered(2):
            if st.button("Save answer and ask assistant ➜", key="submit_stage2", use_container_width=True):
                if st.session_state.selected_parts:
                    add_patient("Pain locations: " + ", ".join(sorted(st.session_state.selected_parts)) + ".")
                else:
                    add_patient("I'm not sure of the exact pain location.")
                st.session_state.answered_stages.add(stage_key(2))
                with st.spinner("Assistant is thinking…"):
                    gpt_followup(stage_key(2))
                st.rerun()
        else:
            st.success("Saved. The assistant is ready on the right.")
            if st.button("Continue to next question ➜", key="next_from_2", use_container_width=True):
                st.session_state.stage = next_stage_after(2)
                st.rerun()

    # Stage 3 — Symptoms
    elif stage == 3:
        st.markdown('<div class="panel-title">Any of these symptoms today?</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-note">Tap any symptoms that apply. Tap again to remove.</div>', unsafe_allow_html=True)

        symptom_options = [
            "Fatigue / low energy",
            "Nausea",
            "Vomiting",
            "Poor appetite",
            "Mouth sores",
            "Trouble swallowing",
            "Shortness of breath",
            "Fever / chills",
            "Constipation",
            "Diarrhea",
            "Sleep problems",
            "Anxiety / low mood",
        ]

        for symptom in symptom_options:
            label = f"✓ {symptom}" if symptom in st.session_state.symptoms else symptom
            if st.button(label, key=f"symptom_{symptom}", use_container_width=True):
                if symptom in st.session_state.symptoms:
                    st.session_state.symptoms.remove(symptom)
                else:
                    st.session_state.symptoms.append(symptom)
                st.rerun()

        st.markdown(
            '<div class="answer-box"><b>Selected symptoms:</b> '
            + ("; ".join(st.session_state.symptoms) if st.session_state.symptoms else "None")
            + "</div>",
            unsafe_allow_html=True,
        )

        if not stage_answered(3):
            if st.button("Save answer and ask assistant ➜", key="submit_stage3", use_container_width=True):
                if st.session_state.symptoms:
                    add_patient("Symptoms today: " + "; ".join(st.session_state.symptoms) + ".")
                else:
                    add_patient("No symptoms from the checklist today.")
                st.session_state.answered_stages.add(stage_key(3))
                with st.spinner("Assistant is thinking…"):
                    gpt_followup(stage_key(3))
                st.rerun()
        else:
            st.success("Saved. The assistant is ready on the right.")
            if st.button("Continue to final review ➜", key="next_from_3", use_container_width=True):
                st.session_state.stage = next_stage_after(3)
                st.rerun()

    # Stage 4 — Final review + submit
    elif stage == 4:
        st.markdown('<div class="panel-title">Final review</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-note">Use the assistant on the right for any last detail, then submit when ready.</div>', unsafe_allow_html=True)

        summary_html = f"""
        <div class="answer-box">
            <div><b>Feeling level:</b> {st.session_state.feeling_level if st.session_state.feeling_level is not None else "Not answered"}</div>
            <div><b>Pain today:</b> {("Yes" if st.session_state.pain_yesno else "No") if st.session_state.pain_yesno is not None else "Not answered"}</div>
            <div><b>Pain locations:</b> {", ".join(sorted(st.session_state.selected_parts)) or "None"}</div>
            <div><b>Symptoms:</b> {"; ".join(st.session_state.symptoms) if st.session_state.symptoms else "None"}</div>
        </div>
        """
        st.markdown(summary_html, unsafe_allow_html=True)

        if not stage_answered(4):
            if st.button("Ask assistant for final follow-up ➜", key="submit_stage4", use_container_width=True):
                st.session_state.answered_stages.add(stage_key(4))
                with st.spinner("Assistant is thinking…"):
                    gpt_followup(stage_key(4))
                st.rerun()
        else:
            st.success("Final review is ready. You can still reply on the right if needed.")

        if st.session_state.submitted:
            st.success("✅ Your check-in has been submitted. Thank you — your care team will review this shortly.")
        else:
            if st.button("✅ Submit Check-In", key="submit_checkin", use_container_width=True):
                try:
                    save_to_sheet()
                    st.session_state.submitted = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save to Google Sheets: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    render_right_panel(stage)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
