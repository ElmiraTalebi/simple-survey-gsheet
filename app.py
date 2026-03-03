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
# STREAMLIT PAGE CONFIG (must be near top)
# ============================================================
st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

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
# Support multiple common secret key names (you used different ones across drafts).
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
# GOOGLE SHEETS SETUP (robust + lazy error handling)
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
            # Create if missing (keeps app from crashing)
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
        for row in all_rows[1:]:  # skip header row
            # Expect [timestamp, name, json]
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
# PROMPTING
# ============================================================
def build_system_prompt() -> str:
    """
    Build the GPT system prompt.
    Hybrid approach: widgets collect data, GPT reacts and asks 1 follow-up at a time.
    Past sessions are injected as memory.
    """
    name = st.session_state.get("patient_name", "the patient")

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

You are part of a HYBRID interface. The patient fills in structured widgets (a feeling slider, a
body pain map, a symptom checklist), and YOU handle the conversational layer — reacting naturally
to each widget submission and asking ONE thoughtful follow-up question at a time.

This is a light daily check-in, not a deep formal survey.

=== TODAY'S STRUCTURED DATA (collected so far via widgets) ===
{session_str}

=== PATIENT HISTORY / MEMORY (past sessions) ===
{memory_str}

=== YOUR RULES ===
1. Be warm, natural, and conversational. Short sentences.
2. React to new widget submissions and ask ONE focused follow-up question.
3. Use memory to follow up on recurring issues.
4. If the patient goes off-topic, redirect gently back to symptoms.
5. NEVER give medical advice, diagnoses, medication suggestions, or treatment guidance.
   If asked, say: "I'm not able to give medical advice — I'm here so your care team can follow up with you."
6. Ask ONE question at a time. Do not list multiple questions.
7. When complete, say: "It sounds like we have a good picture of how you're doing today. Feel free to hit Submit when you're ready."
"""

def _openai_ready() -> bool:
    return (openai_client is not None) and (openai_init_error is None)

def get_gpt_reply() -> str:
    """
    Call the GPT API with the conversation history.
    - Limits message history to reduce prompt bloat.
    """
    if not _openai_ready():
        return "(LLM is not configured right now. Please check app secrets for an OpenAI API key.)"

    # Build messages with truncation (keep last N turns)
    openai_messages = [{"role": "system", "content": build_system_prompt()}]

    history = st.session_state.messages[-20:]  # last 20 bubbles is enough for this app
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
.header{ font-size:24px; font-weight:700; margin:8px 0 14px 0; }
.chat-shell{ max-width:840px; margin:0 auto; }
.chat-window{
    max-height:55vh; overflow-y:auto; padding:18px 14px; border-radius:18px;
    background:rgba(255,255,255,0.55); border:1px solid rgba(200,210,230,0.55);
    backdrop-filter:blur(10px);
}
.row-left{ display:flex; justify-content:flex-start; align-items:flex-end; margin:10px 0; gap:10px; }
.row-right{ display:flex; justify-content:flex-end; align-items:flex-end; margin:10px 0; gap:10px; }
.avatar{
    width:36px; height:36px; border-radius:50%;
    display:flex; justify-content:center; align-items:center;
    background:rgba(255,255,255,0.9); border:1px solid rgba(210,220,240,0.9);
    box-shadow:0 2px 8px rgba(0,0,0,0.08); font-size:18px; flex:0 0 auto;
}
.bubble-doc{
    background:#ffffff; border:1px solid rgba(220,225,235,0.95);
    border-radius:18px; padding:12px 14px; max-width:72%;
    box-shadow:0 2px 10px rgba(0,0,0,0.05); white-space:pre-wrap;
}
.bubble-pat{
    background:#1f7aff; color:white; border-radius:18px;
    padding:12px 14px; max-width:72%;
    box-shadow:0 2px 10px rgba(0,0,0,0.08); white-space:pre-wrap;
}
.small-note{ color:rgba(0,0,0,0.55); font-size:12px; margin-top:6px; }
.panel{
    margin-top:14px; padding:14px; border-radius:16px;
    background:rgba(255,255,255,0.65); border:1px solid rgba(200,210,230,0.55);
    backdrop-filter:blur(10px);
}
.panel-title{ font-weight:700; margin-bottom:10px; }
.stButton>button{ border-radius:14px; padding:0.55rem 0.9rem; }
[data-testid="stChatInput"]{
    position:sticky; bottom:0; background:rgba(255,255,255,0.6);
    backdrop-filter:blur(10px); border-top:1px solid rgba(200,210,230,0.55);
    padding-top:10px;
}
.stButton>button {
    font-size: 18px !important;
    padding: 16px 12px !important;
    border-radius: 16px !important;
    font-weight: 600 !important;
}

.scale-button button {
    font-size: 20px !important;
    padding: 18px !important;
    border-radius: 14px !important;
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
    "gpt_followup_done": set(),  # ensures each stage only fires one GPT follow-up
    "last_audio_hash": None,     # prevents double-processing voice recordings
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

def gpt_followup(stage_key: str) -> None:
    """Fire a GPT follow-up after a structured widget is submitted (once per stage)."""
    if stage_key in st.session_state.gpt_followup_done:
        return
    st.session_state.gpt_followup_done.add(stage_key)
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

# ============================================================
# TOP WARNINGS (show helpful errors instead of crashing)
# ============================================================
if openai_init_error:
    st.warning(f"LLM not ready: {openai_init_error}")

_init_sheets()
if sheets_init_error:
    st.warning(f"Google Sheets not ready: {sheets_init_error}")

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="chat-shell"><div class="header">🩺 Cancer Symptom Check-In</div>', unsafe_allow_html=True)

# ============================================================
# STAGE -1 — Name entry
# ============================================================
if st.session_state.stage == -1:
    st.markdown('<div class="panel"><div class="panel-title">Welcome · Please enter your name</div>', unsafe_allow_html=True)
    name_input = st.text_input("Your name:", value=st.session_state.patient_name)

    if st.button("Start Check-In"):
        if name_input.strip():
            st.session_state.patient_name = name_input.strip()

            with st.spinner("Loading your history…"):
                st.session_state.past_checkins = load_past_checkins(name_input.strip())

            with st.spinner("Getting your assistant ready…"):
                opening = get_gpt_reply()

            if not opening or opening.startswith("(LLM is not configured"):
                opening = (
                    f"Hi {st.session_state.patient_name}! I'm your virtual check-in assistant. "
                    "Let's do a quick check-in. How have you been feeling? "
                    "Use the slider below to give me a number from 0 to 10."
                )

            add_doctor(opening)
            st.session_state.stage = 0
            st.rerun()
        else:
            st.warning("Please enter your name to continue.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ============================================================
# CHAT WINDOW — render all messages
# ============================================================
st.markdown('<div class="chat-window">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg.get("role") == "doctor":
        st.markdown(
            f"""
        <div class="row-left">
          <div class="avatar">🩺</div>
          <div class="bubble-doc">{msg.get("content","")}</div>
        </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="row-right">
          <div class="bubble-pat">{msg.get("content","")}</div>
          <div class="avatar">🙂</div>
        </div>""",
            unsafe_allow_html=True,
        )
st.markdown("</div>", unsafe_allow_html=True)

stage = st.session_state.stage

# ============================================================
# INPUT AREA — Text OR Voice (voice optional depending on Streamlit version)
# ============================================================
if not st.session_state.submitted:

    cols = st.columns([5, 1])
    with cols[0]:
        user_text = st.chat_input("Reply to the assistant or type anything…")

    audio_value = None
    with cols[1]:
        # st.audio_input exists in newer Streamlit; fall back gracefully if missing.
        if hasattr(st, "audio_input"):
            audio_value = st.audio_input("🎙️", key="mic_input")
        else:
            st.caption("🎙️ Upgrade Streamlit for voice")

    # Typed text
    if user_text:
        add_patient(user_text)
        with st.spinner("Assistant is thinking…"):
            reply = get_gpt_reply()
        add_doctor(reply)
        st.rerun()

    # Voice recording
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
                add_patient(transcribed)
                with st.spinner("Assistant is thinking…"):
                    reply = get_gpt_reply()
                add_doctor(reply)
                st.rerun()
            else:
                st.warning(f"Could not transcribe audio. {transcribed} Please try again or type your message.")

# ============================================================
# STAGE PANELS — structured widgets sit below the chat
# ============================================================

# Stage 0 — Feeling scale (replaces slider with clickable boxes)
if stage == 0:
    st.markdown('<div class="panel"><div class="panel-title">How are you feeling today?</div>', unsafe_allow_html=True)
    st.markdown("Tap a number from 0 (worst) to 10 (best):")

    scale_cols = st.columns(11)
    for i in range(11):
        with scale_cols[i]:
            if st.button(str(i), key=f"feeling_{i}", use_container_width=True):
                st.session_state.feeling_level = i

    st.markdown(
        f"<div style='font-size:18px;margin-top:8px;'>Selected: <b>{st.session_state.feeling_level}/10</b></div>",
        unsafe_allow_html=True
    )

    if st.button("Send feeling level ➜", use_container_width=True):
        add_patient(f"My feeling level today is {st.session_state.feeling_level}/10.")
        st.session_state.stage = 1
        with st.spinner("Assistant is thinking…"):
            gpt_followup("stage0")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# Stage 1 — Pain yes/no
elif stage == 1:
    st.markdown('<div class="panel"><div class="panel-title">Do you have any pain today?</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Yes, I have pain"):
            st.session_state.pain_yesno = True
            add_patient("Yes, I have pain today.")
            st.session_state.stage = 2
            with st.spinner("Assistant is thinking…"):
                gpt_followup("stage1_yes")
            st.rerun()
    with c2:
        if st.button("🙂 No pain today"):
            st.session_state.pain_yesno = False
            add_patient("No, I don't have any pain today.")
            st.session_state.stage = 3
            with st.spinner("Assistant is thinking…"):
                gpt_followup("stage1_no")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Stage 2 — Body pain map
elif stage == 2:
    st.markdown('<div class="panel"><div class="panel-title">Where do you feel pain?</div>', unsafe_allow_html=True)
    left, right = st.columns([1.2, 1.0])
    with left:
        st.markdown(body_svg(st.session_state.selected_parts), unsafe_allow_html=True)
    with right:
        st.markdown("**Click to toggle regions:**")
        for part in ["Head", "Chest", "Abdomen", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]:
            label = f"✓ {part}" if part in st.session_state.selected_parts else part
            if st.button(label, key=f"toggle_{part}"):
                toggle_body_part(part)
                st.rerun()
        st.markdown(
            '<div class="small-note">Selected: '
            + (", ".join(sorted(st.session_state.selected_parts)) or "None")
            + "</div>",
            unsafe_allow_html=True,
        )

    cA, cB = st.columns(2)
    with cA:
        if st.button("Clear all"):
            st.session_state.selected_parts = set()
            st.rerun()
    with cB:
        if st.button("Send pain locations ➜"):
            if st.session_state.selected_parts:
                add_patient("Pain locations: " + ", ".join(sorted(st.session_state.selected_parts)) + ".")
            else:
                add_patient("I'm not sure of the exact location.")
            st.session_state.stage = 3
            with st.spinner("Assistant is thinking…"):
                gpt_followup("stage2")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    
# Stage 3 — Symptom checklist (large clickable rows)
elif stage == 3:
    st.markdown('<div class="panel"><div class="panel-title">Any of these symptoms today?</div>', unsafe_allow_html=True)

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

    # Ensure list exists
    if "symptoms" not in st.session_state:
        st.session_state.symptoms = []

    for symptom in symptom_options:
        label = f"✓ {symptom}" if symptom in st.session_state.symptoms else symptom
        if st.button(label, key=f"symptom_{symptom}", use_container_width=True):
            if symptom in st.session_state.symptoms:
                st.session_state.symptoms.remove(symptom)
            else:
                st.session_state.symptoms.append(symptom)
            st.rerun()

    if st.button("Send symptoms ➜", use_container_width=True):
        if st.session_state.symptoms:
            add_patient("Symptoms today: " + "; ".join(st.session_state.symptoms) + ".")
        else:
            add_patient("No symptoms from the checklist today.")
        st.session_state.stage = 4
        with st.spinner("Assistant is thinking…"):
            gpt_followup("stage3")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# Stage 4 — Free chat + submit
elif stage == 4:
    if st.session_state.submitted:
        st.success("✅ Your check-in has been submitted. Thank you — your care team will review this shortly.")
    else:
        st.markdown(
            '<div class="panel">'
            '<div class="panel-title">💬 Anything else to share?</div>'
            '<div class="small-note">Chat freely with the assistant, or click Submit when ready.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("✅ Submit Check-In"):
            try:
                save_to_sheet()
                st.session_state.submitted = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save to Google Sheets: {e}")
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
# GOOGLE SHEETS
# ============================================================
sheet = None
sheets_init_error: Optional[str] = None

def _init_sheets():
    global sheet, sheets_init_error
    if sheet is not None or sheets_init_error is not None:
        return
    try:
        gcp_sa    = _require_secret("gcp_service_account")
        gsheet_id = _require_secret("gsheet_id")
        scope     = ["https://www.googleapis.com/auth/spreadsheets"]
        creds     = Credentials.from_service_account_info(gcp_sa, scopes=scope)
        gs_client = gspread.authorize(creds)
        book      = gs_client.open_by_key(gsheet_id)
        try:
            sheet_local = book.worksheet("Form")
        except Exception:
            sheet_local = book.add_worksheet(title="Form", rows=2000, cols=20)
            sheet_local.append_row(["timestamp", "name", "json"])
        sheet = sheet_local
    except Exception as e:
        sheets_init_error = str(e)

def load_past_checkins(name: str) -> List[Dict]:
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
    _init_sheets()
    if sheet is None:
        raise RuntimeError(f"Google Sheets not available: {sheets_init_error}")
    chat_dict = {
        "feeling_level":  st.session_state.feeling_level,
        "pain":           st.session_state.pain_yesno,
        "pain_locations": sorted(list(st.session_state.selected_parts)),
        "symptoms":       st.session_state.symptoms,
        "conversation":   st.session_state.free_chat,   # only save the free-chat turns
    }
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name      = st.session_state.get("patient_name", "Unknown")
    sheet.append_row([timestamp, name, json.dumps(chat_dict)])

# ============================================================
# GPT
# ============================================================
def build_system_prompt() -> str:
    name     = st.session_state.get("patient_name", "the patient")
    feeling  = st.session_state.get("feeling_level", None)
    pain     = st.session_state.get("pain_yesno", None)
    locs     = sorted(list(st.session_state.get("selected_parts", set())))
    symptoms = st.session_state.get("symptoms", [])

    lines = []
    if feeling  is not None: lines.append(f"- Feeling level: {feeling}/10")
    if pain     is not None: lines.append(f"- Pain today: {'yes' if pain else 'no'}")
    if locs:                 lines.append(f"- Pain locations: {', '.join(locs)}")
    if symptoms:             lines.append(f"- Symptoms: {', '.join(symptoms)}")
    session_str = "\n".join(lines) if lines else "Nothing collected yet."

    past = st.session_state.get("past_checkins", [])
    if past:
        mem = []
        for p in past:
            ts   = p.get("timestamp", "?")
            fl   = p.get("feeling_level", "?")
            pn   = "yes" if p.get("pain") else "no"
            ploc = ", ".join(p.get("pain_locations", [])) or "none"
            sym  = ", ".join(p.get("symptoms", [])) or "none"
            mem.append(f"  [{ts}] Feeling:{fl}/10 | Pain:{pn} | Locs:{ploc} | Symptoms:{sym}")
        memory_str = "\n".join(mem)
    else:
        memory_str = "No previous check-ins — may be first session."

    return f"""You are a warm, empathetic virtual symptom-intake assistant for a cancer care clinic.
You are having a brief daily check-in with: {name}.

TODAY'S DATA (from structured widgets):
{session_str}

PATIENT HISTORY (memory from past sessions):
{memory_str}

RULES:
1. Be warm and conversational. Short sentences. Empathetic tone.
2. Ask exactly ONE focused follow-up question based on what was just submitted.
3. Use memory to personalise — mention recurring issues if relevant.
4. Never give medical advice. If asked: "I'm not able to give medical advice — your care team will follow up."
5. One question only. Never list multiple questions.
6. When the check-in feels complete: "It sounds like we have a good picture of how you're doing. Feel free to submit when you're ready."
"""

def _openai_ready() -> bool:
    return openai_client is not None and openai_init_error is None

def get_gpt_reply(extra_history: list = None) -> str:
    if not _openai_ready():
        return "(Assistant not available — check OpenAI API key in secrets.)"
    msgs = [{"role": "system", "content": build_system_prompt()}]
    for m in (extra_history or [])[-20:]:
        role = "assistant" if m.get("role") == "doctor" else "user"
        msgs.append({"role": role, "content": m.get("content", "")})
    try:
        r = openai_client.chat.completions.create(
            model=_secret("openai_model", default="gpt-4o-mini"),
            messages=msgs,
            max_tokens=300,
            temperature=0.6,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(Error: {e})"

def transcribe_audio(audio_bytes: bytes) -> str:
    if not _openai_ready():
        return "(Transcription unavailable.)"
    try:
        import io
        f = io.BytesIO(audio_bytes)
        f.name = "recording.wav"
        r = openai_client.audio.transcriptions.create(
            model=_secret("whisper_model", default="whisper-1"),
            file=f, language="en",
        )
        return (r.text or "").strip()
    except Exception as e:
        return f"(Transcription failed: {e})"

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #eef4ff, #f6fbff);
}

/* ── Page wrapper ── */
.page { max-width: 620px; margin: 0 auto; padding: 0 8px 80px; }

/* ── Header ── */
.app-header {
    font-size: 22px; font-weight: 700;
    margin: 16px 0 24px;
    display: flex; align-items: center; gap: 10px;
}

/* ── Step pill ── */
.step-pill {
    display: inline-block;
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.07em; text-transform: uppercase;
    color: #1f7aff;
    background: rgba(31,122,255,0.08);
    border-radius: 20px; padding: 3px 10px;
    margin-bottom: 10px;
}

/* ── GPT message card (shown at TOP of each step) ── */
@keyframes card-in {
    from { opacity:0; transform: translateY(-8px); }
    to   { opacity:1; transform: translateY(0);    }
}
.gpt-card {
    background: #ffffff;
    border: 1px solid rgba(220,228,245,0.9);
    border-radius: 18px;
    padding: 16px 18px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(31,122,255,0.07);
    display: flex; gap: 12px; align-items: flex-start;
    animation: card-in 0.3s ease both;
}
.gpt-card-avatar {
    font-size: 22px; flex: 0 0 auto; margin-top: 1px;
}
.gpt-card-body { flex: 1; }
.gpt-card-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: rgba(31,122,255,0.55); margin-bottom: 5px;
}
.gpt-card-text {
    font-size: 15px; line-height: 1.6; color: #1a2540;
    white-space: pre-wrap;
}

/* ── Widget card ── */
@keyframes widget-in {
    from { opacity:0; transform: translateY(10px); }
    to   { opacity:1; transform: translateY(0);    }
}
.widget-card {
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(200,215,240,0.6);
    border-radius: 18px;
    padding: 20px 18px;
    backdrop-filter: blur(10px);
    animation: widget-in 0.35s ease both;
}
.widget-title {
    font-size: 17px; font-weight: 700;
    color: #1a2540; margin-bottom: 16px;
}

/* ── GPT follow-up box (shown BELOW widget after answering) ── */
@keyframes followup-in {
    0%   { opacity:0; transform: translateY(14px) scale(0.97); }
    65%  { opacity:1; transform: translateY(-3px) scale(1.01); }
    100% { opacity:1; transform: translateY(0)    scale(1);    }
}
.followup-box {
    margin-top: 18px;
    padding: 16px 18px;
    border-radius: 18px;
    background: linear-gradient(135deg, #f0f7ff, #e8f0fe);
    border: 1.5px solid rgba(31,122,255,0.15);
    box-shadow: 0 4px 20px rgba(31,122,255,0.09);
    display: flex; gap: 12px; align-items: flex-start;
    animation: followup-in 0.45s cubic-bezier(0.34,1.56,0.64,1) both;
}
.followup-avatar { font-size: 22px; flex: 0 0 auto; margin-top: 2px; }
.followup-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: rgba(31,122,255,0.55); margin-bottom: 4px;
}
.followup-text {
    font-size: 15px; line-height: 1.6; color: #1a2540; white-space: pre-wrap;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 14px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 16px !important;
}
.send-btn > button {
    background: #1f7aff !important;
    color: white !important;
    margin-top: 10px;
}

/* ── Pain-location small note ── */
.small-note { color: rgba(0,0,0,0.5); font-size: 12px; margin-top: 6px; }

/* ── Free chat bubbles ── */
.row-left  { display:flex; justify-content:flex-start; align-items:flex-end; margin:10px 0; gap:10px; }
.row-right { display:flex; justify-content:flex-end;   align-items:flex-end; margin:10px 0; gap:10px; }
.avatar {
    width:34px; height:34px; border-radius:50%;
    display:flex; justify-content:center; align-items:center;
    background:rgba(255,255,255,0.9); border:1px solid rgba(210,225,245,0.9);
    box-shadow:0 2px 6px rgba(0,0,0,0.07); font-size:17px; flex:0 0 auto;
}
.bubble-doc {
    background:#fff; border:1px solid rgba(220,230,245,0.9);
    border-radius:18px; padding:11px 14px; max-width:78%;
    box-shadow:0 2px 8px rgba(0,0,0,0.04); white-space:pre-wrap; font-size:15px;
}
.bubble-pat {
    background:#1f7aff; color:#fff; border-radius:18px;
    padding:11px 14px; max-width:78%;
    box-shadow:0 2px 8px rgba(0,0,0,0.08); white-space:pre-wrap; font-size:15px;
}
.chat-area {
    border-radius:18px; padding:14px;
    background:rgba(255,255,255,0.5);
    border:1px solid rgba(200,215,240,0.55);
    margin-bottom:12px;
}

/* ── Submit success ── */
.submitted-box {
    background: linear-gradient(135deg,#edfff4,#d6f5e3);
    border: 1.5px solid rgba(34,197,94,0.25);
    border-radius: 18px; padding: 24px 20px; text-align:center;
    font-size:17px; color:#166534;
    box-shadow: 0 4px 18px rgba(34,197,94,0.08);
    animation: card-in 0.4s ease both;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
defaults: dict = {
    "stage":              -1,
    "patient_name":       "",
    "feeling_level":      None,
    "pain_yesno":         None,
    "selected_parts":     set(),
    "symptoms":           [],
    "submitted":          False,
    "past_checkins":      [],
    # GPT message shown at the TOP of the current step (greeting / transition)
    "current_gpt_msg":    "",
    # GPT follow-up shown BELOW the widget after the patient answers
    "followup_msg":       "",
    # Free-chat history (stage 4 only)
    "free_chat":          [],
    # Whisper dedup
    "last_audio_hash":    None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# RENDER HELPERS
# ============================================================
def render_gpt_card(text: str) -> None:
    """Big white card at the TOP of the step — the assistant speaking first."""
    if not text:
        return
    safe = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    st.markdown(f"""
<div class="gpt-card">
  <div class="gpt-card-avatar">🩺</div>
  <div class="gpt-card-body">
    <div class="gpt-card-label">Assistant</div>
    <div class="gpt-card-text">{safe}</div>
  </div>
</div>""", unsafe_allow_html=True)

def render_followup_box(text: str) -> None:
    """Animated blue-tinted box that pops up BELOW the widget after answering."""
    if not text:
        return
    safe = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    st.markdown(f"""
<div class="followup-box">
  <div class="followup-avatar">🩺</div>
  <div>
    <div class="followup-label">Assistant</div>
    <div class="followup-text">{safe}</div>
  </div>
</div>""", unsafe_allow_html=True)

def body_svg(selected: Set[str]) -> str:
    def fill(p: str) -> str:
        return "#1f7aff" if p in selected else "#cfd8e6"
    s = "#6b7a90"
    return f"""
<svg width="260" height="430" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">
  <defs><filter id="sh"><feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="rgba(0,0,0,0.12)"/></filter></defs>
  <g filter="url(#sh)"><circle cx="160" cy="70" r="38" fill="{fill('Head')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><rect x="110" y="120" width="100" height="70" rx="24" fill="{fill('Chest')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><rect x="115" y="195" width="90" height="70" rx="22" fill="{fill('Abdomen')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M110 132 C80 145,72 180,78 220 C82 250,92 270,100 290 C108 310,115 320,120 320 L120 130Z" fill="{fill('Left Arm')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M210 132 C240 145,248 180,242 220 C238 250,228 270,220 290 C212 310,205 320,200 320 L200 130Z" fill="{fill('Right Arm')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M135 265 C120 310,118 360,126 410 C132 445,132 475,128 500 L155 500 C158 470,160 435,156 405 C150 355,152 312,165 265Z" fill="{fill('Left Leg')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M185 265 C200 310,202 360,194 410 C188 445,188 475,192 500 L165 500 C162 470,160 435,164 405 C170 355,168 312,155 265Z" fill="{fill('Right Leg')}" stroke="{s}" stroke-width="2"/></g>
</svg>""".strip()

# ============================================================
# WARNINGS
# ============================================================
if openai_init_error:
    st.warning(f"LLM not ready: {openai_init_error}")
_init_sheets()
if sheets_init_error:
    st.warning(f"Google Sheets not ready: {sheets_init_error}")

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="page">', unsafe_allow_html=True)
st.markdown('<div class="app-header">🩺 Cancer Symptom Check-In</div>', unsafe_allow_html=True)

# ============================================================
# STAGE -1 — Name entry
# ============================================================
if st.session_state.stage == -1:
    st.markdown('<div class="widget-card">', unsafe_allow_html=True)
    st.markdown('<div class="widget-title">Welcome — please enter your name</div>', unsafe_allow_html=True)
    name_input = st.text_input("Your name:", value=st.session_state.patient_name, label_visibility="collapsed", placeholder="Your name")

    if st.button("Start Check-In →", use_container_width=True):
        if name_input.strip():
            st.session_state.patient_name = name_input.strip()
            with st.spinner("Loading your history…"):
                st.session_state.past_checkins = load_past_checkins(name_input.strip())
            with st.spinner("Getting your assistant ready…"):
                opening = get_gpt_reply()
            if not opening or opening.startswith("("):
                opening = (f"Hi {st.session_state.patient_name}! I'm your virtual check-in assistant. "
                           "Let's do a quick check-in. How are you feeling today — tap a number below.")
            st.session_state.current_gpt_msg = opening
            st.session_state.followup_msg    = ""
            st.session_state.stage           = 0
            st.rerun()
        else:
            st.warning("Please enter your name to continue.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

stage = st.session_state.stage

# ============================================================
# STAGE 0 — Feeling scale
# ============================================================
if stage == 0:
    st.markdown('<div class="step-pill">Step 1 of 4 · How are you feeling?</div>', unsafe_allow_html=True)

    # GPT greeting at top
    render_gpt_card(st.session_state.current_gpt_msg)

    # Only show widget if no followup yet (patient hasn't answered)
    if not st.session_state.followup_msg:
        st.markdown('<div class="widget-card">', unsafe_allow_html=True)
        st.markdown('<div class="widget-title">Tap a number — 0 is worst, 10 is best</div>', unsafe_allow_html=True)

        cols = st.columns(11)
        for i in range(11):
            with cols[i]:
                selected = st.session_state.feeling_level == i
                label = f"**{i}**" if selected else str(i)
                if st.button(label, key=f"feel_{i}", use_container_width=True):
                    st.session_state.feeling_level = i
                    st.rerun()

        if st.session_state.feeling_level is not None:
            st.markdown(f"<div style='margin-top:10px;font-size:17px;'>Selected: <b>{st.session_state.feeling_level} / 10</b></div>", unsafe_allow_html=True)
            st.markdown('<div class="send-btn">', unsafe_allow_html=True)
            if st.button("Send ➜", use_container_width=True, key="send_feeling"):
                with st.spinner("Assistant is thinking…"):
                    # Build context for GPT: patient just answered feeling level
                    ctx = [{"role": "patient", "content": f"My feeling level today is {st.session_state.feeling_level}/10."}]
                    reply = get_gpt_reply(extra_history=ctx)
                st.session_state.followup_msg = reply
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Followup box + Next button appear AFTER answering
    if st.session_state.followup_msg:
        render_followup_box(st.session_state.followup_msg)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("Next →", use_container_width=True, key="next_0"):
            st.session_state.current_gpt_msg = st.session_state.followup_msg
            st.session_state.followup_msg    = ""
            st.session_state.stage           = 1
            st.rerun()

# ============================================================
# STAGE 1 — Pain yes/no
# ============================================================
elif stage == 1:
    st.markdown('<div class="step-pill">Step 2 of 4 · Pain check</div>', unsafe_allow_html=True)
    render_gpt_card(st.session_state.current_gpt_msg)

    if not st.session_state.followup_msg:
        st.markdown('<div class="widget-card">', unsafe_allow_html=True)
        st.markdown('<div class="widget-title">Do you have any pain today?</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅  Yes, I have pain", use_container_width=True):
                st.session_state.pain_yesno = True
                with st.spinner("Assistant is thinking…"):
                    ctx   = [{"role": "patient", "content": "Yes, I have pain today."}]
                    reply = get_gpt_reply(extra_history=ctx)
                st.session_state.followup_msg = reply
                st.rerun()
        with c2:
            if st.button("🙂  No pain today", use_container_width=True):
                st.session_state.pain_yesno = False
                with st.spinner("Assistant is thinking…"):
                    ctx   = [{"role": "patient", "content": "No, I don't have any pain today."}]
                    reply = get_gpt_reply(extra_history=ctx)
                st.session_state.followup_msg = reply
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.followup_msg:
        render_followup_box(st.session_state.followup_msg)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        # Skip body map if no pain
        next_stage = 2 if st.session_state.pain_yesno else 3
        if st.button("Next →", use_container_width=True, key="next_1"):
            st.session_state.current_gpt_msg = st.session_state.followup_msg
            st.session_state.followup_msg    = ""
            st.session_state.stage           = next_stage
            st.rerun()

# ============================================================
# STAGE 2 — Body pain map
# ============================================================
elif stage == 2:
    st.markdown('<div class="step-pill">Step 3a of 4 · Where is the pain?</div>', unsafe_allow_html=True)
    render_gpt_card(st.session_state.current_gpt_msg)

    if not st.session_state.followup_msg:
        st.markdown('<div class="widget-card">', unsafe_allow_html=True)
        st.markdown('<div class="widget-title">Select all areas where you feel pain</div>', unsafe_allow_html=True)

        left, right = st.columns([1, 1])
        with left:
            st.markdown(body_svg(st.session_state.selected_parts), unsafe_allow_html=True)
        with right:
            for part in ["Head", "Chest", "Abdomen", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]:
                label = f"✓  {part}" if part in st.session_state.selected_parts else f"   {part}"
                if st.button(label, key=f"bp_{part}", use_container_width=True):
                    if part in st.session_state.selected_parts:
                        st.session_state.selected_parts.remove(part)
                    else:
                        st.session_state.selected_parts.add(part)
                    st.rerun()
            if st.session_state.selected_parts:
                st.markdown(f'<div class="small-note">Selected: {", ".join(sorted(st.session_state.selected_parts))}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear", use_container_width=True):
                st.session_state.selected_parts = set()
                st.rerun()
        with c2:
            if st.button("Send locations ➜", use_container_width=True):
                loc_text = ", ".join(sorted(st.session_state.selected_parts)) if st.session_state.selected_parts else "not sure of exact location"
                with st.spinner("Assistant is thinking…"):
                    ctx   = [{"role": "patient", "content": f"Pain locations: {loc_text}."}]
                    reply = get_gpt_reply(extra_history=ctx)
                st.session_state.followup_msg = reply
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.followup_msg:
        render_followup_box(st.session_state.followup_msg)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("Next →", use_container_width=True, key="next_2"):
            st.session_state.current_gpt_msg = st.session_state.followup_msg
            st.session_state.followup_msg    = ""
            st.session_state.stage           = 3
            st.rerun()

# ============================================================
# STAGE 3 — Symptom checklist
# ============================================================
elif stage == 3:
    step_label = "Step 3b of 4" if st.session_state.pain_yesno else "Step 3 of 4"
    st.markdown(f'<div class="step-pill">{step_label} · Symptoms</div>', unsafe_allow_html=True)
    render_gpt_card(st.session_state.current_gpt_msg)

    if not st.session_state.followup_msg:
        st.markdown('<div class="widget-card">', unsafe_allow_html=True)
        st.markdown('<div class="widget-title">Any of these symptoms today? (tap to select)</div>', unsafe_allow_html=True)

        symptom_options = [
            "Fatigue / low energy", "Nausea", "Vomiting", "Poor appetite",
            "Mouth sores", "Trouble swallowing", "Shortness of breath",
            "Fever / chills", "Constipation", "Diarrhea",
            "Sleep problems", "Anxiety / low mood",
        ]
        for sym in symptom_options:
            label = f"✓  {sym}" if sym in st.session_state.symptoms else f"   {sym}"
            if st.button(label, key=f"sym_{sym}", use_container_width=True):
                if sym in st.session_state.symptoms:
                    st.session_state.symptoms.remove(sym)
                else:
                    st.session_state.symptoms.append(sym)
                st.rerun()

        if st.button("Send symptoms ➜", use_container_width=True, key="send_sym"):
            sym_text = "; ".join(st.session_state.symptoms) if st.session_state.symptoms else "no symptoms from the checklist"
            with st.spinner("Assistant is thinking…"):
                ctx   = [{"role": "patient", "content": f"Symptoms today: {sym_text}."}]
                reply = get_gpt_reply(extra_history=ctx)
            st.session_state.followup_msg = reply
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.followup_msg:
        render_followup_box(st.session_state.followup_msg)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("Next →", use_container_width=True, key="next_3"):
            st.session_state.current_gpt_msg = st.session_state.followup_msg
            st.session_state.followup_msg    = ""
            st.session_state.stage           = 4
            st.rerun()

# ============================================================
# STAGE 4 — Free chat + submit
# ============================================================
elif stage == 4:
    if st.session_state.submitted:
        st.markdown("""
<div class="submitted-box">
  ✅ <strong>Check-in submitted!</strong><br>
  Thank you — your care team will review this shortly.
</div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="step-pill">Step 4 of 4 · Anything else?</div>', unsafe_allow_html=True)

        # Show the last GPT message as the opening card
        render_gpt_card(st.session_state.current_gpt_msg)

        # Free chat history
        if st.session_state.free_chat:
            st.markdown('<div class="chat-area">', unsafe_allow_html=True)
            for msg in st.session_state.free_chat:
                if msg["role"] == "doctor":
                    st.markdown(f'<div class="row-left"><div class="avatar">🩺</div><div class="bubble-doc">{msg["content"]}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="row-right"><div class="bubble-pat">{msg["content"]}</div><div class="avatar">🙂</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Voice + text input
        cols = st.columns([5, 1])
        with cols[0]:
            user_text = st.chat_input("Type anything, or use the mic →")
        audio_value = None
        with cols[1]:
            if hasattr(st, "audio_input"):
                audio_value = st.audio_input("🎙️", key="mic_input")
            else:
                st.caption("🎙️ upgrade Streamlit for voice")

        # Handle typed text
        if user_text:
            st.session_state.free_chat.append({"role": "patient", "content": user_text})
            with st.spinner("Assistant is thinking…"):
                reply = get_gpt_reply(extra_history=st.session_state.free_chat)
            st.session_state.free_chat.append({"role": "doctor", "content": reply})
            st.rerun()

        # Handle voice
        if audio_value is not None:
            try:
                ab   = audio_value.getvalue()
                ahsh = hashlib.sha1(ab).hexdigest()
            except Exception:
                ab = ahsh = None
            if ab and ahsh and ahsh != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = ahsh
                with st.spinner("Transcribing…"):
                    transcribed = transcribe_audio(ab)
                if transcribed and not transcribed.startswith("("):
                    st.info(f'🎙️ Heard: "{transcribed}"')
                    st.session_state.free_chat.append({"role": "patient", "content": transcribed})
                    with st.spinner("Assistant is thinking…"):
                        reply = get_gpt_reply(extra_history=st.session_state.free_chat)
                    st.session_state.free_chat.append({"role": "doctor", "content": reply})
                    st.rerun()
                else:
                    st.warning(f"Could not transcribe. {transcribed}")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("✅  Submit Check-In", use_container_width=True):
            try:
                save_to_sheet()
                st.session_state.submitted = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")

st.markdown('</div>', unsafe_allow_html=True)  # close .page
