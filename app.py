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
# UTIL: Secrets helpers
# ============================================================
def _secret(*keys: str, default=None):
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

# ============================================================
# PERSISTENCE: load/save check-ins
# ============================================================
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
        "conversation":   st.session_state.gpt_history,
    }
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name      = st.session_state.get("patient_name", "Unknown")
    sheet.append_row([timestamp, name, json.dumps(chat_dict)])

# ============================================================
# PROMPTING
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
4. STRICT TOPIC ENFORCEMENT: You ONLY discuss how the patient is feeling physically and emotionally
   as it relates to their health and symptoms. If the patient says ANYTHING off-topic — including
   general chat, jokes, questions about you, news, weather, other people, or any unrelated subject —
   do NOT engage with it at all. Immediately redirect with a warm but firm response, for example:
   "I'm only here to help with your symptom check-in today. Let's stay focused — [follow-up question]."
   Never answer off-topic questions even briefly before redirecting.
5. Never give medical advice. If asked: "I'm not able to give medical advice — your care team will follow up."
6. One question only. Never list multiple questions.
7. When the check-in feels complete: "It sounds like we have a good picture of how you're doing. Feel free to submit when you're ready."
"""

def _openai_ready() -> bool:
    return openai_client is not None and openai_init_error is None

def get_gpt_reply() -> str:
    if not _openai_ready():
        return "(Assistant not available — check OpenAI API key in secrets.)"
    msgs = [{"role": "system", "content": build_system_prompt()}]
    for m in st.session_state.gpt_history[-20:]:
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
.header { font-size: 24px; font-weight: 700; margin: 8px 0 14px 0; }
.chat-shell { max-width: 840px; margin: 0 auto; }

/* Widget panel */
.panel {
    margin-top: 14px; padding: 14px; border-radius: 16px;
    background: rgba(255,255,255,0.65); border: 1px solid rgba(200,210,230,0.55);
    backdrop-filter: blur(10px);
}
.panel-title { font-weight: 700; margin-bottom: 10px; }

/* GPT reply box — animated, appears BELOW the widget */
@keyframes gpt-pop {
    0%   { opacity: 0; transform: translateY(14px) scale(0.97); }
    65%  { opacity: 1; transform: translateY(-3px) scale(1.01); }
    100% { opacity: 1; transform: translateY(0)    scale(1);    }
}
.gpt-reply-box {
    margin-top: 16px;
    padding: 16px 18px;
    border-radius: 18px;
    background: linear-gradient(135deg, #f0f7ff, #e8f0fe);
    border: 1.5px solid rgba(31, 122, 255, 0.18);
    box-shadow: 0 4px 20px rgba(31, 122, 255, 0.10);
    display: flex;
    gap: 12px;
    align-items: flex-start;
    animation: gpt-pop 0.45s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}
.gpt-reply-avatar { font-size: 22px; flex: 0 0 auto; margin-top: 2px; }
.gpt-reply-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: rgba(31,122,255,0.55); margin-bottom: 4px;
}
.gpt-reply-text {
    font-size: 15px; line-height: 1.6; color: #1a2540; white-space: pre-wrap;
}

/* Free chat bubbles (stage 4) */
.chat-window {
    max-height: 55vh; overflow-y: auto; padding: 18px 14px; border-radius: 18px;
    background: rgba(255,255,255,0.55); border: 1px solid rgba(200,210,230,0.55);
    backdrop-filter: blur(10px);
}
.row-left  { display:flex; justify-content:flex-start; align-items:flex-end; margin:10px 0; gap:10px; }
.row-right { display:flex; justify-content:flex-end;   align-items:flex-end; margin:10px 0; gap:10px; }
.avatar {
    width:36px; height:36px; border-radius:50%;
    display:flex; justify-content:center; align-items:center;
    background:rgba(255,255,255,0.9); border:1px solid rgba(210,220,240,0.9);
    box-shadow:0 2px 8px rgba(0,0,0,0.08); font-size:18px; flex:0 0 auto;
}
.bubble-doc {
    background:#ffffff; border:1px solid rgba(220,225,235,0.95);
    border-radius:18px; padding:12px 14px; max-width:72%;
    box-shadow:0 2px 10px rgba(0,0,0,0.05); white-space:pre-wrap;
}
.bubble-pat {
    background:#1f7aff; color:white; border-radius:18px;
    padding:12px 14px; max-width:72%;
    box-shadow:0 2px 10px rgba(0,0,0,0.08); white-space:pre-wrap;
}

.small-note { color: rgba(0,0,0,0.55); font-size: 12px; margin-top: 6px; }
.stButton > button { border-radius: 14px; padding: 0.55rem 0.9rem; }
[data-testid="stChatInput"] {
    position: sticky; bottom: 0; background: rgba(255,255,255,0.6);
    backdrop-filter: blur(10px); border-top: 1px solid rgba(200,210,230,0.55);
    padding-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INIT
# ============================================================
defaults = {
    "stage":           -1,
    "patient_name":    "",
    "feeling_level":   None,
    "pain_yesno":      None,
    "selected_parts":  set(),
    "symptoms":        [],
    "submitted":       False,
    "past_checkins":   [],
    "last_audio_hash": None,
    # gpt_history: full conversation used as GPT context (never displayed directly)
    "gpt_history":     [],
    # Per-stage GPT replies — stored here so they show BELOW the widget, not in a chat log
    "reply_stage0":    "",   # after feeling level submitted
    "reply_stage1":    "",   # after pain yes/no submitted
    "reply_stage2":    "",   # after body map submitted
    "reply_stage3":    "",   # after symptom checklist submitted
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# HELPERS
# ============================================================
def gpt_reply_box(text: str) -> None:
    """Render an animated GPT reply box below the widget."""
    if not text:
        return
    safe = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    st.markdown(f"""
<div class="gpt-reply-box">
  <div class="gpt-reply-avatar">🩺</div>
  <div>
    <div class="gpt-reply-label">Assistant</div>
    <div class="gpt-reply-text">{safe}</div>
  </div>
</div>""", unsafe_allow_html=True)

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
            st.session_state.stage = 0
            st.rerun()
        else:
            st.warning("Please enter your name to continue.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

stage = st.session_state.stage

# ============================================================
# STAGE 0 — Feeling scale
# ============================================================
if stage == 0:
    st.markdown('<div class="panel"><div class="panel-title">How are you feeling today?</div>', unsafe_allow_html=True)
    st.markdown("Tap a number from 0 (worst) to 10 (best):")

    scale_cols = st.columns(11)
    for i in range(11):
        with scale_cols[i]:
            label = f"✓ {i}" if st.session_state.feeling_level == i else str(i)
            if st.button(label, key=f"feeling_{i}", use_container_width=True):
                st.session_state.feeling_level = i
                st.rerun()

    if st.session_state.feeling_level is not None:
        st.markdown(
            f"<div style='font-size:16px; margin-top:10px;'>Selected: <b>{st.session_state.feeling_level} / 10</b></div>",
            unsafe_allow_html=True
        )

    # Only show Send button if no GPT reply yet
    if not st.session_state.reply_stage0:
        if st.button("Send ➜", key="send_feeling", use_container_width=True):
            if st.session_state.feeling_level is not None:
                patient_msg = f"My feeling level today is {st.session_state.feeling_level}/10."
                st.session_state.gpt_history.append({"role": "patient", "content": patient_msg})
                with st.spinner("Assistant is thinking…"):
                    reply = get_gpt_reply()
                st.session_state.gpt_history.append({"role": "doctor", "content": reply})
                st.session_state.reply_stage0 = reply
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # GPT reply appears BELOW the panel after submitting
    if st.session_state.reply_stage0:
        gpt_reply_box(st.session_state.reply_stage0)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Next →", key="next_0", use_container_width=True):
            st.session_state.stage = 1
            st.rerun()

# ============================================================
# STAGE 1 — Pain yes/no
# ============================================================
elif stage == 1:
    st.markdown('<div class="panel"><div class="panel-title">Do you have any pain today?</div>', unsafe_allow_html=True)

    if not st.session_state.reply_stage1:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Yes, I have pain", use_container_width=True):
                st.session_state.pain_yesno = True
                patient_msg = "Yes, I have pain today."
                st.session_state.gpt_history.append({"role": "patient", "content": patient_msg})
                with st.spinner("Assistant is thinking…"):
                    reply = get_gpt_reply()
                st.session_state.gpt_history.append({"role": "doctor", "content": reply})
                st.session_state.reply_stage1 = reply
                st.rerun()
        with c2:
            if st.button("🙂 No pain today", use_container_width=True):
                st.session_state.pain_yesno = False
                patient_msg = "No, I don't have any pain today."
                st.session_state.gpt_history.append({"role": "patient", "content": patient_msg})
                with st.spinner("Assistant is thinking…"):
                    reply = get_gpt_reply()
                st.session_state.gpt_history.append({"role": "doctor", "content": reply})
                st.session_state.reply_stage1 = reply
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.reply_stage1:
        gpt_reply_box(st.session_state.reply_stage1)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Next →", key="next_1", use_container_width=True):
            # Skip body map if no pain
            st.session_state.stage = 2 if st.session_state.pain_yesno else 3
            st.rerun()

# ============================================================
# STAGE 2 — Body pain map
# ============================================================
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

    if not st.session_state.reply_stage2:
        cA, cB = st.columns(2)
        with cA:
            if st.button("Clear all"):
                st.session_state.selected_parts = set()
                st.rerun()
        with cB:
            if st.button("Send locations ➜"):
                loc_text = ", ".join(sorted(st.session_state.selected_parts)) if st.session_state.selected_parts else "not sure of exact location"
                patient_msg = f"Pain locations: {loc_text}."
                st.session_state.gpt_history.append({"role": "patient", "content": patient_msg})
                with st.spinner("Assistant is thinking…"):
                    reply = get_gpt_reply()
                st.session_state.gpt_history.append({"role": "doctor", "content": reply})
                st.session_state.reply_stage2 = reply
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.reply_stage2:
        gpt_reply_box(st.session_state.reply_stage2)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Next →", key="next_2", use_container_width=True):
            st.session_state.stage = 3
            st.rerun()

# ============================================================
# STAGE 3 — Symptom checklist
# ============================================================
elif stage == 3:
    st.markdown('<div class="panel"><div class="panel-title">Any of these symptoms today?</div>', unsafe_allow_html=True)

    symptom_options = [
        "Fatigue / low energy", "Nausea", "Vomiting", "Poor appetite",
        "Mouth sores", "Trouble swallowing", "Shortness of breath",
        "Fever / chills", "Constipation", "Diarrhea",
        "Sleep problems", "Anxiety / low mood",
    ]

    for symptom in symptom_options:
        label = f"✓ {symptom}" if symptom in st.session_state.symptoms else symptom
        if st.button(label, key=f"symptom_{symptom}", use_container_width=True):
            if symptom in st.session_state.symptoms:
                st.session_state.symptoms.remove(symptom)
            else:
                st.session_state.symptoms.append(symptom)
            st.rerun()

    if not st.session_state.reply_stage3:
        if st.button("Send symptoms ➜", use_container_width=True):
            sym_text = "; ".join(st.session_state.symptoms) if st.session_state.symptoms else "no symptoms from the checklist"
            patient_msg = f"Symptoms today: {sym_text}."
            st.session_state.gpt_history.append({"role": "patient", "content": patient_msg})
            with st.spinner("Assistant is thinking…"):
                reply = get_gpt_reply()
            st.session_state.gpt_history.append({"role": "doctor", "content": reply})
            st.session_state.reply_stage3 = reply
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.reply_stage3:
        gpt_reply_box(st.session_state.reply_stage3)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Next →", key="next_3", use_container_width=True):
            st.session_state.stage = 4
            st.rerun()

# ============================================================
# STAGE 4 — Free chat + submit
# ============================================================
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

        # Show free-chat history (only stage-4 turns)
        stage4_msgs = [m for m in st.session_state.gpt_history if m.get("stage4")]
        if stage4_msgs:
            st.markdown('<div class="chat-window">', unsafe_allow_html=True)
            for msg in stage4_msgs:
                if msg["role"] == "doctor":
                    st.markdown(f'<div class="row-left"><div class="avatar">🩺</div><div class="bubble-doc">{msg["content"]}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="row-right"><div class="bubble-pat">{msg["content"]}</div><div class="avatar">🙂</div></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Voice + text input
        cols = st.columns([5, 1])
        with cols[0]:
            user_text = st.chat_input("Reply to the assistant or type anything…", key="chat_input_main")
        audio_value = None
        with cols[1]:
            if hasattr(st, "audio_input"):
                audio_value = st.audio_input("🎙️", key="mic_input")
            else:
                st.caption("🎙️ Upgrade Streamlit for voice")

        if user_text:
            msg = {"role": "patient", "content": user_text, "stage4": True}
            st.session_state.gpt_history.append(msg)
            with st.spinner("Assistant is thinking…"):
                reply = get_gpt_reply()
            st.session_state.gpt_history.append({"role": "doctor", "content": reply, "stage4": True})
            st.rerun()

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
                    st.session_state.gpt_history.append({"role": "patient", "content": transcribed, "stage4": True})
                    with st.spinner("Assistant is thinking…"):
                        reply = get_gpt_reply()
                    st.session_state.gpt_history.append({"role": "doctor", "content": reply, "stage4": True})
                    st.rerun()
                else:
                    st.warning(f"Could not transcribe. {transcribed}")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("✅ Submit Check-In", use_container_width=True):
            try:
                save_to_sheet()
                st.session_state.submitted = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save to Google Sheets: {e}")
