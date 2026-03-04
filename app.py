import hashlib
import io
import json
from datetime import datetime
from typing import Dict, List, Optional, Set

import gspread
import streamlit as st
from google.oauth2.service_account import Credentials
from openai import OpenAI

st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ============================================================
# Secrets helpers
# ============================================================
def _secret(*keys, default=None):
    for k in keys:
        if k in st.secrets:
            return st.secrets[k]
    return default


def _require_secret(*keys):
    value = _secret(*keys)
    if value is None:
        raise KeyError(f"Missing secret. Tried: {', '.join(keys)}")
    return value


def _service_account_info() -> dict:
    raw = _require_secret("gcp_service_account")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    raise ValueError("gcp_service_account must be a dict or JSON string.")


# ============================================================
# OpenAI
# ============================================================
OPENAI_API_KEY = _secret("openai_api_key", "OPENAI_API_KEY", "openai_key")
openai_client: Optional[OpenAI] = None
openai_init_error: Optional[str] = None

if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as exc:
        openai_init_error = str(exc)
else:
    openai_init_error = "OpenAI API key not found."


def _openai_ready() -> bool:
    return openai_client is not None and openai_init_error is None


# ============================================================
# Google Sheets
# ============================================================
sheet = None
sheets_init_error: Optional[str] = None


def _init_sheets():
    global sheet, sheets_init_error
    if sheet is not None or sheets_init_error is not None:
        return

    try:
        creds = Credentials.from_service_account_info(
            _service_account_info(),
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        book = gspread.authorize(creds).open_by_key(_require_secret("gsheet_id"))
        try:
            sheet_local = book.worksheet("Form")
        except Exception:
            sheet_local = book.add_worksheet(title="Form", rows=2000, cols=20)
            sheet_local.append_row(["timestamp", "name", "json"])
        sheet = sheet_local
    except Exception as exc:
        sheets_init_error = str(exc)



def load_past_checkins(name: str) -> List[Dict]:
    _init_sheets()
    if sheet is None:
        return []

    try:
        rows = sheet.get_all_values()
        past = []
        for row in rows[1:]:
            if len(row) < 3:
                continue
            if row[1].strip().lower() != name.strip().lower():
                continue
            try:
                payload = json.loads(row[2])
                payload["timestamp"] = row[0]
                past.append(payload)
            except Exception:
                continue
        return past[-5:]
    except Exception:
        return []



def save_to_sheet():
    _init_sheets()
    if sheet is None:
        raise RuntimeError(f"Sheets unavailable: {sheets_init_error}")

    payload = {
        "feeling_level": st.session_state.feeling_level,
        "pain": st.session_state.pain_yesno,
        "pain_locations": sorted(list(st.session_state.selected_parts)),
        "symptoms": st.session_state.symptoms,
        "conversation": st.session_state.messages,
    }
    sheet.append_row(
        [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            st.session_state.get("patient_name", "Unknown"),
            json.dumps(payload),
        ]
    )


# ============================================================
# Prompting
# ============================================================
def build_system_prompt(extra_context: str = "") -> str:
    name = st.session_state.get("patient_name", "the patient")
    feeling = st.session_state.get("feeling_level")
    pain = st.session_state.get("pain_yesno")
    locations = sorted(list(st.session_state.get("selected_parts", set())))
    symptoms = st.session_state.get("symptoms", [])

    today_lines = []
    if feeling is not None:
        today_lines.append(f"- Feeling: {feeling}/10")
    if pain is not None:
        today_lines.append(f"- Pain: {'yes' if pain else 'no'}")
    if locations:
        today_lines.append(f"- Pain location: {', '.join(locations)}")
    if symptoms:
        today_lines.append(f"- Symptoms: {', '.join(symptoms)}")
    today_summary = "\n".join(today_lines) if today_lines else "Nothing collected yet."

    past = st.session_state.get("past_checkins", [])
    if past:
        memory_lines = []
        for p in past:
            memory_lines.append(
                f"[{p.get('timestamp', '?')}] "
                f"Feeling:{p.get('feeling_level', '?')}/10 | "
                f"Pain:{'yes' if p.get('pain') else 'no'} | "
                f"Locations:{', '.join(p.get('pain_locations', [])) or 'none'} | "
                f"Symptoms:{', '.join(p.get('symptoms', [])) or 'none'}"
            )
        memory_summary = "\n".join(memory_lines)
    else:
        memory_summary = "No previous check-ins."

    ctx = f"\nCURRENT TASK:\n{extra_context}\n" if extra_context else ""

    return f"""You are a virtual symptom-intake assistant for a cancer care clinic.
You are doing a brief daily check-in with {name}.

TODAY'S DATA:
{today_summary}

PATIENT HISTORY:
{memory_summary}
{ctx}
RULES:
1. Ask exactly ONE question at a time.
2. Be brief. No pleasantries or filler such as 'Thank you', 'I\'m sorry to hear that', or 'Great'.
3. Stay on physical and emotional symptoms relevant to the check-in.
4. Do not give medical advice. If asked for advice, say: 'Your care team will follow up with you.'
5. Keep each reply concise and focused.
"""



def get_gpt_reply(extra_context: str = "") -> str:
    if not _openai_ready():
        return "(Assistant unavailable — check OpenAI API key.)"

    messages = [{"role": "system", "content": build_system_prompt(extra_context)}]

    for m in st.session_state.messages[-20:]:
        messages.append(
            {
                "role": "assistant" if m.get("role") == "doctor" else "user",
                "content": m.get("content", ""),
            }
        )

    try:
        response = openai_client.chat.completions.create(
            model=_secret("openai_model", default="gpt-4o-mini"),
            messages=messages,
            max_tokens=200,
            temperature=0.4,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"(Error: {exc})"



def transcribe_audio(audio_bytes: bytes) -> str:
    if not _openai_ready():
        return "(Transcription unavailable.)"
    try:
        wav_file = io.BytesIO(audio_bytes)
        wav_file.name = "recording.wav"
        result = openai_client.audio.transcriptions.create(
            model=_secret("whisper_model", default="whisper-1"),
            file=wav_file,
            language="en",
        )
        return (result.text or "").strip()
    except Exception as exc:
        return f"(Transcription failed: {exc})"


# ============================================================
# Styles
# ============================================================
st.markdown(
    """
<style>
[data-testid="stAppViewContainer"]{ background:linear-gradient(135deg,#eef4ff,#f8fbff); }
.header{ font-size:26px; font-weight:700; margin:8px 0 6px 0; }
.subheader{ color:#61708a; font-size:14px; margin-bottom:14px; }
.chat-window{
    max-height:50vh; overflow-y:auto; padding:16px 14px; border-radius:18px;
    background:rgba(255,255,255,0.55); border:1px solid rgba(200,210,230,0.55);
    backdrop-filter:blur(10px); margin-bottom:12px;
}
.row-left{ display:flex; justify-content:flex-start; align-items:flex-end; margin:8px 0; gap:8px; }
.row-right{ display:flex; justify-content:flex-end; align-items:flex-end; margin:8px 0; gap:8px; }
.avatar{
    width:32px; height:32px; border-radius:50%; display:flex; justify-content:center;
    align-items:center; background:rgba(255,255,255,0.9); border:1px solid rgba(210,220,240,0.9);
    box-shadow:0 2px 8px rgba(0,0,0,0.08); font-size:16px; flex:0 0 auto;
}
.bubble-doc{
    background:#fff; border:1px solid rgba(220,225,235,0.95); border-radius:16px;
    padding:10px 13px; max-width:74%; box-shadow:0 2px 8px rgba(0,0,0,0.05); white-space:pre-wrap;
}
.bubble-pat{
    background:#1f7aff; color:white; border-radius:16px; padding:10px 13px;
    max-width:74%; box-shadow:0 2px 8px rgba(0,0,0,0.08); white-space:pre-wrap;
}
.panel{
    margin-top:10px; padding:14px 16px; border-radius:16px;
    background:rgba(255,255,255,0.7); border:1px solid rgba(200,210,230,0.6);
    backdrop-filter:blur(10px);
}
.panel-title{ font-weight:700; font-size:16px; margin-bottom:10px; }
.small-note{ color:#627089; font-size:12px; margin-top:4px; line-height:1.45; }
.divider{ color:#6b7a90; font-size:12px; text-align:center; margin:14px 0 8px 0; font-weight:600; }
.stButton>button{
    border-radius:20px !important; padding:0.45rem 1rem !important;
    font-size:14px !important; border:1px solid rgba(180,195,220,0.8) !important;
    background:white !important; color:#1a2540 !important;
    box-shadow:0 1px 4px rgba(0,0,0,0.06) !important;
    transition: all 0.15s ease !important; min-height:42px !important;
}
.stButton>button:hover{ background:#f0f6ff !important; border-color:#1f7aff !important; }
[data-testid="stAudioInput"]{ margin:0 !important; padding:0 !important; }
[data-testid="stAudioInput"]>label{ display:none !important; }
[data-testid="stAudioInput"]>div{
    height:36px !important; border-radius:20px !important;
    display:flex !important; align-items:center !important; justify-content:center !important;
    border:1px solid rgba(180,195,220,0.8) !important;
}
.tag{display:inline-block;background:rgba(31,122,255,0.09);color:#1f5acc;border-radius:20px;padding:2px 10px;font-size:13px;margin:2px 3px 2px 0;}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Session state
# ============================================================
DEFAULTS = {
    "messages": [],
    "stage": -1,
    "patient_name": "",
    "selected_parts": set(),
    "pain_yesno": None,
    "feeling_level": None,
    "symptoms": [],
    "submitted": False,
    "past_checkins": [],
    "last_audio_hash": None,
    "mic_key_counter": 0,
    "followup_counts": {},
    "stage_answered": {},
    "current_stage_prompt": {},
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


MAX_FOLLOWUPS = {0: 1, 1: 3, 2: 3, 3: 3, 4: 3}


# ============================================================
# State helpers
# ============================================================
def add_doctor(text: str):
    st.session_state.messages.append({"role": "doctor", "content": text})



def add_patient(text: str):
    st.session_state.messages.append({"role": "patient", "content": text})



def stage_answered(stage_id: int) -> bool:
    return bool(st.session_state.stage_answered.get(stage_id, False))



def mark_stage_answered(stage_id: int):
    st.session_state.stage_answered[stage_id] = True



def set_stage_prompt(stage_id: int, text: str):
    st.session_state.current_stage_prompt[stage_id] = text



def get_stage_prompt(stage_id: int) -> str:
    return st.session_state.current_stage_prompt.get(stage_id, "")



def complete_followups(stage_id: int):
    st.session_state.followup_counts[stage_id] = MAX_FOLLOWUPS.get(stage_id, 0)



def followup_count(stage_id: int) -> int:
    return int(st.session_state.followup_counts.get(stage_id, 0))



def can_ask_followup(stage_id: int) -> bool:
    return followup_count(stage_id) < MAX_FOLLOWUPS.get(stage_id, 0)



def record_followup(stage_id: int):
    st.session_state.followup_counts[stage_id] = followup_count(stage_id) + 1



def toggle_body_part(part: str):
    if part in st.session_state.selected_parts:
        st.session_state.selected_parts.remove(part)
    else:
        st.session_state.selected_parts.add(part)



def next_stage_after(current_stage: int) -> int:
    if current_stage == 0:
        return 1
    if current_stage == 1:
        return 2
    if current_stage == 2:
        return 4 if st.session_state.pain_yesno is False else 3
    if current_stage == 3:
        return 4
    if current_stage == 4:
        return 5
    return current_stage



def advance_stage():
    st.session_state.stage = next_stage_after(st.session_state.stage)



def submit_stage_response(patient_text: str, stage_id: int, extra_context: str = ""):
    add_patient(patient_text)
    mark_stage_answered(stage_id)

    if can_ask_followup(stage_id):
        record_followup(stage_id)
        with st.spinner("Writing the next question…"):
            reply = get_gpt_reply(extra_context=extra_context).strip()

        if reply == "READY_TO_CONTINUE":
            complete_followups(stage_id)
            st.session_state.current_stage_prompt.pop(stage_id, None)
        elif reply:
            add_doctor(reply)
            set_stage_prompt(stage_id, reply)



def handle_audio(audio_value, stage_id: int, extra_context: str = "") -> bool:
    if audio_value is None:
        return False
    try:
        audio_bytes = audio_value.getvalue()
        audio_hash = hashlib.sha1(audio_bytes).hexdigest()
    except Exception:
        return False

    if not audio_bytes or audio_hash == st.session_state.last_audio_hash:
        return False

    st.session_state.last_audio_hash = audio_hash
    st.session_state.mic_key_counter += 1
    with st.spinner("Transcribing your recording…"):
        transcript = transcribe_audio(audio_bytes)

    if transcript and not transcript.startswith("(Transcription failed"):
        st.info(f'Heard: "{transcript}"')
        submit_stage_response(transcript, stage_id, extra_context)
        return True

    st.warning("I couldn't transcribe that. Please try again.")
    return False



def render_chat_window():
    st.markdown('<div class="chat-window">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg.get("role") == "doctor":
            st.markdown(
                f'<div class="row-left"><div class="avatar">🩺</div><div class="bubble-doc">{msg.get("content", "")}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="row-right"><div class="bubble-pat">{msg.get("content", "")}</div><div class="avatar">🙂</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)





def render_inline_followup(stage_id: int):
    prompt = get_stage_prompt(stage_id)
    if not prompt:
        return
    st.markdown(
        f'<div class="row-left" style="margin-top:10px;"><div class="avatar">🩺</div><div class="bubble-doc" style="max-width:100%;">{prompt}</div></div>',
        unsafe_allow_html=True,
    )


def render_structured_history():
    with st.expander("Show conversation so far", expanded=False):
        render_chat_window()

def render_type_or_speak(stage_id: int, extra_context: str, placeholder: str):
    text_col, send_col, mic_col = st.columns([5, 1, 1], gap="small")
    with text_col:
        typed = st.text_input(
            "",
            placeholder=placeholder,
            key=f"inline_buf_{stage_id}_{followup_count(stage_id)}",
            label_visibility="collapsed",
        )
    with send_col:
        send_clicked = st.button("↑", key=f"inline_send_{stage_id}_{followup_count(stage_id)}", use_container_width=True)
    with mic_col:
        audio_value = None
        if hasattr(st, "audio_input"):
            audio_value = st.audio_input(
                "",
                key=f"mic_{stage_id}_{st.session_state.mic_key_counter}",
                label_visibility="collapsed",
            )

    if send_clicked and typed and typed.strip():
        submit_stage_response(typed.strip(), stage_id, extra_context)
        st.rerun()

    if handle_audio(audio_value, stage_id, extra_context):
        st.rerun()



def render_stage_footer(stage_id: int, next_label: str, allow_skip_while_followups_remain: bool = False):
    if not stage_answered(stage_id):
        return

    ready_for_next = (not can_ask_followup(stage_id)) or allow_skip_while_followups_remain

    if can_ask_followup(stage_id):
        st.markdown('<div class="small-note">You can answer the question above, or continue when you are ready.</div>', unsafe_allow_html=True)
    else:
        st.markdown("<div class='small-note'>You're ready for the next step.</div>", unsafe_allow_html=True)

    if ready_for_next:
        if st.button(next_label, use_container_width=True, type="primary", key=f"next_{stage_id}"):
            advance_stage()
            st.rerun()



def body_svg(selected: Set[str]) -> str:
    def fill(part):
        return "#1f7aff" if part in selected else "#cfd8e6"

    stroke = "#6b7a90"
    return f"""<svg width="220" height="360" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">
  <defs><filter id="sh"><feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="rgba(0,0,0,0.12)"/></filter></defs>
  <g filter="url(#sh)"><circle cx="160" cy="70" r="38" fill="{fill('Head')}" stroke="{stroke}" stroke-width="2"/></g>
  <g filter="url(#sh)"><rect x="110" y="120" width="100" height="70" rx="24" fill="{fill('Chest')}" stroke="{stroke}" stroke-width="2"/></g>
  <g filter="url(#sh)"><rect x="115" y="195" width="90" height="70" rx="22" fill="{fill('Abdomen')}" stroke="{stroke}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M110 132 C80 145,72 180,78 220 C82 250,92 270,100 290 C108 310,115 320,120 320 L120 130Z" fill="{fill('Left Arm')}" stroke="{stroke}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M210 132 C240 145,248 180,242 220 C238 250,228 270,220 290 C212 310,205 320,200 320 L200 130Z" fill="{fill('Right Arm')}" stroke="{stroke}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M135 265 C120 310,118 360,126 410 C132 445,132 475,128 500 L155 500 C158 470,160 435,156 405 C150 355,152 312,165 265Z" fill="{fill('Left Leg')}" stroke="{stroke}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M185 265 C200 310,202 360,194 410 C188 445,188 475,192 500 L165 500 C162 470,160 435,164 405 C170 355,168 312,155 265Z" fill="{fill('Right Leg')}" stroke="{stroke}" stroke-width="2"/></g>
</svg>""".strip()


# ============================================================
# Global messages
# ============================================================
if openai_init_error:
    st.warning(f"LLM not ready: {openai_init_error}")
_init_sheets()
if sheets_init_error:
    st.warning(f"Sheets not ready: {sheets_init_error}")

st.markdown('<div class="header">🩺 Cancer Symptom Check-In</div><div class="subheader">A quick daily check-in to help your care team understand how you are feeling today.</div>', unsafe_allow_html=True)


# ============================================================
# Stage -1 — Name entry
# ============================================================
if st.session_state.stage == -1:
    st.markdown('<div class="panel"><div class="panel-title">Welcome</div>', unsafe_allow_html=True)
    st.markdown("<div class='small-note'>Please enter the name you would like used for today's check-in.</div>", unsafe_allow_html=True)
    name_input = st.text_input("Your name", value=st.session_state.patient_name, placeholder="Enter your name")

    if st.button("Begin check-in", use_container_width=True, type="primary"):
        if not name_input.strip():
            st.warning("Please enter your name to continue.")
            st.stop()

        clean_name = name_input.strip()
        st.session_state.patient_name = clean_name
        with st.spinner("Loading recent check-ins…"):
            st.session_state.past_checkins = load_past_checkins(clean_name)

        past = st.session_state.past_checkins
        if past:
            last = past[-1]
            context = (
                f"Most recent visit: {last.get('timestamp', '?')}. "
                f"Feeling {last.get('feeling_level', '?')}/10. "
                f"Pain {'yes' if last.get('pain') else 'no'}. "
                f"Locations: {', '.join(last.get('pain_locations', [])) or 'none'}. "
                f"Symptoms: {', '.join(last.get('symptoms', [])) or 'none'}. "
                f"Briefly summarize the last visit, then ask one follow-up question about how they have been since then."
            )
            with st.spinner("Preparing your check-in…"):
                opening = get_gpt_reply(extra_context=context)
            add_doctor(opening)
            st.session_state.stage = 0
        else:
            context = (
                f"This is {clean_name}'s first check-in. Introduce the check-in briefly and say you will ask a few quick questions."
            )
            with st.spinner("Preparing your check-in…"):
                opening = get_gpt_reply(extra_context=context)
            if not opening or opening.startswith("("):
                opening = f"Hi {clean_name}. I'll ask a few quick questions for today's check-in."
            add_doctor(opening)
            st.session_state.stage = 1
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


stage = st.session_state.stage
if stage in (0, 5):
    render_chat_window()


# ============================================================
# Stage 0 — History recap
# ============================================================
if stage == 0:
    history_ctx = (
        "The patient is replying about how they have been since the last visit. "
        "Ask at most one focused follow-up question. Do not begin today's structured questionnaire yet."
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">💬 Since your last check-in, how have you been feeling?</div>', unsafe_allow_html=True)
    render_type_or_speak(0, history_ctx, "Type or speak your update…")
    st.markdown("</div>", unsafe_allow_html=True)

    if stage_answered(0):
        label = "Start today\'s check-in →" if not can_ask_followup(0) else "Skip to check-in →"
        render_stage_footer(0, label, allow_skip_while_followups_remain=True)
    else:
        if st.button("Skip to today's questions →", use_container_width=True, type="primary", key="skip_stage_0"):
            advance_stage()
            st.rerun()


# ============================================================
# Stage 1 — Feeling scale
# ============================================================
elif stage == 1:
    feeling_ctx = (
        "The patient just answered the 0-10 feeling question. "
        "Ask one focused follow-up question about what is driving the score only if more detail is still needed. "
        "If you already have enough information for this stage, respond exactly READY_TO_CONTINUE."
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">How are you feeling overall today?</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Choose a number, or answer in your own words below.</div>', unsafe_allow_html=True)

    scale_cols = st.columns(11)
    for i in range(11):
        with scale_cols[i]:
            label = f"✓ {i}" if st.session_state.feeling_level == i else str(i)
            if st.button(label, key=f"feel_{i}", use_container_width=True):
                st.session_state.feeling_level = i
                st.rerun()

    if st.session_state.feeling_level is not None:
        st.markdown(
            f"<div class='small-note'>Selected score: <b>{st.session_state.feeling_level}/10</b></div>",
            unsafe_allow_html=True,
        )
        if not stage_answered(1):
            if st.button("Use this score ➜", use_container_width=True, key="send_feeling_score"):
                submit_stage_response(
                    f"My feeling level today is {st.session_state.feeling_level}/10.",
                    1,
                    feeling_ctx,
                )
                st.rerun()

    if stage_answered(1):
        render_inline_followup(1)
        if can_ask_followup(1):
            st.markdown('<div class="divider">— answer the question above, or type / speak here —</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="divider">— you can add more detail, or continue below —</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="divider">— or type / speak your answer —</div>', unsafe_allow_html=True)
    render_type_or_speak(1, feeling_ctx, "Describe how you're feeling today…")
    st.markdown("</div>", unsafe_allow_html=True)
    render_structured_history()
    render_stage_footer(1, "Next question →", allow_skip_while_followups_remain=False)


# ============================================================
# Stage 2 — Pain yes/no
# ============================================================
elif stage == 2:
    pain_ctx = (
        "The patient just answered whether they have pain today. "
        "Ask one adaptive follow-up question only if more detail is still needed. If they said yes, focus on pain details. If they said no, confirm whether they have any other physical discomfort worth noting. "
        "If you already have enough information for this stage, respond exactly READY_TO_CONTINUE."
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Are you having any pain today?</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Choose an option, or answer in your own words below.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Yes", use_container_width=True, key="pain_yes"):
            st.session_state.pain_yesno = True
            submit_stage_response("Yes, I have pain today.", 2, pain_ctx)
            st.rerun()
    with c2:
        if st.button("🙂 No", use_container_width=True, key="pain_no"):
            st.session_state.pain_yesno = False
            submit_stage_response("No, I do not have pain today.", 2, pain_ctx)
            st.rerun()

    if stage_answered(2):
        render_inline_followup(2)
        if can_ask_followup(2):
            st.markdown('<div class="divider">— answer the question above, or type / speak here —</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="divider">— you can add more detail, or continue below —</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="divider">— or type / speak your answer —</div>', unsafe_allow_html=True)
    render_type_or_speak(2, pain_ctx, "Type or speak your answer…")
    st.markdown("</div>", unsafe_allow_html=True)
    render_structured_history()
    render_stage_footer(2, "Next question →", allow_skip_while_followups_remain=False)


# ============================================================
# Stage 3 — Body map
# ============================================================
elif stage == 3:
    location_ctx = (
        "The patient just identified where the pain is located. "
        "Ask one focused follow-up question about those locations, such as severity, duration, pattern, or triggers, only if more detail is still needed. "
        "If you already have enough information for this stage, respond exactly READY_TO_CONTINUE."
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Where is the pain located?</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Select body areas, or describe the location below.</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown(body_svg(st.session_state.selected_parts), unsafe_allow_html=True)
    with right:
        st.markdown("**Select body areas:**")
        for part in ["Head", "Chest", "Abdomen", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]:
            label = f"✓ {part}" if part in st.session_state.selected_parts else part
            if st.button(label, key=f"toggle_{part}", use_container_width=True):
                toggle_body_part(part)
                st.rerun()
        st.markdown(
            '<div class="small-note">Current selection: '
            + (", ".join(sorted(st.session_state.selected_parts)) or "None")
            + "</div>",
            unsafe_allow_html=True,
        )

    if not stage_answered(3):
        if st.button("Use these locations ➜", use_container_width=True, key="send_locations"):
            selected_text = ", ".join(sorted(st.session_state.selected_parts)) or "I am not sure where the pain is"
            submit_stage_response(f"Pain location: {selected_text}.", 3, location_ctx)
            st.rerun()

    if stage_answered(3):
        render_inline_followup(3)
        if can_ask_followup(3):
            st.markdown('<div class="divider">— answer the question above, or type / speak here —</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="divider">— you can add more detail, or continue below —</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="divider">— or type / speak your answer —</div>', unsafe_allow_html=True)
    render_type_or_speak(3, location_ctx, "Describe where the pain is…")
    st.markdown("</div>", unsafe_allow_html=True)
    render_structured_history()
    render_stage_footer(3, "Next question →", allow_skip_while_followups_remain=False)


# ============================================================
# Stage 4 — Symptom checklist
# ============================================================
elif stage == 4:
    symptom_ctx = (
        "The patient just submitted symptom information. "
        "Ask one focused follow-up question about the most important symptom mentioned only if more detail is still needed. "
        "If you already have enough information for this stage, respond exactly READY_TO_CONTINUE."
    )
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

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Are you having any of these symptoms today?</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Choose any that apply, then continue. You can also describe symptoms below.</div>', unsafe_allow_html=True)

    cols_symp = st.columns(2)
    for idx, symptom in enumerate(symptom_options):
        with cols_symp[idx % 2]:
            label = f"✓ {symptom}" if symptom in st.session_state.symptoms else symptom
            if st.button(label, key=f"sym_{idx}", use_container_width=True):
                if symptom in st.session_state.symptoms:
                    st.session_state.symptoms.remove(symptom)
                else:
                    st.session_state.symptoms.append(symptom)
                st.rerun()

    if st.session_state.symptoms:
        st.markdown(
            '<div class="small-note">Current selection: ' + "; ".join(st.session_state.symptoms) + "</div>",
            unsafe_allow_html=True,
        )

    if not stage_answered(4):
        if st.button("Use these symptoms ➜", use_container_width=True, key="send_symptoms"):
            symptoms_text = "; ".join(st.session_state.symptoms) if st.session_state.symptoms else "no symptoms from the checklist"
            submit_stage_response(f"Symptoms today: {symptoms_text}.", 4, symptom_ctx)
            st.rerun()

    if stage_answered(4):
        render_inline_followup(4)
        if can_ask_followup(4):
            st.markdown('<div class="divider">— answer the question above, or type / speak here —</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="divider">— you can add more detail, or continue below —</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="divider">— or type / speak your answer —</div>', unsafe_allow_html=True)
    render_type_or_speak(4, symptom_ctx, "Describe any symptoms you want to mention…")
    st.markdown("</div>", unsafe_allow_html=True)
    render_structured_history()
    render_stage_footer(4, "Continue to final notes →", allow_skip_while_followups_remain=False)


# ============================================================
# Stage 5 — Free chat + submit
# ============================================================
elif stage == 5:
    if st.session_state.submitted:
        name = st.session_state.get("patient_name", "—")
        feeling = st.session_state.get("feeling_level")
        pain = st.session_state.get("pain_yesno")
        locations = sorted(list(st.session_state.get("selected_parts", set())))
        symptoms = st.session_state.get("symptoms", [])

        widget_msgs = {
            f"My feeling level today is {feeling}/10.",
            "Yes, I have pain today.",
            "No, I do not have pain today.",
            f"Pain location: {', '.join(locations)}." if locations else "",
            f"Symptoms today: {'; '.join(symptoms)}." if symptoms else "Symptoms today: no symptoms from the checklist.",
        }

        patient_lines = [
            m["content"]
            for m in st.session_state.messages
            if m.get("role") == "patient" and m.get("content", "") not in widget_msgs
        ]

        summary_text = "None"
        if patient_lines and _openai_ready():
            try:
                response = openai_client.chat.completions.create(
                    model=_secret("openai_model", default="gpt-4o-mini"),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Extract only clinically relevant facts from the patient's free-text messages. "
                                "Focus on symptom details, severity, duration, triggers, appetite, sleep, energy, bowel issues, mood, and breathing. "
                                "Return a bullet list. If nothing clinically relevant is present, return 'None'."
                            ),
                        },
                        {"role": "user", "content": "\n".join(f"- {line}" for line in patient_lines)},
                    ],
                    max_tokens=300,
                    temperature=0.2,
                )
                summary_text = (response.choices[0].message.content or "").strip()
            except Exception:
                summary_text = "None"

        if summary_text and summary_text != "None":
            items = [line.lstrip("•-– ").strip() for line in summary_text.split("\n") if line.strip() and line.strip() != "None"]
            conv_cell = "<ul style='margin:0;padding-left:18px;'>" + "".join(
                f"<li style='margin-bottom:4px;font-size:14px;color:#1a2540'>{item}</li>" for item in items
            ) + "</ul>"
        else:
            conv_cell = "<span style='opacity:.4'>No additional details provided</span>"

        feeling_display = f"{feeling}/10" if feeling is not None else "—"
        pain_display = "Yes" if pain is True else ("No" if pain is False else "—")
        locations_html = "".join(f'<span class="tag">{loc}</span>' for loc in locations) or "<span style='opacity:.4'>N/A</span>"
        symptoms_html = "".join(f'<span class="tag">{sym}</span>' for sym in symptoms) or "<span style='opacity:.4'>None</span>"

        st.markdown(
            f"""
<div class="panel">
  <div class="panel-title">✅ Check-in summary — {name}</div>
  <table style="width:100%;border-collapse:collapse;font-size:14px;">
    <tr><td style="padding:8px 6px;font-weight:600;width:36%;">Name</td><td style="padding:8px 6px;">{name}</td></tr>
    <tr><td style="padding:8px 6px;font-weight:600;">Feeling score</td><td style="padding:8px 6px;">{feeling_display}</td></tr>
    <tr><td style="padding:8px 6px;font-weight:600;">Pain today</td><td style="padding:8px 6px;">{pain_display}</td></tr>
    <tr><td style="padding:8px 6px;font-weight:600;">Pain location</td><td style="padding:8px 6px;">{locations_html}</td></tr>
    <tr><td style="padding:8px 6px;font-weight:600;">Symptoms</td><td style="padding:8px 6px;">{symptoms_html}</td></tr>
    <tr><td style="padding:8px 6px;font-weight:600;vertical-align:top;">Additional notes</td><td style="padding:8px 6px;">{conv_cell}</td></tr>
  </table>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='panel'><div class='panel-title'>💬 Is there anything else you would like to share?</div><div class='small-note'>You can add any final details here, then submit when you are ready.</div></div>",
            unsafe_allow_html=True,
        )
        cols = st.columns([5, 1], vertical_alignment="bottom")
        with cols[0]:
            user_text = st.chat_input("Type any final details…", key="chat_input_5")
        with cols[1]:
            audio_5 = None
            if hasattr(st, "audio_input"):
                audio_5 = st.audio_input("", key=f"mic_5_{st.session_state.mic_key_counter}", label_visibility="collapsed")

        if user_text:
            add_patient(user_text)
            with st.spinner("Writing the next question…"):
                reply = get_gpt_reply()
            add_doctor(reply)
            st.rerun()

        if audio_5 is not None:
            try:
                audio_bytes = audio_5.getvalue()
                audio_hash = hashlib.sha1(audio_bytes).hexdigest()
            except Exception:
                audio_bytes = None
                audio_hash = None

            if audio_bytes and audio_hash and audio_hash != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = audio_hash
                st.session_state.mic_key_counter += 1
                with st.spinner("Transcribing your recording…"):
                    transcript = transcribe_audio(audio_bytes)
                if transcript and not transcript.startswith("(Transcription failed"):
                    st.info(f'Heard: "{transcript}"')
                    add_patient(transcript)
                    with st.spinner("Writing the next question…"):
                        reply = get_gpt_reply()
                    add_doctor(reply)
                    st.rerun()
                else:
                    st.warning("I couldn't transcribe that. Please try again.")

        if st.button("✅ Submit today's check-in", use_container_width=True, type="primary"):
            try:
                save_to_sheet()
                st.session_state.submitted = True
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save the check-in: {exc}")
