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
st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ============================================================
# SECRETS HELPERS
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
        "conversation":   st.session_state.messages,
    }
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name      = st.session_state.get("patient_name", "Unknown")
    sheet.append_row([timestamp, name, json.dumps(chat_dict)])

# ============================================================
# PROMPTING
# ============================================================
def build_system_prompt(extra_context: str = "") -> str:
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
            mem.append(f"  [{ts}] Feeling:{fl}/10 | Pain:{pn} | Locations:{ploc} | Symptoms:{sym}")
        memory_str = "\n".join(mem)
    else:
        memory_str = "No previous check-ins — this may be their first session."

    context_block = f"\nCURRENT TASK:\n{extra_context}\n" if extra_context else ""

    return f"""You are a warm, empathetic virtual symptom-intake assistant for a cancer care clinic.
You are conducting a brief daily check-in with: {name}.

TODAY'S DATA (from structured widgets so far):
{session_str}

PATIENT HISTORY (past sessions):
{memory_str}
{context_block}
RULES:
1. Be warm, natural, conversational. Short sentences.
2. Ask exactly ONE focused question at a time — never list multiple questions.
3. Personalise using memory — mention recurring issues if relevant.
4. STRICT TOPIC ENFORCEMENT: Only discuss the patient's physical and emotional health.
   If the patient says ANYTHING off-topic, immediately redirect:
   "I'm only here to help with your check-in today. Let's stay focused — [question]."
   Never engage with off-topic content even briefly.
5. Never give medical advice, diagnoses, or treatment guidance.
   If asked: "I'm not able to give medical advice — your care team will follow up with you."
"""

def _openai_ready() -> bool:
    return openai_client is not None and openai_init_error is None

def get_gpt_reply(extra_context: str = "") -> str:
    if not _openai_ready():
        return "(Assistant not available — check OpenAI API key in secrets.)"

    msgs = [{"role": "system", "content": build_system_prompt(extra_context)}]

    # Inject past visits as synthetic turns for memory continuity
    past = st.session_state.get("past_checkins", [])
    for p in past:
        ts   = p.get("timestamp", "unknown date")
        fl   = p.get("feeling_level", "?")
        pn   = "yes" if p.get("pain") else "no"
        locs = ", ".join(p.get("pain_locations", [])) or "none"
        syms = ", ".join(p.get("symptoms", [])) or "none"
        msgs.append({"role": "user",
                     "content": f"[Past visit {ts}] Feeling:{fl}/10. Pain:{pn}. Locations:{locs}. Symptoms:{syms}."})
        msgs.append({"role": "assistant",
                     "content": f"Thank you, I've noted your check-in from {ts}."})

    for m in st.session_state.messages[-20:]:
        role = "assistant" if m.get("role") == "doctor" else "user"
        msgs.append({"role": role, "content": m.get("content", "")})

    try:
        r = openai_client.chat.completions.create(
            model=_secret("openai_model", default="gpt-4o-mini"),
            messages=msgs,
            max_tokens=350,
            temperature=0.6,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(Sorry, I couldn't connect right now. Error: {e})"

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
.divider-label{
    font-size:11px; font-weight:600; letter-spacing:0.07em; text-transform:uppercase;
    color:rgba(0,0,0,0.35); text-align:center; margin:12px 0 8px 0;
}
.stButton>button{ border-radius:14px; padding:0.55rem 0.9rem; }
[data-testid="stAudioInput"] {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
[data-testid="stAudioInput"] > label { display: none !important; }
[data-testid="stAudioInput"] > div {
    height: 52px !important;
    border-radius: 12px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stChatInput"]{ position:sticky; bottom:0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
# Stage map:
#  -1  name entry
#   0  history recap conversation (skip if no history)
#   1  feeling scale questionnaire + adaptive follow-ups
#   2  pain yes/no questionnaire + adaptive follow-ups
#   3  body map questionnaire + adaptive follow-ups (skip if no pain)
#   4  symptom checklist questionnaire + adaptive follow-ups
#   5  free chat + submit

defaults = {
    "messages":        [],
    "stage":           -1,
    "patient_name":    "",
    "selected_parts":  set(),
    "pain_yesno":      None,
    "feeling_level":   None,
    "symptoms":        [],
    "submitted":       False,
    "past_checkins":   [],
    "last_audio_hash": None,
    "mic_key_counter": 0,
    # followup_counts[stage_id] = number of GPT follow-ups fired so far in that stage
    "followup_counts": {},
    # stage_answered[stage_id] = True once the patient has given their first answer
    "stage_answered":  {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# CORE HELPERS
# ============================================================
def add_doctor(text: str) -> None:
    st.session_state.messages.append({"role": "doctor", "content": text})

def add_patient(text: str) -> None:
    st.session_state.messages.append({"role": "patient", "content": text})

def toggle_body_part(part: str) -> None:
    if part in st.session_state.selected_parts:
        st.session_state.selected_parts.remove(part)
    else:
        st.session_state.selected_parts.add(part)

# --- Follow-up budget helpers ---
MAX_FOLLOWUPS = {0: 1, 1: 3, 2: 3, 3: 3, 4: 3}  # stage → max GPT follow-ups

def followup_count(stage_id: int) -> int:
    return st.session_state.followup_counts.get(stage_id, 0)

def can_followup(stage_id: int) -> bool:
    return followup_count(stage_id) < MAX_FOLLOWUPS.get(stage_id, 0)

def record_followup(stage_id: int) -> None:
    st.session_state.followup_counts[stage_id] = followup_count(stage_id) + 1

def is_answered(stage_id: int) -> bool:
    return st.session_state.stage_answered.get(stage_id, False)

def mark_answered(stage_id: int) -> None:
    st.session_state.stage_answered[stage_id] = True

def advance_stage() -> None:
    s = st.session_state.stage
    if   s == 0: st.session_state.stage = 1
    elif s == 1: st.session_state.stage = 2
    elif s == 2:
        # Skip body map if patient said no pain
        if st.session_state.pain_yesno is False:
            st.session_state.stage = 4
        else:
            st.session_state.stage = 3
    elif s == 3: st.session_state.stage = 4
    elif s == 4: st.session_state.stage = 5

def on_patient_answer(text: str, stage_id: int, extra_context: str = "") -> None:
    """
    Handle any patient answer — widget click, typed, or voiced.
    Records the answer, then fires a GPT follow-up if budget allows,
    otherwise advances to the next stage automatically.
    """
    add_patient(text)
    mark_answered(stage_id)

    if can_followup(stage_id):
        record_followup(stage_id)
        with st.spinner("Assistant is thinking…"):
            reply = get_gpt_reply(extra_context=extra_context)
        add_doctor(reply)
    else:
        advance_stage()

def process_typed_input(user_text: str, stage_id: int, extra_context: str = "") -> None:
    if user_text:
        on_patient_answer(user_text, stage_id, extra_context)
        st.rerun()

def process_voice_input(audio_value, stage_id: int, extra_context: str = "") -> None:
    if audio_value is None:
        return
    try:
        audio_bytes = audio_value.getvalue()
        audio_hash  = hashlib.sha1(audio_bytes).hexdigest()
    except Exception:
        return
    if not audio_bytes or not audio_hash or audio_hash == st.session_state.last_audio_hash:
        return
    st.session_state.last_audio_hash = audio_hash
    st.session_state.mic_key_counter += 1
    with st.spinner("Transcribing your voice…"):
        transcribed = transcribe_audio(audio_bytes)
    if transcribed and not transcribed.startswith("(Transcription failed"):
        st.info(f'Heard: "{transcribed}"')
        on_patient_answer(transcribed, stage_id, extra_context)
        st.rerun()
    else:
        st.warning("Could not transcribe. Please try again or type your message.")

def render_chat_window() -> None:
    st.markdown('<div class="chat-window">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg.get("role") == "doctor":
            st.markdown(f"""
        <div class="row-left">
          <div class="avatar">🩺</div>
          <div class="bubble-doc">{msg.get("content","")}</div>
        </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
        <div class="row-right">
          <div class="bubble-pat">{msg.get("content","")}</div>
          <div class="avatar">🙂</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_free_input(stage_id: int, placeholder: str, extra_context: str = "") -> None:
    """Render the shared text + mic input bar with divider label."""
    st.markdown('<div class="divider-label">— or type / speak your answer —</div>', unsafe_allow_html=True)
    cols = st.columns([5, 1], vertical_alignment="bottom")
    with cols[0]:
        user_text = st.chat_input(placeholder, key=f"chat_input_{stage_id}")
    audio_value = None
    with cols[1]:
        if hasattr(st, "audio_input"):
            audio_value = st.audio_input(
                "", key=f"mic_{stage_id}_{st.session_state.mic_key_counter}",
                label_visibility="collapsed"
            )
        else:
            st.caption("Upgrade Streamlit for voice")
    process_typed_input(user_text, stage_id, extra_context)
    process_voice_input(audio_value, stage_id, extra_context)

def render_next_button(label: str = "Next →") -> None:
    """Show blue Next button that advances the stage."""
    if st.button(label, use_container_width=True, key=f"next_{st.session_state.stage}",
                 type="primary"):
        advance_stage()
        st.rerun()

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
  <text x="160" y="518" text-anchor="middle" font-size="12" fill="rgba(0,0,0,0.45)">Click buttons to toggle regions</text>
</svg>""".strip()

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

            past = st.session_state.past_checkins
            if past:
                # Build opening message grounded in last visit
                last = past[-1]
                ts   = last.get("timestamp", "your last visit")
                fl   = last.get("feeling_level", "?")
                pn   = "yes" if last.get("pain") else "no"
                ploc = ", ".join(last.get("pain_locations", [])) or "none"
                sym  = ", ".join(last.get("symptoms", [])) or "none"
                context = (
                    f"The patient's most recent visit was on {ts}. They reported: "
                    f"feeling level {fl}/10, pain: {pn}, pain locations: {ploc}, symptoms: {sym}. "
                    f"Greet {name_input.strip()} warmly by name, give a brief friendly summary of how "
                    f"they were doing at their last visit, and ask ONE follow-up question about how "
                    f"they have been since then."
                )
                with st.spinner("Getting your assistant ready…"):
                    opening = get_gpt_reply(extra_context=context)
                add_doctor(opening)
                st.session_state.stage = 0          # history recap
            else:
                # No history — go straight to questionnaires
                context = (
                    f"This is {name_input.strip()}'s very first check-in. "
                    f"Greet them warmly by name, briefly introduce yourself as a virtual check-in assistant, "
                    f"and let them know you'll ask a few quick questions about how they're feeling today."
                )
                with st.spinner("Getting your assistant ready…"):
                    opening = get_gpt_reply(extra_context=context)
                if not opening or opening.startswith("("):
                    opening = (
                        f"Hi {name_input.strip()}! I'm your virtual check-in assistant. "
                        "I'll ask you a few quick questions about how you're feeling today."
                    )
                add_doctor(opening)
                st.session_state.stage = 1          # skip history recap
            st.rerun()
        else:
            st.warning("Please enter your name to continue.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

stage = st.session_state.stage

# ============================================================
# CHAT WINDOW — always visible across all stages
# ============================================================
render_chat_window()

# ============================================================
# STAGE 0 — History recap
# GPT already asked the opening question (from name entry).
# Patient can reply (type or voice). GPT gets up to 1 follow-up.
# After the budget is spent, or the patient clicks Next, move to stage 1.
# ============================================================
if stage == 0:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">💬 Catching up from your last visit</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Reply to the assistant about how you\'ve been, then we\'ll move to today\'s questions.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Context for GPT follow-ups in this stage
    history_context = (
        "You are in the history recap stage. The patient is responding about how they have been "
        "since their last visit. React empathetically and ask ONE relevant follow-up question if needed. "
        "Do not start the structured check-in questions yet."
    )
    render_free_input(stage_id=0, placeholder="How have you been since your last visit?",
                      extra_context=history_context)

    # Show Next button once patient has answered AND either GPT has no more budget
    # or they want to skip ahead
    if is_answered(0):
        if not can_followup(0):
            st.markdown('<div class="small-note" style="margin-top:12px;">Great — let\'s move to today\'s check-in questions.</div>', unsafe_allow_html=True)
            render_next_button("Start today's check-in →")
        else:
            # Budget remains but patient can still skip
            st.markdown('<div class="small-note" style="margin-top:12px;">You can reply above, or skip to the check-in questions.</div>', unsafe_allow_html=True)
            render_next_button("Skip to check-in →")

# ============================================================
# STAGE 1 — Feeling scale
# ============================================================
elif stage == 1:
    feeling_context = (
        "The patient just answered the feeling scale question (0–10). "
        "React warmly to their score — if low, express empathy; if high, positive reinforcement. "
        "Ask ONE specific follow-up question to understand what is driving that score."
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">How are you feeling today?</div>', unsafe_allow_html=True)

    if not is_answered(1):
        st.markdown("Tap a number from **0** (worst) to **10** (best):")
        scale_cols = st.columns(11)
        for i in range(11):
            with scale_cols[i]:
                label = f"✓{i}" if st.session_state.feeling_level == i else str(i)
                if st.button(label, key=f"feeling_{i}", use_container_width=True):
                    st.session_state.feeling_level = i
                    st.rerun()

        if st.session_state.feeling_level is not None:
            st.markdown(
                f"<div style='font-size:18px;margin-top:8px;'>Selected: <b>{st.session_state.feeling_level}/10</b></div>",
                unsafe_allow_html=True
            )
            if st.button("Send feeling level ➜", use_container_width=True, key="send_feeling"):
                on_patient_answer(
                    f"My feeling level today is {st.session_state.feeling_level}/10.",
                    stage_id=1, extra_context=feeling_context
                )
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    render_free_input(stage_id=1, placeholder="Or describe how you feel in your own words…",
                      extra_context=feeling_context)

    if is_answered(1) and not can_followup(1):
        render_next_button("Next question →")

# ============================================================
# STAGE 2 — Pain yes/no
# ============================================================
elif stage == 2:
    pain_context = (
        "The patient just answered whether they have pain today. "
        "React empathetically and ask ONE follow-up question to understand more about their pain situation."
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Do you have any pain today?</div>', unsafe_allow_html=True)

    if not is_answered(2):
        st.markdown("Choose an option:")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Yes, I have pain", use_container_width=True, key="pain_yes"):
                st.session_state.pain_yesno = True
                on_patient_answer("Yes, I have pain today.", stage_id=2, extra_context=pain_context)
                st.rerun()
        with c2:
            if st.button("🙂 No pain today", use_container_width=True, key="pain_no"):
                st.session_state.pain_yesno = False
                on_patient_answer("No, I don't have any pain today.", stage_id=2, extra_context=pain_context)
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    render_free_input(stage_id=2, placeholder="Or describe your pain situation in your own words…",
                      extra_context=pain_context)

    if is_answered(2) and not can_followup(2):
        render_next_button("Next question →")

# ============================================================
# STAGE 3 — Body pain map (only if pain_yesno is True or unknown)
# ============================================================
elif stage == 3:
    location_context = (
        "The patient just indicated where they feel pain on a body map. "
        "React with empathy and ask ONE specific follow-up question about those pain locations "
        "(e.g. severity, duration, quality of pain)."
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Where do you feel pain?</div>', unsafe_allow_html=True)

    if not is_answered(3):
        st.markdown("Select all areas where you feel pain:")
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
                + "</div>", unsafe_allow_html=True)

        cA, cB = st.columns(2)
        with cA:
            if st.button("Clear all", key="clear_body"):
                st.session_state.selected_parts = set()
                st.rerun()
        with cB:
            if st.button("Send pain locations ➜", key="send_locations"):
                loc_txt = ", ".join(sorted(st.session_state.selected_parts)) if st.session_state.selected_parts else "not sure of exact location"
                on_patient_answer(f"Pain locations: {loc_txt}.", stage_id=3, extra_context=location_context)
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    render_free_input(stage_id=3, placeholder="Or describe where you feel pain in your own words…",
                      extra_context=location_context)

    if is_answered(3) and not can_followup(3):
        render_next_button("Next question →")

# ============================================================
# STAGE 4 — Symptom checklist
# ============================================================
elif stage == 4:
    symptom_context = (
        "The patient just submitted their symptom checklist. "
        "React with empathy to what they reported and ask ONE specific follow-up question "
        "about their most concerning or notable symptom."
    )

    symptom_options = [
        "Fatigue / low energy", "Nausea", "Vomiting", "Poor appetite",
        "Mouth sores", "Trouble swallowing", "Shortness of breath",
        "Fever / chills", "Constipation", "Diarrhea",
        "Sleep problems", "Anxiety / low mood",
    ]

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Any of these symptoms today?</div>', unsafe_allow_html=True)

    if not is_answered(4):
        st.markdown("Tap to toggle — select all that apply, then click Send:")
        for symptom in symptom_options:
            label = f"✓ {symptom}" if symptom in st.session_state.symptoms else symptom
            if st.button(label, key=f"symptom_{symptom}", use_container_width=True):
                if symptom in st.session_state.symptoms:
                    st.session_state.symptoms.remove(symptom)
                else:
                    st.session_state.symptoms.append(symptom)
                st.rerun()

        if st.button("Send symptoms ➜", use_container_width=True, key="send_symptoms"):
            sym_txt = "; ".join(st.session_state.symptoms) if st.session_state.symptoms else "no symptoms from the checklist"
            on_patient_answer(f"Symptoms today: {sym_txt}.", stage_id=4, extra_context=symptom_context)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    render_free_input(stage_id=4, placeholder="Or describe your symptoms in your own words…",
                      extra_context=symptom_context)

    if is_answered(4) and not can_followup(4):
        render_next_button("Finish check-in →")

# ============================================================
# STAGE 5 — Free chat + submit
# ============================================================
elif stage == 5:
    if st.session_state.submitted:
        # ── Summary table ─────────────────────────────────────────────────
        name      = st.session_state.get("patient_name", "—")
        feeling   = st.session_state.get("feeling_level", None)
        pain      = st.session_state.get("pain_yesno", None)
        locations = sorted(list(st.session_state.get("selected_parts", set())))
        symptoms  = st.session_state.get("symptoms", [])

        widget_msgs = {
            f"My feeling level today is {feeling}/10.",
            "Yes, I have pain today.",
            "No, I don't have any pain today.",
            "No symptoms from the checklist today.",
        }
        if locations:
            widget_msgs.add(f"Pain locations: {', '.join(locations)}.")
        if symptoms:
            widget_msgs.add(f"Symptoms today: {'; '.join(symptoms)}.")

        pain_str        = "Yes" if pain is True else ("No" if pain is False else "—")
        feeling_display = f"{feeling}/10" if feeling is not None else "—"
        sym_html  = "".join(f'<span class="tag">{s}</span>' for s in symptoms) if symptoms \
                    else "<span style='color:rgba(0,0,0,0.4)'>None reported</span>"
        loc_html  = "".join(f'<span class="tag">{l}</span>' for l in locations) if locations \
                    else "<span style='color:rgba(0,0,0,0.4)'>None / N/A</span>"

        patient_lines = [
            m["content"] for m in st.session_state.messages
            if m.get("role") == "patient" and m.get("content", "") not in widget_msgs
        ]

        if patient_lines and _openai_ready():
            patient_text = "\n".join(f"- {l}" for l in patient_lines)
            try:
                sr = openai_client.chat.completions.create(
                    model=_secret("openai_model", default="gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": (
                            "You are a clinical notes assistant. "
                            "Below are free-text messages typed or spoken by a cancer patient during a symptom check-in. "
                            "Extract ONLY medically relevant facts: pain details, severity, duration, "
                            "what helps or worsens it, mood, appetite, sleep, energy, or other health details. "
                            "Output a clean bullet-point list, one fact per line. "
                            "Remove greetings, filler, and repetition. "
                            "If nothing clinically relevant beyond the widgets, reply exactly: None"
                        )},
                        {"role": "user", "content": patient_text}
                    ],
                    max_tokens=300, temperature=0.2,
                )
                summary_text = (sr.choices[0].message.content or "").strip()
            except Exception:
                summary_text = "None"
        else:
            summary_text = "None"

        if summary_text and summary_text != "None":
            lines = [l.lstrip("\u2022-\u2013 ").strip() for l in summary_text.split("\n")
                     if l.strip() and l.strip() != "None"]
            conv_cell = "<ul style='margin:0;padding-left:18px;'>" + "".join(
                f"<li style='margin-bottom:5px;font-size:14px;color:#1a2540;line-height:1.5'>{l}</li>"
                for l in lines
            ) + "</ul>"
        else:
            conv_cell = "<span style='color:rgba(0,0,0,0.4)'>No additional details shared</span>"

        st.markdown("""
<style>
.summary-wrap{background:linear-gradient(135deg,#f0f7ff,#eaf3ff);border:1.5px solid rgba(31,122,255,0.15);border-radius:20px;padding:28px 24px 20px;margin-top:8px;box-shadow:0 4px 20px rgba(31,122,255,0.08);}
.summary-title{font-size:20px;font-weight:700;color:#1a2540;margin-bottom:4px;}
.summary-sub{font-size:13px;color:rgba(0,0,0,0.45);margin-bottom:20px;}
.summary-table{width:100%;border-collapse:collapse;font-size:15px;}
.summary-table tr{border-bottom:1px solid rgba(200,215,240,0.6);}
.summary-table tr:last-child{border-bottom:none;}
.summary-table td{padding:11px 8px;vertical-align:top;}
.summary-table td:first-child{font-weight:600;color:#4a6080;width:36%;white-space:nowrap;}
.summary-table td:last-child{color:#1a2540;}
.tag{display:inline-block;background:rgba(31,122,255,0.09);color:#1f5acc;border-radius:20px;padding:3px 10px;font-size:13px;margin:3px 4px 3px 0;}
.submitted-badge{display:inline-block;background:#22c55e;color:white;border-radius:20px;padding:4px 14px;font-size:13px;font-weight:700;margin-bottom:16px;}
</style>
""", unsafe_allow_html=True)

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
        st.markdown(
            '<div class="panel">'
            '<div class="panel-title">💬 Anything else to share?</div>'
            '<div class="small-note">Chat freely with the assistant, or click Submit when ready.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        # Free chat input for stage 5
        cols = st.columns([5, 1], vertical_alignment="bottom")
        with cols[0]:
            user_text_5 = st.chat_input("Anything else you'd like to mention…", key="chat_input_5")
        audio_value_5 = None
        with cols[1]:
            if hasattr(st, "audio_input"):
                audio_value_5 = st.audio_input(
                    "", key=f"mic_5_{st.session_state.mic_key_counter}",
                    label_visibility="collapsed"
                )
        if user_text_5:
            add_patient(user_text_5)
            with st.spinner("Assistant is thinking…"):
                reply = get_gpt_reply()
            add_doctor(reply)
            st.rerun()
        if audio_value_5 is not None:
            try:
                ab = audio_value_5.getvalue()
                ah = hashlib.sha1(ab).hexdigest()
            except Exception:
                ab = ah = None
            if ab and ah and ah != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = ah
                st.session_state.mic_key_counter += 1
                with st.spinner("Transcribing your voice…"):
                    transcribed = transcribe_audio(ab)
                if transcribed and not transcribed.startswith("(Transcription failed"):
                    st.info(f'Heard: "{transcribed}"')
                    add_patient(transcribed)
                    with st.spinner("Assistant is thinking…"):
                        reply = get_gpt_reply()
                    add_doctor(reply)
                    st.rerun()

        if st.button("✅ Submit Check-In", use_container_width=True):
            try:
                save_to_sheet()
                st.session_state.submitted = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save to Google Sheets: {e}")
