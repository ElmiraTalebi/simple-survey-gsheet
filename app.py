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
    if feeling  is not None: lines.append(f"- Feeling: {feeling}/10")
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
You are doing a brief daily check-in with: {name}.

TODAY'S DATA: {session_str}
PATIENT HISTORY: {memory_str}
{ctx}
STRICT OUTPUT RULES — violating any of these is unacceptable:
1. Output ONLY a single question. Nothing else.
2. NEVER start with "Thank you", "I'm sorry", "Great", "I see", "I understand", "Of course",
   "That's", "I'm glad", or ANY acknowledgement phrase whatsoever.
3. NEVER explain what you are going to do. Just do it.
4. You may use ONE short empathetic phrase (max 8 words) before the question IF truly needed,
   but only if it adds clinical value — otherwise skip it entirely.
5. Only discuss the patient's physical and emotional health.
   Off-topic? Say only: "Let's stay focused — [question]."
6. Never give medical advice. If asked: "Your care team will follow up."
"""

def _openai_ready():
    return openai_client is not None and openai_init_error is None

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
        msgs.append({"role": "assistant" if m.get("role")=="doctor" else "user", "content": m.get("content","")})
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
<style>
[data-testid="stAppViewContainer"]{ background:linear-gradient(135deg,#eef4ff,#f6fbff); }
.header{ font-size:24px; font-weight:700; margin:8px 0 14px 0; }
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
    margin-top:10px; padding:14px 16px 16px; border-radius:16px;
    background:rgba(255,255,255,0.7); border:1px solid rgba(200,210,230,0.6);
    backdrop-filter:blur(10px);
}
.panel-title{ font-weight:700; font-size:15px; margin-bottom:12px; color:#1a2540; }
.small-note{ color:rgba(0,0,0,0.4); font-size:12px; margin-top:4px; }
/* All answer-type buttons styled as uniform rounded pills */
.stButton>button{
    border-radius:20px !important; padding:0.42rem 0.9rem !important;
    font-size:14px !important; border:1px solid rgba(180,195,220,0.8) !important;
    background:white !important; color:#1a2540 !important;
    box-shadow:0 1px 4px rgba(0,0,0,0.06) !important;
    transition: all 0.15s ease !important; white-space: nowrap !important;
}
.stButton>button:hover{ background:#f0f6ff !important; border-color:#1f7aff !important; color:#1f7aff !important; }
/* Mic widget — compact pill matching button height */
[data-testid="stAudioInput"]{ margin:0 !important; padding:0 !important; }
[data-testid="stAudioInput"]>label{ display:none !important; }
[data-testid="stAudioInput"]>div{
    height:36px !important; min-height:36px !important;
    border-radius:20px !important; border:1px solid rgba(180,195,220,0.8) !important;
    display:flex !important; align-items:center !important; justify-content:center !important;
    background:white !important; box-shadow:0 1px 4px rgba(0,0,0,0.06) !important;
}
/* Text input styled as pill */
[data-testid="stTextInput"]>div>div>input{
    border-radius:20px !important; border:1px solid rgba(180,195,220,0.8) !important;
    padding:6px 14px !important; font-size:14px !important;
    background:white !important; height:36px !important;
    box-shadow:0 1px 4px rgba(0,0,0,0.06) !important;
}
[data-testid="stTextInput"]>label{ display:none !important; }
[data-testid="stChatInput"]{ position:sticky; bottom:0; }
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
def add_doctor(text): st.session_state.messages.append({"role":"doctor","content":text})
def add_patient(text): st.session_state.messages.append({"role":"patient","content":text})
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
    """Record patient answer; fire GPT follow-up if budget remains, else advance stage."""
    add_patient(text)
    mark_answered(stage_id)
    if can_followup(stage_id):
        record_followup(stage_id)
        with st.spinner("Assistant is thinking…"):
            reply = get_gpt_reply(extra_context=extra_context)
        add_doctor(reply)
    else:
        advance_stage()

def handle_voice(audio_value, stage_id: int, extra_context: str = "") -> bool:
    """Process mic recording. Returns True if a new clip was processed."""
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
        on_patient_answer(t, stage_id, extra_context)
        return True
    st.warning("Could not transcribe. Please try again.")
    return False

def render_chat_window():
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

def render_text_mic_row(stage_id: int, extra_context: str = "",
                        placeholder: str = "Or type your answer…",
                        send_label: str = "↑ Send"):
    """
    Compact single row: [text input ──────] [Send] [🎤]
    All styled as pills matching the clickable buttons.
    Returns True if a submission happened (caller should st.rerun()).
    """
    c_txt, c_send, c_mic = st.columns([6, 1, 1], gap="small")
    with c_txt:
        typed = st.text_input("", placeholder=placeholder,
                              key=f"txt_{stage_id}", label_visibility="collapsed")
    with c_send:
        send_clicked = st.button(send_label, key=f"txtsend_{stage_id}", use_container_width=True)
    with c_mic:
        audio_val = None
        if hasattr(st, "audio_input"):
            audio_val = st.audio_input("", key=f"mic_{stage_id}_{st.session_state.mic_key_counter}",
                                       label_visibility="collapsed")

    submitted = False
    if send_clicked and typed and typed.strip():
        on_patient_answer(typed.strip(), stage_id, extra_context)
        submitted = True
    if not submitted and handle_voice(audio_val, stage_id, extra_context):
        submitted = True
    if submitted:
        st.rerun()

def render_next_button(label="Next →"):
    if st.button(label, use_container_width=True, key=f"next_{st.session_state.stage}", type="primary"):
        advance_stage(); st.rerun()

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

# ── Warnings ────────────────────────────────────────────────
if openai_init_error: st.warning(f"LLM not ready: {openai_init_error}")
_init_sheets()
if sheets_init_error: st.warning(f"Sheets not ready: {sheets_init_error}")

st.markdown('<div class="header">🩺 Cancer Symptom Check-In</div>', unsafe_allow_html=True)

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
                context = (
                    f"Most recent visit: {last.get('timestamp','?')}. "
                    f"Feeling:{last.get('feeling_level','?')}/10, "
                    f"pain:{'yes' if last.get('pain') else 'no'}, "
                    f"locations:{', '.join(last.get('pain_locations',[])) or 'none'}, "
                    f"symptoms:{', '.join(last.get('symptoms',[])) or 'none'}. "
                    f"Greet {name_input.strip()} by name, briefly summarise their last visit in 1-2 sentences, "
                    f"then ask ONE follow-up question about how they have been since then. "
                    f"No filler phrases."
                )
                with st.spinner("Getting your assistant ready…"):
                    opening = get_gpt_reply(extra_context=context)
                add_doctor(opening)
                st.session_state.stage = 0
            else:
                context = (
                    f"This is {name_input.strip()}'s first check-in. "
                    f"Greet them by name and say you'll start with a few quick questions. One sentence max."
                )
                with st.spinner("Getting your assistant ready…"):
                    opening = get_gpt_reply(extra_context=context)
                if not opening or opening.startswith("("):
                    opening = f"Hi {name_input.strip()}! I'm your check-in assistant. Let's go through a few quick questions."
                add_doctor(opening)
                st.session_state.stage = 1
            st.rerun()
        else:
            st.warning("Please enter your name to continue.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

stage = st.session_state.stage

# ── Chat window — always visible ────────────────────────────
render_chat_window()

# ════════════════════════════════════════════════════════════
# STAGE 0 — History recap (only if patient has past visits)
# ONE panel. Text+mic row as the only input.
# GPT gets up to 1 follow-up. Then Next button appears.
# ════════════════════════════════════════════════════════════
if stage == 0:
    history_ctx = (
        "The patient is replying about how they have been since their last visit. "
        "Ask ONE focused follow-up question if clinically warranted. "
        "Do NOT start the structured questionnaire questions yet. No filler."
    )
    if not is_answered(0):
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">💬 How have you been since your last visit?</div>', unsafe_allow_html=True)
        render_text_mic_row(stage_id=0, extra_context=history_ctx, placeholder="Tell us how you've been…")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        if not can_followup(0):
            render_next_button("Start today's check-in →")
        else:
            st.markdown('<div class="small-note" style="margin-top:6px;">Reply above, or skip ahead.</div>', unsafe_allow_html=True)
            render_next_button("Skip to check-in →")

# ════════════════════════════════════════════════════════════
# STAGE 1 — Feeling scale
# ONE panel. Number buttons (0-10) on top row.
# Text+mic row below as the free-input alternative.
# Follow-ups appear in chat AFTER the patient answers.
# ════════════════════════════════════════════════════════════
elif stage == 1:
    feeling_ctx = (
        "The patient just answered the 0-10 feeling scale. "
        "Ask ONE specific follow-up about what is driving that score. "
        "No filler at all. Start directly with the question or max 5 words of context."
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">How are you feeling today?</div>', unsafe_allow_html=True)

    if not is_answered(1):
        st.markdown('<div class="small-note">Tap a number (0 = worst · 10 = best) or type / speak below</div>', unsafe_allow_html=True)

        # ── Number buttons row ──────────────────────────────
        scale_cols = st.columns(11, gap="small")
        for i in range(11):
            with scale_cols[i]:
                label = f"✓{i}" if st.session_state.feeling_level == i else str(i)
                if st.button(label, key=f"feel_{i}", use_container_width=True):
                    st.session_state.feeling_level = i
                    st.rerun()

        # Show selected value + Send button when a number is chosen
        if st.session_state.feeling_level is not None:
            c_sel, c_send = st.columns([5, 2], gap="small")
            with c_sel:
                st.markdown(f"<div style='padding:6px 2px;font-size:14px;'>Selected: <b>{st.session_state.feeling_level}/10</b></div>",
                            unsafe_allow_html=True)
            with c_send:
                if st.button("Send number ➜", key="send_feeling", use_container_width=True):
                    on_patient_answer(f"My feeling level today is {st.session_state.feeling_level}/10.", 1, feeling_ctx)
                    st.rerun()

        # ── Free text + mic row (same panel, below buttons) ─
        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        render_text_mic_row(stage_id=1, extra_context=feeling_ctx,
                            placeholder="Or describe how you feel in your own words…")

    st.markdown("</div>", unsafe_allow_html=True)
    if is_answered(1) and not can_followup(1):
        render_next_button("Next question →")

# ════════════════════════════════════════════════════════════
# STAGE 2 — Pain yes/no
# ONE panel. [Yes] [No] + text+mic all on the same row.
# ════════════════════════════════════════════════════════════
elif stage == 2:
    pain_ctx = (
        "The patient just answered whether they have pain. "
        "Ask ONE focused follow-up question about their pain. No filler."
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Do you have any pain today?</div>', unsafe_allow_html=True)

    if not is_answered(2):
        st.markdown('<div class="small-note">Choose an option, or describe in your own words</div>', unsafe_allow_html=True)
        # [Yes, pain] [No pain] [text──────────] [Send] [🎤]  all on one row
        c1, c2, c_txt, c_send, c_mic = st.columns([1.6, 1.6, 4, 1, 1], gap="small")
        with c1:
            if st.button("✅ Yes, pain", use_container_width=True, key="pain_yes"):
                st.session_state.pain_yesno = True
                on_patient_answer("Yes, I have pain today.", 2, pain_ctx)
                st.rerun()
        with c2:
            if st.button("🙂 No pain", use_container_width=True, key="pain_no"):
                st.session_state.pain_yesno = False
                on_patient_answer("No, I don't have any pain today.", 2, pain_ctx)
                st.rerun()
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

    st.markdown("</div>", unsafe_allow_html=True)
    if is_answered(2) and not can_followup(2):
        render_next_button("Next question →")

# ════════════════════════════════════════════════════════════
# STAGE 3 — Body pain map
# ONE panel. SVG on left, toggle buttons + text/mic on right.
# ════════════════════════════════════════════════════════════
elif stage == 3:
    location_ctx = (
        "The patient just indicated pain locations on a body map. "
        "Ask ONE follow-up about those locations (severity, duration, character). No filler."
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Where do you feel pain?</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Select areas on the map, or describe in your own words below</div>', unsafe_allow_html=True)

    if not is_answered(3):
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

        # Text+mic row + Send button — all in one row below the map
        c_txt3, c_send3a, c_mic3, c_send3b = st.columns([5, 1, 1, 2], gap="small")
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

    st.markdown("</div>", unsafe_allow_html=True)
    if is_answered(3) and not can_followup(3):
        render_next_button("Next question →")

# ════════════════════════════════════════════════════════════
# STAGE 4 — Symptom checklist
# ONE panel. 2-column grid of symptom toggle buttons.
# Text+mic + Send all on one row at the bottom of the panel.
# ════════════════════════════════════════════════════════════
elif stage == 4:
    symptom_ctx = (
        "The patient just submitted their symptom checklist. "
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
        st.markdown('<div class="small-note">Tap to select all that apply, then click Send — or describe in your own words below</div>', unsafe_allow_html=True)

        # Symptom toggle buttons — 2-column grid
        sc = st.columns(2, gap="small")
        for idx, symptom in enumerate(symptom_options):
            with sc[idx % 2]:
                label = f"✓ {symptom}" if symptom in st.session_state.symptoms else symptom
                if st.button(label, key=f"sym_{idx}", use_container_width=True):
                    if symptom in st.session_state.symptoms: st.session_state.symptoms.remove(symptom)
                    else: st.session_state.symptoms.append(symptom)
                    st.rerun()

        # Bottom row: [text──────────────] [↑] [🎤] [Send symptoms ➜]
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

    st.markdown("</div>", unsafe_allow_html=True)
    if is_answered(4) and not can_followup(4):
        render_next_button("Finish check-in →")

# ════════════════════════════════════════════════════════════
# STAGE 5 — Free chat + submit → summary table
# ════════════════════════════════════════════════════════════
elif stage == 5:
    if st.session_state.submitted:
        # ── Summary table ──────────────────────────────────
        name      = st.session_state.get("patient_name","—")
        feeling   = st.session_state.get("feeling_level",None)
        pain      = st.session_state.get("pain_yesno",None)
        locations = sorted(list(st.session_state.get("selected_parts",set())))
        symptoms  = st.session_state.get("symptoms",[])

        widget_msgs = {
            f"My feeling level today is {feeling}/10.",
            "Yes, I have pain today.", "No, I don't have any pain today.",
            "No symptoms from the checklist today.",
        }
        if locations: widget_msgs.add(f"Pain locations: {', '.join(locations)}.")
        if symptoms:  widget_msgs.add(f"Symptoms today: {'; '.join(symptoms)}.")

        feeling_display = f"{feeling}/10" if feeling is not None else "—"
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
                            "Clinical notes assistant. Extract ONLY medically relevant facts from the patient's "
                            "free-text messages: pain details, severity, duration, triggers, mood, appetite, sleep, energy. "
                            "Output a clean bullet list, one fact per line. No greetings or filler. "
                            "If nothing clinically relevant, reply: None"
                        )},
                        {"role":"user","content":"\n".join(f"- {l}" for l in patient_lines)}
                    ], max_tokens=300, temperature=0.2,
                )
                summary_text = (sr.choices[0].message.content or "").strip()
            except: pass

        if summary_text and summary_text != "None":
            items = [l.lstrip("•-– ").strip() for l in summary_text.split("\n") if l.strip() and l.strip()!="None"]
            conv_cell = "<ul style='margin:0;padding-left:18px;'>"+"".join(
                f"<li style='margin-bottom:4px;font-size:14px;color:#1a2540'>{l}</li>" for l in items)+"</ul>"
        else:
            conv_cell = "<span style='opacity:.4'>No additional details shared</span>"

        st.markdown("""
<style>
.summary-wrap{background:linear-gradient(135deg,#f0f7ff,#eaf3ff);border:1.5px solid rgba(31,122,255,0.15);border-radius:20px;padding:26px 22px 18px;margin-top:8px;box-shadow:0 4px 20px rgba(31,122,255,0.08);}
.summary-title{font-size:19px;font-weight:700;color:#1a2540;margin-bottom:3px;}
.summary-sub{font-size:12px;color:rgba(0,0,0,0.4);margin-bottom:18px;}
.summary-table{width:100%;border-collapse:collapse;font-size:14px;}
.summary-table tr{border-bottom:1px solid rgba(200,215,240,0.6);}
.summary-table tr:last-child{border-bottom:none;}
.summary-table td{padding:10px 8px;vertical-align:top;}
.summary-table td:first-child{font-weight:600;color:#4a6080;width:36%;}
.tag{display:inline-block;background:rgba(31,122,255,0.09);color:#1f5acc;border-radius:20px;padding:2px 10px;font-size:13px;margin:2px 3px 2px 0;}
.submitted-badge{display:inline-block;background:#22c55e;color:white;border-radius:20px;padding:3px 13px;font-size:13px;font-weight:700;margin-bottom:14px;}
</style>""", unsafe_allow_html=True)

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
            add_patient(user_text_5)
            with st.spinner("Assistant is thinking…"):
                reply = get_gpt_reply()
            add_doctor(reply); st.rerun()
        if audio_5 is not None:
            try: ab=audio_5.getvalue(); ah=hashlib.sha1(ab).hexdigest()
            except: ab=ah=None
            if ab and ah and ah != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash=ah; st.session_state.mic_key_counter+=1
                with st.spinner("Transcribing…"): t=transcribe_audio(ab)
                if t and not t.startswith("(Transcription failed"):
                    st.info(f'Heard: "{t}"')
                    add_patient(t)
                    with st.spinner("Assistant is thinking…"): reply=get_gpt_reply()
                    add_doctor(reply); st.rerun()
        if st.button("✅ Submit Check-In", use_container_width=True):
            try: save_to_sheet(); st.session_state.submitted=True; st.rerun()
            except Exception as e: st.error(f"Failed to save: {e}")
