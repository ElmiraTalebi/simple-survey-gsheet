import hashlib
import io
import json
from datetime import datetime
from typing import Dict, List, Optional, Set

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI

st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ============================================================
# MODERN UI STYLE
# ============================================================

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    max-width:900px;
}

h1{
    font-weight:600;
    margin-bottom:0.5rem;
}

h2,h3{
    font-weight:500;
}

.stButton>button{
    width:100%;
    border-radius:10px;
    padding:0.6rem 0.8rem;
    font-size:15px;
    border:1px solid #e1e5ee;
    background:#f8f9fc;
}

.stButton>button:hover{
    border:1px solid #7aa6ff;
    background:#eef3ff;
}

.symptom-btn button{
    border-radius:20px;
}

.card{
    padding:20px;
    border-radius:12px;
    border:1px solid #e6e9f2;
    background:white;
    margin-bottom:15px;
}

.doctor-box{
    padding:18px;
    border-radius:12px;
    background:#f1f6ff;
    border:1px solid #cfe0ff;
    font-size:15px;
}

.section-title{
    font-size:18px;
    font-weight:600;
    margin-bottom:10px;
}

.success-box{
    padding:18px;
    border-radius:12px;
    background:#ecfff1;
    border:1px solid #a8e6b9;
    font-size:16px;
    text-align:center;
}

.followup-box{
    padding:18px;
    border-radius:12px;
    background:#fff8f0;
    border:1px solid #ffd6a0;
    font-size:15px;
    margin-bottom:12px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# Secrets helpers
# ============================================================

def _secret(*keys, default=None):
    for k in keys:
        if k in st.secrets:
            return st.secrets[k]
    return default

def _require_secret(*keys):
    v = _secret(*keys)
    if v is None:
        raise KeyError(f"Missing secret. Tried: {', '.join(keys)}")
    return v


# ============================================================
# OpenAI
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
    openai_init_error = "OpenAI API key not found."


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
            _require_secret("gcp_service_account"),
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        book = gspread.authorize(creds).open_by_key(_require_secret("gsheet_id"))
        try:
            ws = book.worksheet("Form")
        except Exception:
            ws = book.add_worksheet(title="Form", rows=2000, cols=20)
            ws.append_row(["timestamp", "name", "json"])
        sheet = ws
    except Exception as e:
        sheets_init_error = str(e)

def load_past_checkins(name: str) -> List[Dict]:
    _init_sheets()
    if sheet is None:
        return []
    try:
        past: List[Dict] = []
        rows = sheet.get_all_values()
        for row in rows[1:]:
            if len(row) >= 3 and row[1].strip().lower() == name.strip().lower():
                try:
                    d = json.loads(row[2])
                    d["timestamp"] = row[0]
                    past.append(d)
                except Exception:
                    continue
        return past[-5:]
    except Exception:
        return []

def save_to_sheet(payload: Dict):
    _init_sheets()
    if sheet is None:
        raise RuntimeError(f"Sheets unavailable: {sheets_init_error}")
    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        payload.get("name", "Unknown"),
        json.dumps(payload),
    ])


# ============================================================
# Body map SVG
# ============================================================

def body_svg(colors: Dict[str, str]) -> str:

    def c(p): return colors.get(p, "#cfd8e6")
    stroke = "#6b7a90"

    return f"""
<svg width="260" height="410" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">

  <circle cx="160" cy="70" r="38" fill="{c('Head')}" stroke="{stroke}" stroke-width="2"/>
  <rect x="110" y="120" width="100" height="70" rx="24" fill="{c('Chest')}" stroke="{stroke}" stroke-width="2"/>
  <rect x="115" y="195" width="90" height="70" rx="22" fill="{c('Abdomen')}" stroke="{stroke}" stroke-width="2"/>

  <rect x="60" y="140" width="40" height="120" rx="20" fill="{c('Left Arm')}" stroke="{stroke}" stroke-width="2"/>
  <rect x="220" y="140" width="40" height="120" rx="20" fill="{c('Right Arm')}" stroke="{stroke}" stroke-width="2"/>

  <rect x="130" y="270" width="35" height="150" rx="20" fill="{c('Left Leg')}" stroke="{stroke}" stroke-width="2"/>
  <rect x="165" y="270" width="35" height="150" rx="20" fill="{c('Right Leg')}" stroke="{stroke}" stroke-width="2"/>

</svg>
"""


# ============================================================
# Session state
# ============================================================

DEFAULTS = {
    "stage": -1,
    "name": "",
    "past_checkins": [],
    "last_summary": None,
    "last_pain_severity": {},
    "feeling_level": None,
    "pain_yesno": None,
    "selected_parts": set(),
    "pain_severity": {},
    "pain_reason": {},
    "symptoms": set(),
    "symptom_answers": {},          # {symptom_name: answer_text} for inline follow-ups
    "submitted": False,
    # Follow-up stage state
    "followup_questions": [],       # list of question strings from GPT
    "followup_answers": {},         # {question_index: answer_text}
    "followup_generated": False,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# Body map logic
# ============================================================

REGIONS = ["Head", "Chest", "Abdomen", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]

GREEN = "#6fd08c"
ORANGE = "#f5a623"
RED = "#e74c3c"

# Symptoms that always trigger a follow-up question when newly reported
HIGH_CONCERN_SYMPTOMS = {"Difficulty swallowing", "Mouth sores", "Hoarseness", "Nausea"}


def region_color_state(region: str) -> str:

    last = st.session_state.last_pain_severity
    selected = region in st.session_state.selected_parts

    if not selected:
        if region in last:
            return ORANGE
        return GREEN

    # Once selected, always RED — persistent or new pain is clinically significant
    return RED


def current_svg_colors():
    return {r: region_color_state(r) for r in REGIONS}


# ============================================================
# Worsening detection
# ============================================================

def detect_worsening(payload: Dict, last_summary: Optional[Dict]) -> Dict:
    """
    Returns a dict describing what got worse, or empty dict if nothing significant.
    High-concern symptom follow-ups are handled inline in Stage 4.
    """
    worse = {}

    if not last_summary:
        return worse

    # Pain severity worsened
    last_sev = last_summary.get("pain_severity", {})
    worsened_pain = {}
    for region, sev in payload.get("pain_severity", {}).items():
        prev = last_sev.get(region, 0)
        if sev >= prev + 2:
            worsened_pain[region] = {"from": prev, "to": sev}
    if worsened_pain:
        worse["worsened_pain"] = worsened_pain

    # New pain locations
    last_locs = set(last_summary.get("pain_locations", []))
    cur_locs = set(payload.get("pain_locations", []))
    new_locs = cur_locs - last_locs
    if new_locs:
        worse["new_pain_locations"] = list(new_locs)

    # Feeling level dropped
    last_feeling = last_summary.get("feeling_level")
    cur_feeling = payload.get("feeling_level")
    if last_feeling is not None and cur_feeling is not None:
        if cur_feeling <= last_feeling - 2:
            worse["feeling_dropped"] = {"from": last_feeling, "to": cur_feeling}

    # New symptoms — general (non-high-concern) new symptoms only
    # High-concern symptoms are handled inline in Stage 4, not via stage 4.5
    last_symptoms = set(last_summary.get("symptoms", []))
    cur_symptoms = set(payload.get("symptoms", []))
    new_symptoms = cur_symptoms - last_symptoms
    new_general = [s for s in new_symptoms if s not in HIGH_CONCERN_SYMPTOMS]
    if new_general:
        worse["new_symptoms"] = new_general

    return worse


# ============================================================
# GPT follow-up question generation
# ============================================================

def generate_followup_questions(payload: Dict, worse: Dict, last_summary: Optional[Dict]) -> List[str]:
    """
    Generates at most 1 follow-up question, only when clinically necessary
    and only when the patient has NOT already explained the issue.

    Priority order (first match wins):
      1. Severe pain (>=8) with no explanation
      2. Pain jumped 3+ points with no explanation
      3. New pain location (severity >=6) with no explanation
      4. New concerning symptom (Difficulty swallowing, Mouth sores) with no prior record
    """
    if openai_client is None:
        return []

    pain_reason   = payload.get("pain_reason", {})
    pain_severity = payload.get("pain_severity", {})

    critical_issue = None

    # 1. Severe pain with no reason
    for region, sev in pain_severity.items():
        if sev >= 8 and not pain_reason.get(region, "").strip():
            critical_issue = (
                f"The patient has severe pain in their {region} ({sev}/10) "
                f"and has not explained why."
            )
            break

    # 2. Pain worsened significantly with no reason
    if not critical_issue and "worsened_pain" in worse:
        for region, change in worse["worsened_pain"].items():
            jump = change["to"] - change["from"]
            if jump >= 3 and not pain_reason.get(region, "").strip():
                critical_issue = (
                    f"The patient's pain in their {region} jumped from "
                    f"{change['from']}/10 to {change['to']}/10 with no explanation."
                )
                break

    # 3. New pain location, moderate-to-severe, no reason
    if not critical_issue and "new_pain_locations" in worse:
        for region in worse["new_pain_locations"]:
            sev = pain_severity.get(region, 0)
            if sev >= 6 and not pain_reason.get(region, "").strip():
                critical_issue = (
                    f"The patient has new pain in their {region} at {sev}/10 "
                    f"that wasn't present in the previous visit."
                )
                break

    # 4. (High-concern symptoms are handled inline in Stage 4 — not triggered here)

    # Nothing triggered or everything already explained
    if not critical_issue:
        return []

    prompt = (
        "You are a compassionate oncology nurse conducting a brief patient check-in. "
        f"The following needs clarification: {critical_issue} "
        "Ask the patient ONE short, kind, open-ended question about this — "
        "it must be easy to answer out loud in one or two sentences. "
        "Do NOT use medical jargon. Do NOT ask multiple questions. "
        "Return ONLY a JSON array with exactly one question string. "
        'Example: ["Can you describe what the pain feels like?"]'
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        questions = json.loads(raw)
        if isinstance(questions, list) and questions:
            return [str(questions[0])]
    except Exception:
        pass

    return []




# ============================================================
# Voice input — st.audio_input + OpenAI Whisper
# ============================================================

def _transcribe_audio(audio_bytes: bytes) -> str:
    """Send raw audio bytes to Whisper and return the transcript."""
    if openai_client is None:
        return ""
    try:
        buf = io.BytesIO(audio_bytes)
        buf.name = "audio.wav"
        result = openai_client.audio.transcriptions.create(
            model="whisper-1", file=buf
        )
        return result.text.strip()
    except Exception:
        return ""


def voice_input_widget(answer_key: str, label: str = "🎤 Or speak your answer"):
    """
    Renders Streamlit's native audio recorder.
    On new audio: transcribes via Whisper → stores in _transcript_{answer_key}.
    The caller reads st.session_state.get(f"_transcript_{answer_key}", "")
    and passes it as value= to the associated text_input / text_area.
    Uses SHA-1 dedup so the same clip is never transcribed twice.
    """
    transcript_key = f"_transcript_{answer_key}"
    hash_key       = f"_audiohash_{answer_key}"

    if hash_key not in st.session_state:
        st.session_state[hash_key] = None

    audio = st.audio_input(label, key=f"_rec_{answer_key}")
    if audio is None:
        return

    try:
        ab = audio.getvalue()
    except Exception:
        return
    if not ab:
        return

    ah = hashlib.sha1(ab).hexdigest()
    if ah == st.session_state[hash_key]:
        return  # already processed this clip

    st.session_state[hash_key] = ah
    with st.spinner("Transcribing…"):
        text = _transcribe_audio(ab)

    if text:
        st.session_state[transcript_key] = text
        st.rerun()
    else:
        st.warning("Could not transcribe — please try again or type your answer.")

# ============================================================
# UI
# ============================================================

st.title("🩺 Cancer Symptom Check-In")

_init_sheets()

# ------------------------------------------------------------
# Stage -1 : Name
# ------------------------------------------------------------

if st.session_state.stage == -1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    name = st.text_input("Enter your name")

    if st.button("Start Check-In"):

        if name.strip():

            st.session_state.name = name.strip()

            past = load_past_checkins(name)
            st.session_state.past_checkins = past

            if past:
                last = past[-1]
                st.session_state.last_summary = last
                st.session_state.last_pain_severity = last.get("pain_severity", {})
            else:
                st.session_state.last_summary = None
                st.session_state.last_pain_severity = {}

            st.session_state.stage = 0
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ------------------------------------------------------------
# Stage 0 : Doctor recap
# ------------------------------------------------------------

if st.session_state.stage == 0:

    last = st.session_state.last_summary

    if last:

        ts = last.get("timestamp", "")
        pain_locs = last.get("pain_locations", [])
        symptoms = last.get("symptoms", [])
        feeling = last.get("feeling_level")

        message = f"Hi {st.session_state.name}, I reviewed your last check-in"

        if ts:
            message += f" from {ts.split()[0]}"

        message += ". "

        if pain_locs:
            message += f"You mentioned pain in your {', '.join(pain_locs)}. "

        if symptoms:
            message += f"You also reported {', '.join(symptoms)}. "

        if feeling is not None:
            message += f"Your overall feeling level was {feeling}/10. "

        message += "Before we continue, has anything changed since then?"

        st.markdown(f'<div class="doctor-box">👩‍⚕️ {message}</div>', unsafe_allow_html=True)

    else:

        welcome = (
            f"Welcome, {st.session_state.name}! 👋 It looks like this is your first check-in with us. "
            f"We'll ask you a few short questions about how you're feeling today — "
            f"it should only take a minute or two."
        )
        st.markdown(f'<div class="doctor-box">👩‍⚕️ {welcome}</div>', unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns(2)

    if last:

        with c1:
            if st.button("Same as yesterday"):

                payload = {
                    "name": st.session_state.name,
                    "note": "Same as yesterday"
                }

                save_to_sheet(payload)

                st.session_state.stage = 5
                st.rerun()

        with c2:
            if st.button("Something changed"):
                st.session_state.stage = 1
                st.rerun()

    else:

        with c1:
            if st.button("Start Check-In"):
                st.session_state.stage = 1
                st.rerun()


# ------------------------------------------------------------
# Stage 1
# ------------------------------------------------------------

elif st.session_state.stage == 1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">How are you feeling today?</div>', unsafe_allow_html=True)

    last_feeling = st.session_state.last_summary.get("feeling_level", None) if st.session_state.last_summary else None
    default_feeling = int(last_feeling) if last_feeling is not None else 7
    feeling = st.number_input("Feeling (0-10)", 0, 10, value=default_feeling)

    feeling_improved = (last_feeling is not None and feeling > int(last_feeling))
    feeling_worse = (last_feeling is not None and feeling < int(last_feeling))

    if feeling_improved:
        st.markdown(
            f'<div class="doctor-box">\U0001f60a That\'s great to hear! Last time you were at {int(last_feeling)}/10 '
            f'and now you\'re at {feeling}/10. Would you like to finish here, or continue to log any symptoms?</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("I'm feeling better — submit"):
                st.session_state.feeling_level = feeling
                payload = {
                    "name": st.session_state.name,
                    "feeling_level": feeling,
                    "note": "Patient reported feeling better than last visit. No further symptoms logged.",
                }
                save_to_sheet(payload)
                st.session_state.stage = 5
                st.rerun()
        with col_b:
            if st.button("Continue to log symptoms"):
                st.session_state.feeling_level = feeling
                st.session_state.stage = 2
                st.rerun()

    elif feeling_worse:
        st.markdown(
            f'<div class="doctor-box">\U0001f614 I\'m sorry to hear that. Last time you were at {int(last_feeling)}/10 '
            f'and today you\'re at {feeling}/10. Let\'s make sure we capture what\'s going on. '
            f'Do you have any pain today?</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("No pain"):
                st.session_state.feeling_level = feeling
                st.session_state.pain_yesno = False
                st.session_state.stage = 4
                st.rerun()
        with col_b:
            if st.button("Yes, I have pain"):
                st.session_state.feeling_level = feeling
                st.session_state.pain_yesno = True
                st.session_state.stage = 3
                st.rerun()

    else:
        if st.button("Next"):
            st.session_state.feeling_level = feeling
            st.session_state.stage = 2
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Stage 2
# ------------------------------------------------------------

elif st.session_state.stage == 2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Do you have pain today?</div>', unsafe_allow_html=True)

    pain = st.radio("", ["No", "Yes"])

    if st.button("Next"):

        st.session_state.pain_yesno = (pain == "Yes")

        if pain == "Yes":
            st.session_state.stage = 3
        else:
            st.session_state.stage = 4

        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Stage 3
# ------------------------------------------------------------

elif st.session_state.stage == 3:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Where do you feel pain?</div>', unsafe_allow_html=True)

    # Color legend
    st.markdown("""
    <div style="display:flex; gap:18px; margin-bottom:14px; font-size:14px; flex-wrap:wrap;">
        <span>🟢 <b>No pain</b> (not reported before)</span>
        <span>🟠 <b>Known pain</b> (reported in last visit)</span>
        <span>🔴 <b>New or worsened pain</b> (significantly worse or new)</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown(body_svg(current_svg_colors()), unsafe_allow_html=True)

    with col2:

        last = st.session_state.last_pain_severity

        for r in REGIONS:

            icon = "🟢"

            col = region_color_state(r)

            if col == ORANGE:
                icon = "🟠"
            if col == RED:
                icon = "🔴"

            if st.button(f"{icon} {r}", key=f"btn_{r}"):

                if r in st.session_state.selected_parts:
                    st.session_state.selected_parts.remove(r)
                else:
                    st.session_state.selected_parts.add(r)

                st.rerun()

            if r in st.session_state.selected_parts:

                last_val = last.get(r, 0)

                sev = st.number_input(
                    f"{r} severity",
                    0,
                    10,
                    value=last_val,
                    key=f"sev_{r}"
                )

                st.session_state.pain_severity[r] = sev

                # Ask GPT-generated follow-up only when severity > 5 or jumped 2+ from last visit
                needs_followup = sev > 5 or (last_val > 0 and sev >= last_val + 2)
                if needs_followup:
                    q_key = f"gpt_question_{r}"
                    if q_key not in st.session_state:
                        st.session_state[q_key] = None

                    # Generate question once per region if not yet done
                    if st.session_state[q_key] is None and openai_client is not None:
                        try:
                            context = (
                                f"A cancer patient has reported pain in their {r} "
                                f"with severity {sev}/10."
                            )
                            if last_val > 0:
                                context += f" Last visit it was {last_val}/10."
                            prompt = (
                                f"You are a compassionate oncology nurse. {context} "
                                f"Ask ONE short, empathetic, open-ended question to understand "
                                f"this pain better. Return ONLY the question string, no extra text."
                            )
                            resp = openai_client.chat.completions.create(
                                model="gpt-4.1-mini",
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=80,
                                temperature=0.5,
                            )
                            st.session_state[q_key] = resp.choices[0].message.content.strip().strip('"')
                        except Exception:
                            st.session_state[q_key] = "Can you describe what makes this pain better or worse?"

                    question   = st.session_state.get(q_key) or "Can you describe what makes this pain better or worse?"
                    _tr_reason = st.session_state.get(f"_transcript_reason_{r}", "")
                    _saved     = st.session_state.pain_reason.get(r, "")

                    if _saved:
                        # Already answered — show a compact summary, no input needed
                        st.markdown(
                            f'<div style="background:#f0fff4; border:1px solid #a8e6b9; '
                            f'border-radius:8px; padding:8px 12px; font-size:14px; '
                            f'color:#1a5c2a; margin-top:4px;">'
                            f'✅ <em>{question}</em><br><strong>{_saved}</strong></div>',
                            unsafe_allow_html=True,
                        )
                        # Small "edit" link so patient can re-open if needed
                        if st.button("✏️ Edit answer", key=f"edit_{r}"):
                            st.session_state.pain_reason.pop(r, None)
                            st.session_state.pop(f"_transcript_reason_{r}", None)
                            st.rerun()
                    else:
                        # widget_key is OWNED by st.text_input — never write to it directly
                        widget_key = f"_draft_reason_{r}"
                        if widget_key not in st.session_state:
                            st.session_state[widget_key] = _tr_reason or ""

                        # Voice landed: pre-fill widget before it renders
                        if _tr_reason and not st.session_state[widget_key]:
                            st.session_state[widget_key] = _tr_reason

                        def _save_reason(rr=r, wk=widget_key):
                            val = st.session_state.get(wk, "").strip()
                            if val:
                                st.session_state.pain_reason[rr] = val

                        st.text_input(question, key=widget_key, on_change=_save_reason)
                        voice_input_widget(f"reason_{r}", label="🎤 Speak your answer")

                        # Read value AFTER widget renders (never set it after this point)
                        current_val = st.session_state.get(widget_key, "").strip()

                        # Voice auto-save: if transcript has landed and matches widget, commit
                        if _tr_reason and current_val == _tr_reason:
                            st.session_state.pain_reason[r] = current_val
                            del st.session_state[widget_key]
                            st.rerun()

        # Other / custom pain location
        st.markdown("---")
        other_selected = "Other" in st.session_state.selected_parts
        other_icon = "🔴" if other_selected else "🟢"

        if st.button(f"{other_icon} Other", key="btn_Other"):
            if other_selected:
                st.session_state.selected_parts.remove("Other")
                st.session_state.pain_severity.pop("Other", None)
                st.session_state.pain_reason.pop("Other", None)
            else:
                st.session_state.selected_parts.add("Other")
            st.rerun()

        if other_selected:
            other_desc = st.text_input(
                "Describe the location",
                placeholder="e.g. lower back, behind the ear...",
                key="other_location_desc"
            )
            if other_desc:
                st.session_state.pain_reason["Other"] = other_desc

            other_sev = st.number_input(
                "Other severity",
                0, 10,
                value=0,
                key="sev_Other"
            )
            st.session_state.pain_severity["Other"] = other_sev

    if st.button("Next"):
        st.session_state.stage = 4
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Stage 4
# ------------------------------------------------------------

elif st.session_state.stage == 4:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Symptoms today</div>', unsafe_allow_html=True)

    symptom_options = [
        "Fatigue",
        "Nausea",
        "Dry mouth",
        "Difficulty swallowing",
        "Hoarseness",
        "Mouth sores",
        "Skin irritation",
        "Loss of taste",
    ]

    # Symptoms reported in the last session (for orange highlight)
    last_symptoms_set = set(
        st.session_state.last_summary.get("symptoms", [])
        if st.session_state.last_summary else []
    )

    cols = st.columns(2)

    for i, sym in enumerate(symptom_options):

        selected = sym in st.session_state.symptoms
        was_previous = sym in last_symptoms_set
        is_high_concern = sym in HIGH_CONCERN_SYMPTOMS

        if selected:
            label = f"🔴 {sym}"
        elif was_previous:
            label = f"🟠 {sym}"
        else:
            label = f"🟢 {sym}"

        with cols[i % 2]:

            if st.button(label, key=f"sym_btn_{sym}"):
                if selected:
                    st.session_state.symptoms.remove(sym)
                    # Clear follow-up state when deselecting
                    st.session_state.symptom_answers.pop(sym, None)
                    st.session_state.pop(f"gpt_sym_question_{sym}", None)
                    st.session_state.pop(f"_transcript_sym_{sym}", None)
                else:
                    st.session_state.symptoms.add(sym)
                st.rerun()

            # Show caption for previously reported but not yet re-selected
            if was_previous and not selected:
                st.caption("⬆️ Reported last visit — tap to confirm again")

            # --- Inline follow-up for newly selected high-concern symptoms ---
            if selected and is_high_concern and not was_previous:

                q_key = f"gpt_sym_question_{sym}"
                if q_key not in st.session_state:
                    st.session_state[q_key] = None

                # Generate GPT question once, only when first needed
                if st.session_state[q_key] is None and openai_client is not None:
                    try:
                        prompt = (
                            f"You are a compassionate oncology nurse doing a brief patient check-in. "
                            f"The patient has just reported a new symptom: {sym}. "
                            f"Ask ONE short, empathetic, open-ended question to understand it better. "
                            f"Keep it conversational and easy to answer in one or two sentences. "
                            f"Do NOT use medical jargon. Return ONLY the question string, no extra text."
                        )
                        resp = openai_client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=80,
                            temperature=0.5,
                        )
                        st.session_state[q_key] = resp.choices[0].message.content.strip().strip('"')
                    except Exception:
                        st.session_state[q_key] = f"Can you tell me a little more about the {sym.lower()}?"

                question = st.session_state.get(q_key) or f"Can you tell me a little more about the {sym.lower()}?"
                _tr_sym  = st.session_state.get(f"_transcript_sym_{sym}", "")
                _saved   = st.session_state.symptom_answers.get(sym, "")

                if _saved:
                    # Already answered — show compact green summary with edit option
                    st.markdown(
                        f'<div style="background:#f0fff4; border:1px solid #a8e6b9; '
                        f'border-radius:8px; padding:8px 12px; font-size:14px; '
                        f'color:#1a5c2a; margin-top:4px;">'
                        f'✅ <em>{question}</em><br><strong>{_saved}</strong></div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("✏️ Edit answer", key=f"edit_sym_{sym}"):
                        st.session_state.symptom_answers.pop(sym, None)
                        st.session_state.pop(f"_transcript_sym_{sym}", None)
                        st.rerun()
                else:
                    widget_key = f"_draft_sym_{sym}"
                    if widget_key not in st.session_state:
                        st.session_state[widget_key] = _tr_sym or ""

                    if _tr_sym and not st.session_state[widget_key]:
                        st.session_state[widget_key] = _tr_sym

                    def _save_sym(ss=sym, wk=widget_key):
                        val = st.session_state.get(wk, "").strip()
                        if val:
                            st.session_state.symptom_answers[ss] = val

                    st.text_input(question, key=widget_key, on_change=_save_sym)
                    voice_input_widget(f"sym_{sym}", label="🎤 Speak your answer")

                    current_val = st.session_state.get(widget_key, "").strip()

                    if _tr_sym and current_val == _tr_sym:
                        st.session_state.symptom_answers[sym] = current_val
                        del st.session_state[widget_key]
                        st.rerun()

    st.markdown("---")

    if st.button("Submit Check-In"):

        payload = {
            "name": st.session_state.name,
            "feeling_level": st.session_state.feeling_level,
            "pain": st.session_state.pain_yesno,
            "pain_locations": list(st.session_state.selected_parts),
            "pain_severity": st.session_state.pain_severity,
            "pain_reason": st.session_state.pain_reason,
            "symptoms": list(st.session_state.symptoms),
            "symptom_followup": [
                {"symptom": sym, "question": st.session_state.get(f"gpt_sym_question_{sym}", ""), "answer": ans}
                for sym, ans in st.session_state.symptom_answers.items()
            ],
        }

        # Check if things got worse — if so, go to follow-up stage
        worse = detect_worsening(payload, st.session_state.last_summary)

        if worse and openai_client is not None:
            # Store payload temporarily so follow-up stage can access it
            st.session_state["_pending_payload"] = payload
            st.session_state["_worse_context"] = worse

            # Generate follow-up questions via GPT
            questions = generate_followup_questions(
                payload, worse, st.session_state.last_summary
            )
            st.session_state.followup_questions = questions
            st.session_state.followup_answers = {}
            st.session_state.followup_generated = True

            if questions:
                st.session_state.stage = 4.5
            else:
                # GPT returned nothing — save and move on
                save_to_sheet(payload)
                st.session_state.stage = 5
        else:
            # No worsening detected — save directly
            save_to_sheet(payload)
            st.session_state.stage = 5

        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Stage 4.5 : GPT Follow-up questions
# ------------------------------------------------------------

elif st.session_state.stage == 4.5:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">One follow-up question</div>', unsafe_allow_html=True)

    questions = st.session_state.followup_questions

    for i, question in enumerate(questions):
        # Pre-seed from voice transcript buffer if available
        _tr_key  = f"_transcript_followup_answer_{i}"
        _tr_text = st.session_state.get(_tr_key, "")
        _default = _tr_text or st.session_state.followup_answers.get(i, "")

        answer = st.text_area(
            question,
            value=_default,
            key=f"followup_answer_{i}",
            height=80,
        )
        voice_input_widget(f"followup_answer_{i}")

        # Save whichever is non-empty: typed answer or voice transcript
        saved = answer or _tr_text
        if saved:
            st.session_state.followup_answers[i] = saved

    st.markdown("---")

    if st.button("Submit Answers"):

        payload = st.session_state.get("_pending_payload", {})

        # Attach follow-up Q&A to the payload before saving
        payload["followup_qa"] = [
            {
                "question": questions[i],
                "answer": st.session_state.followup_answers.get(i, ""),
            }
            for i in range(len(questions))
        ]

        save_to_sheet(payload)

        # Clean up temp keys
        st.session_state.pop("_pending_payload", None)
        st.session_state.pop("_worse_context", None)

        st.session_state.stage = 5
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Stage 5
# ------------------------------------------------------------

elif st.session_state.stage == 5:

    st.markdown('<div class="success-box">✅ Check-in complete.</div>', unsafe_allow_html=True)

    if st.button("Start another check-in"):

        for k, v in DEFAULTS.items():
            st.session_state[k] = v

        st.rerun()
