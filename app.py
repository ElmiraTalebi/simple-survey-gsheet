import hashlib
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
# Audio / Voice
# ============================================================

def transcribe_audio(audio_bytes: bytes) -> str:
    """Send raw audio bytes to OpenAI Whisper and return transcript."""
    if openai_client is None:
        return "(Transcription failed: no OpenAI client)"
    try:
        import io
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"
        result = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
        return result.text.strip()
    except Exception as e:
        return f"(Transcription failed: {e})"


def interpret_voice_answer(transcript: str, question: str) -> str:
    """
    Passes the raw Whisper transcript + the question to GPT to produce
    a clean, concise written answer. Falls back to raw transcript on error.
    """
    if openai_client is None:
        return transcript
    try:
        prompt = (
            f"A cancer patient was asked the following question during a symptom check-in:\n"
            f"Question: {question}\n"
            f"The patient spoke this answer (raw speech transcription): {transcript}\n\n"
            f"Rewrite their answer as a clean, concise written response in first person. "
            f"Keep all medical details exactly as stated. Do not add anything not mentioned. "
            f"Return only the cleaned answer, nothing else."
        )
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return transcript


def voice_input_widget(answer_key: str, label: str = "🎤 Or speak your answer", question: str = ""):
    """
    Renders a mic input using st.audio_input.
    Flow: record → Whisper transcription → GPT cleans answer → stored in _transcript_ buffer.
    The text input above should use value=st.session_state.get(f"_transcript_{answer_key}", "")
    """
    transcript_key = f"_transcript_{answer_key}"
    last_hash_key  = f"_audio_hash_{answer_key}"

    if last_hash_key not in st.session_state:
        st.session_state[last_hash_key] = None

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
    if ah == st.session_state[last_hash_key]:
        return  # same clip already processed

    st.session_state[last_hash_key] = ah

    with st.spinner("Transcribing…"):
        raw_text = transcribe_audio(ab)

    if not raw_text or raw_text.startswith("(Transcription failed"):
        st.warning("Could not transcribe — please try again or type your answer.")
        return

    # Pass transcript through GPT to clean it up
    with st.spinner("Processing your answer…"):
        cleaned = interpret_voice_answer(raw_text, question) if question else raw_text

    # Write to buffer — never to the widget key directly (Streamlit restriction)
    st.session_state[transcript_key] = cleaned
    st.rerun()


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
    """
    if not last_summary:
        return {}

    worse = {}

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

    # New symptoms
    last_symptoms = set(last_summary.get("symptoms", []))
    cur_symptoms = set(payload.get("symptoms", []))
    new_symptoms = cur_symptoms - last_symptoms
    if len(new_symptoms) >= 1:
        worse["new_symptoms"] = list(new_symptoms)

    return worse


# ============================================================
# GPT follow-up question generation
# ============================================================

def generate_followup_questions(payload: Dict, worse: Dict, last_summary: Dict) -> List[str]:
    """
    Generates at most 1 specific follow-up question, only when clinically necessary
    and not already answered by the patient in the body map stage.
    """
    if openai_client is None:
        return []

    pain_reason = payload.get("pain_reason", {})
    pain_severity = payload.get("pain_severity", {})

    # Find the single most critical unanswered issue — strict priority order
    critical_issue = None

    # 1. Severity >= 8 with no reason given
    for region, sev in pain_severity.items():
        if sev >= 8 and region not in pain_reason:
            critical_issue = f"Pain in {region} is severe ({sev}/10) and no reason was provided."
            break

    # 2. Pain jumped by 3+ with no reason given
    if not critical_issue and "worsened_pain" in worse:
        for region, change in worse["worsened_pain"].items():
            jump = change["to"] - change["from"]
            if jump >= 3 and region not in pain_reason:
                critical_issue = (
                    f"Pain in {region} jumped from {change['from']}/10 to {change['to']}/10 "
                    f"with no explanation provided."
                )
                break

    # 3. New pain location with severity >= 6 and no reason
    if not critical_issue and "new_pain_locations" in worse:
        for region in worse["new_pain_locations"]:
            sev = pain_severity.get(region, 0)
            if sev >= 6 and region not in pain_reason:
                critical_issue = f"New pain in {region} at {sev}/10 with no explanation."
                break

    # Nothing critical or everything already explained — no follow-up needed
    if not critical_issue:
        return []

    prompt = (
        f"You are a concise oncology nurse. A cancer patient has this unresolved clinical issue: "
        f"{critical_issue} "
        f"Ask exactly ONE specific, short question to clarify only this issue. "
        f"Do not ask anything already covered. Do not be vague or generic. "
        f'Return ONLY a JSON array with one question string. Example: ["Your question here?"]'
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
    default_feeling = int(last_feeling) if last_feeling is not None else 0
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

                # Ask GPT-generated follow-up only when severity is high or significantly worsened
                needs_followup = sev > 6 or sev >= last_val + 2
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

                    question = st.session_state.get(q_key) or "Can you describe what makes this pain better or worse?"
                    _tr_reason = st.session_state.get(f"_transcript_reason_{r}", "")
                    _default_reason = _tr_reason or st.session_state.pain_reason.get(r, "")
                    answer = st.text_input(
                        question,
                        value=_default_reason,
                        key=f"reason_{r}"
                    )
                    voice_input_widget(f"reason_{r}", label="🎤 Or speak", question=question)
                    if answer:
                        st.session_state.pain_reason[r] = answer
                    elif _tr_reason:
                        st.session_state.pain_reason[r] = _tr_reason

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

    cols = st.columns(2)

    for i, sym in enumerate(symptom_options):

        selected = sym in st.session_state.symptoms
        label = f"🔴 {sym}" if selected else f"🟢 {sym}"

        with cols[i % 2]:

            if st.button(label):

                if selected:
                    st.session_state.symptoms.remove(sym)
                else:
                    st.session_state.symptoms.add(sym)

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
        _tr_fup = st.session_state.get(f"_transcript_followup_answer_{i}", "")
        _default_fup = _tr_fup or st.session_state.followup_answers.get(i, "")
        answer = st.text_area(
            question,
            value=_default_fup,
            key=f"followup_answer_{i}",
            height=80,
        )
        voice_input_widget(f"followup_answer_{i}", question=question)
        saved = answer or _tr_fup
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
