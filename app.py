import json
from datetime import datetime
from typing import Dict, List, Optional, Set

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ============================================================
# MODERN UI STYLE
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    padding-top: 2.5rem;
    max-width: 860px;
}

/* Page background */
.stApp {
    background: linear-gradient(135deg, #f0f5fb 0%, #e8f0f9 100%);
}

/* Title */
h1 {
    font-family: 'DM Serif Display', serif;
    font-weight: 400;
    font-size: 2rem;
    color: #1a3a5c;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}

h2, h3 {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: #1a3a5c;
}

/* Buttons – base */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    padding: 0.65rem 1rem;
    font-size: 14.5px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    border: 1.5px solid #d4deef;
    background: #ffffff;
    color: #1a3a5c;
    box-shadow: 0 2px 6px rgba(26,58,92,0.06);
    transition: all 0.18s ease;
    letter-spacing: 0.01em;
}

.stButton > button:hover {
    border-color: #3d8fc0;
    background: #eef5fc;
    color: #1a5a8a;
    box-shadow: 0 4px 14px rgba(61,143,192,0.15);
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0px);
    box-shadow: 0 2px 6px rgba(26,58,92,0.08);
}

/* Cards */
.card {
    padding: 28px 30px;
    border-radius: 18px;
    border: 1.5px solid #dde8f5;
    background: #ffffff;
    margin-bottom: 18px;
    box-shadow: 0 4px 20px rgba(26,58,92,0.07);
}

/* Doctor box */
.doctor-box {
    padding: 22px 24px;
    border-radius: 16px;
    background: linear-gradient(135deg, #eaf4fc 0%, #f0f8ff 100%);
    border: 1.5px solid #b8d9f0;
    font-size: 15px;
    font-family: 'DM Sans', sans-serif;
    color: #1a3a5c;
    line-height: 1.65;
    box-shadow: 0 3px 12px rgba(61,143,192,0.1);
    position: relative;
}

.doctor-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, #3d8fc0, #5ab0d8);
    border-radius: 16px 0 0 16px;
}

/* Section titles */
.section-title {
    font-size: 17px;
    font-weight: 600;
    color: #1a3a5c;
    margin-bottom: 14px;
    letter-spacing: -0.1px;
}

/* Success box */
.success-box {
    padding: 32px 24px;
    border-radius: 18px;
    background: linear-gradient(135deg, #e8fbf0 0%, #f0fff6 100%);
    border: 1.5px solid #8dd9ae;
    font-size: 18px;
    font-family: 'DM Serif Display', serif;
    font-weight: 400;
    color: #1a4d32;
    text-align: center;
    box-shadow: 0 4px 18px rgba(61,180,110,0.12);
    letter-spacing: 0.01em;
}

/* Input fields */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border-radius: 10px;
    border: 1.5px solid #d4deef;
    font-family: 'DM Sans', sans-serif;
    font-size: 14.5px;
    color: #1a3a5c;
    padding: 0.5rem 0.8rem;
    background: #fafcff;
    transition: border 0.15s ease, box-shadow 0.15s ease;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #3d8fc0;
    box-shadow: 0 0 0 3px rgba(61,143,192,0.12);
    background: #ffffff;
}

/* Radio buttons */
.stRadio > div {
    gap: 10px;
}

.stRadio > div > label {
    font-family: 'DM Sans', sans-serif;
    font-size: 14.5px;
    color: #1a3a5c;
}

/* Divider */
hr {
    border: none;
    border-top: 1.5px solid #e4edf7;
    margin: 18px 0;
}

/* Number input label */
.stNumberInput label, .stTextInput label, .stRadio label {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: #3a5a7a;
    font-size: 13.5px;
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

    last_val = last.get(region, 0)
    cur = st.session_state.pain_severity.get(region, last_val)

    if region not in last:
        return RED

    if cur >= last_val + 2:
        return RED

    return ORANGE


def current_svg_colors():
    return {r: region_color_state(r) for r in REGIONS}


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

    st.markdown("---")

    c1, c2 = st.columns(2)

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


# ------------------------------------------------------------
# Stage 1
# ------------------------------------------------------------

elif st.session_state.stage == 1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">How are you feeling today?</div>', unsafe_allow_html=True)

    feeling = st.number_input("Feeling (0-10)", 0, 10, value=7)

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

            if st.button(f"{icon} {r}"):

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

                if sev > 6 or sev >= last_val + 2:

                    reason = st.text_input("Reason", key=f"reason_{r}")

                    if reason:
                        st.session_state.pain_reason[r] = reason

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

        save_to_sheet(payload)

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
