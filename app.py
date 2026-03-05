import json
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials


st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")


# ============================================================
# Secrets
# ============================================================

def _secret(*keys, default=None):
    for k in keys:
        if k in st.secrets:
            return st.secrets[k]
    return default


def _require_secret(*keys):
    v = _secret(*keys)
    if v is None:
        raise KeyError(f"Missing secret: {keys}")
    return v


# ============================================================
# Google Sheets
# ============================================================

sheet = None
sheet_error = None


def init_sheet():
    global sheet, sheet_error

    if sheet or sheet_error:
        return

    try:
        creds = Credentials.from_service_account_info(
            _require_secret("gcp_service_account"),
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )

        book = gspread.authorize(creds).open_by_key(_require_secret("gsheet_id"))

        try:
            sheet = book.worksheet("Form")
        except:
            sheet = book.add_worksheet("Form", rows=2000, cols=20)
            sheet.append_row(["timestamp", "name", "json"])

    except Exception as e:
        sheet_error = str(e)


def save_to_sheet(data: Dict):

    init_sheet()

    if sheet is None:
        return

    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data["name"],
        json.dumps(data)
    ])


def load_past(name: str):

    init_sheet()

    if sheet is None:
        return []

    rows = sheet.get_all_values()

    results = []

    for r in rows[1:]:

        if len(r) < 3:
            continue

        if r[1].strip().lower() == name.strip().lower():

            try:
                d = json.loads(r[2])
                d["timestamp"] = r[0]
                results.append(d)
            except:
                pass

    return results[-5:]


# ============================================================
# Body Map SVG
# ============================================================

def body_map_svg(colors):

    def c(x):
        return colors.get(x, "#cfd8e6")

    return f"""
<svg width="260" height="420" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">

<circle cx="160" cy="70" r="38" fill="{c('Head')}" stroke="#444"/>

<rect x="110" y="120" width="100" height="70" rx="24"
fill="{c('Chest')}" stroke="#444"/>

<rect x="115" y="195" width="90" height="70" rx="22"
fill="{c('Abdomen')}" stroke="#444"/>

<rect x="75" y="130" width="35" height="160"
fill="{c('Left Arm')}" stroke="#444"/>

<rect x="210" y="130" width="35" height="160"
fill="{c('Right Arm')}" stroke="#444"/>

<rect x="135" y="265" width="35" height="200"
fill="{c('Left Leg')}" stroke="#444"/>

<rect x="180" y="265" width="35" height="200"
fill="{c('Right Leg')}" stroke="#444"/>

</svg>
"""


# ============================================================
# State Initialization
# ============================================================

defaults = {

    "stage": -1,
    "name": "",
    "past": [],

    "last": None,
    "last_pain": {},

    "feeling": None,

    "pain_yes": None,

    "selected": set(),
    "severity": {},
    "reason": {},

    "symptoms": set(),

    "submitted": False
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# Body color logic
# ============================================================

GREEN = "#6fd08c"
ORANGE = "#f5a623"
RED = "#e74c3c"


regions = [
    "Head",
    "Chest",
    "Abdomen",
    "Left Arm",
    "Right Arm",
    "Left Leg",
    "Right Leg"
]


def region_color(r):

    last = st.session_state.last_pain

    if r not in st.session_state.selected:

        if r in last:
            return ORANGE
        return GREEN

    last_val = last.get(r, 0)
    cur = st.session_state.severity.get(r, last_val)

    if r not in last:
        return RED

    if cur >= last_val + 2:
        return RED

    return ORANGE


def svg_colors():
    return {r: region_color(r) for r in regions}


# ============================================================
# UI
# ============================================================

st.title("🩺 Cancer Symptom Check-In")

init_sheet()

if sheet_error:
    st.warning("Google Sheets not available")


# ============================================================
# Stage -1 Name
# ============================================================

if st.session_state.stage == -1:

    name = st.text_input("Enter your name")

    if st.button("Start Check-In"):

        if name:

            st.session_state.name = name

            past = load_past(name)

            st.session_state.past = past

            if past:

                last = past[-1]

                st.session_state.last = last
                st.session_state.last_pain = last.get("pain_severity", {})

            st.session_state.stage = 0
            st.rerun()

    st.stop()


stage = st.session_state.stage


# ============================================================
# Stage 0 Doctor Conversation Recap
# ============================================================

if stage == 0:

    last = st.session_state.last

    if last:

        pain = last.get("pain_locations", [])
        symptoms = last.get("symptoms", [])
        feeling = last.get("feeling_level")
        ts = last.get("timestamp", "")

        msg = f"""
👩‍⚕️ **Doctor**

Hi {st.session_state.name}, I reviewed your last check-in"""

        if ts:
            msg += f" from {ts.split()[0]}."

        if pain:
            msg += f" You mentioned pain in your **{', '.join(pain)}**."

        if symptoms:
            msg += f" You also reported **{', '.join(symptoms)}**."

        if feeling:
            msg += f" Your overall feeling level was **{feeling}/10**."

        msg += " Before we continue, has anything changed since then?"

        st.info(msg)

    else:

        st.info("Hello! Let's start your symptom check-in.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Same as yesterday"):

            save_to_sheet({
                "name": st.session_state.name,
                "note": "same"
            })

            st.session_state.stage = 5
            st.rerun()

    with col2:
        if st.button("Something changed"):
            st.session_state.stage = 1
            st.rerun()


# ============================================================
# Stage 1 Feeling
# ============================================================

elif stage == 1:

    st.subheader("How are you feeling today (0-10)?")

    val = st.number_input("Feeling", 0, 10, 7)

    if st.button("Next"):

        st.session_state.feeling = val

        st.session_state.stage = 2
        st.rerun()


# ============================================================
# Stage 2 Pain yes/no
# ============================================================

elif stage == 2:

    st.subheader("Do you have pain today?")

    pain = st.radio("", ["No", "Yes"])

    if st.button("Next"):

        st.session_state.pain_yes = pain == "Yes"

        st.session_state.stage = 3 if pain == "Yes" else 4

        st.rerun()


# ============================================================
# Stage 3 Body Map
# ============================================================

elif stage == 3:

    st.subheader("Where do you feel pain?")

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown(body_map_svg(svg_colors()), unsafe_allow_html=True)

    with col2:

        last = st.session_state.last_pain

        for r in regions:

            icon = "🟢"
            col = region_color(r)

            if col == ORANGE:
                icon = "🟠"

            if col == RED:
                icon = "🔴"

            if st.button(f"{icon} {r}"):

                if r in st.session_state.selected:
                    st.session_state.selected.remove(r)
                else:
                    st.session_state.selected.add(r)

                st.rerun()

            if r in st.session_state.selected:

                last_val = last.get(r, 0)

                sev = st.number_input(
                    f"{r} severity",
                    0,
                    10,
                    last_val,
                    key=f"sev_{r}"
                )

                st.session_state.severity[r] = sev

                if sev > 6 or sev >= last_val + 2:

                    reason = st.text_input(
                        "What caused the worsening?",
                        key=f"why_{r}"
                    )

                    st.session_state.reason[r] = reason

                st.markdown("---")

    if st.button("Next"):
        st.session_state.stage = 4
        st.rerun()


# ============================================================
# Stage 4 Symptoms
# ============================================================

elif stage == 4:

    st.subheader("Symptoms today")

    options = [
        "Fatigue",
        "Nausea",
        "Dry mouth",
        "Difficulty swallowing",
        "Hoarseness",
        "Mouth sores",
        "Skin irritation",
        "Loss of taste"
    ]

    cols = st.columns(2)

    for i, s in enumerate(options):

        col = cols[i % 2]

        with col:

            label = f"🔴 {s}" if s in st.session_state.symptoms else f"🟢 {s}"

            if st.button(label):

                if s in st.session_state.symptoms:
                    st.session_state.symptoms.remove(s)
                else:
                    st.session_state.symptoms.add(s)

                st.rerun()

    if st.button("Submit Check-In"):

        payload = {

            "name": st.session_state.name,
            "feeling_level": st.session_state.feeling,
            "pain": st.session_state.pain_yes,
            "pain_locations": list(st.session_state.selected),
            "pain_severity": st.session_state.severity,
            "pain_reason": st.session_state.reason,
            "symptoms": list(st.session_state.symptoms),
        }

        save_to_sheet(payload)

        st.session_state.stage = 5
        st.rerun()


# ============================================================
# Stage 5 Complete
# ============================================================

elif stage == 5:

    st.success("Check-in complete.")

    if st.button("Start new check-in"):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()
