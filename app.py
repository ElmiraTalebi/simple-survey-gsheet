import json
from datetime import datetime
from typing import Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI

st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ============================================================
# Secrets helper
# ============================================================

def _secret(*keys, default=None):
    for k in keys:
        if k in st.secrets:
            return st.secrets[k]
    return default


# ============================================================
# OpenAI (ONLY for optional summarization)
# ============================================================

OPENAI_API_KEY = _secret("OPENAI_API_KEY", "openai_api_key")
openai_client: Optional[OpenAI] = None

if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        pass


# ============================================================
# Google Sheets
# ============================================================

sheet = None


def init_sheets():
    global sheet

    if sheet:
        return

    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )

        gc = gspread.authorize(creds)
        book = gc.open_by_key(st.secrets["gsheet_id"])

        try:
            sheet = book.worksheet("Form")
        except Exception:
            sheet = book.add_worksheet(title="Form", rows=2000, cols=20)
            sheet.append_row(["timestamp", "name", "json"])

    except Exception as e:
        st.warning(f"Sheets unavailable: {e}")


def load_last_checkin(name):

    init_sheets()

    if sheet is None:
        return None

    rows = sheet.get_all_values()

    for row in reversed(rows[1:]):
        if row[1].lower() == name.lower():
            try:
                return json.loads(row[2])
            except:
                return None

    return None


def save_to_sheet(data):

    init_sheets()

    if sheet is None:
        return

    sheet.append_row(
        [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data["name"],
            json.dumps(data),
        ]
    )


# ============================================================
# Clinical logic
# ============================================================


def is_same_as_yesterday(text):

    text = text.lower()

    keywords = [
        "same",
        "same as yesterday",
        "about the same",
        "no change",
        "nothing changed",
        "all good",
    ]

    return any(k in text for k in keywords)


# ============================================================
# Session state
# ============================================================

defaults = dict(
    stage=-1,
    name="",
    feeling=None,
    pain=None,
    pain_locations=set(),
    pain_severity=0,
    symptoms="",
    submitted=False,
    show_other_pain=False,
    last_pain_severity=0,
)

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# UI header
# ============================================================

st.title("🩺 Cancer Symptom Check-In")

# ============================================================
# Stage -1 : name entry
# ============================================================

if st.session_state.stage == -1:

    name = st.text_input("Enter your name")

    if st.button("Start Check-In"):

        if name:

            st.session_state.name = name

            last = load_last_checkin(name)

            if last:
                st.session_state.last_pain_severity = last.get("pain_severity", 0)

            st.session_state.stage = 0
            st.rerun()

    st.stop()

stage = st.session_state.stage

# ============================================================
# Stage 0 : fast exit
# ============================================================

if stage == 0:

    st.subheader("How have you been since your last visit?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Same as yesterday"):

            st.success("Check-in complete. Nothing changed.")

            st.session_state.submitted = True
            st.session_state.stage = 5
            st.rerun()

    with col2:
        if st.button("Something changed"):

            st.session_state.stage = 1
            st.rerun()

    text = st.text_input("Or type your answer")

    if text:

        if is_same_as_yesterday(text):

            st.success("Check-in complete.")

            st.session_state.submitted = True
            st.session_state.stage = 5

        else:

            st.session_state.stage = 1

        st.rerun()


# ============================================================
# Stage 1 : feeling
# ============================================================

elif stage == 1:

    feeling = st.radio(
        "How are you feeling today?",
        ["Excellent", "Very good", "Good", "Fair", "Poor"],
    )

    if st.button("Next"):

        st.session_state.feeling = feeling

        st.session_state.stage = 2
        st.rerun()


# ============================================================
# Stage 2 : pain yes/no
# ============================================================

elif stage == 2:

    pain = st.radio("Do you have pain today?", ["No", "Yes"])

    if st.button("Next"):

        st.session_state.pain = pain

        if pain == "Yes":

            st.session_state.stage = 3

        else:

            st.session_state.stage = 4

        st.rerun()


# ============================================================
# Stage 3 : body map
# ============================================================

elif stage == 3:

    st.subheader("Mark where you feel pain")

    parts = [
        "Head",
        "Neck",
        "Chest",
        "Abdomen",
        "Left Arm",
        "Right Arm",
        "Left Leg",
        "Right Leg",
    ]

    cols = st.columns(2)

    for i, part in enumerate(parts):

        selected = part in st.session_state.pain_locations

        color = "🔴" if selected else "🟢"

        with cols[i % 2]:

            if st.button(f"{color} {part}", key=part):

                if selected:
                    st.session_state.pain_locations.remove(part)
                else:
                    st.session_state.pain_locations.add(part)

                st.rerun()

    st.markdown("---")

    st.subheader("Pain severity")

    severity = st.slider(
        "",
        0,
        10,
        value=st.session_state.last_pain_severity,
    )

    st.session_state.pain_severity = severity

    if st.button("Other location"):

        st.session_state.show_other_pain = True

    if st.session_state.show_other_pain:

        other = st.text_input("Describe other pain location")

        if other:
            st.session_state.pain_locations.add(other)

    if st.button("Next"):

        st.session_state.stage = 4
        st.rerun()


# ============================================================
# Stage 4 : symptoms
# ============================================================

elif stage == 4:

    st.subheader("Symptoms today")

    symptoms = st.text_area("Describe any symptoms")

    if st.button("Submit Check-In"):

        st.session_state.symptoms = symptoms

        data = dict(
            name=st.session_state.name,
            feeling=st.session_state.feeling,
            pain=st.session_state.pain,
            pain_locations=list(st.session_state.pain_locations),
            pain_severity=st.session_state.pain_severity,
            symptoms=symptoms,
        )

        save_to_sheet(data)

        st.session_state.submitted = True
        st.session_state.stage = 5

        st.rerun()


# ============================================================
# Stage 5 : completed
# ============================================================

elif stage == 5:

    st.success("Check-in complete.")

    st.write("Your care team will review your responses.")

    if st.button("Start another check-in"):

        for k in defaults:
            st.session_state[k] = defaults[k]

        st.rerun()
