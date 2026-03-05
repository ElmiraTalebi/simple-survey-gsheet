import json
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI

st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺")

# ============================================================
# Secrets helper
# ============================================================

def _secret(*keys, default=None):
    for k in keys:
        if k in st.secrets:
            return st.secrets[k]
    return default


# ============================================================
# OpenAI (used ONLY for optional summarization)
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
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
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


def save_to_sheet(data):

    init_sheets()

    if sheet is None:
        return

    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data["name"],
        json.dumps(data)
    ])


# ============================================================
# Clinical Rules
# ============================================================

def is_same_as_yesterday(text: str) -> bool:

    text = text.lower()

    keywords = [
        "same",
        "same as yesterday",
        "about the same",
        "no change",
        "nothing changed",
        "all good"
    ]

    return any(k in text for k in keywords)


def is_concerning(stage: int, text: str):

    t = text.lower()

    if stage == 1:
        return "poor" in t or "fair" in t or "worse" in t

    if stage == 2:
        return "yes" in t

    if stage == 3:
        return "severe" in t or "worse" in t

    if stage == 4:
        return "yes" in t

    return False


FOLLOWUPS = {

    1: "What seems to be making you feel this way?",
    2: "How severe is the pain?",
    3: "Has the pain been getting worse?",
    4: "Which symptom is bothering you most?"
}


QUESTIONS = {

    0: "How have you been since your last visit?",
    1: "How are you feeling today?",
    2: "Do you have any pain today?",
    3: "Where do you feel pain?",
    4: "Are you experiencing any symptoms today?"
}


# ============================================================
# Session State
# ============================================================

defaults = dict(

    stage=-1,
    name="",
    feeling=None,
    pain=None,
    pain_location="",
    symptoms="",
    submitted=False,
    messages=[]
)

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# Helpers
# ============================================================

def add_doctor(text):
    st.session_state.messages.append(("doctor", text))


def add_patient(text):
    st.session_state.messages.append(("patient", text))


def advance():
    st.session_state.stage += 1


# ============================================================
# UI Header
# ============================================================

st.title("🩺 Cancer Symptom Check-In")


# ============================================================
# Stage -1: Name
# ============================================================

if st.session_state.stage == -1:

    name = st.text_input("Enter your name")

    if st.button("Start Check-In"):

        if name:

            st.session_state.name = name
            st.session_state.stage = 0
            add_doctor(QUESTIONS[0])

            st.rerun()

    st.stop()


# ============================================================
# Chat History
# ============================================================

for role, msg in st.session_state.messages:

    if role == "doctor":
        st.info(msg)

    else:
        st.success(msg)


stage = st.session_state.stage


# ============================================================
# Stage 0 — Fast Exit
# ============================================================

if stage == 0:

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Same as yesterday"):

            add_patient("Same as yesterday")

            add_doctor("Great. Your check-in is complete.")

            st.session_state.submitted = True
            st.session_state.stage = 5
            st.rerun()

    with col2:
        if st.button("Something changed"):

            add_patient("Something changed")

            advance()
            add_doctor(QUESTIONS[1])

            st.rerun()

    text = st.text_input("Or type your answer")

    if text:

        add_patient(text)

        if is_same_as_yesterday(text):

            add_doctor("Thanks for checking in. Your check-in is complete.")
            st.session_state.submitted = True
            st.session_state.stage = 5

        else:

            advance()
            add_doctor(QUESTIONS[1])

        st.rerun()


# ============================================================
# Stage 1 — Feeling
# ============================================================

elif stage == 1:

    feeling = st.radio(

        "How are you feeling today?",
        ["Excellent", "Very good", "Good", "Fair", "Poor"]
    )

    if st.button("Next"):

        st.session_state.feeling = feeling

        add_patient(feeling)

        if is_concerning(1, feeling):
            add_doctor(FOLLOWUPS[1])
        else:
            advance()
            add_doctor(QUESTIONS[2])

        st.rerun()


# ============================================================
# Stage 2 — Pain
# ============================================================

elif stage == 2:

    pain = st.radio("Do you have pain today?", ["No", "Yes"])

    if st.button("Next"):

        st.session_state.pain = pain

        add_patient(pain)

        if pain == "Yes":
            add_doctor(FOLLOWUPS[2])
        else:
            advance()
            add_doctor(QUESTIONS[4])

        st.rerun()


# ============================================================
# Stage 3 — Pain follow-up
# ============================================================

elif stage == 3:

    severity = st.slider("Pain severity", 0, 10, 3)

    if st.button("Next"):

        add_patient(f"Pain severity {severity}/10")

        advance()
        add_doctor(QUESTIONS[4])

        st.rerun()


# ============================================================
# Stage 4 — Symptoms
# ============================================================

elif stage == 4:

    symptoms = st.text_area("Describe any symptoms")

    if st.button("Submit"):

        st.session_state.symptoms = symptoms

        add_patient(symptoms if symptoms else "No symptoms")

        add_doctor("Thank you. Your check-in has been recorded.")

        data = dict(

            name=st.session_state.name,
            feeling=st.session_state.feeling,
            pain=st.session_state.pain,
            symptoms=st.session_state.symptoms
        )

        save_to_sheet(data)

        st.session_state.submitted = True
        st.session_state.stage = 5

        st.rerun()


# ============================================================
# Stage 5 — Completed
# ============================================================

elif stage == 5:

    st.success("Check-in complete.")

    st.write("Your care team will review your responses.")

    if st.button("Start another check-in"):

        for k in defaults:
            st.session_state[k] = defaults[k]

        st.rerun()
