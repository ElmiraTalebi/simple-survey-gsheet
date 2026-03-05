import json
from datetime import datetime
from typing import Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
import numpy as np

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
# OpenAI (only for optional summarization)
# ============================================================

OPENAI_API_KEY = _secret("OPENAI_API_KEY", "openai_api_key")
openai_client: Optional[OpenAI] = None

if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except:
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
        except:
            sheet = book.add_worksheet(title="Form", rows=2000, cols=20)
            sheet.append_row(["timestamp", "name", "json"])

    except Exception as e:
        st.warning(f"Sheets unavailable: {e}")


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
# Session state
# ============================================================

defaults = dict(
    stage=-1,
    name="",
    pain_map={},
    last_severity=0,
    submitted=False,
)

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# Clinical helpers
# ============================================================

def is_same_as_yesterday(text):

    text = text.lower()

    keywords = [
        "same",
        "same as yesterday",
        "about the same",
        "no change",
        "nothing changed",
    ]

    return any(k in text for k in keywords)


# ============================================================
# Body regions (coordinate mapping)
# ============================================================

BODY_REGIONS = {

    "Head": (120, 40, 200, 110),
    "Chest": (100, 110, 220, 200),
    "Abdomen": (110, 200, 210, 280),
    "Left Arm": (40, 120, 100, 260),
    "Right Arm": (220, 120, 280, 260),
    "Left Leg": (110, 280, 150, 420),
    "Right Leg": (170, 280, 210, 420),
}


def get_clicked_region(x, y):

    for part, (x1, y1, x2, y2) in BODY_REGIONS.items():

        if x1 <= x <= x2 and y1 <= y <= y2:
            return part

    return None


# ============================================================
# UI Header
# ============================================================

st.title("🩺 Cancer Symptom Check-In")


# ============================================================
# Stage -1 Name
# ============================================================

if st.session_state.stage == -1:

    name = st.text_input("Enter your name")

    if st.button("Start Check-In"):

        if name:

            st.session_state.name = name
            st.session_state.stage = 0
            st.rerun()

    st.stop()


stage = st.session_state.stage


# ============================================================
# Stage 0 Fast Exit
# ============================================================

if stage == 0:

    st.subheader("How have you been since your last visit?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Same as yesterday"):

            st.success("Check-in complete.")

            st.session_state.submitted = True
            st.session_state.stage = 5
            st.rerun()

    with col2:
        if st.button("Something changed"):

            st.session_state.stage = 2
            st.rerun()


# ============================================================
# Stage 2 Pain Question
# ============================================================

elif stage == 2:

    pain = st.radio("Do you have pain today?", ["No", "Yes"])

    if st.button("Next"):

        if pain == "Yes":
            st.session_state.stage = 3
        else:
            st.session_state.stage = 4

        st.rerun()


# ============================================================
# Stage 3 Clickable Body Map
# ============================================================

elif stage == 3:

    st.subheader("Click where you feel pain")

    body_img = Image.open("body_map.png")

    click = streamlit_image_coordinates(body_img)

    if click:

        region = get_clicked_region(click["x"], click["y"])

        if region:

            if region not in st.session_state.pain_map:
                st.session_state.pain_map[region] = {
                    "severity": None,
                    "reason": None
                }

    for part, data in st.session_state.pain_map.items():

        st.markdown(f"### {part}")

        if data["severity"] is None:

            st.markdown(
                f"""
                <div style="background:#eef2ff;padding:10px;border-radius:10px">
                🩺 How severe is the pain in your {part.lower()}? (0–10)
                </div>
                """,
                unsafe_allow_html=True
            )

            sev = st.number_input(
                "",
                min_value=0,
                max_value=10,
                key=f"sev_{part}"
            )

            if st.button("Submit", key=f"submit_{part}"):

                st.session_state.pain_map[part]["severity"] = sev
                st.rerun()

        else:

            sev = data["severity"]
            st.write(f"Severity: {sev}/10")

            if sev > 6 or sev > st.session_state.last_severity:

                if data["reason"] is None:

                    st.markdown(
                        f"""
                        <div style="background:#eef2ff;padding:10px;border-radius:10px">
                        🩺 What seems to be causing the increase in your {part.lower()} pain?
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    reason = st.text_input("", key=f"reason_{part}")

                    if st.button("Submit reason", key=f"reasonbtn_{part}"):

                        st.session_state.pain_map[part]["reason"] = reason
                        st.rerun()

    if st.button("Next"):
        st.session_state.stage = 4
        st.rerun()


# ============================================================
# Stage 4 Symptoms
# ============================================================

elif stage == 4:

    symptoms = st.text_area("Any symptoms today?")

    if st.button("Submit Check-In"):

        data = dict(
            name=st.session_state.name,
            pain_map=st.session_state.pain_map,
            symptoms=symptoms
        )

        save_to_sheet(data)

        st.session_state.stage = 5
        st.session_state.submitted = True
        st.rerun()


# ============================================================
# Stage 5 Completed
# ============================================================

elif stage == 5:

    st.success("Check-in complete.")

    if st.button("Start another check-in"):

        for k in defaults:
            st.session_state[k] = defaults[k]

        st.rerun()
