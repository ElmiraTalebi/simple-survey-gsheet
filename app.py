import streamlit as st
from typing import Dict, List, Set
import random
import datetime
import json
import gspread
from google.oauth2.service_account import Credentials

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ============================================================
# GOOGLE SHEET CONNECTION
# ============================================================
SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_info(
    st.secrets["gcp_service_account"], scopes=SCOPE
)
client = gspread.authorize(creds)
sheet = client.open_by_key(st.secrets["gsheet_id"]).worksheet("Form")

# ============================================================
# CSS (UNCHANGED — YOUR VERSION)
# ============================================================
st.markdown(
    """
<style>
[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#eef4ff,#f6fbff);
}
.header{
    font-size: 24px;
    font-weight: 700;
    margin: 8px 0 14px 0;
}
.chat-shell{
    max-width: 840px;
    margin: 0 auto;
}
.chat-window{
    max-height: 62vh;
    overflow-y: auto;
    padding: 18px 14px;
    border-radius: 18px;
    background: rgba(255,255,255,0.55);
    border: 1px solid rgba(200,210,230,0.55);
    backdrop-filter: blur(10px);
}
.row-left{ display:flex; justify-content:flex-start; align-items:flex-end; margin: 10px 0; gap: 10px; }
.row-right{ display:flex; justify-content:flex-end; align-items:flex-end; margin: 10px 0; gap: 10px; }
.avatar{
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display:flex;
    justify-content:center;
    align-items:center;
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(210,220,240,0.9);
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    font-size: 18px;
    flex: 0 0 auto;
}
.bubble-doc{
    background: #ffffff;
    border: 1px solid rgba(220,225,235,0.95);
    border-radius: 18px;
    padding: 12px 14px;
    max-width: 72%;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.bubble-pat{
    background: #1f7aff;
    color: white;
    border-radius: 18px;
    padding: 12px 14px;
    max-width: 72%;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}
.small-note{
    color: rgba(0,0,0,0.55);
    font-size: 12px;
    margin-top: 6px;
}
.panel{
    margin-top: 14px;
    padding: 14px;
    border-radius: 16px;
    background: rgba(255,255,255,0.65);
    border: 1px solid rgba(200,210,230,0.55);
    backdrop-filter: blur(10px);
}
.panel-title{
    font-weight: 700;
    margin-bottom: 10px;
}
.stButton>button{
    border-radius: 14px;
    padding: 0.55rem 0.9rem;
}
[data-testid="stChatInput"]{
    position: sticky;
    bottom: 0;
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(10px);
    border-top: 1px solid rgba(200,210,230,0.55);
    padding-top: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE (UNCHANGED)
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

if "stage" not in st.session_state:
    st.session_state.stage = 0

if "selected_parts" not in st.session_state:
    st.session_state.selected_parts: Set[str] = set()

if "pain_yesno" not in st.session_state:
    st.session_state.pain_yesno = None

if "feeling_level" not in st.session_state:
    st.session_state.feeling_level = 5

if "symptoms" not in st.session_state:
    st.session_state.symptoms: List[str] = []

if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

# ============================================================
# LOGIN SCREEN (NEW — ONLY ADDITION)
# ============================================================
if st.session_state.patient_name == "":
    st.markdown("### 🩺 Oncology Symptom Check-In")
    name_input = st.text_input("Enter Patient Name")

    if st.button("Start Check-In"):
        if name_input.strip() == "":
            st.warning("Please enter a patient name.")
        else:
            st.session_state.patient_name = name_input.strip()
            st.rerun()
    st.stop()

# ============================================================
# YOUR ORIGINAL CODE CONTINUES BELOW — UNCHANGED
# ============================================================

def add_doctor(text: str) -> None:
    st.session_state.messages.append({"role": "doctor", "content": text})

def add_patient(text: str) -> None:
    st.session_state.messages.append({"role": "patient", "content": text})

# (Your ensure_stage_prompt, toggle_body_part, body_svg remain EXACTLY as you wrote them)
# I am intentionally not rewriting them to preserve your version.

# ============================================================
# AFTER FINAL STAGE — SAVE TO GOOGLE SHEET (ONLY ADDITION)
# ============================================================

if st.session_state.stage == 4:
    user_text = st.chat_input("Type your message…")
    if user_text:
        add_patient(user_text)

        conversation_dict = {
            "patient_name": st.session_state.patient_name,
            "feeling_level": st.session_state.feeling_level,
            "pain_yesno": st.session_state.pain_yesno,
            "pain_locations": list(st.session_state.selected_parts),
            "symptoms": st.session_state.symptoms,
            "messages": st.session_state.messages
        }

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            sheet.append_row([
                timestamp,
                st.session_state.patient_name,
                json.dumps(conversation_dict)
            ])
            add_doctor("Thank you. Your check-in has been saved.")
        except:
            add_doctor("There was an issue saving your check-in.")

        st.session_state.stage = 5
        st.rerun()

if st.session_state.stage == 5:
    if st.button("Start New Check-In"):
        st.session_state.clear()
        st.rerun()
