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

body {
    background-color:#f5f7fb;
}

.block-container{
    padding-top:2rem;
    max-width:900px;
}

/* Header */

.app-header{
    background:linear-gradient(135deg,#4c8df5,#6aa9ff);
    padding:20px 25px;
    border-radius:14px;
    color:white;
    margin-bottom:25px;
}

.app-title{
    font-size:26px;
    font-weight:600;
}

.app-subtitle{
    opacity:0.9;
    font-size:14px;
}

/* Cards */

.card{
    padding:22px;
    border-radius:14px;
    border:1px solid #e7ebf3;
    background:white;
    margin-bottom:18px;
    box-shadow:0 4px 12px rgba(0,0,0,0.04);
}

/* Buttons */

.stButton>button{
    width:100%;
    border-radius:10px;
    padding:0.6rem 0.8rem;
    font-size:15px;
    border:1px solid #e1e5ee;
    background:#f8f9fc;
    transition:0.2s;
}

.stButton>button:hover{
    border:1px solid #4c8df5;
    background:#eef3ff;
}

/* Doctor chat bubble */

.doctor-box{
    padding:18px;
    border-radius:14px;
    background:#f0f5ff;
    border:1px solid #cfe0ff;
    font-size:15px;
    line-height:1.6;
}

/* Section titles */

.section-title{
    font-size:18px;
    font-weight:600;
    margin-bottom:12px;
}

/* Success box */

.success-box{
    padding:22px;
    border-radius:14px;
    background:#ecfff1;
    border:1px solid #a8e6b9;
    font-size:18px;
    text-align:center;
    font-weight:500;
}

/* Symptom pills */

.symptom-pill button{
    border-radius:25px;
}

/* SVG container */

.svg-box{
    display:flex;
    justify-content:center;
    margin-bottom:15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="app-header">
<div class="app-title">🩺 Cancer Symptom Check-In</div>
<div class="app-subtitle">Daily monitoring for your care team</div>
</div>
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
            ws.append_row(["timestamp","name","json"])

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
                except:
                    continue

        return past[-5:]

    except:
        return []


def save_to_sheet(payload: Dict):

    _init_sheets()

    if sheet is None:
        raise RuntimeError(f"Sheets unavailable: {sheets_init_error}")

    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        payload.get("name","Unknown"),
        json.dumps(payload)
    ])


# ============================================================
# Body map SVG
# ============================================================

def body_svg(colors: Dict[str,str]) -> str:

    def c(p): return colors.get(p,"#cfd8e6")

    stroke="#6b7a90"

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
    "stage":-1,
    "name":"",
    "past_checkins":[],
    "last_summary":None,
    "last_pain_severity":{},
    "feeling_level":None,
    "pain_yesno":None,
    "selected_parts":set(),
    "pain_severity":{},
    "pain_reason":{},
    "symptoms":set(),
    "submitted":False
}

for k,v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k]=v


# ============================================================
# UI
# ============================================================

_init_sheets()


# ------------------------------------------------------------
# Stage -1
# ------------------------------------------------------------

if st.session_state.stage == -1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    name = st.text_input("Enter your name")

    if st.button("Start Check-In"):

        if name.strip():

            st.session_state.name=name.strip()

            past = load_past_checkins(name)

            st.session_state.past_checkins=past

            if past:
                last=past[-1]
                st.session_state.last_summary=last
                st.session_state.last_pain_severity=last.get("pain_severity",{})
            else:
                st.session_state.last_summary=None
                st.session_state.last_pain_severity={}

            st.session_state.stage=0
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ------------------------------------------------------------
# Stage 5
# ------------------------------------------------------------

elif st.session_state.stage == 5:

    st.markdown('<div class="success-box">✅ Check-in complete.</div>', unsafe_allow_html=True)

    if st.button("Start another check-in"):

        for k,v in DEFAULTS.items():
            st.session_state[k]=v

        st.rerun()
