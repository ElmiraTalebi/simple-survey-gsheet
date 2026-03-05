import json
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cancer Symptom Check-In",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# MODERN UI STYLE
# ============================================================

st.markdown("""
<style>

body {
    background-color:#f6f8fc;
}

.block-container{
    max-width:900px;
    padding-top:2rem;
}

/* HEADER */

.header-box{
    background:linear-gradient(135deg,#5b8def,#7ab0ff);
    padding:22px 25px;
    border-radius:16px;
    color:white;
    margin-bottom:25px;
}

.header-title{
    font-size:28px;
    font-weight:600;
}

.header-sub{
    opacity:0.9;
    font-size:14px;
}

/* CARD */

.card{
    padding:22px;
    border-radius:14px;
    border:1px solid #e7ebf3;
    background:white;
    margin-bottom:18px;
    box-shadow:0 4px 14px rgba(0,0,0,0.05);
}

/* BUTTON */

.stButton>button{
    width:100%;
    border-radius:10px;
    padding:0.6rem 0.8rem;
    font-size:15px;
    border:1px solid #e1e5ee;
    background:#f8f9fc;
}

.stButton>button:hover{
    border:1px solid #5b8def;
    background:#eef3ff;
}

/* CHAT BUBBLE */

.doctor-bubble{
    padding:18px;
    border-radius:14px;
    background:#f1f6ff;
    border:1px solid #d5e4ff;
    margin-bottom:15px;
}

/* SECTION */

.section-title{
    font-size:18px;
    font-weight:600;
    margin-bottom:10px;
}

/* SUCCESS */

.success-box{
    padding:22px;
    border-radius:14px;
    background:#eaffef;
    border:1px solid #a8e6b9;
    font-size:18px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="header-box">
<div class="header-title">🩺 Cancer Symptom Check-In</div>
<div class="header-sub">Daily monitoring for your care team</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PROGRESS BAR
# ============================================================

def progress(stage):

    mapping = {
        -1:0.0,
        0:0.1,
        1:0.25,
        2:0.45,
        3:0.65,
        4:0.85,
        5:1.0
    }

    st.progress(mapping.get(stage,0))


# ============================================================
# SECRET HELPERS
# ============================================================

def _secret(*keys, default=None):
    for k in keys:
        if k in st.secrets:
            return st.secrets[k]
    return default


def _require_secret(*keys):
    v = _secret(*keys)
    if v is None:
        raise KeyError(f"Missing secret {keys}")
    return v


# ============================================================
# GOOGLE SHEETS
# ============================================================

sheet=None
sheets_init_error=None


def _init_sheets():

    global sheet, sheets_init_error

    if sheet or sheets_init_error:
        return

    try:

        creds = Credentials.from_service_account_info(
            _require_secret("gcp_service_account"),
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )

        book = gspread.authorize(creds).open_by_key(
            _require_secret("gsheet_id")
        )

        try:
            ws = book.worksheet("Form")
        except:
            ws = book.add_worksheet(title="Form", rows=2000, cols=20)
            ws.append_row(["timestamp","name","json"])

        sheet = ws

    except Exception as e:

        sheets_init_error=str(e)


def load_past_checkins(name:str):

    _init_sheets()

    if sheet is None:
        return []

    try:

        rows=sheet.get_all_values()

        past=[]

        for r in rows[1:]:

            if len(r)>=3 and r[1].lower()==name.lower():

                try:

                    data=json.loads(r[2])
                    data["timestamp"]=r[0]

                    past.append(data)

                except:
                    pass

        return past[-5:]

    except:
        return []


def save_to_sheet(payload):

    _init_sheets()

    if sheet is None:
        return

    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        payload.get("name"),
        json.dumps(payload)
    ])


# ============================================================
# BODY MAP
# ============================================================

REGIONS=[
"Head",
"Chest",
"Abdomen",
"Left Arm",
"Right Arm",
"Left Leg",
"Right Leg"
]


def body_svg():

    return """
<svg width="250" height="380" viewBox="0 0 320 520">

<circle cx="160" cy="70" r="38" fill="#d7deef"/>
<rect x="110" y="120" width="100" height="70" rx="24" fill="#d7deef"/>
<rect x="115" y="195" width="90" height="70" rx="22" fill="#d7deef"/>

<rect x="60" y="140" width="40" height="120" rx="20" fill="#d7deef"/>
<rect x="220" y="140" width="40" height="120" rx="20" fill="#d7deef"/>

<rect x="130" y="270" width="35" height="150" rx="20" fill="#d7deef"/>
<rect x="165" y="270" width="35" height="150" rx="20" fill="#d7deef"/>

</svg>
"""


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS={
"stage":-1,
"name":"",
"past_checkins":[],
"last_summary":None,
"feeling_level":None,
"pain_yesno":None,
"selected_parts":set(),
"pain_severity":{},
"pain_reason":{},
"symptoms":set()
}

for k,v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k]=v


progress(st.session_state.stage)


# ============================================================
# STAGE -1 NAME
# ============================================================

if st.session_state.stage==-1:

    st.markdown('<div class="card">',unsafe_allow_html=True)

    name=st.text_input("Enter your name")

    if st.button("Start Check-In"):

        if name:

            st.session_state.name=name

            past=load_past_checkins(name)

            st.session_state.past_checkins=past

            if past:
                st.session_state.last_summary=past[-1]

            st.session_state.stage=0
            st.rerun()

    st.markdown('</div>',unsafe_allow_html=True)


# ============================================================
# STAGE 0 RECAP
# ============================================================

elif st.session_state.stage==0:

    st.markdown('<div class="card">',unsafe_allow_html=True)

    last=st.session_state.last_summary

    if last:

        msg=f"""
Hi **{st.session_state.name}** 👋  

I reviewed your previous check-in.

Before we continue, **has anything changed since then?**
"""

        st.markdown(
        f'<div class="doctor-bubble">{msg}</div>',
        unsafe_allow_html=True
        )

    c1,c2=st.columns(2)

    with c1:

        if st.button("Same as yesterday"):

            payload={
            "name":st.session_state.name,
            "note":"Same as yesterday"
            }

            save_to_sheet(payload)

            st.session_state.stage=5
            st.rerun()

    with c2:

        if st.button("Something changed"):

            st.session_state.stage=1
            st.rerun()

    st.markdown('</div>',unsafe_allow_html=True)


# ============================================================
# STAGE 1 FEELING
# ============================================================

elif st.session_state.stage==1:

    st.markdown('<div class="card">',unsafe_allow_html=True)

    st.markdown("### How are you feeling today?")

    feeling=st.slider("Feeling (0-10)",0,10,7)

    if st.button("Next"):

        st.session_state.feeling_level=feeling
        st.session_state.stage=2
        st.rerun()

    st.markdown('</div>',unsafe_allow_html=True)


# ============================================================
# STAGE 2 PAIN
# ============================================================

elif st.session_state.stage==2:

    st.markdown('<div class="card">',unsafe_allow_html=True)

    pain=st.radio("Do you have pain today?",["No","Yes"])

    if st.button("Next"):

        st.session_state.pain_yesno=(pain=="Yes")

        if pain=="Yes":
            st.session_state.stage=3
        else:
            st.session_state.stage=4

        st.rerun()

    st.markdown('</div>',unsafe_allow_html=True)


# ============================================================
# STAGE 3 BODY MAP
# ============================================================

elif st.session_state.stage==3:

    st.markdown('<div class="card">',unsafe_allow_html=True)

    st.markdown("### Where do you feel pain?")

    col1,col2=st.columns([1,1])

    with col1:
        st.markdown(body_svg(),unsafe_allow_html=True)

    with col2:

        for r in REGIONS:

            if st.button(r):

                if r in st.session_state.selected_parts:
                    st.session_state.selected_parts.remove(r)
                else:
                    st.session_state.selected_parts.add(r)

                st.rerun()

            if r in st.session_state.selected_parts:

                sev=st.slider(f"{r} severity",0,10,5)

                st.session_state.pain_severity[r]=sev

    if st.button("Next"):

        st.session_state.stage=4
        st.rerun()

    st.markdown('</div>',unsafe_allow_html=True)


# ============================================================
# STAGE 4 SYMPTOMS
# ============================================================

elif st.session_state.stage==4:

    st.markdown('<div class="card">',unsafe_allow_html=True)

    st.markdown("### Symptoms today")

    options=[
    "Fatigue",
    "Nausea",
    "Dry mouth",
    "Difficulty swallowing",
    "Hoarseness",
    "Mouth sores",
    "Skin irritation",
    "Loss of taste"
    ]

    cols=st.columns(2)

    for i,s in enumerate(options):

        with cols[i%2]:

            if st.button(s):

                if s in st.session_state.symptoms:
                    st.session_state.symptoms.remove(s)
                else:
                    st.session_state.symptoms.add(s)

                st.rerun()

    if st.button("Submit Check-In"):

        payload={
        "name":st.session_state.name,
        "feeling_level":st.session_state.feeling_level,
        "pain":st.session_state.pain_yesno,
        "pain_locations":list(st.session_state.selected_parts),
        "pain_severity":st.session_state.pain_severity,
        "symptoms":list(st.session_state.symptoms)
        }

        save_to_sheet(payload)

        st.session_state.stage=5
        st.rerun()

    st.markdown('</div>',unsafe_allow_html=True)


# ============================================================
# STAGE 5 SUCCESS
# ============================================================

elif st.session_state.stage==5:

    st.markdown(
    '<div class="success-box">✅ Check-in complete. Thank you!</div>',
    unsafe_allow_html=True
    )

    if st.button("Start another check-in"):

        for k,v in DEFAULTS.items():
            st.session_state[k]=v

        st.rerun()
