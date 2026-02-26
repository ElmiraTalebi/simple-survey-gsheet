import streamlit as st
import datetime
import json
import gspread
from google.oauth2.service_account import Credentials

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Oncology Check-In Prototype", layout="centered")

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
# CSS (Professional Messenger UI)
# ============================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#eef4ff,#f6fbff);
}

/* Login Card */
.login-card{
    max-width:400px;
    margin:100px auto;
    padding:30px;
    border-radius:18px;
    background:rgba(255,255,255,0.85);
    box-shadow:0 10px 30px rgba(0,0,0,0.1);
}

/* Chat Window */
.chat-window{
    max-height:60vh;
    overflow-y:auto;
    padding:20px;
    border-radius:16px;
    background:rgba(255,255,255,0.6);
}

/* Message Rows */
.row-left{display:flex;justify-content:flex-start;margin:10px 0;}
.row-right{display:flex;justify-content:flex-end;margin:10px 0;}

/* Bubbles */
.bubble-doc{
    background:white;
    padding:12px;
    border-radius:18px;
    max-width:70%;
}
.bubble-pat{
    background:#1f7aff;
    color:white;
    padding:12px;
    border-radius:18px;
    max-width:70%;
}

.avatar{margin:0 8px;font-size:18px;}

.panel{
    margin-top:15px;
    padding:15px;
    border-radius:12px;
    background:rgba(255,255,255,0.7);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "stage" not in st.session_state:
    st.session_state.stage = 0

if "selected_parts" not in st.session_state:
    st.session_state.selected_parts = set()

if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

if "feeling_level" not in st.session_state:
    st.session_state.feeling_level = 5

if "pain_yesno" not in st.session_state:
    st.session_state.pain_yesno = None

if "symptoms" not in st.session_state:
    st.session_state.symptoms = []

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def add_doctor(text):
    st.session_state.messages.append({"role":"doctor","content":text})

def add_patient(text):
    st.session_state.messages.append({"role":"patient","content":text})

def toggle_part(part):
    if part in st.session_state.selected_parts:
        st.session_state.selected_parts.remove(part)
    else:
        st.session_state.selected_parts.add(part)

def body_svg(selected):
    def fill(p): return "#1f7aff" if p in selected else "#cfd8e6"
    stroke = "#6b7a90"
    return f"""
<svg width="250" height="450" viewBox="0 0 250 450">
  <circle cx="125" cy="50" r="30" fill="{fill('Head')}" stroke="{stroke}" stroke-width="2"/>
  <rect x="85" y="90" width="80" height="70" rx="20" fill="{fill('Chest')}" stroke="{stroke}" stroke-width="2"/>
  <rect x="90" y="165" width="70" height="60" rx="20" fill="{fill('Abdomen')}" stroke="{stroke}" stroke-width="2"/>
  <rect x="40" y="95" width="40" height="120" rx="20" fill="{fill('Left Arm')}" stroke="{stroke}" stroke-width="2"/>
  <rect x="170" y="95" width="40" height="120" rx="20" fill="{fill('Right Arm')}" stroke="{stroke}" stroke-width="2"/>
  <rect x="95" y="225" width="25" height="150" rx="20" fill="{fill('Left Leg')}" stroke="{stroke}" stroke-width="2"/>
  <rect x="130" y="225" width="25" height="150" rx="20" fill="{fill('Right Leg')}" stroke="{stroke}" stroke-width="2"/>
</svg>
"""

# ============================================================
# LOGIN SCREEN
# ============================================================
if st.session_state.patient_name == "":
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown("## 🩺 Oncology Symptom Check-In")
    st.caption("Research prototype — not a medical device")

    name_input = st.text_input("Patient Name")

    if st.button("Start Check-In"):
        if name_input.strip() == "":
            st.warning("Please enter a patient name.")
        else:
            st.session_state.patient_name = name_input.strip()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ============================================================
# INITIAL MESSAGE
# ============================================================
if len(st.session_state.messages) == 0:
    add_doctor("Hello. Let’s begin your symptom check-in.")
    add_doctor("How are you feeling today from 0 to 10?")

# ============================================================
# CHAT DISPLAY
# ============================================================
st.markdown('<div class="chat-window">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "doctor":
        st.markdown(f"""
        <div class="row-left">
            <div class="avatar">🩺</div>
            <div class="bubble-doc">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="row-right">
            <div class="bubble-pat">{msg["content"]}</div>
            <div class="avatar">🙂</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# STAGE FLOW
# ============================================================

if st.session_state.stage == 0:
    st.markdown('<div class="panel">Feeling Level</div>', unsafe_allow_html=True)
    st.session_state.feeling_level = st.slider("0 = worst, 10 = best", 0, 10, 5)

    if st.button("Submit"):
        add_patient(f"My feeling level is {st.session_state.feeling_level}/10.")
        add_doctor("Do you have any pain today?")
        st.session_state.stage = 1
        st.rerun()

elif st.session_state.stage == 1:
    st.markdown('<div class="panel">Pain?</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    if col1.button("Yes"):
        st.session_state.pain_yesno = True
        add_patient("Yes, I have pain.")
        add_doctor("Select where you feel pain.")
        st.session_state.stage = 2
        st.rerun()

    if col2.button("No"):
        st.session_state.pain_yesno = False
        add_patient("No pain today.")
        add_doctor("Select your symptoms.")
        st.session_state.stage = 3
        st.rerun()

elif st.session_state.stage == 2:
    st.markdown('<div class="panel">Pain Location</div>', unsafe_allow_html=True)
    st.markdown(body_svg(st.session_state.selected_parts), unsafe_allow_html=True)

    parts = ["Head","Chest","Abdomen","Left Arm","Right Arm","Left Leg","Right Leg"]
    for p in parts:
        if st.button(p):
            toggle_part(p)
            st.rerun()

    if st.button("Submit Locations"):
        add_patient("Pain locations: " + ", ".join(st.session_state.selected_parts))
        add_doctor("Select your symptoms.")
        st.session_state.stage = 3
        st.rerun()

elif st.session_state.stage == 3:
    st.markdown('<div class="panel">Symptoms</div>', unsafe_allow_html=True)
    options = ["Fatigue","Nausea","Vomiting","Shortness of breath","Fever"]
    st.session_state.symptoms = st.multiselect("Select symptoms:", options)

    if st.button("Submit Symptoms"):
        add_patient("Symptoms: " + ", ".join(st.session_state.symptoms))
        add_doctor("Anything else you'd like to share?")
        st.session_state.stage = 4
        st.rerun()

elif st.session_state.stage == 4:
    text = st.chat_input("Type your message")
    if text:
        add_patient(text)
        add_doctor("Thank you. Saving your check-in.")
        st.session_state.stage = 5
        st.rerun()

elif st.session_state.stage == 5:

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
        st.success("Check-in saved successfully.")
    except:
        st.error("Failed to save to Google Sheet.")

    if st.button("Start New Check-In"):
        st.session_state.clear()
        st.rerun()
