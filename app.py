import streamlit as st
import random

# ------------------------------------------------
# Page config
# ------------------------------------------------
st.set_page_config(page_title="Virtual Doctor", page_icon="🩺", layout="centered")

# ------------------------------------------------
# CSS (Telegram-style)
# ------------------------------------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#eef4ff,#f9fbff);
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}
.chat-wrapper {
    max-width:720px;
    margin:auto;
}
.msg-row-left {display:flex; justify-content:flex-start; margin:6px 0;}
.msg-row-right {display:flex; justify-content:flex-end; margin:6px 0;}
.bot-bubble {
    background:white;
    padding:12px 16px;
    border-radius:16px;
    border:1px solid #e6e6e6;
    max-width:70%;
}
.user-bubble {
    background:#4c9aff;
    color:white;
    padding:12px 16px;
    border-radius:16px;
    max-width:70%;
}
.title-box {
    text-align:center;
    font-size:26px;
    font-weight:600;
    padding:10px;
}
.selector-box{
    background:white;
    border-radius:12px;
    padding:15px;
    border:1px solid #e6e6e6;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Doctor questions
# ------------------------------------------------
doctor_questions = [
    "How are you feeling today on a scale from 1 to 10?",
    "Are you experiencing any pain today?",
    "Do you feel fatigued or low energy?",
    "Have you noticed nausea or difficulty eating?",
]

followups = [
    "Thanks for sharing that. Can you tell me more?",
    "Where exactly do you feel it?",
    "How severe would you say it is?",
]

# ------------------------------------------------
# Session state
# ------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "doctor", "text": random.choice(doctor_questions)}
    ]

# ------------------------------------------------
# Helper: detect pain question
# ------------------------------------------------
def last_doctor_asked_pain():
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "doctor":
            return "pain" in msg["text"].lower()
    return False

# ------------------------------------------------
# Title
# ------------------------------------------------
st.markdown('<div class="title-box">🩺 Virtual Doctor Check-In</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

# ------------------------------------------------
# Render chat history
# ------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "doctor":
        st.markdown(f"""
        <div class="msg-row-left">
            <div class="bot-bubble">{msg["text"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-row-right">
            <div class="user-bubble">{msg["text"]}</div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------
# BODY SELECTOR UI (appears only when pain asked)
# ------------------------------------------------
show_selector = last_doctor_asked_pain()

if show_selector:
    st.markdown('<div class="selector-box">🧍 Select where you feel pain:</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🧠 Head"):
            selected = "Pain in Head"
    with col2:
        if st.button("💪 Arm"):
            selected = "Pain in Arm"
    with col3:
        if st.button("🫀 Chest"):
            selected = "Pain in Chest"

    col4, col5, col6 = st.columns(3)
    with col4:
        if st.button("🫁 Back"):
            selected = "Pain in Back"
    with col5:
        if st.button("🦵 Leg"):
            selected = "Pain in Leg"
    with col6:
        if st.button("🦶 Foot"):
            selected = "Pain in Foot"

    # If any body part selected
    if "selected" in locals():
        st.session_state.messages.append({"role":"user","text":selected})

        next_msg = random.choice(followups)
        st.session_state.messages.append({"role":"doctor","text":next_msg})

        st.rerun()

# ------------------------------------------------
# Normal text chat input
# ------------------------------------------------
user_input = st.chat_input("Type your response...")

if user_input:
    st.session_state.messages.append({"role":"user","text":user_input})

    # fake continuation
    if len(st.session_state.messages) % 3 == 0:
        next_msg = random.choice(doctor_questions)
    else:
        next_msg = random.choice(followups)

    st.session_state.messages.append({"role":"doctor","text":next_msg})
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
