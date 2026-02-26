import streamlit as st
import random

# ------------------------------------------------
# Page config
# ------------------------------------------------
st.set_page_config(
    page_title="Virtual Doctor Messenger",
    page_icon="🩺",
    layout="centered"
)

# ------------------------------------------------
# TELEGRAM-LIKE CSS
# ------------------------------------------------
st.markdown("""
<style>

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#eef4ff,#f9fbff);
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}

/* CHAT WRAPPER */
.chat-wrapper {
    max-width: 720px;
    margin: auto;
}

/* MESSAGE ROWS */
.msg-row-left {
    display: flex;
    justify-content: flex-start;
    margin: 6px 0;
}

.msg-row-right {
    display: flex;
    justify-content: flex-end;
    margin: 6px 0;
}

/* DOCTOR BUBBLE (LEFT) */
.bot-bubble {
    background-color: white;
    padding: 12px 16px;
    border-radius: 16px;
    border: 1px solid #e6e6e6;
    max-width: 70%;
    box-shadow: 0px 1px 3px rgba(0,0,0,0.05);
}

/* PATIENT BUBBLE (RIGHT) */
.user-bubble {
    background-color: #4c9aff;
    color: white;
    padding: 12px 16px;
    border-radius: 16px;
    max-width: 70%;
    box-shadow: 0px 1px 3px rgba(0,0,0,0.05);
}

/* TITLE */
.title-box {
    text-align:center;
    font-size:26px;
    font-weight:600;
    padding:10px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Fake Doctor Questions
# ------------------------------------------------
doctor_questions = [
    "How are you feeling today on a scale from 1 to 10?",
    "Are you experiencing any pain today?",
    "Do you feel fatigued or low energy?",
    "Have you noticed nausea or difficulty eating?",
    "Is anything bothering you right now?",
    "Can you describe your main symptom today?"
]

followups = [
    "Thanks for sharing that. Can you tell me more?",
    "Where exactly do you feel it?",
    "How severe would you say it is?",
    "Did this start recently or earlier this week?",
    "Anything else you'd like to add?"
]

# ------------------------------------------------
# Session State
# ------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "doctor", "text": random.choice(doctor_questions)}
    )

# ------------------------------------------------
# Title
# ------------------------------------------------
st.markdown('<div class="title-box">🩺 Virtual Doctor Check-In</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

# ------------------------------------------------
# Render Chat (LEFT / RIGHT layout)
# ------------------------------------------------
for msg in st.session_state.messages:

    if msg["role"] == "doctor":
        st.markdown(
            f"""
            <div class="msg-row-left">
                <div class="bot-bubble">{msg["text"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div class="msg-row-right">
                <div class="user-bubble">{msg["text"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------------------------------------
# Chat Input
# ------------------------------------------------
user_input = st.chat_input("Type your response...")

if user_input:

    # Add patient message (RIGHT)
    st.session_state.messages.append(
        {"role": "user", "text": user_input}
    )

    # Fake continuation logic
    if len(st.session_state.messages) % 3 == 0:
        next_msg = random.choice(doctor_questions)
    else:
        next_msg = random.choice(followups)

    # Add doctor message (LEFT)
    st.session_state.messages.append(
        {"role": "doctor", "text": next_msg}
    )

    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
