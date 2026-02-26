import streamlit as st
import random
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Virtual Doctor Messenger",
    layout="centered"
)

# ---------------------------------------------------
# TELEGRAM STYLE CSS
# ---------------------------------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#eef2f7,#dde6f1);
}

/* CHAT AREA */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-bottom: 80px;
}

/* LEFT MESSAGE (DOCTOR) */
.bot-row {
    display: flex;
    justify-content: flex-start;
}

.bot-bubble {
    background: white;
    border-radius: 18px;
    padding: 12px 16px;
    max-width: 70%;
    border: 1px solid #e0e0e0;
    font-size: 15px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

/* RIGHT MESSAGE (PATIENT) */
.user-row {
    display: flex;
    justify-content: flex-end;
}

.user-bubble {
    background: #0084ff;
    color: white;
    border-radius: 18px;
    padding: 12px 16px;
    max-width: 70%;
    font-size: 15px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

.header {
    font-size:22px;
    font-weight:600;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "step" not in st.session_state:
    st.session_state.step = 0

# ---------------------------------------------------
# FAKE VIRTUAL DOCTOR LOGIC (NO API)
# ---------------------------------------------------
def virtual_doctor_reply(user_text, step):

    intro = [
        "Hi — I'm your virtual doctor check-in assistant.",
        "How are you feeling today from 0 to 10?",
        "Do you have any pain today? (yes/no)",
        "Any new or worsening symptoms?",
        "Anything else you'd like your care team to know?"
    ]

    followups_yes = [
        "Where is the pain located?",
        "How severe is it from 0 to 10?",
        "Thanks — I'm recording this for your team."
    ]

    neutral = [
        "Understood.",
        "Thanks for sharing.",
        "Got it — anything else?"
    ]

    if step < len(intro):
        return intro[step]

    if "yes" in user_text.lower():
        return random.choice(followups_yes)

    return random.choice(neutral)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<div class="header">🩺 Virtual Doctor Messenger</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# CHAT DISPLAY
# ---------------------------------------------------
chat_area = st.container()

with chat_area:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

    for msg in st.session_state.messages:

        if msg["role"] == "bot":
            st.markdown(
                f"""
                <div class="bot-row">
                    <div class="bot-bubble">
                        {msg["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f"""
                <div class="user-row">
                    <div class="user-bubble">
                        {msg["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
user_input = st.chat_input("Type your message...")

# ---------------------------------------------------
# CHAT FLOW
# ---------------------------------------------------
if user_input:

    # add patient message (RIGHT)
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    time.sleep(0.4)

    reply = virtual_doctor_reply(
        user_input,
        st.session_state.step
    )

    # add doctor message (LEFT)
    st.session_state.messages.append(
        {"role": "bot", "content": reply}
    )

    st.session_state.step += 1
    st.rerun()

# ---------------------------------------------------
# INITIAL MESSAGE
# ---------------------------------------------------
if len(st.session_state.messages) == 0:
    st.session_state.messages.append(
        {"role": "bot",
         "content": "Hello — I'm your virtual doctor. Let's start your check-in."}
    )
    st.rerun()
