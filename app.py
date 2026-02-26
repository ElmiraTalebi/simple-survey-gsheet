import streamlit as st
import random
import time

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Virtual Doctor Prototype",
    layout="centered",
)

# ---------------------------------------------------
# Custom CSS (Nice Background + Chat Style)
# ---------------------------------------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f4f7fb, #e6edf5);
}

.chat-bubble-user {
    background-color: #007AFF;
    color: white;
    padding: 12px;
    border-radius: 15px;
    margin: 5px;
    max-width: 70%;
    align-self: flex-end;
}

.chat-bubble-bot {
    background-color: #ffffff;
    color: black;
    padding: 12px;
    border-radius: 15px;
    margin: 5px;
    max-width: 70%;
    border: 1px solid #e0e0e0;
}

.chat-container {
    display: flex;
    flex-direction: column;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Initialize Session State
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "step" not in st.session_state:
    st.session_state.step = 0

# ---------------------------------------------------
# Fake Doctor Logic (NO API)
# ---------------------------------------------------
def virtual_doctor_reply(user_text, step):

    intro_questions = [
        "Hi, I’m your virtual check-in assistant. How are you feeling today from 0 to 10?",
        "Do you have any pain today? (yes/no)",
        "Any new or worsening symptoms?",
        "Is there anything else you want your care team to know?"
    ]

    followups_yes = [
        "Can you describe where the pain is located?",
        "On a scale from 0-10 how severe is it?",
        "Thanks — I will summarize this for your care team."
    ]

    neutral_followups = [
        "Thanks for sharing. Anything else bothering you today?",
        "Got it. I’m recording that information.",
        "Understood. Let’s continue."
    ]

    # Structured flow (simulate adaptive logic)
    if step < len(intro_questions):
        return intro_questions[step]

    if "yes" in user_text.lower():
        return random.choice(followups_yes)

    return random.choice(neutral_followups)

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.title("🩺 Virtual Doctor Chat Prototype")
st.write("This is a UI prototype — simulated responses only.")

# ---------------------------------------------------
# Chat Display
# ---------------------------------------------------
chat_container = st.container()

with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-bubble-user">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-bubble-bot">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# User Input
# ---------------------------------------------------
user_input = st.chat_input("Type your message...")

# ---------------------------------------------------
# Conversation Flow
# ---------------------------------------------------
if user_input:

    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # Simulated delay (feels realistic)
    time.sleep(0.5)

    # Generate fake doctor reply
    reply = virtual_doctor_reply(
        user_input,
        st.session_state.step
    )

    st.session_state.messages.append(
        {"role": "bot", "content": reply}
    )

    st.session_state.step += 1

    st.rerun()

# ---------------------------------------------------
# Initial Bot Message
# ---------------------------------------------------
if len(st.session_state.messages) == 0:
    first_msg = "Hello — I’m your virtual check-in assistant. Let’s begin."
    st.session_state.messages.append(
        {"role": "bot", "content": first_msg}
    )
    st.rerun()
