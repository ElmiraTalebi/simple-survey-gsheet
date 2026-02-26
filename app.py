import streamlit as st
import random
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Virtual Doctor Messenger", layout="centered")

# ---------------------------------------------------
# TELEGRAM STYLE CSS
# ---------------------------------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#eef2f7,#dde6f1);
}

.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-bottom: 80px;
}

.bot-row { display:flex; justify-content:flex-start; }
.user-row { display:flex; justify-content:flex-end; }

.bot-bubble {
    background:white;
    border-radius:18px;
    padding:12px 16px;
    max-width:70%;
    border:1px solid #e0e0e0;
}

.user-bubble {
    background:#0084ff;
    color:white;
    border-radius:18px;
    padding:12px 16px;
    max-width:70%;
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

if "show_body_selector" not in st.session_state:
    st.session_state.show_body_selector = False

# ---------------------------------------------------
# FAKE DOCTOR LOGIC
# ---------------------------------------------------
def virtual_doctor_reply(user_text, step):

    intro = [
        "Hi — I'm your virtual doctor check-in assistant.",
        "How are you feeling today from 0 to 10?",
        "Do you have any pain today? (yes/no)"
    ]

    neutral = [
        "Thanks for sharing.",
        "Got it — anything else?",
        "Understood."
    ]

    # STEP LOGIC
    if step < len(intro):
        return intro[step]

    if "pain" in user_text.lower():
        return "Please select where you feel pain on the body below."

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
            st.markdown(f"""
            <div class="bot-row">
                <div class="bot-bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="user-row">
                <div class="user-bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# HUMAN BODY SELECTOR UI (appears inside chat)
# ---------------------------------------------------
if st.session_state.show_body_selector:

    st.markdown("### Select Body Location")

    col1, col2, col3 = st.columns(3)

    body_parts = [
        "Head", "Chest", "Abdomen",
        "Left Arm", "Right Arm",
        "Left Leg", "Right Leg"
    ]

    selected = None

    for i, part in enumerate(body_parts):
        if st.button(part, key=f"body_{part}"):
            selected = part

    if selected:
        # add patient message
        st.session_state.messages.append(
            {"role":"user", "content":f"Pain located at: {selected}"}
        )

        st.session_state.show_body_selector = False

        # doctor followup
        st.session_state.messages.append(
            {"role":"bot", "content":"Thanks — how severe is the pain from 0 to 10?"}
        )

        st.rerun()

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
user_input = st.chat_input("Type your message...")

if user_input:

    st.session_state.messages.append({"role":"user","content":user_input})

    time.sleep(0.4)

    reply = virtual_doctor_reply(user_input, st.session_state.step)

    st.session_state.messages.append({"role":"bot","content":reply})

    # 🔥 Trigger body selector when pain question happens
    if "pain today" in reply.lower():
        st.session_state.show_body_selector = True

    st.session_state.step += 1
    st.rerun()

# ---------------------------------------------------
# INITIAL MESSAGE
# ---------------------------------------------------
if len(st.session_state.messages) == 0:
    st.session_state.messages.append(
        {"role":"bot","content":"Hello — I'm your virtual doctor. Let's start your check-in."}
    )
    st.rerun()
