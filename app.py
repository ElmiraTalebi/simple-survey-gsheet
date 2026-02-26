import streamlit as st
import random
import time
from streamlit.components.v1 import html

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
    display:flex;
    flex-direction:column;
    gap:8px;
    padding-bottom:80px;
}

.bot-row {display:flex; justify-content:flex-start;}
.user-row {display:flex; justify-content:flex-end;}

.bot-bubble{
    background:white;
    border-radius:18px;
    padding:12px 16px;
    max-width:70%;
    border:1px solid #e0e0e0;
}

.user-bubble{
    background:#0084ff;
    color:white;
    border-radius:18px;
    padding:12px 16px;
    max-width:70%;
}

.header{
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

if "selected_parts" not in st.session_state:
    st.session_state.selected_parts = []

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

    if step < len(intro):
        return intro[step]

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
# HUMAN BODY SILHOUETTE SELECTOR (MULTI SELECT)
# ---------------------------------------------------
if st.session_state.show_body_selector:

    st.markdown("### Select Pain Locations (click multiple)")

    silhouette_html = """
    <style>
    .zone { cursor:pointer; fill:rgba(0,0,255,0.15); }
    .zone:hover { fill:rgba(255,0,0,0.35); }
    </style>

    <svg width="260" height="500" viewBox="0 0 200 500">

        <circle cx="100" cy="40" r="25"
        class="zone"
        onclick="window.location.search='?part=Head'"/>

        <rect x="60" y="70" width="80" height="70"
        class="zone"
        onclick="window.location.search='?part=Chest'"/>

        <rect x="60" y="140" width="80" height="70"
        class="zone"
        onclick="window.location.search='?part=Abdomen'"/>

        <rect x="20" y="80" width="35" height="120"
        class="zone"
        onclick="window.location.search='?part=Left Arm'"/>

        <rect x="145" y="80" width="35" height="120"
        class="zone"
        onclick="window.location.search='?part=Right Arm'"/>

        <rect x="70" y="210" width="25" height="180"
        class="zone"
        onclick="window.location.search='?part=Left Leg'"/>

        <rect x="105" y="210" width="25" height="180"
        class="zone"
        onclick="window.location.search='?part=Right Leg'"/>

    </svg>
    """

    html(silhouette_html, height=520)

    # READ CLICKED PART FROM URL PARAM
    query_params = st.query_params

    if "part" in query_params:
        clicked = query_params["part"]

        if clicked not in st.session_state.selected_parts:
            st.session_state.selected_parts.append(clicked)

        st.query_params.clear()
        st.rerun()

    # SHOW SELECTED PARTS
    if st.session_state.selected_parts:
        st.write("Selected:", ", ".join(st.session_state.selected_parts))

    # SUBMIT BUTTON
    if st.button("Submit Pain Locations"):

        chosen = ", ".join(st.session_state.selected_parts)

        st.session_state.messages.append(
            {"role":"user","content":f"Pain located at: {chosen}"}
        )

        st.session_state.messages.append(
            {"role":"bot","content":"Thanks — how severe is the pain from 0 to 10?"}
        )

        st.session_state.selected_parts = []
        st.session_state.show_body_selector = False

        st.rerun()

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
user_input = st.chat_input("Type your message...")

if user_input:

    st.session_state.messages.append(
        {"role":"user","content":user_input}
    )

    time.sleep(0.4)

    reply = virtual_doctor_reply(user_input, st.session_state.step)

    st.session_state.messages.append(
        {"role":"bot","content":reply}
    )

    # TRIGGER BODY SELECTOR
    if "pain today" in reply.lower():
        st.session_state.show_body_selector = True

    st.session_state.step += 1
    st.rerun()

# ---------------------------------------------------
# INITIAL MESSAGE
# ---------------------------------------------------
if len(st.session_state.messages) == 0:
    st.session_state.messages.append(
        {"role":"bot","content":"Ready when you are. Type your first message."}
    )
    st.rerun()
