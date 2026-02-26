import streamlit as st
import random
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Virtual Doctor Messenger", layout="centered")

# ---------------------------------------------------
# GLASS TELEGRAM STYLE CSS (UPGRADED)
# ---------------------------------------------------
st.markdown("""
<style>

/* Glass hospital dashboard background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#e8f0fb,#dde6f1,#f4f8ff);
}

/* Header */
.header {
    font-size:24px;
    font-weight:600;
    margin-bottom:10px;
}

/* Smooth scrolling chat window */
.chat-wrapper {
    display:flex;
    flex-direction:column;
    gap:10px;
    max-height:65vh;
    overflow-y:auto;
    padding:20px;
    border-radius:18px;
    backdrop-filter: blur(12px);
    background: rgba(255,255,255,0.45);
    border:1px solid rgba(255,255,255,0.6);
}

/* Message rows */
.bot-row { display:flex; align-items:flex-end; justify-content:flex-start; }
.user-row { display:flex; align-items:flex-end; justify-content:flex-end; }

/* Avatar circles */
.avatar {
    width:36px;
    height:36px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:18px;
    margin:0 8px;
    background:white;
    box-shadow:0 1px 4px rgba(0,0,0,0.15);
}

/* Bubbles */
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

/* Sticky input bar like Telegram */
[data-testid="stChatInput"] {
    position:fixed;
    bottom:0;
    left:0;
    right:0;
    padding:12px;
    backdrop-filter: blur(12px);
    background: rgba(255,255,255,0.6);
    border-top:1px solid #e0e0e0;
}

/* Pain slider box */
.glass-card {
    backdrop-filter: blur(10px);
    background: rgba(255,255,255,0.6);
    border-radius:16px;
    padding:15px;
    border:1px solid rgba(255,255,255,0.7);
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

if "show_pain_slider" not in st.session_state:
    st.session_state.show_pain_slider = False

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

    if "pain located" in user_text.lower():
        st.session_state.show_pain_slider = True
        return "Thanks — how severe is the pain?"

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
                <div class="avatar">🩺</div>
                <div class="bot-bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="user-row">
                <div class="user-bubble">{msg["content"]}</div>
                <div class="avatar">🙂</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# HUMAN BODY SELECTOR
# ---------------------------------------------------
if st.session_state.show_body_selector:

    st.markdown('<div class="glass-card">🧍 Select Body Location</div>', unsafe_allow_html=True)

    body_parts = ["Head","Chest","Abdomen","Left Arm","Right Arm","Left Leg","Right Leg"]

    cols = st.columns(3)
    selected = None

    for i, part in enumerate(body_parts):
        if cols[i%3].button(part, key=f"body_{part}"):
            selected = part

    if selected:
        st.session_state.messages.append(
            {"role":"user","content":f"Pain located at: {selected}"}
        )
        st.session_state.show_body_selector = False
        st.session_state.show_pain_slider = True
        st.session_state.messages.append(
            {"role":"bot","content":"Doctor is typing..."}
        )
        st.rerun()

# ---------------------------------------------------
# PAIN SLIDER + SYMPTOM BUTTONS
# ---------------------------------------------------
if st.session_state.show_pain_slider:

    st.markdown('<div class="glass-card">Rate your pain severity</div>', unsafe_allow_html=True)

    severity = st.slider("Pain Level",0,10,5)

    colA,colB,colC = st.columns(3)

    symptoms = ["Sharp","Burning","Constant","Intermittent","Radiating","Pressure"]

    chosen = None
    for i,s in enumerate(symptoms):
        if [colA,colB,colC][i%3].button(s,key=f"sym_{s}"):
            chosen = s

    if st.button("Submit Pain Info"):
        st.session_state.messages.append(
            {"role":"user","content":f"Pain severity {severity}/10 — {chosen if chosen else 'No descriptor'}"}
        )
        st.session_state.show_pain_slider = False

        st.session_state.messages.append(
            {"role":"bot","content":"Doctor is typing..."}
        )
        st.rerun()

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
user_input = st.chat_input("Type your message...")

if user_input:

    st.session_state.messages.append({"role":"user","content":user_input})

    # 🔥 Typing animation
    st.session_state.messages.append({"role":"bot","content":"Doctor is typing..."})
    st.rerun()

# ---------------------------------------------------
# HANDLE TYPING STATE
# ---------------------------------------------------
if len(st.session_state.messages)>0 and st.session_state.messages[-1]["content"]=="Doctor is typing...":

    time.sleep(0.6)

    last_user = ""
    for m in reversed(st.session_state.messages):
        if m["role"]=="user":
            last_user = m["content"]
            break

    reply = virtual_doctor_reply(last_user, st.session_state.step)

    st.session_state.messages[-1] = {"role":"bot","content":reply}

    if "pain today" in reply.lower():
        st.session_state.show_body_selector = True

    st.session_state.step += 1
    st.rerun()

# ---------------------------------------------------
# INITIAL MESSAGE
# ---------------------------------------------------
if len(st.session_state.messages)==0:
    st.session_state.messages.append(
        {"role":"bot","content":"Hello — I'm your virtual doctor. Let's start your check-in."}
    )
    st.rerun()
