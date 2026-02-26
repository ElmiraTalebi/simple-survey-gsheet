import streamlit as st
from streamlit.components.v1 import html
import time

# ---------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------
st.set_page_config(page_title="Slobodan Demo", layout="centered")

# ---------------------------------------------------
# CSS (Messenger UI)
# ---------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#eef2f7,#dde6f1);
}
.chat-wrapper {display:flex;flex-direction:column;gap:8px;}
.bot-row{display:flex;justify-content:flex-start;}
.user-row{display:flex;justify-content:flex-end;}
.bot-bubble{background:white;border-radius:18px;padding:12px 16px;max-width:70%;border:1px solid #ddd;}
.user-bubble{background:#0084ff;color:white;border-radius:18px;padding:12px 16px;max-width:70%;}
.header{font-size:22px;font-weight:600;margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">🩺 Virtual Doctor Feature Demo</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "stage" not in st.session_state:
    st.session_state.stage = 0

if "selected_parts" not in st.session_state:
    st.session_state.selected_parts = []

# ---------------------------------------------------
# CHAT DISPLAY
# ---------------------------------------------------
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "bot":
        st.markdown(f'<div class="bot-row"><div class="bot-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="user-row"><div class="user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# STAGE 0 — INTRO + SLIDER (STRUCTURED SCALE)
# ---------------------------------------------------
if st.session_state.stage == 0:

    if not st.session_state.messages:
        st.session_state.messages.append({"role":"bot","content":"Hello — I'm your virtual doctor assistant."})
        st.session_state.messages.append({"role":"bot","content":"How are you feeling today from 0 to 10?"})
        st.rerun()

    feeling = st.slider("Feeling scale",0,10,5)

    if st.button("Submit feeling"):
        st.session_state.messages.append({"role":"user","content":f"Feeling level: {feeling}"})
        st.session_state.messages.append({"role":"bot","content":"Do you have any pain today?"})
        st.session_state.stage = 1
        st.rerun()

# ---------------------------------------------------
# STAGE 1 — YES/NO QUICK REPLIES
# ---------------------------------------------------
elif st.session_state.stage == 1:

    col1,col2 = st.columns(2)

    if col1.button("Yes"):
        st.session_state.messages.append({"role":"user","content":"Yes"})
        st.session_state.messages.append({"role":"bot","content":"Select where you feel pain on the body."})
        st.session_state.stage = 2
        st.rerun()

    if col2.button("No"):
        st.session_state.messages.append({"role":"user","content":"No"})
        st.session_state.messages.append({"role":"bot","content":"Any new symptoms today?"})
        st.session_state.stage = 3
        st.rerun()

# ---------------------------------------------------
# STAGE 2 — CLICKABLE HUMAN BODY (MULTI SELECT)
# ---------------------------------------------------
elif st.session_state.stage == 2:

    st.write("### Select Pain Locations")

    query = st.query_params

    # HANDLE CLICK
    if "part" in query:
        part = query["part"]

        if part in st.session_state.selected_parts:
            st.session_state.selected_parts.remove(part)
        else:
            st.session_state.selected_parts.append(part)

        st.query_params.clear()
        st.rerun()

    # COLOR FUNCTION
    def color(name):
        if name in st.session_state.selected_parts:
            return "rgba(255,0,0,0.6)"
        return "rgba(0,0,255,0.2)"

    silhouette_html = f"""
    <svg width="260" height="500" viewBox="0 0 200 500">

        <circle cx="100" cy="40" r="25"
        fill="{color('Head')}"
        onclick="window.location.search='?part=Head'"/>

        <rect x="60" y="70" width="80" height="70"
        fill="{color('Chest')}"
        onclick="window.location.search='?part=Chest'"/>

        <rect x="60" y="140" width="80" height="70"
        fill="{color('Abdomen')}"
        onclick="window.location.search='?part=Abdomen'"/>

        <rect x="20" y="80" width="35" height="120"
        fill="{color('Left Arm')}"
        onclick="window.location.search='?part=Left Arm'"/>

        <rect x="145" y="80" width="35" height="120"
        fill="{color('Right Arm')}"
        onclick="window.location.search='?part=Right Arm'"/>

        <rect x="70" y="210" width="25" height="180"
        fill="{color('Left Leg')}"
        onclick="window.location.search='?part=Left Leg'"/>

        <rect x="105" y="210" width="25" height="180"
        fill="{color('Right Leg')}"
        onclick="window.location.search='?part=Right Leg'"/>

    </svg>
    """

    html(silhouette_html, height=520)

    if st.session_state.selected_parts:
        st.success("Selected: " + ", ".join(st.session_state.selected_parts))
    else:
        st.info("No body parts selected.")

    if st.button("Submit Pain Areas"):

        chosen = ", ".join(st.session_state.selected_parts)

        st.session_state.messages.append({"role":"user","content":f"Pain at {chosen}"})
        st.session_state.messages.append({"role":"bot","content":"Any new symptoms today?"})

        st.session_state.selected_parts = []
        st.session_state.stage = 3
        st.rerun()

# ---------------------------------------------------
# STAGE 3 — MULTI SELECT SYMPTOMS
# ---------------------------------------------------
elif st.session_state.stage == 3:

    symptoms = st.multiselect(
        "Select symptoms",
        ["Fatigue","Nausea","Fever","Shortness of Breath","None"]
    )

    if st.button("Submit Symptoms"):
        st.session_state.messages.append({"role":"user","content":", ".join(symptoms)})
        st.session_state.messages.append({"role":"bot","content":"Anything else you'd like your care team to know?"})
        st.session_state.stage = 4
        st.rerun()

# ---------------------------------------------------
# STAGE 4 — FREE TEXT CHAT (CONVERSATIONAL)
# ---------------------------------------------------
elif st.session_state.stage == 4:

    user_text = st.chat_input("Type message...")

    if user_text:
        st.session_state.messages.append({"role":"user","content":user_text})
        time.sleep(0.3)
        st.session_state.messages.append({"role":"bot","content":"Thank you. Your check-in is complete."})
        st.session_state.stage = 5
        st.rerun()

# ---------------------------------------------------
# END
# ---------------------------------------------------
elif st.session_state.stage == 5:
    st.success("Have a nice day!")
