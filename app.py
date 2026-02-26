import streamlit as st
from streamlit.components.v1 import html
import time

# ---------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------
st.set_page_config(page_title="Virtual Doctor Demo", layout="centered")

# ---------------------------------------------------
# CSS - TELEGRAM STYLE
# ---------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#eef2f7,#dde6f1);
}
.chat-wrapper {display:flex;flex-direction:column;gap:8px;padding-bottom:18px;}
.bot-row{display:flex;justify-content:flex-start;}
.user-row{display:flex;justify-content:flex-end;}
.bot-bubble{
    background:white;border-radius:18px;padding:12px 16px;max-width:72%;
    border:1px solid #ddd; box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.user-bubble{
    background:#0084ff;color:white;border-radius:18px;padding:12px 16px;max-width:72%;
    box-shadow: 0 2px 6px rgba(0,0,0,0.10);
}
.header{font-size:22px;font-weight:600;margin-bottom:10px;}
.pill {
    display:inline-block;
    padding:6px 10px;
    margin:4px 6px 0 0;
    border-radius:999px;
    background:#ffffff;
    border:1px solid #d9d9d9;
    font-size:13px;
}
.pill-selected {
    background: rgba(0,132,255,0.12);
    border: 1px solid rgba(0,132,255,0.55);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">🩺 Slobodan Feature Demo (Fixed Body Selector)</div>', unsafe_allow_html=True)

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
# HELPERS
# ---------------------------------------------------
def add_bot(text: str):
    st.session_state.messages.append({"role": "bot", "content": text})

def add_user(text: str):
    st.session_state.messages.append({"role": "user", "content": text})

def read_clicked_part_from_query() -> str | None:
    """
    Streamlit query params can behave like a dict with values sometimes being str or list-like.
    This safely extracts a single string if present.
    """
    qp = st.query_params
    if "part" not in qp:
        return None

    val = qp.get("part")
    # val might be "Head" or ["Head"]
    if isinstance(val, (list, tuple)):
        return val[0] if len(val) > 0 else None
    return val

def render_silhouette(selected_parts: list[str]) -> str:
    """
    Returns an SVG with clickable zones. Selected parts are highlighted.
    Uses window.location.search to set ?part=... which Streamlit reads.
    """
    selected = set(selected_parts)

    def fill(part: str) -> str:
        # highlight selected zones
        return "rgba(0,132,255,0.45)" if part in selected else "rgba(0,0,255,0.18)"

    def stroke(part: str) -> str:
        return "rgba(0,132,255,0.95)" if part in selected else "rgba(0,0,0,0.25)"

    # NOTE: This is a simple silhouette; zones are clickable and highlight on selection.
    return f"""
    <style>
      .zone {{ cursor:pointer; transition: 0.12s ease; }}
      .zone:hover {{ filter: brightness(1.05); }}
      .label {{ font: 12px sans-serif; fill: rgba(0,0,0,0.55); }}
    </style>

    <svg width="290" height="520" viewBox="0 0 220 520">
      <!-- Body outline (very light) -->
      <path d="M110 15
               C90 15, 75 30, 75 50
               C75 70, 90 85, 110 85
               C130 85, 145 70, 145 50
               C145 30, 130 15, 110 15 Z"
            fill="rgba(0,0,0,0.05)"/>

      <path d="M80 95
               C70 105, 65 120, 65 140
               L65 210
               C65 230, 75 245, 90 255
               L90 410
               C90 430, 100 450, 110 450
               C120 450, 130 430, 130 410
               L130 255
               C145 245, 155 230, 155 210
               L155 140
               C155 120, 150 105, 140 95
               C130 85, 90 85, 80 95 Z"
            fill="rgba(0,0,0,0.05)"/>

      <!-- CLICKABLE ZONES -->
      <!-- Head -->
      <circle cx="110" cy="50" r="32"
        class="zone"
        fill="{fill('Head')}"
        stroke="{stroke('Head')}" stroke-width="2"
        onclick="window.location.search='?part=Head'"/>

      <!-- Chest -->
      <rect x="75" y="105" width="70" height="65" rx="14"
        class="zone"
        fill="{fill('Chest')}"
        stroke="{stroke('Chest')}" stroke-width="2"
        onclick="window.location.search='?part=Chest'"/>

      <!-- Abdomen -->
      <rect x="75" y="175" width="70" height="65" rx="14"
        class="zone"
        fill="{fill('Abdomen')}"
        stroke="{stroke('Abdomen')}" stroke-width="2"
        onclick="window.location.search='?part=Abdomen'"/>

      <!-- Left Arm -->
      <rect x="30" y="115" width="38" height="140" rx="14"
        class="zone"
        fill="{fill('Left Arm')}"
        stroke="{stroke('Left Arm')}" stroke-width="2"
        onclick="window.location.search='?part=Left%20Arm'"/>

      <!-- Right Arm -->
      <rect x="152" y="115" width="38" height="140" rx="14"
        class="zone"
        fill="{fill('Right Arm')}"
        stroke="{stroke('Right Arm')}" stroke-width="2"
        onclick="window.location.search='?part=Right%20Arm'"/>

      <!-- Left Leg -->
      <rect x="85" y="255" width="28" height="210" rx="14"
        class="zone"
        fill="{fill('Left Leg')}"
        stroke="{stroke('Left Leg')}" stroke-width="2"
        onclick="window.location.search='?part=Left%20Leg'"/>

      <!-- Right Leg -->
      <rect x="117" y="255" width="28" height="210" rx="14"
        class="zone"
        fill="{fill('Right Leg')}"
        stroke="{stroke('Right Leg')}" stroke-width="2"
        onclick="window.location.search='?part=Right%20Leg'"/>

      <!-- Optional labels (subtle) -->
      <text x="110" y="95" text-anchor="middle" class="label">click areas</text>
    </svg>
    """

# ---------------------------------------------------
# CHAT DISPLAY
# ---------------------------------------------------
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "bot":
        st.markdown(
            f'<div class="bot-row"><div class="bot-bubble">{msg["content"]}</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="user-row"><div class="user-bubble">{msg["content"]}</div></div>',
            unsafe_allow_html=True
        )
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# STAGES
# ---------------------------------------------------

# Stage 0: intro + slider
if st.session_state.stage == 0:
    if not st.session_state.messages:
        add_bot("Hello — I'm your virtual doctor assistant.")
        add_bot("How are you feeling today from 0 to 10?")
        st.rerun()

    feeling = st.slider("Feeling scale", 0, 10, 5)

    if st.button("Submit feeling"):
        add_user(f"Feeling level: {feeling}")
        add_bot("Do you have any pain today?")
        st.session_state.stage = 1
        st.rerun()

# Stage 1: yes/no
elif st.session_state.stage == 1:
    c1, c2 = st.columns(2)
    if c1.button("Yes"):
        add_user("Yes")
        add_bot("Select where you feel pain (click multiple areas).")
        st.session_state.stage = 2
        st.rerun()
    if c2.button("No"):
        add_user("No")
        add_bot("Any new symptoms today?")
        st.session_state.stage = 3
        st.rerun()

# Stage 2: body selector with visual highlight + multi-select
elif st.session_state.stage == 2:
    st.markdown("### Body map (multi-select)")

    # 1) If user clicked a zone, record it
    clicked = read_clicked_part_from_query()
    if clicked:
        clicked = clicked.replace("%20", " ")
        if clicked not in st.session_state.selected_parts:
            st.session_state.selected_parts.append(clicked)
        # clear URL params so you can click multiple times
        st.query_params.clear()
        st.rerun()

    # 2) Render silhouette with current selections highlighted
    html(render_silhouette(st.session_state.selected_parts), height=540)

    # 3) Show selections as pills (visual confirmation)
    if st.session_state.selected_parts:
        pills = []
        for p in st.session_state.selected_parts:
            pills.append(f'<span class="pill pill-selected">{p}</span>')
        st.markdown("".join(pills), unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill">No areas selected yet</span>', unsafe_allow_html=True)

    # 4) Controls: clear / submit
    colA, colB = st.columns(2)

    if colA.button("Clear selection"):
        st.session_state.selected_parts = []
        st.rerun()

    if colB.button("Submit Pain Areas"):
        if not st.session_state.selected_parts:
            st.warning("Please select at least one body area.")
        else:
            chosen = ", ".join(st.session_state.selected_parts)
            add_user(f"Pain at: {chosen}")
            add_bot("Thanks — how severe is the pain from 0 to 10?")
            st.session_state.selected_parts = []
            st.session_state.stage = 3
            st.rerun()

# Stage 3: symptoms + free text wrap-up
elif st.session_state.stage == 3:
    symptoms = st.multiselect(
        "Select symptoms",
        ["Fatigue", "Nausea", "Fever", "Shortness of Breath", "None"]
    )
    if st.button("Submit Symptoms"):
        add_user("Symptoms: " + (", ".join(symptoms) if symptoms else "None selected"))
        add_bot("Anything else you'd like your care team to know?")
        st.session_state.stage = 4
        st.rerun()

elif st.session_state.stage == 4:
    user_text = st.chat_input("Type message...")
    if user_text:
        add_user(user_text)
        time.sleep(0.25)
        add_bot("Thank you. Your check-in is complete.")
        st.session_state.stage = 5
        st.rerun()

elif st.session_state.stage == 5:
    st.success("Demo complete.")
    if st.button("Restart demo"):
        st.session_state.messages = []
        st.session_state.stage = 0
        st.session_state.selected_parts = []
        st.query_params.clear()
        st.rerun()
