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
.chat-wrapper {display:flex;flex-direction:column;gap:8px;padding-bottom:10px;}
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
.smallhint { color: rgba(0,0,0,0.55); font-size: 13px; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">🩺 Slobodan Feature Demo — Clickable Body Map (Works)</div>', unsafe_allow_html=True)

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

def toggle_part(part: str):
    # multi-select toggle
    if part in st.session_state.selected_parts:
        st.session_state.selected_parts.remove(part)
    else:
        st.session_state.selected_parts.append(part)

def _get_part_from_query():
    qp = st.query_params
    if "part" not in qp:
        return None
    val = qp.get("part")
    if isinstance(val, (list, tuple)):
        return val[0] if val else None
    return val

def render_clickable_silhouette(selected_parts: list[str]) -> str:
    """
    IMPORTANT:
    - Each zone is wrapped in an <a href="/?part=..." target="_top"> ... </a>
    - target="_top" navigates the main Streamlit page (not the iframe),
      so Streamlit receives the query param and can record selection.
    """
    sel = set(selected_parts)

    def fill(part: str) -> str:
        return "rgba(0,132,255,0.55)" if part in sel else "rgba(0,0,255,0.18)"

    def stroke(part: str) -> str:
        return "rgba(0,132,255,0.95)" if part in sel else "rgba(0,0,0,0.25)"

    # Use href values without spaces (URL-encoded)
    return f"""
    <style>
      .zone {{ cursor:pointer; transition: 0.12s ease; }}
      .zone:hover {{ filter: brightness(1.08); }}
      .hint {{ font: 12px sans-serif; fill: rgba(0,0,0,0.45); }}
      a {{ text-decoration:none; }}
    </style>

    <svg width="290" height="520" viewBox="0 0 220 520" xmlns="http://www.w3.org/2000/svg">

      <!-- faint silhouette -->
      <path d="M110 18
               C90 18, 74 34, 74 54
               C74 74, 90 90, 110 90
               C130 90, 146 74, 146 54
               C146 34, 130 18, 110 18 Z"
            fill="rgba(0,0,0,0.05)"/>
      <path d="M82 98
               C72 110, 68 126, 68 144
               L68 220
               C68 242, 80 258, 96 268
               L96 414
               C96 438, 104 462, 110 462
               C116 462, 124 438, 124 414
               L124 268
               C140 258, 152 242, 152 220
               L152 144
               C152 126, 148 110, 138 98
               C128 88, 92 88, 82 98 Z"
            fill="rgba(0,0,0,0.05)"/>

      <!-- CLICKABLE ZONES (navigate TOP window) -->
      <a href="/?part=Head" target="_top">
        <circle cx="110" cy="54" r="34"
                class="zone"
                fill="{fill('Head')}" stroke="{stroke('Head')}" stroke-width="2"/>
      </a>

      <a href="/?part=Chest" target="_top">
        <rect x="76" y="110" width="68" height="62" rx="14"
              class="zone"
              fill="{fill('Chest')}" stroke="{stroke('Chest')}" stroke-width="2"/>
      </a>

      <a href="/?part=Abdomen" target="_top">
        <rect x="76" y="176" width="68" height="62" rx="14"
              class="zone"
              fill="{fill('Abdomen')}" stroke="{stroke('Abdomen')}" stroke-width="2"/>
      </a>

      <a href="/?part=Left%20Arm" target="_top">
        <rect x="28" y="122" width="38" height="140" rx="16"
              class="zone"
              fill="{fill('Left Arm')}" stroke="{stroke('Left Arm')}" stroke-width="2"/>
      </a>

      <a href="/?part=Right%20Arm" target="_top">
        <rect x="154" y="122" width="38" height="140" rx="16"
              class="zone"
              fill="{fill('Right Arm')}" stroke="{stroke('Right Arm')}" stroke-width="2"/>
      </a>

      <a href="/?part=Left%20Leg" target="_top">
        <rect x="86" y="266" width="28" height="210" rx="14"
              class="zone"
              fill="{fill('Left Leg')}" stroke="{stroke('Left Leg')}" stroke-width="2"/>
      </a>

      <a href="/?part=Right%20Leg" target="_top">
        <rect x="118" y="266" width="28" height="210" rx="14"
              class="zone"
              fill="{fill('Right Leg')}" stroke="{stroke('Right Leg')}" stroke-width="2"/>
      </a>

      <text x="110" y="505" text-anchor="middle" class="hint">
        click zones to toggle selection
      </text>
    </svg>
    """

# ---------------------------------------------------
# CHAT DISPLAY
# ---------------------------------------------------
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "bot":
        st.markdown(f'<div class="bot-row"><div class="bot-bubble">{msg["content"]}</div></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="user-row"><div class="user-bubble">{msg["content"]}</div></div>',
                    unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# STAGES
# ---------------------------------------------------

# Stage 0: intro + feeling slider
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

# Stage 1: yes/no quick reply
elif st.session_state.stage == 1:
    c1, c2 = st.columns(2)
    if c1.button("Yes"):
        add_user("Yes")
        add_bot("Select where you feel pain (click multiple body areas).")
        st.session_state.stage = 2
        st.rerun()
    if c2.button("No"):
        add_user("No")
        add_bot("Any new symptoms today?")
        st.session_state.stage = 3
        st.rerun()

# Stage 2: CLICKABLE silhouette zones (works) + highlight + multi-select
elif st.session_state.stage == 2:
    st.markdown("### Body map (multi-select)")
    st.markdown('<div class="smallhint">Click a zone to toggle it. Selected zones become blue.</div>',
                unsafe_allow_html=True)

    # 1) If we navigated with ?part=..., record it
    clicked = _get_part_from_query()
    if clicked:
        clicked = clicked.replace("%20", " ")
        toggle_part(clicked)

        # Remove query param from URL so repeated clicks work cleanly
        st.query_params.clear()
        st.rerun()

    # 2) Render clickable silhouette with highlights
    html(render_clickable_silhouette(st.session_state.selected_parts), height=540)

    # 3) Show selected parts as pills
    if st.session_state.selected_parts:
        pills = "".join([f'<span class="pill pill-selected">{p}</span>' for p in st.session_state.selected_parts])
        st.markdown(pills, unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill">No areas selected yet</span>', unsafe_allow_html=True)

    # 4) Controls
    ca, cb = st.columns(2)

    if ca.button("Clear selection", use_container_width=True):
        st.session_state.selected_parts = []
        st.rerun()

    if cb.button("Submit Pain Areas", use_container_width=True):
        if not st.session_state.selected_parts:
            st.warning("Please select at least one body area.")
        else:
            chosen = ", ".join(st.session_state.selected_parts)
            add_user(f"Pain at: {chosen}")
            add_bot("Thanks — how severe is the pain from 0 to 10?")
            st.session_state.selected_parts = []
            st.session_state.stage = 3
            st.rerun()

# Stage 3: symptoms
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

# Stage 4: free text wrap-up
elif st.session_state.stage == 4:
    user_text = st.chat_input("Type message...")
    if user_text:
        add_user(user_text)
        time.sleep(0.25)
        add_bot("Thank you. Your check-in is complete.")
        st.session_state.stage = 5
        st.rerun()

# Stage 5: end
elif st.session_state.stage == 5:
    st.success("Demo complete.")
    if st.button("Restart demo"):
        st.session_state.messages = []
        st.session_state.stage = 0
        st.session_state.selected_parts = []
        st.query_params.clear()
        st.rerun()
