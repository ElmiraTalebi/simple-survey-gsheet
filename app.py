import streamlit as st
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
.small-note{font-size:12px;color:#666;margin-top:-6px;margin-bottom:8px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">🩺 Virtual Doctor Feature Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="small-note">Demo prototype (no APIs). Click body regions to select/deselect.</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# QUERY PARAM HELPERS (works across Streamlit versions)
# ---------------------------------------------------
def qp_get() -> dict:
    """
    Returns dict of query params. Compatible with both:
    - st.query_params (newer Streamlit)
    - st.experimental_get_query_params (older)
    """
    if hasattr(st, "query_params"):
        # st.query_params behaves like a dict-like object
        return dict(st.query_params)
    return st.experimental_get_query_params()

def qp_clear():
    """
    Clears query params. Compatible with both:
    - st.query_params.clear() (newer)
    - st.experimental_set_query_params() (older)
    """
    if hasattr(st, "query_params"):
        st.query_params.clear()
    else:
        st.experimental_set_query_params()

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
# STAGE 0 — INTRO + SLIDER
# ---------------------------------------------------
if st.session_state.stage == 0:

    if not st.session_state.messages:
        st.session_state.messages.append({"role": "bot", "content": "Hello — I'm your virtual doctor assistant."})
        st.session_state.messages.append({"role": "bot", "content": "How are you feeling today from 0 to 10?"})
        st.rerun()

    feeling = st.slider("Feeling scale", 0, 10, 5)

    if st.button("Submit feeling"):
        st.session_state.messages.append({"role": "user", "content": f"Feeling level: {feeling}"})
        st.session_state.messages.append({"role": "bot", "content": "Do you have any pain today?"})
        st.session_state.stage = 1
        st.rerun()

# ---------------------------------------------------
# STAGE 1 — YES/NO QUICK REPLIES
# ---------------------------------------------------
elif st.session_state.stage == 1:
    col1, col2 = st.columns(2)

    if col1.button("Yes"):
        st.session_state.messages.append({"role": "user", "content": "Yes"})
        st.session_state.messages.append({"role": "bot", "content": "Select where you feel pain (you can click multiple areas)."})
        st.session_state.stage = 2
        st.rerun()

    if col2.button("No"):
        st.session_state.messages.append({"role": "user", "content": "No"})
        st.session_state.messages.append({"role": "bot", "content": "Any new symptoms today?"})
        st.session_state.stage = 3
        st.rerun()

# ---------------------------------------------------
# STAGE 2 — CLICKABLE HUMAN BODY (MULTI-SELECT + VISUAL HIGHLIGHT)
# ---------------------------------------------------
elif st.session_state.stage == 2:

    # 1) If a part was clicked (via ?part=...), toggle selection
    q = qp_get()
    if "part" in q:
        # Depending on Streamlit version, value may be list-like or string
        part_val = q["part"]
        if isinstance(part_val, (list, tuple)):
            part = part_val[0]
        else:
            part = str(part_val)

        if part in st.session_state.selected_parts:
            st.session_state.selected_parts.remove(part)
        else:
            st.session_state.selected_parts.append(part)

        qp_clear()
        st.rerun()

    # 2) Color helper
    def fill(name: str) -> str:
        return "rgba(255,0,0,0.65)" if name in st.session_state.selected_parts else "rgba(0,0,255,0.18)"

    # 3) SVG rendered in MAIN DOM (NOT iframe), using <a href="?part=...">
    silhouette_svg = f"""
    <style>
      .zone {{ cursor: pointer; }}
      .zone:hover {{ filter: brightness(0.92); }}
      .label {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial; font-size: 12px; fill: #444; }}
    </style>

    <div style="display:flex; justify-content:center; margin-top:6px;">
      <svg width="260" height="520" viewBox="0 0 200 520" role="img" aria-label="Human body selector">

        <!-- HEAD -->
        <a href="?part=Head">
          <circle class="zone" cx="100" cy="48" r="24" fill="{fill('Head')}" />
        </a>
        <text class="label" x="100" y="20" text-anchor="middle">Head</text>

        <!-- CHEST -->
        <a href="?part=Chest">
          <rect class="zone" x="60" y="80" width="80" height="70" rx="10" fill="{fill('Chest')}" />
        </a>
        <text class="label" x="100" y="75" text-anchor="middle">Chest</text>

        <!-- ABDOMEN -->
        <a href="?part=Abdomen">
          <rect class="zone" x="60" y="155" width="80" height="70" rx="10" fill="{fill('Abdomen')}" />
        </a>
        <text class="label" x="100" y="150" text-anchor="middle">Abdomen</text>

        <!-- LEFT ARM -->
        <a href="?part=Left Arm">
          <rect class="zone" x="18" y="95" width="36" height="145" rx="12" fill="{fill('Left Arm')}" />
        </a>
        <text class="label" x="36" y="90" text-anchor="middle">L arm</text>

        <!-- RIGHT ARM -->
        <a href="?part=Right Arm">
          <rect class="zone" x="146" y="95" width="36" height="145" rx="12" fill="{fill('Right Arm')}" />
        </a>
        <text class="label" x="164" y="90" text-anchor="middle">R arm</text>

        <!-- LEFT LEG -->
        <a href="?part=Left Leg">
          <rect class="zone" x="72" y="235" width="26" height="220" rx="10" fill="{fill('Left Leg')}" />
        </a>
        <text class="label" x="85" y="470" text-anchor="middle">L leg</text>

        <!-- RIGHT LEG -->
        <a href="?part=Right Leg">
          <rect class="zone" x="102" y="235" width="26" height="220" rx="10" fill="{fill('Right Leg')}" />
        </a>
        <text class="label" x="115" y="470" text-anchor="middle">R leg</text>

        <!-- optional faint torso outline -->
        <rect x="55" y="70" width="90" height="395" rx="22" fill="none" stroke="rgba(0,0,0,0.08)" stroke-width="2"/>
      </svg>
    </div>
    """

    st.markdown(silhouette_svg, unsafe_allow_html=True)

    # 4) Show selected parts clearly
    if st.session_state.selected_parts:
        st.success("Selected: " + ", ".join(st.session_state.selected_parts))
    else:
        st.info("No body parts selected yet. Click one or more areas.")

    colA, colB = st.columns(2)

    with colA:
        if st.button("Clear selections"):
            st.session_state.selected_parts = []
            st.rerun()

    with colB:
        if st.button("Submit Pain Areas"):
            chosen = ", ".join(st.session_state.selected_parts) if st.session_state.selected_parts else "None"

            st.session_state.messages.append({"role": "user", "content": f"Pain at: {chosen}"})
            st.session_state.messages.append({"role": "bot", "content": "Any new symptoms today?"})

            st.session_state.selected_parts = []
            st.session_state.stage = 3
            st.rerun()

# ---------------------------------------------------
# STAGE 3 — MULTI SELECT SYMPTOMS
# ---------------------------------------------------
elif st.session_state.stage == 3:

    symptoms = st.multiselect(
        "Select symptoms",
        ["Fatigue", "Nausea", "Fever", "Shortness of Breath", "None"]
    )

    if st.button("Submit Symptoms"):
        st.session_state.messages.append({"role": "user", "content": ", ".join(symptoms) if symptoms else "No selection"})
        st.session_state.messages.append({"role": "bot", "content": "Anything else you'd like your care team to know?"})
        st.session_state.stage = 4
        st.rerun()

# ---------------------------------------------------
# STAGE 4 — FREE TEXT CHAT
# ---------------------------------------------------
elif st.session_state.stage == 4:

    user_text = st.chat_input("Type message...")

    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        time.sleep(0.25)
        st.session_state.messages.append({"role": "bot", "content": "Thank you. Your check-in is complete."})
        st.session_state.stage = 5
        st.rerun()

# ---------------------------------------------------
# END
# ---------------------------------------------------
elif st.session_state.stage == 5:
    st.success("Demo complete — all features shown.")
