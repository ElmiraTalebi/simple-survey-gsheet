import streamlit as st
from typing import Dict, List, Set

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Cancer Symptom Check-In Prototype", page_icon="🩺", layout="centered")

# ============================================================
# CSS — messenger look + soft medical background
# ============================================================
st.markdown(
    """
<style>
/* Soft medical background */
[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#eef4ff,#f6fbff);
}

/* Header */
.header{
    font-size: 24px;
    font-weight: 700;
    margin: 8px 0 14px 0;
}

/* Chat window */
.chat-shell{
    max-width: 840px;
    margin: 0 auto;
}

.chat-window{
    max-height: 62vh;
    overflow-y: auto;
    padding: 18px 14px;
    border-radius: 18px;
    background: rgba(255,255,255,0.55);
    border: 1px solid rgba(200,210,230,0.55);
    backdrop-filter: blur(10px);
}

/* Message rows */
.row-left{ display:flex; justify-content:flex-start; align-items:flex-end; margin: 10px 0; gap: 10px; }
.row-right{ display:flex; justify-content:flex-end; align-items:flex-end; margin: 10px 0; gap: 10px; }

/* Avatars */
.avatar{
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display:flex;
    justify-content:center;
    align-items:center;
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(210,220,240,0.9);
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    font-size: 18px;
    flex: 0 0 auto;
}

/* Bubbles */
.bubble-doc{
    background: #ffffff;
    border: 1px solid rgba(220,225,235,0.95);
    border-radius: 18px;
    padding: 12px 14px;
    max-width: 72%;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.bubble-pat{
    background: #1f7aff;
    color: white;
    border-radius: 18px;
    padding: 12px 14px;
    max-width: 72%;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}

.small-note{
    color: rgba(0,0,0,0.55);
    font-size: 12px;
    margin-top: 6px;
}

/* Stage panels */
.panel{
    margin-top: 14px;
    padding: 14px;
    border-radius: 16px;
    background: rgba(255,255,255,0.65);
    border: 1px solid rgba(200,210,230,0.55);
    backdrop-filter: blur(10px);
}

.panel-title{
    font-weight: 700;
    margin-bottom: 10px;
}

/* Make default button spacing nicer */
.stButton>button{
    border-radius: 14px;
    padding: 0.55rem 0.9rem;
}

/* Sticky input feel (Streamlit chat_input sits at bottom already; we just style a bit) */
[data-testid="stChatInput"]{
    position: sticky;
    bottom: 0;
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(10px);
    border-top: 1px solid rgba(200,210,230,0.55);
    padding-top: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

if "stage" not in st.session_state:
    st.session_state.stage = 0  # 0..4

if "selected_parts" not in st.session_state:
    st.session_state.selected_parts: Set[str] = set()

if "pain_yesno" not in st.session_state:
    st.session_state.pain_yesno = None  # True/False/None

if "feeling_level" not in st.session_state:
    st.session_state.feeling_level = 5

if "symptoms" not in st.session_state:
    st.session_state.symptoms: List[str] = []

# ============================================================
# HELPERS
# ============================================================
def add_doctor(text: str) -> None:
    st.session_state.messages.append({"role": "doctor", "content": text})

def add_patient(text: str) -> None:
    st.session_state.messages.append({"role": "patient", "content": text})

def ensure_stage_prompt() -> None:
    """Make sure the doctor has asked the current stage question (once)."""
    stage = st.session_state.stage
    if len(st.session_state.messages) == 0:
        add_doctor("Hi — I’m your virtual doctor check-in assistant. Let’s do a quick symptom check-in.")
        return

    # If last message is already a doctor prompt for the stage, do nothing.
    # We use a simple marker check based on stage number and key phrasing.
    last_doc = None
    for m in reversed(st.session_state.messages):
        if m["role"] == "doctor":
            last_doc = m["content"]
            break

    prompts = {
        0: "How are you feeling today from 0 to 10?",
        1: "Do you have any pain today?",
        2: "Please select where you feel pain on the body.",
        3: "Which symptoms are you experiencing today? (Select all that apply.)",
        4: "Anything else you want to tell me in your own words?",
    }

    want = prompts.get(stage, None)
    if not want:
        return

    if last_doc is None or want not in last_doc:
        add_doctor(want)

def toggle_body_part(part: str) -> None:
    if part in st.session_state.selected_parts:
        st.session_state.selected_parts.remove(part)
    else:
        st.session_state.selected_parts.add(part)

def body_svg(selected: Set[str]) -> str:
    """
    Simple human silhouette made of separate SVG regions.
    Regions highlight blue if selected.
    """
    def fill(part: str) -> str:
        return "#1f7aff" if part in selected else "#cfd8e6"  # blue vs light gray

    stroke = "#6b7a90"
    # A clean, simple silhouette (not anatomically perfect, but clear + regioned)
    return f"""
<svg width="320" height="520" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="rgba(0,0,0,0.15)"/>
    </filter>
  </defs>

  <!-- HEAD -->
  <g filter="url(#shadow)">
    <circle cx="160" cy="70" r="38" fill="{fill('Head')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <!-- CHEST -->
  <g filter="url(#shadow)">
    <rect x="110" y="120" width="100" height="70" rx="24" fill="{fill('Chest')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <!-- ABDOMEN -->
  <g filter="url(#shadow)">
    <rect x="115" y="195" width="90" height="70" rx="22" fill="{fill('Abdomen')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <!-- LEFT ARM -->
  <g filter="url(#shadow)">
    <path d="M110 132
             C 80 145, 72 180, 78 220
             C 82 250, 92 270, 100 290
             C 108 310, 115 320, 120 320
             L 120 130 Z"
          fill="{fill('Left Arm')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <!-- RIGHT ARM -->
  <g filter="url(#shadow)">
    <path d="M210 132
             C 240 145, 248 180, 242 220
             C 238 250, 228 270, 220 290
             C 212 310, 205 320, 200 320
             L 200 130 Z"
          fill="{fill('Right Arm')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <!-- LEFT LEG -->
  <g filter="url(#shadow)">
    <path d="M135 265
             C 120 310, 118 360, 126 410
             C 132 445, 132 475, 128 500
             L 155 500
             C 158 470, 160 435, 156 405
             C 150 355, 152 312, 165 265 Z"
          fill="{fill('Left Leg')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <!-- RIGHT LEG -->
  <g filter="url(#shadow)">
    <path d="M185 265
             C 200 310, 202 360, 194 410
             C 188 445, 188 475, 192 500
             L 165 500
             C 162 470, 160 435, 164 405
             C 170 355, 168 312, 155 265 Z"
          fill="{fill('Right Leg')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <!-- Label line -->
  <text x="160" y="520" text-anchor="middle" font-size="12" fill="rgba(0,0,0,0.55)">
    Click buttons to toggle regions (highlights update live)
  </text>
</svg>
""".strip()

# ============================================================
# HEADER + ENSURE PROMPT
# ============================================================
st.markdown('<div class="chat-shell"><div class="header">🩺 Cancer Symptom Check-In (Research Prototype)</div>', unsafe_allow_html=True)
ensure_stage_prompt()

# ============================================================
# CHAT RENDER
# ============================================================
st.markdown('<div class="chat-window">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "doctor":
        st.markdown(
            f"""
            <div class="row-left">
              <div class="avatar">🩺</div>
              <div class="bubble-doc">{msg["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="row-right">
              <div class="bubble-pat">{msg["content"]}</div>
              <div class="avatar">🙂</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("</div></div>", unsafe_allow_html=True)

# ============================================================
# STAGE PANELS (step-based conversation)
# ============================================================
stage = st.session_state.stage

# -------------------------------
# Stage 0 — feeling slider
# -------------------------------
if stage == 0:
    st.markdown('<div class="panel"><div class="panel-title">Stage 0 · Feeling level</div>', unsafe_allow_html=True)
    st.session_state.feeling_level = st.slider("Feeling (0 = worst, 10 = best)", 0, 10, int(st.session_state.feeling_level))
    if st.button("Send feeling level"):
        add_patient(f"My feeling level is {st.session_state.feeling_level}/10.")
        st.session_state.stage = 1
        ensure_stage_prompt()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Stage 1 — yes/no pain
# -------------------------------
elif stage == 1:
    st.markdown('<div class="panel"><div class="panel-title">Stage 1 · Pain today?</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Yes, I have pain"):
            st.session_state.pain_yesno = True
            add_patient("Yes, I have pain today.")
            st.session_state.stage = 2
            ensure_stage_prompt()
            st.rerun()
    with c2:
        if st.button("No pain"):
            st.session_state.pain_yesno = False
            add_patient("No, I don’t have pain today.")
            # If no pain, we can skip body selector but still keep the stage flow
            # We still proceed to symptom checklist (stage 3) per your requested stages.
            add_doctor("Okay — we’ll skip the body pain map.")
            st.session_state.stage = 3
            ensure_stage_prompt()
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Stage 2 — body selector
# -------------------------------
elif stage == 2:
    st.markdown('<div class="panel"><div class="panel-title">Stage 2 · Body pain map</div>', unsafe_allow_html=True)

    left, right = st.columns([1.2, 1.0], vertical_alignment="top")

    with left:
        # Visible SVG silhouette
        st.markdown(body_svg(st.session_state.selected_parts), unsafe_allow_html=True)

    with right:
        st.markdown("**Click to toggle regions** (multiple selections allowed):")
        # Real Streamlit buttons + callbacks (no URL hacks, no query params)
        buttons = [
            "Head", "Chest", "Abdomen",
            "Left Arm", "Right Arm",
            "Left Leg", "Right Leg",
        ]
        for part in buttons:
            label = f"✓ {part}" if part in st.session_state.selected_parts else part
            if st.button(label, key=f"toggle_{part}"):
                toggle_body_part(part)
                st.rerun()

        st.markdown('<div class="small-note">Selected: ' +
                    (", ".join(sorted(st.session_state.selected_parts)) if st.session_state.selected_parts else "None") +
                    "</div>", unsafe_allow_html=True)

    cA, cB = st.columns([1, 1])
    with cA:
        if st.button("Clear selections"):
            st.session_state.selected_parts = set()
            st.rerun()

    with cB:
        if st.button("Send selected pain locations"):
            if st.session_state.selected_parts:
                add_patient("Pain locations: " + ", ".join(sorted(st.session_state.selected_parts)) + ".")
            else:
                add_patient("I’m not sure / I didn’t select a specific location.")
            st.session_state.stage = 3
            ensure_stage_prompt()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Stage 3 — symptom checklist (multi-select)
# -------------------------------
elif stage == 3:
    st.markdown('<div class="panel"><div class="panel-title">Stage 3 · Symptom checklist</div>', unsafe_allow_html=True)

    symptom_options = [
        "Fatigue / low energy",
        "Nausea",
        "Vomiting",
        "Poor appetite",
        "Mouth sores",
        "Trouble swallowing",
        "Shortness of breath",
        "Fever / chills",
        "Constipation",
        "Diarrhea",
        "Sleep problems",
        "Anxiety / low mood",
    ]

    st.session_state.symptoms = st.multiselect("Select symptoms you have today:", symptom_options, default=st.session_state.symptoms)

    if st.button("Send symptoms"):
        if st.session_state.symptoms:
            add_patient("Symptoms today: " + "; ".join(st.session_state.symptoms) + ".")
        else:
            add_patient("No significant symptoms from the checklist.")
        st.session_state.stage = 4
        ensure_stage_prompt()
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Stage 4 — free text chat input
# -------------------------------
elif stage == 4:
    st.markdown(
        '<div class="panel"><div class="panel-title">Stage 4 · Free text</div>'
        '<div class="small-note">Type anything else you want your care team to know. (Prototype: simulated reply.)</div></div>',
        unsafe_allow_html=True
    )

    # Use Streamlit chat input for a messenger-like feel
    user_text = st.chat_input("Type your message…")

    if user_text:
        add_patient(user_text)

        # Local simulated response (NO API calls)
        canned = [
            "Thanks — I’ve recorded that. If symptoms worsen, consider contacting your care team.",
            "Got it. I’m logging this for your check-in summary.",
            "Thank you for sharing. Is there anything else you want to mention?",
        ]
        add_doctor(random.choice(canned))
        st.rerun()
