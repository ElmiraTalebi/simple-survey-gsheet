import streamlit as st
from typing import Dict, List, Set, Any
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import json
from openai import OpenAI

# ============================================================
# PAGE CONFIG (keep near top)
# ============================================================
st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ============================================================
# OPENAI / LLM SETUP  (API is the main chatbot in Stage 4)
# ============================================================
# Put OPENAI_API_KEY in Streamlit secrets:
# - local: .streamlit/secrets.toml
# - Streamlit Cloud: App -> Settings -> Secrets
client_llm = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

SYSTEM_PROMPT = """
You are a clinical symptom check-in assistant for cancer patients.
Your ONLY job is to collect symptoms and report them clearly for the care team.

Hard rules:
- Do NOT give medical advice.
- Do NOT recommend medications, treatments, or diagnoses.
- If the patient asks for advice/treatment/diagnosis, say exactly:
  "Please contact your healthcare provider for medical advice."
  Then immediately continue symptom intake with ONE short question.
- Keep responses short (1–4 sentences).
- Ask ONE intake question at a time.
- Use simple language. Make it easy to answer.

Emergency safety:
If the patient reports any emergency symptom (e.g., trouble breathing, chest pain, heavy bleeding, fainting, confusion),
tell them: "Please seek emergency medical attention immediately."
Then ask a short safety question: "Are you currently safe?"

Output format:
You MUST return valid JSON only (no extra text), exactly with keys:
{
  "assistant_message": "...",
  "extracted": {
    "symptoms": ["..."],
    "pain_locations": ["..."],
    "severity_0_to_10": null or number,
    "duration": "... or empty string",
    "urgency": "none" | "monitor" | "urgent" | "emergency",
    "red_flags": ["..."]
  },
  "summary_for_clinician": "1-3 short bullet-like sentences"
}

Notes:
- symptoms should be short phrases (e.g. "nausea", "fatigue", "shortness of breath").
- pain_locations should be among: Head, Chest, Abdomen, Left Arm, Right Arm, Left Leg, Right Leg, Other.
- If you are unsure, set fields to empty lists / null / "monitor".
"""

def build_api_messages() -> List[Dict[str, str]]:
    """Convert our UI chat history into OpenAI chat messages (limit history for cost/speed)."""
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    recent = st.session_state.messages[-20:]  # last 20 bubbles

    for m in recent:
        if m["role"] == "doctor":
            msgs.append({"role": "assistant", "content": m["content"]})
        else:
            msgs.append({"role": "user", "content": m["content"]})

    return msgs

def llm_turn() -> Dict[str, Any]:
    """
    Call the API for the next assistant turn.
    Returns a parsed dict containing:
      assistant_message, extracted{}, summary_for_clinician
    """
    try:
        response = client_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=build_api_messages(),
            temperature=0.2,
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)  # must be JSON per prompt
        return data
    except Exception:
        # Safe fallback if API or JSON parsing fails
        return {
            "assistant_message": (
                "Thanks — I recorded that. Please contact your healthcare provider for medical advice. "
                "What symptom is bothering you most right now?"
            ),
            "extracted": {
                "symptoms": [],
                "pain_locations": [],
                "severity_0_to_10": None,
                "duration": "",
                "urgency": "monitor",
                "red_flags": []
            },
            "summary_for_clinician": ""
        }

# ============================================================
# GOOGLE SHEETS SETUP
# ============================================================
SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPE)
gs_client = gspread.authorize(creds)
sheet = gs_client.open_by_key(st.secrets["gsheet_id"]).worksheet("Form")

def save_to_sheet():
    """
    Save timestamp + name + structured fields + API-extracted JSON + full conversation.
    Recommended Google Sheet columns:
    Timestamp | Name | Feeling | PainYesNo | PainLocations | ChecklistSymptoms | API_Structured | API_Summary | FullChat
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = st.session_state.get("patient_name", "Unknown")

    sheet.append_row([
        timestamp,
        name,
        int(st.session_state.feeling_level),
        str(st.session_state.pain_yesno),
        ", ".join(sorted(st.session_state.selected_parts)),
        "; ".join(st.session_state.symptoms),
        json.dumps(st.session_state.structured),  # API structured
        st.session_state.structured.get("summary_for_clinician", ""),
        json.dumps(st.session_state.messages),     # full chat
    ])

# ============================================================
# CSS — messenger look + soft medical background
# ============================================================
st.markdown(
    """
<style>
[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#eef4ff,#f6fbff);
}
.header{
    font-size: 24px;
    font-weight: 700;
    margin: 8px 0 14px 0;
}
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
.row-left{ display:flex; justify-content:flex-start; align-items:flex-end; margin: 10px 0; gap: 10px; }
.row-right{ display:flex; justify-content:flex-end; align-items:flex-end; margin: 10px 0; gap: 10px; }
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
.stButton>button{
    border-radius: 14px;
    padding: 0.55rem 0.9rem;
}
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
    st.session_state.stage = -1  # -1 = name entry, 0..4 = main stages

if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

if "selected_parts" not in st.session_state:
    st.session_state.selected_parts: Set[str] = set()

if "pain_yesno" not in st.session_state:
    st.session_state.pain_yesno = None  # True/False/None

if "feeling_level" not in st.session_state:
    st.session_state.feeling_level = 5

if "symptoms" not in st.session_state:
    st.session_state.symptoms: List[str] = []

if "submitted" not in st.session_state:
    st.session_state.submitted = False

# Used in Stage 4 gating (Yes/No)
if "free_text_permission" not in st.session_state:
    st.session_state.free_text_permission = None

# API-driven structured capture
if "structured" not in st.session_state:
    st.session_state.structured = {
        "symptoms": [],
        "pain_locations": [],
        "severity_0_to_10": None,
        "duration": "",
        "urgency": "unknown",
        "red_flags": [],
        "summary_for_clinician": ""
    }

if "api_chat_started" not in st.session_state:
    st.session_state.api_chat_started = False

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
        add_doctor(
            f"Hi {st.session_state.patient_name} — I'm your virtual check-in assistant. "
            "Let's do a quick symptom check-in."
        )
        return

    last_doc = None
    for m in reversed(st.session_state.messages):
        if m["role"] == "doctor":
            last_doc = m["content"]
            break

    prompts = {
        0: "How are you feeling today?",
        1: "Do you have any pain today?",
        2: "Please select where you feel pain on the body.",
        3: "Which symptoms are you experiencing today? (Select all that apply.)",
        4: "Now you can chat with me so I can collect details for your care team.",
    }

    want = prompts.get(stage, None)
    if want and (last_doc is None or want not in last_doc):
        add_doctor(want)

def toggle_body_part(part: str) -> None:
    if part in st.session_state.selected_parts:
        st.session_state.selected_parts.remove(part)
    else:
        st.session_state.selected_parts.add(part)

def body_svg(selected: Set[str]) -> str:
    """Simple human silhouette made of separate SVG regions."""
    def fill(part: str) -> str:
        return "#1f7aff" if part in selected else "#cfd8e6"

    stroke = "#6b7a90"
    return f"""
<svg width="320" height="520" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="rgba(0,0,0,0.15)"/>
    </filter>
  </defs>

  <g filter="url(#shadow)">
    <circle cx="160" cy="70" r="38" fill="{fill('Head')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <g filter="url(#shadow)">
    <rect x="110" y="120" width="100" height="70" rx="24" fill="{fill('Chest')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <g filter="url(#shadow)">
    <rect x="115" y="195" width="90" height="70" rx="22" fill="{fill('Abdomen')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <g filter="url(#shadow)">
    <path d="M110 132
             C 80 145, 72 180, 78 220
             C 82 250, 92 270, 100 290
             C 108 310, 115 320, 120 320
             L 120 130 Z"
          fill="{fill('Left Arm')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <g filter="url(#shadow)">
    <path d="M210 132
             C 240 145, 248 180, 242 220
             C 238 250, 228 270, 220 290
             C 212 310, 205 320, 200 320
             L 200 130 Z"
          fill="{fill('Right Arm')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <g filter="url(#shadow)">
    <path d="M135 265
             C 120 310, 118 360, 126 410
             C 132 445, 132 475, 128 500
             L 155 500
             C 158 470, 160 435, 156 405
             C 150 355, 152 312, 165 265 Z"
          fill="{fill('Left Leg')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <g filter="url(#shadow)">
    <path d="M185 265
             C 200 310, 202 360, 194 410
             C 188 445, 188 475, 192 500
             L 165 500
             C 162 470, 160 435, 164 405
             C 170 355, 168 312, 155 265 Z"
          fill="{fill('Right Leg')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <text x="160" y="520" text-anchor="middle" font-size="12" fill="rgba(0,0,0,0.55)">
    Click buttons to toggle regions (highlights update live)
  </text>
</svg>
""".strip()

def merge_extracted(extracted: Dict[str, Any]) -> None:
    """Merge model-extracted fields into our session state."""
    if not extracted:
        return

    # symptoms
    sym = extracted.get("symptoms", []) or []
    for s in sym:
        if s and s not in st.session_state.structured["symptoms"]:
            st.session_state.structured["symptoms"].append(s)

    # pain locations -> we also sync to selected_parts so your UI remains consistent
    locs = extracted.get("pain_locations", []) or []
    for loc in locs:
        if loc:
            st.session_state.structured["pain_locations"].append(loc) if loc not in st.session_state.structured["pain_locations"] else None
            st.session_state.selected_parts.add(loc)

    # severity + duration
    sev = extracted.get("severity_0_to_10", None)
    if isinstance(sev, (int, float)):
        # clamp
        sev = max(0, min(10, int(sev)))
        st.session_state.structured["severity_0_to_10"] = sev

    dur = extracted.get("duration", "")
    if isinstance(dur, str) and dur.strip():
        st.session_state.structured["duration"] = dur.strip()

    # urgency + red flags
    urg = extracted.get("urgency", "")
    if urg in ["none", "monitor", "urgent", "emergency"]:
        st.session_state.structured["urgency"] = urg

    rfs = extracted.get("red_flags", []) or []
    for rf in rfs:
        if rf and rf not in st.session_state.structured["red_flags"]:
            st.session_state.structured["red_flags"].append(rf)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="chat-shell"><div class="header">🩺 Cancer Symptom Check-In</div>', unsafe_allow_html=True)

# ============================================================
# STAGE -1 — Patient name entry
# ============================================================
if st.session_state.stage == -1:
    st.markdown('<div class="panel"><div class="panel-title">Welcome · Please enter your name</div>', unsafe_allow_html=True)
    name_input = st.text_input("Your name:", value=st.session_state.patient_name)
    if st.button("Start Check-In"):
        if name_input.strip():
            st.session_state.patient_name = name_input.strip()
            st.session_state.stage = 0
            ensure_stage_prompt()
            st.rerun()
        else:
            st.warning("Please enter your name to continue.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ============================================================
# ENSURE PROMPT FOR CURRENT STAGE
# ============================================================
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
# STAGE PANELS
# ============================================================
stage = st.session_state.stage

# -------------------------------
# Stage 0 — feeling buttons
# -------------------------------
if stage == 0:
    st.markdown('<div class="panel"><div class="panel-title">Stage 0 · How are you feeling today?</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    scale_labels = ["Very Bad", "Bad", "Okay", "Good", "Very Good"]
    scale_values = [0, 2, 5, 8, 10]

    for i in range(5):
        with cols[i]:
            if st.button(scale_labels[i], key=f"feel_{i}"):
                st.session_state.feeling_level = scale_values[i]
                add_patient(f"I feel {scale_labels[i]} today ({scale_values[i]}/10).")
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
            add_patient("No, I don't have pain today.")
            add_doctor("Okay — we'll skip the body pain map.")
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
        st.markdown(body_svg(st.session_state.selected_parts), unsafe_allow_html=True)

    with right:
        st.markdown("**Click to toggle regions** (multiple selections allowed):")
        buttons = ["Head", "Chest", "Abdomen", "Left Arm", "Right Arm", "Left Leg", "Right Leg", "Other"]
        for part in buttons:
            label = f"✓ {part}" if part in st.session_state.selected_parts else part
            if st.button(label, key=f"toggle_{part}"):
                toggle_body_part(part)
                st.rerun()

        st.markdown(
            '<div class="small-note">Selected: ' +
            (", ".join(sorted(st.session_state.selected_parts)) if st.session_state.selected_parts else "None") +
            "</div>",
            unsafe_allow_html=True
        )

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
                add_patient("I'm not sure / I didn't select a specific location.")
            st.session_state.stage = 3
            ensure_stage_prompt()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Stage 3 — symptom checklist
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

    st.session_state.symptoms = st.multiselect(
        "Select symptoms you have today:",
        symptom_options,
        default=st.session_state.symptoms
    )

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
# Stage 4 — API is the main chatbot here
# -------------------------------
elif stage == 4:

    if st.session_state.submitted:
        st.success("✅ Your check-in has been submitted. Thank you!")
        st.stop()

    st.markdown('<div class="panel"><div class="panel-title">Stage 4 · Chat (symptom intake)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-note">Describe how you feel. I will ask a few short questions and log your symptoms for your care team.</div></div>',
        unsafe_allow_html=True
    )

    # Ask a first question once
    if not st.session_state.api_chat_started:
        st.session_state.api_chat_started = True
        add_doctor("To start, what symptom is bothering you most right now?")
        st.rerun()

    # Patient types messages; EVERY patient message triggers an API reply
    user_text = st.chat_input("Type your message…")

    if user_text:
        add_patient(user_text)

        # Call API to get the next assistant message + extracted fields
        data = llm_turn()

        assistant_message = data.get("assistant_message", "")
        extracted = data.get("extracted", {}) or {}
        summary = data.get("summary_for_clinician", "") or ""

        if assistant_message:
            add_doctor(assistant_message)

        # Merge extracted structure into session_state.structured
        merge_extracted(extracted)

        # Save/refresh clinician summary
        if summary.strip():
            st.session_state.structured["summary_for_clinician"] = summary.strip()

        # Also show an on-screen warning if the model flags emergency
        if st.session_state.structured.get("urgency") == "emergency":
            st.error("⚠️ Emergency warning: Please seek emergency medical attention immediately.")

        st.rerun()

    # Optional: show a tiny “what we captured” panel (helpful for debugging; you can remove later)
    with st.expander("What the assistant has captured so far (debug)"):
        st.json(st.session_state.structured)

    # Finish/submit
    st.markdown('<div class="panel"><div class="panel-title">Finish</div>', unsafe_allow_html=True)
    if st.button("✅ Submit Check-In"):
        save_to_sheet()
        st.session_state.submitted = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
