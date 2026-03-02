import streamlit as st
from typing import Dict, List, Set
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import json
from openai import OpenAI

# ============================================================
# GOOGLE SHEETS SETUP
# ============================================================
SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPE)
gs_client = gspread.authorize(creds)
sheet = gs_client.open_by_key(st.secrets["gsheet_id"]).worksheet("Form")

def load_past_checkins(name: str) -> List[Dict]:
    """
    Load previous check-ins for this patient from Google Sheets.
    Returns up to the last 5 sessions — injected into GPT's system prompt as memory.
    From transcript: 'if you have memory, you know which is the biggest problem
    for that patient, so just follow up on that.'
    """
    try:
        all_rows = sheet.get_all_values()
        past = []
        for row in all_rows[1:]:  # skip header row if present
            if len(row) >= 3 and row[1].strip().lower() == name.strip().lower():
                try:
                    data = json.loads(row[2])
                    data["timestamp"] = row[0]
                    past.append(data)
                except Exception:
                    pass
        return past[-5:]  # keep last 5 sessions to avoid oversized prompt
    except Exception:
        return []

def save_to_sheet():
    """Save patient name, timestamp, and full chat as a dict to Google Sheets."""
    chat_dict = {
        "feeling_level": st.session_state.feeling_level,
        "pain": st.session_state.pain_yesno,
        "pain_locations": sorted(list(st.session_state.selected_parts)),
        "symptoms": st.session_state.symptoms,
        "conversation": st.session_state.messages,
    }
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = st.session_state.get("patient_name", "Unknown")
    sheet.append_row([timestamp, name, json.dumps(chat_dict)])

# ============================================================
# OPENAI CLIENT
# ============================================================
openai_client = OpenAI(api_key=st.secrets["openai_api_key"])

def build_system_prompt() -> str:
    """
    Build the GPT system prompt.

    Architecture (from transcript):
    - GPT is the symptom intake agent for the WHOLE conversation, not just the free-text stage.
    - Hybrid approach: structured widgets collect data, GPT reacts and asks follow-ups after each one.
    - Memory: past check-in data from Sheets is injected so GPT can follow up on recurring issues.
    - Tone: warm, light daily check-in — not a deep formal survey. Natural conversation.
    - Constraints: no medical advice, no diagnoses, redirect off-topic gently.
    """
    name = st.session_state.get("patient_name", "the patient")

    # Structured data collected so far this session
    feeling = st.session_state.get("feeling_level", None)
    pain = st.session_state.get("pain_yesno", None)
    locations = sorted(list(st.session_state.get("selected_parts", set())))
    symptoms = st.session_state.get("symptoms", [])

    session_lines = []
    if feeling is not None:
        session_lines.append(f"- Feeling level: {feeling}/10")
    if pain is not None:
        session_lines.append(f"- Pain today: {'yes' if pain else 'no'}")
    if locations:
        session_lines.append(f"- Pain locations: {', '.join(locations)}")
    if symptoms:
        session_lines.append(f"- Symptoms from checklist: {', '.join(symptoms)}")
    session_str = "\n".join(session_lines) if session_lines else "Check-in just started — nothing collected yet."

    # Past session memory from Google Sheets
    past = st.session_state.get("past_checkins", [])
    if past:
        mem_lines = []
        for p in past:
            ts = p.get("timestamp", "unknown date")
            fl = p.get("feeling_level", "?")
            pn = "yes" if p.get("pain") else "no"
            locs = ", ".join(p.get("pain_locations", [])) or "none"
            syms = ", ".join(p.get("symptoms", [])) or "none"
            mem_lines.append(f"  [{ts}] Feeling: {fl}/10 | Pain: {pn} | Locations: {locs} | Symptoms: {syms}")
        memory_str = "\n".join(mem_lines)
    else:
        memory_str = "No previous check-ins on record — this may be their first session."

    return f"""You are a warm, empathetic virtual symptom-intake assistant for a cancer care clinic.
Your role is to conduct a brief, natural daily check-in with the patient: {name}.

You are part of a HYBRID interface. The patient fills in structured widgets (a feeling slider, a
body pain map, a symptom checklist), and YOU handle the conversational layer — reacting naturally
to each widget submission and asking one thoughtful follow-up question at a time.

Think of yourself as a caring companion, not a formal questionnaire. The goal is a light daily
check-in: "Hey, how are you today? Anything bothering you? Okay great, take care." — not a deep
clinical interview.

=== TODAY'S STRUCTURED DATA (collected so far via widgets) ===
{session_str}

=== PATIENT HISTORY / MEMORY (past sessions from Google Sheets) ===
{memory_str}

=== YOUR RULES ===
1. Be warm, natural, and conversational. Short sentences. Empathetic tone.
2. After the patient submits widget data, react to it naturally, then ask ONE focused follow-up.
   Examples:
   - Feeling level 3/10 → "I'm sorry to hear that — what's been the hardest part today?"
   - Pain in abdomen → "Is the abdominal pain sharp, or more of a dull ache?"
   - Nausea on checklist → "Has the nausea been affecting your eating today?"
3. Use memory: if a patient previously had recurring symptoms, check in on those specifically.
   Example: "Last time you mentioned fatigue was really affecting you — how has that been this week?"
4. If the patient goes off-topic, redirect gently: "That's good to know — let's stay focused on
   how you're feeling physically today. Is there anything else symptom-wise to share?"
5. NEVER give medical advice, diagnoses, medication suggestions, or treatment guidance. If asked,
   say: "I'm not able to give medical advice — I'm here so your care team can follow up with you."
6. Ask ONE question at a time. Do not list multiple questions.
7. When the conversation feels complete, say something like: "It sounds like we have a good picture
   of how you're doing today. Feel free to hit Submit when you're ready."
8. Brief warm small talk is fine (e.g. if they mention something personal), but always steer
   naturally back to the check-in.
"""

def get_gpt_reply() -> str:
    """
    Call the GPT API with the full conversation history as memory.
    Converts app message roles (doctor/patient) → OpenAI roles (assistant/user).
    """
    openai_messages = [{"role": "system", "content": build_system_prompt()}]
    for msg in st.session_state.messages:
        if msg["role"] == "doctor":
            openai_messages.append({"role": "assistant", "content": msg["content"]})
        elif msg["role"] == "patient":
            openai_messages.append({"role": "user", "content": msg["content"]})
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=openai_messages,
            max_tokens=350,
            temperature=0.6,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Sorry, I couldn't connect right now. Please try again. Error: {e})"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#eef4ff,#f6fbff);
}
.header{ font-size:24px; font-weight:700; margin:8px 0 14px 0; }
.chat-shell{ max-width:840px; margin:0 auto; }
.chat-window{
    max-height:55vh; overflow-y:auto; padding:18px 14px; border-radius:18px;
    background:rgba(255,255,255,0.55); border:1px solid rgba(200,210,230,0.55);
    backdrop-filter:blur(10px);
}
.row-left{ display:flex; justify-content:flex-start; align-items:flex-end; margin:10px 0; gap:10px; }
.row-right{ display:flex; justify-content:flex-end; align-items:flex-end; margin:10px 0; gap:10px; }
.avatar{
    width:36px; height:36px; border-radius:50%;
    display:flex; justify-content:center; align-items:center;
    background:rgba(255,255,255,0.9); border:1px solid rgba(210,220,240,0.9);
    box-shadow:0 2px 8px rgba(0,0,0,0.08); font-size:18px; flex:0 0 auto;
}
.bubble-doc{
    background:#ffffff; border:1px solid rgba(220,225,235,0.95);
    border-radius:18px; padding:12px 14px; max-width:72%;
    box-shadow:0 2px 10px rgba(0,0,0,0.05); white-space:pre-wrap;
}
.bubble-pat{
    background:#1f7aff; color:white; border-radius:18px;
    padding:12px 14px; max-width:72%;
    box-shadow:0 2px 10px rgba(0,0,0,0.08); white-space:pre-wrap;
}
.small-note{ color:rgba(0,0,0,0.55); font-size:12px; margin-top:6px; }
.panel{
    margin-top:14px; padding:14px; border-radius:16px;
    background:rgba(255,255,255,0.65); border:1px solid rgba(200,210,230,0.55);
    backdrop-filter:blur(10px);
}
.panel-title{ font-weight:700; margin-bottom:10px; }
.stButton>button{ border-radius:14px; padding:0.55rem 0.9rem; }
[data-testid="stChatInput"]{
    position:sticky; bottom:0; background:rgba(255,255,255,0.6);
    backdrop-filter:blur(10px); border-top:1px solid rgba(200,210,230,0.55);
    padding-top:10px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INIT
# ============================================================
defaults = {
    "messages": [],
    "stage": -1,
    "patient_name": "",
    "selected_parts": set(),
    "pain_yesno": None,
    "feeling_level": 5,
    "symptoms": [],
    "submitted": False,
    "past_checkins": [],
    "gpt_followup_done": set(),  # ensures each stage only fires one GPT follow-up
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# HELPERS
# ============================================================
def add_doctor(text: str) -> None:
    st.session_state.messages.append({"role": "doctor", "content": text})

def add_patient(text: str) -> None:
    st.session_state.messages.append({"role": "patient", "content": text})

def gpt_followup(stage_key: str) -> None:
    """
    Fire a GPT follow-up after a structured widget is submitted.
    Only fires once per stage_key. GPT sees updated structured data via system prompt,
    so it knows exactly what was just submitted and responds accordingly.
    """
    if stage_key not in st.session_state.gpt_followup_done:
        st.session_state.gpt_followup_done.add(stage_key)
        reply = get_gpt_reply()
        add_doctor(reply)

def toggle_body_part(part: str) -> None:
    if part in st.session_state.selected_parts:
        st.session_state.selected_parts.remove(part)
    else:
        st.session_state.selected_parts.add(part)

def body_svg(selected: Set[str]) -> str:
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
    <path d="M110 132 C 80 145, 72 180, 78 220 C 82 250, 92 270, 100 290 C 108 310, 115 320, 120 320 L 120 130 Z"
          fill="{fill('Left Arm')}" stroke="{stroke}" stroke-width="2"/>
  </g>
  <g filter="url(#shadow)">
    <path d="M210 132 C 240 145, 248 180, 242 220 C 238 250, 228 270, 220 290 C 212 310, 205 320, 200 320 L 200 130 Z"
          fill="{fill('Right Arm')}" stroke="{stroke}" stroke-width="2"/>
  </g>
  <g filter="url(#shadow)">
    <path d="M135 265 C 120 310, 118 360, 126 410 C 132 445, 132 475, 128 500 L 155 500 C 158 470, 160 435, 156 405 C 150 355, 152 312, 165 265 Z"
          fill="{fill('Left Leg')}" stroke="{stroke}" stroke-width="2"/>
  </g>
  <g filter="url(#shadow)">
    <path d="M185 265 C 200 310, 202 360, 194 410 C 188 445, 188 475, 192 500 L 165 500 C 162 470, 160 435, 164 405 C 170 355, 168 312, 155 265 Z"
          fill="{fill('Right Leg')}" stroke="{stroke}" stroke-width="2"/>
  </g>
  <text x="160" y="518" text-anchor="middle" font-size="12" fill="rgba(0,0,0,0.45)">
    Click buttons to toggle regions
  </text>
</svg>""".strip()

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="chat-shell"><div class="header">🩺 Cancer Symptom Check-In</div>', unsafe_allow_html=True)

# ============================================================
# STAGE -1 — Name entry
# ============================================================
if st.session_state.stage == -1:
    st.markdown('<div class="panel"><div class="panel-title">Welcome · Please enter your name</div>', unsafe_allow_html=True)
    name_input = st.text_input("Your name:", value=st.session_state.patient_name)
    if st.button("Start Check-In"):
        if name_input.strip():
            st.session_state.patient_name = name_input.strip()

            # Load past sessions from Google Sheets for GPT memory
            with st.spinner("Loading your history…"):
                st.session_state.past_checkins = load_past_checkins(name_input.strip())

            # GPT opens the conversation with a personalized greeting
            # It already has access to memory via build_system_prompt()
            with st.spinner("Getting your assistant ready…"):
                opening = get_gpt_reply()
            if not opening:
                opening = f"Hi {st.session_state.patient_name}! I'm your virtual check-in assistant. Let's do a quick check-in. How have you been feeling? Use the slider below to give me a number from 0 to 10."
            add_doctor(opening)
            st.session_state.stage = 0
            st.rerun()
        else:
            st.warning("Please enter your name to continue.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ============================================================
# CHAT WINDOW — render all messages
# ============================================================
st.markdown('<div class="chat-window">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "doctor":
        st.markdown(f"""
        <div class="row-left">
          <div class="avatar">🩺</div>
          <div class="bubble-doc">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="row-right">
          <div class="bubble-pat">{msg["content"]}</div>
          <div class="avatar">🙂</div>
        </div>""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

stage = st.session_state.stage

# ============================================================
# PERSISTENT FREE-TEXT INPUT — available at every stage
# The patient can reply to GPT follow-ups naturally at any point,
# not just at the final stage. This is the "normal conversation"
# that was discussed in the transcript.
# ============================================================
if not st.session_state.submitted:
    user_text = st.chat_input("Reply to the assistant or type anything…")
    if user_text:
        add_patient(user_text)
        with st.spinner("Assistant is thinking…"):
            reply = get_gpt_reply()
        add_doctor(reply)
        st.rerun()

# ============================================================
# STAGE PANELS — structured widgets sit below the chat
# ============================================================

# ----------------------------------------------------------
# Stage 0 — Feeling slider
# ----------------------------------------------------------
if stage == 0:
    st.markdown('<div class="panel"><div class="panel-title">How are you feeling today?</div>', unsafe_allow_html=True)
    st.session_state.feeling_level = st.slider(
        "0 = worst, 10 = best", 0, 10, int(st.session_state.feeling_level)
    )
    if st.button("Send feeling level ➜"):
        add_patient(f"My feeling level today is {st.session_state.feeling_level}/10.")
        st.session_state.stage = 1
        with st.spinner("Assistant is thinking…"):
            gpt_followup("stage0")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# Stage 1 — Pain yes/no
# ----------------------------------------------------------
elif stage == 1:
    st.markdown('<div class="panel"><div class="panel-title">Do you have any pain today?</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Yes, I have pain"):
            st.session_state.pain_yesno = True
            add_patient("Yes, I have pain today.")
            st.session_state.stage = 2
            with st.spinner("Assistant is thinking…"):
                gpt_followup("stage1_yes")
            st.rerun()
    with c2:
        if st.button("🙂 No pain today"):
            st.session_state.pain_yesno = False
            add_patient("No, I don't have any pain today.")
            st.session_state.stage = 3
            with st.spinner("Assistant is thinking…"):
                gpt_followup("stage1_no")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# Stage 2 — Body pain map
# ----------------------------------------------------------
elif stage == 2:
    st.markdown('<div class="panel"><div class="panel-title">Where do you feel pain?</div>', unsafe_allow_html=True)
    left, right = st.columns([1.2, 1.0], vertical_alignment="top")
    with left:
        st.markdown(body_svg(st.session_state.selected_parts), unsafe_allow_html=True)
    with right:
        st.markdown("**Click to toggle regions:**")
        for part in ["Head", "Chest", "Abdomen", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]:
            label = f"✓ {part}" if part in st.session_state.selected_parts else part
            if st.button(label, key=f"toggle_{part}"):
                toggle_body_part(part)
                st.rerun()
        st.markdown(
            '<div class="small-note">Selected: ' +
            (", ".join(sorted(st.session_state.selected_parts)) or "None") +
            "</div>", unsafe_allow_html=True
        )
    cA, cB = st.columns(2)
    with cA:
        if st.button("Clear all"):
            st.session_state.selected_parts = set()
            st.rerun()
    with cB:
        if st.button("Send pain locations ➜"):
            if st.session_state.selected_parts:
                add_patient("Pain locations: " + ", ".join(sorted(st.session_state.selected_parts)) + ".")
            else:
                add_patient("I'm not sure of the exact location.")
            st.session_state.stage = 3
            with st.spinner("Assistant is thinking…"):
                gpt_followup("stage2")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# Stage 3 — Symptom checklist
# ----------------------------------------------------------
elif stage == 3:
    st.markdown('<div class="panel"><div class="panel-title">Any of these symptoms today?</div>', unsafe_allow_html=True)
    symptom_options = [
        "Fatigue / low energy", "Nausea", "Vomiting", "Poor appetite",
        "Mouth sores", "Trouble swallowing", "Shortness of breath",
        "Fever / chills", "Constipation", "Diarrhea",
        "Sleep problems", "Anxiety / low mood",
    ]
    st.session_state.symptoms = st.multiselect(
        "Select all that apply (leave empty if none):",
        symptom_options, default=st.session_state.symptoms
    )
    if st.button("Send symptoms ➜"):
        if st.session_state.symptoms:
            add_patient("Symptoms today: " + "; ".join(st.session_state.symptoms) + ".")
        else:
            add_patient("No symptoms from the checklist today.")
        st.session_state.stage = 4
        with st.spinner("Assistant is thinking…"):
            gpt_followup("stage3")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# Stage 4 — Free chat + submit
# Structured widgets are done. GPT continues the conversation
# naturally. Patient can type freely or submit.
# ----------------------------------------------------------
elif stage == 4:
    if st.session_state.submitted:
        st.success("✅ Your check-in has been submitted. Thank you — your care team will review this shortly.")
    else:
        st.markdown(
            '<div class="panel">'
            '<div class="panel-title">💬 Anything else to share?</div>'
            '<div class="small-note">Chat freely with the assistant, or click Submit when ready.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("✅ Submit Check-In"):
            try:
                save_to_sheet()
                st.session_state.submitted = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save to Google Sheets: {e}")
