import streamlit as st
from typing import Dict, List, Set, Any
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import json
from openai import OpenAI

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ============================================================
# OPENAI / LLM SETUP
# ============================================================
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
Then ask: "Are you currently safe?"

Output format — MUST be valid JSON only, no extra text:
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
- symptoms: short phrases (e.g. "nausea", "fatigue").
- pain_locations: Head, Chest, Abdomen, Left Arm, Right Arm, Left Leg, Right Leg, Other.
- If unsure, set fields to empty lists / null / "monitor".
"""

def build_api_messages() -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages[-20:]:
        role = "assistant" if m["role"] == "doctor" else "user"
        msgs.append({"role": role, "content": m["content"]})
    return msgs

def llm_turn() -> Dict[str, Any]:
    try:
        response = client_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=build_api_messages(),
            temperature=0.2,
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {
            "assistant_message": (
                "Thanks — I recorded that. Please contact your healthcare provider for medical advice. "
                "What symptom is bothering you most right now?"
            ),
            "extracted": {"symptoms": [], "pain_locations": [], "severity_0_to_10": None,
                          "duration": "", "urgency": "monitor", "red_flags": []},
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
    pc = st.session_state.pain_characteristics
    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        st.session_state.get("patient_name", "Unknown"),
        str(st.session_state.pain_yesno),
        ", ".join(sorted(st.session_state.selected_parts)),
        pc.get("severity", ""),
        ", ".join(pc.get("types", [])),
        ", ".join(pc.get("timing", [])),
        ", ".join(pc.get("triggers", [])),
        "; ".join(st.session_state.symptoms),
        st.session_state.get("anything_else_text", ""),
        json.dumps(st.session_state.structured),
        st.session_state.structured.get("summary_for_clinician", ""),
        json.dumps(st.session_state.messages),
    ])

# ============================================================
# CSS — accessible, large tap targets, chip-first design
# ============================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #f0f6ff 0%, #f7fbff 60%, #edf4f0 100%);
    min-height: 100vh;
}
.progress-bar-wrap { max-width:580px; margin:0 auto 24px auto; padding:0 8px; }
.progress-label {
    display:flex; justify-content:space-between;
    font-size:12px; color:#6b7a90; margin-bottom:6px; font-weight:500;
}
.progress-track { height:6px; background:#dce5f0; border-radius:99px; overflow:hidden; }
.progress-fill {
    height:100%; background:linear-gradient(90deg, #3b82f6, #06b6d4);
    border-radius:99px; transition:width 0.4s ease;
}
.card {
    max-width:580px; margin:0 auto 20px auto;
    background:white; border-radius:24px; padding:28px 28px 20px 28px;
    box-shadow:0 4px 24px rgba(30,60,120,0.07),0 1px 4px rgba(0,0,0,0.04);
    border:1px solid rgba(200,215,240,0.5);
}
.card-title { font-size:21px; font-weight:700; color:#1a2540; margin-bottom:6px; }
.card-sub   { font-size:14px; color:#6b7a90; margin-bottom:20px; line-height:1.5; }
.section-head {
    font-size:12px; font-weight:700; color:#6b7a90;
    text-transform:uppercase; letter-spacing:0.06em; margin:20px 0 8px 0;
}
/* base buttons */
.stButton > button {
    border-radius:12px !important; font-size:15px !important; font-weight:600 !important;
    padding:0.7rem 1rem !important; border:2px solid transparent !important;
    transition:all 0.15s ease !important; width:100% !important; cursor:pointer !important;
}
/* greeting CTA */
.big-btn .stButton > button {
    background:linear-gradient(135deg, #3b82f6, #2563eb) !important; color:white !important;
    font-size:18px !important; padding:1rem 1.5rem !important; border-radius:16px !important;
    box-shadow:0 4px 16px rgba(59,130,246,0.35) !important;
}
/* pain yes */
.pain-yes .stButton > button {
    background:#fff0f0 !important; color:#c0392b !important;
    border:2px solid #f5b7b1 !important; font-size:16px !important;
    padding:1rem !important; min-height:62px !important;
}
.pain-yes .stButton > button:hover { background:#ffe4e1 !important; }
/* pain no */
.pain-no .stButton > button {
    background:#f0fff4 !important; color:#196f3d !important;
    border:2px solid #a9dfbf !important; font-size:16px !important;
    padding:1rem !important; min-height:62px !important;
}
.pain-no .stButton > button:hover { background:#d5f5e3 !important; }
/* severity chips */
.sev-chip .stButton > button {
    background:#f3f6fb !important; color:#2c3e6b !important;
    border:2px solid #cdd9ed !important; font-size:13px !important;
    min-height:52px !important;
}
.sev-chip-active .stButton > button {
    background:linear-gradient(135deg,#3b82f6,#1d4ed8) !important;
    color:white !important; border:2px solid transparent !important;
    box-shadow:0 2px 10px rgba(59,130,246,0.35) !important;
}
/* generic chips */
.chip-btn .stButton > button {
    background:#f3f6fb !important; color:#2c3e6b !important;
    border:2px solid #cdd9ed !important; font-size:13px !important;
    padding:0.55rem 0.5rem !important; border-radius:10px !important;
    min-height:48px !important;
}
.chip-btn .stButton > button:hover { background:#e8f0fe !important; border-color:#93c5fd !important; }
.chip-btn-active .stButton > button {
    background:#eff6ff !important; color:#1d4ed8 !important;
    border:2px solid #3b82f6 !important; font-weight:700 !important;
}
/* nav */
.nav-next .stButton > button {
    background:#1d4ed8 !important; color:white !important;
    border-radius:12px !important; min-height:52px !important;
}
.nav-clear .stButton > button {
    background:#f8fafc !important; color:#6b7a90 !important;
    border:2px solid #e2e8f0 !important;
}
/* body map buttons */
.body-btn .stButton > button {
    background:#f3f6fb !important; color:#2c3e6b !important;
    border:2px solid #cdd9ed !important; font-size:14px !important;
    min-height:46px !important;
}
.body-btn-active .stButton > button {
    background:#dbeafe !important; color:#1e3a8a !important;
    border:2px solid #3b82f6 !important;
}
/* review rows */
.review-row {
    display:flex; gap:12px; align-items:flex-start;
    padding:10px 0; border-bottom:1px solid #f0f4f8; font-size:14px;
}
.review-row:last-child { border-bottom:none; }
.review-label { min-width:155px; color:#6b7a90; font-weight:500; }
.review-value  { color:#1a2540; font-weight:600; flex:1; }
/* submit */
.submit-btn .stButton > button {
    background:linear-gradient(135deg,#10b981,#059669) !important;
    color:white !important; font-size:17px !important;
    padding:0.9rem !important; border-radius:14px !important;
    box-shadow:0 4px 16px rgba(16,185,129,0.35) !important; min-height:58px !important;
}
/* chat */
.chat-window {
    max-width:580px; margin:0 auto;
    max-height:50vh; overflow-y:auto; padding:12px 4px;
}
.row-left  { display:flex; justify-content:flex-start;  align-items:flex-end; margin:8px 0; gap:8px; }
.row-right { display:flex; justify-content:flex-end;    align-items:flex-end; margin:8px 0; gap:8px; }
.avatar {
    width:34px; height:34px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    background:#f0f6ff; border:1px solid #cdd9ed; font-size:17px; flex:0 0 auto;
}
.bubble-doc {
    background:white; border:1px solid #dce5f0;
    border-radius:18px 18px 18px 4px; padding:11px 14px; max-width:78%;
    box-shadow:0 2px 8px rgba(0,0,0,0.05); font-size:14px; line-height:1.5; color:#1a2540;
}
.bubble-pat {
    background:linear-gradient(135deg,#3b82f6,#2563eb); color:white;
    border-radius:18px 18px 4px 18px; padding:11px 14px; max-width:78%;
    box-shadow:0 2px 8px rgba(59,130,246,0.25); font-size:14px; line-height:1.5;
}
.greeting-wrap { max-width:580px; margin:40px auto 0 auto; text-align:center; }
.greeting-icon  { font-size:54px; margin-bottom:10px; }
.greeting-title { font-size:28px; font-weight:800; color:#1a2540; margin-bottom:8px; }
.greeting-text  { font-size:15px; color:#6b7a90; line-height:1.7; margin-bottom:28px; }
#MainMenu, footer, header { visibility:hidden; }
[data-testid="stChatInput"] {
    position:sticky; bottom:0; background:rgba(255,255,255,0.88);
    backdrop-filter:blur(12px); border-top:1px solid #e5eaf3; padding-top:10px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# STAGE MAP
# -1 = Greeting/name   0 = Pain gate   1 = Body map
#  2 = Pain characteristics (pain only)
#  3 = Symptom checklist   4 = "Anything else?" gate
#  5 = Free-text / LLM chat (gate=yes only)
#  6 = Review + Submit
# ============================================================
STAGE_LABELS = {
    -1:"Welcome", 0:"Pain check", 1:"Pain location",
     2:"Pain details", 3:"Symptoms", 4:"Wrap-up",
     5:"More details", 6:"Review & Submit",
}
TOTAL_STAGES = 6

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "stage": -1,
    "patient_name": "",
    "messages": [],
    "selected_parts": set(),
    "pain_yesno": None,
    "symptoms": [],
    "anything_else": None,
    "anything_else_text": "",
    "submitted": False,
    "api_chat_started": False,
    "pain_characteristics": {"severity": "", "types": [], "timing": [], "triggers": []},
    "structured": {"symptoms":[], "pain_locations":[], "severity_0_to_10":None,
                   "duration":"", "urgency":"unknown", "red_flags":[], "summary_for_clinician":""},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# HELPERS
# ============================================================
def add_doctor(t): st.session_state.messages.append({"role":"doctor","content":t})
def add_patient(t): st.session_state.messages.append({"role":"patient","content":t})

def toggle_body_part(p):
    if p in st.session_state.selected_parts: st.session_state.selected_parts.remove(p)
    else: st.session_state.selected_parts.add(p)

def toggle_pc(field, item):
    lst = st.session_state.pain_characteristics[field]
    if item in lst: lst.remove(item)
    else: lst.append(item)

def toggle_symptom(s):
    if s in st.session_state.symptoms: st.session_state.symptoms.remove(s)
    else: st.session_state.symptoms.append(s)

def merge_extracted(ex: Dict[str, Any]):
    if not ex: return
    s = st.session_state.structured
    for sym in (ex.get("symptoms") or []):
        if sym and sym not in s["symptoms"]: s["symptoms"].append(sym)
    for loc in (ex.get("pain_locations") or []):
        if loc and loc not in s["pain_locations"]:
            s["pain_locations"].append(loc)
            st.session_state.selected_parts.add(loc)
    sev = ex.get("severity_0_to_10")
    if isinstance(sev, (int, float)): s["severity_0_to_10"] = max(0, min(10, int(sev)))
    dur = ex.get("duration","")
    if isinstance(dur, str) and dur.strip(): s["duration"] = dur.strip()
    urg = ex.get("urgency","")
    if urg in ["none","monitor","urgent","emergency"]: s["urgency"] = urg
    for rf in (ex.get("red_flags") or []):
        if rf and rf not in s["red_flags"]: s["red_flags"].append(rf)

def body_svg(selected: Set[str]) -> str:
    def f(p): return "#3b82f6" if p in selected else "#cfd8e6"
    sk = "#6b7a90"
    return f"""<svg width="220" height="400" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">
  <defs><filter id="sh"><feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="rgba(0,0,0,0.12)"/></filter></defs>
  <g filter="url(#sh)"><circle cx="160" cy="70" r="38" fill="{f('Head')}" stroke="{sk}" stroke-width="2"/></g>
  <g filter="url(#sh)"><rect x="110" y="120" width="100" height="70" rx="24" fill="{f('Chest')}" stroke="{sk}" stroke-width="2"/></g>
  <g filter="url(#sh)"><rect x="115" y="195" width="90" height="70" rx="22" fill="{f('Abdomen')}" stroke="{sk}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M110 132 C80 145,72 180,78 220 C82 250,92 270,100 290 C108 310,115 320,120 320 L120 130Z" fill="{f('Left Arm')}" stroke="{sk}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M210 132 C240 145,248 180,242 220 C238 250,228 270,220 290 C212 310,205 320,200 320 L200 130Z" fill="{f('Right Arm')}" stroke="{sk}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M135 265 C120 310,118 360,126 410 C132 445,132 475,128 500 L155 500 C158 470,160 435,156 405 C150 355,152 312,165 265Z" fill="{f('Left Leg')}" stroke="{sk}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M185 265 C200 310,202 360,194 410 C188 445,188 475,192 500 L165 500 C162 470,160 435,164 405 C170 355,168 312,155 265Z" fill="{f('Right Leg')}" stroke="{sk}" stroke-width="2"/></g>
</svg>""".strip()

def chip_grid(prefix: str, options: List[str], active_list: List[str], on_toggle, ncols=3):
    for row in [options[i:i+ncols] for i in range(0, len(options), ncols)]:
        cs = st.columns(len(row))
        for col, opt in zip(cs, row):
            with col:
                cls = "chip-btn-active" if opt in active_list else "chip-btn"
                st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
                if st.button(("✓ " if opt in active_list else "") + opt, key=f"{prefix}__{opt}"):
                    on_toggle(opt); st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

def render_progress(stage: int):
    pct = max(0, min(100, int((max(stage,0) / TOTAL_STAGES) * 100)))
    st.markdown(f"""
    <div class="progress-bar-wrap">
      <div class="progress-label">
        <span>Step {max(stage,0)+1} of {TOTAL_STAGES+1} · <strong>{STAGE_LABELS.get(stage,"")}</strong></span>
        <span>{pct}%</span>
      </div>
      <div class="progress-track"><div class="progress-fill" style="width:{pct}%"></div></div>
    </div>""", unsafe_allow_html=True)

# ============================================================
# SCREEN -1: GREETING
# ============================================================
if st.session_state.stage == -1:
    st.markdown("""
    <div class="greeting-wrap">
      <div class="greeting-icon">🩺</div>
      <div class="greeting-title">Daily Symptom Check-In</div>
      <div class="greeting-text">
        Hi — I'm your virtual check-in assistant.<br>
        I'll ask a few quick questions about how you're feeling today.<br>
        <strong>It only takes 2–3 minutes.</strong>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="max-width:360px;margin:0 auto;">', unsafe_allow_html=True)
    name_input = st.text_input("Your name (optional):", value=st.session_state.patient_name,
                               placeholder="e.g. Alex")
    st.markdown('<div class="big-btn">', unsafe_allow_html=True)
    if st.button("▶  Start Check-In", key="start_btn"):
        st.session_state.patient_name = name_input.strip() or "Patient"
        st.session_state.stage = 0
        add_doctor(
            f"Hi {st.session_state.patient_name} 👋 — I'm your virtual check-in assistant. "
            "Let's do a quick symptom check-in. I'll ask a few short questions."
        )
        st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()

# ============================================================
# PROGRESS BAR (stages 0–6)
# ============================================================
render_progress(st.session_state.stage)

# ============================================================
# SCREEN 0: PAIN GATE
# ============================================================
if st.session_state.stage == 0:
    st.markdown("""
    <div class="card">
      <div class="card-title">Do you have any pain today?</div>
      <div class="card-sub">Tap one button below.</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="pain-yes">', unsafe_allow_html=True)
        if st.button("😣  Yes, I have pain", key="pain_yes"):
            st.session_state.pain_yesno = True
            add_patient("Yes, I have pain today.")
            add_doctor("I'm sorry to hear that. Please show me where on your body you feel pain.")
            st.session_state.stage = 1; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="pain-no">', unsafe_allow_html=True)
        if st.button("😊  No pain today", key="pain_no"):
            st.session_state.pain_yesno = False
            add_patient("No pain today.")
            add_doctor("Great! Let's check on any other symptoms you might have.")
            st.session_state.stage = 3; st.rerun()  # skip body map + pain chars
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SCREEN 1: BODY MAP
# ============================================================
elif st.session_state.stage == 1:
    st.markdown("""
    <div class="card">
      <div class="card-title">Where do you feel pain?</div>
      <div class="card-sub">Tap body regions — select as many as apply. The figure highlights your selections.</div>
    </div>""", unsafe_allow_html=True)

    col_svg, col_btns = st.columns([1, 1.1], gap="large")
    with col_svg:
        st.markdown(body_svg(st.session_state.selected_parts), unsafe_allow_html=True)
    with col_btns:
        for part in ["Head","Chest","Abdomen","Left Arm","Right Arm","Left Leg","Right Leg","Other"]:
            is_on = part in st.session_state.selected_parts
            cls = "body-btn-active" if is_on else "body-btn"
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            if st.button(("✓ " if is_on else "   ") + part, key=f"bp_{part}"):
                toggle_body_part(part); st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        if st.session_state.selected_parts:
            st.markdown(f'<p style="font-size:12px;color:#6b7a90;margin-top:4px;">✓ {", ".join(sorted(st.session_state.selected_parts))}</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="nav-clear">', unsafe_allow_html=True)
        if st.button("Clear all", key="clear_body"):
            st.session_state.selected_parts = set(); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="nav-next">', unsafe_allow_html=True)
        if st.button("Next →", key="body_next"):
            loc_str = ", ".join(sorted(st.session_state.selected_parts)) if st.session_state.selected_parts else "not specified"
            add_patient(f"Pain locations: {loc_str}.")
            for loc in st.session_state.selected_parts:
                if loc not in st.session_state.structured["pain_locations"]:
                    st.session_state.structured["pain_locations"].append(loc)
            add_doctor("Thanks. Can you tell me more about the pain?")
            st.session_state.stage = 2; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SCREEN 2: PAIN CHARACTERISTICS
# ============================================================
elif st.session_state.stage == 2:
    pc = st.session_state.pain_characteristics
    st.markdown("""
    <div class="card">
      <div class="card-title">Tell me about your pain</div>
      <div class="card-sub">Tap all that apply — no typing needed.</div>
    """, unsafe_allow_html=True)

    # Severity — single-select segmented bar
    st.markdown('<div class="section-head">Severity</div>', unsafe_allow_html=True)
    sev_opts = ["Mild (1–3)", "Moderate (4–6)", "Severe (7–9)", "Worst ever (10)"]
    sev_cols = st.columns(4)
    for col, sev in zip(sev_cols, sev_opts):
        with col:
            cls = "sev-chip-active" if pc["severity"] == sev else "sev-chip"
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            if st.button(sev, key=f"sev_{sev}"):
                pc["severity"] = sev; st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # Type
    st.markdown('<div class="section-head">Type of pain</div>', unsafe_allow_html=True)
    chip_grid("pt", ["Burning","Stabbing","Dull / Aching","Cramping","Pressure","Throbbing","Tingling","Shooting"],
              pc["types"], lambda x: toggle_pc("types", x), ncols=4)

    # Timing
    st.markdown('<div class="section-head">Timing</div>', unsafe_allow_html=True)
    chip_grid("tim", ["New / Getting worse","Same as usual","Getting better","Constant","Comes and goes"],
              pc["timing"], lambda x: toggle_pc("timing", x), ncols=3)

    # Triggers
    st.markdown('<div class="section-head">What makes it worse or better?</div>', unsafe_allow_html=True)
    chip_grid("trig", ["Swallowing","Eating","Movement","Rest helps","Medication helps","Nothing helps"],
              pc["triggers"], lambda x: toggle_pc("triggers", x), ncols=3)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="nav-next" style="max-width:200px;margin:0 auto;">', unsafe_allow_html=True)
    if st.button("Next →", key="pain_chars_next"):
        parts = []
        if pc["severity"]: parts.append(f"Severity: {pc['severity']}")
        if pc["types"]:    parts.append(f"Type: {', '.join(pc['types'])}")
        if pc["timing"]:   parts.append(f"Timing: {', '.join(pc['timing'])}")
        if pc["triggers"]: parts.append(f"Triggers/relief: {', '.join(pc['triggers'])}")
        add_patient(". ".join(parts) + "." if parts else "No additional pain details.")
        add_doctor("Got it. Let's now check on any other symptoms you may have today.")
        st.session_state.stage = 3; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SCREEN 3: SYMPTOM CHECKLIST
# ============================================================
elif st.session_state.stage == 3:
    st.markdown("""
    <div class="card">
      <div class="card-title">Any other symptoms today?</div>
      <div class="card-sub">Tap everything that applies. Select as many as you like, then tap Next.</div>
    </div>""", unsafe_allow_html=True)

    chip_grid("sym",
        ["Fatigue / low energy","Nausea","Vomiting","Poor appetite",
         "Mouth sores","Trouble swallowing","Shortness of breath","Fever / chills",
         "Constipation","Diarrhea","Sleep problems","Anxiety / low mood","Dizziness","Headache"],
        st.session_state.symptoms, toggle_symptom, ncols=3)

    st.markdown('<div class="nav-next" style="max-width:220px;margin:16px auto 0 auto;">', unsafe_allow_html=True)
    if st.button("Next →", key="sym_next"):
        sym_str = "; ".join(st.session_state.symptoms) if st.session_state.symptoms else "None"
        add_patient(f"Symptoms: {sym_str}.")
        for s in st.session_state.symptoms:
            if s not in st.session_state.structured["symptoms"]:
                st.session_state.structured["symptoms"].append(s)
        add_doctor("Thanks for sharing. Is there anything else you want to report to your care team?")
        st.session_state.stage = 4; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SCREEN 4: "ANYTHING ELSE?" GATE
# ============================================================
elif st.session_state.stage == 4:
    st.markdown("""
    <div class="card">
      <div class="card-title">Anything else to report?</div>
      <div class="card-sub">Is there anything else you want your care team to know?</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="pain-no">', unsafe_allow_html=True)
        if st.button("No, that's everything", key="ae_no"):
            st.session_state.anything_else = False
            add_patient("Nothing else to report.")
            add_doctor("Perfect — let's review your check-in before you submit.")
            st.session_state.stage = 6; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="pain-yes">', unsafe_allow_html=True)
        if st.button("Yes, I want to add more", key="ae_yes"):
            st.session_state.anything_else = True
            add_patient("Yes, I have more to share.")
            add_doctor("Of course — please tell me. What else is on your mind?")
            st.session_state.api_chat_started = True
            st.session_state.stage = 5; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SCREEN 5: FREE-TEXT / LLM CHAT (gated)
# ============================================================
elif st.session_state.stage == 5:
    if st.session_state.submitted:
        st.success("✅ Your check-in has been submitted. Thank you!")
        st.stop()

    st.markdown("""
    <div class="card">
      <div class="card-title">Tell me more</div>
      <div class="card-sub">Describe anything else you're experiencing. I'll ask a few short follow-up questions and log your answers for your care team.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="chat-window">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "doctor":
            st.markdown(f'<div class="row-left"><div class="avatar">🩺</div><div class="bubble-doc">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="row-right"><div class="bubble-pat">{msg["content"]}</div><div class="avatar">🙂</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    user_text = st.chat_input("Type your message…")
    if user_text:
        add_patient(user_text)
        data = llm_turn()
        am = data.get("assistant_message","")
        if am: add_doctor(am)
        merge_extracted(data.get("extracted",{}) or {})
        sm = data.get("summary_for_clinician","") or ""
        if sm.strip(): st.session_state.structured["summary_for_clinician"] = sm.strip()
        if st.session_state.structured.get("urgency") == "emergency":
            st.error("⚠️ Please seek emergency medical attention immediately.")
        st.rerun()

    st.markdown('<div class="nav-next" style="max-width:280px;margin:16px auto 0 auto;">', unsafe_allow_html=True)
    if st.button("Done — Review my check-in →", key="chat_done"):
        add_patient("I'm done sharing.")
        add_doctor("Great — let's review everything before you submit.")
        st.session_state.stage = 6; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SCREEN 6: REVIEW + SUBMIT
# ============================================================
elif st.session_state.stage == 6:
    if st.session_state.submitted:
        st.markdown("""
        <div class="card" style="text-align:center;padding:40px 28px;">
          <div style="font-size:54px;margin-bottom:14px;">✅</div>
          <div class="card-title" style="font-size:24px;">Check-in submitted!</div>
          <div class="card-sub" style="font-size:15px;">Your care team has been notified.<br>Thank you for checking in today.</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    pc = st.session_state.pain_characteristics
    locs = ", ".join(sorted(st.session_state.selected_parts)) or "—"
    syms = ", ".join(st.session_state.symptoms) or "None"
    ae   = st.session_state.anything_else_text or ("Notes shared in chat" if st.session_state.anything_else else "No")

    st.markdown(f"""
    <div class="card">
      <div class="card-title">Review your check-in</div>
      <div class="card-sub">Please review before submitting — this will be sent to your care team.</div>
      <div class="review-row"><div class="review-label">🕐 Date &amp; time</div><div class="review-value">{datetime.now().strftime("%B %d, %Y · %I:%M %p")}</div></div>
      <div class="review-row"><div class="review-label">👤 Patient</div><div class="review-value">{st.session_state.patient_name}</div></div>
      <div class="review-row"><div class="review-label">🔴 Pain today?</div><div class="review-value">{"Yes" if st.session_state.pain_yesno else "No"}</div></div>
      <div class="review-row"><div class="review-label">📍 Pain location(s)</div><div class="review-value">{locs}</div></div>
      <div class="review-row"><div class="review-label">📊 Severity</div><div class="review-value">{pc.get("severity") or "—"}</div></div>
      <div class="review-row"><div class="review-label">🔥 Pain type</div><div class="review-value">{", ".join(pc.get("types",[])) or "—"}</div></div>
      <div class="review-row"><div class="review-label">⏱ Timing</div><div class="review-value">{", ".join(pc.get("timing",[])) or "—"}</div></div>
      <div class="review-row"><div class="review-label">💊 Triggers / relief</div><div class="review-value">{", ".join(pc.get("triggers",[])) or "—"}</div></div>
      <div class="review-row"><div class="review-label">🩹 Other symptoms</div><div class="review-value">{syms}</div></div>
      <div class="review-row"><div class="review-label">💬 Additional notes</div><div class="review-value">{ae}</div></div>
    </div>""", unsafe_allow_html=True)

    with st.expander("View AI clinical summary"):
        st.json(st.session_state.structured)

    st.markdown('<div class="submit-btn" style="max-width:580px;margin:0 auto;">', unsafe_allow_html=True)
    if st.button("✅  Submit Check-In", key="final_submit"):
        save_to_sheet()
        st.session_state.submitted = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="max-width:580px;margin:10px auto 0 auto;">', unsafe_allow_html=True)
    if st.button("← Go back to add more", key="back_to_chat"):
        st.session_state.stage = 5
        st.session_state.anything_else = True
        if not st.session_state.api_chat_started:
            add_doctor("What else would you like to share?")
            st.session_state.api_chat_started = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)If the patient reports any emergency symptom (e.g., trouble breathing, chest pain, heavy bleeding, fainting, confusion),
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
