import hashlib
import json
from datetime import datetime
from typing import Dict, List, Set, Optional

import streamlit as st

# Third-party
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI

# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ============================================================
# UTIL: Secrets helpers
# ============================================================
def _secret(*keys: str, default=None):
    for k in keys:
        if k in st.secrets:
            return st.secrets[k]
    return default

def _require_secret(*keys: str) -> str:
    v = _secret(*keys)
    if v is None:
        raise KeyError(f"Missing required secret. Tried: {', '.join(keys)}")
    return v

# ============================================================
# OPENAI CLIENT
# ============================================================
OPENAI_API_KEY = _secret("openai_api_key", "OPENAI_API_KEY", "openai_key")
openai_client: Optional[OpenAI] = None
openai_init_error: Optional[str] = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        openai_init_error = str(e)
else:
    openai_init_error = "OpenAI API key not found in Streamlit secrets."

# ============================================================
# GOOGLE SHEETS SETUP
# ============================================================
sheet = None
sheets_init_error: Optional[str] = None

def _init_sheets():
    global sheet, sheets_init_error
    if sheet is not None or sheets_init_error is not None:
        return
    try:
        gcp_sa = _require_secret("gcp_service_account")
        gsheet_id = _require_secret("gsheet_id")
        scope = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(gcp_sa, scopes=scope)
        gs_client = gspread.authorize(creds)
        book = gs_client.open_by_key(gsheet_id)
        try:
            sheet_local = book.worksheet("Form")
        except Exception:
            sheet_local = book.add_worksheet(title="Form", rows=2000, cols=20)
            sheet_local.append_row(["timestamp", "name", "json"])
        sheet = sheet_local
    except Exception as e:
        sheets_init_error = str(e)

# ============================================================
# PERSISTENCE: load/save check-ins
# ============================================================
def load_past_checkins(name: str) -> List[Dict]:
    _init_sheets()
    if sheet is None: return []
    try:
        all_rows = sheet.get_all_values()
        past: List[Dict] = []
        for row in all_rows[1:]:
            if len(row) >= 3 and row[1].strip().lower() == name.strip().lower():
                try:
                    data = json.loads(row[2])
                    data["timestamp"] = row[0]
                    past.append(data)
                except Exception: continue
        return past[-5:]
    except Exception: return []

def save_to_sheet() -> None:
    _init_sheets()
    if sheet is None: raise RuntimeError(f"Google Sheets not available: {sheets_init_error}")
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
# PROMPTING
# ============================================================
def build_system_prompt() -> str:
    name = st.session_state.get("patient_name", "the patient")
    feeling = st.session_state.get("feeling_level", None)
    pain = st.session_state.get("pain_yesno", None)
    locations = sorted(list(st.session_state.get("selected_parts", set())))
    symptoms = st.session_state.get("symptoms", [])

    session_lines = []
    if feeling is not None: session_lines.append(f"- Feeling level: {feeling}/10")
    if pain is not None: session_lines.append(f"- Pain today: {'yes' if pain else 'no'}")
    if locations: session_lines.append(f"- Pain locations: {', '.join(locations)}")
    if symptoms: session_lines.append(f"- Symptoms: {', '.join(symptoms)}")
    session_str = "\n".join(session_lines) if session_lines else "Check-in just started."

    past = st.session_state.get("past_checkins", [])
    memory_str = "\n".join([f"  [{p.get('timestamp')}] Feeling: {p.get('feeling_level')}/10 | Pain: {'yes' if p.get('pain') else 'no'}" for p in past]) if past else "No history."

    return f"""You are a warm, empathetic assistant for {name}. React to widget submissions and ask ONE follow-up.
TODAY: {session_str}
HISTORY: {memory_str}
RULES: 1. Warm/Short. 2. NO medical advice. 3. ONE question at a time."""

def get_gpt_reply() -> str:
    if not (openai_client and not openai_init_error): return "LLM not configured."
    msgs = [{"role": "system", "content": build_system_prompt()}]
    for msg in st.session_state.messages[-15:]:
        role = "assistant" if msg["role"] == "doctor" else "user"
        msgs.append({"role": role, "content": msg["content"]})
    try:
        response = openai_client.chat.completions.create(
            model=_secret("openai_model", default="gpt-4o-mini"),
            messages=msgs, max_tokens=350, temperature=0.6
        )
        return response.choices[0].message.content.strip()
    except Exception as e: return f"Error: {e}"

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{ background: linear-gradient(135deg,#eef4ff,#f6fbff); }
.chat-window{ max-height:40vh; overflow-y:auto; padding:18px; border-radius:18px; background:rgba(255,255,255,0.5); border:1px solid #d0d8e6; margin-bottom: 20px;}
.bubble-doc{ background:white; border-radius:18px; padding:12px 15px; max-width:80%; border:1px solid #eee; margin-bottom:10px; }
.bubble-pat{ background:#1f7aff; color:white; border-radius:18px; padding:12px 15px; max-width:80%; margin-left:auto; margin-bottom:10px; }
.panel{ background:white; padding:20px; border-radius:18px; border:1px solid #d0d8e6; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
.stButton>button { width: 100%; border-radius: 14px; font-weight: 600; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INIT
# ============================================================
defaults = {"messages": [], "stage": -1, "patient_name": "", "selected_parts": set(), "pain_yesno": None, "feeling_level": None, "symptoms": [], "submitted": False, "past_checkins": [], "gpt_followup_done": set()}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

def add_doctor(t): st.session_state.messages.append({"role": "doctor", "content": t})
def add_patient(t): st.session_state.messages.append({"role": "patient", "content": t})

def body_svg(selected):
    def f(p): return "#1f7aff" if p in selected else "#cfd8e6"
    return f'<svg width="220" height="400" viewBox="0 0 320 520"><circle cx="160" cy="70" r="38" fill="{f("Head")}"/><rect x="110" y="120" width="100" height="70" rx="20" fill="{f("Chest")}"/><rect x="115" y="195" width="90" height="70" rx="20" fill="{f("Abdomen")}"/><path d="M110 132 L 80 220 L 120 130 Z" fill="{f("Left Arm")}"/><path d="M210 132 L 240 220 L 200 130 Z" fill="{f("Right Arm")}"/><path d="M135 265 L 128 500 L 165 265 Z" fill="{f("Left Leg")}"/><path d="M185 265 L 192 500 L 155 265 Z" fill="{f("Right Leg")}"/></svg>'

# ============================================================
# MAIN PAGE ROUTING
# ============================================================
st.markdown('<div class="header">🩺 Cancer Symptom Check-In</div>', unsafe_allow_html=True)

# Always show Chat Window at the top (except during name entry)
if st.session_state.stage >= 0:
    st.markdown('<div class="chat-window">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        cls = "bubble-doc" if m["role"] == "doctor" else "bubble-pat"
        st.markdown(f'<div class="{cls}">{m["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: NAME ENTRY ---
if st.session_state.stage == -1:
    st.markdown('<div class="panel"><h3>Welcome</h3>', unsafe_allow_html=True)
    name = st.text_input("Enter your name:")
    if st.button("Start Check-In"):
        if name:
            st.session_state.patient_name = name
            st.session_state.past_checkins = load_past_checkins(name)
            add_doctor(get_gpt_reply())
            st.session_state.stage = 0
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 0: FEELING ---
elif st.session_state.stage == 0:
    st.markdown('<div class="panel"><b>How are you feeling today? (0-10)</b>', unsafe_allow_html=True)
    cols = st.columns(6)
    for i in range(11):
        with cols[i % 6]:
            if st.button(str(i), key=f"f_{i}"):
                st.session_state.feeling_level = i
                add_patient(f"I'm at a {i}/10.")
                add_doctor(get_gpt_reply())
                st.session_state.stage = 1
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 1: PAIN BINARY ---
elif st.session_state.stage == 1:
    st.markdown('<div class="panel"><b>Are you experiencing any pain?</b>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("✅ Yes"):
        st.session_state.pain_yesno = True
        add_patient("Yes, I have pain.")
        add_doctor(get_gpt_reply())
        st.session_state.stage = 2
        st.rerun()
    if c2.button("❌ No"):
        st.session_state.pain_yesno = False
        add_patient("No pain today.")
        add_doctor(get_gpt_reply())
        st.session_state.stage = 3
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 2: BODY MAP ---
elif st.session_state.stage == 2:
    st.markdown('<div class="panel"><b>Where is the pain located?</b>', unsafe_allow_html=True)
    l, r = st.columns([1,1])
    l.markdown(body_svg(st.session_state.selected_parts), unsafe_allow_html=True)
    for p in ["Head", "Chest", "Abdomen", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]:
        label = f"✓ {p}" if p in st.session_state.selected_parts else p
        if r.button(label, key=f"p_{p}"):
            if p in st.session_state.selected_parts: st.session_state.selected_parts.remove(p)
            else: st.session_state.selected_parts.add(p)
            st.rerun()
    if st.button("Confirm Locations ➜"):
        add_patient(f"Locations: {', '.join(st.session_state.selected_parts)}")
        add_doctor(get_gpt_reply())
        st.session_state.stage = 3
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 3: SYMPTOMS ---
elif st.session_state.stage == 3:
    st.markdown('<div class="panel"><b>Select other symptoms:</b>', unsafe_allow_html=True)
    opts = ["Fatigue", "Nausea", "Poor appetite", "Shortness of breath", "Constipation", "Sleep problems", "Anxiety"]
    cols = st.columns(2)
    for idx, s in enumerate(opts):
        with cols[idx % 2]:
            sel = s in st.session_state.symptoms
            if st.button(f"{'✅ ' if sel else ''}{s}", key=f"s_{s}"):
                if sel: st.session_state.symptoms.remove(s)
                else: st.session_state.symptoms.append(s)
                st.rerun()
    if st.button("Send Symptoms ➜"):
        add_patient(f"Symptoms: {', '.join(st.session_state.symptoms) if st.session_state.symptoms else 'None'}")
        add_doctor(get_gpt_reply())
        st.session_state.stage = 4
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 4: FINAL SUBMIT ---
elif st.session_state.stage == 4:
    if st.session_state.submitted:
        st.success("✅ Submitted to your care team. Thank you!")
    else:
        st.markdown('<div class="panel"><b>Final step: Anything else?</b>', unsafe_allow_html=True)
        txt = st.chat_input("Message your doctor...")
        if txt:
            add_patient(txt); add_doctor(get_gpt_reply()); st.rerun()
        if st.button("✅ Finish & Submit"):
            save_to_sheet(); st.session_state.submitted = True; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
