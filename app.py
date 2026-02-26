# chatreport_app.py
# pip install streamlit

"""
ChatReport — Head & Neck Cancer Symptom Chatbot
================================================

Production-style Streamlit chatbot for structured symptom reporting.
All logic is local and rule-based (no APIs).

Run:
    streamlit run chatreport_app.py

Author: ChatGPT (Streamlit Healthcare UX)
"""

# ================================================================
# SECTION 0 — IMPORTS & CONFIG
# ================================================================

import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Callable

# MUST be first Streamlit command
st.set_page_config(
    page_title="ChatReport",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# SECTION 1 — CONSTANTS
# ================================================================

BOT_AVATAR = "🤖"
USER_AVATAR = "🧑"
APP_NAME = "ChatReport"

POSITIVE_KEYWORDS = [
    "yes",
    "a little",
    "somewhat",
    "a lot",
    "often",
    "constantly",
    "most",
    "on and off",
    "only",
    "occasionally",
    "noticeably",
    "quite",
    "very",
    "difficulty",
    "trouble",
    "reduced",
    "poor",
    "lost",
    "gained",
    "different",
]

NEGATIVE_KEYWORDS = [
    "no",
    "not really",
    "good",
    "well",
    "none",
]

BODY_REGIONS = [
    "Head", "Left Ear", "Right Ear", "Jaw", "Throat", "Neck",
    "Left Shoulder", "Right Shoulder", "Chest", "Left Arm",
    "Right Arm", "Abdomen"
]

DOMAINS = [
    ("pain", "Pain"),
    ("mouth", "Mouth"),
    ("swallow", "Swallowing"),
    ("nutrition", "Nutrition"),
    ("breathing", "Breathing"),
    ("fatigue", "Fatigue"),
    ("mood", "Mood"),
    ("other", "Other"),
]

# ================================================================
# SECTION 2 — CSS
# ================================================================

st.markdown("""
<style>
.chat-header {
    position: sticky;
    top: 0;
    z-index: 999;
    background: #008b8b;
    color: white;
    padding: 10px 15px;
    border-radius: 10px;
    font-weight: bold;
}
.bot-msg, .user-msg {
    padding: 10px 14px;
    border-radius: 18px;
    margin: 8px 0;
    max-width: 70%;
    font-size: 15px;
}
.bot-msg {
    background: #f3f4f6;
    margin-right: auto;
}
.user-msg {
    background: linear-gradient(135deg,#00bfa6,#00a0a0);
    color:white;
    margin-left:auto;
}
.timestamp {
    font-size:11px;
    opacity:0.6;
}
.chip button {
    border-radius:20px !important;
    margin:4px !important;
}
.scale-btn {
    border-radius:50% !important;
    width:38px !important;
    height:38px !important;
}
.report-box {
    background:#111;
    color:#00ffcc;
    font-family:monospace;
    padding:15px;
    border-radius:8px;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# SECTION 3 — HELPER FUNCTIONS
# ================================================================

def now_ts() -> str:
    """Return timestamp string."""
    return datetime.now().strftime("%H:%M")


def is_positive(answer: str) -> bool:
    """Determine if answer triggers conditional follow-ups."""
    if not answer:
        return False
    a = answer.lower()
    for w in POSITIVE_KEYWORDS:
        if w in a:
            return True
    for w in NEGATIVE_KEYWORDS:
        if w in a:
            return False
    return False


def add_message(role: str, text: str):
    """Append message to chat history."""
    st.session_state.messages.append({
        "role": role,
        "text": text,
        "ts": now_ts()
    })


# ================================================================
# SECTION 4 — REPORT GENERATION
# ================================================================

def generate_report(symptoms: Dict) -> str:
    """
    Create structured clinical report text.
    Standalone & testable.
    """
    name = symptoms.get("patient_name", "Unknown")
    dt = datetime.now().strftime("%Y-%m-%d %H:%M")

    def get(k):
        return symptoms.get(k, "N/A")

    lines = []
    lines.append(f"CHATREPORT CLINICAL SUMMARY")
    lines.append(f"Patient: {name}")
    lines.append(f"Generated: {dt}")
    lines.append("-" * 60)

    # 🔴 Symptoms present
    lines.append("🔴 SYMPTOMS PRESENT")
    if is_positive(get("pain_any")):
        lines.append(f"Pain location: {get('pain_location')}")
        lines.append(f"Pain severity: {get('pain_scale')}")
        lines.append(f"Pain frequency: {get('pain_freq')}")
    if is_positive(get("mouth_any")):
        lines.append(f"Mouth symptoms: {get('mouth_any')}")
    if is_positive(get("swallow_any")):
        lines.append(f"Swallowing difficulty: {get('swallow_any')}")
    if is_positive(get("breath_any")):
        lines.append(f"Breathing issues: {get('breath_any')}")

    # ✅ Not reported
    lines.append("\n✅ SYMPTOMS NOT REPORTED")
    if not is_positive(get("pain_any")):
        lines.append("No pain reported")
    if not is_positive(get("mouth_any")):
        lines.append("No mouth symptoms")
    if not is_positive(get("swallow_any")):
        lines.append("No swallowing difficulty")

    # 📊 Nutrition
    lines.append("\n📊 NUTRITIONAL STATUS")
    lines.append(f"Appetite: {get('appetite')}")
    lines.append(f"Weight change: {get('weight_change')}")
    lines.append(f"Nausea: {get('nausea')}")

    # 😴 Fatigue
    lines.append("\n😴 FATIGUE")
    lines.append(f"Fatigue level: {get('fatigue_scale')}")
    lines.append(f"Impact: {get('fatigue_text')}")

    # 💭 Mood
    lines.append("\n💭 EMOTIONAL WELLBEING")
    lines.append(f"Mood: {get('mood')}")
    lines.append(f"Anxiety: {get('anxiety')}")
    lines.append(f"Sleep: {get('sleep')}")
    lines.append(f"Support: {get('support')}")

    # 🩺 Other
    lines.append("\n🩺 OTHER SYMPTOMS")
    lines.append(f"Cough: {get('cough')}")
    lines.append(f"Skin: {get('skin')}")
    lines.append(f"Concentration: {get('focus')}")

    # Notes
    lines.append("\n📝 ADDITIONAL NOTES")
    lines.append(get("closing"))

    # Alerts
    alerts_high = []
    alerts_monitor = []

    try:
        pain_val = int(get("pain_scale"))
        if pain_val >= 7:
            alerts_high.append("Severe pain — review pain management")
        elif 4 <= pain_val <= 6:
            alerts_monitor.append("Moderate pain level")
    except:
        pass

    if "Liquids only" in str(get("swallow_food")) or "Feeding tube" in str(get("swallow_food")):
        alerts_high.append("Nutritional consult may be needed")

    if get("mood") in ["Very distressed", "Quite sad"]:
        alerts_high.append("Consider psychosocial referral")

    if is_positive(get("breath_any")):
        alerts_high.append("Assess for obstruction/infection")

    if "lost" in str(get("weight_change")).lower():
        alerts_monitor.append("Weight loss reported")

    if get("anxiety") in ["Yes, a lot", "Sometimes"]:
        alerts_monitor.append("Elevated anxiety")

    lines.append("\n🚨 CLINICAL ALERTS")
    for a in alerts_high:
        lines.append(f"🔴 {a}")
    for a in alerts_monitor:
        lines.append(f"⚠️ {a}")

    return "\n".join(lines)

# ================================================================
# SECTION 5 — STATE INIT
# ================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_step" not in st.session_state:
    st.session_state.current_step = "step1"
if "symptoms" not in st.session_state:
    st.session_state.symptoms = {}
if "conversation_done" not in st.session_state:
    st.session_state.conversation_done = False
if "report" not in st.session_state:
    st.session_state.report = None
if "selected_regions" not in st.session_state:
    st.session_state.selected_regions = []

# ================================================================
# SECTION 6 — FLOW DEFINITION
# ================================================================

FLOW = [
    {"id":"step1","type":"info","msg":
     "Hello! I'm ChatReport — a supportive tool to help you share how you've been feeling before your visit. "
     "This tool is for symptom reporting only. For urgent concerns, contact your care team immediately. "
     "Your responses are confidential and help your team prepare.","key":None,"next":"step2"},

    {"id":"step2","type":"text","msg":"What is your first name?","key":"patient_name","next":"step3"},
    {"id":"step3","type":"info","msg":"Nice to meet you, {name}! Let's go through some questions.","key":None,"next":"step4"},

    {"id":"step4","type":"chips","msg":"Have you had any pain since your last appointment?",
     "chips":["Yes","No","A little"],"key":"pain_any","next":"step5"},

    {"id":"step5","type":"body","msg":"I'm sorry to hear that. Can you show me where you're feeling the pain?",
     "key":"pain_location","cond":lambda s:is_positive(s.get("pain_any","")),"next":"step6"},

    {"id":"step6","type":"scale","msg":"On a scale of 0 to 10, how would you rate your pain at its worst?",
     "key":"pain_scale","cond":lambda s:is_positive(s.get("pain_any","")),"next":"step7"},

    {"id":"step7","type":"chips","msg":"How often do you experience this pain?",
     "chips":["Constantly","Most of the day","On and off","Only when swallowing","Only at night"],
     "key":"pain_freq","cond":lambda s:is_positive(s.get("pain_any","")),"next":"step8"},

    {"id":"step8","type":"text","msg":"Are you doing anything to manage the pain? (e.g., medication, ice, heat)",
     "key":"pain_manage","cond":lambda s:is_positive(s.get("pain_any","")),"next":"step9"},

    {"id":"step9","type":"text","msg":"How much is the pain affecting your daily life — eating, sleeping, activities?",
     "key":"pain_impact","cond":lambda s:is_positive(s.get("pain_any","")),"next":"step10"},

    {"id":"step10","type":"chips","msg":"Have you noticed any dryness, sores, or changes in your mouth?",
     "chips":["Yes","No","A little"],"key":"mouth_any","next":"step11"},

    {"id":"step11","type":"chips","msg":"Is your mouth feeling very dry?",
     "chips":["Yes, very dry","Somewhat dry","Not really"],
     "key":"mouth_dry","cond":lambda s:is_positive(s.get("mouth_any","")),"next":"step12"},

    {"id":"step12","type":"chips","msg":"Do you have any sores or ulcers inside your mouth?",
     "chips":["Yes","No","Not sure"],
     "key":"mouth_sores","cond":lambda s:is_positive(s.get("mouth_any","")),"next":"step13"},

    {"id":"step13","type":"chips","msg":"Have you noticed changes in how food tastes?",
     "chips":["Yes, very different","A little different","No change"],
     "key":"mouth_taste","cond":lambda s:is_positive(s.get("mouth_any","")),"next":"step14"},

    {"id":"step14","type":"text","msg":"How much are these mouth symptoms affecting your ability to eat or drink?",
     "key":"mouth_impact","cond":lambda s:is_positive(s.get("mouth_any","")),"next":"step15"},

    {"id":"step15","type":"chips","msg":"Have you had any difficulty swallowing?",
     "chips":["Yes","No","A little"],"key":"swallow_any","next":"step16"},

    {"id":"step16","type":"chips","msg":"Does swallowing cause you pain?",
     "chips":["Yes, a lot","A little","No"],
     "key":"swallow_pain","cond":lambda s:is_positive(s.get("swallow_any","")),"next":"step17"},

    {"id":"step17","type":"chips","msg":"What types of food are you able to eat right now?",
     "chips":["Regular food","Soft foods only","Pureed foods","Liquids only","Feeding tube only"],
     "key":"swallow_food","cond":lambda s:is_positive(s.get("swallow_any","")),"next":"step18"},

    {"id":"step18","type":"chips","msg":"Have you had any choking or coughing when eating?",
     "chips":["Yes, often","Occasionally","No"],
     "key":"swallow_choke","cond":lambda s:is_positive(s.get("swallow_any","")),"next":"step19"},

    {"id":"step19","type":"chips","msg":"How is your appetite lately?",
     "chips":["Good","Reduced","Very poor","No appetite at all"],"key":"appetite","next":"step20"},

    {"id":"step20","type":"chips","msg":"Have you noticed any weight changes recently?",
     "chips":["Yes, lost weight","Yes, gained weight","No change","Not sure"],"key":"weight_change","next":"step21"},

    {"id":"step21","type":"text","msg":"About how much weight have you lost, and over how long?",
     "key":"weight_loss_text","cond":lambda s:"lost" in s.get("weight_change","").lower(),"next":"step22"},

    {"id":"step22","type":"chips","msg":"Have you experienced any nausea or vomiting?",
     "chips":["Yes, often","Occasionally","No"],"key":"nausea","next":"step23"},

    {"id":"step23","type":"chips","msg":"Are you using any nutritional supplements (like Ensure or Boost)?",
     "chips":["Yes","No","Sometimes"],"key":"supplements","next":"step24"},

    {"id":"step24","type":"chips","msg":"Have you had any shortness of breath or difficulty breathing?",
     "chips":["Yes","No","A little"],"key":"breath_any","next":"step25"},

    {"id":"step25","type":"text","msg":"Can you describe when it happens — at rest, walking, at night?",
     "key":"breath_desc","cond":lambda s:is_positive(s.get("breath_any","")),"next":"step26"},

    {"id":"step26","type":"scale","msg":"On a scale of 0–10, how would you rate your fatigue or tiredness?",
     "key":"fatigue_scale","next":"step27"},

    {"id":"step27","type":"text","msg":"How is the fatigue affecting your daily activities?",
     "key":"fatigue_text","cond":lambda s:int(s.get("fatigue_scale","0"))>=4 if s.get("fatigue_scale") else False,"next":"step28"},

    {"id":"step28","type":"chips","msg":"How would you describe your mood lately?",
     "chips":["Good / Positive","A bit down","Anxious or worried","Quite sad","Very distressed"],
     "key":"mood","next":"step29"},

    {"id":"step29","type":"chips","msg":"Have you been feeling worried or anxious about your treatment?",
     "chips":["Yes, a lot","Sometimes","Not really"],"key":"anxiety","next":"step30"},

    {"id":"step30","type":"chips","msg":"How has your sleep been?",
     "chips":["Sleeping well","Some trouble","Difficulty most nights","Can't sleep at all"],
     "key":"sleep","next":"step31"},

    {"id":"step31","type":"chips","msg":"Do you feel you have enough support from family or your care team?",
     "chips":["Yes, feel supported","Somewhat","No, need more support"],
     "key":"support","next":"step32"},

    {"id":"step32","type":"chips","msg":"Have you had any bothersome cough?",
     "chips":["Yes","No","A little"],"key":"cough","next":"step33"},

    {"id":"step33","type":"chips","msg":"Have you noticed any skin changes in the treatment area?",
     "chips":["Yes","No","A little"],"key":"skin","next":"step34"},

    {"id":"step34","type":"chips","msg":"Have you had difficulty concentrating or remembering things?",
     "chips":["Yes, noticeably","A little","No"],"key":"focus","next":"step35"},

    {"id":"step35","type":"text","msg":"Is there anything else you'd like your care team to know?",
     "key":"closing","next":"step36"},

    {"id":"step36","type":"info","msg":
     "Thank you so much, {name}! Your responses have been recorded. "
     "Your care team will review your report before your appointment. "
     "This tool is for symptom reporting only. For urgent concerns, contact your care team immediately. 💙",
     "key":None,"next":None},
]

FLOW_MAP = {s["id"]:s for s in FLOW}

# ================================================================
# SECTION 7 — NAVIGATION
# ================================================================

def get_next_step(current_id:str, symptoms:Dict)->Optional[str]:
    """Determine next valid step considering conditions."""
    step = FLOW_MAP[current_id]
    nxt = step.get("next")
    while nxt:
        nxt_step = FLOW_MAP[nxt]
        cond = nxt_step.get("cond")
        if cond is None or cond(symptoms):
            return nxt
        nxt = nxt_step.get("next")
    return None

# ================================================================
# SECTION 8 — RENDER CHAT HEADER
# ================================================================

total_steps = len(FLOW)
idx = [i for i,s in enumerate(FLOW) if s["id"]==st.session_state.current_step]
progress = (idx[0]+1)/total_steps if idx else 0

st.markdown(f"""
<div class="chat-header">
{APP_NAME} — Symptom Reporting
<br>
<progress value="{progress}" max="1" style="width:100%"></progress>
</div>
""", unsafe_allow_html=True)

# ================================================================
# SECTION 9 — SIDEBAR PROGRESS
# ================================================================

st.sidebar.title("Progress")

for key,label in DOMAINS:
    done = any(k.startswith(key) for k in st.session_state.symptoms.keys())
    st.sidebar.write(f"{'✅' if done else '⏳'} {label}")

# ================================================================
# SECTION 10 — RENDER CHAT HISTORY
# ================================================================

for m in st.session_state.messages:
    if m["role"]=="bot":
        st.markdown(f"""
        <div class="bot-msg">{BOT_AVATAR} {m['text']}<div class="timestamp">{m['ts']}</div></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="user-msg">{m['text']} {USER_AVATAR}<div class="timestamp">{m['ts']}</div></div>
        """, unsafe_allow_html=True)

# ================================================================
# SECTION 11 — CURRENT STEP LOGIC
# ================================================================

def show_bot_message_if_needed():
    """Show bot question once."""
    if not st.session_state.messages or \
       st.session_state.messages[-1]["role"]=="user":

        step = FLOW_MAP[st.session_state.current_step]
        txt = step["msg"]
        name = st.session_state.symptoms.get("patient_name","")
        txt = txt.replace("{name}", name if name else "")
        add_message("bot", txt)

show_bot_message_if_needed()

step = FLOW_MAP[st.session_state.current_step]

# ================================================================
# SECTION 12 — INPUT WIDGETS
# ================================================================

def submit_answer(val:str):
    """Handle answer submission and move flow."""
    add_message("user", val)
    key = step.get("key")
    if key:
        st.session_state.symptoms[key] = val

    # empathetic safety message
    if key=="pain_scale":
        try:
            if int(val)>=8:
                add_message("bot","I'm really sorry you're dealing with severe pain. Your care team will see this.")
        except: pass
    if key=="mood" and val=="Very distressed":
        add_message("bot","Thank you for sharing how you're feeling — your care team will review this and support you.")

    nxt = get_next_step(step["id"], st.session_state.symptoms)
    if nxt:
        st.session_state.current_step = nxt
    else:
        st.session_state.conversation_done=True
        st.session_state.report = generate_report(st.session_state.symptoms)
    st.rerun()


if not st.session_state.conversation_done:

    if step["type"]=="chips":
        cols = st.columns(len(step["chips"]))
        for i,ch in enumerate(step["chips"]):
            if cols[i].button(ch, key=f"{step['id']}_{i}"):
                submit_answer(ch)

    elif step["type"]=="scale":
        cols = st.columns(11)
        for i in range(11):
            color="#00aa55" if i<=3 else "#ffaa00" if i<=6 else "#ff4444"
            if cols[i].button(str(i), key=f"{step['id']}_{i}"):
                submit_answer(str(i))

    elif step["type"]=="body":
        st.write("Select areas:")
        selected = []
        cols = st.columns(3)
        for idx,r in enumerate(BODY_REGIONS):
            if cols[idx%3].checkbox(r, key=f"body_{r}"):
                selected.append(r)
        if st.button("Confirm Selection"):
            submit_answer(", ".join(selected))

    elif step["type"]=="text":
        user_text = st.text_input("Type here", key=f"text_{step['id']}")
        if st.button("Send"):
            if user_text.strip():
                submit_answer(user_text.strip())

    elif step["type"]=="info":
        # auto advance
        nxt = get_next_step(step["id"], st.session_state.symptoms)
        if nxt:
            st.session_state.current_step = nxt
            st.rerun()

# ================================================================
# SECTION 13 — REPORT VIEW
# ================================================================

if st.session_state.conversation_done and st.session_state.report:
    st.markdown("### Clinical Report")
    st.markdown(f"<div class='report-box'><pre>{st.session_state.report}</pre></div>", unsafe_allow_html=True)

    st.download_button(
        "Download Report (.txt)",
        st.session_state.report,
        file_name="chatreport.txt"
    )

    if st.button("Start Over"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
