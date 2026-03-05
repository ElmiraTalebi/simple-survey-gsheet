import hashlib
import json
from datetime import datetime
from typing import Dict, List, Set, Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI

st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ── Secrets ────────────────────────────────────────────────
def _secret(*keys, default=None):
    for k in keys:
        if k in st.secrets: return st.secrets[k]
    return default

def _require_secret(*keys):
    v = _secret(*keys)
    if v is None: raise KeyError(f"Missing secret. Tried: {', '.join(keys)}")
    return v

# ── OpenAI ─────────────────────────────────────────────────
OPENAI_API_KEY = _secret("openai_api_key", "OPENAI_API_KEY", "openai_key")
openai_client: Optional[OpenAI] = None
openai_init_error: Optional[str] = None
if OPENAI_API_KEY:
    try: openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e: openai_init_error = str(e)
else: openai_init_error = "OpenAI API key not found."

# ── Google Sheets ───────────────────────────────────────────
sheet = None
sheets_init_error: Optional[str] = None

def _init_sheets():
    global sheet, sheets_init_error
    if sheet is not None or sheets_init_error is not None: return
    try:
        creds = Credentials.from_service_account_info(
            _require_secret("gcp_service_account"),
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        book = gspread.authorize(creds).open_by_key(_require_secret("gsheet_id"))
        try: sheet_local = book.worksheet("Form")
        except Exception:
            sheet_local = book.add_worksheet(title="Form", rows=2000, cols=20)
            sheet_local.append_row(["timestamp", "name", "json"])
        sheet = sheet_local
    except Exception as e: sheets_init_error = str(e)

def load_past_checkins(name: str) -> List[Dict]:
    _init_sheets()
    if sheet is None: return []
    try:
        past = []
        for row in sheet.get_all_values()[1:]:
            if len(row) >= 3 and row[1].strip().lower() == name.strip().lower():
                try:
                    d = json.loads(row[2]); d["timestamp"] = row[0]; past.append(d)
                except: continue
        return past[-5:]
    except: return []

def save_to_sheet():
    _init_sheets()
    if sheet is None: raise RuntimeError(f"Sheets unavailable: {sheets_init_error}")
    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        st.session_state.get("patient_name", "Unknown"),
        json.dumps({
            "feeling_level":  st.session_state.feeling_level,
            "pain":           st.session_state.pain_yesno,
            "pain_trend":     st.session_state.get("pain_trend"),
            "pain_locations": sorted(list(st.session_state.selected_parts)),
            "pain_severities": st.session_state.get("pain_severities", {}),
            "symptoms":       st.session_state.symptoms,
            "conversation":   st.session_state.messages,
        })
    ])

# ── Helpers ─────────────────────────────────────────────────
def _openai_ready():
    return openai_client is not None and openai_init_error is None

def build_system_prompt(extra_context: str = "") -> str:
    name     = st.session_state.get("patient_name", "the patient")
    feeling  = st.session_state.get("feeling_level", None)
    pain     = st.session_state.get("pain_yesno", None)
    locs     = sorted(list(st.session_state.get("selected_parts", set())))
    symptoms = st.session_state.get("symptoms", [])

    lines = []
    if feeling  is not None: lines.append(f"- Feeling: {feeling}")
    if pain     is not None: lines.append(f"- Pain: {'yes' if pain else 'no'}")
    if locs:                 lines.append(f"- Pain locations: {', '.join(locs)}")
    if symptoms:             lines.append(f"- Symptoms: {', '.join(symptoms)}")
    session_str = "\n".join(lines) if lines else "Nothing collected yet."

    past = st.session_state.get("past_checkins", [])
    if past:
        mem = [f"  [{p.get('timestamp','?')}] Feeling:{p.get('feeling_level','?')} | "
               f"Pain:{'yes' if p.get('pain') else 'no'} | "
               f"Locations:{', '.join(p.get('pain_locations',[])) or 'none'} | "
               f"Symptoms:{', '.join(p.get('symptoms',[])) or 'none'}" for p in past]
        memory_str = "\n".join(mem)
    else:
        memory_str = "No previous check-ins."

    ctx = f"\nCURRENT TASK:\n{extra_context}\n" if extra_context else ""

    return f"""You are a virtual symptom-intake assistant for a cancer care clinic.
Daily check-in with: {name}.

TODAY'S DATA: {session_str}
PATIENT HISTORY: {memory_str}
{ctx}
STRICT OUTPUT RULES:
1. Output ONLY a single short question. Nothing else.
2. NEVER start with "Thank you", "I'm sorry", "Great", "I see", "I understand",
   "Of course", "That's", "I'm glad", or ANY acknowledgement.
3. No preamble. No explanation. Just the question.
4. A very brief empathetic lead-in (max 6 words) is allowed only when the patient
   shares something emotionally significant — otherwise skip it.
5. Never give medical advice. If asked: "Your care team will follow up."
"""

# ── Curated Follow-Up Question Banks ───────────────────────
# Instead of letting GPT freely generate questions, we use curated clinical
# questions and only ask them when the answer is CONCERNING.

PAIN_FOLLOWUPS = {
    "severity_high": [
        "How long has this pain been this intense?",
        "Does it affect your ability to eat or swallow?",
        "Is this pain constant or does it come and go?",
    ],
    "new_location": [
        "When did you first notice pain in this area?",
        "Is it sharp, dull, or aching?",
        "Does anything make it better or worse?",
    ],
    "worsening": [
        "When did it start getting worse?",
        "Has anything changed in your treatment recently?",
        "Is the worsening sudden or gradual?",
    ],
}

SYMPTOM_FOLLOWUPS = {
    "eating_drinking_less": "Are you mostly on liquids, or are you able to eat soft foods too?",
    "mouth_tongue_sore": "Is the sore making it hard to eat or drink?",
    "trouble_swallowing": "Are you coughing or choking when you swallow?",
    "dry_mouth_thick_mucus": "Is the dryness or mucus making swallowing more painful?",
    "constipation": "About how many days has it been since your last bowel movement?",
    "nausea_vomiting": "Are you able to keep food and fluids down?",
    "dizziness_weakness": "Does it happen when you stand up or walk around?",
    "fever_chills": "Have you had any recent fevers or chills?",
    "cough_breathing": "Are you breathing okay, or is the cough getting worse?",
    "hearing_ringing": "Is it affecting your hearing, or is it more like ringing or fullness?",
    "numbness_tingling": "Is the numbness or tingling getting worse or staying mild?",
    "trouble_sleeping": "Is pain or another symptom keeping you awake?",
}

# Map UI symptom labels to canonical keys in SYMPTOM_FOLLOWUPS
SYMPTOM_KEY_MAP = {
    "Eating or drinking less": "eating_drinking_less",
    "Mouth or tongue sore": "mouth_tongue_sore",
    "Trouble swallowing": "trouble_swallowing",
    "Dry mouth / thick mucus": "dry_mouth_thick_mucus",
    "Constipation": "constipation",
    "Nausea / vomiting": "nausea_vomiting",
    "Dizziness / weakness": "dizziness_weakness",
    "Fever / chills": "fever_chills",
    "Cough / breathing trouble": "cough_breathing",
    "Hearing change / ringing": "hearing_ringing",
    "Numbness / tingling": "numbness_tingling",
    "Trouble sleeping": "trouble_sleeping",
}

FEELING_FOLLOWUPS = {
    "a little worse": "What has been harder this week — pain, eating, or energy?",
    "much worse": "What is bothering you the most right now?",
}

def get_curated_followup(stage_id: int) -> Optional[str]:
    """
    Return a curated follow-up question based on the patient's answers,
    ONLY if the answer is concerning. Returns None if no follow-up needed.
    """
    if stage_id == 1:
        energy = st.session_state.get("feeling_level")
        if energy in ("a little worse", "much worse"):
            return FEELING_FOLLOWUPS.get(energy)
        return None

    if stage_id == 2:
        if st.session_state.pain_yesno is False:
            return None
        return "What is your pain right now, from 0 to 10?"

    if stage_id == 3:
        locs = st.session_state.get("selected_parts", set())
        past = st.session_state.get("past_checkins", [])
        prev_locs = set(past[-1].get("pain_locations", [])) if past else set()
        new_locs = locs - prev_locs
        if new_locs:
            return f"You mentioned new pain in {', '.join(sorted(new_locs))}. When did that start?"
        # Check if severity is high
        sevs = st.session_state.get("pain_severities", {})
        high_sev = [loc for loc, s in sevs.items() if s >= 5]
        if high_sev:
            return f"Your {', '.join(high_sev)} pain is quite high. Is it affecting your daily activities?"
        return None

    if stage_id == 4:
        symptoms = st.session_state.get("symptoms", [])
        past = st.session_state.get("past_checkins", [])
        prev_syms = set(past[-1].get("symptoms", [])) if past else set()
        new_syms = set(symptoms) - prev_syms

        priority = [
            "Trouble swallowing",
            "Cough / breathing trouble",
            "Fever / chills",
            "Dizziness / weakness",
            "Eating or drinking less",
            "Mouth or tongue sore",
        ]
        for s in priority:
            if s in symptoms:
                key = SYMPTOM_KEY_MAP.get(s)
                if key and key in SYMPTOM_FOLLOWUPS:
                    return SYMPTOM_FOLLOWUPS[key]

        if new_syms:
            first_new = sorted(new_syms)[0]
            key = SYMPTOM_KEY_MAP.get(first_new)
            if key and key in SYMPTOM_FOLLOWUPS:
                return SYMPTOM_FOLLOWUPS[key]

        return None

    return None

def get_gpt_reply_with_suggestions(extra_context: str = ""):
    """GPT-generated reply with suggestions — used only for free-text follow-ups."""
    if not _openai_ready():
        return "(Assistant unavailable — check OpenAI API key.)", []

    suggestion_instruction = (
        "\n\nRESPONSE FORMAT — reply with ONLY valid JSON, no markdown fences:\n"
        '{"question": "Your single follow-up question here", '
        '"suggested_answers": ["short answer 1", "short answer 2", "short answer 3"]}\n'
        "suggested_answers MUST be MAX 4 words each. "
        "Output ONLY the JSON object."
    )
    system_prompt = build_system_prompt(extra_context) + suggestion_instruction

    msgs = [{"role": "system", "content": system_prompt}]
    for p in st.session_state.get("past_checkins", []):
        ts = p.get("timestamp","?"); fl = p.get("feeling_level","?")
        pn = "yes" if p.get("pain") else "no"
        locs = ", ".join(p.get("pain_locations",[])) or "none"
        syms = ", ".join(p.get("symptoms",[])) or "none"
        msgs.append({"role":"user",      "content": f"[Past visit {ts}] Feeling:{fl}. Pain:{pn}. Locations:{locs}. Symptoms:{syms}."})
        msgs.append({"role":"assistant", "content": f"Noted your check-in from {ts}."})
    for m in st.session_state.messages[-20:]:
        msgs.append({"role": "assistant" if m.get("role")=="doctor" else "user",
                     "content": m.get("content","")})
    try:
        r = openai_client.chat.completions.create(
            model=_secret("openai_model", default="gpt-4o-mini"),
            messages=msgs, max_tokens=200, temperature=0.5,
        )
        raw = (r.choices[0].message.content or "").strip()
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

        parsed = json.loads(cleaned)
        question = parsed.get("question", "").strip()
        suggestions = parsed.get("suggested_answers", [])
        if not question: return raw, []
        if not isinstance(suggestions, list): suggestions = []
        suggestions = [str(s).strip() for s in suggestions if str(s).strip()][:3]
        return question, suggestions
    except (json.JSONDecodeError, KeyError, TypeError):
        return raw if raw else "(No response)", []
    except Exception as e:
        return f"(Error: {e})", []

def get_opening_message(last: Dict, name: str) -> str:
    """Generate brief opening greeting referencing last visit data."""
    if not _openai_ready():
        return f"Hi {name}! Good to see you again."

    fl   = last.get("feeling_level", "?")
    pn   = "yes" if last.get("pain") else "no"
    ploc = ", ".join(last.get("pain_locations", [])) or "none"
    sym  = ", ".join(last.get("symptoms", [])) or "none"
    ts   = last.get("timestamp", "your last visit")

    system = (
        "You are a warm symptom-intake assistant for a cancer care clinic. "
        "Write ONE sentence greeting the patient by name, briefly mentioning "
        "their specific data from last time (actual symptoms, pain locations, or feeling). "
        "End with: 'Let\\'s see how things are today.' "
        "Do NOT ask a question. Do NOT say 'Thank you' or filler. Max 2 sentences."
    )
    try:
        r = openai_client.chat.completions.create(
            model=_secret("openai_model", default="gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Patient: {name}. Last visit: {ts}. "
                                               f"Feeling: {fl}. Pain: {pn}. "
                                               f"Locations: {ploc}. Symptoms: {sym}."}
            ],
            max_tokens=80, temperature=0.5,
        )
        return (r.choices[0].message.content or "").strip()
    except:
        return f"Hi {name}! Let's see how things are today."

def transcribe_audio(audio_bytes: bytes) -> str:
    if not _openai_ready(): return "(Transcription unavailable.)"
    try:
        import io
        f = io.BytesIO(audio_bytes); f.name = "recording.wav"
        return (openai_client.audio.transcriptions.create(
            model=_secret("whisper_model", default="whisper-1"), file=f, language="en"
        ).text or "").strip()
    except Exception as e: return f"(Transcription failed: {e})"

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

:root {
    --bg:        #f5f3ef;
    --surface:   #ffffff;
    --border:    rgba(180,170,155,0.30);
    --accent:    #2a9d8f;
    --accent-lt: rgba(42,157,143,0.09);
    --accent-md: rgba(42,157,143,0.22);
    --patient:   #264653;
    --text:      #2c2c2c;
    --muted:     #8a8075;
    --shadow-sm: 0 2px 10px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 24px rgba(0,0,0,0.09);
    --r-sm: 12px;
    --r-md: 18px;
    --r-lg: 24px;
    --warn:     #e76f51;
    --ok:       #2a9d8f;
    --neutral:  #e9c46a;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Nunito', sans-serif !important;
    background: var(--bg) !important;
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; z-index: -1;
    background:
        radial-gradient(ellipse 70% 50% at 88% 8%, rgba(42,157,143,0.12) 0%, transparent 65%),
        radial-gradient(ellipse 55% 40% at 8% 92%, rgba(38,70,83,0.08) 0%, transparent 65%),
        #f5f3ef;
}
[data-testid="stHeader"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stToolbar"] {
    display: none !important; height: 0 !important;
}
[data-testid="stMainBlockContainer"] { padding-top: 1rem !important; }
.block-container { max-width: 680px !important; padding: 0 1.2rem 3rem !important; }
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stAppDeployButton"], .stDeployButton,
.stApp > header, iframe[title="streamlitApp"],
[data-testid="manage-app-button"], [data-testid="stRunningManWidget"],
[data-testid="collapsedControl"], [data-testid="stAppIframe"],
div:has(> iframe[title="streamlitApp"]) {
    display: none !important; height: 0 !important;
    max-height: 0 !important; overflow: hidden !important;
}
.stApp > div:first-child:not(:has([data-testid="stMainBlockContainer"])) {
    display: none !important; height: 0 !important;
}

.app-header {
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 26px; padding-bottom: 18px;
    border-bottom: 1.5px solid var(--border);
}
.app-header-icon {
    width: 46px; height: 46px; border-radius: 13px;
    background: linear-gradient(135deg, var(--accent) 0%, #1d7a6e 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 21px; box-shadow: 0 3px 12px rgba(42,157,143,0.30);
    flex-shrink: 0;
}
.app-header-title {
    font-family: 'Lora', serif; font-size: 21px; font-weight: 600;
    color: var(--text); line-height: 1.2; letter-spacing: -0.3px;
}
.app-header-sub {
    font-size: 11px; color: var(--muted); font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; margin-top: 3px;
}

.chat-window {
    max-height: 38vh; overflow-y: auto;
    padding: 14px; border-radius: var(--r-lg);
    background: var(--surface);
    border: 1.5px solid var(--border);
    box-shadow: var(--shadow-sm);
    margin-bottom: 14px;
    scrollbar-width: thin;
    scrollbar-color: rgba(0,0,0,0.10) transparent;
}
.row-left  { display:flex; justify-content:flex-start; align-items:flex-end; margin:7px 0; gap:9px; }
.row-right { display:flex; justify-content:flex-end;   align-items:flex-end; margin:7px 0; gap:9px; }
.avatar {
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
    background: var(--accent-lt); border: 1.5px solid var(--accent-md);
}
.bubble-doc {
    background: var(--surface); border: 1.5px solid var(--border);
    border-radius: var(--r-md); border-bottom-left-radius: 4px;
    padding: 10px 14px; max-width: 78%;
    box-shadow: var(--shadow-sm);
    font-size: 14px; line-height: 1.6; color: var(--text);
    white-space: pre-wrap; animation: fadeUp 0.22s ease both;
}
.bubble-pat {
    background: var(--patient); color: #fff;
    border-radius: var(--r-md); border-bottom-right-radius: 4px;
    padding: 10px 14px; max-width: 78%;
    box-shadow: var(--shadow-sm);
    font-size: 14px; line-height: 1.6;
    white-space: pre-wrap; animation: fadeUp 0.22s ease both;
}

.panel {
    background: transparent; border: none; border-radius: 0;
    padding: 0 0 10px; box-shadow: none;
    animation: fadeUp 0.28s ease both;
}
.panel-title {
    display: flex; align-items: flex-end; gap: 9px;
    margin-bottom: 14px;
    font-family: 'Nunito', sans-serif !important;
    font-size: 14px; font-weight: 400;
}
.panel-title-avatar {
    width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    background: var(--accent-lt); border: 1.5px solid var(--accent-md);
}
.panel-title-bubble {
    background: var(--surface); border: 1.5px solid var(--border);
    border-radius: var(--r-md); border-bottom-left-radius: 4px;
    padding: 10px 15px; box-shadow: var(--shadow-sm);
    font-family: 'Nunito', sans-serif !important;
    font-size: 15px; font-weight: 600; color: var(--text);
    line-height: 1.5; max-width: 82%;
    animation: fadeUp 0.22s ease both;
}
.small-note {
    font-size: 12px; color: var(--muted); font-weight: 500;
    margin: 0 0 10px; letter-spacing: 0.02em;
}
.divider { border: none; border-top: 1.5px solid var(--border); margin: 14px 0 12px; }

.inline-followup {
    background: var(--accent-lt); border-left: 3px solid var(--accent);
    border-radius: 0 var(--r-sm) var(--r-sm) 0;
    padding: 11px 14px; margin: 12px 0 8px;
    font-size: 14px; line-height: 1.6; color: var(--text);
    animation: fadeUp 0.22s ease both;
}
.inline-patient {
    background: rgba(38,70,83,0.07);
    border-radius: var(--r-sm) var(--r-sm) 0 var(--r-sm);
    padding: 9px 13px; margin: 6px 0; text-align: right;
    font-size: 14px; line-height: 1.6; color: var(--text);
    animation: fadeUp 0.18s ease both;
}

.stButton > button {
    font-family: 'Nunito', sans-serif !important;
    border-radius: var(--r-sm) !important;
    padding: 0.45rem 1rem !important;
    font-size: 14px !important; font-weight: 600 !important;
    border: 1.5px solid var(--border) !important;
    background: var(--surface) !important; color: var(--text) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.14s ease !important; white-space: nowrap !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important; color: var(--accent) !important;
    background: var(--accent-lt) !important;
    box-shadow: 0 2px 12px rgba(42,157,143,0.18) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent) 0%, #1d7a6e 100%) !important;
    color: #fff !important; border-color: transparent !important;
    box-shadow: 0 3px 14px rgba(42,157,143,0.32) !important;
    font-size: 15px !important; padding: 0.55rem 1rem !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #30ada0 0%, #1a6e64 100%) !important;
    color: #fff !important; transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px rgba(42,157,143,0.42) !important;
}

[data-testid="stAudioInput"] { margin: 0 !important; padding: 0 !important; }
[data-testid="stAudioInput"] > label { display: none !important; }
[data-testid="stAudioInput"] > div {
    height: 38px !important; min-height: 38px !important;
    border-radius: var(--r-sm) !important;
    border: 1.5px solid var(--border) !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    background: var(--surface) !important; box-shadow: var(--shadow-sm) !important;
    width: 100% !important; min-width: 0 !important;
    overflow: hidden !important;
}
[data-testid="stAudioInput"] > div:hover { border-color: var(--accent) !important; }
[data-testid="stAudioInput"] > div > * { transform: scale(0.88); transform-origin: center; }

[data-testid="stTextInput"] > div > div > input {
    font-family: 'Nunito', sans-serif !important;
    border-radius: var(--r-sm) !important;
    border: 1.5px solid var(--border) !important;
    padding: 7px 14px !important; font-size: 14px !important;
    background: var(--surface) !important; height: 38px !important;
    box-shadow: var(--shadow-sm) !important; color: var(--text) !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-lt) !important;
    outline: none !important;
}
[data-testid="stTextInput"] > label { display: none !important; }

[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton) {
    position: relative !important;
}
[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton)
  [data-testid="stElementContainer"]:has(.stButton) {
    position: absolute !important; right: 8px !important; top: 4px !important;
    z-index: 2 !important; width: auto !important;
}
[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton)
  .stButton { display: flex !important; justify-content: flex-end !important; }
[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton)
  .stButton button {
    width: 30px !important; height: 30px !important; min-height: 30px !important;
    padding: 0 !important; border-radius: 50% !important;
    background: linear-gradient(135deg, var(--accent) 0%, #1d7a6e 100%) !important;
    color: #fff !important; border: none !important;
    font-size: 15px !important; font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(42,157,143,0.28) !important;
    display: inline-flex !important; align-items: center !important;
    justify-content: center !important;
}
[data-testid="stColumn"]:has(.stButton) [data-testid="stTextInput"] > div > div > input {
    padding-right: 42px !important;
}

/* ── Previous-value indicator on body map ── */
.prev-indicator {
    display: inline-block; font-size: 11px; font-weight: 600;
    padding: 1px 7px; border-radius: 6px; margin-left: 6px;
}
.prev-ok   { background: #d1fae5; color: #065f46; }
.prev-warn { background: #fee2e2; color: #991b1b; }
.prev-same { background: #fef3c7; color: #92400e; }

.summary-wrap {
    background: var(--surface); border: 1.5px solid var(--border);
    border-radius: var(--r-lg); padding: 26px 22px 20px;
    box-shadow: var(--shadow-md); animation: fadeUp 0.35s ease both;
}
.summary-title {
    font-family: 'Lora', serif; font-size: 20px; font-weight: 600;
    color: var(--text); margin-bottom: 4px;
}
.summary-sub {
    font-size: 11px; color: var(--muted); font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 18px;
}
.summary-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.summary-table tr { border-bottom: 1.5px solid var(--border); }
.summary-table tr:last-child { border-bottom: none; }
.summary-table td { padding: 11px 8px; vertical-align: top; line-height: 1.5; }
.summary-table td:first-child {
    font-weight: 700; color: var(--muted); width: 36%;
    font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.07em; padding-top: 14px;
}
.tag {
    display: inline-block; background: var(--accent-lt); color: var(--accent);
    border: 1px solid var(--accent-md); border-radius: 8px;
    padding: 2px 10px; font-size: 13px; font-weight: 600;
    margin: 2px 3px 2px 0;
}
.submitted-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #d1fae5; color: #065f46;
    border: 1.5px solid #6ee7b7; border-radius: 10px;
    padding: 5px 14px; font-size: 13px; font-weight: 700;
    margin-bottom: 16px;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────
# REDESIGNED STAGES:
#  -1 = name entry
#   0 = welcome + "anything changed?" fast-path (returning patients only)
#   1 = feeling scale (triage: only follow-up if fair/poor)
#   2 = pain yes/no (only follow-up if yes)
#   3 = body pain map + severity sliders (pre-populated; follow-up only if concerning)
#   4 = symptom checklist (pre-populated; "Other" button; follow-up only if concerning)
#   5 = submit (direct submit with optional note)
defaults = {
    "messages": [], "stage": -1, "patient_name": "",
    "selected_parts": set(), "pain_yesno": None, "pain_trend": None, "feeling_level": None,
    "symptoms": [], "submitted": False, "past_checkins": [],
    "last_audio_hash": None, "mic_key_counter": 0,
    "followup_counts": {}, "stage_answered": {},
    "pain_severities": {},  # {location: severity_int}
    "active_body_part": None,  # which body option is currently expanded inline
    "show_other_text": {},  # {stage_id: bool} — whether "Other" text is expanded
    "fast_path": False,     # True if patient said "nothing changed"
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ── Core helpers ────────────────────────────────────────────
def add_doctor(text, stage=None, suggestions=None):
    s = stage if stage is not None else st.session_state.get("stage", -1)
    msg = {"role":"doctor","content":text,"stage":s}
    if suggestions: msg["suggestions"] = suggestions
    st.session_state.messages.append(msg)

def add_patient(text, stage=None):
    s = stage if stage is not None else st.session_state.get("stage", -1)
    st.session_state.messages.append({"role":"patient","content":text,"stage":s})

def toggle_body_part(part):
    # Keep selected parts and per-location severities consistent.
    if part in st.session_state.selected_parts:
        st.session_state.selected_parts.remove(part)
        try:
            st.session_state.pain_severities.pop(part, None)
        except Exception:
            pass
        if st.session_state.get("active_body_part") == part:
            st.session_state.active_body_part = None
    else:
        st.session_state.selected_parts.add(part)
        st.session_state.active_body_part = part

def followup_count(sid): return st.session_state.followup_counts.get(sid, 0)
def record_followup(sid): st.session_state.followup_counts[sid] = followup_count(sid)+1
def is_answered(sid):    return st.session_state.stage_answered.get(sid, False)
def mark_answered(sid):  st.session_state.stage_answered[sid] = True

# Max 1 curated follow-up per stage (not 3 random GPT questions)
MAX_FOLLOWUPS = {0:0, 1:1, 2:1, 3:1, 4:1}

def can_followup(sid): return followup_count(sid) < MAX_FOLLOWUPS.get(sid, 0)

def advance_stage():
    s = st.session_state.stage
    if st.session_state.fast_path:
        st.session_state.stage = 5  # skip everything
        return
    if   s == 0: st.session_state.stage = 1
    elif s == 1: st.session_state.stage = 2
    elif s == 2: st.session_state.stage = 4 if st.session_state.pain_yesno is False else 3
    elif s == 3: st.session_state.stage = 4
    elif s == 4: st.session_state.stage = 5

def on_patient_answer(text: str, stage_id: int):
    """Record answer. Use CURATED follow-up only if concerning."""
    add_patient(text, stage=stage_id)
    mark_answered(stage_id)
    if can_followup(stage_id):
        followup_q = get_curated_followup(stage_id)
        if followup_q:
            record_followup(stage_id)
            add_doctor(followup_q, stage=stage_id)
        # If no curated follow-up needed → don't ask anything

def on_followup_reply(text: str, stage_id: int):
    """Patient replying to a curated follow-up. No further follow-ups."""
    add_patient(text, stage=stage_id)

def handle_voice(audio_value, stage_id: int, is_followup: bool = False) -> bool:
    if audio_value is None: return False
    try:
        ab = audio_value.getvalue()
        ah = hashlib.sha1(ab).hexdigest()
    except: return False
    if not ab or not ah or ah == st.session_state.last_audio_hash: return False
    st.session_state.last_audio_hash = ah
    st.session_state.mic_key_counter += 1
    with st.spinner("Transcribing…"):
        t = transcribe_audio(ab)
    if t and not t.startswith("(Transcription failed"):
        st.info(f'Heard: "{t}"')
        if is_followup:
            on_followup_reply(t, stage_id)
        else:
            on_patient_answer(t, stage_id)
        return True
    st.warning("Could not transcribe. Please try again.")
    return False

def render_chat_window():
    current = st.session_state.stage
    past_msgs = [m for m in st.session_state.messages if m.get("stage", -99) < current]
    if not past_msgs: return
    st.markdown('<div class="chat-window">', unsafe_allow_html=True)
    for msg in past_msgs:
        if msg.get("role") == "doctor":
            st.markdown(f'<div class="row-left"><div class="avatar">🩺</div>'
                        f'<div class="bubble-doc">{msg.get("content","")}</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="row-right"><div class="bubble-pat">{msg.get("content","")}</div>'
                        f'<div class="avatar">🙂</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def render_inline_stage_messages(stage_id: int):
    stage_msgs = [m for m in st.session_state.messages if m.get("stage") == stage_id]
    for msg in stage_msgs:
        if msg.get("role") == "doctor":
            st.markdown(f'<div class="inline-followup">🩺 {msg.get("content","")}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="inline-patient">{msg.get("content","")} 🙂</div>',
                        unsafe_allow_html=True)

def render_followup_input(stage_id: int):
    """Compact text + mic row for answering a curated follow-up."""
    c_main, c_mic = st.columns([7, 2.5], gap="small")
    with c_main:
        typed = st.text_input("", placeholder="Reply…",
                              key=f"fu_txt_{stage_id}_{followup_count(stage_id)}",
                              label_visibility="collapsed")
        send_clicked = st.button("↑", key=f"fu_send_{stage_id}_{followup_count(stage_id)}")
    with c_mic:
        audio_val = None
        if hasattr(st, "audio_input"):
            audio_val = st.audio_input("", label_visibility="collapsed",
                                       key=f"fu_mic_{stage_id}_{st.session_state.mic_key_counter}")
    if send_clicked and typed and typed.strip():
        on_followup_reply(typed.strip(), stage_id)
        st.rerun()
    if handle_voice(audio_val, stage_id, is_followup=True):
        st.rerun()

def render_other_text(stage_id: int, placeholder: str = "Describe…"):
    """Show text input only when 'Other' is expanded."""
    c_main, c_mic = st.columns([7, 2.5], gap="small")
    with c_main:
        typed = st.text_input("", placeholder=placeholder,
                              key=f"other_txt_{stage_id}", label_visibility="collapsed")
        send_clicked = st.button("↑", key=f"other_send_{stage_id}")
    with c_mic:
        audio_val = None
        if hasattr(st, "audio_input"):
            audio_val = st.audio_input("", key=f"other_mic_{stage_id}_{st.session_state.mic_key_counter}",
                                       label_visibility="collapsed")
    if send_clicked and typed and typed.strip():
        on_patient_answer(typed.strip(), stage_id)
        st.rerun()
    if handle_voice(audio_val, stage_id):
        st.rerun()

def body_svg(selected: Set[str], prev_locs: Set[str] = set()) -> str:
    """Body SVG with color coding:
    - red    = selected pain areas today
    - orange = pain areas from last visit (history)
    - green  = other body areas
    """

    COLOR_SELECTED_TODAY = "#e63946"   # red
    COLOR_LAST_VISIT     = "#f4a261"   # orange
    COLOR_OTHER          = "#7bc96f"   # green

    def fill(p):
        if p in selected:
            return COLOR_SELECTED_TODAY
        if p in prev_locs:
            return COLOR_LAST_VISIT
        return COLOR_OTHER
    s = "#6b7a90"
    return f"""<svg width="200" height="325" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">
  <defs><filter id="sh"><feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="rgba(0,0,0,0.12)"/></filter></defs>
  <g filter="url(#sh)"><circle cx="160" cy="70" r="38" fill="{fill('Head')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><rect x="110" y="120" width="100" height="70" rx="24" fill="{fill('Chest')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><rect x="115" y="195" width="90" height="70" rx="22" fill="{fill('Abdomen')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M110 132 C80 145,72 180,78 220 C82 250,92 270,100 290 C108 310,115 320,120 320 L120 130Z" fill="{fill('Left Arm')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M210 132 C240 145,248 180,242 220 C238 250,228 270,220 290 C212 310,205 320,200 320 L200 130Z" fill="{fill('Right Arm')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M135 265 C120 310,118 360,126 410 C132 445,132 475,128 500 L155 500 C158 470,160 435,156 405 C150 355,152 312,165 265Z" fill="{fill('Left Leg')}" stroke="{s}" stroke-width="2"/></g>
  <g filter="url(#sh)"><path d="M185 265 C200 310,202 360,194 410 C188 445,188 475,192 500 L165 500 C162 470,160 435,164 405 C170 355,168 312,155 265Z" fill="{fill('Right Leg')}" stroke="{s}" stroke-width="2"/></g>
</svg>""".strip()

# ── Warnings ─────────────────────────────────────────────────
if openai_init_error: st.warning(f"LLM not ready: {openai_init_error}")
_init_sheets()
if sheets_init_error: st.warning(f"Sheets not ready: {sheets_init_error}")

st.markdown('''
<div class="app-header">
  <div class="app-header-icon">🩺</div>
  <div>
    <div class="app-header-title">Symptom Check-In</div>
    <div class="app-header-sub">Cancer Care · Daily Check-In</div>
  </div>
</div>''', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE -1 — Name entry
# ════════════════════════════════════════════════════════════
if st.session_state.stage == -1:
    st.markdown('<div class="panel"><div class="panel-title"><div class="panel-title-avatar">🩺</div>'
                '<div class="panel-title-bubble">Welcome · Please enter your name</div></div>',
                unsafe_allow_html=True)
    name_input = st.text_input("Your name:", value=st.session_state.patient_name)
    if st.button("Start Check-In"):
        if name_input.strip():
            st.session_state.patient_name = name_input.strip()
            with st.spinner("Loading your history…"):
                st.session_state.past_checkins = load_past_checkins(name_input.strip())
            past = st.session_state.past_checkins
            if past:
                last = past[-1]
                # Carry over prior symptoms for quick confirmation, but do NOT pre-select
                # prior pain locations as today's pain. They should appear orange first and
                # only turn red if the patient clicks them today.
                st.session_state.selected_parts = set()
                st.session_state.symptoms = list(last.get("symptoms", []))
                # Keep today's pain severities empty until a body part is confirmed today.
                # Previous severities are still shown from past_checkins inside Stage 3.
                st.session_state.pain_severities = {}
                with st.spinner("Getting your assistant ready…"):
                    opening = get_opening_message(last, name_input.strip())
                add_doctor(opening, stage=0)
                st.session_state.stage = 0
            else:
                add_doctor(f"Hi {name_input.strip()}! Let's go through a few quick questions.", stage=1)
                st.session_state.stage = 1
            st.rerun()
        else:
            st.warning("Please enter your name to continue.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

stage = st.session_state.stage
render_chat_window()

# ════════════════════════════════════════════════════════════
# STAGE 0 — Quick recap + "nothing changed" fast path
# ════════════════════════════════════════════════════════════
if stage == 0:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><div class="panel-title-avatar">🩺</div>'
                '<div class="panel-title-bubble">💬 Quick check-in</div></div>',
                unsafe_allow_html=True)

    render_inline_stage_messages(stage_id=0)

    if not is_answered(0):
        st.markdown('<div class="small-note">Compared to your last visit:</div>',
                    unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            if st.button("✅ Nothing new", use_container_width=True, key="fast_nothing"):
                st.session_state.fast_path = True
                add_patient("Nothing has changed.", stage=0)
                mark_answered(0)
                advance_stage(); st.rerun()
        with c2:
            if st.button("📋 Some changes", use_container_width=True, key="fast_some"):
                add_patient("A few things have changed.", stage=0)
                mark_answered(0)
                advance_stage(); st.rerun()
        with c3:
            if st.button("⚠️ Getting worse", use_container_width=True, key="fast_worse"):
                add_patient("Things are getting worse.", stage=0)
                mark_answered(0)
                advance_stage(); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 1 — Feeling scale (triage — follow-up ONLY if fair/poor)
# ════════════════════════════════════════════════════════════
elif stage == 1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><div class="panel-title-avatar">🩺</div>'
                '<div class="panel-title-bubble">How has your energy been since your last visit?</div></div>',
                unsafe_allow_html=True)

    FEELING_OPTIONS = [
        ("Much more energy", "much better"),
        ("A little more energy", "a little better"),
        ("About the same", "about the same"),
        ("A little less energy", "a little worse"),
        ("Much less energy", "much worse"),
    ]

    if not is_answered(1):
        render_inline_stage_messages(stage_id=1)

        opt_cols = st.columns(5, gap="small")
        for idx, (label, value) in enumerate(FEELING_OPTIONS):
            with opt_cols[idx]:
                if st.button(label, key=f"feel_{idx}", use_container_width=True):
                    st.session_state.feeling_level = value
                    on_patient_answer(f"I'm feeling {value} today.", 1)
                    st.rerun()
    else:
        render_inline_stage_messages(stage_id=1)
        stage1_msgs = [m for m in st.session_state.messages if m.get("stage") == 1]
        last_is_doctor = stage1_msgs and stage1_msgs[-1].get("role") == "doctor"
        if last_is_doctor:
            # Curated follow-up only when energy is worse
            feeling = st.session_state.feeling_level
            replies = []
            if feeling == "a little worse":
                replies = ["Pain is worse", "Eating is harder", "More tired"]
            elif feeling == "much worse":
                replies = ["Pain is bad", "Very weak", "Can't eat"]
            if replies:
                cols = st.columns(len(replies), gap="small")
                for idx, r in enumerate(replies):
                    with cols[idx]:
                        if st.button(r, key=f"feel_fu_{idx}", use_container_width=True):
                            on_followup_reply(r, 1); st.rerun()
                render_followup_input(stage_id=1)
        else:
            advance_stage(); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 2 — Pain yes/no (follow-up only if yes → severity)
# ════════════════════════════════════════════════════════════
elif stage == 2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><div class="panel-title-avatar">🩺</div>'
                '<div class="panel-title-bubble">How has your pain been since your last visit?</div></div>',
                unsafe_allow_html=True)

    if not is_answered(2):
        render_inline_stage_messages(stage_id=2)

        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            if st.button("🙂 No pain", use_container_width=True, key="pain_no"):
                st.session_state.pain_yesno = False
                st.session_state.pain_trend = "no pain"
                on_patient_answer("No pain today.", 2); st.rerun()
        with c2:
            if st.button("⬇️ Better", use_container_width=True, key="pain_better"):
                st.session_state.pain_yesno = True
                st.session_state.pain_trend = "better"
                on_patient_answer("Pain is better than last visit.", 2); st.rerun()
        with c3:
            if st.button("➡️ About the same", use_container_width=True, key="pain_same"):
                st.session_state.pain_yesno = True
                st.session_state.pain_trend = "about the same"
                on_patient_answer("Pain is about the same as last visit.", 2); st.rerun()
        with c4:
            if st.button("⬆️ Worse", use_container_width=True, key="pain_worse"):
                st.session_state.pain_yesno = True
                st.session_state.pain_trend = "worse"
                on_patient_answer("Pain is worse than last visit.", 2); st.rerun()
    else:
        render_inline_stage_messages(stage_id=2)
        stage2_msgs = [m for m in st.session_state.messages if m.get("stage") == 2]
        last_is_doctor = stage2_msgs and stage2_msgs[-1].get("role") == "doctor"
        if last_is_doctor and st.session_state.pain_yesno:
            # Severity rating as buttons (0-10 scale, show only key values)
            st.markdown('<div class="small-note">Rate your overall pain (0 = none, 10 = worst):</div>',
                        unsafe_allow_html=True)
            sev_cols = st.columns(6, gap="small")
            for idx, val in enumerate([0, 2, 4, 6, 8, 10]):
                with sev_cols[idx]:
                    if st.button(str(val), key=f"pain_sev_{val}", use_container_width=True):
                        on_followup_reply(f"Pain level: {val}/10", 2)
                        st.rerun()
        else:
            advance_stage(); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 3 — Body pain map (PRE-POPULATED from last visit)
# ════════════════════════════════════════════════════════════
elif stage == 3:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><div class="panel-title-avatar">🩺</div>'
                '<div class="panel-title-bubble">Where is the pain right now?</div></div>',
                unsafe_allow_html=True)

    past = st.session_state.get("past_checkins", [])
    prev_locs = set(past[-1].get("pain_locations", [])) if past else set()

    if not is_answered(3):
        render_inline_stage_messages(stage_id=3)

        if prev_locs:
            st.markdown(
                '<div class="small-note">🟠 Orange = pain areas from last visit · 🟢 Green = other areas · 🔴 Red = pain today. '
                'Tap a body part to confirm or change.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="small-note">Select areas where you feel pain:</div>',
                        unsafe_allow_html=True)

        col_svg, col_btns = st.columns([1, 1], gap="medium")
        with col_svg:
            st.markdown(body_svg(st.session_state.selected_parts, prev_locs),
                        unsafe_allow_html=True)
        with col_btns:
            # Pre-populate per-location severities from last visit.
            prev_sevs = dict(past[-1].get("pain_severities", {})) if past else {}
            body_parts = ["Head","Chest","Abdomen","Left Arm","Right Arm","Left Leg","Right Leg"]
            for part in body_parts:
                is_selected = part in st.session_state.selected_parts
                label = f"✓ {part}" if is_selected else part
                if part in prev_locs:
                    label += " · last visit"
                if st.button(label, key=f"toggle_{part}", use_container_width=True):
                    if is_selected:
                        # Clicking the same selected option collapses it and removes the pain mark.
                        toggle_body_part(part)
                    else:
                        toggle_body_part(part)
                    st.rerun()

                # Inline slider directly from this option, not in a separate section.
                if part in st.session_state.selected_parts and st.session_state.get("active_body_part") == part:
                    default_val = int(prev_sevs.get(part, st.session_state.pain_severities.get(part, 3)))
                    current_val = int(st.session_state.pain_severities.get(part, default_val))
                    st.markdown(
                        f"<div class='small-note' style='margin:4px 0 2px 8px;'>Set {part.lower()} pain now"
                        + (f" &nbsp;•&nbsp; last visit: {prev_sevs.get(part)}/10" if prev_sevs.get(part) is not None else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                    val = st.slider(
                        label=f"{part} severity",
                        min_value=0,
                        max_value=10,
                        value=current_val,
                        key=f"sev_{part}",
                        label_visibility="collapsed",
                    )
                    st.session_state.pain_severities[part] = int(val)
                    slider_cols = st.columns([1,1])
                    with slider_cols[0]:
                        if st.button(f"Done with {part}", key=f"done_{part}", use_container_width=True):
                            st.session_state.active_body_part = None
                            st.rerun()
                    with slider_cols[1]:
                        if st.button(f"Remove {part}", key=f"remove_{part}", use_container_width=True):
                            toggle_body_part(part)
                            st.rerun()

            # "Other" button for unlisted locations
            if st.button("➕ Other location", key="other_loc_btn", use_container_width=True):
                st.session_state.show_other_text[3] = True
                st.rerun()

        if st.session_state.show_other_text.get(3, False):
            render_other_text(stage_id=3, placeholder="Describe other pain location…")

        # Ensure every selected location has a default severity even if the drawer isn't opened.
        if st.session_state.selected_parts:
            prev_sevs = dict(past[-1].get("pain_severities", {})) if past else {}
            for part in st.session_state.selected_parts:
                st.session_state.pain_severities.setdefault(part, int(prev_sevs.get(part, 3)))

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        if st.button("Save pain areas", key="send_locs", use_container_width=True, type="primary"):
            locs = sorted(st.session_state.selected_parts)
            sevs = st.session_state.pain_severities
            loc_txt = ", ".join(f"{l} ({sevs.get(l,'?')}/10)" for l in locs) if locs else "no pain areas selected"
            on_patient_answer(f"Pain locations right now: {loc_txt}", 3)
            st.rerun()

    else:
        render_inline_stage_messages(stage_id=3)
        stage3_msgs = [m for m in st.session_state.messages if m.get("stage") == 3]
        last_is_doctor = stage3_msgs and stage3_msgs[-1].get("role") == "doctor"
        if last_is_doctor:
            render_followup_input(stage_id=3)
        else:
            advance_stage(); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 4 — Symptom checklist (PRE-POPULATED + "Other" button)
# ════════════════════════════════════════════════════════════
elif stage == 4:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><div class="panel-title-avatar">🩺</div>'
                '<div class="panel-title-bubble">Are any of these bothering you since your last visit?</div></div>',
                unsafe_allow_html=True)

    symptom_options = [
        "Eating or drinking less", "Mouth or tongue sore", "Trouble swallowing",
        "Dry mouth / thick mucus", "Constipation", "Nausea / vomiting",
        "Dizziness / weakness", "Fever / chills", "Cough / breathing trouble",
        "Hearing change / ringing", "Numbness / tingling", "Trouble sleeping",
    ]

    past = st.session_state.get("past_checkins", [])
    prev_syms = set(past[-1].get("symptoms", [])) if past else set()

    if not is_answered(4):
        render_inline_stage_messages(stage_id=4)

        if prev_syms:
            st.markdown('<div class="small-note">✓ = carried over from last visit. '
                        'Tap to remove if resolved.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="small-note">Tap any issues that sound like what the clinic usually asks about.</div>',
                        unsafe_allow_html=True)

        sc = st.columns(2, gap="small")
        for idx, symptom in enumerate(symptom_options):
            with sc[idx % 2]:
                if symptom in st.session_state.symptoms:
                    # ✓ indicates selected; if it was also present last visit, it is 'carried over'
                    if symptom in prev_syms:
                        label = f"✓ {symptom} (carried over)"
                    else:
                        label = f"✓ {symptom}"
                else:
                    # If it was present last visit but not selected now, show a gentle reminder
                    if symptom in prev_syms:
                        label = f"{symptom} (last visit)"
                    else:
                        label = symptom
                if st.button(label, key=f"sym_{idx}", use_container_width=True):
                    if symptom in st.session_state.symptoms:
                        st.session_state.symptoms.remove(symptom)
                    else:
                        st.session_state.symptoms.append(symptom)
                    st.rerun()

        # "Other" button — hidden text input
        if st.button("➕ Other issue", key="other_sym_btn", use_container_width=True):
            st.session_state.show_other_text[4] = True
            st.rerun()

        if st.session_state.show_other_text.get(4, False):
            render_other_text(stage_id=4, placeholder="Describe another issue the care team should know about…")

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        if st.button("Save issues", key="send_syms", use_container_width=True, type="primary"):
            sym_txt = "; ".join(st.session_state.symptoms) if st.session_state.symptoms else "no symptoms"
            on_patient_answer(f"Current issues: {sym_txt}", 4)
            st.rerun()

    else:
        render_inline_stage_messages(stage_id=4)
        stage4_msgs = [m for m in st.session_state.messages if m.get("stage") == 4]
        last_is_doctor = stage4_msgs and stage4_msgs[-1].get("role") == "doctor"
        if last_is_doctor:
            # Show quick replies for the curated follow-up
            render_followup_input(stage_id=4)
        else:
            advance_stage(); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 5 — Submit (optional note + direct submit)
# ════════════════════════════════════════════════════════════
elif stage == 5:
    if st.session_state.submitted:
        name      = st.session_state.get("patient_name","—")
        feeling   = st.session_state.get("feeling_level",None)
        pain      = st.session_state.get("pain_yesno",None)
        locations = sorted(list(st.session_state.get("selected_parts",set())))
        symptoms  = st.session_state.get("symptoms",[])
        severities = st.session_state.get("pain_severities", {})

        feeling_display = feeling if feeling is not None else "—"
        pain_str  = "Yes" if pain is True else ("No" if pain is False else "—")
        sym_html  = "".join(f'<span class="tag">{s}</span>' for s in symptoms) or "<span style='opacity:.4'>None</span>"
        loc_html  = "".join(f'<span class="tag">{l} ({severities.get(l,"?")}/10)</span>' for l in locations) or "<span style='opacity:.4'>N/A</span>"

        # Extract conversation notes
        widget_patterns = {"Nothing has changed.", "A few things have changed.",
                           "Things are getting worse.", "No pain today."}
        patient_lines = [m["content"] for m in st.session_state.messages
                         if m.get("role")=="patient" and m["content"] not in widget_patterns]

        conv_cell = "<span style='opacity:.4'>No additional details shared</span>"
        if patient_lines and _openai_ready():
            try:
                sr = openai_client.chat.completions.create(
                    model=_secret("openai_model", default="gpt-4o-mini"),
                    messages=[
                        {"role":"system","content":(
                            "Clinical notes assistant. Extract ONLY medically relevant facts. "
                            "One bullet per fact. No greetings. If nothing relevant: None"
                        )},
                        {"role":"user","content":"\n".join(f"- {l}" for l in patient_lines)}
                    ], max_tokens=300, temperature=0.2,
                )
                summary_text = (sr.choices[0].message.content or "").strip()
                if summary_text and summary_text != "None":
                    items = [l.lstrip("•-– ").strip() for l in summary_text.split("\n")
                             if l.strip() and l.strip()!="None"]
                    if items:
                        conv_cell = "<ul style='margin:0;padding-left:18px;'>"+"".join(
                            f"<li style='margin-bottom:4px;font-size:14px;'>{l}</li>" for l in items)+"</ul>"
            except: pass

        fast_note = ""
        if st.session_state.fast_path:
            fast_note = "<tr><td>Status</td><td><span class='tag'>No changes from last visit</span></td></tr>"

        st.markdown(f"""
<div class="summary-wrap">
  <div class="submitted-badge">✅ Submitted</div>
  <div class="summary-title">Check-In Summary — {name}</div>
  <div class="summary-sub">Your care team will review this shortly.</div>
  <table class="summary-table">
    <tr><td>Patient</td><td>{name}</td></tr>
    {fast_note}
    <tr><td>Energy since last visit</td><td>{feeling_display}</td></tr>
    <tr><td>Pain</td><td>{pain_str}</td></tr>
    <tr><td>Pain locations</td><td>{loc_html}</td></tr>
    <tr><td>Symptoms</td><td>{sym_html}</td></tr>
    <tr><td>Notes</td><td>{conv_cell}</td></tr>
  </table>
</div>""", unsafe_allow_html=True)

    else:
        # Show chat history
        st.markdown('<div class="chat-window">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg.get("role") == "doctor":
                st.markdown(f'<div class="row-left"><div class="avatar">🩺</div>'
                            f'<div class="bubble-doc">{msg.get("content","")}</div></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="row-right"><div class="bubble-pat">{msg.get("content","")}</div>'
                            f'<div class="avatar">🙂</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="panel-title"><div class="panel-title-avatar">🩺</div>'
                    '<div class="panel-title-bubble">Ready to submit — anything else?</div></div>',
                    unsafe_allow_html=True)

        # Optional note field (hidden unless patient wants it)
        if 'show_final_note' not in st.session_state:
            st.session_state.show_final_note = False

        if st.button("➕ Add a note (optional)", use_container_width=True, key="btn_show_note"):
            st.session_state.show_final_note = True
            st.rerun()

        note = ""
        if st.session_state.show_final_note:
            note = st.text_input("", placeholder="Add a note for your care team (optional)…",
                                 key="final_note", label_visibility="collapsed")

        if st.button("✅ Submit Check-In", use_container_width=True, type="primary"):
            if note and note.strip():
                add_patient(note.strip(), stage=5)
            try:
                save_to_sheet()
                st.session_state.submitted = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
