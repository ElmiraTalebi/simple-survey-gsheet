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
            "pain_locations": sorted(list(st.session_state.selected_parts)),
            "symptoms":       st.session_state.symptoms,
            "conversation":   st.session_state.messages,
        })
    ])

# ── Prompting ───────────────────────────────────────────────
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
        mem = [f"  [{p.get('timestamp','?')}] Feeling:{p.get('feeling_level','?')}/10 | "
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
   "Of course", "That's", "I'm glad", or ANY acknowledgement whatsoever.
3. No preamble. No explanation. Just the question.
4. A very brief empathetic lead-in (max 6 words) is allowed only when the patient
   shares something emotionally significant — otherwise skip it entirely and go
   straight to the question.
5. "Let's stay focused" is ONLY for genuinely off-topic messages (e.g. jokes, news,
   questions about you). NEVER use it for normal health-related replies.
6. Never give medical advice. If asked: "Your care team will follow up."
"""

def _openai_ready():
    return openai_client is not None and openai_init_error is None

def get_opening_message(last: Dict, name: str) -> str:
    """
    Generate the opening history-recap message with a lighter, non-restrictive prompt.
    Explicitly names the patient's last-visit data and asks a targeted question about it.
    """
    if not _openai_ready():
        return f"Hi {name}! Good to see you again. How have you been since your last visit?"

    fl   = last.get("feeling_level", "?")
    pn   = "yes" if last.get("pain") else "no"
    ploc = ", ".join(last.get("pain_locations", [])) or "none"
    sym  = ", ".join(last.get("symptoms", [])) or "none"
    ts   = last.get("timestamp", "your last visit")

    # Pick the most specific thing to follow up on
    if last.get("symptoms"):
        focus = f"ask specifically how their {sym} have been — better, worse, or the same?"
    elif last.get("pain") and last.get("pain_locations"):
        focus = f"ask specifically about their {ploc} pain — is it still there, how severe?"
    elif fl not in ("?", None) and str(fl).isdigit() and int(fl) <= 4:
        focus = f"ask what was contributing to their low feeling score of {fl}/10 and whether it has changed"
    else:
        focus = f"ask how they have been feeling since then"

    system = (
        "You are a warm, empathetic virtual symptom-intake assistant for a cancer care clinic. "
        "Write a short opening message (2–3 sentences max). "
        "Sentence 1: greet the patient by name and briefly mention their SPECIFIC data from last time "
        "(name the actual symptoms, pain locations, or feeling score — do NOT be generic). "
        "Sentence 2-3: " + focus + ". "
        "Do NOT say 'Thank you', 'Great', 'I see', or any filler. Be warm but direct."
    )
    try:
        r = openai_client.chat.completions.create(
            model=_secret("openai_model", default="gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Patient name: {name}. Last visit: {ts}. "
                                               f"Feeling: {fl}/10. Pain: {pn}. "
                                               f"Locations: {ploc}. Symptoms: {sym}."}
            ],
            max_tokens=120, temperature=0.5,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Hi {name}! Last time you reported {sym if sym != 'none' else 'some symptoms'}. How have things been since then?"

def get_gpt_reply(extra_context: str = "") -> str:
    if not _openai_ready():
        return "(Assistant unavailable — check OpenAI API key.)"
    msgs = [{"role": "system", "content": build_system_prompt(extra_context)}]
    for p in st.session_state.get("past_checkins", []):
        ts = p.get("timestamp","?"); fl = p.get("feeling_level","?")
        pn = "yes" if p.get("pain") else "no"
        locs = ", ".join(p.get("pain_locations",[])) or "none"
        syms = ", ".join(p.get("symptoms",[])) or "none"
        msgs.append({"role":"user",      "content": f"[Past visit {ts}] Feeling:{fl}/10. Pain:{pn}. Locations:{locs}. Symptoms:{syms}."})
        msgs.append({"role":"assistant", "content": f"Noted your check-in from {ts}."})
    for m in st.session_state.messages[-20:]:
        msgs.append({"role": "assistant" if m.get("role")=="doctor" else "user",
                     "content": m.get("content","")})
    try:
        r = openai_client.chat.completions.create(
            model=_secret("openai_model", default="gpt-4o-mini"),
            messages=msgs, max_tokens=120, temperature=0.5,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"(Error: {e})"

def get_gpt_reply_with_suggestions(extra_context: str = ""):
    """
    Like get_gpt_reply but asks GPT to also provide 2-3 likely patient answers.
    Returns (question_text, suggestions_list).
    """
    if not _openai_ready():
        return "(Assistant unavailable — check OpenAI API key.)", []

    # Build messages exactly like get_gpt_reply but with structured output instruction
    suggestion_instruction = (
        "\n\nRESPONSE FORMAT — you MUST reply with ONLY valid JSON, no markdown fences:\n"
        '{"question": "Your single follow-up question here", '
        '"suggested_answers": ["short answer 1", "short answer 2", "short answer 3"]}\n'
        "The suggested_answers MUST be 2-3 very brief patient responses — MAX 4 words each. "
        "Examples: \"Much better\", \"About the same\", \"Worse than before\", \"Yes, a little\", \"No, not really\". "
        "Keep them short enough to fit on a button. Range from positive to concerning. "
        "Output ONLY the JSON object — no extra text, no code fences."
    )
    system_prompt = build_system_prompt(extra_context) + suggestion_instruction

    msgs = [{"role": "system", "content": system_prompt}]
    for p in st.session_state.get("past_checkins", []):
        ts = p.get("timestamp","?"); fl = p.get("feeling_level","?")
        pn = "yes" if p.get("pain") else "no"
        locs = ", ".join(p.get("pain_locations",[])) or "none"
        syms = ", ".join(p.get("symptoms",[])) or "none"
        msgs.append({"role":"user",      "content": f"[Past visit {ts}] Feeling:{fl}/10. Pain:{pn}. Locations:{locs}. Symptoms:{syms}."})
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
        # Strip markdown code fences if present
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
        # Validate
        if not question:
            return raw, []
        if not isinstance(suggestions, list):
            suggestions = []
        # Ensure suggestions are strings and reasonable length
        suggestions = [str(s).strip() for s in suggestions if str(s).strip()][:3]
        return question, suggestions
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: GPT didn't return valid JSON — use raw text, no suggestions
        return raw if raw else "(No response)", []
    except Exception as e:
        return f"(Error: {e})", []

def get_opening_message_with_suggestions(last: Dict, name: str):
    """Deterministic opening (no GPT-generated questions).

    Professor feedback: questions must align to the baseline questionnaire.
    The opening is a short recap + a single standardized question about change.

    Returns (opening_text, suggestions_list).
    """
    ts = last.get("timestamp", "your last visit")
    fl = last.get("feeling_level", "—")
    pn = "Yes" if last.get("pain") else "No"
    ploc = ", ".join(last.get("pain_locations", [])) or "None"
    sym = ", ".join(last.get("symptoms", [])) or "None"

    opening = (
        f"Hi {name}. Last time ({ts}) you reported: "
        f"feeling = {fl}, pain = {pn} (locations: {ploc}), symptoms = {sym}. "
        "Compared to last time, are you feeling better, about the same, or worse today?"
    )
    suggestions = ["Better", "About the same", "Worse"]
    return opening, suggestions


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
}

/* ── Base ── */
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
[data-testid="stHeader"]  { display: none !important; height: 0 !important; }
[data-testid="stDecoration"] { display: none !important; height: 0 !important; }
[data-testid="stStatusWidget"] { display: none !important; height: 0 !important; }
[data-testid="stToolbar"] { display: none !important; height: 0 !important; }
[data-testid="stMainBlockContainer"] { padding-top: 1rem !important; }
.block-container { max-width: 680px !important; padding: 0 1.2rem 3rem !important; }
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stAppDeployButton"], [data-testid="stStatusWidget"],
.stDeployButton, .stApp > header,
iframe[title="streamlitApp"] { display: none !important; }
/* Kill any fixed/sticky bars at top */
.stApp > header, .stApp > div:first-child > header {
    display: none !important;
}
header[data-testid="stHeader"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"] {
    display: none !important;
    height: 0 !important;
    max-height: 0 !important;
    overflow: hidden !important;
    visibility: hidden !important;
    position: absolute !important;
}
/* Hide the Streamlit top bar / connection status bar / iframe bar */
.stApp [data-testid="stHeader"],
.stApp [data-testid="stStatusWidget"],
[data-testid="collapsedControl"],
.stApp iframe[title="streamlitApp"],
.stAppHeader, .stHeader,
iframe[title="streamlitApp"],
[data-testid="stAppIframe"],
div:has(> iframe[title="streamlitApp"]) {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}
/* Catch any remaining top chrome — iframe bar, running indicator */
.stApp > div:first-child:not(:has([data-testid="stMainBlockContainer"])) {
    display: none !important;
    height: 0 !important;
}
/* NUCLEAR: hide any white bar at top that contains "streamlitApp" badge */
.stApp [data-testid="stBottom"] ~ div,
[data-testid="manage-app-button"],
.stApp [data-baseweb="tag"],
.stApp a[href*="streamlit"],
[data-testid="stRunningManWidget"],
[data-testid="stConnectionStatus"],
[data-testid="manage-app-button"],
.reportview-container .main > div:first-child {
    display: none !important;
    height: 0 !important;
}
/* Hide the absolute top bar (white strip) that Streamlit Cloud injects */
.stApp::before { display: none !important; }
.stApp > div:first-child {
    min-height: 0 !important;
}
.stApp > div:first-child > div:first-child:not([data-testid="stMainBlockContainer"]):not(:has([data-testid="stMainBlockContainer"])) {
    display: none !important;
    height: 0 !important;
    max-height: 0 !important;
    overflow: hidden !important;
}
/* Last resort: any fixed bar at top of the viewport */
.stApp > header,
.stApp > div > header,
.stApp > div:first-child > div:first-child > div:first-child:empty,
[data-testid="stAppViewBlockContainer"] > div:first-child:empty {
    display: none !important; height: 0 !important;
}
/* Pill badge / "streamlitApp" label */
.stApp [data-baseweb="tag"],
.stApp [data-baseweb="badge"],
.stApp span:first-child:only-child[style*="background"] {
    display: none !important;
}

/* ── App header ── */
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

/* ── Chat history ── */
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
.chat-window::-webkit-scrollbar { width: 3px; }
.chat-window::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.10); border-radius: 3px; }

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

/* ── Panel (active stage card) ── */
.panel {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0 0 10px;
    box-shadow: none;
    animation: fadeUp 0.28s ease both;
}
.panel-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: var(--r-lg);
    padding: 22px 20px 18px;
    box-shadow: var(--shadow-md);
    animation: fadeUp 0.28s ease both;
}
.panel-title {
    display: flex; align-items: flex-end; gap: 9px;
    margin-bottom: 14px;
    /* Reset — ensure no heading-like styling leaks */
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

/* ── Inline follow-up messages ── */
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

/* ── Buttons ── */
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

/* ── Mic widget ── */
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

/* ── Text inputs ── */
[data-testid="stTextInput"] > div > div > input {
    font-family: 'Nunito', sans-serif !important;
    border-radius: var(--r-sm) !important;
    border: 1.5px solid var(--border) !important;
    padding: 7px 14px !important; font-size: 14px !important;
    background: var(--surface) !important; height: 38px !important;
    box-shadow: var(--shadow-sm) !important; color: var(--text) !important;
    transition: border-color 0.14s !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-lt) !important;
    outline: none !important;
}
[data-testid="stTextInput"] > label { display: none !important; }

/* ── Embedded send button (↑ inside text input) ── */
/* The column contains a text input followed by a button.
   We overlay the button on top of the input, aligned right. */
[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton) {
    position: relative !important;
}
[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton)
  [data-testid="stElementContainer"]:has(.stButton) {
    position: absolute !important;
    right: 8px !important;
    top: 4px !important;
    z-index: 2 !important;
    width: auto !important;
}
[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton)
  .stButton {
    display: flex !important;
    justify-content: flex-end !important;
}
[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton)
  .stButton button {
    width: 30px !important;
    height: 30px !important;
    min-height: 30px !important;
    padding: 0 !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, var(--accent) 0%, #1d7a6e 100%) !important;
    color: #fff !important;
    border: none !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(42,157,143,0.28) !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.14s ease !important;
}
[data-testid="stColumn"]:has([data-testid="stTextInput"]):has(.stButton)
  .stButton button:hover {
    background: linear-gradient(135deg, #30ada0 0%, #1a6e64 100%) !important;
    transform: scale(1.08) !important;
    box-shadow: 0 3px 12px rgba(42,157,143,0.40) !important;
}
/* Add right padding to text inputs in columns so text doesn't go under the button */
[data-testid="stColumn"]:has(.stButton) [data-testid="stTextInput"] > div > div > input {
    padding-right: 42px !important;
}

/* ── Summary card ── */
.summary-wrap {
    background: var(--surface); border: 1.5px solid var(--border);
    border-radius: var(--r-lg); padding: 26px 22px 20px;
    box-shadow: var(--shadow-md); animation: fadeUp 0.35s ease both;
}
.summary-title {
    font-family: 'Lora', serif; font-size: 20px; font-weight: 600;
    color: var(--text); margin-bottom: 4px; letter-spacing: -0.3px;
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
    display: inline-block;
    background: var(--accent-lt); color: var(--accent);
    border: 1px solid var(--accent-md); border-radius: 8px;
    padding: 2px 10px; font-size: 13px; font-weight: 600;
    margin: 2px 3px 2px 0;
}
.submitted-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #d1fae5; color: #065f46;
    border: 1.5px solid #6ee7b7; border-radius: 10px;
    padding: 5px 14px; font-size: 13px; font-weight: 700;
    margin-bottom: 16px; letter-spacing: 0.02em;
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────
# Stages:
#  -1 = name entry
#   0 = history recap (skip if no history)
#   1 = feeling scale
#   2 = pain yes/no
#   3 = body pain map (skip if no pain)
#   4 = symptom checklist
#   5 = free chat + submit
#
# KEY DESIGN: every message is tagged with the stage it belongs to.
#   {"role": "doctor"|"patient", "content": "...", "stage": 1}
# The top chat window shows messages from COMPLETED stages only.
# Messages from the CURRENT stage are rendered inline inside the panel.
defaults = {
    "messages": [], "stage": -1, "patient_name": "",
    "selected_parts": set(), "pain_yesno": None, "feeling_level": None,
    "symptoms": [], "submitted": False, "past_checkins": [],
    "last_audio_hash": None, "mic_key_counter": 0,
    "followup_counts": {}, "stage_answered": {},
    "other_input_open": {}, "note_open": False,
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# ── Core helpers ────────────────────────────────────────────
def add_doctor(text, stage=None, suggestions=None):
    s = stage if stage is not None else st.session_state.get("stage", -1)
    msg = {"role":"doctor","content":text,"stage":s}
    if suggestions:
        msg["suggestions"] = suggestions
    st.session_state.messages.append(msg)

def add_patient(text, stage=None):
    s = stage if stage is not None else st.session_state.get("stage", -1)
    st.session_state.messages.append({"role":"patient","content":text,"stage":s})

def toggle_body_part(part):
    if part in st.session_state.selected_parts: st.session_state.selected_parts.remove(part)
    else: st.session_state.selected_parts.add(part)


def is_other_open(key: str) -> bool:
    return bool(st.session_state.other_input_open.get(key, False))

def open_other(key: str):
    st.session_state.other_input_open[key] = True

def render_secondary_input_toggle(key: str, label: str = "Other"):
    if not is_other_open(key):
        if st.button(label, key=f"open_{key}", use_container_width=True):
            open_other(key)
            st.rerun()


MAX_FOLLOWUPS = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}  # Professor feedback: no GPT-driven follow-ups; branching handled deterministically

def followup_count(sid): return st.session_state.followup_counts.get(sid, 0)
def can_followup(sid):   return followup_count(sid) < MAX_FOLLOWUPS.get(sid, 0)
def record_followup(sid): st.session_state.followup_counts[sid] = followup_count(sid)+1
def is_answered(sid):    return st.session_state.stage_answered.get(sid, False)
def mark_answered(sid):  st.session_state.stage_answered[sid] = True

def advance_stage():
    s = st.session_state.stage
    if   s == 0: st.session_state.stage = 1
    elif s == 1: st.session_state.stage = 2
    elif s == 2: st.session_state.stage = 4 if st.session_state.pain_yesno is False else 3
    elif s == 3: st.session_state.stage = 4
    elif s == 4: st.session_state.stage = 5

def on_patient_answer(text: str, stage_id: int, extra_context: str = ""):
    """Record patient answer.

    Professor feedback: the baseline clinical questionnaire controls the flow.
    We do NOT let GPT invent questions. Follow-ups (if any) are deterministic
    and implemented inside each stage UI.
    """
    add_patient(text, stage=stage_id)
    mark_answered(stage_id)


def on_followup_reply(text: str, stage_id: int, extra_context: str = ""):
    """Record a patient reply to a deterministic follow-up inside the same stage."""
    add_patient(text, stage=stage_id)


def handle_voice(audio_value, stage_id: int, extra_context: str = "",
                 is_followup: bool = False) -> bool:
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
            on_followup_reply(t, stage_id, extra_context)
        else:
            on_patient_answer(t, stage_id, extra_context)
        return True
    st.warning("Could not transcribe. Please try again.")
    return False

def render_chat_window():
    """Only show messages from COMPLETED stages (stage < current stage)."""
    current = st.session_state.stage
    past_msgs = [m for m in st.session_state.messages
                 if m.get("stage", -99) < current]
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

def render_inline_stage_messages(stage_id: int, extra_context: str = ""):
    """
    Render messages from the current stage inline inside the panel.
    Doctor messages = left-aligned card. Patient messages = right-aligned.
    Suggestion buttons are shown ONLY on the very last doctor message if it has them
    and the patient hasn't replied yet.
    """
    stage_msgs = [m for m in st.session_state.messages if m.get("stage") == stage_id]
    for i, msg in enumerate(stage_msgs):
        is_last = (i == len(stage_msgs) - 1)
        if msg.get("role") == "doctor":
            st.markdown(f'<div class="inline-followup">🩺 {msg.get("content","")}</div>',
                        unsafe_allow_html=True)
            # Show suggestion buttons only on the last doctor message
            # (i.e. the one awaiting a patient reply)
            if is_last and msg.get("suggestions"):
                render_suggestion_buttons(
                    msg["suggestions"], stage_id,
                    is_followup=is_answered(stage_id),
                    extra_context=extra_context
                )
        else:
            st.markdown(f'<div class="inline-patient">{msg.get("content","")} 🙂</div>',
                        unsafe_allow_html=True)

def render_suggestion_buttons(suggestions: list, stage_id: int,
                               is_followup: bool = False,
                               extra_context: str = ""):
    """
    Render 2-3 GPT-suggested patient answers as clickable buttons.
    """
    if not suggestions:
        return
    st.markdown('<div class="small-note" style="margin-top:8px;">Quick replies:</div>',
                unsafe_allow_html=True)
    n = len(suggestions)
    cols = st.columns(n, gap="small")
    fu_count = followup_count(stage_id)
    for idx, sug in enumerate(suggestions):
        with cols[idx]:
            if st.button(sug, key=f"sug_{stage_id}_{fu_count}_{idx}",
                         use_container_width=True):
                if is_followup:
                    on_followup_reply(sug, stage_id, extra_context)
                else:
                    on_patient_answer(sug, stage_id, extra_context)
                st.rerun()

def render_followup_input(stage_id: int, extra_context: str = ""):
    """
    Secondary free-text / voice for deeper detail only.
    Hidden behind a button so quick replies remain the default interaction.
    """
    key = f"followup_{stage_id}_{followup_count(stage_id)}"
    render_secondary_input_toggle(key, label="Other / add more detail")
    if not is_other_open(key):
        return

    st.markdown('<div class="small-note">Type or speak only if the quick replies do not fit.</div>', unsafe_allow_html=True)
    c_main, c_mic = st.columns([7, 2.5], gap="small")
    with c_main:
        typed = st.text_input("", placeholder="Type more detail…",
                              key=f"fu_txt_{stage_id}_{followup_count(stage_id)}",
                              label_visibility="collapsed")
        send_clicked = st.button("↑", key=f"fu_send_{stage_id}_{followup_count(stage_id)}")
    with c_mic:
        audio_val = None
        if hasattr(st, "audio_input"):
            audio_val = st.audio_input("", label_visibility="collapsed",
                                       key=f"fu_mic_{stage_id}_{st.session_state.mic_key_counter}")
    if send_clicked and typed and typed.strip():
        on_followup_reply(typed.strip(), stage_id, extra_context)
        st.rerun()
    if handle_voice(audio_val, stage_id, extra_context, is_followup=True):
        st.rerun()

def render_next_button(label="Next →"):
    """Next button — always available once a stage is answered."""
    if st.button(label, use_container_width=True,
                 key=f"next_{st.session_state.stage}", type="primary"):
        advance_stage(); st.rerun()

def render_text_mic_row(stage_id: int, extra_context: str = "",
                        placeholder: str = "Type your answer…"):
    """
    Secondary text / voice input. Hidden until the patient taps Other.
    """
    key = f"stage_{stage_id}_other"
    render_secondary_input_toggle(key, label="Other")
    if not is_other_open(key):
        return

    st.markdown('<div class="small-note">Type or speak only if the buttons above do not fit your answer.</div>', unsafe_allow_html=True)
    c_main, c_mic = st.columns([7, 2.5], gap="small")
    with c_main:
        typed = st.text_input("", placeholder=placeholder,
                              key=f"txt_{stage_id}", label_visibility="collapsed")
        send_clicked = st.button("↑", key=f"txtsend_{stage_id}")
    with c_mic:
        audio_val = None
        if hasattr(st, "audio_input"):
            audio_val = st.audio_input("", key=f"mic_{stage_id}_{st.session_state.mic_key_counter}",
                                       label_visibility="collapsed")
    if send_clicked and typed and typed.strip():
        on_patient_answer(typed.strip(), stage_id, extra_context)
        st.rerun()
    if handle_voice(audio_val, stage_id, extra_context):
        st.rerun()

def body_svg(selected: Set[str]) -> str:
    def fill(p): return "#1f7aff" if p in selected else "#cfd8e6"
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
    st.markdown('<div class="panel"><div class="panel-title"><div class="panel-title-avatar">🩺</div><div class="panel-title-bubble">Welcome · Please enter your name</div></div>', unsafe_allow_html=True)
    name_input = st.text_input("Your name:", value=st.session_state.patient_name)
    if st.button("Start Check-In"):
        if name_input.strip():
            st.session_state.patient_name = name_input.strip()
            with st.spinner("Loading your history…"):
                st.session_state.past_checkins = load_past_checkins(name_input.strip())
            past = st.session_state.past_checkins
            if past:
                last = past[-1]
                with st.spinner("Getting your assistant ready…"):
                    opening, opening_suggestions = get_opening_message_with_suggestions(last, name_input.strip())
                add_doctor(opening, stage=0, suggestions=opening_suggestions)
                st.session_state.stage = 0
            else:
                context = (
                    f"First check-in for {name_input.strip()}."
                )
                opening = f"Hi {name_input.strip()}. Let's do a quick symptom check-in." \
                          " Please answer by tapping buttons when you can."
                add_doctor(opening, stage=1)
                st.session_state.stage = 1
            st.rerun()
        else:
            st.warning("Please enter your name to continue.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

stage = st.session_state.stage

# ── Chat window: only past-stage messages ───────────────────
render_chat_window()

# ════════════════════════════════════════════════════════════
# STAGE 0 — History recap
# ════════════════════════════════════════════════════════════
if stage == 0:
    # Professor feedback: keep this short and structured.
    # Goal: determine whether the patient is better / same / worse vs last check-in.
    history_ctx = ""

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><div class="panel-title-avatar">🩺</div><div class="panel-title-bubble">Quick update since your last visit</div></div>', unsafe_allow_html=True)

    # Render the opening recap + question (doctor) and any replies inline
    render_inline_stage_messages(stage_id=0, extra_context=history_ctx)

    # If patient hasn't replied yet, default to click-first (buttons) but allow typing/voice.
    if not is_answered(0):
        st.markdown('<div class="small-note">Tap one option above.</div>', unsafe_allow_html=True)
        render_text_mic_row(stage_id=0, extra_context=history_ctx, placeholder="Type a short reply…")

    # Once answered, offer a fast path when there are no changes.
    if is_answered(0):
        # Detect if the patient chose the structured "About the same" option.
        stage0_patient_msgs = [m for m in st.session_state.messages if m.get('stage') == 0 and m.get('role') == 'patient']
        last_reply = (stage0_patient_msgs[-1].get('content','').strip() if stage0_patient_msgs else '').lower()

        if 'about the same' in last_reply:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="small-note">If nothing is new today, you can finish in 1–2 clicks.</div>', unsafe_allow_html=True)
            no_new = st.checkbox('No new symptoms since last check-in', value=True)
            if st.button('✅ Finish quickly → Review & submit', use_container_width=True, type='primary'):
                # Copy last check-in as baseline (patient can still edit in later stages if they want).
                past = st.session_state.get('past_checkins', [])
                if past:
                    last = past[-1]
                    st.session_state.feeling_level = last.get('feeling_level', st.session_state.feeling_level)
                    st.session_state.pain_yesno = last.get('pain', st.session_state.pain_yesno)
                    st.session_state.selected_parts = set(last.get('pain_locations', []))
                    st.session_state.symptoms = list(last.get('symptoms', []))
                st.session_state.stage = 5
                st.rerun()

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        render_next_button("Start today's check-in →")

    st.markdown('</div>', unsafe_allow_html=True)

elif stage == 1:
    feeling_ctx = (
        "Patient answered how they are feeling today using the PROMIS 5-point scale (excellent/very good/good/fair/poor). "
        "Ask ONE specific follow-up about what is driving that feeling. No filler. Just the question."
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><div class="panel-title-avatar">🩺</div><div class="panel-title-bubble">How are you feeling today?</div></div>', unsafe_allow_html=True)

    # PROMIS Global Health 5-point scale — standard in oncology patient-reported outcomes
    FEELING_OPTIONS = [
        ("Excellent", "excellent"),
        ("Very Good",  "very good"),
        ("Good",      "good"),
        ("Fair",      "fair"),
        ("Poor",      "poor"),
    ]

    if not is_answered(1):
        # Show opening GPT message (tagged stage=1) inline
        render_inline_stage_messages(stage_id=1, extra_context=feeling_ctx)
        st.markdown('<div class="small-note">Choose one option below.</div>',
                    unsafe_allow_html=True)

        # 5 buttons in a single row
        opt_cols = st.columns(5, gap="small")
        for idx, (label, value) in enumerate(FEELING_OPTIONS):
            with opt_cols[idx]:
                btn_label = f"✓ {label}" if st.session_state.feeling_level == value else label
                if st.button(btn_label, key=f"feel_{idx}", use_container_width=True):
                    st.session_state.feeling_level = value
                    st.rerun()

        # Selected indicator + Send
        if st.session_state.feeling_level is not None:
            c_sel, c_send = st.columns([5, 2], gap="small")
            with c_sel:
                st.markdown(f"<div style='padding:5px 2px;font-size:14px;'>"
                            f"Selected: <b>{st.session_state.feeling_level}</b></div>",
                            unsafe_allow_html=True)
            with c_send:
                if st.button("Send ➜", key="send_feeling", use_container_width=True):
                    on_patient_answer(
                        f"I'm feeling {st.session_state.feeling_level} today.",
                        1, feeling_ctx)
                    st.rerun()

        # Free text + mic row
        st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
        render_text_mic_row(stage_id=1, extra_context=feeling_ctx,
                            placeholder="Or describe how you feel in your own words…")

    else:
        render_inline_stage_messages(stage_id=1, extra_context=feeling_ctx)
        stage1_msgs = [m for m in st.session_state.messages if m.get("stage") == 1]
        if stage1_msgs and stage1_msgs[-1].get("role") == "doctor":
            render_followup_input(stage_id=1, extra_context=feeling_ctx)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        render_next_button("Next question →")

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 2 — Pain yes/no
# ════════════════════════════════════════════════════════════
elif stage == 2:
    pain_ctx = (
        "Patient answered whether they have pain. "
        "Ask ONE follow-up about the nature or severity of their pain — "
        "e.g. how intense it is, how long they've had it, or what it feels like. "
        "IMPORTANT: Do NOT ask where the pain is located — location is collected separately in the next step. "
        "No filler."
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><div class="panel-title-avatar">🩺</div><div class="panel-title-bubble">Do you have any pain today?</div></div>', unsafe_allow_html=True)

    if not is_answered(2):
        render_inline_stage_messages(stage_id=2, extra_context=pain_ctx)
        st.markdown('<div class="small-note">Choose one option below.</div>',
                    unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1.7, 1.7, 1.6], gap="small")
        with c1:
            if st.button("✅ Yes, pain", use_container_width=True, key="pain_yes"):
                st.session_state.pain_yesno = True
                on_patient_answer("Yes, I have pain today.", 2, pain_ctx); st.rerun()
        with c2:
            if st.button("🙂 No pain", use_container_width=True, key="pain_no"):
                st.session_state.pain_yesno = False
                on_patient_answer("No, I don't have any pain today.", 2, pain_ctx); st.rerun()
        with c3:
            render_secondary_input_toggle("stage_2_other", label="Other")

        if is_other_open("stage_2_other"):
            st.markdown('<div class="small-note">Type or speak only if Yes / No does not fit.</div>', unsafe_allow_html=True)
            c_main, c_mic = st.columns([7, 2.5], gap="small")
            with c_main:
                typed_pain = st.text_input("", placeholder="Describe your pain answer…",
                                           key="txt_2", label_visibility="collapsed")
                send_pain = st.button("↑", key="txtsend_2")
            with c_mic:
                audio_pain = None
                if hasattr(st, "audio_input"):
                    audio_pain = st.audio_input("", key=f"mic_2_{st.session_state.mic_key_counter}",
                                                label_visibility="collapsed")
            if send_pain and typed_pain and typed_pain.strip():
                on_patient_answer(typed_pain.strip(), 2, pain_ctx); st.rerun()
            if handle_voice(audio_pain, 2, pain_ctx): st.rerun()

    else:
        render_inline_stage_messages(stage_id=2, extra_context=pain_ctx)
        stage2_msgs = [m for m in st.session_state.messages if m.get("stage") == 2]
        if stage2_msgs and stage2_msgs[-1].get("role") == "doctor":
            render_followup_input(stage_id=2, extra_context=pain_ctx)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        render_next_button("Next question →")

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 3 — Body pain map
# ════════════════════════════════════════════════════════════
elif stage == 3:
    location_ctx = (
        "Patient marked pain locations on a body map. "
        "Ask ONE follow-up about those locations (severity, duration, character). No filler."
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><div class="panel-title-avatar">🩺</div><div class="panel-title-bubble">Where do you feel pain?</div></div>', unsafe_allow_html=True)

    if not is_answered(3):
        render_inline_stage_messages(stage_id=3, extra_context=location_ctx)
        st.markdown('<div class="small-note">Select areas on the map, then send your selection.</div>',
                    unsafe_allow_html=True)

        col_svg, col_btns = st.columns([1, 1], gap="medium")
        with col_svg:
            st.markdown(body_svg(st.session_state.selected_parts), unsafe_allow_html=True)
        with col_btns:
            st.markdown("**Toggle regions:**")
            for part in ["Head","Chest","Abdomen","Left Arm","Right Arm","Left Leg","Right Leg"]:
                label = f"✓ {part}" if part in st.session_state.selected_parts else part
                if st.button(label, key=f"toggle_{part}", use_container_width=True):
                    toggle_body_part(part); st.rerun()
            st.markdown(
                '<div class="small-note">Selected: '
                + (", ".join(sorted(st.session_state.selected_parts)) or "None") + "</div>",
                unsafe_allow_html=True)

        c_other3, c_send3b = st.columns([1.6, 2], gap="small")
        with c_other3:
            render_secondary_input_toggle("stage_3_other", label="Other")
        with c_send3b:
            send_locs = st.button("Send locations ➜", key="send_locs", use_container_width=True)

        if is_other_open("stage_3_other"):
            st.markdown('<div class="small-note">Type or speak only if the body map does not capture it well.</div>', unsafe_allow_html=True)
            c_main3, c_mic3 = st.columns([7, 2.5], gap="small")
            with c_main3:
                typed_loc = st.text_input("", placeholder="Describe where you feel pain…",
                                          key="txt_3", label_visibility="collapsed")
                send_txt3 = st.button("↑", key="txtsend_3")
            with c_mic3:
                audio_loc = None
                if hasattr(st, "audio_input"):
                    audio_loc = st.audio_input("", key=f"mic_3_{st.session_state.mic_key_counter}",
                                               label_visibility="collapsed")
            if send_txt3 and typed_loc and typed_loc.strip():
                on_patient_answer(typed_loc.strip(), 3, location_ctx); st.rerun()
            if handle_voice(audio_loc, 3, location_ctx): st.rerun()
        if send_locs:
            loc_txt = ", ".join(sorted(st.session_state.selected_parts)) if st.session_state.selected_parts else "not sure of location"
            on_patient_answer(f"Pain locations: {loc_txt}.", 3, location_ctx); st.rerun()

    else:
        render_inline_stage_messages(stage_id=3, extra_context=location_ctx)
        stage3_msgs = [m for m in st.session_state.messages if m.get("stage") == 3]
        if stage3_msgs and stage3_msgs[-1].get("role") == "doctor":
            render_followup_input(stage_id=3, extra_context=location_ctx)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        render_next_button("Next question →")

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 4 — Symptom checklist
# ════════════════════════════════════════════════════════════
elif stage == 4:
    symptom_ctx = (
        "Patient submitted their symptom checklist. "
        "Ask ONE follow-up about their most notable symptom. No filler."
    )
    symptom_options = [
        "Fatigue / low energy","Nausea","Vomiting","Poor appetite",
        "Mouth sores","Trouble swallowing","Shortness of breath",
        "Fever / chills","Constipation","Diarrhea","Sleep problems","Anxiety / low mood",
    ]
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title"><div class="panel-title-avatar">🩺</div><div class="panel-title-bubble">Any of these symptoms today?</div></div>', unsafe_allow_html=True)

    if not is_answered(4):
        render_inline_stage_messages(stage_id=4, extra_context=symptom_ctx)
        st.markdown('<div class="small-note">Tap to select all that apply, then send your selection.</div>',
                    unsafe_allow_html=True)

        sc = st.columns(2, gap="small")
        for idx, symptom in enumerate(symptom_options):
            with sc[idx % 2]:
                label = f"✓ {symptom}" if symptom in st.session_state.symptoms else symptom
                if st.button(label, key=f"sym_{idx}", use_container_width=True):
                    if symptom in st.session_state.symptoms: st.session_state.symptoms.remove(symptom)
                    else: st.session_state.symptoms.append(symptom)
                    st.rerun()

        c_other4, c_send4b = st.columns([1.6, 2], gap="small")
        with c_other4:
            render_secondary_input_toggle("stage_4_other", label="Other")
        with c_send4b:
            send_syms = st.button("Send symptoms ➜", key="send_syms", use_container_width=True)

        if is_other_open("stage_4_other"):
            st.markdown('<div class="small-note">Type or speak only for a symptom that is not listed.</div>', unsafe_allow_html=True)
            c_main4, c_mic4 = st.columns([7, 2.5], gap="small")
            with c_main4:
                typed_sym = st.text_input("", placeholder="Type the other symptom…",
                                          key="txt_4", label_visibility="collapsed")
                send_txt4 = st.button("↑", key="txtsend_4")
            with c_mic4:
                audio_sym = None
                if hasattr(st, "audio_input"):
                    audio_sym = st.audio_input("", key=f"mic_4_{st.session_state.mic_key_counter}",
                                               label_visibility="collapsed")
            if send_txt4 and typed_sym and typed_sym.strip():
                on_patient_answer(typed_sym.strip(), 4, symptom_ctx); st.rerun()
            if handle_voice(audio_sym, 4, symptom_ctx): st.rerun()
        if send_syms:
            sym_txt = "; ".join(st.session_state.symptoms) if st.session_state.symptoms else "no symptoms from checklist"
            on_patient_answer(f"Symptoms today: {sym_txt}.", 4, symptom_ctx); st.rerun()

    else:
        render_inline_stage_messages(stage_id=4, extra_context=symptom_ctx)
        stage4_msgs = [m for m in st.session_state.messages if m.get("stage") == 4]
        if stage4_msgs and stage4_msgs[-1].get("role") == "doctor":
            render_followup_input(stage_id=4, extra_context=symptom_ctx)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        render_next_button("Finish check-in →")

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# STAGE 5 — Free chat + submit → summary
# ════════════════════════════════════════════════════════════
elif stage == 5:
    # Professor feedback: avoid open-ended free chat.
    # Stage 5 is only: (a) optional extra note for the care team, (b) submit, (c) show summary.

    if "extra_note" not in st.session_state:
        st.session_state.extra_note = ""

    if st.session_state.submitted:
        name      = st.session_state.get("patient_name","—")
        feeling   = st.session_state.get("feeling_level",None)
        pain      = st.session_state.get("pain_yesno",None)
        locations = sorted(list(st.session_state.get("selected_parts",set())))
        symptoms  = st.session_state.get("symptoms",[])

        feeling_display = feeling if feeling is not None else "—"
        pain_str  = "Yes" if pain is True else ("No" if pain is False else "—")
        sym_html  = "".join(f'<span class="tag">{s}</span>' for s in symptoms) or "<span style='opacity:.4'>None</span>"
        loc_html  = "".join(f'<span class="tag">{l}</span>' for l in locations) or "<span style='opacity:.4'>N/A</span>"

        note_text = (st.session_state.get("extra_note") or "").strip()
        conv_cell = "<span style='opacity:.4'>None</span>"
        if note_text:
            conv_cell = f"<div style='font-size:14px;line-height:1.55;color:#1a2540;white-space:pre-wrap'>{note_text}</div>"

        st.markdown(f"""
<div class="summary-wrap">
  <div class="submitted-badge">✅ Submitted</div>
  <div class="summary-title">Check-In Summary — {name}</div>
  <div class="summary-sub">Your care team will review this shortly.</div>
  <table class="summary-table">
    <tr><td>Patient name</td><td>{name}</td></tr>
    <tr><td>Feeling level</td><td>{feeling_display}</td></tr>
    <tr><td>Pain today</td><td>{pain_str}</td></tr>
    <tr><td>Pain locations</td><td>{loc_html}</td></tr>
    <tr><td>Symptoms</td><td>{sym_html}</td></tr>
    <tr><td>Additional note</td><td>{conv_cell}</td></tr>
  </table>
</div>""", unsafe_allow_html=True)

    else:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><div class="panel-title-avatar">🩺</div><div class="panel-title-bubble">Review & submit</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="small-note">Review your answers and submit. Add a note only if something important is still missing.</div>', unsafe_allow_html=True)

        if not st.session_state.note_open:
            if st.button("Other / add a note", use_container_width=True, key="open_note_button"):
                st.session_state.note_open = True
                st.rerun()
        else:
            st.session_state.extra_note = st.text_area(
                "", value=st.session_state.extra_note,
                placeholder="Type a short note for your care team…",
                height=110, label_visibility="collapsed",
            )

        if st.button("✅ Submit Check-In", use_container_width=True, type="primary"):
            # Save the baseline questionnaire data + optional note.
            try:
                _init_sheets()
                if sheet is None:
                    raise RuntimeError(f"Sheets unavailable: {sheets_init_error}")
                sheet.append_row([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    st.session_state.get("patient_name", "Unknown"),
                    json.dumps({
                        "feeling_level":  st.session_state.feeling_level,
                        "pain":           st.session_state.pain_yesno,
                        "pain_locations": sorted(list(st.session_state.selected_parts)),
                        "symptoms":       st.session_state.symptoms,
                        "extra_note":     st.session_state.extra_note,
                    })
                ])
                st.session_state.submitted = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save: {e}")

        st.markdown('</div>', unsafe_allow_html=True)
