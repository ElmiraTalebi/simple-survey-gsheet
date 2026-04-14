import hashlib
import html as _html
import io
import json
from datetime import datetime
from typing import Optional

import streamlit as st
import streamlit.components.v1 as _stc
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI


# ══════════════════════════════════════════════════════════════════
# 🔥 NEW: Adaptive LLM Question Generator
# ══════════════════════════════════════════════════════════════════

def get_llm_adaptive_question(topic_key: str, step: dict, answer: str, data: dict) -> Optional[str]:
    """
    This function lets LLM dynamically decide a follow-up question
    based on patient free-text answers.
    """

    if not openai_client:
        return None

    topic_label = _TOPIC_LABELS_LLM.get(topic_key, topic_key)

    prompt = f"""
{_SYSTEM_CONTEXT}

You are a highly experienced oncology nurse.

A patient just answered a question in a symptom check-in.

TOPIC: {topic_label}
QUESTION ASKED: "{step['text']}"
PATIENT RESPONSE: "{answer}"

CURRENT STRUCTURED DATA:
{json.dumps(data, indent=2)}

---
YOUR TASK:

1. Read the patient's response carefully.
2. Think clinically like a nurse.
3. Decide if a MORE IMPORTANT follow-up question should be asked
   instead of continuing the predefined flow.

You should ONLY ask a follow-up if:
- The answer contains useful clinical detail
- You can ask something MORE specific than the predefined flow
- The question would improve clinical understanding

Examples:
- If patient says "jaw and ear pain" → ask about radiation, infection, nerve pain
- If patient says "sharp pain when swallowing" → ask about solids vs liquids
- If patient says "burning pain" → ask about timing or triggers

If NO better question exists → return NONE

RULES:
- Ask ONLY ONE question
- Be very specific
- Use plain language
- Sound natural and human
- Do NOT repeat the original question
- Do NOT explain your reasoning

OUTPUT:
- Either a single question
- Or exactly: NONE
"""

    try:
        r = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.5,
        )
        out = r.choices[0].message.content.strip()
        if out.upper() == "NONE":
            return None
        return out
    except Exception:
        return None







def interpret_user_input_with_options(step, user_input):
    if not openai_client:
        return user_input

    options = step.get("opts", [])
    if not options:
        return user_input

    prompt = f"""
    You are a clinical assistant.
    
    QUESTION:
    "{step['text']}"
    
    OPTIONS:
    {options}
    
    PATIENT RESPONSE:
    "{user_input}"
    
    TASK:
    - If the response clearly matches ONE option → return that option EXACTLY
    - Otherwise return the original response
    
    ONLY return one line.
    """

    try:
        r = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            temperature=0
        )
        mapped = r.choices[0].message.content.strip()

        if mapped in options:
            return mapped

        return user_input

    except:
        return user_input


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ChatReport — HNC Symptom Check-In",
    page_icon="🩺",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg1: #f6f9fc;
    --bg2: #eef4fb;
    --card: rgba(255,255,255,0.88);
    --card-solid: #ffffff;
    --border: #dbe5f1;
    --border-strong: #c6d5e6;
    --text: #142033;
    --muted: #66758a;
    --primary: #2563eb;
    --primary-strong: #1d4ed8;
    --primary-soft: #eff6ff;
    --success: #0f9f6e;
    --success-soft: #ecfdf5;
    --shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
    --shadow-sm: 0 6px 18px rgba(15, 23, 42, 0.05);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

html, body, .stApp {
    background:
        radial-gradient(circle at top left, rgba(37,99,235,0.08), transparent 28%),
        radial-gradient(circle at top right, rgba(16,185,129,0.05), transparent 25%),
        linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
}

/* ── Layout ── */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1380px;
    padding-left: 1rem;
    padding-right: 1rem;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 0.65rem;
        padding-right: 0.65rem;
    }
}

/* ── Sidebar nav ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fbff 0%, #f1f6fd 100%);
    border-right: 1px solid #dce5f1;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
}

/* ── Buttons (main content area) ── */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 0.65rem 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    font-weight: 600;
    border: 1px solid var(--border);
    background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
    color: var(--text);
    transition: all 0.15s ease;
    text-align: left !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
}
.stButton > button:hover {
    border-color: #9fbfeb;
    background: #ffffff;
    color: #123664;
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(37, 99, 235, 0.08);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary) 0%, #3b82f6 100%);
    color: white;
    border: none;
    box-shadow: 0 12px 24px rgba(37, 99, 235, 0.18);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--primary-strong) 0%, var(--primary) 100%);
    color: white;
}

/* ── Sidebar nav buttons ── */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    padding: 10px 12px !important;
    font-size: 12.4px !important;
    font-weight: 600 !important;
    line-height: 1.45 !important;
    min-height: 0 !important;
    border-radius: 13px !important;
    margin-bottom: 6px !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    text-align: left !important;
    color: #173457 !important;
    border-color: #dbe5f1 !important;
    background: rgba(255,255,255,0.84) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    border-color: #9ec1ef !important;
    background: #ffffff !important;
    color: #173457 !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primaryFormSubmit"],
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
    font-size: 13.5px !important;
    font-weight: 700 !important;
    padding: 11px 14px !important;
    white-space: normal !important;
    color: white !important;
    margin-top: 4px !important;
}

/* ── Inputs ── */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
div[data-baseweb="select"] > div {
    border-radius: 14px !important;
    border: 1px solid #d6e0ec !important;
    background: rgba(255,255,255,0.98) !important;
}

/* ── Chat message wrappers ── */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    margin-bottom: 10px;
    padding: 0;
    background: transparent;
}

/* ── Topic status pills ── */
.status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    margin-left: 6px;
}
.pill-done   { background: #d1fae5; color: #065f46; }
.pill-active { background: #dbeafe; color: #1e40af; }
.pill-todo   { background: #f3f4f6; color: #6b7280; }

/* ── Modern shell cards ── */
.card {
    background: var(--card);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.6);
    border-radius: 20px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
}

.soft-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: var(--shadow-sm);
}

.chat-shell {
    background: rgba(255,255,255,0.82);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 14px;
    box-shadow: var(--shadow);
}

.assistant-chip {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    background: linear-gradient(135deg, #ffffff 0%, #f7fbff 100%);
    border: 1px solid #dbe8f7;
    border-radius: 18px;
    box-shadow: var(--shadow-sm);
    margin-bottom: 12px;
}
.assistant-chip .avatar {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #dbeafe, #e0f2fe);
    font-size: 22px;
}
.assistant-chip .name {
    font-size: 14px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 2px;
}
.assistant-chip .role {
    font-size: 12px;
    color: #64748b;
}

.memory-banner {
    background: linear-gradient(135deg, #eff6ff 0%, #f8fbff 100%);
    border: 1px solid #cfe0f6;
    border-radius: 16px;
    padding: 12px 14px;
    color: #1e3a5f;
    font-size: 13px;
    margin-bottom: 12px;
}

.report-box {
    background: rgba(255,255,255,0.94);
    border: 1px solid #dbe5f1;
    border-radius: 22px;
    padding: 24px 28px;
    font-size: 14.5px;
    line-height: 1.72;
    white-space: pre-wrap;
    box-shadow: var(--shadow);
}

/* ── Progress / completion ── */
.prog-label {
    font-size: 12px;
    color: #6b7280;
    margin-bottom: 4px;
    font-weight: 600;
}
.completion-badge {
    background: linear-gradient(135deg, #10b981, #2563eb);
    border-radius: 16px;
    padding: 16px 20px;
    color: white;
    font-weight: 700;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 12px 24px rgba(37, 99, 235, 0.16);
}

/* ── Welcome / login ── */
.welcome-card {
    background:
        radial-gradient(circle at top right, rgba(59,130,246,0.12), transparent 30%),
        linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,251,255,0.98) 100%);
    border: 1px solid #d9e6f5;
    border-radius: 26px;
    padding: 30px 34px;
    max-width: 720px;
    margin: 56px auto 24px auto;
    box-shadow: var(--shadow);
}

.subtle-note {
    background: #f8fbff;
    border: 1px dashed #ccd9ea;
    border-radius: 14px;
    padding: 10px 12px;
    font-size: 12.5px;
    color: #6b7d92;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)




def render_assistant_chip():
    st.markdown("""
    <div class="assistant-chip">
        <div class="avatar">👩‍⚕️</div>
        <div>
            <div class="name">Nurse Assistant</div>
            <div class="role">Head &amp; Neck Oncology Care Team</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_memory_banner(message: str):
    st.markdown(f'<div class="memory-banner">📋 {message}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECRETS / OPENAI
# ══════════════════════════════════════════════════════════════════

def _secret(*keys, default=None):
    for k in keys:
        if k in st.secrets:
            return st.secrets[k]
    return default


OPENAI_API_KEY = _secret("openai_api_key", "OPENAI_API_KEY", "openai_key")
openai_client: Optional[OpenAI] = None
_openai_error: Optional[str] = None

if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        _openai_error = str(e)
else:
    _openai_error = "OpenAI API key not configured."


# ══════════════════════════════════════════════════════════════════
# GOOGLE SHEETS
# ══════════════════════════════════════════════════════════════════

_sheet = None
_sheet_error: Optional[str] = None


def _init_sheets():
    global _sheet, _sheet_error
    if _sheet is not None or _sheet_error is not None:
        return
    try:
        creds = Credentials.from_service_account_info(
            _secret("gcp_service_account"),
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        book = gspread.authorize(creds).open_by_key(_secret("gsheet_id"))
        try:
            ws = book.worksheet("ChatReport")
        except Exception:
            ws = book.add_worksheet(title="ChatReport", rows=2000, cols=5)
            ws.append_row(["timestamp", "name", "all_data_json", "report"])
        _sheet = ws
    except Exception as e:
        _sheet_error = str(e)


def save_to_sheet(name: str, all_data: dict, report: str = "") -> bool:
    """
    Append one row to the Google Sheet.
    Columns: timestamp | name | all_data_json | report
    Returns True on success, False on failure.
    """
    _init_sheets()
    if _sheet is None:
        st.error(f"Could not connect to Google Sheets: {_sheet_error}")
        return False
    try:
        _sheet.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            name,
            json.dumps(all_data, ensure_ascii=False),
            report,
        ])
        return True
    except Exception as e:
        st.error(f"Failed to save to Google Sheets: {e}")
        return False


# ══════════════════════════════════════════════════════════════════
# VOICE / WHISPER
# ══════════════════════════════════════════════════════════════════

def _transcribe(audio_bytes: bytes) -> str:
    if not openai_client:
        return ""
    try:
        buf = io.BytesIO(audio_bytes)
        buf.name = "audio.wav"
        return openai_client.audio.transcriptions.create(
            model="whisper-1", file=buf
        ).text.strip()
    except Exception:
        return ""


def voice_widget(key_suffix: str) -> Optional[str]:
    """Renders voice recorder. Returns transcript string if new audio was processed."""
    transcript_key = f"_vt_{key_suffix}"
    hash_key = f"_vh_{key_suffix}"
    if hash_key not in st.session_state:
        st.session_state[hash_key] = None

    audio = st.audio_input("🎤 Speak your answer", key=f"_vrec_{key_suffix}")
    if not audio:
        return st.session_state.get(transcript_key)

    try:
        ab = audio.getvalue()
    except Exception:
        return st.session_state.get(transcript_key)

    if not ab:
        return st.session_state.get(transcript_key)

    ah = hashlib.sha1(ab).hexdigest()
    if ah == st.session_state[hash_key]:
        return st.session_state.get(transcript_key)

    st.session_state[hash_key] = ah
    with st.spinner("Transcribing…"):
        text = _transcribe(ab)

    if text:
        st.session_state[transcript_key] = text
        st.rerun()

    return st.session_state.get(transcript_key)


# ══════════════════════════════════════════════════════════════════
# TOPIC & FLOW DEFINITIONS
# ══════════════════════════════════════════════════════════════════

# Each entry: (display_label, internal_key)
TOPICS = [
    ("🩹 Pain & Medications",   "pain"),
    ("🍽️  Nutrition & Fluids",   "nutrition"),
    ("👄 Oral Symptoms",         "oral"),
    ("🤢 GI Symptoms",           "gi"),
    ("😴 Fatigue & Sleep",       "fatigue"),
    ("🚶 Activity Level",        "activity"),
    ("🧠 Mood",                  "mood"),
    ("💊 Other Symptoms",        "other"),
]

TOPIC_INTROS = {
    "pain":      "Let's talk about any pain you've been having and the medications you're using.",
    "nutrition": "I'd like to ask about your eating, drinking, and weight.",
    "oral":      "Let's go over any problems in your mouth — sores, dryness, mucus, etc.",
    "gi":        "I'll ask about nausea, vomiting, and bowel habits.",
    "fatigue":   "Let's discuss how your energy and sleep have been.",
    "activity":  "Tell me about how your daily activities have been going.",
    "mood":      "This section covers how you've been feeling emotionally and your support system.",
    "other":     "Finally, let's cover any other symptoms — breathing, skin, hearing, and more.",
}


def _q(id, text, type="options", opts=None, when=None,
        placeholder="Please describe...", min_v=0, max_v=10, default_v=0):
    """Helper to build a question step dict."""
    return {
        "id": id, "text": text, "type": type,
        "opts": opts or [], "when": when,
        "placeholder": placeholder,
        "min_v": min_v, "max_v": max_v, "default_v": default_v,
    }


def _safe_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ── PAIN & MEDICATIONS (Main 2, 3, 12, 38) ────────────────────────
FLOW_PAIN = [
    # Main 2
    _q("has_pain", "Do you have any pain today?", opts=["Yes", "No"]),

    # Main 3 — location
    _q("pain_location", "Where exactly is the pain?",
       opts=["Throat", "Tongue", "Somewhere else"],
       when=lambda d: d.get("has_pain") == "Yes"),

    # ── Throat branch ──
    _q("throat_timing",
       "Is the throat pain there all the time, or only when you swallow or eat?",
       opts=["All the time", "Only when swallowing", "Only when eating",
             "Both swallowing and eating"],
       when=lambda d: d.get("pain_location") == "Throat"),

    _q("throat_severity",
       "On a scale of 0–10, how bad is the throat pain at its worst?",
       type="number", min_v=0, max_v=10, default_v=5,
       when=lambda d: d.get("pain_location") == "Throat"),

    _q("throat_med_helps",
       "Are you taking pain medication for this? Is it helping?",
       opts=["Yes, it helps", "Yes, but it's not enough", "No, I'm not taking anything"],
       when=lambda d: (d.get("pain_location") == "Throat"
                       and _safe_int(d.get("throat_severity", 0)) > 4)),

    # ── Tongue branch ──
    _q("tongue_type",
       "Is it a sore or ulcer on the tongue, or a general painful feeling?",
       opts=["There's a sore/ulcer", "Just pain, no visible sore"],
       when=lambda d: d.get("pain_location") == "Tongue"),

    _q("tongue_spot",
       "Is the pain in one specific spot, or does it spread?",
       opts=["One spot", "Spreads across tongue", "Whole mouth"],
       when=lambda d: d.get("pain_location") == "Tongue"),

    _q("tongue_severity",
       "On a scale of 0–10, how bad is the tongue pain at its worst?",
       type="number", min_v=0, max_v=10, default_v=5,
       when=lambda d: d.get("pain_location") == "Tongue"),

    # ── Somewhere else branch ──
    _q("other_pain_desc",
       "Can you describe where the pain is?",
       type="free_text", placeholder="e.g., near my jaw and ear…",
       when=lambda d: d.get("pain_location") == "Somewhere else"),

    _q("ear_pain", "Do you have ear pain or hearing changes?",
       opts=["Yes", "No"],
       when=lambda d: d.get("pain_location") == "Somewhere else"),

    _q("jaw_swelling", "Do you feel any swelling near your jaw?",
       opts=["Yes", "No"],
       when=lambda d: d.get("pain_location") == "Somewhere else"),

    _q("pain_with_chewing",
       "Does the pain worsen when chewing or opening your mouth?",
       opts=["Yes", "No"],
       when=lambda d: d.get("pain_location") == "Somewhere else"),

    _q("pain_start",                        # ← added (Main 3, Somewhere else branch)
       "When did this pain start?",
       type="free_text",
       placeholder="e.g., about a week ago, since I started radiation…",
       when=lambda d: d.get("pain_location") == "Somewhere else"),

    # Main 12 — Medications
    _q("pain_medications",
       "Which medications are you currently taking for pain?",
       type="multi_select",
       opts=["Gabapentin", "Oxycodone", "Butrans patch", "Other", "No pain medication"]),

    _q("med_dose_freq",
       "How often are you taking your pain medication, and at what dose?",
       type="free_text", placeholder="e.g., Oxycodone 5mg every 6 hours…",
       when=lambda d: (bool(d.get("pain_medications"))
                       and "No pain medication" not in (d.get("pain_medications") or []))),

    # Main 38 — Adherence
    _q("taking_as_prescribed",
       "Are you taking your medications as prescribed?",
       opts=["Yes", "No"]),

    _q("med_adherence_issue",
       "What is making it difficult to take your medications?",
       opts=["Side effects", "Schedule", "Access issues", "Other"],
       when=lambda d: d.get("taking_as_prescribed") == "No"),

    _q("med_side_effects",
       "Are you experiencing any side effects from your medications?",
       opts=["Yes", "No"],
       when=lambda d: d.get("taking_as_prescribed") == "Yes"),
]

# ── NUTRITION & FLUIDS (Main 5, 6, 8, 25, 26, 27, 34) ─────────────
FLOW_NUTRITION = [
    # Main 5 — Eating ability
    _q("eating_ability",
       "How has your eating been since your last visit?",
       opts=["Eating normally — no problems",
             "Eating less than usual, but managing",
             "Struggling — only liquids or very little",
             "Not eating — using a feeding tube only"]),

    # Branch: Eating less
    _q("fluid_intake_managing",
       "Are you drinking enough fluids throughout the day — water, shakes, or other drinks?",
       opts=["Yes, drinking well", "A little less than usual", "Struggling to drink enough"],
       when=lambda d: d.get("eating_ability") == "Eating less than usual, but managing"),

    _q("food_type",
       "What are you able to eat right now?",
       opts=["Mostly normal food", "Soft foods only (yogurt, soup, pudding)",
             "Mix of soft and liquid", "Mainly liquids"],
       when=lambda d: d.get("eating_ability") == "Eating less than usual, but managing"),

    # Branch: Struggling
    _q("nutritional_shakes",
       "How many nutritional shakes or Boost/Ensure drinks are you having per day?",
       opts=["None", "1–2", "3–4", "More than 4"],
       when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),

    _q("eating_barrier",
       "What is stopping you from eating more?",
       opts=["Pain when eating/swallowing", "Feel full very quickly",
             "No appetite", "Nausea", "Too tired to prepare food"],
       when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),

    _q("fluid_struggling",
       "Are you drinking enough fluids — water, juice, or anything?",
       opts=["Yes, drinking well", "A little", "Very little, hard to drink"],
       when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),

    _q("fluid_barrier",
       "What's making it hard to drink?",
       opts=["Pain when swallowing", "Dry mouth", "Nausea", "Just not thirsty"],
       when=lambda d: (d.get("eating_ability") == "Struggling — only liquids or very little"
                       and d.get("fluid_struggling") in ["A little", "Very little, hard to drink"])),

    _q("pain_med_timing",
       "Are you timing your pain medication before meals to make eating easier?",
       opts=["Yes, it helps", "I try, but it's not enough",
             "No, I didn't know to do this", "No, I don't take pain medication"],
       when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),

    # Branch: Tube only
    _q("tube_issues",
       "Is the tube feeding going well — no blockages, leaks, or discomfort around the site?",
       opts=["Working fine", "Some issues — leaking or blockage",
             "Discomfort/soreness around the tube"],
       when=lambda d: d.get("eating_ability") == "Not eating — using a feeding tube only"),

    _q("tube_oral_sips",
       "Are you still able to take any sips of water or liquids by mouth at all?",
       opts=["Yes, small amounts", "Very occasionally for comfort", "No, nothing by mouth"],
       when=lambda d: d.get("eating_ability") == "Not eating — using a feeding tube only"),

    # Main 6 — Weight
    _q("weight",
       "What has your weight been recently? (Enter in pounds)",
       type="number", min_v=50, max_v=500, default_v=150),

    _q("weight_impact",
       "Has any weight change been affecting how you feel or your energy levels?",
       opts=["Yes, I've noticed a difference", "Not really"]),

    # Main 8 — Swallowing
    _q("swallowing_difficulty",
       "Are you having any difficulty swallowing — liquids, food, or pills?",
       opts=["Yes", "No"]),

    _q("swallowing_type",
       "Is it painful to swallow, or just mechanically difficult?",
       opts=["Painful to swallow", "Mechanically difficult"],
       when=lambda d: d.get("swallowing_difficulty") == "Yes"),

    _q("choking_with_eating",
       "Do you cough or choke when you eat?",
       opts=["Yes", "No"],
       when=lambda d: d.get("swallowing_difficulty") == "Yes"),

    _q("swallowing_method",
       "Are you still able to swallow liquids by mouth, or is everything through a feeding tube?",
       opts=["I swallow by mouth", "Everything through the feeding tube"],
       when=lambda d: d.get("swallowing_difficulty") == "Yes"),

    # Main 25 — Choking/Coughing (standalone — separate from Main 8)
    _q("choking_coughing",
       "Are you having any difficulty with choking or coughing when eating or drinking?",
       opts=["Yes", "No"]),

    _q("choking_type",
       "Does it happen with liquids, solids, or both?",
       opts=["Liquids", "Solids", "Both"],
       when=lambda d: d.get("choking_coughing") == "Yes"),

    _q("choking_frequency",
       "Does it happen every time you eat, or only occasionally?",
       opts=["Every time", "Occasionally"],
       when=lambda d: d.get("choking_coughing") == "Yes"),

    _q("choking_pills",                     # ← added (Main 25)
       "Does it also happen when you take pills?",
       opts=["Yes", "No"],
       when=lambda d: d.get("choking_coughing") == "Yes"),

    # Main 26 — IV Fluids
    _q("iv_fluids",
       "Are you currently receiving IV fluids or hydration treatments?",
       opts=["Yes", "No"]),

    _q("iv_frequency",
       "How often are you receiving IV fluids?",
       type="free_text", placeholder="e.g., twice a week…",
       when=lambda d: d.get("iv_fluids") == "Yes"),

    _q("iv_helping",
       "Do you feel the IV fluids are helping?",
       opts=["Yes", "No"],
       when=lambda d: d.get("iv_fluids") == "Yes"),

    _q("iv_adjust",                         # ← added (Main 26)
       "Would you like to adjust the frequency of your hydration visits?",
       opts=["Yes", "No"],
       when=lambda d: d.get("iv_fluids") == "Yes"),

    _q("need_hydration",
       "Do you feel like you might need hydration support?",
       opts=["Yes", "No"],
       when=lambda d: d.get("iv_fluids") == "No"),

    # Main 27 — Feeding tube (for patients not already exclusively tube-fed)
    _q("feeding_tube",
       "Are you currently using a feeding tube?",
       opts=["Yes", "No"],
       when=lambda d: d.get("eating_ability") != "Not eating — using a feeding tube only"),

    _q("tube_status",
       "Is the feeding tube working well or are there issues?",
       opts=["Working well", "Leakage", "Blockage", "Discomfort"],
       when=lambda d: (d.get("feeding_tube") == "Yes"
                       and d.get("eating_ability") != "Not eating — using a feeding tube only")),

    _q("tube_oral",
       "Are you able to take anything by mouth at all?",
       opts=["Yes, some", "No, nothing by mouth"],
       when=lambda d: (d.get("feeding_tube") == "Yes"
                       and d.get("eating_ability") != "Not eating — using a feeding tube only")),

    # Main 34 — Taste
    _q("taste_changes",
       "Have you noticed any changes in your sense of taste?",
       opts=["Yes", "No"]),

    _q("taste_type",
       "Does food taste different, bland, or unpleasant?",
       opts=["Different", "Bland", "Unpleasant"],
       when=lambda d: d.get("taste_changes") == "Yes"),

    _q("taste_eating_impact",
       "Is the taste change affecting your ability to eat?",
       opts=["Yes", "No"],
       when=lambda d: d.get("taste_changes") == "Yes"),
]

# ── ORAL SYMPTOMS (Main 4, 7, 10, 24, 33) ─────────────────────────
FLOW_ORAL = [
    # Main 4 — Mouth sores
    _q("mouth_sores",
       "Do you have any mouth sores or ulcers right now?",
       opts=["Yes", "No"]),

    _q("sore_new_or_old",
       "Is this sore new since your last visit, or have you had it for a while?",
       opts=["New", "Not sure", "Same one as before"],
       when=lambda d: d.get("mouth_sores") == "Yes"),

    _q("sore_location",
       "Where exactly is the sore?",
       opts=["Inside the mouth/cheek", "On the tongue", "Back of the throat",
             "Gums/lips", "Multiple spots"],
       when=lambda d: (d.get("mouth_sores") == "Yes"
                       and d.get("sore_new_or_old") in ["New", "Not sure"])),

    _q("sore_pain_impact",
       "Is the sore painful? Is it affecting your ability to eat or drink?",
       opts=["No pain, just noticed it", "A little, but manageable",
             "Yes, can't eat/drink comfortably"],
       when=lambda d: (d.get("mouth_sores") == "Yes"
                       and d.get("sore_new_or_old") in ["New", "Not sure"])),

    _q("magic_mouthwash",
       "Are you using magic mouthwash? If yes, is it helping?",
       opts=["Yes, it helps", "Yes, but not enough",
             "No, I don't have it", "No, I don't use it"],
       when=lambda d: (d.get("mouth_sores") == "Yes"
                       and d.get("sore_new_or_old") in ["New", "Not sure"])),

    _q("sore_progression",
       "Is the sore getting better, staying the same, or getting worse?",
       opts=["Getting better", "About the same", "Getting worse", "Not sure"],
       when=lambda d: (d.get("mouth_sores") == "Yes"
                       and d.get("sore_new_or_old") == "Same one as before")),

    _q("sore_eating_impact_old",
       "Is it still preventing you from eating or drinking comfortably?",
       opts=["Yes", "A little", "No"],
       when=lambda d: (d.get("mouth_sores") == "Yes"
                       and d.get("sore_new_or_old") == "Same one as before"
                       and d.get("sore_progression") in ["About the same", "Getting worse"])),

    # Main 7 — Dry mouth
    _q("dry_mouth",
       "Are you experiencing any dryness in your mouth?",
       opts=["Yes", "No"]),

    _q("dry_mouth_timing",
       "Is the dryness worse at night or all day?",
       opts=["Worse at night", "All day"],
       when=lambda d: d.get("dry_mouth") == "Yes"),

    _q("dry_mouth_med",
       "Are you using any medication like Biotene or a saliva substitute?",
       opts=["Yes", "No"],
       when=lambda d: d.get("dry_mouth") == "Yes"),

    _q("dry_mouth_impact",
       "Is the dryness making it harder to eat, talk, or sleep?",
       opts=["Yes", "No"],
       when=lambda d: d.get("dry_mouth") == "Yes"),

    # Main 10 — Mucus / thick secretions
    _q("mucus_issues",
       "Are you having problems with mucus or thick secretions in your throat?",
       opts=["Yes", "No"]),

    _q("mucus_type",
       "Is the mucus thick and hard to clear, or more watery?",
       opts=["Thick", "More watery"],
       when=lambda d: d.get("mucus_issues") == "Yes"),

    _q("mucus_impact",
       "Is the mucus affecting your ability to swallow or sleep?",
       opts=["Yes", "No"],
       when=lambda d: d.get("mucus_issues") == "Yes"),

    _q("mucus_management",
       "Are you using anything to manage it — like Robitussin or saline rinses?",
       opts=["Yes", "No"],
       when=lambda d: d.get("mucus_issues") == "Yes"),

    # Main 24 — Teeth / Gums
    _q("teeth_gum_issues",
       "Are you having any problems with your teeth or gums?",
       opts=["Yes", "No"]),

    _q("teeth_issue_type",
       "Is there pain, bleeding, or sores with your teeth or gums?",
       opts=["Pain", "Bleeding", "Sores", "Multiple issues"],
       when=lambda d: d.get("teeth_gum_issues") == "Yes"),

    _q("brushing_difficult",
       "Is it making brushing difficult?",
       opts=["Yes", "No"],
       when=lambda d: d.get("teeth_gum_issues") == "Yes"),

    _q("avoiding_brushing",                 # ← added (Main 24)
       "Are you avoiding brushing because of the discomfort?",
       opts=["Yes", "No"],
       when=lambda d: d.get("teeth_gum_issues") == "Yes"),

    # Main 33 — Oral rinses
    _q("oral_rinse_use",
       "Are you using mouthwash or oral rinses regularly?",
       opts=["Yes", "No"]),

    _q("oral_rinse_type",
       "What type are you using?",
       type="free_text",
       placeholder="e.g., magic mouthwash, salt/baking soda rinse…",
       when=lambda d: d.get("oral_rinse_use") == "Yes"),

    _q("oral_rinse_helping",
       "Is it helping?",
       opts=["Yes", "No"],
       when=lambda d: d.get("oral_rinse_use") == "Yes"),

    _q("oral_rinse_open",
       "Would you be open to trying an oral rinse to help with symptoms?",
       opts=["Yes", "No"],
       when=lambda d: d.get("oral_rinse_use") == "No"),
]

# ── GI SYMPTOMS (Main 11, 18) ─────────────────────────────────────
FLOW_GI = [
    # Main 11 — Nausea / Vomiting / Blood
    _q("nausea_vomiting",
       "Have you had any nausea, vomiting, or noticed any blood when you cough?",
       type="multi_select",
       opts=["Nausea", "Vomiting", "Blood when coughing", "None of these"]),

    _q("nausea_frequency",
       "How often are you feeling nauseated?",
       type="free_text",
       placeholder="e.g., a few times a day, mostly in the mornings…",
       when=lambda d: "Nausea" in (d.get("nausea_vomiting") or [])),

    _q("vomiting_frequency",
       "How often are you vomiting and how much?",
       type="free_text",
       placeholder="e.g., once or twice a day, small amounts…",
       when=lambda d: "Vomiting" in (d.get("nausea_vomiting") or [])),

    _q("blood_cough_amount",
       "How much blood have you noticed when coughing?",
       type="free_text",
       placeholder="e.g., small streaks, about a teaspoon…",
       when=lambda d: "Blood when coughing" in (d.get("nausea_vomiting") or [])),

    # Main 18 — Constipation
    _q("constipation",
       "Have you had any constipation or trouble moving your bowels?",
       opts=["Yes", "No"]),

    _q("bowel_frequency",
       "How often are you having bowel movements?",
       type="free_text",
       placeholder="e.g., once every 3 days…",
       when=lambda d: d.get("constipation") == "Yes"),

    _q("constipation_meds",
       "Are you taking anything like Senna, Miralax, or other medications for constipation?",
       opts=["Yes", "No"],
       when=lambda d: d.get("constipation") == "Yes"),

    _q("bloating",
       "Are you feeling bloated or uncomfortable?",
       opts=["Yes", "No"],
       when=lambda d: d.get("constipation") == "Yes"),
]

# ── FATIGUE & SLEEP (Main 13, 14) ─────────────────────────────────
FLOW_FATIGUE = [
    # Main 13 — Fatigue / Weakness
    _q("fatigue",
       "Are you feeling more tired or weak than usual?",
       opts=["Yes", "No"]),

    _q("fatigue_type",
       "Is it a general tiredness, or weakness in specific parts of your body?",
       opts=["General tiredness", "Weakness in specific parts"],
       when=lambda d: d.get("fatigue") == "Yes"),

    _q("weakness_location",
       "In which parts of your body do you feel weakness?",
       type="free_text", placeholder="e.g., legs, arms…",
       when=lambda d: (d.get("fatigue") == "Yes"
                       and d.get("fatigue_type") == "Weakness in specific parts")),

    _q("fatigue_daily_impact",
       "Is the fatigue affecting your daily activities — getting dressed, moving around?",
       opts=["Yes", "No"],
       when=lambda d: d.get("fatigue") == "Yes"),

    # Main 14 — Drowsiness + Sleep
    _q("medication_drowsy",
       "Are your pain medications making you feel drowsy?",
       opts=["Yes", "No", "Sometimes"]),

    _q("sleep_quality",
       "Are you able to sleep through the night?",
       opts=["Yes", "No"]),

    _q("sleep_wake_reason",
       "Are you waking up at night due to pain, dry mouth, or coughing?",
       type="free_text",
       placeholder="e.g., pain wakes me up around 3am…",
       when=lambda d: d.get("sleep_quality") == "No"),

    _q("drowsy_schedule",                   # ← fixed: now conditional on sleep_quality, not drowsiness
       "Is drowsiness from medication affecting your normal wake/sleep schedule?",
       opts=["Yes", "No"],
       when=lambda d: d.get("sleep_quality") == "No"),
]

# ── ACTIVITY LEVEL (Main 30) ───────────────────────────────────────
FLOW_ACTIVITY = [
    _q("activity_level",
       "How is your daily life — are you able to do your usual activities?",
       opts=["Doing everything normally", "Doing less than usual",
             "Struggling with daily tasks"]),

    _q("difficult_activities",
       "What activities are most difficult right now?",
       type="free_text",
       placeholder="e.g., climbing stairs, cooking, getting dressed…",
       when=lambda d: d.get("activity_level") in
             ["Doing less than usual", "Struggling with daily tasks"]),

    _q("activity_limiting_factor",
       "Is the difficulty mainly due to pain, fatigue, or something else?",
       opts=["Pain", "Fatigue", "Both", "Something else"],
       when=lambda d: d.get("activity_level") in
             ["Doing less than usual", "Struggling with daily tasks"]),

    _q("activity_other_desc",
       "Can you tell me more about what's limiting your activities?",
       type="free_text", placeholder="e.g., balance issues, weakness…",
       when=lambda d: d.get("activity_limiting_factor") == "Something else"),
]

# ── MOOD (Main 15, 35, 39) ─────────────────────────────────────────
FLOW_MOOD = [
    # Main 15 — Emotional state / Anxiety
    _q("emotional_state",
       "How are you feeling emotionally? Are you feeling anxious or worried about anything?",
       type="free_text",
       placeholder="Please share how you've been feeling — there are no wrong answers…"),

    _q("anxiety_impact",
       "Is anxiety or worry affecting your sleep, eating, or daily activities?",
       opts=["Yes", "No", "A little"]),

    _q("social_support_quality",
       "Do you have people around you who you can talk to about how you're feeling?",
       opts=["Yes, I have good support", "Some support", "Not really"]),

    # Main 35 — Depression
    _q("feeling_down",
       "Have you been feeling down or depressed?",
       opts=["Yes", "No"]),

    _q("depression_frequency",
       "How often have you been feeling this way?",
       type="free_text",
       placeholder="e.g., most days, occasionally, mostly in the evenings…",
       when=lambda d: d.get("feeling_down") == "Yes"),

    _q("depression_daily_impact",
       "Is it affecting your daily activities or motivation?",
       opts=["Yes", "No"],
       when=lambda d: d.get("feeling_down") == "Yes"),

    # Main 39 — Support between visits
    _q("support_adequate",
       "Do you feel you have enough support between visits?",
       opts=["Yes", "No"]),

    _q("who_supports",
       "Who is supporting you — family, friends, or caregivers?",
       type="free_text",
       placeholder="e.g., my wife and daughter…",
       when=lambda d: d.get("support_adequate") == "Yes"),

    _q("needed_support",
       "What kind of support would be most helpful right now?",
       type="free_text",
       placeholder="e.g., emotional support, help with transportation, more info about treatment…",
       when=lambda d: d.get("support_adequate") == "No"),
]

# ── OTHER SYMPTOMS (Main 9, 16, 17, 19, 20, 21, 22, 23, 36, 37) ───
FLOW_OTHER = [
    # Main 9 — Breathing
    _q("breathing_issues",
       "Are you having any difficulty breathing or shortness of breath?",
       opts=["Yes", "No"]),

    _q("breathing_timing",
       "Is the breathing difficulty constant, or does it come on with activity?",
       opts=["It's constant", "It comes on with activity"],
       when=lambda d: d.get("breathing_issues") == "Yes"),

    _q("wheezing",
       "Are you wheezing or feeling like something is blocking your airway?",
       opts=["Yes", "No"],
       when=lambda d: d.get("breathing_issues") == "Yes"),

    # Main 16 — Hearing
    _q("hearing_changes",
       "Do you have any hearing problems or changes recently?",
       opts=["Yes", "No"]),

    _q("hearing_type",
       "Is it ringing in your ears, hearing loss, or both?",
       opts=["Ringing in ears", "Hearing loss", "Both"],
       when=lambda d: d.get("hearing_changes") == "Yes"),

    _q("hearing_constant",
       "Is it constant or does it come and go?",
       opts=["Constant", "Comes and goes"],
       when=lambda d: d.get("hearing_changes") == "Yes"),

    _q("hearing_worsening",
       "Has it gotten worse compared to your last visit?",
       opts=["Yes", "No"],
       when=lambda d: d.get("hearing_changes") == "Yes"),

    # Main 17 — Dizziness
    _q("dizziness",
       "Have you been feeling dizzy or lightheaded?",
       opts=["Yes", "No"]),

    _q("dizziness_timing",
       "Is it constant or only when you stand up or change position?",
       opts=["Constant", "Only when standing or changing position"],
       when=lambda d: d.get("dizziness") == "Yes"),

    _q("dizziness_worsening",               # ← added (Main 17)
       "Has the dizziness gotten worse recently?",
       opts=["Yes", "No"],
       when=lambda d: d.get("dizziness") == "Yes"),

    _q("falls",
       "Have you had any falls or felt like you might fall?",
       opts=["Yes", "No"],
       when=lambda d: d.get("dizziness") == "Yes"),

    # Main 19 — Numbness / Tingling
    _q("numbness",
       "Have you noticed any numbness or tingling in your hands or feet?",
       opts=["Yes", "No"]),

    _q("numbness_location",
       "Is it in your hands, feet, or both?",
       opts=["Hands", "Feet", "Both"],
       when=lambda d: d.get("numbness") == "Yes"),

    _q("numbness_new",
       "Is it new or getting worse?",
       opts=["New", "Getting worse", "Same as before"],
       when=lambda d: d.get("numbness") == "Yes"),

    _q("numbness_daily_impact",
       "Is it affecting your daily activities?",
       opts=["Yes", "No"],
       when=lambda d: d.get("numbness") == "Yes"),

    # Main 20 — Fever / Chills
    _q("fever_chills",
       "Have you had any fever or chills recently?",
       opts=["Yes", "No"]),

    _q("fever_start", "When did the fever or chills start?",
       type="free_text", placeholder="e.g., two days ago…",
       when=lambda d: d.get("fever_chills") == "Yes"),

    _q("fever_temp", "How high was the fever?",
       type="free_text", placeholder="e.g., 101.5°F…",
       when=lambda d: d.get("fever_chills") == "Yes"),

    _q("fever_other_symptoms",
       "Do you have any other symptoms like cough or signs of infection?",
       opts=["Yes", "No"],
       when=lambda d: d.get("fever_chills") == "Yes"),

    # Main 21 — Blood pressure
    _q("bp_monitoring",
       "Are you checking your blood pressure at home?",
       opts=["Yes", "No"]),

    _q("bp_reading", "What has your blood pressure been recently?",
       type="free_text", placeholder="e.g., 130/85…",
       when=lambda d: d.get("bp_monitoring") == "Yes"),

    _q("bp_dizziness",
       "Have you felt dizzy or lightheaded with blood pressure changes?",
       opts=["Yes", "No"],
       when=lambda d: d.get("bp_monitoring") == "Yes"),

    _q("bp_home_monitor",
       "Do you have a way to check your blood pressure at home?",
       opts=["Yes", "No"],
       when=lambda d: d.get("bp_monitoring") == "No"),

    # Main 22 — Skin
    _q("skin_issues",
       "Have you had any skin problems — like irritation, wounds, or redness?",
       opts=["Yes", "No"]),

    _q("skin_location", "Where is the skin issue located?",
       type="free_text", placeholder="e.g., neck, shoulder, near jaw…",
       when=lambda d: d.get("skin_issues") == "Yes"),

    _q("skin_start",                        # ← added (Main 22)
       "When did it start?",
       type="free_text",
       placeholder="e.g., about a week ago, at the start of radiation…",
       when=lambda d: d.get("skin_issues") == "Yes"),

    _q("skin_progression",
       "Is it getting better, worse, or staying the same?",
       opts=["Getting better", "About the same", "Getting worse"],
       when=lambda d: d.get("skin_issues") == "Yes"),

    _q("skin_drainage",
       "Any drainage, bleeding, or open areas?",
       opts=["Yes", "No"],
       when=lambda d: d.get("skin_issues") == "Yes"),

    # Main 23 — Voice / Hoarseness
    _q("voice_hoarseness",
       "How is your voice? Have you noticed any hoarseness or trouble speaking?",
       opts=["Yes, problems with my voice", "No, voice is fine"]),

    _q("voice_timing",
       "Is the hoarseness constant or only when you're talking?",
       opts=["Constant", "Only when talking"],
       when=lambda d: d.get("voice_hoarseness") == "Yes, problems with my voice"),

    _q("voice_progression",
       "Has your voice improved or worsened since your last visit?",
       opts=["Improved", "About the same", "Worse"],
       when=lambda d: d.get("voice_hoarseness") == "Yes, problems with my voice"),

    _q("voice_communication_impact",
       "Is it affecting your ability to communicate with others?",
       opts=["Yes", "No"],
       when=lambda d: d.get("voice_hoarseness") == "Yes, problems with my voice"),

    # Main 36 — Concentration / Memory
    _q("concentration",
       "Have you had trouble concentrating or remembering things?",
       opts=["Yes", "No"]),

    _q("concentration_new",
       "Is it new or ongoing?",
       opts=["New", "Ongoing"],
       when=lambda d: d.get("concentration") == "Yes"),

    _q("concentration_daily_impact",
       "Is it affecting your daily tasks?",
       opts=["Yes", "No"],
       when=lambda d: d.get("concentration") == "Yes"),

    # Main 37 — Sexual health
    _q("sexual_health",
       "Have you had any sexual health concerns or changes?",
       opts=["Yes", "Prefer not to say", "No"]),

    _q("sexual_discuss",
       "Would you like to discuss this further with your provider?",
       opts=["Yes", "No"],
       when=lambda d: d.get("sexual_health") == "Yes"),

    _q("sexual_cause",                      # ← added (Main 37)
       "Is it related to treatment, energy levels, or something else?",
       opts=["Treatment side effects", "Energy levels", "Other"],
       when=lambda d: d.get("sexual_health") == "Yes"),
]

# Master flow registry
FLOWS = {
    "pain":      FLOW_PAIN,
    "nutrition": FLOW_NUTRITION,
    "oral":      FLOW_ORAL,
    "gi":        FLOW_GI,
    "fatigue":   FLOW_FATIGUE,
    "activity":  FLOW_ACTIVITY,
    "mood":      FLOW_MOOD,
    "other":     FLOW_OTHER,
}



# ══════════════════════════════════════════════════════════════════
# FLOW ENGINE
# ══════════════════════════════════════════════════════════════════

def get_next_step(topic_key: str, data: dict) -> Optional[dict]:
    """Return the first unanswered applicable step for this topic."""
    for step in FLOWS.get(topic_key, []):
        when = step.get("when")
        if when and not when(data):
            continue
        if step["id"] not in data:
            return step
    return None


def topic_is_complete(topic_key: str, data: dict) -> bool:
    return get_next_step(topic_key, data) is None


def get_topic_progress(topic_key: str, data: dict) -> tuple[int, int]:
    """Returns (answered, applicable) counts."""
    flow = FLOWS.get(topic_key, [])
    applicable = [s for s in flow if not s.get("when") or s["when"](data)]
    answered = [s for s in applicable if s["id"] in data]
    return len(answered), len(applicable)


# ══════════════════════════════════════════════════════════════════
# LLM FUNCTIONS
# ══════════════════════════════════════════════════════════════════

# ── Shared clinical context injected into every LLM call ─────────
_SYSTEM_CONTEXT = (
    "You are a compassionate, clinically trained nurse at a head and neck cancer (HNC) center. "
    "You are conducting a structured symptom check-in with a patient currently receiving "
    "chemoradiation for head and neck cancer. "
    "This patient population frequently experiences: severe mucositis, dysphagia, pain, "
    "significant weight loss, fatigue, depression, and impaired communication. "
    "Many patients have low health literacy or face barriers to care. "
    "Your tone is always warm, clear, and non-alarming — even when probing for clinically "
    "urgent information. Never use medical jargon without explaining it simply. "
    "Never minimize a patient's reported symptom."
)

# ── Human-readable topic labels for prompt context ───────────────
_TOPIC_LABELS_LLM = {
    "pain":      "Pain & Pain Medications",
    "nutrition": "Nutrition, Fluids & Weight",
    "oral":      "Oral Symptoms (mouth sores, dryness, mucus)",
    "gi":        "GI Symptoms (nausea, vomiting, constipation)",
    "fatigue":   "Fatigue & Sleep",
    "activity":  "Daily Activity & Independence",
    "mood":      "Emotional Health & Support",
    "other":     "Other Symptoms (breathing, skin, hearing, etc.)",
}

# ── Red flag criteria — shared between clarification and report ───
_RED_FLAGS = (
    "- Pain severity ≥ 7/10, uncontrolled or worsening despite medication\n"
    "- Blood when coughing (hemoptysis) — any amount\n"
    "- Fever ≥ 100.4°F / 38°C or chills with possible infection signs\n"
    "- Significant unintentional weight loss (> 5 lbs since last visit)\n"
    "- Complete inability to swallow liquids or take any oral intake\n"
    "- Feeding tube complications: leakage, blockage, site infection\n"
    "- Breathing difficulty at rest or worsening shortness of breath / wheezing\n"
    "- Falls or near-falls, especially with dizziness\n"
    "- Suicidal ideation or expression of wanting to harm oneself\n"
    "- Severe depression or distress that is interfering with daily functioning\n"
    "- New neurological symptoms: sudden weakness, numbness, confusion\n"
    "- No bowel movement for > 3 days with discomfort\n"
    "- Medication non-adherence affecting symptom control"
)


def _call_openai(prompt: str, max_tokens: int = 120, temp: float = 0.4) -> str:
    if not openai_client:
        return ""
    try:
        r = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temp,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return ""


def get_llm_clarification(topic_key: str, step: dict,
                           answer: str, chat_history: list) -> Optional[str]:
    """
    After a patient gives a free-text answer, decide:
      (A) Answer is complete -> return a warm 1-sentence acknowledgment.
      (B) Answer is vague/incomplete -> return one targeted follow-up probe.
      (C) Answer contains a RED FLAG -> return a calm, flagging sentence.

    Returns a single string (acknowledgment or question), or None to skip.
    """
    if len(answer.strip().split()) < 3:
        return None

    topic_label = _TOPIC_LABELS_LLM.get(topic_key, topic_key)

    # Compact recent conversation history (last 6 turns, excluding current answer)
    recent = chat_history[-6:] if len(chat_history) > 6 else chat_history
    history_str = "\n".join(
        f"{'Nurse' if m['role'] == 'assistant' else 'Patient'}: {m['content']}"
        for m in recent
        if m["content"] != answer
    )

    prompt = (
        f"{_SYSTEM_CONTEXT}\n\n"
        f"=== CURRENT TOPIC: {topic_label} ===\n\n"
        f"=== RECENT CONVERSATION ===\n"
        f"{history_str}\n\n"
        f"Nurse asked: \"{step['text']}\"\n"
        f"Patient said: \"{answer}\"\n\n"
        f"=== RED FLAG CRITERIA TO WATCH FOR ===\n"
        f"{_RED_FLAGS}\n\n"
        f"=== YOUR TASK ===\n"
        f"Decide which ONE of these applies:\n\n"
        f"A) The patient's answer is COMPLETE and clinically sufficient for this question.\n"
        f"   -> Write one warm, specific sentence acknowledging what they shared.\n"
        f"   Do NOT ask another question. Do NOT use hollow openers like 'I see' or "
        f"'Thank you for sharing'. Reference something specific they said.\n\n"
        f"B) The answer is INCOMPLETE or vague -- a single targeted follow-up would capture "
        f"important clinical detail not yet provided.\n"
        f"   -> Ask exactly ONE short, plain-language follow-up question.\n"
        f"   It must be directly relevant to what they said and stay within this topic.\n"
        f"   Do not introduce a new subject.\n\n"
        f"C) The answer contains or strongly implies a RED FLAG listed above.\n"
        f"   -> Write a calm, caring sentence that gently acknowledges the concern and lets "
        f"them know their care team will be notified. Example: 'That's something your team "
        f"will want to know about before your visit -- I've made sure it's flagged in your "
        f"report so they can follow up with you.'\n\n"
        f"Return ONLY your chosen response -- one sentence or one question. "
        f"No labels, no preamble, no explanation of which option you chose."
    )

    return _call_openai(prompt, max_tokens=120, temp=0.3) or None


def generate_report(name: str, all_data: dict) -> str:
    """
    Generate a structured clinical pre-visit report from all collected topic data.
    Falls back to a plain-text report if OpenAI is unavailable.
    """
    topic_summaries = {}
    for label, key in TOPICS:
        d = all_data.get(key, {})
        if d:
            topic_summaries[label] = d

    if not openai_client:
        # Plain-text fallback
        lines = [
            "CHATREPORT -- PRE-VISIT CLINICAL SUMMARY",
            f"Patient: {name}",
            f"Date: {datetime.now().strftime('%B %d, %Y')}",
            "=" * 56, "",
        ]
        for label, data in topic_summaries.items():
            lines.append(f"[ {label.upper()} ]")
            for k, v in data.items():
                val = ", ".join(v) if isinstance(v, list) else str(v)
                lines.append(f"  - {k.replace('_', ' ').title()}: {val}")
            lines.append("")
        return "\n".join(lines)

    data_json = json.dumps(topic_summaries, indent=2)
    today = datetime.now().strftime("%B %d, %Y")

    prompt = (
        f"{_SYSTEM_CONTEXT}\n\n"
        f"You are now generating a structured pre-visit clinical summary for a provider "
        f"(oncologist, radiation oncologist, or NP). This report will be reviewed BEFORE "
        f"the patient's appointment and must be concise, clinically precise, and "
        f"provider-ready. Providers will skim this in under 2 minutes.\n\n"
        f"=== PATIENT: {name} ===\n"
        f"=== DATE: {today} ===\n\n"
        f"=== PATIENT-REPORTED DATA ===\n"
        f"{data_json}\n\n"
        f"=== RED FLAGS TO SCREEN FOR ===\n"
        f"{_RED_FLAGS}\n\n"
        f"=== REPORT FORMAT INSTRUCTIONS ===\n"
        f"Use the EXACT structure below. Use bullet points within each section.\n"
        f"Convert patient-language answers into accurate clinical language where appropriate "
        f"(e.g., 'sore in my mouth' -> 'oral mucositis', 'can't swallow' -> 'dysphagia').\n"
        f"Include the patient's own words in quotes only when clinically meaningful.\n"
        f"If a topic was not completed or has no data, omit it entirely -- do not write N/A.\n\n"
        f"---\n"
        f"CHATREPORT -- PRE-VISIT CLINICAL SUMMARY\n"
        f"Patient: {name}  |  Date: {today}\n"
        f"{'=' * 56}\n\n"
        f"CLINICAL OVERVIEW\n"
        f"[2-3 sentence high-level summary of the patient's current status, most prominent "
        f"issues, and any notable changes. Written for a clinician who has 10 seconds to "
        f"orient before walking into the room.]\n\n"
        f"FLAGS FOR PROVIDER ATTENTION\n"
        f"[List ONLY items matching the red flag criteria above, each as a brief bullet. "
        f"If none, write: No urgent flags identified.]\n\n"
        f"SYMPTOM DETAILS BY DOMAIN\n"
        f"[One subsection per completed topic. Use the topic name as a bold header. "
        f"Bullets should include: symptom presence/severity, patient-reported management "
        f"strategies, medications mentioned, and functional impact where reported.]\n\n"
        f"SUGGESTED DISCUSSION POINTS\n"
        f"[2-4 bullets: items the provider may want to address or follow up on based on "
        f"the data -- e.g., medication adjustment, referral, patient education need, "
        f"unresolved concern. Do not repeat red flags already listed above.]\n"
        f"---\n\n"
        f"Write only the completed report. Do not include these instructions in your output. "
        f"Do not add AI disclaimers or notes about report generation."
    )

    return _call_openai(prompt, max_tokens=2000, temp=0.2) or \
        "Report generation failed -- please check your OpenAI API configuration."


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "app_stage":           "login",
        "patient_name":        "",
        "selected_topic":      None,
        "topic_states": {
            key: {"status": "not_started", "data": {}, "chat": []}
            for _, key in TOPICS
        },
        "report":              "",
        "report_saved":        False,
        # ── New: previous check-in data ──
        "last_checkin":        {},    # {topic_key: {q_id: answer}} from last session
        "has_prev_checkin":    False, # True if a previous session was found in Sheets
        # ── New: free-form chat ──
        "freeform_chat":       [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ══════════════════════════════════════════════════════════════════
# LOAD PREVIOUS CHECK-IN
# ══════════════════════════════════════════════════════════════════

def load_last_checkin(name: str) -> dict:
    """
    Fetch the most recent saved check-in for this patient from Google Sheets.
    Returns a dict keyed by topic_key -> {q_id: answer}, or {} if none found.
    """
    _init_sheets()
    if _sheet is None:
        return {}
    try:
        rows = _sheet.get_all_values()
        last_row = None
        for row in rows[1:]:
            if len(row) >= 3 and row[1].strip().lower() == name.strip().lower():
                last_row = row          # keep iterating — last match wins
        if last_row:
            raw = json.loads(last_row[2])
            # raw is {topic_key: {q_id: answer}}
            return raw
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════════════════════
# TOPIC SUMMARY FORMATTER  (rule-based, no LLM)
# ══════════════════════════════════════════════════════════════════

# Key fields to surface per topic — (field_id, short_label)
_SUMMARY_FIELDS = {
    "pain": [
        ("has_pain",            "Pain today"),
        ("pain_location",       "Location"),
        ("throat_severity",     "Throat severity"),
        ("tongue_severity",     "Tongue severity"),
        ("pain_medications",    "Medications"),
        ("taking_as_prescribed","Adherence"),
    ],
    "nutrition": [
        ("eating_ability",        "Eating"),
        ("weight",                "Weight (lbs)"),
        ("swallowing_difficulty", "Swallowing"),
        ("feeding_tube",          "Feeding tube"),
        ("iv_fluids",             "IV fluids"),
        ("taste_changes",         "Taste changes"),
    ],
    "oral": [
        ("mouth_sores",    "Mouth sores"),
        ("dry_mouth",      "Dry mouth"),
        ("mucus_issues",   "Mucus"),
        ("oral_rinse_use", "Oral rinse"),
    ],
    "gi": [
        ("nausea_vomiting", "Nausea/vomiting"),
        ("constipation",    "Constipation"),
    ],
    "fatigue": [
        ("fatigue",           "Fatigue"),
        ("sleep_quality",     "Sleep"),
        ("medication_drowsy", "Medication drowsiness"),
    ],
    "activity": [
        ("activity_level",           "Activity level"),
        ("activity_limiting_factor", "Limiting factor"),
    ],
    "mood": [
        ("feeling_down",      "Feeling down"),
        ("support_adequate",  "Support"),
        ("anxiety_impact",    "Anxiety impact"),
    ],
    "other": [
        ("breathing_issues",  "Breathing"),
        ("hearing_changes",   "Hearing"),
        ("dizziness",         "Dizziness"),
        ("skin_issues",       "Skin"),
        ("voice_hoarseness",  "Voice"),
        ("fever_chills",      "Fever/chills"),
    ],
}


def _checkin_summary_html(topic_key: str, data: dict) -> str:
    """
    Build a chip-grid HTML block showing key facts from the previous check-in.
    Each field becomes a small pill: Label on top, value below.
    Returns an HTML string, or "" if no data to show.
    """
    fields = _SUMMARY_FIELDS.get(topic_key, [])
    chips  = []

    for field_id, label in fields:
        val = data.get(field_id)
        if val is None:
            continue
        if isinstance(val, list):
            val_str = ", ".join(str(v) for v in val)
        else:
            val_str = str(val)
        val_str = val_str.strip()
        if not val_str:
            continue
        if len(val_str) > 35:
            val_str = val_str[:32] + "…"

        chips.append(
            f'<div style="display:inline-flex;flex-direction:column;'
            f'background:#f4f8ff;border:1px solid #d0e0f8;'
            f'border-radius:10px;padding:5px 13px 6px 13px;'
            f'min-width:70px;max-width:200px;">'            f'<span style="font-size:10px;color:#8fa8c8;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.4px;'
            f'margin-bottom:2px;">{_html.escape(label)}</span>'            f'<span style="font-size:13px;color:#1e3a5f;font-weight:600;'
            f'line-height:1.3;">{_html.escape(val_str)}</span>'            f'</div>'
        )

    if not chips:
        return ""

    return (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;padding:4px 0 6px 0;">'        + "".join(chips)        + '</div>'
    )


# ── Sidebar summary: natural-language sentence per topic ─────────

def _natural_summary(topic_key: str, data: dict) -> str:
    """Return a short natural-language sentence summarising a topic's previous answers."""

    def v(field):
        val = data.get(field)
        if isinstance(val, list):
            return val if val else None
        return val if val is not None else None

    def yn(field):
        return v(field) == "Yes"

    if topic_key == "pain":
        if v("has_pain") == "No":
            return "No pain"
        loc  = (v("pain_location") or "").lower()
        sev  = v("throat_severity") or v("tongue_severity")
        meds = v("pain_medications")
        med_str = ""
        if isinstance(meds, list) and "No pain medication" not in meds:
            med_str = f", on {meds[0]}" if len(meds) == 1 else f", on {meds[0]} + {len(meds)-1} more"
        if loc and sev is not None:
            return f"{loc.capitalize()} pain ({sev}/10){med_str}"
        elif loc:
            return f"{loc.capitalize()} pain{med_str}"
        return f"Pain reported{med_str}"

    elif topic_key == "nutrition":
        eating = v("eating_ability") or ""
        weight = v("weight")
        w_str  = f", {weight} lbs" if weight else ""
        if "normally" in eating:
            return f"Eating normally{w_str}"
        elif "less" in eating:
            return f"Eating less than usual{w_str}"
        elif "Struggling" in eating:
            return f"Struggling, liquids only{w_str}"
        elif "tube" in eating.lower():
            return f"On feeding tube{w_str}"
        return f"Nutrition assessed{w_str}"

    elif topic_key == "oral":
        syms = []
        if yn("mouth_sores"):      syms.append("mouth sores")
        if yn("dry_mouth"):        syms.append("dry mouth")
        if yn("mucus_issues"):     syms.append("thick mucus")
        if yn("teeth_gum_issues"): syms.append("gum problems")
        return ", ".join(syms).capitalize() if syms else "No oral symptoms"

    elif topic_key == "gi":
        syms = []
        nv = v("nausea_vomiting") or []
        if "Nausea" in nv:              syms.append("nausea")
        if "Vomiting" in nv:            syms.append("vomiting")
        if "Blood when coughing" in nv: syms.append("blood when coughing")
        if yn("constipation"):          syms.append("constipation")
        return ", ".join(syms).capitalize() if syms else "No GI symptoms"

    elif topic_key == "fatigue":
        fatigue = v("fatigue")
        sleep   = v("sleep_quality")
        if fatigue == "No" and sleep == "Yes":
            return "No fatigue, sleeping well"
        elif fatigue == "Yes" and sleep == "No":
            return "Fatigued, trouble sleeping"
        elif fatigue == "Yes":
            return "Feeling fatigued"
        elif sleep == "No":
            return "Trouble sleeping"
        return "Fatigue assessed"

    elif topic_key == "activity":
        level = v("activity_level") or ""
        if "normally" in level:
            return "Fully active"
        elif "less" in level:
            cause = (v("activity_limiting_factor") or "").lower()
            return f"Less active — {cause}" if cause else "Less active than usual"
        elif "Struggling" in level:
            return "Struggling with daily tasks"
        return "Activity assessed"

    elif topic_key == "mood":
        parts = []
        if v("feeling_down") == "Yes":       parts.append("feeling depressed")
        if v("support_adequate") == "No":    parts.append("limited support")
        elif v("support_adequate") == "Yes": parts.append("good support")
        if v("anxiety_impact") == "Yes":     parts.append("anxiety affecting daily life")
        return ", ".join(parts[:2]).capitalize() if parts else "Mood assessed"

    elif topic_key == "other":
        syms = []
        if yn("breathing_issues"): syms.append("breathing difficulty")
        if yn("hearing_changes"):  syms.append("hearing changes")
        if yn("dizziness"):        syms.append("dizziness")
        if yn("fever_chills"):     syms.append("fever/chills")
        if yn("skin_issues"):      syms.append("skin issues")
        if v("voice_hoarseness") == "Yes, problems with my voice":
            syms.append("voice changes")
        return ", ".join(syms[:3]).capitalize() if syms else "No other symptoms"

    return ""


# ══════════════════════════════════════════════════════════════════
# FREE-FORM CHAT LLM
# ══════════════════════════════════════════════════════════════════

def _freeform_llm_response(messages: list) -> str:
    """
    Generate a nurse reply in the free-form "Anything else?" chat.
    messages is the full conversation history: [{role, content}, ...].
    """
    if not openai_client:
        return "I'm not able to respond right now — please let your care team know directly."

    system = (
        f"{_SYSTEM_CONTEXT}\n\n"
        "You are now in an open conversation with the patient. They may raise anything not "
        "covered by the structured check-in — a new symptom, a question about their treatment, "
        "a concern about a medication, or just something they want their provider to know.\n\n"
        "Guidelines:\n"
        "- Listen carefully and respond with warmth and clinical awareness.\n"
        "- If they mention a symptom that sounds urgent (e.g., chest pain, breathing difficulty, "
        "  high fever, blood, suicidal thoughts), acknowledge it calmly and tell them it will be "
        "  flagged for their care team.\n"
        "- Do NOT diagnose or prescribe. You are gathering information, not treating.\n"
        "- Keep responses short — 2-4 sentences maximum.\n"
        "- If the patient seems to be done, gently close: 'Is there anything else you'd like "
        "  to share with your team before your visit?'"
    )

    api_messages = [{"role": "system", "content": system}]
    for m in messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    try:
        r = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=api_messages,
            max_tokens=200,
            temperature=0.5,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return "I'm having trouble responding right now — please share this with your care team directly."


# ══════════════════════════════════════════════════════════════════
# 🔥 MODIFY: handle_answer (inject adaptive questioning)
# ══════════════════════════════════════════════════════════════════

def handle_answer(topic_key: str, step: dict, answer, llm_ack: Optional[str] = None):
    state = st.session_state.topic_states[topic_key]

    if isinstance(answer, list):
        display = ", ".join(answer) if answer else "(none)"
    else:
        display = str(answer)

    state["chat"].append({"role": "assistant", "content": step["text"]})
    state["chat"].append({"role": "user", "content": display})

    # ✅ NEW: LLM adaptive follow-up
    adaptive_q = None
    if isinstance(answer, str) and len(answer.split()) > 2:
        with st.spinner("Thinking…"):
            adaptive_q = get_llm_adaptive_question(
                topic_key, step, answer, state["data"]
            )

    if adaptive_q:
        state["chat"].append({
            "role": "assistant",
            "content": adaptive_q
        })

        # store temporary follow-up step
        state["pending_llm_question"] = adaptive_q
        state["pending_llm_key"] = f"llm_{len(state['data'])}"

        st.rerun()

    # fallback to original behavior
    if llm_ack:
        state["chat"].append({"role": "assistant", "content": llm_ack})
        # 🔥 STOP here — do NOT move to next question yet
        state["waiting_for_followup"] = True
        return

    state["data"][step["id"]] = answer
    state["status"] = "in_progress"

    if topic_is_complete(topic_key, state["data"]):
        state["status"] = "completed"
        state["chat"].append({
            "role": "assistant",
            "content": "✅ Thank you — I have everything I need for this topic."
        })

    st.rerun()


# ══════════════════════════════════════════════════════════════════
# INPUT RENDERING  (accepts prev_answer for pre-filling)
# ══════════════════════════════════════════════════════════════════

def _fmt_prev(val) -> str:
    """Format a previous answer value for display."""
    if val is None:
        return ""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val)


def render_input(topic_key: str, step: dict, prev_answer=None):
    """Render the appropriate input widget, pre-filled with prev_answer where possible."""
    stype = step["type"]
    sid   = step["id"]

    # ── Previous-answer hint (shown for all types) ───────────────
    if prev_answer is not None:
        hint = _fmt_prev(prev_answer)
        if hint:
            if len(hint) > 80:
                hint = hint[:77] + "…"
            st.markdown(
                f'<div style="background:#f0f7ff; border-left:3px solid #93c5fd; '                f'padding:6px 10px; border-radius:6px; font-size:13px; '                f'color:#1e40af; margin-bottom:8px;">'                f'📋 <strong>Last time:</strong> {hint}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("ℹ️ No previous answer for this question.")

    # ── Options ─────────────────────────────────────────────────
    if stype == "options":

        st.markdown("**Choose an option OR type your answer:**")
    
        # ── OPTION BUTTONS ──
        cols = st.columns(2 if len(step["opts"]) <= 4 else 1)
    
        for i, opt in enumerate(step["opts"]):
            with cols[i % len(cols)]:
                if st.button(opt, key=f"opt_{topic_key}_{sid}_{i}"):
                    handle_answer(topic_key, step, opt)
    
        # ── FREE TEXT INPUT ──
        user_text = st.text_input(
            "Or type your answer:",
            key=f"opt_text_{topic_key}_{sid}"
        )
    
        col1, col2 = st.columns([2,1])
    
        # ── TEXT SUBMIT ──
        with col1:
            if st.button("Submit ✎", key=f"opt_submit_{topic_key}_{sid}"):
    
                if user_text.strip():
    
                    # 🔥 LLM interprets text
                    interpreted = interpret_user_input_with_options(step, user_text)
    
                    # optional clarification (your existing logic)
                    ack = None
                    if openai_client:
                        with st.spinner("Understanding your response…"):
                            ack = get_llm_clarification(
                                topic_key,
                                step,
                                user_text,
                                st.session_state.topic_states[topic_key]["chat"]
                            )
    
                    handle_answer(topic_key, step, interpreted, llm_ack=ack)
    
                else:
                    st.warning("Please type something or select an option.")
    
        # ── VOICE INPUT ──
        with col2:
            voice_text = voice_widget(f"{topic_key}_{sid}_opt")
    
            if voice_text:
                interpreted = interpret_user_input_with_options(step, voice_text)
                handle_answer(topic_key, step, interpreted)
                

    # ── Multi-select ─────────────────────────────────────────────
    elif stype == "multi_select":
        # Pre-select previous choices that still exist in options
        safe_prev = (
            [v for v in prev_answer if v in step["opts"]]
            if isinstance(prev_answer, list) else []
        )
        chosen = st.multiselect(
            "Select all that apply:",
            step["opts"],
            default=safe_prev,
            key=f"ms_{topic_key}_{sid}",
        )
        if st.button("Confirm ✓", key=f"ms_submit_{topic_key}_{sid}"):
            if chosen:
                handle_answer(topic_key, step, chosen)
            else:
                st.warning("Please select at least one option, or choose 'None of these'.")

    # ── Number ───────────────────────────────────────────────────
    elif stype == "number":
        # Use previous answer as default if valid
        if prev_answer is not None:
            try:
                num_default = float(prev_answer)
                num_default = max(float(step["min_v"]), min(float(step["max_v"]), num_default))
            except (TypeError, ValueError):
                num_default = float(step["default_v"])
        else:
            num_default = float(step["default_v"])

        val = st.number_input(
            "Enter value:",
            min_value=float(step["min_v"]),
            max_value=float(step["max_v"]),
            value=num_default,
            step=1.0,
            key=f"num_{topic_key}_{sid}",
        )
        if st.button("Submit ✓", key=f"num_submit_{topic_key}_{sid}"):
            handle_answer(topic_key, step, int(val))

    # ── Free text ────────────────────────────────────────────────
    elif stype == "free_text":
        transcript_key = f"_vt_{topic_key}_{sid}"
        widget_key     = f"ft_{topic_key}_{sid}"

        # Priority for pre-fill: voice transcript > previous answer > empty
        if widget_key not in st.session_state:
            transcript  = st.session_state.get(transcript_key, "")
            prev_str    = str(prev_answer) if prev_answer is not None else ""
            st.session_state[widget_key] = transcript or prev_str

        st.text_area(
            "Your response:",
            placeholder=step.get("placeholder", "Please describe…"),
            key=widget_key,
            height=100,
        )

        col_submit, col_voice = st.columns([2, 1])
        with col_submit:
            if st.button("Submit ✓", key=f"ft_submit_{topic_key}_{sid}"):
                text = st.session_state.get(widget_key, "").strip()
                if text:
                    ack = None
                    if openai_client:
                        with st.spinner("Processing…"):
                            ack = get_llm_clarification(
                                topic_key, step, text,
                                st.session_state.topic_states[topic_key]["chat"],
                            )
                    handle_answer(topic_key, step, text, llm_ack=ack)
                else:
                    st.warning("Please enter a response before submitting.")
        with col_voice:
            voice_widget(f"{topic_key}_{sid}")


# ══════════════════════════════════════════════════════════════════
# FREE-FORM CHAT PANEL
# ══════════════════════════════════════════════════════════════════

def render_freeform_chat():
    """Render the open-ended 'Anything else?' chatbot panel."""
    _stc.html("""<script>
    (function(){
        var s=['section[data-testid=\"stMain\"]',
               'div[data-testid=\"stAppViewContainer\"]','.main'];
        for(var i=0;i<s.length;i++){
            var e=window.parent.document.querySelector(s[i]);
            if(e){e.scrollTop=0;break;}
        }
    })();
    </script>""", height=0)
    render_assistant_chip()
    st.subheader("💬 Anything else you’d like to share?")
    st.caption(
        "Mention any other symptoms, questions, or concerns you’d like your care team "
        "to know about before your visit."
    )

    # ── Initialise conversation ──────────────────────────────────
    if not st.session_state.freeform_chat:
        opening = (
            "Is there anything else you'd like your care team to know before your visit? "
            "Feel free to share any concerns, questions, or symptoms we haven't covered yet."
        )
        st.session_state.freeform_chat = [{"role": "assistant", "content": opening}]

    # ── Show history ─────────────────────────────────────────────
    chat_container = st.container(border=False)
    with chat_container:
        for msg in st.session_state.freeform_chat:
            avatar = "👩‍⚕️" if msg["role"] == "assistant" else "🧑‍💼"
            with st.chat_message(msg["role"], avatar=avatar):
                st.write(msg["content"])

    # ── Input ────────────────────────────────────────────────────
    user_input = st.chat_input("Type here, or use the voice button below…",
                                key="freeform_chat_input")

    col_text, col_voice = st.columns([3, 1])
    with col_voice:
        vt = voice_widget("freeform")
        if vt and not user_input:
            user_input = vt
            # Clear the transcript so it doesn't re-fire
            st.session_state.pop("_vt_freeform", None)

    if user_input and user_input.strip():
        # Avoid re-appending if already in history (Streamlit reruns)
        last_user = next(
            (m["content"] for m in reversed(st.session_state.freeform_chat)
             if m["role"] == "user"), None
        )
        if user_input.strip() != last_user:
            st.session_state.freeform_chat.append(
                {"role": "user", "content": user_input.strip()}
            )
            if openai_client:
                with st.spinner("…"):
                    reply = _freeform_llm_response(st.session_state.freeform_chat)
            else:
                reply = (
                    "Got it — I've noted that for your care team. "
                    "Is there anything else you'd like to add?"
                )
            st.session_state.freeform_chat.append(
                {"role": "assistant", "content": reply}
            )
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# TOPIC DETAIL PANEL
# ══════════════════════════════════════════════════════════════════

def render_topic_detail(topic_label: str, topic_key: str):
    """Render the chat + current question for the selected topic."""
    _stc.html("""<script>
    (function(){
        var s=['section[data-testid=\"stMain\"]',
               'div[data-testid=\"stAppViewContainer\"]','.main'];
        for(var i=0;i<s.length;i++){
            var e=window.parent.document.querySelector(s[i]);
            if(e){e.scrollTop=0;break;}
        }
    })();
    </script>""", height=0)
    state        = st.session_state.topic_states[topic_key]
    last_data    = st.session_state.last_checkin.get(topic_key, {})
    has_prev     = st.session_state.has_prev_checkin

    render_assistant_chip()

    # ── Previous check-in summary card ────────────────────────────
    if has_prev:
        if last_data:
            chips_html = _checkin_summary_html(topic_key, last_data)
            if chips_html:
                render_memory_banner("From your last check-in, here are the main details we already have on file.")
                st.markdown(
                    '<div class="soft-card" style="margin-bottom:12px;">'
                    '<div style="font-size:11px;font-weight:700;color:#7a9ec5;'
                    'text-transform:uppercase;letter-spacing:0.6px;margin-bottom:8px;">'
                    'Your last check-in</div>' + chips_html + '</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="subtle-note">📋 No information from your last visit was recorded for this section.</div>',
                unsafe_allow_html=True,
            )

    # ── Initialize topic on first visit ─────────────────────────
    if state["status"] == "not_started":
        state["status"] = "in_progress"
        intro = TOPIC_INTROS.get(topic_key, "Let's go through this section together.")
        state["chat"] = [{"role": "assistant", "content": intro}]

    # ── Header with progress bar ─────────────────────────────────
    answered, applicable = get_topic_progress(topic_key, state["data"])
    col_title, col_prog = st.columns([3, 1])
    with col_title:
        st.subheader(topic_label)
    with col_prog:
        if applicable > 0:
            st.progress(answered / applicable, text=f"{answered}/{applicable} answered")

    # ── Chat history ─────────────────────────────────────────────
    if state["chat"]:
        with st.container(border=False):
            for msg in state["chat"]:
                avatar = "👩‍⚕️" if msg["role"] == "assistant" else "🧑‍💼"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.write(msg["content"])

    # ── Completed ────────────────────────────────────────────────
    if state["status"] == "completed":
        st.markdown(
            '<div class="completion-badge">✅ This topic is complete</div>',
            unsafe_allow_html=True,
        )
        if st.button("✏️ Add a note or correction", key=f"reopen_{topic_key}"):
            state["status"] = "in_progress"
            state["chat"].append({
                "role": "assistant",
                "content": "Of course — please share any correction or additional detail.",
            })
            state["data"].pop("_correction_note", None)
            st.rerun()
        return

    # ── Current question ─────────────────────────────────────────
    # 🔥 If waiting for LLM follow-up, do not ask next question
    if state.get("waiting_for_followup"):
        user_input = st.chat_input("Your response...")
    
        if user_input:
            state["chat"].append({"role": "user", "content": user_input})
            state["waiting_for_followup"] = False
            st.rerun()
    
        return
    next_step = get_next_step(topic_key, state["data"])
    if next_step:
        # Look up previous answer for this specific question
        prev_answer = last_data.get(next_step["id"]) if last_data else None

        with st.container(border=False):
            with st.chat_message("assistant", avatar="👩‍⚕️"):
                st.write(next_step["text"])
            render_input(topic_key, next_step, prev_answer=prev_answer)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR  (MASTER PANEL)
# ══════════════════════════════════════════════════════════════════



def render_sidebar():
    with st.sidebar:
        # ── Header ───────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:18px;font-weight:800;color:#10233d;margin:0 0 2px 0;letter-spacing:-0.03em;">🩺 ChatReport</div>'
            '<div style="font-size:11px;color:#6b7b92;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.08em;">Clinical symptom assistant</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.patient_name:
            st.markdown(
                f'<div style="font-size:11.5px;color:#6b7280;margin-bottom:8px;">'                f'Patient: <strong>{_html.escape(st.session_state.patient_name)}</strong></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<hr style="margin:6px 0 10px 0;border:none;border-top:1px solid #dde6f5;">',
            unsafe_allow_html=True,
        )

        # ── Overall progress ─────────────────────────────────────
        completed   = sum(1 for _, k in TOPICS
                          if st.session_state.topic_states[k]["status"] == "completed")
        in_progress = sum(1 for _, k in TOPICS
                          if st.session_state.topic_states[k]["status"] == "in_progress")
        total = len(TOPICS)

        st.markdown(
            f'<div class="prog-label">{completed}/{total} topics complete</div>',
            unsafe_allow_html=True,
        )
        st.progress(completed / total if total > 0 else 0)
        st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

        # ── Topic nav buttons ─────────────────────────────────────
        # Using st.button (not HTML anchors) so session state is preserved.
        # The label has two parts separated by \n:
        #   line 1: "{selected_marker}{icon} {topic_name}"
        #   line 2: summary snippet or "No prior data" (when prev check-in exists)
        # CSS white-space:pre-wrap renders the \n as a real line break.

        has_prev = st.session_state.get("has_prev_checkin", False)
        last_ck  = st.session_state.get("last_checkin", {})

        for label, key in TOPICS:
            status = st.session_state.topic_states[key]["status"]
            icon   = {"completed": "✅", "in_progress": "🔵"}.get(status, "⚪")
            dname  = label.split(" ", 1)[1] if " " in label else label
            marker = "▶ " if st.session_state.selected_topic == key else "   "

            # Build button label
            btn = f"{marker}{icon} {dname}"
            if has_prev:
                prev_data = last_ck.get(key, {})
                if prev_data:
                    snip = _natural_summary(key, prev_data)
                    btn += f"\n   {snip}" if snip else "\n   No data recorded"
                else:
                    btn += "\n   No prior data"

            if st.button(btn, key=f"nav_{key}", use_container_width=True):
                st.session_state.selected_topic = key
                st.rerun()

        # ── Anything else? ────────────────────────────────────────
        ff_msgs  = [m for m in st.session_state.freeform_chat if m["role"] == "user"]
        ff_badge = f" ({len(ff_msgs)})" if ff_msgs else ""
        ff_mark  = "▶ " if st.session_state.selected_topic == "freeform" else "   "
        if st.button(f"{ff_mark}💬 Anything else?{ff_badge}",
                     key="nav_freeform", use_container_width=True):
            st.session_state.selected_topic = "freeform"
            st.rerun()

        # ── Submit ────────────────────────────────────────────────
        st.markdown(
            '<hr style="margin:8px 0 8px 0;border:none;border-top:1px solid #dde6f5;">',
            unsafe_allow_html=True,
        )
        any_started = completed >= 1 or in_progress >= 1
        if any_started:
            if st.button("📤 Submit Check-In", use_container_width=True,
                         type="primary", key="sidebar_submit"):
                all_data = {k: st.session_state.topic_states[k]["data"]
                            for _, k in TOPICS}
                if ff_msgs:
                    all_data["freeform_notes"] = [
                        m["content"] for m in st.session_state.freeform_chat
                        if m["role"] == "user"
                    ]
                with st.spinner("Generating report…"):
                    report = generate_report(st.session_state.patient_name, all_data)
                st.session_state.report = report
                with st.spinner("Saving…"):
                    save_to_sheet(st.session_state.patient_name, all_data, report)
                st.session_state.report_saved = True
                st.session_state.app_stage = "report"
                st.rerun()

# ══════════════════════════════════════════════════════════════════
# SCREENS
# ══════════════════════════════════════════════════════════════════

TOPIC_LABELS = {key: label for label, key in TOPICS}
TOPIC_KEYS   = [k for _, k in TOPICS]


def screen_login():
    st.markdown("""
    <div class="welcome-card">
        <div style="font-size:12px;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;color:#6b7b92;margin-bottom:8px;">
            Pre-visit symptom check-in
        </div>
        <h1 style="margin:0 0 10px 0; color:#10233d; font-size:34px; letter-spacing:-0.04em;">🩺 ChatReport</h1>
        <p style="color:#56667d; margin-bottom:14px; font-size:15px; line-height:1.75;">
        A modern clinical check-in for patients receiving head and neck cancer treatment.
        Your answers help your care team review symptoms before the visit.
        </p>
        <div class="subtle-note">
            You can answer by typing or by voice. You may switch topics at any time, and your most recent prior check-in will be used to guide the conversation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        name = st.text_input("Please enter your name:", placeholder="First and last name…")
        if st.button("Begin Check-In →", type="primary", use_container_width=True):
            if name.strip():
                st.session_state.patient_name = name.strip()

                # ── Load previous check-in from Sheets ──────────
                with st.spinner("Loading your previous check-in…"):
                    prev = load_last_checkin(name.strip())

                if prev:
                    st.session_state.last_checkin     = prev
                    st.session_state.has_prev_checkin = True
                else:
                    st.session_state.last_checkin     = {}
                    st.session_state.has_prev_checkin = False

                st.session_state.selected_topic = TOPIC_KEYS[0] if TOPIC_KEYS else None
                st.session_state.app_stage      = "main"
                st.rerun()
            else:
                st.warning("Please enter your name to continue.")


def screen_main():
    render_sidebar()

    selected = st.session_state.selected_topic

    if not selected:
        st.markdown('<div class="card"><div style="font-size:12px;font-weight:800;color:#6b7b92;text-transform:uppercase;letter-spacing:0.08em;">Getting started</div><div style="font-size:28px;font-weight:800;letter-spacing:-0.03em;margin-top:6px;">Choose a symptom area from the sidebar</div><div style="font-size:14px;color:#5f6f84;line-height:1.7;margin-top:8px;">Move through the check-in in any order you prefer. Your answers are saved into a provider-ready summary for the care team.</div></div>', unsafe_allow_html=True)
        return

    # Route to free-form chat or regular topic
    if selected == "freeform":
        render_freeform_chat()
    else:
        topic_label = TOPIC_LABELS.get(selected, selected)
        render_topic_detail(topic_label, selected)


def screen_report():
    render_sidebar()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📄 Clinical Check-In Report")
    st.markdown(
        f"**Patient:** {st.session_state.patient_name} &nbsp;|&nbsp; "
        f"**Date:** {datetime.now().strftime('%B %d, %Y')}"
    )
    st.markdown(
        '<div style="font-size:13px;color:#627287;line-height:1.7;">'
        'This report is formatted for quick clinical review before the appointment.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    all_data = {key: st.session_state.topic_states[key]["data"] for _, key in TOPICS}
    ff_msgs  = [m for m in st.session_state.freeform_chat if m["role"] == "user"]
    if ff_msgs:
        all_data["freeform_notes"] = [m["content"] for m in ff_msgs]

    if not st.session_state.report:
        with st.spinner("Generating clinical report…"):
            st.session_state.report = generate_report(
                st.session_state.patient_name, all_data
            )

    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown(st.session_state.report)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅️ Back to Check-In"):
            st.session_state.app_stage = "main"
            st.rerun()

    with col2:
        saved     = st.session_state.get("report_saved", False)
        btn_label = "✅ Saved" if saved else "💾 Save to Google Sheets"
        if st.button(btn_label, type="primary", disabled=saved):
            with st.spinner("Saving…"):
                _init_sheets()
                save_to_sheet(
                    st.session_state.patient_name,
                    all_data,
                    st.session_state.report,
                )
            st.session_state.report_saved = True
            st.success("Saved successfully!")
            st.rerun()

    with col3:
        if st.button("📋 Copy to Clipboard (manual)"):
            st.info("Select the report text above and copy (Ctrl+C / Cmd+C).")


# ══════════════════════════════════════════════════════════════════
# MAIN DISPATCH
# ══════════════════════════════════════════════════════════════════

_init_sheets()

stage = st.session_state.get("app_stage", "login")

if stage == "login":
    screen_login()
elif stage == "main":
    screen_main()
elif stage == "report":
    screen_report()
