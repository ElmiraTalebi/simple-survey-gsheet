"""
╔══════════════════════════════════════════════════════════════════╗
║         ChatReport - HNC Symptom Reporting Chatbot              ║
║         Fox Chase Cancer Center × Temple University             ║
╚══════════════════════════════════════════════════════════════════╝

A Streamlit-based conversational chatbot prototype for head and neck
cancer patients undergoing chemoradiation treatment.

Features:
  • Telegram-style chat UI (bot left / patient right)
  • Sequential questioning — next question only after answer
  • Interactive SVG human body diagram for pain location
  • Adaptive follow-up logic per symptom domain
  • Structured clinical report generation
  • Clinical alert flagging (pain ≥7, weight loss, mood concerns)

Run:
    pip install streamlit
    streamlit run chatreport_app.py
"""

import streamlit as st
from datetime import datetime
import json, re

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatReport | Fox Chase × Temple",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ───────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #EEF2F7 !important; }

/* ── Hide Streamlit chrome ──────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 780px !important; }

/* ── Chat shell ─────────────────────────────────── */
.chat-shell {
    background: #fff;
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(0,0,0,0.12);
    margin: 24px auto;
    display: flex; flex-direction: column;
    min-height: 90vh;
}

/* ── Header ─────────────────────────────────────── */
.chat-header {
    background: linear-gradient(135deg, #0C5460 0%, #17A2B8 100%);
    padding: 16px 22px;
    display: flex; align-items: center; gap: 14px;
    flex-shrink: 0;
}
.header-avatar {
    width: 48px; height: 48px; border-radius: 50%;
    background: rgba(255,255,255,0.2);
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; flex-shrink: 0;
    border: 2px solid rgba(255,255,255,0.4);
}
.header-text h2 { margin:0; font-size:17px; font-weight:700; color:#fff; }
.header-text p  { margin:0; font-size:12px; color:rgba(255,255,255,0.78); }
.header-badge {
    margin-left: auto;
    background: rgba(255,255,255,0.2);
    color: #fff;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.3);
}

/* ── Message area ───────────────────────────────── */
.chat-area {
    flex: 1;
    padding: 20px 18px;
    overflow-y: auto;
    background: #F0F4F8;
    display: flex;
    flex-direction: column;
    gap: 6px;
}

/* ── Rows ───────────────────────────────────────── */
.row { display:flex; align-items:flex-end; gap:8px; margin-bottom:4px; }
.row-bot  { justify-content: flex-start; }
.row-user { justify-content: flex-end; }

/* ── Avatars ─────────────────────────────────────── */
.av {
    width:34px; height:34px; border-radius:50%; flex-shrink:0;
    display:flex; align-items:center; justify-content:center; font-size:16px;
}
.av-bot  { background: linear-gradient(135deg,#0C5460,#17A2B8); color:#fff; }
.av-user { background: linear-gradient(135deg,#28A745,#20C997); color:#fff; }

/* ── Bubbles ─────────────────────────────────────── */
.bubble {
    max-width: 68%;
    padding: 11px 15px;
    border-radius: 18px;
    font-size: 14px;
    line-height: 1.55;
    word-break: break-word;
}
.bubble-bot {
    background: #fff;
    color: #1a1a2e;
    border-bottom-left-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.bubble-user {
    background: linear-gradient(135deg, #0C5460, #17A2B8);
    color: #fff;
    border-bottom-right-radius: 4px;
    box-shadow: 0 1px 4px rgba(23,162,184,0.3);
}
.ts { font-size: 10px; color: #9ca3af; margin-top: 4px; }
.ts-right { text-align: right; }

/* ── System message ─────────────────────────────── */
.sys-msg {
    text-align: center;
    font-size: 11px;
    color: #6b7280;
    background: rgba(0,0,0,0.05);
    border-radius: 10px;
    padding: 5px 14px;
    margin: 8px auto;
    width: fit-content;
}

/* ── Input row ──────────────────────────────────── */
.input-row {
    background: #fff;
    border-top: 1px solid #e5e7eb;
    padding: 12px 16px;
    display: flex; gap: 10px; align-items: center;
    flex-shrink: 0;
}

/* ── Streamlit widget overrides ─────────────────── */
.stTextInput > div > div > input {
    border-radius: 24px !important;
    border: 1.5px solid #d1d5db !important;
    padding: 10px 18px !important;
    font-size: 14px !important;
    background: #f9fafb !important;
    transition: border-color .2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #17A2B8 !important;
    box-shadow: 0 0 0 3px rgba(23,162,184,0.15) !important;
    outline: none !important;
}
div[data-testid="stTextInput"] label { display: none !important; }

.stButton > button {
    border-radius: 50% !important;
    width: 44px !important; height: 44px !important;
    padding: 0 !important;
    background: linear-gradient(135deg,#0C5460,#17A2B8) !important;
    color: #fff !important;
    border: none !important;
    font-size: 18px !important;
    line-height: 1 !important;
    box-shadow: 0 3px 10px rgba(23,162,184,0.4) !important;
    transition: transform .15s, box-shadow .15s !important;
}
.stButton > button:hover {
    transform: scale(1.08) !important;
    box-shadow: 0 5px 16px rgba(23,162,184,0.5) !important;
}

/* ── Quick-reply chips ──────────────────────────── */
.chips { display:flex; flex-wrap:wrap; gap:7px; margin-top:10px; }
.chip {
    background: #EEF2F7;
    border: 1.5px solid #CBD5E0;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 13px;
    color: #374151;
    cursor: pointer;
    transition: all .15s;
}
.chip:hover { background:#17A2B8; color:#fff; border-color:#17A2B8; }

/* ── Pain scale ─────────────────────────────────── */
.pain-scale {
    display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px;
}
.pain-btn {
    width: 38px; height: 38px;
    border-radius: 50%;
    border: none;
    font-size: 13px; font-weight: 600;
    cursor: pointer;
    color: #fff;
    display: flex; align-items:center; justify-content:center;
}

/* ── Body diagram ───────────────────────────────── */
.body-diagram-wrap {
    background: #fff;
    border: 1.5px solid #e5e7eb;
    border-radius: 14px;
    padding: 18px;
    margin-top: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.body-diagram-wrap h4 {
    margin: 0 0 12px;
    font-size: 14px;
    color: #0C5460;
    font-weight: 600;
}

/* ── Report ─────────────────────────────────────── */
.report-container {
    background: #0d1117;
    border-radius: 14px;
    padding: 24px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: #e6edf3;
    white-space: pre-wrap;
    line-height: 1.6;
    max-height: 70vh;
    overflow-y: auto;
}

/* ── Progress bar ───────────────────────────────── */
.progress-wrap {
    background: #e5e7eb;
    border-radius: 99px;
    height: 6px;
    margin: 0;
}
.progress-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #17A2B8, #20C997);
    transition: width .4s ease;
}
.progress-label {
    font-size: 11px;
    color: #6b7280;
    text-align: right;
    margin-top: 3px;
    padding-right: 18px;
}

/* ── Alert box ──────────────────────────────────── */
.alert-box {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #856404;
    margin-top: 8px;
}
.alert-red {
    background: #f8d7da;
    border-color: #f5c2c7;
    color: #842029;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# BODY DIAGRAM SVG  (clickable regions — head/neck anatomy)
# ─────────────────────────────────────────────────────────────────
BODY_SVG = """
<svg viewBox="0 0 220 420" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:220px;display:block;margin:0 auto;cursor:pointer">
  <defs>
    <style>
      .region { fill: #EEF2F7; stroke: #CBD5E0; stroke-width:1.5;
                transition: fill .2s; cursor: pointer; }
      .region:hover { fill: #BAE6FD; stroke: #0EA5E9; }
      .region.selected { fill: #17A2B8; stroke: #0C5460; }
      .label { font-family:Inter,sans-serif; font-size:9px;
               fill:#374151; text-anchor:middle; pointer-events:none; }
    </style>
  </defs>

  <!-- HEAD -->
  <ellipse class="region" id="head" cx="110" cy="52" rx="38" ry="44"/>
  <text class="label" x="110" y="55">Head</text>

  <!-- NECK -->
  <rect class="region" id="neck" x="90" y="94" width="40" height="34" rx="6"/>
  <text class="label" x="110" y="115">Neck</text>

  <!-- JAW (overlaid on lower head) -->
  <ellipse class="region" id="jaw" cx="110" cy="88" rx="30" ry="12" opacity=".85"/>
  <text class="label" x="110" y="91">Jaw</text>

  <!-- THROAT (front of neck) -->
  <rect class="region" id="throat" x="96" y="100" width="28" height="22" rx="5" opacity=".9"/>
  <text class="label" x="110" y="114">Throat</text>

  <!-- SHOULDERS -->
  <ellipse class="region" id="shoulder_left"  cx="60"  cy="132" rx="28" ry="16"/>
  <text class="label" x="60"  y="135">L Shoulder</text>
  <ellipse class="region" id="shoulder_right" cx="160" cy="132" rx="28" ry="16"/>
  <text class="label" x="160" y="135">R Shoulder</text>

  <!-- CHEST -->
  <rect class="region" id="chest" x="72" y="128" width="76" height="70" rx="8"/>
  <text class="label" x="110" y="167">Chest</text>

  <!-- LEFT ARM -->
  <rect class="region" id="arm_left"  x="34" y="140" width="26" height="80" rx="10"/>
  <text class="label" x="47" y="185">L Arm</text>

  <!-- RIGHT ARM -->
  <rect class="region" id="arm_right" x="160" y="140" width="26" height="80" rx="10"/>
  <text class="label" x="173" y="185">R Arm</text>

  <!-- ABDOMEN -->
  <rect class="region" id="abdomen" x="76" y="198" width="68" height="58" rx="8"/>
  <text class="label" x="110" y="231">Abdomen</text>

  <!-- PELVIS -->
  <rect class="region" id="pelvis" x="76" y="256" width="68" height="40" rx="8"/>
  <text class="label" x="110" y="280">Pelvis</text>

  <!-- LEFT LEG -->
  <rect class="region" id="leg_left"  x="72"  y="296" width="42" height="100" rx="12"/>
  <text class="label" x="93"  y="350">L Leg</text>

  <!-- RIGHT LEG -->
  <rect class="region" id="leg_right" x="106" y="296" width="42" height="100" rx="12"/>
  <text class="label" x="127" y="350">R Leg</text>

  <!-- EAR indicators (small) -->
  <ellipse class="region" id="ear_left"  cx="72"  cy="62" rx="10" ry="14"/>
  <text class="label" x="72"  y="65">L Ear</text>
  <ellipse class="region" id="ear_right" cx="148" cy="62" rx="10" ry="14"/>
  <text class="label" x="148" y="65">R Ear</text>
</svg>
"""

# Region → human-readable label mapping
REGION_LABELS = {
    "head": "Head", "neck": "Neck", "jaw": "Jaw", "throat": "Throat",
    "shoulder_left": "Left Shoulder", "shoulder_right": "Right Shoulder",
    "chest": "Chest", "arm_left": "Left Arm", "arm_right": "Right Arm",
    "abdomen": "Abdomen", "pelvis": "Pelvis",
    "leg_left": "Left Leg", "leg_right": "Right Leg",
    "ear_left": "Left Ear", "ear_right": "Right Ear",
}

# ─────────────────────────────────────────────────────────────────
# CONVERSATION STATE MACHINE
# ─────────────────────────────────────────────────────────────────
# Each "step" is an entry in FLOW:
# {
#   "id":       unique step id,
#   "msg":      bot message text,
#   "key":      session_state key to store answer,
#   "type":     "text" | "scale" | "chips" | "body" | "info",
#   "chips":    [...] for type=chips,
#   "cond":     lambda symptoms → bool (True = include step),
#   "branch":   {answer_keyword: next_step_id} (optional),
#   "next":     next step id (default sequential),
# }

def make_flow():
    """Build the ordered conversation flow."""
    return [
        # ── GREETING ────────────────────────────────────────────
        {
            "id": "welcome", "type": "info",
            "msg": (
                "👋 Hello! I'm **ChatReport**, a symptom-reporting assistant "
                "from Fox Chase Cancer Center and Temple University.\n\n"
                "I'm here to help your care team understand how you've been "
                "feeling before your upcoming appointment. This should take "
                "about **10–15 minutes**.\n\n"
                "Everything you share stays private and goes only to your "
                "medical team. You can be as brief or as detailed as you'd like."
            ),
            "next": "name",
        },
        {
            "id": "name", "type": "text",
            "msg": "To get started — what's your **first name**?",
            "key": "patient_name",
            "next": "intro2",
        },
        {
            "id": "intro2", "type": "info",
            "msg": (
                "Nice to meet you, {name}! 😊 Thank you for taking the time "
                "to do this. Let's go through some questions about how you've "
                "been feeling. There are no right or wrong answers — just share "
                "what's true for you."
            ),
            "next": "pain_q1",
        },

        # ── PAIN ─────────────────────────────────────────────────
        {
            "id": "pain_q1", "type": "chips",
            "msg": "**Pain** 😣\n\nHave you had any pain since your last appointment?",
            "key": "pain_present",
            "chips": ["Yes", "No", "A little"],
            "next": "pain_body",
        },
        {
            "id": "pain_body", "type": "body",
            "msg": "I'm sorry to hear that. 💙 Can you show me **where** you're feeling the pain? Tap the area(s) on the diagram below.",
            "key": "pain_location",
            "cond": lambda s: _yes(s.get("pain_present", "")),
            "next": "pain_severity",
        },
        {
            "id": "pain_severity", "type": "scale",
            "msg": "On a scale of **0 to 10** — where 0 is no pain and 10 is the worst pain you can imagine — how would you rate your pain **at its worst**?",
            "key": "pain_severity",
            "cond": lambda s: _yes(s.get("pain_present", "")),
            "next": "pain_frequency",
        },
        {
            "id": "pain_frequency", "type": "chips",
            "msg": "How often do you experience this pain?",
            "key": "pain_frequency",
            "chips": ["Constantly", "Most of the day", "On and off", "Only when swallowing", "Only at night"],
            "cond": lambda s: _yes(s.get("pain_present", "")),
            "next": "pain_management",
        },
        {
            "id": "pain_management", "type": "text",
            "msg": "Are you doing anything to manage the pain? For example, medication, ice packs, or anything else?",
            "key": "pain_management",
            "cond": lambda s: _yes(s.get("pain_present", "")),
            "next": "pain_impact",
        },
        {
            "id": "pain_impact", "type": "text",
            "msg": "How much is the pain affecting your day-to-day life — things like eating, sleeping, or doing activities?",
            "key": "pain_impact",
            "cond": lambda s: _yes(s.get("pain_present", "")),
            "next": "mouth_q1",
        },

        # ── MOUTH ────────────────────────────────────────────────
        {
            "id": "mouth_q1", "type": "chips",
            "msg": "**Mouth Symptoms** 👄\n\nHave you noticed any dryness, sores, or changes in your mouth?",
            "key": "mouth_present",
            "chips": ["Yes", "No", "A little"],
            "next": "mouth_dry",
        },
        {
            "id": "mouth_dry", "type": "chips",
            "msg": "Is your mouth feeling **very dry** (like you can't make enough saliva)?",
            "key": "mouth_dry",
            "chips": ["Yes, very dry", "Somewhat dry", "Not really"],
            "cond": lambda s: _yes(s.get("mouth_present", "")),
            "next": "mouth_sores",
        },
        {
            "id": "mouth_sores", "type": "chips",
            "msg": "Do you have any **sores or ulcers** inside your mouth?",
            "key": "mouth_sores",
            "chips": ["Yes", "No", "Not sure"],
            "cond": lambda s: _yes(s.get("mouth_present", "")),
            "next": "mouth_taste",
        },
        {
            "id": "mouth_taste", "type": "chips",
            "msg": "Have you noticed any **changes in how food tastes**?",
            "key": "mouth_taste",
            "chips": ["Yes, very different", "A little different", "No change"],
            "cond": lambda s: _yes(s.get("mouth_present", "")),
            "next": "mouth_impact",
        },
        {
            "id": "mouth_impact", "type": "text",
            "msg": "How much are these mouth symptoms affecting your ability to **eat or drink**?",
            "key": "mouth_impact",
            "cond": lambda s: _yes(s.get("mouth_present", "")),
            "next": "swallow_q1",
        },

        # ── SWALLOWING ───────────────────────────────────────────
        {
            "id": "swallow_q1", "type": "chips",
            "msg": "**Swallowing** 🥤\n\nHave you had any difficulty swallowing since your last visit?",
            "key": "swallow_present",
            "chips": ["Yes", "No", "A little"],
            "next": "swallow_pain",
        },
        {
            "id": "swallow_pain", "type": "chips",
            "msg": "Does swallowing cause you **pain**?",
            "key": "swallow_pain",
            "chips": ["Yes, a lot", "A little", "No"],
            "cond": lambda s: _yes(s.get("swallow_present", "")),
            "next": "swallow_diet",
        },
        {
            "id": "swallow_diet", "type": "chips",
            "msg": "What types of food are you able to eat right now?",
            "key": "swallow_diet",
            "chips": ["Regular food", "Soft foods only", "Pureed foods", "Liquids only", "Feeding tube only"],
            "cond": lambda s: _yes(s.get("swallow_present", "")),
            "next": "swallow_choking",
        },
        {
            "id": "swallow_choking", "type": "chips",
            "msg": "Have you had any **choking or coughing** episodes when eating or drinking?",
            "key": "swallow_choking",
            "chips": ["Yes, often", "Occasionally", "No"],
            "cond": lambda s: _yes(s.get("swallow_present", "")),
            "next": "nutrition_q1",
        },

        # ── NUTRITION ────────────────────────────────────────────
        {
            "id": "nutrition_q1", "type": "chips",
            "msg": "**Nutrition & Appetite** 🍽️\n\nHow is your appetite lately?",
            "key": "nutrition_appetite",
            "chips": ["Good", "Reduced", "Very poor", "No appetite at all"],
            "next": "nutrition_weight",
        },
        {
            "id": "nutrition_weight", "type": "chips",
            "msg": "Have you noticed any **weight changes** recently?",
            "key": "nutrition_weight",
            "chips": ["Yes, lost weight", "Yes, gained weight", "No change", "Not sure"],
            "next": "nutrition_weight_amt",
        },
        {
            "id": "nutrition_weight_amt", "type": "text",
            "msg": "About how much weight have you lost, and over how long?",
            "key": "nutrition_weight_amt",
            "cond": lambda s: "lost" in s.get("nutrition_weight", "").lower(),
            "next": "nutrition_nausea",
        },
        {
            "id": "nutrition_nausea", "type": "chips",
            "msg": "Have you experienced any **nausea or vomiting**?",
            "key": "nutrition_nausea",
            "chips": ["Yes, often", "Occasionally", "No"],
            "next": "nutrition_supplements",
        },
        {
            "id": "nutrition_supplements", "type": "chips",
            "msg": "Are you using any **nutritional supplements** (like Ensure, Boost, or shakes)?",
            "key": "nutrition_supplements",
            "chips": ["Yes", "No", "Sometimes"],
            "next": "breathing_q1",
        },

        # ── BREATHING ────────────────────────────────────────────
        {
            "id": "breathing_q1", "type": "chips",
            "msg": "**Breathing** 🫁\n\nHave you had any shortness of breath or difficulty breathing?",
            "key": "breathing_present",
            "chips": ["Yes", "No", "A little"],
            "next": "breathing_detail",
        },
        {
            "id": "breathing_detail", "type": "text",
            "msg": "Can you describe when it happens — for example, at rest, when walking, or at night?",
            "key": "breathing_detail",
            "cond": lambda s: _yes(s.get("breathing_present", "")),
            "next": "fatigue_q1",
        },

        # ── FATIGUE ──────────────────────────────────────────────
        {
            "id": "fatigue_q1", "type": "scale",
            "msg": "**Energy & Fatigue** 😴\n\nOn a scale of 0–10, how would you rate your **fatigue** or tiredness? (0 = not tired at all, 10 = completely exhausted)",
            "key": "fatigue_level",
            "next": "fatigue_impact",
        },
        {
            "id": "fatigue_impact", "type": "text",
            "msg": "How is the fatigue affecting your daily activities — like taking care of yourself, household tasks, or going out?",
            "key": "fatigue_impact",
            "cond": lambda s: int(s.get("fatigue_level", 0) or 0) >= 4,
            "next": "mood_q1",
        },

        # ── MOOD / EMOTIONAL WELLBEING ───────────────────────────
        {
            "id": "mood_q1", "type": "chips",
            "msg": (
                "**Emotional Wellbeing** 💙\n\n"
                "Treatment can be really tough, and it's completely normal to "
                "have a range of feelings. How would you describe your **mood** lately?"
            ),
            "key": "mood_general",
            "chips": ["Good / Positive", "A bit down", "Anxious or worried", "Quite sad", "Very distressed"],
            "next": "mood_anxiety",
        },
        {
            "id": "mood_anxiety", "type": "chips",
            "msg": "Have you been feeling **worried or anxious** about your treatment or health?",
            "key": "mood_anxiety",
            "chips": ["Yes, a lot", "Sometimes", "Not really"],
            "next": "mood_sleep",
        },
        {
            "id": "mood_sleep", "type": "chips",
            "msg": "How has your **sleep** been?",
            "key": "mood_sleep",
            "chips": ["Sleeping well", "Some trouble sleeping", "Difficulty most nights", "Can't sleep at all"],
            "next": "mood_support",
        },
        {
            "id": "mood_support", "type": "chips",
            "msg": "Do you feel you have enough **support** from family, friends, or your care team?",
            "key": "mood_support",
            "chips": ["Yes, I feel supported", "Somewhat", "No, I need more support"],
            "next": "other_q1",
        },

        # ── OTHER SYMPTOMS ────────────────────────────────────────
        {
            "id": "other_q1", "type": "chips",
            "msg": "**Other Symptoms** 🩺\n\nHave you had any **cough** that's been bothersome?",
            "key": "other_cough",
            "chips": ["Yes", "No", "A little"],
            "next": "other_skin",
        },
        {
            "id": "other_skin", "type": "chips",
            "msg": "Have you noticed any **skin changes** in the area being treated — like redness, peeling, or soreness?",
            "key": "other_skin",
            "chips": ["Yes", "No", "A little"],
            "next": "other_concentration",
        },
        {
            "id": "other_concentration", "type": "chips",
            "msg": "Have you had **difficulty concentrating** or remembering things?",
            "key": "other_concentration",
            "chips": ["Yes, noticeably", "A little", "No"],
            "next": "closing_q1",
        },

        # ── CLOSING ──────────────────────────────────────────────
        {
            "id": "closing_q1", "type": "text",
            "msg": (
                "We're almost done! 🎉\n\nIs there anything else you'd like your "
                "care team to know before your appointment — anything we haven't "
                "covered, or anything you want to make sure they see?"
            ),
            "key": "additional_notes",
            "next": "done",
        },
        {
            "id": "done", "type": "info",
            "msg": (
                "Thank you so much, **{name}**! 🙏 You've done a wonderful job "
                "sharing how you've been feeling.\n\n"
                "Your responses have been recorded and a summary report will be "
                "sent to your care team before your appointment. They'll review "
                "it and be ready to discuss your concerns with you.\n\n"
                "If you have any urgent concerns before your appointment, please "
                "contact your care team directly. Take care of yourself! 💙"
            ),
            "next": None,
        },
    ]


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def _yes(val: str) -> bool:
    """Return True if the value indicates a positive/present response."""
    val = (val or "").lower().strip()
    return any(w in val for w in ["yes", "a little", "somewhat", "a lot", "often",
                                   "constantly", "most", "on and off", "only",
                                   "occasionally", "noticeably", "quite", "very",
                                   "difficulty", "trouble"])


def now_ts() -> str:
    return datetime.now().strftime("%I:%M %p")


def pain_color(n: int) -> str:
    """Traffic-light color for pain scale."""
    if n <= 3:   return "#22c55e"   # green
    elif n <= 6: return "#f59e0b"   # amber
    else:        return "#ef4444"   # red


def total_steps() -> int:
    return len([s for s in make_flow() if s["type"] != "info"])


def completed_steps(symptoms: dict) -> int:
    return len([k for k in symptoms if symptoms[k]])


def get_step(step_id: str):
    """Find a step by id."""
    for s in make_flow():
        if s["id"] == step_id:
            return s
    return None


def next_step_id(current_id: str, symptoms: dict) -> str | None:
    """
    Walk forward from current step, skipping steps whose cond() is False.
    Returns next visible step id, or None if conversation is over.
    """
    flow = make_flow()
    ids  = [s["id"] for s in flow]
    try:
        idx = ids.index(current_id)
    except ValueError:
        return None

    i = idx + 1
    while i < len(flow):
        step = flow[i]
        cond = step.get("cond", None)
        if cond is None or cond(symptoms):
            return step["id"]
        i += 1
    return None


# ─────────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────
def generate_report(s: dict) -> str:
    """Build the structured clinical report from collected symptom data."""
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    name = s.get("patient_name", "Unknown")
    lines = []
    add = lines.append

    add("=" * 62)
    add("  CHATREPORT SYMPTOM SUMMARY")
    add("  Fox Chase Cancer Center × Temple University")
    add("=" * 62)
    add(f"  Patient Name  : {name}")
    add(f"  Report Date   : {now}")
    add(f"  Report Type   : Pre-Appointment Symptom Checkin")
    add("=" * 62)
    add("")

    # ── Collect alerts ───────────────────────────────────────────
    high_alerts  = []
    watch_alerts = []

    pain_sev = int(s.get("pain_severity", 0) or 0)
    if pain_sev >= 7:
        high_alerts.append(f"Severe pain reported ({pain_sev}/10) — review pain management")
    elif pain_sev >= 4:
        watch_alerts.append(f"Moderate pain ({pain_sev}/10) — monitor closely")

    if "only" in (s.get("swallow_diet") or "").lower() or "tube" in (s.get("swallow_diet") or "").lower():
        high_alerts.append("Patient on liquids only or feeding tube — nutritional consult may be needed")

    if "lost" in (s.get("nutrition_weight") or "").lower():
        watch_alerts.append(f"Weight loss reported: {s.get('nutrition_weight_amt','amount not specified')}")

    mood = (s.get("mood_general") or "").lower()
    if "very distressed" in mood or "quite sad" in mood:
        high_alerts.append("Patient reports significant emotional distress — consider psychosocial referral")
    elif "anxious" in mood or "a bit down" in mood:
        watch_alerts.append("Elevated anxiety/low mood — check in during appointment")

    breathing = (s.get("breathing_present") or "").lower()
    if _yes(breathing):
        high_alerts.append("Breathing difficulties reported — assess for obstruction or infection")

    # ── PAIN ─────────────────────────────────────────────────────
    add("🔴  SYMPTOMS PRESENT")
    add("-" * 62)
    present_any = False

    if _yes(s.get("pain_present", "")):
        present_any = True
        add("")
        add("  PAIN — PRESENT")
        location = s.get("pain_location") or "Not specified"
        add(f"    Location    : {location}")
        add(f"    Severity    : {pain_sev}/10" + (" ⚠️  HIGH" if pain_sev >= 7 else ""))
        add(f"    Frequency   : {s.get('pain_frequency','Not reported')}")
        add(f"    Management  : {s.get('pain_management','None reported')}")
        add(f"    Impact      : {s.get('pain_impact','Not described')}")

    # ── MOUTH ────────────────────────────────────────────────────
    if _yes(s.get("mouth_present", "")):
        present_any = True
        add("")
        add("  MOUTH SYMPTOMS — PRESENT")
        add(f"    Dry Mouth   : {s.get('mouth_dry','Not specified')}")
        add(f"    Mouth Sores : {s.get('mouth_sores','Not specified')}")
        add(f"    Taste Change: {s.get('mouth_taste','Not specified')}")
        add(f"    Impact      : {s.get('mouth_impact','Not described')}")

    # ── SWALLOWING ───────────────────────────────────────────────
    if _yes(s.get("swallow_present", "")):
        present_any = True
        add("")
        add("  SWALLOWING DIFFICULTY — PRESENT")
        add(f"    Pain w/ swallow: {s.get('swallow_pain','Not specified')}")
        add(f"    Diet level     : {s.get('swallow_diet','Not specified')}")
        add(f"    Choking/cough  : {s.get('swallow_choking','Not reported')}")

    # ── BREATHING ────────────────────────────────────────────────
    if _yes(s.get("breathing_present", "")):
        present_any = True
        add("")
        add("  BREATHING DIFFICULTY — PRESENT ⚠️")
        add(f"    Details: {s.get('breathing_detail','Not described')}")

    if not present_any:
        add("  (No major physical symptoms reported)")

    add("")
    add("=" * 62)

    # ── NUTRITION ────────────────────────────────────────────────
    add("")
    add("📊  NUTRITIONAL STATUS")
    add("-" * 62)
    add(f"  Appetite      : {s.get('nutrition_appetite','Not reported')}")
    add(f"  Weight        : {s.get('nutrition_weight','Not reported')}")
    if s.get("nutrition_weight_amt"):
        add(f"  Weight amt    : {s.get('nutrition_weight_amt')}")
    add(f"  Nausea/Vomit  : {s.get('nutrition_nausea','Not reported')}")
    add(f"  Supplements   : {s.get('nutrition_supplements','Not reported')}")

    # ── FATIGUE ──────────────────────────────────────────────────
    fatigue = s.get("fatigue_level", "Not rated")
    add("")
    add("😴  FATIGUE")
    add("-" * 62)
    add(f"  Fatigue Level : {fatigue}/10")
    if s.get("fatigue_impact"):
        add(f"  Impact        : {s.get('fatigue_impact')}")

    # ── MOOD ─────────────────────────────────────────────────────
    add("")
    add("💭  EMOTIONAL WELLBEING")
    add("-" * 62)
    add(f"  General Mood  : {s.get('mood_general','Not reported')}")
    add(f"  Anxiety       : {s.get('mood_anxiety','Not reported')}")
    add(f"  Sleep         : {s.get('mood_sleep','Not reported')}")
    add(f"  Support       : {s.get('mood_support','Not reported')}")

    # ── OTHER ────────────────────────────────────────────────────
    add("")
    add("🩺  OTHER SYMPTOMS")
    add("-" * 62)
    add(f"  Cough         : {s.get('other_cough','Not reported')}")
    add(f"  Skin changes  : {s.get('other_skin','Not reported')}")
    add(f"  Concentration : {s.get('other_concentration','Not reported')}")

    # ── ADDITIONAL NOTES ─────────────────────────────────────────
    if s.get("additional_notes"):
        add("")
        add("📝  ADDITIONAL NOTES (patient's words)")
        add("-" * 62)
        add(f"  {s.get('additional_notes')}")

    # ── NOT REPORTED ─────────────────────────────────────────────
    not_present = []
    checks = {
        "Pain": "pain_present", "Mouth symptoms": "mouth_present",
        "Swallowing difficulty": "swallow_present",
        "Breathing problems": "breathing_present",
    }
    for label, key in checks.items():
        val = s.get(key, "")
        if val and not _yes(val):
            not_present.append(label)
    if not_present:
        add("")
        add("✅  SYMPTOMS NOT REPORTED")
        add("-" * 62)
        for item in not_present:
            add(f"  • {item}")

    # ── CLINICAL ALERTS ──────────────────────────────────────────
    add("")
    add("=" * 62)
    add("  CLINICAL ALERTS")
    add("=" * 62)
    if high_alerts:
        for a in high_alerts:
            add(f"  🔴 HIGH PRIORITY: {a}")
    if watch_alerts:
        for a in watch_alerts:
            add(f"  ⚠️  MONITOR: {a}")
    if not high_alerts and not watch_alerts:
        add("  ✅ No critical alerts at this time.")

    add("")
    add("=" * 62)
    add("  This report was generated by ChatReport (Phase 1 prototype).")
    add("  For clinical decisions, consult the treating physician.")
    add("=" * 62)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages":        [],   # [{role, text, ts, extra}]
        "current_step":    "welcome",
        "symptoms":        {},
        "awaiting_input":  False,
        "conversation_done": False,
        "report":          None,
        "show_body":       False,
        "selected_regions":[],
        "input_clear":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────
# MESSAGE HELPERS
# ─────────────────────────────────────────────────────────────────
def push_bot(text: str, extra=None):
    """Add a bot message to the history."""
    # Personalise {name} placeholders
    name = st.session_state.symptoms.get("patient_name", "")
    text = text.replace("{name}", name)
    st.session_state.messages.append({
        "role": "bot", "text": text,
        "ts": now_ts(), "extra": extra
    })


def push_user(text: str):
    st.session_state.messages.append({
        "role": "user", "text": text,
        "ts": now_ts(), "extra": None
    })


# ─────────────────────────────────────────────────────────────────
# ADVANCE THE CONVERSATION
# ─────────────────────────────────────────────────────────────────
def advance(user_answer: str | None = None):
    """
    Process the user's answer (if any) and push the next bot question.
    Called once per user interaction.
    """
    symptoms = st.session_state.symptoms
    current  = st.session_state.current_step

    step = get_step(current)
    if step is None:
        return

    # ── Store the user's answer for the current step ─────────────
    if user_answer is not None and step.get("key"):
        symptoms[step["key"]] = user_answer
        push_user(user_answer)

    # ── Find the next step ───────────────────────────────────────
    nid = next_step_id(current, symptoms)

    if nid is None:
        # Conversation finished
        st.session_state.conversation_done = True
        st.session_state.report = generate_report(symptoms)
        return

    st.session_state.current_step = nid
    nstep = get_step(nid)

    # ── Push bot message ─────────────────────────────────────────
    extra = None
    if nstep["type"] == "body":
        extra = "body_diagram"
    elif nstep["type"] == "scale":
        extra = "pain_scale"
    elif nstep["type"] == "chips":
        extra = {"chips": nstep.get("chips", [])}

    push_bot(nstep["msg"], extra)

    # ── If it's an info step (no input needed) advance further ───
    if nstep["type"] == "info":
        advance()  # recurse to next real question

    st.session_state.awaiting_input = True


# ─────────────────────────────────────────────────────────────────
# KICK OFF (first load)
# ─────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    # Seed the welcome message
    welcome = get_step("welcome")
    push_bot(welcome["msg"])
    # Advance to name question
    st.session_state.current_step = "welcome"
    advance()   # moves from welcome (info) → name


# ─────────────────────────────────────────────────────────────────
# RENDER CHAT MESSAGES
# ─────────────────────────────────────────────────────────────────
def render_bubble(msg: dict, idx: int):
    role  = msg["role"]
    text  = msg["text"]
    ts    = msg["ts"]
    extra = msg["extra"]
    is_last = (idx == len(st.session_state.messages) - 1)

    # Convert **bold** markdown to <strong>
    def md_bold(t):
        return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', t).replace("\n", "<br>")

    if role == "bot":
        st.markdown(f"""
        <div class="row row-bot">
          <div class="av av-bot">🤖</div>
          <div>
            <div class="bubble bubble-bot">{md_bold(text)}</div>
            <div class="ts">{ts}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Render interactive extras only for the LAST bot message ──
        if is_last and extra and not st.session_state.conversation_done:

            # Body diagram
            if extra == "body_diagram":
                with st.container():
                    st.markdown('<div class="body-diagram-wrap"><h4>👆 Select pain location(s)</h4>', unsafe_allow_html=True)
                    cols = st.columns([1, 1])
                    with cols[0]:
                        st.markdown(BODY_SVG, unsafe_allow_html=True)
                    with cols[1]:
                        selected = st.multiselect(
                            "Regions:",
                            options=list(REGION_LABELS.values()),
                            default=st.session_state.selected_regions,
                            key="body_multiselect",
                            label_visibility="collapsed",
                        )
                        st.session_state.selected_regions = selected
                        if selected:
                            st.markdown(
                                "**Selected:** " + ", ".join(selected),
                                help="You can select multiple areas"
                            )
                        if st.button("Confirm location →", key="confirm_body"):
                            loc = ", ".join(selected) if selected else "Not specified"
                            st.session_state.selected_regions = []
                            advance(loc)
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

            # Pain scale
            elif extra == "pain_scale":
                st.markdown('<div style="margin-top:8px">', unsafe_allow_html=True)
                scale_cols = st.columns(11)
                for n in range(0, 11):
                    clr = pain_color(n)
                    with scale_cols[n]:
                        if st.button(
                            str(n),
                            key=f"pain_scale_{n}",
                            help=["No pain","","","Mild","","Moderate","","","Severe","","Worst"][n],
                        ):
                            advance(str(n))
                            st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption("0 = No pain at all  ·  10 = Worst pain imaginable")

            # Quick-reply chips
            elif isinstance(extra, dict) and "chips" in extra:
                cols = st.columns(len(extra["chips"]))
                for ci, chip in enumerate(extra["chips"]):
                    with cols[ci]:
                        if st.button(chip, key=f"chip_{idx}_{ci}"):
                            advance(chip)
                            st.rerun()

    else:  # user
        st.markdown(f"""
        <div class="row row-user">
          <div>
            <div class="bubble bubble-user">{md_bold(text)}</div>
            <div class="ts ts-right">{ts} ✓✓</div>
          </div>
          <div class="av av-user">👤</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SIDEBAR — progress tracker
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Progress")
    domains = [
        ("Pain",          "pain_present"),
        ("Mouth",         "mouth_present"),
        ("Swallowing",    "swallow_present"),
        ("Nutrition",     "nutrition_appetite"),
        ("Breathing",     "breathing_present"),
        ("Fatigue",       "fatigue_level"),
        ("Mood",          "mood_general"),
        ("Other",         "other_cough"),
    ]
    for domain, key in domains:
        done = bool(st.session_state.symptoms.get(key))
        icon = "✅" if done else "⏳"
        st.markdown(f"{icon} {domain}")

    st.divider()
    if st.button("🔄 Start Over"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ─────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────
progress_pct = min(
    int(len([v for v in st.session_state.symptoms.values() if v]) / max(total_steps(), 1) * 100),
    100
)
st.markdown(f"""
<div class="chat-header">
  <div class="header-avatar">🩺</div>
  <div class="header-text">
    <h2>ChatReport</h2>
    <p>Fox Chase Cancer Center × Temple University</p>
  </div>
  <div class="header-badge">Phase 1 Prototype</div>
</div>
<div style="padding: 6px 0 0">
  <div class="progress-wrap">
    <div class="progress-fill" style="width:{progress_pct}%"></div>
  </div>
  <div class="progress-label">{progress_pct}% complete</div>
</div>
""", unsafe_allow_html=True)

# ── Chat messages ─────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)

    for i, msg in enumerate(st.session_state.messages):
        render_bubble(msg, i)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Input row (only when conversation is active) ──────────────────
if not st.session_state.conversation_done:
    current_step = get_step(st.session_state.current_step)
    step_type = current_step["type"] if current_step else "text"

    # Only show free-text input for "text" type steps
    if step_type == "text":
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        in_col, btn_col = st.columns([9, 1])
        with in_col:
            placeholder = "Type your answer here…"
            user_input = st.text_input(
                "msg",
                key="text_input",
                placeholder=placeholder,
                label_visibility="collapsed",
            )
        with btn_col:
            send = st.button("➤", key="send_btn")

        if (send or user_input) and user_input.strip():
            advance(user_input.strip())
            st.rerun()
    else:
        st.info("👆 Please use the buttons above to answer.", icon="ℹ️")

# ── Conversation done — show report ──────────────────────────────
else:
    st.markdown("---")
    st.markdown("### 📄 Clinical Report Generated")
    st.markdown(
        f'<div class="report-container">{st.session_state.report}</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="⬇️ Download Report (.txt)",
            data=st.session_state.report,
            file_name=f"ChatReport_{st.session_state.symptoms.get('patient_name','Patient')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
        )
    with col2:
        if st.button("🔄 New Session"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)  # close chat-shell
