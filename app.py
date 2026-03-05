import json
from datetime import datetime
from typing import Dict, List, Optional, Set

import streamlit as st
import streamlit.components.v1 as components
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ============================================================
# MODERN UI STYLE
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    padding-top: 2.5rem;
    max-width: 860px;
}

/* Page background */
.stApp {
    background: linear-gradient(135deg, #f0f5fb 0%, #e8f0f9 100%);
}

/* Title */
h1 {
    font-family: 'DM Serif Display', serif;
    font-weight: 400;
    font-size: 2rem;
    color: #1a3a5c;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}

h2, h3 {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: #1a3a5c;
}

/* Buttons – base */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    padding: 0.65rem 1rem;
    font-size: 14.5px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    border: 1.5px solid #d4deef;
    background: #ffffff;
    color: #1a3a5c;
    box-shadow: 0 2px 6px rgba(26,58,92,0.06);
    transition: all 0.18s ease;
    letter-spacing: 0.01em;
}

.stButton > button:hover {
    border-color: #3d8fc0;
    background: #eef5fc;
    color: #1a5a8a;
    box-shadow: 0 4px 14px rgba(61,143,192,0.15);
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0px);
    box-shadow: 0 2px 6px rgba(26,58,92,0.08);
}

/* Cards */
.card {
    padding: 28px 30px;
    border-radius: 18px;
    border: 1.5px solid #dde8f5;
    background: #ffffff;
    margin-bottom: 18px;
    box-shadow: 0 4px 20px rgba(26,58,92,0.07);
}

/* Doctor box */
.doctor-box {
    padding: 22px 24px;
    border-radius: 16px;
    background: linear-gradient(135deg, #eaf4fc 0%, #f0f8ff 100%);
    border: 1.5px solid #b8d9f0;
    font-size: 15px;
    font-family: 'DM Sans', sans-serif;
    color: #1a3a5c;
    line-height: 1.65;
    box-shadow: 0 3px 12px rgba(61,143,192,0.1);
    position: relative;
}

.doctor-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, #3d8fc0, #5ab0d8);
    border-radius: 16px 0 0 16px;
}

/* Section titles */
.section-title {
    font-size: 17px;
    font-weight: 600;
    color: #1a3a5c;
    margin-bottom: 14px;
    letter-spacing: -0.1px;
}

/* Success box */
.success-box {
    padding: 32px 24px;
    border-radius: 18px;
    background: linear-gradient(135deg, #e8fbf0 0%, #f0fff6 100%);
    border: 1.5px solid #8dd9ae;
    font-size: 18px;
    font-family: 'DM Serif Display', serif;
    font-weight: 400;
    color: #1a4d32;
    text-align: center;
    box-shadow: 0 4px 18px rgba(61,180,110,0.12);
    letter-spacing: 0.01em;
}

/* Input fields */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border-radius: 10px;
    border: 1.5px solid #d4deef;
    font-family: 'DM Sans', sans-serif;
    font-size: 14.5px;
    color: #1a3a5c;
    padding: 0.5rem 0.8rem;
    background: #fafcff;
    transition: border 0.15s ease, box-shadow 0.15s ease;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #3d8fc0;
    box-shadow: 0 0 0 3px rgba(61,143,192,0.12);
    background: #ffffff;
}

/* Radio buttons */
.stRadio > div {
    gap: 10px;
}

.stRadio > div > label {
    font-family: 'DM Sans', sans-serif;
    font-size: 14.5px;
    color: #1a3a5c;
}

/* Divider */
hr {
    border: none;
    border-top: 1.5px solid #e4edf7;
    margin: 18px 0;
}

/* Number input label */
.stNumberInput label, .stTextInput label, .stRadio label {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    color: #3a5a7a;
    font-size: 13.5px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# Secrets helpers
# ============================================================

def _secret(*keys, default=None):
    for k in keys:
        if k in st.secrets:
            return st.secrets[k]
    return default

def _require_secret(*keys):
    v = _secret(*keys)
    if v is None:
        raise KeyError(f"Missing secret. Tried: {', '.join(keys)}")
    return v


# ============================================================
# Google Sheets
# ============================================================

sheet = None
sheets_init_error: Optional[str] = None

def _init_sheets():
    global sheet, sheets_init_error
    if sheet is not None or sheets_init_error is not None:
        return
    try:
        creds = Credentials.from_service_account_info(
            _require_secret("gcp_service_account"),
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        book = gspread.authorize(creds).open_by_key(_require_secret("gsheet_id"))
        try:
            ws = book.worksheet("Form")
        except Exception:
            ws = book.add_worksheet(title="Form", rows=2000, cols=20)
            ws.append_row(["timestamp", "name", "json"])
        sheet = ws
    except Exception as e:
        sheets_init_error = str(e)

def load_past_checkins(name: str) -> List[Dict]:
    _init_sheets()
    if sheet is None:
        return []
    try:
        past: List[Dict] = []
        rows = sheet.get_all_values()
        for row in rows[1:]:
            if len(row) >= 3 and row[1].strip().lower() == name.strip().lower():
                try:
                    d = json.loads(row[2])
                    d["timestamp"] = row[0]
                    past.append(d)
                except Exception:
                    continue
        return past[-5:]
    except Exception:
        return []

def save_to_sheet(payload: Dict):
    _init_sheets()
    if sheet is None:
        raise RuntimeError(f"Sheets unavailable: {sheets_init_error}")
    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        payload.get("name", "Unknown"),
        json.dumps(payload),
    ])


# ============================================================
# Body map SVG
# ============================================================

def body_svg(colors: Dict[str, str]) -> str:

    def c(p): return colors.get(p, "#cfd8e6")
    stroke = "#6b7a90"

    return f"""
<svg width="260" height="410" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">

  <circle cx="160" cy="70" r="38" fill="{c('Head')}" stroke="{stroke}" stroke-width="2"/>
  <rect x="110" y="120" width="100" height="70" rx="24" fill="{c('Chest')}" stroke="{stroke}" stroke-width="2"/>
  <rect x="115" y="195" width="90" height="70" rx="22" fill="{c('Abdomen')}" stroke="{stroke}" stroke-width="2"/>

  <rect x="60" y="140" width="40" height="120" rx="20" fill="{c('Left Arm')}" stroke="{stroke}" stroke-width="2"/>
  <rect x="220" y="140" width="40" height="120" rx="20" fill="{c('Right Arm')}" stroke="{stroke}" stroke-width="2"/>

  <rect x="130" y="270" width="35" height="150" rx="20" fill="{c('Left Leg')}" stroke="{stroke}" stroke-width="2"/>
  <rect x="165" y="270" width="35" height="150" rx="20" fill="{c('Right Leg')}" stroke="{stroke}" stroke-width="2"/>

</svg>
"""



# ============================================================
# Voice input widget
# ============================================================

def voice_input_widget(placeholder: str = "Click the mic and speak…") -> None:
    """Renders a Web Speech API microphone widget. Transcript is auto-copied to clipboard."""
    html = f"""
<style>
  body {{ margin:0; padding:0; background:transparent; font-family:'DM Sans',sans-serif; }}
  .vbox {{
    display:flex; align-items:center; gap:12px;
    padding:12px 16px; border-radius:12px;
    background:#f0f5fb; border:1.5px solid #d4deef;
  }}
  #mic {{
    width:42px; height:42px; border-radius:50%; flex-shrink:0;
    background:linear-gradient(135deg,#3d8fc0,#5ab0d8);
    border:none; cursor:pointer; font-size:18px;
    box-shadow:0 3px 10px rgba(61,143,192,0.3);
    transition:transform 0.15s;
  }}
  #mic:hover {{ transform:scale(1.08); }}
  #mic.on {{ background:linear-gradient(135deg,#e74c3c,#ff6b6b); animation:pulse 1s infinite; }}
  @keyframes pulse {{
    0%,100% {{ box-shadow:0 3px 10px rgba(231,76,60,0.3); }}
    50%      {{ box-shadow:0 3px 22px rgba(231,76,60,0.65); }}
  }}
  #txt {{
    flex:1; font-size:14px; color:#1a3a5c; line-height:1.5;
    min-height:20px; word-break:break-word;
  }}
  #txt.ph {{ color:#9aafc7; font-style:italic; }}
  #cpybtn {{
    background:#1a3a5c; color:#fff; border:none; cursor:pointer;
    border-radius:8px; padding:6px 14px; font-size:12px; font-weight:500;
    opacity:0; transition:opacity 0.2s; white-space:nowrap;
  }}
  #cpybtn.show {{ opacity:1; }}
  #cpybtn:hover {{ background:#3d8fc0; }}
  #note {{
    font-size:11px; color:#9aafc7; margin-top:6px; padding-left:2px;
  }}
</style>
<div class="vbox">
  <button id="mic" onclick="toggleMic()">🎤</button>
  <div id="txt" class="ph">{placeholder}</div>
  <button id="cpybtn" onclick="doCopy()">Copy ↑</button>
</div>
<div id="note"></div>
<script>
var rec = null, final = '', going = false;
function toggleMic() {{
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {{
    document.getElementById('txt').textContent = '⚠ Voice not supported in this browser.';
    document.getElementById('txt').classList.remove('ph');
    return;
  }}
  if (going) {{ rec.stop(); return; }}
  rec = new SR(); rec.continuous = false; rec.interimResults = true; rec.lang = 'en-US';
  rec.onstart = function() {{
    going = true; final = '';
    document.getElementById('mic').classList.add('on');
    document.getElementById('mic').textContent = '⏹';
    document.getElementById('txt').textContent = 'Listening…';
    document.getElementById('txt').classList.remove('ph');
    document.getElementById('cpybtn').classList.remove('show');
    document.getElementById('note').textContent = '';
  }};
  rec.onresult = function(e) {{
    var interim = ''; final = '';
    for (var i = e.resultIndex; i < e.results.length; i++) {{
      if (e.results[i].isFinal) final += e.results[i][0].transcript;
      else interim += e.results[i][0].transcript;
    }}
    document.getElementById('txt').textContent = final || interim || 'Listening…';
  }};
  rec.onend = function() {{
    going = false;
    document.getElementById('mic').classList.remove('on');
    document.getElementById('mic').textContent = '🎤';
    if (final) {{
      document.getElementById('cpybtn').classList.add('show');
      document.getElementById('note').textContent = '✔ Copied to clipboard — paste into the field above';
      navigator.clipboard.writeText(final).catch(function(){{}});
    }} else {{
      document.getElementById('txt').textContent = '{placeholder}';
      document.getElementById('txt').classList.add('ph');
    }}
  }};
  rec.onerror = function(e) {{
    going = false;
    document.getElementById('mic').classList.remove('on');
    document.getElementById('mic').textContent = '🎤';
    document.getElementById('txt').textContent = 'Error: ' + e.error;
  }};
  rec.start();
}}
function doCopy() {{
  if (!final) return;
  navigator.clipboard.writeText(final).then(function() {{
    var b = document.getElementById('cpybtn');
    b.textContent = 'Copied!';
    setTimeout(function(){{ b.textContent = 'Copy ↑'; }}, 1500);
  }});
}}
</script>
"""
    components.html(html, height=90)



DEFAULTS = {
    "stage": -1,
    "name": "",
    "past_checkins": [],
    "last_summary": None,
    "last_pain_severity": {},
    "feeling_level": None,
    "pain_yesno": None,
    "selected_parts": set(),
    "pain_severity": {},
    "pain_reason": {},
    "symptoms": set(),
    "submitted": False,
    "show_other_pain": False,
    "other_pain_text": "",
    "show_other_symptom": False,
    "other_symptom_text": "",
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# Body map logic
# ============================================================

REGIONS = ["Head", "Chest", "Abdomen", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]

GREEN = "#6fd08c"
ORANGE = "#f5a623"
RED = "#e74c3c"


def region_color_state(region: str) -> str:

    last = st.session_state.last_pain_severity
    selected = region in st.session_state.selected_parts

    if not selected:
        if region in last:
            return ORANGE
        return GREEN

    last_val = last.get(region, 0)
    cur = st.session_state.pain_severity.get(region, last_val)

    if region not in last:
        return RED

    if cur >= last_val + 2:
        return RED

    return ORANGE


def current_svg_colors():
    return {r: region_color_state(r) for r in REGIONS}


# ============================================================
# UI
# ============================================================

st.title("🩺 Cancer Symptom Check-In")

_init_sheets()

# ------------------------------------------------------------
# Stage -1 : Name
# ------------------------------------------------------------

if st.session_state.stage == -1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    name = st.text_input("Enter your name")

    if st.button("Start Check-In"):

        if name.strip():

            st.session_state.name = name.strip()

            past = load_past_checkins(name)
            st.session_state.past_checkins = past

            if past:
                last = past[-1]
                st.session_state.last_summary = last
                st.session_state.last_pain_severity = last.get("pain_severity", {})
            else:
                st.session_state.last_summary = None
                st.session_state.last_pain_severity = {}

            st.session_state.stage = 0
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ------------------------------------------------------------
# Stage 0 : Doctor recap
# ------------------------------------------------------------

if st.session_state.stage == 0:

    last = st.session_state.last_summary

    if last:

        ts = last.get("timestamp", "")
        pain_locs = last.get("pain_locations", [])
        symptoms = last.get("symptoms", [])
        feeling = last.get("feeling_level")

        message = f"Hi {st.session_state.name}, I reviewed your last check-in"

        if ts:
            message += f" from {ts.split()[0]}"

        message += ". "

        if pain_locs:
            message += f"You mentioned pain in your {', '.join(pain_locs)}. "

        if symptoms:
            message += f"You also reported {', '.join(symptoms)}. "

        if feeling is not None:
            message += f"Your overall feeling level was {feeling}/10. "

        message += "Before we continue, has anything changed since then?"

        st.markdown(f'<div class="doctor-box">👩‍⚕️ {message}</div>', unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Same as yesterday"):

            payload = {
                "name": st.session_state.name,
                "note": "Same as yesterday"
            }

            save_to_sheet(payload)

            st.session_state.stage = 5
            st.rerun()

    with c2:
        if st.button("Something changed"):
            st.session_state.stage = 1
            st.rerun()


# ------------------------------------------------------------
# Stage 1
# ------------------------------------------------------------

elif st.session_state.stage == 1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">How are you feeling today?</div>', unsafe_allow_html=True)

    feeling = st.number_input("Feeling (0-10)", 0, 10, value=7)

    if st.button("Next"):
        st.session_state.feeling_level = feeling
        st.session_state.stage = 2
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Stage 2
# ------------------------------------------------------------

elif st.session_state.stage == 2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Do you have pain today?</div>', unsafe_allow_html=True)

    pain = st.radio("", ["No", "Yes"])

    if st.button("Next"):

        st.session_state.pain_yesno = (pain == "Yes")

        if pain == "Yes":
            st.session_state.stage = 3
        else:
            st.session_state.stage = 4

        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Stage 3
# ------------------------------------------------------------

elif st.session_state.stage == 3:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Where do you feel pain?</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown(body_svg(current_svg_colors()), unsafe_allow_html=True)

    with col2:

        last = st.session_state.last_pain_severity

        for r in REGIONS:

            icon = "🟢"

            col = region_color_state(r)

            if col == ORANGE:
                icon = "🟠"
            if col == RED:
                icon = "🔴"

            if st.button(f"{icon} {r}"):

                if r in st.session_state.selected_parts:
                    st.session_state.selected_parts.remove(r)
                else:
                    st.session_state.selected_parts.add(r)

                st.rerun()

            if r in st.session_state.selected_parts:

                last_val = last.get(r, 0)

                sev = st.number_input(
                    f"{r} severity",
                    0,
                    10,
                    value=last_val,
                    key=f"sev_{r}"
                )

                st.session_state.pain_severity[r] = sev

                if sev > 6 or sev >= last_val + 2:

                    reason = st.text_input("Reason", key=f"reason_{r}")

                    if reason:
                        st.session_state.pain_reason[r] = reason

        # ── Other pain location ──────────────────────────────
        other_pain_selected = st.session_state.show_other_pain
        other_pain_icon = "🔴 Other" if other_pain_selected else "🟢 Other"

        if st.button(other_pain_icon, key="other_pain_btn"):
            st.session_state.show_other_pain = not st.session_state.show_other_pain
            if not st.session_state.show_other_pain:
                st.session_state.other_pain_text = ""
            st.rerun()

        if st.session_state.show_other_pain:
            st.session_state.other_pain_text = st.text_input(
                "Describe the pain location",
                value=st.session_state.other_pain_text,
                placeholder="Type here, or use the mic below and paste…",
                key="other_pain_input",
            )
            voice_input_widget("Click mic, speak the location, then paste above ↑")

    if st.button("Next"):
        st.session_state.stage = 4
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Stage 4
# ------------------------------------------------------------

elif st.session_state.stage == 4:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Symptoms today</div>', unsafe_allow_html=True)

    symptom_options = [
        "Fatigue",
        "Nausea",
        "Dry mouth",
        "Difficulty swallowing",
        "Hoarseness",
        "Mouth sores",
        "Skin irritation",
        "Loss of taste",
    ]

    cols = st.columns(2)

    for i, sym in enumerate(symptom_options):

        selected = sym in st.session_state.symptoms
        label = f"🔴 {sym}" if selected else f"🟢 {sym}"

        with cols[i % 2]:

            if st.button(label):

                if selected:
                    st.session_state.symptoms.remove(sym)
                else:
                    st.session_state.symptoms.add(sym)

                st.rerun()

    st.markdown("---")

    # ── Other symptom ────────────────────────────────────────
    other_sym_selected = st.session_state.show_other_symptom
    other_sym_icon = "🔴 Other" if other_sym_selected else "🟢 Other"

    if st.button(other_sym_icon, key="other_sym_btn"):
        st.session_state.show_other_symptom = not st.session_state.show_other_symptom
        if not st.session_state.show_other_symptom:
            st.session_state.other_symptom_text = ""
        st.rerun()

    if st.session_state.show_other_symptom:
        st.session_state.other_symptom_text = st.text_input(
            "Describe the symptom",
            value=st.session_state.other_symptom_text,
            placeholder="Type here, or use the mic below and paste…",
            key="other_sym_input",
        )
        voice_input_widget("Click mic, speak the symptom, then paste above ↑")

    st.markdown("---")

    if st.button("Submit Check-In"):

        all_symptoms = list(st.session_state.symptoms)
        if st.session_state.other_symptom_text.strip():
            all_symptoms.append(st.session_state.other_symptom_text.strip())

        all_pain_locations = list(st.session_state.selected_parts)
        if st.session_state.other_pain_text.strip():
            all_pain_locations.append(st.session_state.other_pain_text.strip())

        payload = {
            "name": st.session_state.name,
            "feeling_level": st.session_state.feeling_level,
            "pain": st.session_state.pain_yesno,
            "pain_locations": all_pain_locations,
            "pain_severity": st.session_state.pain_severity,
            "pain_reason": st.session_state.pain_reason,
            "symptoms": all_symptoms,
        }

        save_to_sheet(payload)

        st.session_state.stage = 5
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Stage 5
# ------------------------------------------------------------

elif st.session_state.stage == 5:

    st.markdown('<div class="success-box">✅ Check-in complete.</div>', unsafe_allow_html=True)

    if st.button("Start another check-in"):

        for k, v in DEFAULTS.items():
            st.session_state[k] = v

        st.rerun()
