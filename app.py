import json
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(
    page_title="Symptom Check-In",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Custom CSS — Modern Dark Medical Theme
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@500;600&display=swap');

/* ── Global Reset ───────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117 !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(56,189,248,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 60% at 80% 90%, rgba(99,102,241,0.07) 0%, transparent 60%),
        #0d1117 !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
section[data-testid="stMain"] > div { padding-top: 2rem; }

/* ── Typography ─────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    letter-spacing: -0.02em;
}

/* ── Card / Panel ───────────────────────────────────────── */
.card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom: 1.5rem;
}

/* ── Header ─────────────────────────────────────────────── */
.app-header {
    text-align: center;
    margin-bottom: 2.5rem;
}
.app-header .badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.3rem 0.85rem;
    border-radius: 100px;
    margin-bottom: 1rem;
}
.app-header h1 {
    font-size: 2.5rem !important;
    background: linear-gradient(135deg, #e8eaf0 30%, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem !important;
    padding: 0 !important;
}
.app-header p {
    color: #64748b;
    font-size: 0.95rem;
    margin: 0;
}

/* ── Input ──────────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(56,189,248,0.5) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.1) !important;
    outline: none !important;
}
[data-testid="stTextInput"] label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.4rem !important;
}

/* ── Buttons ────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 12px !important;
    transition: all 0.2s !important;
    border: none !important;
    cursor: pointer !important;
}

/* Primary CTA */
[data-testid="stButton"][data-baseweb] > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"],
div[data-testid="stButton"]:has(button:not(.region-btn)) button {
    background: linear-gradient(135deg, #38bdf8, #6366f1) !important;
    color: white !important;
    padding: 0.7rem 2rem !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 20px rgba(56,189,248,0.25) !important;
}

/* ── Region Buttons ─────────────────────────────────────── */
.region-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.region-row:last-child { border-bottom: none; }

/* ── Severity Slider Area ───────────────────────────────── */
[data-testid="stNumberInput"] {
    margin-top: 0.25rem;
}
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stNumberInput"] label {
    color: #64748b !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Radio ──────────────────────────────────────────────── */
[data-testid="stRadio"] label {
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
}
[data-testid="stRadio"] > label {
    color: #64748b !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── Success / Info Messages ────────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(56,189,248,0.08) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 12px !important;
    color: #7dd3fc !important;
}

/* ── Columns ────────────────────────────────────────────── */
[data-testid="stHorizontalBlock"] {
    gap: 1.5rem;
    align-items: flex-start;
}

/* ── SVG body map container ─────────────────────────────── */
.body-map-wrap {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    display: flex;
    justify-content: center;
    align-items: center;
}

/* ── Section label ──────────────────────────────────────── */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.75rem;
}

/* ── Legend ─────────────────────────────────────────────── */
.legend {
    display: flex;
    gap: 1.2rem;
    flex-wrap: wrap;
    margin-bottom: 1.25rem;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    color: #64748b;
}
.legend-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Subheader override ─────────────────────────────────── */
[data-testid="stSubheader"] {
    color: #e2e8f0 !important;
    font-size: 1.3rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 1.2rem !important;
}

/* ── Past check-in pill ─────────────────────────────────── */
.checkin-pill {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-size: 0.82rem;
    color: #a5b4fc;
    margin-bottom: 0.5rem;
}

/* ── Divider ────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.07) !important; }
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
        sheet = book.worksheet("Form")
    except Exception as e:
        sheets_init_error = str(e)


def load_past_checkins(name: str) -> List[Dict]:
    _init_sheets()
    if sheet is None:
        return []
    rows = sheet.get_all_values()
    past = []
    for r in rows[1:]:
        if r[1].strip().lower() == name.lower():
            try:
                d = json.loads(r[2])
                d["timestamp"] = r[0]
                past.append(d)
            except:
                pass
    return past[-5:]


def save_to_sheet(payload: Dict):
    _init_sheets()
    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        payload.get("name"),
        json.dumps(payload)
    ])


# ============================================================
# Body Map SVG
# ============================================================

def body_svg(colors: Dict[str, str]):
    def c(p): return colors.get(p, "#1e293b")

    return f"""
<svg width="200" height="340" viewBox="0 0 200 380" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="glow">
      <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
      <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <!-- Head -->
  <circle cx="100" cy="45" r="30" fill="{c('Head')}" stroke="rgba(255,255,255,0.15)" stroke-width="1.5" filter="url(#glow)"/>

  <!-- Neck -->
  <rect x="89" y="73" width="22" height="18" fill="{c('Chest')}" stroke="none" rx="4"/>

  <!-- Chest -->
  <rect x="68" y="89" width="64" height="65" fill="{c('Chest')}" stroke="rgba(255,255,255,0.15)" stroke-width="1.5" rx="8" filter="url(#glow)"/>

  <!-- Abdomen -->
  <rect x="72" y="152" width="56" height="55" fill="{c('Abdomen')}" stroke="rgba(255,255,255,0.15)" stroke-width="1.5" rx="8" filter="url(#glow)"/>

  <!-- Left Arm -->
  <rect x="36" y="95" width="28" height="95" fill="{c('Left Arm')}" stroke="rgba(255,255,255,0.15)" stroke-width="1.5" rx="10" filter="url(#glow)"/>

  <!-- Right Arm -->
  <rect x="136" y="95" width="28" height="95" fill="{c('Right Arm')}" stroke="rgba(255,255,255,0.15)" stroke-width="1.5" rx="10" filter="url(#glow)"/>

  <!-- Left Leg -->
  <rect x="74" y="205" width="26" height="130" fill="{c('Left Leg')}" stroke="rgba(255,255,255,0.15)" stroke-width="1.5" rx="10" filter="url(#glow)"/>

  <!-- Right Leg -->
  <rect x="100" y="205" width="26" height="130" fill="{c('Right Leg')}" stroke="rgba(255,255,255,0.15)" stroke-width="1.5" rx="10" filter="url(#glow)"/>
</svg>
"""


# ============================================================
# Colors
# ============================================================

GREEN  = "#10b981"   # ok
ORANGE = "#f59e0b"   # mild / was painful
RED    = "#ef4444"   # new or worsening

REGIONS = ["Head", "Chest", "Abdomen", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]


def region_color_state(region):
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
# Session state
# ============================================================

defaults = {
    "stage": -1,
    "name": "",
    "past_checkins": [],
    "last_summary": None,
    "last_pain_severity": {},
    "selected_parts": set(),
    "pain_severity": {},
    "pain_reason": {},
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# App Header (always visible)
# ============================================================

st.markdown("""
<div class="app-header">
  <div class="badge">Oncology Care · Daily Tracker</div>
  <h1>Symptom Check-In</h1>
  <p>Track how you're feeling today — takes less than 2 minutes.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Stage -1 — Welcome / Name Entry
# ============================================================

if st.session_state.stage == -1:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Your Name</div>', unsafe_allow_html=True)
    name = st.text_input("Full name", placeholder="e.g. Jane Smith", label_visibility="collapsed")

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Begin Check-In →", use_container_width=True):
            if name.strip():
                st.session_state.name = name.strip()
                past = load_past_checkins(name.strip())
                st.session_state.past_checkins = past
                if past:
                    last = past[-1]
                    st.session_state.last_summary = last
                    st.session_state.last_pain_severity = last.get("pain_severity", {})
                st.session_state.stage = 3
                st.rerun()
            else:
                st.warning("Please enter your name to continue.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; color:#334155; font-size:0.82rem; margin-top:2rem;">
        🔒 &nbsp;Your data is private and securely stored.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Stage 3 — Pain Map
# ============================================================

elif st.session_state.stage == 3:

    # Greeting
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
        <span style="color:#64748b; font-size:0.85rem;">Checking in as</span>
        <span style="color:#e2e8f0; font-weight:600; margin-left:0.4rem;">{st.session_state.name}</span>
    </div>
    """, unsafe_allow_html=True)

    # Legend
    st.markdown("""
    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:#10b981;"></div> No pain</div>
      <div class="legend-item"><div class="legend-dot" style="background:#f59e0b;"></div> Previously reported</div>
      <div class="legend-item"><div class="legend-dot" style="background:#ef4444;"></div> New / worsening</div>
    </div>
    """, unsafe_allow_html=True)

    col_map, col_opts = st.columns([1, 1.3])

    with col_map:
        st.markdown('<div class="body-map-wrap">', unsafe_allow_html=True)
        st.markdown(body_svg(current_svg_colors()), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_opts:
        st.markdown('<div class="section-label">Select affected areas</div>', unsafe_allow_html=True)
        last = st.session_state.last_pain_severity

        for r in REGIONS:
            col = region_color_state(r)
            icon = "🟢" if col == GREEN else ("🟠" if col == ORANGE else "🔴")
            is_selected = r in st.session_state.selected_parts

            btn_label = f"{'✓ ' if is_selected else ''}{icon} {r}"
            if st.button(btn_label, key=f"btn_{r}", use_container_width=True):
                if r in st.session_state.selected_parts:
                    st.session_state.selected_parts.remove(r)
                    st.session_state.pain_severity.pop(r, None)
                    st.session_state.pain_reason.pop(r, None)
                else:
                    st.session_state.selected_parts.add(r)
                st.rerun()

            if r in st.session_state.selected_parts:
                last_val = last.get(r, 0)
                sev = st.number_input(
                    f"Severity (0–10)",
                    0, 10,
                    value=last_val,
                    key=f"sev_{r}"
                )
                st.session_state.pain_severity[r] = sev

                if sev > 6 or sev >= last_val + 2:
                    st.markdown("**What seems to be causing this?**")
                    reason_options = [
                        "Physical activity / strain",
                        "Treatment side effect",
                        "Sleeping position",
                        "Stress / anxiety",
                        "Unknown",
                        "Other",
                    ]
                    reason = st.radio("Reason", reason_options, key=f"reason_select_{r}", label_visibility="collapsed")
                    if reason == "Other":
                        txt = st.text_input("Describe reason", key=f"reason_text_{r}", placeholder="Describe...")
                        st.session_state.pain_reason[r] = txt
                    else:
                        st.session_state.pain_reason[r] = reason

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.divider()

    col_a, col_b = st.columns([3, 1])
    with col_a:
        if st.button("Save & Submit Check-In →", use_container_width=True):
            payload = {
                "name": st.session_state.name,
                "pain_severity": st.session_state.pain_severity,
                "pain_reason": st.session_state.pain_reason,
                "selected_parts": list(st.session_state.selected_parts),
            }
            try:
                save_to_sheet(payload)
                st.success("✓ Your check-in has been saved. Thank you.")
            except Exception as e:
                st.error(f"Could not save: {e}")
    with col_b:
        if st.button("← Start over"):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()
