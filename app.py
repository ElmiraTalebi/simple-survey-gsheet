import json
from datetime import datetime
from typing import Dict, List, Optional, Set

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

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
# Body map SVG (fills regions by state)
# ============================================================

def body_svg(colors: Dict[str, str]) -> str:
    """
    colors: dict mapping region -> hex color
    Regions used: Head, Chest, Abdomen, Left Arm, Right Arm, Left Leg, Right Leg
    """
    def c(p): return colors.get(p, "#cfd8e6")
    stroke = "#6b7a90"

    return f"""
<svg width="260" height="410" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="sh">
      <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="rgba(0,0,0,0.12)"/>
    </filter>
  </defs>

  <g filter="url(#sh)">
    <circle cx="160" cy="70" r="38" fill="{c('Head')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <g filter="url(#sh)">
    <rect x="110" y="120" width="100" height="70" rx="24" fill="{c('Chest')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <g filter="url(#sh)">
    <rect x="115" y="195" width="90" height="70" rx="22" fill="{c('Abdomen')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <g filter="url(#sh)">
    <path d="M110 132 C80 145,72 180,78 220 C82 250,92 270,100 290 C108 310,115 320,120 320 L120 130Z"
          fill="{c('Left Arm')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <g filter="url(#sh)">
    <path d="M210 132 C240 145,248 180,242 220 C238 250,228 270,220 290 C212 310,205 320,200 320 L200 130Z"
          fill="{c('Right Arm')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <g filter="url(#sh)">
    <path d="M135 265 C120 310,118 360,126 410 C132 445,132 475,128 500 L155 500 C158 470,160 435,156 405 C150 355,152 312,165 265Z"
          fill="{c('Left Leg')}" stroke="{stroke}" stroke-width="2"/>
  </g>

  <g filter="url(#sh)">
    <path d="M185 265 C200 310,202 360,194 410 C188 445,188 475,192 500 L165 500 C162 470,160 435,164 405 C170 355,168 312,155 265Z"
          fill="{c('Right Leg')}" stroke="{stroke}" stroke-width="2"/>
  </g>
</svg>
""".strip()


# ============================================================
# Session state defaults
# ============================================================

DEFAULTS = {
    "stage": -1,
    "name": "",
    "past_checkins": [],

    "last_summary": None,
    "last_pain_severity": {},  # dict region -> int

    "feeling_level": None,
    "pain_yesno": None,

    "selected_parts": set(),   # regions chosen today
    "pain_severity": {},       # region -> int
    "pain_reason": {},         # region -> str

    "other_pain_enabled": False,
    "other_pain_location": "",
    "other_pain_severity": None,
    "other_pain_reason": "",

    "symptoms": set(),
    "other_symptom_enabled": False,
    "other_symptom_text": "",

    "submitted": False,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# Deterministic highlight recap
# ============================================================

def make_highlight_recap(last: Dict) -> List[str]:
    bullets: List[str] = []

    fl = last.get("feeling_level")
    pain = last.get("pain")
    pain_locs = last.get("pain_locations", []) or []
    syms = last.get("symptoms", []) or []

    try:
        if fl is not None and str(fl).isdigit() and int(fl) <= 4:
            bullets.append(f"Feeling was low: {int(fl)}/10.")
    except Exception:
        pass

    if pain and pain_locs:
        bullets.append(f"Pain reported in: {', '.join(pain_locs[:3])}{'…' if len(pain_locs) > 3 else ''}.")

    if syms:
        bullets.append(f"Symptoms included: {', '.join(syms[:3])}{'…' if len(syms) > 3 else ''}.")

    if not bullets:
        if fl is not None:
            bullets.append(f"Last time feeling: {fl}/10.")
        else:
            bullets.append("You have a previous check-in on file.")

    return bullets[:3]


# ============================================================
# Helpers for body-map colors and inline logic
# ============================================================

REGIONS = ["Head", "Chest", "Abdomen", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]

GREEN = "#6fd08c"
ORANGE = "#f5a623"
RED = "#e74c3c"
DEFAULT = "#cfd8e6"

def region_color_state(region: str) -> str:
    """
    Returns hex color based on rules:
    - Green: never reported before (not in last visit) and not selected today
    - Orange: reported last visit but unchanged
    - Red: new today OR severity increased by >= 2
    """
    last = st.session_state.last_pain_severity or {}
    selected = region in st.session_state.selected_parts

    if not selected:
        if region in last:
            return ORANGE
        return GREEN

    # selected today
    last_val = int(last.get(region, 0))
    cur = int(st.session_state.pain_severity.get(region, last_val))

    if region not in last:
        return RED  # new pain

    if cur >= last_val + 2:
        return RED  # worsening

    return ORANGE  # same as before


def current_svg_colors() -> Dict[str, str]:
    colors = {}
    for r in REGIONS:
        colors[r] = region_color_state(r)
    return colors


# ============================================================
# UI
# ============================================================

st.title("🩺 Cancer Symptom Check-In")

_init_sheets()
if sheets_init_error:
    st.warning(f"Sheets not ready: {sheets_init_error}")

# ---------------------------
# Stage -1: Name
# ---------------------------
if st.session_state.stage == -1:
    name = st.text_input("Enter your name", value=st.session_state.name)

    if st.button("Start Check-In", type="primary"):
        if name.strip():
            st.session_state.name = name.strip()

            past = load_past_checkins(st.session_state.name)
            st.session_state.past_checkins = past

            if past:
                last = past[-1]
                st.session_state.last_summary = last
                lps = last.get("pain_severity", {})
                st.session_state.last_pain_severity = lps if isinstance(lps, dict) else {}
            else:
                st.session_state.last_summary = None
                st.session_state.last_pain_severity = {}

            st.session_state.stage = 0
            st.rerun()

    st.stop()

stage = st.session_state.stage


# ---------------------------
# Stage 0: Highlight recap + shortcut
# ---------------------------
if stage == 0:
    last = st.session_state.last_summary

    if last:
        st.subheader("Since your last check-in")
        bullets = make_highlight_recap(last)
        ts = last.get("timestamp", "")
        if ts:
            st.caption(f"Last check-in: {ts}")

        for b in bullets:
            st.write(f"• {b}")

        st.markdown("---")

    st.subheader("Is anything changed today?")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Same as yesterday", use_container_width=True, type="primary"):
            payload = {
                "name": st.session_state.name,
                "feeling_level": None,
                "pain": None,
                "pain_locations": [],
                "pain_severity": {},
                "pain_reason": {},
                "symptoms": [],
                "note": "Same as yesterday",
            }
            try:
                save_to_sheet(payload)
            except Exception as e:
                st.warning(f"Could not save: {e}")

            st.session_state.submitted = True
            st.session_state.stage = 5
            st.rerun()

    with c2:
        if st.button("Something changed", use_container_width=True):
            st.session_state.stage = 1
            st.rerun()


# ---------------------------
# Stage 1: Feeling (0–10)
# ---------------------------
elif stage == 1:
    st.subheader("How are you feeling today (0–10)?")
    st.caption("0 = worst, 10 = best")

    default_val = 7
    last = st.session_state.last_summary
    if last and str(last.get("feeling_level", "")).isdigit():
        default_val = int(last["feeling_level"])

    feeling = st.number_input("Feeling (0–10)", min_value=0, max_value=10, value=default_val, step=1)

    if st.button("Next", type="primary"):
        st.session_state.feeling_level = int(feeling)
        st.session_state.stage = 2
        st.rerun()


# ---------------------------
# Stage 2: Pain yes/no
# ---------------------------
elif stage == 2:
    st.subheader("Do you have pain today?")
    pain = st.radio("", ["No", "Yes"], index=0)

    if st.button("Next", type="primary"):
        st.session_state.pain_yesno = (pain == "Yes")
        st.session_state.stage = 3 if st.session_state.pain_yesno else 4
        st.rerun()


# ---------------------------
# Stage 3: Body map + inline per-option UI
# ---------------------------
elif stage == 3:
    st.subheader("Where do you feel pain?")

    col_svg, col_opts = st.columns([1.1, 1], gap="large")

    with col_svg:
        st.markdown(body_svg(current_svg_colors()), unsafe_allow_html=True)
        st.caption("🟢 New area  •  🟠 Same as last visit  •  🔴 New or worsened")

    with col_opts:
        st.write("**Tap a region**")

        last = st.session_state.last_pain_severity or {}

        for r in REGIONS:
            icon = "🟢"
            col_hex = region_color_state(r)
            if col_hex == ORANGE:
                icon = "🟠"
            elif col_hex == RED:
                icon = "🔴"

            clicked = st.button(f"{icon} {r}", key=f"toggle_{r}", use_container_width=True)

            if clicked:
                if r in st.session_state.selected_parts:
                    st.session_state.selected_parts.remove(r)
                    st.session_state.pain_severity.pop(r, None)
                    st.session_state.pain_reason.pop(r, None)
                else:
                    st.session_state.selected_parts.add(r)
                st.rerun()

            # Inline block for THIS option only
            if r in st.session_state.selected_parts:
                last_val = int(last.get(r, 0))
                cur_default = int(st.session_state.pain_severity.get(r, last_val))

                sev = st.number_input(
                    f"{r} severity (0–10)",
                    min_value=0,
                    max_value=10,
                    value=cur_default,
                    step=1,
                    key=f"sev_{r}"
                )
                st.session_state.pain_severity[r] = int(sev)

                if int(sev) > 6 or int(sev) >= last_val + 2:
                    st.warning("What seems to be causing the worsening?")
                    reason = st.text_input("Reason (optional)", key=f"why_{r}")
                    if reason.strip():
                        st.session_state.pain_reason[r] = reason.strip()

                st.markdown("---")

        if st.button("Other location", use_container_width=True):
            st.session_state.other_pain_enabled = True
            st.rerun()

        if st.session_state.other_pain_enabled:
            other = st.text_input("Describe other pain location", value=st.session_state.other_pain_location)
            st.session_state.other_pain_location = other

            if other.strip():
                sev = st.number_input("Severity (0–10)", 0, 10, value=0, step=1, key="other_pain_sev")
                st.session_state.other_pain_severity = int(sev)

                if int(sev) > 6:
                    st.warning("What seems to be causing the worsening?")
                    other_reason = st.text_input("Reason (optional)", key="other_pain_why")
                    st.session_state.other_pain_reason = other_reason.strip()

    if st.button("Next", type="primary"):
        st.session_state.stage = 4
        st.rerun()


# ---------------------------
# Stage 4: Symptoms (clickable)
# ---------------------------
elif stage == 4:
    st.subheader("Symptoms today")

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

    c1, c2 = st.columns(2)

    for i, sym in enumerate(symptom_options):
        col = c1 if i % 2 == 0 else c2
        with col:
            selected = sym in st.session_state.symptoms
            label = f"🔴 {sym}" if selected else f"🟢 {sym}"
            if st.button(label, key=f"sym_{sym}", use_container_width=True):
                if selected:
                    st.session_state.symptoms.remove(sym)
                else:
                    st.session_state.symptoms.add(sym)
                st.rerun()

    st.markdown("---")

    if st.button("Other symptom", use_container_width=True):
        st.session_state.other_symptom_enabled = True
        st.rerun()

    if st.session_state.other_symptom_enabled:
        other = st.text_input("Describe other symptom", value=st.session_state.other_symptom_text)
        st.session_state.other_symptom_text = other
        if other.strip():
            st.session_state.symptoms.add(other.strip())

    st.markdown("---")

    if st.button("Submit Check-In", type="primary"):
        pain_locations = sorted(list(st.session_state.selected_parts))
        pain_severity = dict(st.session_state.pain_severity)
        pain_reason = dict(st.session_state.pain_reason)

        if st.session_state.other_pain_enabled and st.session_state.other_pain_location.strip():
            loc = st.session_state.other_pain_location.strip()
            pain_locations.append(loc)
            if st.session_state.other_pain_severity is not None:
                pain_severity[loc] = int(st.session_state.other_pain_severity)
            if st.session_state.other_pain_reason.strip():
                pain_reason[loc] = st.session_state.other_pain_reason.strip()

        payload = {
            "name": st.session_state.name,
            "feeling_level": st.session_state.feeling_level,
            "pain": bool(st.session_state.pain_yesno),
            "pain_locations": pain_locations,
            "pain_severity": pain_severity,
            "pain_reason": pain_reason,
            "symptoms": sorted(list(st.session_state.symptoms)),
        }

        try:
            save_to_sheet(payload)
        except Exception as e:
            st.warning(f"Could not save: {e}")

        st.session_state.submitted = True
        st.session_state.stage = 5
        st.rerun()


# ---------------------------
# Stage 5: Done
# ---------------------------
elif stage == 5:
    st.success("Check-in complete.")
    st.write("Your care team will review your responses.")

    if st.button("Start another check-in"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()
