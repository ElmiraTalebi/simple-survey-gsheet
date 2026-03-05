import json
from datetime import datetime
from typing import Dict, List, Optional

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
# Body Map
# ============================================================

def body_svg(colors: Dict[str, str]):
    def c(p): return colors.get(p, "#d5dbe3")

    return f"""
<svg width="260" height="420" viewBox="0 0 300 500">

<circle cx="150" cy="60" r="35" fill="{c('Head')}" stroke="black"/>

<rect x="120" y="100" width="60" height="80" fill="{c('Chest')}" stroke="black"/>
<rect x="120" y="180" width="60" height="70" fill="{c('Abdomen')}" stroke="black"/>

<rect x="70" y="110" width="35" height="110" fill="{c('Left Arm')}" stroke="black"/>
<rect x="195" y="110" width="35" height="110" fill="{c('Right Arm')}" stroke="black"/>

<rect x="125" y="250" width="35" height="150" fill="{c('Left Leg')}" stroke="black"/>
<rect x="160" y="250" width="35" height="150" fill="{c('Right Leg')}" stroke="black"/>

</svg>
"""


# ============================================================
# Colors
# ============================================================

GREEN = "#6fd08c"
ORANGE = "#f5a623"
RED = "#e74c3c"


REGIONS = [
    "Head",
    "Chest",
    "Abdomen",
    "Left Arm",
    "Right Arm",
    "Left Leg",
    "Right Leg"
]


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
# UI
# ============================================================

st.title("🩺 Cancer Symptom Check-In")

# ------------------------------------------------------------
# Stage -1
# ------------------------------------------------------------

if st.session_state.stage == -1:

    name = st.text_input("Enter your name")

    if st.button("Start Check-In"):

        st.session_state.name = name

        past = load_past_checkins(name)
        st.session_state.past_checkins = past

        if past:
            last = past[-1]
            st.session_state.last_summary = last
            st.session_state.last_pain_severity = last.get("pain_severity", {})

        st.session_state.stage = 3
        st.rerun()


# ------------------------------------------------------------
# Stage 3 Pain Map
# ------------------------------------------------------------

elif st.session_state.stage == 3:

    st.subheader("Where do you feel pain?")

    col_map, col_opts = st.columns([1.1, 1])

    with col_map:

        st.markdown(
            body_svg(current_svg_colors()),
            unsafe_allow_html=True
        )

    with col_opts:

        last = st.session_state.last_pain_severity

        for r in REGIONS:

            col = region_color_state(r)

            if col == GREEN:
                icon = "🟢"
            elif col == ORANGE:
                icon = "🟠"
            else:
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

                    st.markdown("**What seems to be causing the worsening?**")

                    reason_options = [
                        "Physical activity / strain",
                        "Treatment side effect",
                        "Sleeping position",
                        "Stress / anxiety",
                        "Unknown",
                        "Other"
                    ]

                    reason = st.radio(
                        "Reason",
                        reason_options,
                        key=f"reason_select_{r}"
                    )

                    if reason == "Other":
                        txt = st.text_input(
                            "Describe reason",
                            key=f"reason_text_{r}"
                        )
                        st.session_state.pain_reason[r] = txt
                    else:
                        st.session_state.pain_reason[r] = reason

    if st.button("Next"):
        st.success("Saved.")
