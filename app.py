import json
from datetime import datetime
from typing import Dict, List, Optional, Set

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ============================================================
# Secrets helper
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
            sheet_local = book.worksheet("Form")
        except Exception:
            sheet_local = book.add_worksheet(title="Form", rows=2000, cols=20)
            sheet_local.append_row(["timestamp", "name", "json"])
        sheet = sheet_local
    except Exception as e:
        sheets_init_error = str(e)

def load_past_checkins(name: str) -> List[Dict]:
    _init_sheets()
    if sheet is None:
        return []
    try:
        past: List[Dict] = []
        for row in sheet.get_all_values()[1:]:
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
# Body map SVG (from your original version, with fill-on-select)
# ============================================================

def body_svg(selected: Set[str]) -> str:
    # Same style/colors as your original file
    def fill(p): 
        return "#1f7aff" if p in selected else "#cfd8e6"
    s = "#6b7a90"
    return f"""<svg width="240" height="390" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="sh">
      <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="rgba(0,0,0,0.12)"/>
    </filter>
  </defs>

  <g filter="url(#sh)">
    <circle cx="160" cy="70" r="38" fill="{fill('Head')}" stroke="{s}" stroke-width="2"/>
  </g>

  <g filter="url(#sh)">
    <rect x="110" y="120" width="100" height="70" rx="24" fill="{fill('Chest')}" stroke="{s}" stroke-width="2"/>
  </g>

  <g filter="url(#sh)">
    <rect x="115" y="195" width="90" height="70" rx="22" fill="{fill('Abdomen')}" stroke="{s}" stroke-width="2"/>
  </g>

  <g filter="url(#sh)">
    <path d="M110 132 C80 145,72 180,78 220 C82 250,92 270,100 290 C108 310,115 320,120 320 L120 130Z"
          fill="{fill('Left Arm')}" stroke="{s}" stroke-width="2"/>
  </g>

  <g filter="url(#sh)">
    <path d="M210 132 C240 145,248 180,242 220 C238 250,228 270,220 290 C212 310,205 320,200 320 L200 130Z"
          fill="{fill('Right Arm')}" stroke="{s}" stroke-width="2"/>
  </g>

  <g filter="url(#sh)">
    <path d="M135 265 C120 310,118 360,126 410 C132 445,132 475,128 500 L155 500 C158 470,160 435,156 405 C150 355,152 312,165 265Z"
          fill="{fill('Left Leg')}" stroke="{s}" stroke-width="2"/>
  </g>

  <g filter="url(#sh)">
    <path d="M185 265 C200 310,202 360,194 410 C188 445,188 475,192 500 L165 500 C162 470,160 435,164 405 C170 355,168 312,155 265Z"
          fill="{fill('Right Leg')}" stroke="{s}" stroke-width="2"/>
  </g>
</svg>""".strip()


# ============================================================
# Session state
# ============================================================

DEFAULTS = {
    "stage": -1,
    "name": "",
    "past_checkins": [],

    # last-visit memory (deterministic)
    "last_summary": None,
    "last_pain_severity": {},

    # today's responses
    "feeling_level": None,          # int 0-10
    "pain_yesno": None,             # bool
    "selected_parts": set(),         # body regions
    "pain_severity": {},            # dict region->int
    "pain_reason": {},              # dict region->str
    "other_pain_enabled": False,
    "other_pain_location": "",
    "other_pain_severity": None,
    "other_pain_reason": "",

    "symptoms": set(),              # selected symptoms
    "other_symptom_enabled": False,
    "other_symptom_text": "",

    "submitted": False,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# Deterministic highlight recap (only important items)
# ============================================================

def make_highlight_recap(last: Dict) -> List[str]:
    """
    Return 1-3 short highlight bullets (important only).
    """
    bullets: List[str] = []

    fl = last.get("feeling_level")
    pain = last.get("pain")
    pain_locs = last.get("pain_locations", []) or []
    syms = last.get("symptoms", []) or []

    # 1) Low feeling is important
    try:
        if fl is not None and str(fl).isdigit() and int(fl) <= 4:
            bullets.append(f"Feeling was low: {int(fl)}/10.")
    except Exception:
        pass

    # 2) Pain (if present)
    if pain and pain_locs:
        bullets.append(f"Pain reported in: {', '.join(pain_locs[:3])}{'…' if len(pain_locs) > 3 else ''}.")

    # 3) Symptoms (if any)
    if syms:
        bullets.append(f"Symptoms included: {', '.join(syms[:3])}{'…' if len(syms) > 3 else ''}.")

    # If nothing “important”, keep one neutral anchor
    if not bullets:
        if fl is not None:
            bullets.append(f"Last time feeling: {fl}/10.")
        else:
            bullets.append("You have a previous check-in on file.")

    return bullets[:3]


# ============================================================
# UI
# ============================================================

st.title("🩺 Cancer Symptom Check-In")

# Warn if sheets not ready (but don’t crash)
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
                # last visit pain severity mapping (if present)
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
# Stage 0: Highlight recap + same-as-yesterday shortcut
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
            # Save a minimal “no change” check-in (still deterministic)
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
                st.warning(f"Could not save (Sheets issue): {e}")

            st.session_state.submitted = True
            st.session_state.stage = 5
            st.rerun()

    with c2:
        if st.button("Something changed", use_container_width=True):
            st.session_state.stage = 1
            st.rerun()


# ---------------------------
# Stage 1: Feeling (0-10)
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
# Stage 3: Body map (pretty SVG) + mark-on-select + per-option bubbles
# ---------------------------
elif stage == 3:
    st.subheader("Where do you feel pain?")

    parts = ["Head", "Chest", "Abdomen", "Left Arm", "Right Arm", "Left Leg", "Right Leg"]

    col_svg, col_btns = st.columns([1.05, 1], gap="large")

    with col_svg:
        st.markdown(body_svg(st.session_state.selected_parts), unsafe_allow_html=True)
        st.caption("Selected areas turn blue on the body map.")

    with col_btns:
        st.write("**Tap a region**")
        for part in parts:
            selected = part in st.session_state.selected_parts
            label = f"🔴 {part}" if selected else f"🟢 {part}"

            if st.button(label, key=f"toggle_{part}", use_container_width=True):
                if selected:
                    st.session_state.selected_parts.remove(part)
                    # also clean stored details
                    st.session_state.pain_severity.pop(part, None)
                    st.session_state.pain_reason.pop(part, None)
                else:
                    st.session_state.selected_parts.add(part)
                st.rerun()

        st.markdown("---")

        # Per-selected-region “bubble” questions under THAT region (deterministic)
        for part in parts:
            if part not in st.session_state.selected_parts:
                continue

            st.info(f"**{part}:** How severe is it (0–10)?")

            # last visit severity for that same region
            last_val = 0
            if isinstance(st.session_state.last_pain_severity, dict):
                try:
                    last_val = int(st.session_state.last_pain_severity.get(part, 0))
                except Exception:
                    last_val = 0

            sev_key = f"sev_{part}"
            if sev_key not in st.session_state:
                st.session_state[sev_key] = last_val

            severity = st.number_input(
                f"{part} severity",
                min_value=0,
                max_value=10,
                value=int(st.session_state[sev_key]),
                step=1,
                key=sev_key
            )
            st.session_state.pain_severity[part] = int(severity)

            # Follow-up only if concerning: >6 OR worse than last time
            if int(severity) > 6 or int(severity) > int(last_val):
                st.warning("What seems to be causing the worsening?")
                reason = st.text_input("Reason (optional)", key=f"why_{part}")
                if reason.strip():
                    st.session_state.pain_reason[part] = reason.strip()

            st.markdown("---")

        # "Other" pain location hidden until clicked
        if st.button("Other location", use_container_width=True):
            st.session_state.other_pain_enabled = True
            st.rerun()

        if st.session_state.other_pain_enabled:
            other_loc = st.text_input("Describe other pain location", value=st.session_state.other_pain_location)
            st.session_state.other_pain_location = other_loc

            if other_loc.strip():
                # severity
                default_other = 0
                other_sev = st.number_input("Severity (0–10)", 0, 10, value=default_other, step=1, key="other_pain_sev")
                st.session_state.other_pain_severity = int(other_sev)

                # follow-up only if concerning
                if int(other_sev) > 6:
                    st.warning("What seems to be causing the worsening?")
                    other_reason = st.text_input("Reason (optional)", key="other_pain_why")
                    st.session_state.other_pain_reason = other_reason.strip()

    if st.button("Next", type="primary"):
        st.session_state.stage = 4
        st.rerun()


# ---------------------------
# Stage 4: Symptoms clickable (like before) + Other button
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

    c1, c2 = st.columns(2, gap="medium")

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
        other_sym = st.text_input("Describe other symptom", value=st.session_state.other_symptom_text)
        st.session_state.other_symptom_text = other_sym
        if other_sym.strip():
            st.session_state.symptoms.add(other_sym.strip())

    st.markdown("---")

    if st.button("Submit Check-In", type="primary"):
        pain_locations = sorted(list(st.session_state.selected_parts))
        pain_severity = dict(st.session_state.pain_severity)
        pain_reason = dict(st.session_state.pain_reason)

        # include "Other pain location" if used
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
            st.warning(f"Could not save (Sheets issue): {e}")

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
        # reset everything
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()
