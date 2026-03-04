import json
from typing import Dict, List, Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI

st.set_page_config(page_title="Provider Dashboard", page_icon="🏥", layout="centered")

# ── Secrets ────────────────────────────────────────────────
def _secret(*keys, default=None):
    for k in keys:
        if k in st.secrets: return st.secrets[k]
    return default

def _require_secret(*keys):
    v = _secret(*keys)
    if v is None: raise KeyError(f"Missing secret. Tried: {', '.join(keys)}")
    return v

# ── OpenAI (for conversation note extraction) ───────────────
OPENAI_API_KEY = _secret("openai_api_key", "OPENAI_API_KEY", "openai_key")
openai_client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    try: openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except: pass

def _openai_ready():
    return openai_client is not None

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

def load_all_visits(name: str) -> List[Dict]:
    """Load ALL visits for a patient (not capped at 5), oldest first."""
    _init_sheets()
    if sheet is None: return []
    try:
        visits = []
        for row in sheet.get_all_values()[1:]:
            if len(row) >= 3 and row[1].strip().lower() == name.strip().lower():
                try:
                    d = json.loads(row[2])
                    d["timestamp"] = row[0]
                    visits.append(d)
                except: continue
        return visits  # oldest → newest
    except: return []

def extract_conversation_notes(visit: Dict) -> str:
    """Use GPT to extract clinical notes from free-text patient messages in a visit."""
    if not _openai_ready():
        return ""
    messages = visit.get("conversation", [])
    feeling   = visit.get("feeling_level")
    locations = visit.get("pain_locations", [])
    symptoms  = visit.get("symptoms", [])

    # Build set of auto-generated widget messages to exclude
    widget_msgs = {
        f"My feeling level today is {feeling}/10.",
        f"I'm feeling {feeling} today.",
        "Yes, I have pain today.",
        "No, I don't have any pain today.",
    }
    if locations:
        widget_msgs.add(f"Pain locations: {', '.join(sorted(locations))}.")
    if symptoms:
        widget_msgs.add(f"Symptoms today: {'; '.join(symptoms)}.")

    patient_lines = [
        m.get("content", "") for m in messages
        if m.get("role") == "patient" and m.get("content", "") not in widget_msgs
    ]
    if not patient_lines:
        return ""
    try:
        r = openai_client.chat.completions.create(
            model=_secret("openai_model", default="gpt-4o-mini"),
            messages=[
                {"role": "system", "content": (
                    "Clinical notes assistant. Extract ONLY medically relevant facts from the "
                    "patient's free-text messages: pain details, severity, duration, triggers, "
                    "mood, appetite, sleep, energy. One bullet per fact. No greetings or filler. "
                    "If nothing clinically relevant, reply: None"
                )},
                {"role": "user", "content": "\n".join(f"- {l}" for l in patient_lines)}
            ], max_tokens=300, temperature=0.2,
        )
        result = (r.choices[0].message.content or "").strip()
        return "" if result == "None" else result
    except:
        return ""

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{ background:linear-gradient(135deg,#f0f4ff,#f8faff); }
.header{ font-size:26px; font-weight:800; color:#1a2540; margin-bottom:4px; }
.subheader{ font-size:14px; color:rgba(0,0,0,0.45); margin-bottom:24px; }
.visit-card{
    background:white; border:1.5px solid rgba(200,215,240,0.7);
    border-radius:18px; padding:22px 24px 18px; margin-bottom:20px;
    box-shadow:0 2px 14px rgba(31,122,255,0.07);
}
.visit-date{
    font-size:13px; font-weight:700; letter-spacing:0.06em; text-transform:uppercase;
    color:#1f7aff; margin-bottom:14px;
}
.visit-number{
    display:inline-block; background:#1f7aff; color:white;
    border-radius:20px; padding:2px 12px; font-size:12px;
    font-weight:700; margin-right:10px;
}
.summary-table{ width:100%; border-collapse:collapse; font-size:14px; }
.summary-table tr{ border-bottom:1px solid rgba(200,215,240,0.5); }
.summary-table tr:last-child{ border-bottom:none; }
.summary-table td{ padding:9px 8px; vertical-align:top; line-height:1.5; }
.summary-table td:first-child{ font-weight:600; color:#4a6080; width:38%; white-space:nowrap; }
.summary-table td:last-child{ color:#1a2540; }
.tag{
    display:inline-block; background:rgba(31,122,255,0.09); color:#1f5acc;
    border-radius:20px; padding:2px 10px; font-size:13px; margin:2px 3px 2px 0;
}
.tag-feeling{
    display:inline-block; border-radius:20px; padding:3px 14px;
    font-size:13px; font-weight:700; margin-right:4px;
}
.panel{
    background:rgba(255,255,255,0.75); border:1px solid rgba(200,215,240,0.6);
    border-radius:16px; padding:20px; margin-bottom:16px;
    backdrop-filter:blur(8px);
}
.no-visits{ color:rgba(0,0,0,0.4); font-size:15px; text-align:center; padding:40px 0; }
.stButton>button{
    border-radius:12px !important; font-weight:600 !important;
    background:#1f7aff !important; color:white !important;
    border:none !important; padding:0.5rem 1.5rem !important;
}
.stButton>button:hover{ background:#1665d8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Feeling colour helper ───────────────────────────────────
def feeling_tag(value) -> str:
    if value is None: return "<span style='opacity:.4'>—</span>"
    colours = {
        "excellent": ("#d1fae5","#065f46"),
        "very good": ("#dbeafe","#1e40af"),
        "good":      ("#e0f2fe","#0369a1"),
        "fair":      ("#fef9c3","#854d0e"),
        "poor":      ("#fee2e2","#991b1b"),
    }
    v = str(value).lower()
    bg, fg = colours.get(v, ("#f3f4f6","#374151"))
    return f"<span class='tag-feeling' style='background:{bg};color:{fg};'>{value}</span>"

def render_visit_card(visit: Dict, visit_num: int, total: int, notes: str):
    timestamp = visit.get("timestamp", "Unknown date")
    feeling   = visit.get("feeling_level")
    pain      = visit.get("pain")
    locations = visit.get("pain_locations", [])
    symptoms  = visit.get("symptoms", [])

    pain_str = "Yes" if pain is True else ("No" if pain is False else "—")
    loc_html = "".join(f'<span class="tag">{l}</span>' for l in locations) \
               if locations else "<span style='opacity:.4'>None / N/A</span>"
    sym_html = "".join(f'<span class="tag">{s}</span>' for s in symptoms) \
               if symptoms else "<span style='opacity:.4'>None reported</span>"

    if notes and notes != "None":
        lines = [l.lstrip("•-– ").strip() for l in notes.split("\n")
                 if l.strip() and l.strip() != "None"]
        notes_html = "<ul style='margin:0;padding-left:16px;'>" + "".join(
            f"<li style='margin-bottom:3px;'>{l}</li>" for l in lines
        ) + "</ul>"
    else:
        notes_html = "<span style='opacity:.4'>No additional details</span>"

    st.markdown(f"""
<div class="visit-card">
  <div class="visit-date">
    <span class="visit-number">Visit {visit_num} of {total}</span>
    {timestamp}
  </div>
  <table class="summary-table">
    <tr><td>Overall feeling</td><td>{feeling_tag(feeling)}</td></tr>
    <tr><td>Pain today</td><td>{pain_str}</td></tr>
    <tr><td>Pain locations</td><td>{loc_html}</td></tr>
    <tr><td>Symptoms</td><td>{sym_html}</td></tr>
    <tr><td>Conversation notes</td><td>{notes_html}</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

# ── App ──────────────────────────────────────────────────────
_init_sheets()

st.markdown('<div class="header">🏥 Provider Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Cancer Symptom Check-In · Patient Visit History</div>',
            unsafe_allow_html=True)

if sheets_init_error:
    st.error(f"Google Sheets connection failed: {sheets_init_error}")
    st.stop()

# ── Search panel ─────────────────────────────────────────────
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("**Search patient by name**")
col_input, col_btn = st.columns([4, 1], gap="small")
with col_input:
    patient_name = st.text_input("", placeholder="Enter patient name…",
                                 label_visibility="collapsed", key="patient_search")
with col_btn:
    search = st.button("Search", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────
if search or st.session_state.get("last_searched"):
    name = patient_name.strip() if search else st.session_state.get("last_searched", "")
    if search and name:
        st.session_state["last_searched"] = name
        st.session_state["visits_cache"] = None  # clear cache on new search

    if not name:
        st.warning("Please enter a patient name.")
        st.stop()

    # Load visits (cache in session so GPT extraction doesn't re-run on reruns)
    if st.session_state.get("visits_cache") is None or \
       st.session_state.get("visits_cache_name") != name:
        with st.spinner(f"Loading visits for **{name}**…"):
            visits = load_all_visits(name)

        if not visits:
            st.markdown(f'<div class="no-visits">No check-in records found for <b>{name}</b>.</div>',
                        unsafe_allow_html=True)
            st.stop()

        # Extract conversation notes for each visit via GPT
        notes_list = []
        with st.spinner("Extracting clinical notes…"):
            for v in visits:
                notes_list.append(extract_conversation_notes(v))

        st.session_state["visits_cache"] = visits
        st.session_state["notes_cache"]  = notes_list
        st.session_state["visits_cache_name"] = name
    else:
        visits     = st.session_state["visits_cache"]
        notes_list = st.session_state["notes_cache"]

    total = len(visits)
    st.markdown(f"### {total} visit{'s' if total != 1 else ''} found for **{name}**")
    st.markdown("---")

    # Show newest visit first
    for i, (visit, notes) in enumerate(zip(reversed(visits), reversed(notes_list))):
        render_visit_card(visit, total - i, total, notes)
