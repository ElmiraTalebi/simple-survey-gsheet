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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background: #f4f6fb; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 900px; }

/* Header */
.dash-header { margin-bottom: 1.5rem; }
.dash-header h1 { font-size: 1.5rem; font-weight: 800; color: #0f1d35; margin: 0; }
.dash-header p { font-size: 0.82rem; color: #8a94b0; margin: 2px 0 0; }

/* Search */
.panel {
    background: white; border: 1px solid #e4e9f4;
    border-radius: 14px; padding: 18px 20px; margin-bottom: 18px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}

/* Patient status banner */
.status-banner {
    border-radius: 14px; padding: 20px 24px; margin-bottom: 20px;
    border: 1.5px solid;
}
.status-banner.green  { background:#f0fdf4; border-color:#86efac; }
.status-banner.orange { background:#fff7ed; border-color:#fdba74; }
.status-banner.red    { background:#fff1f2; border-color:#fca5a5; }
.status-title { font-size: 1rem; font-weight: 700; margin: 0 0 6px; }
.status-title.green  { color: #166534; }
.status-title.orange { color: #9a3412; }
.status-title.red    { color: #991b1b; }

/* Change badges */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    border-radius: 20px; padding: 3px 11px; font-size: 0.78rem;
    font-weight: 600; margin: 3px 4px 3px 0; border: 1px solid;
}
.badge.green  { background:#dcfce7; color:#166534; border-color:#86efac; }
.badge.orange { background:#ffedd5; color:#9a3412; border-color:#fdba74; }
.badge.red    { background:#fee2e2; color:#991b1b; border-color:#fca5a5; }
.badge.blue   { background:#eff6ff; color:#1e40af; border-color:#93c5fd; }
.badge.grey   { background:#f3f4f6; color:#6b7280; border-color:#d1d5db; }

/* Visit card */
.visit-card {
    background: white; border: 1px solid #e4e9f4;
    border-radius: 14px; padding: 16px 20px; margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.visit-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 10px;
}
.visit-num {
    background: #1f7aff; color: white; border-radius: 20px;
    padding: 2px 11px; font-size: 0.72rem; font-weight: 700;
    white-space: nowrap;
}
.visit-num.latest { background: #0f1d35; }
.visit-ts {
    font-size: 0.78rem; color: #8a94b0; font-weight: 500;
}
.feeling-pill {
    margin-left: auto; border-radius: 20px; padding: 3px 12px;
    font-size: 0.78rem; font-weight: 700;
}

/* Detail table inside expander */
.detail-table { width:100%; border-collapse:collapse; font-size:0.84rem; }
.detail-table tr { border-bottom:1px solid #f0f2f8; }
.detail-table tr:last-child { border-bottom:none; }
.detail-table td { padding: 8px 6px; vertical-align:top; line-height:1.5; }
.detail-table td:first-child { font-weight:600; color:#64748b; width:35%; white-space:nowrap; }
.detail-table td:last-child { color:#1a2540; }

.tag {
    display:inline-block; background:rgba(31,122,255,0.08); color:#1f5acc;
    border-radius:16px; padding:2px 9px; font-size:0.78rem; margin:2px 3px 2px 0;
}

.no-visits { color:rgba(0,0,0,0.4); font-size:15px; text-align:center; padding:40px 0; }

.stButton>button {
    border-radius:10px !important; font-weight:600 !important;
    background:#1f7aff !important; color:white !important;
    border:none !important; padding:0.45rem 1.4rem !important;
}
.stButton>button:hover { background:#1665d8 !important; }

/* Trend sparkline label */
.trend-label { font-size:0.72rem; color:#8a94b0; margin-bottom:4px; font-weight:500; }
</style>
""", unsafe_allow_html=True)

# ── Feeling colour helper ────────────────────────────────────────────────────
def feeling_color(value) -> tuple:
    try:
        v = int(float(str(value)))
        if v >= 8: return ("#dcfce7", "#166534")
        if v >= 6: return ("#dbeafe", "#1e40af")
        if v >= 4: return ("#fef9c3", "#854d0e")
        return          ("#fee2e2", "#991b1b")
    except Exception:
        labels = {
            "excellent": ("#dcfce7","#166534"), "very good": ("#dbeafe","#1e40af"),
            "good":      ("#e0f2fe","#0369a1"), "fair":      ("#fef9c3","#854d0e"),
            "poor":      ("#fee2e2","#991b1b"),
        }
        return labels.get(str(value).lower(), ("#f3f4f6","#374151"))


# ── Change detection ──────────────────────────────────────────────────────────
def compute_visit_changes(current: Dict, previous: Optional[Dict]) -> List[Dict]:
    """Compare two visits, return list of change dicts with level: green/orange/red."""
    if previous is None:
        return []
    changes = []

    # Feeling level
    try:
        cur_f = int(float(str(current.get("feeling_level", 0))))
        pre_f = int(float(str(previous.get("feeling_level", 0))))
        d = cur_f - pre_f
        if abs(d) >= 1:
            level = "green" if d > 0 else ("red" if d <= -3 else "orange")
            changes.append({"symptom": "Overall feeling", "current": cur_f,
                            "previous": pre_f, "delta": d, "level": level})
    except Exception:
        pass

    # New / resolved pain locations
    cur_locs = set(current.get("pain_locations", []))
    pre_locs = set(previous.get("pain_locations", []))
    for loc in cur_locs - pre_locs:
        changes.append({"symptom": f"New pain: {loc}", "current": "new",
                        "previous": "none", "delta": None, "level": "red"})
    for loc in pre_locs - cur_locs:
        changes.append({"symptom": f"Pain resolved: {loc}", "current": "none",
                        "previous": "present", "delta": None, "level": "green"})

    # Pain severity per region
    cur_sev = current.get("pain_severity", {})
    pre_sev = previous.get("pain_severity", {})
    for region in set(list(cur_sev.keys()) + list(pre_sev.keys())):
        c_v = cur_sev.get(region)
        p_v = pre_sev.get(region)
        if c_v is not None and p_v is not None:
            try:
                d = int(c_v) - int(p_v)
                if abs(d) >= 1:
                    level = "green" if d < 0 else ("red" if d >= 3 else "orange")
                    changes.append({"symptom": f"{region} pain", "current": c_v,
                                    "previous": p_v, "delta": d, "level": level})
            except Exception:
                pass

    # New / resolved symptoms
    cur_syms = set(current.get("symptoms", []))
    pre_syms = set(previous.get("symptoms", []))
    for s in cur_syms - pre_syms:
        changes.append({"symptom": s, "current": "new", "previous": "none",
                        "delta": None, "level": "orange"})
    for s in pre_syms - cur_syms:
        changes.append({"symptom": s, "current": "resolved", "previous": "present",
                        "delta": None, "level": "green"})

    return changes


def overall_status(changes: List[Dict]) -> str:
    if not changes: return "green"
    levels = [c["level"] for c in changes]
    if "red" in levels:    return "red"
    if "orange" in levels: return "orange"
    return "green"


# ── Badge helpers ─────────────────────────────────────────────────────────────
def badge_html(text: str, level: str) -> str:
    icons = {"green": "🟢", "orange": "🟠", "red": "🔴"}
    icon  = icons.get(level, "")
    return f'<span class="badge {level}">{icon} {text}</span>'


def change_badge_html(c: Dict) -> str:
    sym = c["symptom"]
    d   = c["delta"]
    lvl = c["level"]
    cur = c["current"]
    if d is not None:
        arrow = "↑" if d > 0 else "↓"
        label = f"{sym} {arrow}{abs(d)}"
    elif cur == "new":
        label = f"{sym} (new)"
    elif cur in ("resolved", "none"):
        label = f"{sym} resolved"
    else:
        label = sym
    return badge_html(label, lvl)


# ── Patient status summary banner ─────────────────────────────────────────────
def render_summary_status(name: str, latest: Dict, previous: Optional[Dict]):
    changes = compute_visit_changes(latest, previous)
    status  = overall_status(changes)

    titles = {
        "green":  ("✅ Stable",            "No concerning changes since last visit."),
        "orange": ("🟠 Monitor",           "Some symptoms have changed — review below."),
        "red":    ("🔴 Attention needed",  "Significant changes detected. Review urgently."),
    }
    title, subtitle = titles[status]

    badges = "".join(change_badge_html(c) for c in changes) if changes \
             else '<span style="font-size:0.83rem;color:#6b7280;">No changes from previous visit</span>'

    ts      = latest.get("timestamp", "")
    feeling = latest.get("feeling_level", "—")
    f_bg, f_fg = feeling_color(feeling) if feeling != "—" else ("#f3f4f6","#6b7280")

    st.markdown(f"""
<div class="status-banner {status}">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;
              flex-wrap:wrap;gap:12px;">
    <div style="flex:1;min-width:260px;">
      <div class="status-title {status}">{title}</div>
      <div style="font-size:0.82rem;color:#6b7280;margin-bottom:10px;">{subtitle}</div>
      <div>{badges}</div>
    </div>
    <div style="text-align:right;white-space:nowrap;">
      <div style="font-size:0.72rem;color:#9ca3af;margin-bottom:4px;">Latest visit</div>
      <div style="font-size:0.8rem;font-weight:600;color:#374151;">{ts}</div>
      <div style="margin-top:6px;">
        <span style="background:{f_bg};color:{f_fg};border-radius:16px;
                     padding:3px 12px;font-size:0.78rem;font-weight:700;">
          Feeling {feeling}/10
        </span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Visit card ────────────────────────────────────────────────────────────────
def render_visit_card(visit: Dict, visit_num: int, total: int, notes: str,
                      previous: Optional[Dict] = None):
    timestamp = visit.get("timestamp", "Unknown date")
    feeling   = visit.get("feeling_level")
    pain      = visit.get("pain")
    locations = visit.get("pain_locations", [])
    symptoms  = visit.get("symptoms", [])
    is_latest = (visit_num == total)

    changes = compute_visit_changes(visit, previous)
    status  = overall_status(changes)

    num_cls    = "visit-num latest" if is_latest else "visit-num"
    latest_lbl = " · Latest" if is_latest else ""
    f_bg, f_fg = feeling_color(feeling) if feeling is not None else ("#f3f4f6","#6b7280")

    badges = "".join(change_badge_html(c) for c in changes) if changes \
             else '<span style="font-size:0.78rem;color:#9ca3af;">No changes vs prior visit</span>'

    border_c = {"green":"#86efac","orange":"#fdba74","red":"#fca5a5"}.get(status,"#e4e9f4")

    st.markdown(f"""
<div class="visit-card" style="border-left:4px solid {border_c};">
  <div class="visit-header">
    <span class="{num_cls}">Visit {visit_num}{latest_lbl}</span>
    <span class="visit-ts">{timestamp}</span>
    <span class="feeling-pill" style="background:{f_bg};color:{f_fg};">
      Feeling {feeling if feeling is not None else '—'}/10
    </span>
  </div>
  <div style="margin-bottom:2px;">{badges}</div>
</div>
""", unsafe_allow_html=True)

    with st.expander("Show details", expanded=False):
        pain_str = "Yes" if pain is True else ("No" if pain is False else "—")
        loc_html = "".join(f'<span class="tag">{l}</span>' for l in locations) \
                   if locations else "<span style='opacity:.4'>None / N/A</span>"
        sym_html = "".join(f'<span class="tag">{s}</span>' for s in symptoms) \
                   if symptoms else "<span style='opacity:.4'>None reported</span>"

        if notes and notes.strip() and notes != "None":
            lines = [l.lstrip("•-– ").strip() for l in notes.split("\n")
                     if l.strip() and l.strip() != "None"]
            notes_html = ("<ul style='margin:0;padding-left:16px;line-height:1.8;'>" +
                          "".join(f"<li>{l}</li>" for l in lines) + "</ul>")
        else:
            notes_html = "<span style='opacity:.4'>No additional notes</span>"

        followup_qa = visit.get("followup_qa", [])
        if followup_qa:
            fup_items = [
                f"<li><strong>{item.get('question','')}</strong><br>{item.get('answer','')}</li>"
                for item in followup_qa if item.get("answer","").strip()
            ]
            fup_html = ("<ul style='margin:0;padding-left:16px;line-height:1.8;'>" +
                        "".join(fup_items) + "</ul>") if fup_items \
                       else "<span style='opacity:.4'>None</span>"
        else:
            fup_html = "<span style='opacity:.4'>None</span>"

        # Questionnaire scores (flat int values that aren't standard checkin keys)
        SKIP_KEYS = {"timestamp","feeling_level","pain","pain_locations","pain_severity",
                     "pain_reason","symptoms","conversation","followup_qa","__followup__","name"}
        q_answers = {k: v for k, v in visit.items()
                     if k not in SKIP_KEYS and isinstance(v, (int, float))
                     and not k.startswith("_")}

        rows_data = [
            ("Pain reported",    pain_str),
            ("Pain locations",   loc_html),
            ("Symptoms",         sym_html),
            ("Clinical notes",   notes_html),
            ("Follow-up Q&A",    fup_html),
        ]
        if q_answers:
            q_rows = "".join(
                f"<tr><td style='padding:3px 8px;color:#64748b;'>{k}</td>"
                f"<td style='padding:3px 8px;font-weight:600;'>{v}/5</td></tr>"
                for k, v in sorted(q_answers.items())
            )
            rows_data.append(("Questionnaire scores",
                              f"<table style='border-collapse:collapse;width:100%;font-size:0.8rem;'>"
                              f"{q_rows}</table>"))

        table_rows = "".join(f"<tr><td>{r}</td><td>{v}</td></tr>" for r, v in rows_data)
        st.markdown(f"<table class='detail-table'>{table_rows}</table>",
                    unsafe_allow_html=True)


# ── App ───────────────────────────────────────────────────────────────────────
_init_sheets()

st.markdown("""
<div class="dash-header">
  <h1>🏥 Provider Dashboard</h1>
  <p>Head &amp; Neck Cancer Symptom Check-In &nbsp;·&nbsp; Patient Visit History</p>
</div>
""", unsafe_allow_html=True)

if sheets_init_error:
    st.error(f"Google Sheets connection failed: {sheets_init_error}")
    st.stop()

# ── Search ────────────────────────────────────────────────────────────────────
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown("**Search patient by name**")
col_input, col_btn = st.columns([4, 1], gap="small")
with col_input:
    patient_name = st.text_input("", placeholder="Enter patient name…",
                                 label_visibility="collapsed", key="patient_search")
with col_btn:
    search = st.button("Search", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────
if search or st.session_state.get("last_searched"):
    name = patient_name.strip() if search else st.session_state.get("last_searched", "")
    if search and name:
        st.session_state["last_searched"] = name
        st.session_state["visits_cache"]  = None

    if not name:
        st.warning("Please enter a patient name.")
        st.stop()

    if st.session_state.get("visits_cache") is None or \
       st.session_state.get("visits_cache_name") != name:
        with st.spinner(f"Loading visits for **{name}**…"):
            visits = load_all_visits(name)

        if not visits:
            st.markdown(f'<div class="no-visits">No records found for <b>{name}</b>.</div>',
                        unsafe_allow_html=True)
            st.stop()

        notes_list = []
        with st.spinner("Extracting clinical notes…"):
            for v in visits:
                notes_list.append(extract_conversation_notes(v))

        st.session_state["visits_cache"]      = visits
        st.session_state["notes_cache"]       = notes_list
        st.session_state["visits_cache_name"] = name
    else:
        visits     = st.session_state["visits_cache"]
        notes_list = st.session_state["notes_cache"]

    total    = len(visits)
    latest   = visits[-1]
    previous = visits[-2] if total >= 2 else None

    st.markdown(f"### {name} &nbsp;·&nbsp; {total} visit{'s' if total != 1 else ''}",
                unsafe_allow_html=True)

    # ── Status banner ─────────────────────────────────────────────────────────
    render_summary_status(name, latest, previous)

    st.markdown("---")
    st.markdown("#### Visit history")

    # Newest first
    for i, (visit, notes) in enumerate(zip(reversed(visits), reversed(notes_list))):
        visit_num  = total - i
        prev_visit = visits[total - i - 2] if (total - i - 2) >= 0 else None
        render_visit_card(visit, visit_num, total, notes, previous=prev_visit)
