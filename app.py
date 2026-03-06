import json
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(
    page_title="Symptom Questionnaire",
    page_icon="🏥",
    layout="wide",
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1100px; }

.page-header {
    padding: 1.2rem 1.6rem;
    background: #f8faff;
    border: 1px solid #e2e8f4;
    border-radius: 12px;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 14px;
}
.page-header h1 { font-size: 1.25rem; font-weight: 600; color: #1a2540; margin: 0; }
.page-header p  { font-size: 0.82rem; color: #6b7a99; margin: 2px 0 0 0; }

.name-card {
    max-width: 460px;
    margin: 3rem auto;
    background: #f8faff;
    border: 1px solid #e2e8f4;
    border-radius: 14px;
    padding: 2.2rem 2.4rem;
}
.name-card h2 { font-size: 1.15rem; font-weight: 600; color: #1a2540; margin: 0 0 0.4rem 0; }
.name-card p  { font-size: 0.84rem; color: #6b7a99; margin: 0 0 1.4rem 0; }

.returning-badge {
    display: inline-block; background: #e8f4ee; color: #1a6e40;
    border-radius: 20px; font-size: 0.78rem; font-weight: 500;
    padding: 3px 12px; margin-bottom: 1rem;
}
.firstvisit-badge {
    display: inline-block; background: #e8f0ff; color: #2952cc;
    border-radius: 20px; font-size: 0.78rem; font-weight: 500;
    padding: 3px 12px; margin-bottom: 1rem;
}

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem; font-weight: 500; letter-spacing: 0.12em;
    text-transform: uppercase; color: #8a94b0;
    margin: 1.2rem 0 0.5rem 0; padding-bottom: 4px;
    border-bottom: 1px solid #edf0f7;
}

div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem; font-weight: 500;
    min-height: 42px; min-width: 42px; padding: 0;
    border-radius: 8px; border: 1.5px solid #dde3f0;
    background: #f8faff; color: #4a5578;
    transition: all 0.15s ease; width: 100%;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {
    border-color: #6a8fff; background: #eef3ff; color: #2952cc;
}
/* Selected button — label starts with ● */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[title^="●"],
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"]:has(p:-webkit-any(p)) > button {
    background: #f8faff; color: #4a5578;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button p {
    margin: 0; line-height: 1;
}

.q-label { font-size: 0.92rem; font-weight: 400; color: #1e2d50; line-height: 1.3; padding: 4px 0; }

.scale-header { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #aab3cc; text-align: center; }

.score-badge {
    display: inline-block; font-family: 'DM Mono', monospace;
    font-size: 0.72rem; font-weight: 500;
    background: #e8f0ff; color: #3560e0;
    border-radius: 20px; padding: 1px 8px; margin-left: 8px;
}
.score-badge-warn  { background: #fff3e0; color: #d07000; }
.score-badge-alert { background: #ffeaea; color: #c0392b; }
.no-selection {
    display: inline-block; font-family: 'DM Mono', monospace;
    font-size: 0.72rem; background: #f0f2f8; color: #aab3cc;
    border-radius: 20px; padding: 1px 8px; margin-left: 8px;
}

div.submit-btn > button {
    background: #2952cc !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-size: 1rem !important; font-weight: 600 !important;
    min-height: 48px !important; width: 100% !important;
}
div.submit-btn > button:hover { background: #1e3fa8 !important; }

.followup-card {
    background: #fff8ec; border: 1px solid #ffd28a;
    border-radius: 12px; padding: 18px 20px; margin: 1rem 0;
}
.followup-card h4 { color: #a05e00; margin: 0 0 8px 0; font-size: 0.95rem; }

.success-card {
    background: #f0fff6; border: 1px solid #7ed9a4;
    border-radius: 12px; padding: 22px; text-align: center; margin-top: 1rem;
}
.success-card h3 { color: #1a6e40; margin: 0 0 6px 0; }
.success-card p  { color: #3a8c5c; font-size: 0.88rem; margin: 0; }

.thin-divider { border: none; border-top: 1px solid #edf0f7; margin: 0.3rem 0; }
.delta-up   { color: #c0392b; font-size: 0.75rem; margin-left: 4px; }
.delta-down { color: #1a6e40; font-size: 0.75rem; margin-left: 4px; }
.delta-same { color: #8a94b0; font-size: 0.75rem; margin-left: 4px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Questions
# ============================================================

SECTIONS = {
    "Pain": [
        "Pain in throat / mouth",
        "Pain when swallowing",
        "Jaw pain or tightness",
        "Ear pain",
        "Neck pain or stiffness",
    ],
    "Swallowing & Eating": [
        "Difficulty swallowing solids",
        "Difficulty swallowing liquids",
        "Choking or coughing while eating",
        "Loss of appetite",
        "Unintended weight loss",
    ],
    "Mouth & Saliva": [
        "Dry mouth (xerostomia)",
        "Thick or sticky saliva",
        "Mouth sores or ulcers",
        "Difficulty opening mouth (trismus)",
        "Change in taste",
    ],
    "Voice & Breathing": [
        "Hoarseness or voice changes",
        "Shortness of breath",
        "Coughing",
        "Mucus / phlegm buildup",
    ],
    "Skin & Appearance": [
        "Skin irritation or redness (radiation area)",
        "Skin peeling or blistering",
        "Swelling in face or neck",
    ],
    "Energy & Mood": [
        "Fatigue / tiredness",
        "Difficulty sleeping",
        "Anxiety or worry",
        "Low mood or sadness",
        "Difficulty concentrating",
    ],
    "Overall": [
        "Overall discomfort today",
    ],
}

ALL_QUESTIONS = [q for qs in SECTIONS.values() for q in qs]


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
        raise KeyError(f"Missing secret: {', '.join(keys)}")
    return v


# ============================================================
# Google Sheets
# ============================================================

_sheet_cache: Dict = {}

def _get_sheet():
    if "ws" in _sheet_cache:
        return _sheet_cache["ws"]
    try:
        creds = Credentials.from_service_account_info(
            _require_secret("gcp_service_account"),
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        book = gspread.authorize(creds).open_by_key(_require_secret("gsheet_id"))
        try:
            ws = book.worksheet("Form")
        except Exception:
            try:
                ws = book.worksheet("Questionnaire")
            except Exception:
                ws = book.add_worksheet(title="Form", rows=2000, cols=20)
                ws.append_row(["timestamp", "name", "chat"])
        _sheet_cache["ws"] = ws
        return ws
    except Exception as e:
        st.session_state["_sheets_error"] = str(e)
        return None


def load_previous_answers(name: str) -> Optional[Dict[str, int]]:
    """Return the most recent questionnaire answers for this patient, or None."""
    ws = _get_sheet()
    if ws is None:
        return None
    try:
        rows = ws.get_all_values()
        last_row = None
        for row in rows[1:]:
            if len(row) >= 3 and row[1].strip().lower() == name.strip().lower():
                last_row = row
        if last_row is None:
            return None
        data = json.loads(last_row[2])
        # Support both flat format {question: score} and wrapped {"answers": {...}}
        if "answers" in data and isinstance(data["answers"], dict):
            data = data["answers"]
        if not any(q in data for q in ALL_QUESTIONS):
            return None
        return {q: int(data[q]) for q in ALL_QUESTIONS if q in data}
    except Exception:
        return None


def save_answers(name: str, answers: Dict[str, int], followup_text: str):
    ws = _get_sheet()
    if ws is None:
        return
    # Store questions dict directly (compatible with existing sheet format)
    payload = dict(answers)
    if followup_text:
        payload["__followup__"] = followup_text
    ws.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        name,
        json.dumps(payload),
    ])


# ============================================================
# Session state
# ============================================================

DEFAULTS = {
    "q_stage": "name",         # "name" | "form" | "followup" | "saving" | "done"
    "patient_name": "",
    "previous_answers": None,  # Dict[str,int] or None (None = first visit)
    "answers": {},             # current; absent key = not yet selected
    "followup_needed": [],
    "followup_text": "",
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# Helpers
# ============================================================

def score_class(score: int) -> str:
    if score >= 4: return "score-badge-alert"
    if score >= 2: return "score-badge-warn"
    return "score-badge"

def delta_html(current: int, previous: Optional[int]) -> str:
    if previous is None: return ""
    diff = current - previous
    if diff > 0: return f'<span class="delta-up">▲{diff}</span>'
    if diff < 0: return f'<span class="delta-down">▼{abs(diff)}</span>'
    return '<span class="delta-same">—</span>'

def check_followup(current: Dict[str, int], previous: Optional[Dict[str, int]]):
    flagged = []
    for q, score in current.items():
        prev = (previous or {}).get(q, 0)
        if score >= 5 or (score - prev) >= 3:
            flagged.append((q, score, prev))
    return flagged


# ============================================================
# Page header
# ============================================================

is_returning = st.session_state.previous_answers is not None
sub = (
    "Pre-filled from your last visit · Change any value or keep as is · Under 30 seconds"
    if is_returning else
    "Rate each symptom from 0 (none) to 5 (severe) · Under 30 seconds"
)
st.markdown(f"""
<div class="page-header">
    <div style="font-size:2rem; line-height:1;">🏥</div>
    <div>
        <h1>Head &amp; Neck Cancer — Symptom Check</h1>
        <p>{sub}</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# STAGE: Name entry
# ============================================================

if st.session_state.q_stage == "name":

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("""
        <div class="name-card">
            <h2>Welcome</h2>
            <p>Please enter your name. We'll load your previous answers if available,
               so you only need to update anything that has changed.</p>
        </div>
        """, unsafe_allow_html=True)

        name = st.text_input("Your full name", placeholder="e.g. Jane Smith", key="name_input")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Continue →", key="name_btn"):
            if not name.strip():
                st.warning("Please enter your name.")
            else:
                with st.spinner("Looking up your records…"):
                    prev = load_previous_answers(name.strip())
                st.session_state.patient_name    = name.strip()
                st.session_state.previous_answers = prev
                # Pre-populate if returning; empty dict if first visit
                st.session_state.answers = dict(prev) if prev else {}
                st.session_state.q_stage = "form"
                st.rerun()

    st.stop()


# ============================================================
# STAGE: Questionnaire form
# ============================================================

elif st.session_state.q_stage == "form":

    prev_answers = st.session_state.previous_answers

    if prev_answers is not None:
        st.markdown(
            '<div class="returning-badge">✓ Returning patient · '
            'Pre-filled from last visit · Change anything that has changed</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="firstvisit-badge">✦ First visit · '
            'Please rate each symptom below</div>',
            unsafe_allow_html=True,
        )

    # Scale header
    hc_q, *hc_s = st.columns([6, 1, 1, 1, 1, 1, 1])
    hc_q.markdown(
        '<div class="scale-header" style="text-align:left; font-size:0.72rem;">SYMPTOM</div>',
        unsafe_allow_html=True,
    )
    for i, col in enumerate(hc_s):
        lbl = ["None", "Mild", "Mild+", "Mod", "Severe", "Max"][i]
        col.markdown(
            f'<div class="scale-header">{i}<br>'
            f'<span style="font-size:0.6rem; color:#c0c8de;">{lbl}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    # JS observer that highlights any score button whose label contains ●
    components.html("""
    <script>
    (function() {
        function highlight() {
            const btns = window.parent.document.querySelectorAll(
                'div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button'
            );
            btns.forEach(btn => {
                const txt = (btn.innerText || btn.textContent || "").trim();
                if (txt.startsWith("●")) {
                    btn.style.cssText = "background:#2952cc !important; color:white !important; border-color:#2952cc !important; font-weight:700 !important;";
                } else {
                    if (btn.style.background.includes("41, 82") || btn.style.background === "#2952cc") {
                        btn.style.cssText = "";
                    }
                }
            });
        }
        highlight();
        new MutationObserver(highlight).observe(
            window.parent.document.body, { childList: true, subtree: true }
        );
    })();
    </script>
    """, height=0)

    for section, questions in SECTIONS.items():
        st.markdown(f'<div class="section-label">{section}</div>', unsafe_allow_html=True)

        for q in questions:
            current_val = st.session_state.answers.get(q)

            col_q, *col_s = st.columns([6, 1, 1, 1, 1, 1, 1])

            with col_q:
                st.markdown(f'<div class="q-label">{q}</div>', unsafe_allow_html=True)

            for i, col in enumerate(col_s):
                selected = (current_val == i)
                label = f"● {i}" if selected else str(i)
                if col.button(label, key=f"q_{q}_{i}"):
                    st.session_state.answers[q] = i
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    unanswered = [q for q in ALL_QUESTIONS if q not in st.session_state.answers]
    if unanswered:
        st.markdown(
            f'<div style="font-size:0.82rem; color:#8a94b0; margin-bottom:0.5rem;">'
            f'{len(unanswered)} symptom(s) not yet rated</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="submit-btn">', unsafe_allow_html=True)
    if st.button("Submit questionnaire →", key="submit_btn"):
        if unanswered:
            st.warning(f"Please rate all {len(unanswered)} remaining symptom(s) before submitting.")
        else:
            flagged = check_followup(st.session_state.answers, prev_answers)
            st.session_state.followup_needed = flagged
            st.session_state.q_stage = "followup" if flagged else "saving"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# STAGE: Follow-up
# ============================================================

elif st.session_state.q_stage == "followup":

    st.markdown("""
    <div class="followup-card">
        <h4>⚠️ One quick follow-up</h4>
        <p style="font-size:0.85rem; color:#7a5000; margin:0;">
            One or more symptoms are significantly higher than your last visit.
            A brief note helps your care team prepare.
        </p>
    </div>
    """, unsafe_allow_html=True)

    for q, score, prev in st.session_state.followup_needed:
        diff   = score - prev
        reason = "scored maximum (5/5)" if score >= 5 else f"increased by {diff} ({prev} → {score})"
        st.markdown(
            f'<div style="font-size:0.85rem; color:#1e2d50; padding:3px 0;">'
            f'<strong>{q}</strong> — {reason}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    followup_text = st.text_area(
        "Can you briefly describe what changed?",
        value=st.session_state.followup_text,
        height=90,
        placeholder="e.g. Throat pain started after eating, worse in the morning…",
        key="followup_input",
    )
    st.session_state.followup_text = followup_text

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Submit with my notes →", key="fup_submit"):
            st.session_state.q_stage = "saving"
            st.rerun()
    with c2:
        if st.button("Skip and submit →", key="fup_skip"):
            st.session_state.q_stage = "saving"
            st.rerun()


# ============================================================
# STAGE: Saving
# ============================================================

elif st.session_state.q_stage == "saving":

    with st.spinner("Saving your responses…"):
        save_answers(
            st.session_state.patient_name,
            st.session_state.answers,
            st.session_state.followup_text,
        )
    st.session_state.q_stage = "done"
    st.rerun()


# ============================================================
# STAGE: Done / Summary
# ============================================================

elif st.session_state.q_stage == "done":

    st.markdown(f"""
    <div class="success-card">
        <h3>✅ Questionnaire submitted</h3>
        <p>Thank you, <strong>{st.session_state.patient_name}</strong>.
           Your care team will review your responses.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Your responses — summary</div>', unsafe_allow_html=True)

    hq, hp, hc, hd = st.columns([5, 1, 1, 1])
    hq.markdown('<div class="scale-header" style="text-align:left;">Symptom</div>', unsafe_allow_html=True)
    hp.markdown('<div class="scale-header">Prev</div>', unsafe_allow_html=True)
    hc.markdown('<div class="scale-header">Now</div>', unsafe_allow_html=True)
    hd.markdown('<div class="scale-header">Δ</div>', unsafe_allow_html=True)
    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

    prev_answers = st.session_state.previous_answers or {}

    for q in ALL_QUESTIONS:
        curr = st.session_state.answers.get(q, 0)
        prev = prev_answers.get(q)
        diff = (curr - prev) if prev is not None else None

        if diff is None:
            chg = '<span style="color:#aab3cc; font-size:0.82rem;">—</span>'
        elif diff > 0:
            chg = f'<span style="color:#c0392b; font-size:0.82rem;">▲{diff}</span>'
        elif diff < 0:
            chg = f'<span style="color:#1a6e40; font-size:0.82rem;">▼{abs(diff)}</span>'
        else:
            chg = '<span style="color:#aab3cc; font-size:0.82rem;">—</span>'

        c1, c2, c3, c4 = st.columns([5, 1, 1, 1])
        c1.markdown(f'<div style="font-size:0.85rem; color:#1e2d50; padding:2px 0;">{q}</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="scale-header" style="color:#8a94b0;">{"—" if prev is None else prev}</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="scale-header"><span class="{score_class(curr)} score-badge">{curr}</span></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="scale-header">{chg}</div>', unsafe_allow_html=True)

    if st.session_state.followup_text:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Your notes</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:#f8faff; border:1px solid #e2e8f4; border-radius:8px; '
            f'padding:12px 16px; font-size:0.88rem; color:#1e2d50;">'
            f'{st.session_state.followup_text}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← New check-in", key="restart"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()
