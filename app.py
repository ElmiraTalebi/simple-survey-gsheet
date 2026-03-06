import json
from datetime import datetime
from typing import Dict, Optional

import streamlit as st

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

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 1100px;
}

/* Header */
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
.page-header h1 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1a2540;
    margin: 0;
}
.page-header p {
    font-size: 0.82rem;
    color: #6b7a99;
    margin: 2px 0 0 0;
}

/* Section labels */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8a94b0;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #edf0f7;
}

/* Score button row */
.score-row {
    display: flex;
    align-items: center;
    padding: 5px 0;
    gap: 0;
}

/* Hide Streamlit button defaults and restyle */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    font-weight: 500;
    min-height: 42px;
    min-width: 42px;
    padding: 0;
    border-radius: 8px;
    border: 1.5px solid #dde3f0;
    background: #f8faff;
    color: #4a5578;
    transition: all 0.12s ease;
    width: 100%;
}

div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {
    border-color: #6a8fff;
    background: #eef3ff;
    color: #2952cc;
}

/* Question label styling */
.q-label {
    font-size: 0.92rem;
    font-weight: 400;
    color: #1e2d50;
    line-height: 1.3;
    padding: 4px 0;
}

/* Scale header row */
.scale-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #aab3cc;
    text-align: center;
}

/* Selected score badge */
.score-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    background: #e8f0ff;
    color: #3560e0;
    border-radius: 20px;
    padding: 1px 8px;
    margin-left: 8px;
}
.score-badge-warn {
    background: #fff3e0;
    color: #d07000;
}
.score-badge-alert {
    background: #ffeaea;
    color: #c0392b;
}

/* Submit button */
div.submit-btn > button {
    background: #2952cc !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 2rem !important;
    min-height: 48px !important;
    width: 100% !important;
    transition: background 0.15s ease !important;
}
div.submit-btn > button:hover {
    background: #1e3fa8 !important;
}

/* Follow-up box */
.followup-card {
    background: #fff8ec;
    border: 1px solid #ffd28a;
    border-radius: 12px;
    padding: 18px 20px;
    margin: 1rem 0;
}
.followup-card h4 {
    color: #a05e00;
    margin: 0 0 10px 0;
    font-size: 0.95rem;
}

/* Success */
.success-card {
    background: #f0fff6;
    border: 1px solid #7ed9a4;
    border-radius: 12px;
    padding: 22px;
    text-align: center;
    margin-top: 1rem;
}
.success-card h3 {
    color: #1a6e40;
    margin: 0 0 6px 0;
}
.success-card p {
    color: #3a8c5c;
    font-size: 0.88rem;
    margin: 0;
}

/* Divider */
.thin-divider {
    border: none;
    border-top: 1px solid #edf0f7;
    margin: 0.3rem 0;
}

/* Change indicators */
.delta-up   { color: #c0392b; font-size: 0.75rem; margin-left: 4px; }
.delta-down { color: #1a6e40; font-size: 0.75rem; margin-left: 4px; }
.delta-same { color: #8a94b0; font-size: 0.75rem; margin-left: 4px; }

/* Sticky scale header */
.sticky-scale {
    position: sticky;
    top: 0;
    background: white;
    z-index: 10;
    padding: 6px 0 4px 0;
    border-bottom: 1px solid #edf0f7;
    margin-bottom: 4px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# Questions — 28 Head & Neck Cancer symptoms
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
# Simulated previous visit answers
# ============================================================

PREVIOUS_ANSWERS: Dict[str, int] = {
    "Pain in throat / mouth": 2,
    "Pain when swallowing": 3,
    "Jaw pain or tightness": 1,
    "Ear pain": 0,
    "Neck pain or stiffness": 2,
    "Difficulty swallowing solids": 3,
    "Difficulty swallowing liquids": 1,
    "Choking or coughing while eating": 2,
    "Loss of appetite": 2,
    "Unintended weight loss": 1,
    "Dry mouth (xerostomia)": 4,
    "Thick or sticky saliva": 3,
    "Mouth sores or ulcers": 2,
    "Difficulty opening mouth (trismus)": 1,
    "Change in taste": 3,
    "Hoarseness or voice changes": 2,
    "Shortness of breath": 0,
    "Coughing": 1,
    "Mucus / phlegm buildup": 2,
    "Skin irritation or redness (radiation area)": 3,
    "Skin peeling or blistering": 2,
    "Swelling in face or neck": 1,
    "Fatigue / tiredness": 3,
    "Difficulty sleeping": 2,
    "Anxiety or worry": 2,
    "Low mood or sadness": 1,
    "Difficulty concentrating": 2,
    "Overall discomfort today": 3,
}


# ============================================================
# Session state initialisation
# ============================================================

if "answers" not in st.session_state:
    # Pre-populate with previous visit values
    st.session_state.answers = dict(PREVIOUS_ANSWERS)

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "followup_needed" not in st.session_state:
    st.session_state.followup_needed = []

if "followup_text" not in st.session_state:
    st.session_state.followup_text = ""

if "show_followup" not in st.session_state:
    st.session_state.show_followup = False


# ============================================================
# Helper: score colour class
# ============================================================

def score_class(score: int) -> str:
    if score >= 4:
        return "score-badge-alert"
    if score >= 2:
        return "score-badge-warn"
    return "score-badge"


def delta_html(current: int, previous: Optional[int]) -> str:
    if previous is None:
        return ""
    diff = current - previous
    if diff > 0:
        return f'<span class="delta-up">▲ {diff}</span>'
    if diff < 0:
        return f'<span class="delta-down">▼ {abs(diff)}</span>'
    return '<span class="delta-same">—</span>'


# ============================================================
# Follow-up logic
# ============================================================

def check_followup(current: Dict[str, int], previous: Dict[str, int]):
    flagged = []
    for q, score in current.items():
        prev = previous.get(q, 0)
        if score >= 5 or (score - prev) >= 3:
            flagged.append((q, score, prev))
    return flagged


# ============================================================
# Page header
# ============================================================

st.markdown("""
<div class="page-header">
    <div style="font-size:2rem; line-height:1;">🏥</div>
    <div>
        <h1>Head &amp; Neck Cancer — Symptom Check</h1>
        <p>Rate each symptom from <strong>0</strong> (none) to <strong>5</strong> (severe) &nbsp;·&nbsp;
           Pre-filled from your last visit &nbsp;·&nbsp; Takes under 30 seconds</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Scale legend + sticky header
# ============================================================

col_q, *col_scores = st.columns([6, 1, 1, 1, 1, 1, 1])
with col_q:
    st.markdown('<div class="scale-header" style="text-align:left; color:#aab3cc; font-size:0.72rem;">SYMPTOM</div>', unsafe_allow_html=True)
for i, col in enumerate(col_scores):
    severity_label = ["None", "Mild", "Mild+", "Mod", "Severe", "Max"][i]
    col.markdown(f'<div class="scale-header">{i}<br><span style="font-size:0.6rem; color:#c0c8de;">{severity_label}</span></div>', unsafe_allow_html=True)

st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)


# ============================================================
# Questionnaire
# ============================================================

if not st.session_state.submitted:

    for section, questions in SECTIONS.items():

        st.markdown(f'<div class="section-label">{section}</div>', unsafe_allow_html=True)

        for q in questions:
            current_val = st.session_state.answers.get(q, 0)
            prev_val    = PREVIOUS_ANSWERS.get(q)

            col_q, *col_scores = st.columns([6, 1, 1, 1, 1, 1, 1])

            with col_q:
                badge_cls = score_class(current_val)
                d_html    = delta_html(current_val, prev_val)
                st.markdown(
                    f'<div class="q-label">{q}'
                    f'<span class="{badge_cls} score-badge">{current_val}</span>'
                    f'{d_html}</div>',
                    unsafe_allow_html=True,
                )

            for i, col in enumerate(col_scores):
                selected = (current_val == i)
                # Highlight selected button with inline style injected via a unique key trick
                btn_style = (
                    "background:#2952cc !important; color:white !important; "
                    "border-color:#2952cc !important;"
                    if selected else ""
                )
                label = f"**{i}**" if selected else str(i)
                if col.button(label, key=f"btn_{q}_{i}", help=None):
                    st.session_state.answers[q] = i
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Submit
    st.markdown('<div class="submit-btn">', unsafe_allow_html=True)
    if st.button("Submit questionnaire →", key="submit_btn"):
        flagged = check_followup(st.session_state.answers, PREVIOUS_ANSWERS)
        st.session_state.followup_needed = flagged
        st.session_state.submitted = True
        st.session_state.show_followup = len(flagged) > 0
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Post-submit: follow-up or success
# ============================================================

else:

    if st.session_state.show_followup:

        st.markdown("""
        <div class="followup-card">
            <h4>⚠️ A few quick follow-up questions</h4>
            <p style="font-size:0.85rem; color:#7a5000; margin:0;">
                We noticed one or more symptoms are significantly higher than your last visit.
                Please answer the question below so your care team can follow up.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Show which symptoms triggered follow-up
        for q, score, prev in st.session_state.followup_needed:
            diff = score - prev
            reason = "scored maximum (5/5)" if score >= 5 else f"increased by {diff} points (from {prev} → {score})"
            st.markdown(
                f'<div style="font-size:0.85rem; color:#1e2d50; padding:4px 0;">'
                f'<strong>{q}</strong> — {reason}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        followup_text = st.text_area(
            "Can you briefly describe what changed for these symptoms?",
            value=st.session_state.followup_text,
            height=100,
            placeholder="e.g. Pain started after eating, throat feels more swollen than usual…",
        )
        st.session_state.followup_text = followup_text

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Submit with my notes →", key="submit_followup"):
                st.session_state.show_followup = False
                st.rerun()
        with col2:
            if st.button("Skip and submit →", key="skip_followup"):
                st.session_state.show_followup = False
                st.rerun()

    else:

        # ── Final summary ──────────────────────────────────────
        st.markdown("""
        <div class="success-card">
            <h3>✅ Questionnaire submitted</h3>
            <p>Thank you. Your care team has been notified and will review your responses.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Summary table — highlight changed values
        st.markdown('<div class="section-label">Your responses — summary</div>', unsafe_allow_html=True)

        col_q, col_prev, col_curr, col_chg = st.columns([5, 1, 1, 1])
        col_q.markdown('<div class="scale-header" style="text-align:left;">Symptom</div>', unsafe_allow_html=True)
        col_prev.markdown('<div class="scale-header">Prev</div>', unsafe_allow_html=True)
        col_curr.markdown('<div class="scale-header">Now</div>', unsafe_allow_html=True)
        col_chg.markdown('<div class="scale-header">Δ</div>', unsafe_allow_html=True)

        st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

        for q in ALL_QUESTIONS:
            curr = st.session_state.answers.get(q, 0)
            prev = PREVIOUS_ANSWERS.get(q, 0)
            diff = curr - prev

            badge_cls = score_class(curr)
            if diff > 0:
                chg_html = f'<span style="color:#c0392b; font-size:0.82rem;">▲ {diff}</span>'
            elif diff < 0:
                chg_html = f'<span style="color:#1a6e40; font-size:0.82rem;">▼ {abs(diff)}</span>'
            else:
                chg_html = '<span style="color:#aab3cc; font-size:0.82rem;">—</span>'

            c1, c2, c3, c4 = st.columns([5, 1, 1, 1])
            c1.markdown(f'<div style="font-size:0.85rem; color:#1e2d50; padding:2px 0;">{q}</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="scale-header" style="color:#8a94b0;">{prev}</div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="scale-header"><span class="{badge_cls} score-badge">{curr}</span></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="scale-header">{chg_html}</div>', unsafe_allow_html=True)

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
        if st.button("← Start a new check-in", key="restart"):
            for key in ["answers", "submitted", "followup_needed",
                        "followup_text", "show_followup"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
