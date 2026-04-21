import hashlib
import html as _html
import io
import json
import re
import concurrent.futures as _futures
from datetime import datetime
from typing import Any, Optional

import streamlit as st
import streamlit.components.v1 as _stc
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI
from chatbot_topic_flows import FLOWS, QUESTION_TYPE_BY_ID, STEP_BY_ID, TOPIC_INTROS, TOPICS


# ══════════════════════════════════════════════════════════════════
# 🔥 NEW: Adaptive LLM Question Generator
# ══════════════════════════════════════════════════════════════════

def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


_BODY_LOCATION_PATTERN = re.compile(
    r"\b("
    r"head|face|nose|ear|ears|jaw|chin|mouth|tongue|throat|neck|shoulder|arm|elbow|wrist|hand|hands|finger|fingers|"
    r"chest|back|side|stomach|belly|abdomen|hip|leg|legs|knee|knees|ankle|ankles|foot|feet|toe|toes|rib|ribs|"
    r"cheek|lip|lips|gum|gums|tooth|teeth|palate|scalp"
    r")\b"
)


def _looks_like_body_location_phrase(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    return bool(_BODY_LOCATION_PATTERN.search(normalized))


def _is_redundant_followup(original_question: str, answer: str, followup_question: str) -> bool:
    oq = _norm_text(original_question)
    fq = _norm_text(followup_question)
    if not fq:
        return True
    if fq == oq or fq in oq or oq in fq:
        return True
    return _is_semantically_redundant_question(original_question, followup_question)


def _is_semantically_redundant_question(text_a: str, text_b: str) -> bool:
    a = _norm_text(text_a)
    b = _norm_text(text_b)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    if not openai_client:
        return False
    relation = run_question_relation_agent(text_a, text_b)
    return bool(relation.get("same_intent"))


def _coerce_structured_answer(
    topic_key: str,
    step: dict,
    answer: Any,
    current_data: dict,
    raw_answer: Any = None,
) -> Any:
    if not isinstance(answer, str):
        return answer

    raw = str(raw_answer if raw_answer is not None else answer).strip()
    if not raw:
        return answer

    if topic_key == "pain" and step["id"] == "pain_location" and answer == "Somewhere else":
        if raw not in ("Somewhere else", "somewhere else"):
            current_data["pain_location_raw"] = raw
            current_data["other_pain_desc"] = raw
        return answer

    return answer

def _fallback_clarifying_question(step: dict) -> str:
    text = step.get("text", "").strip()
    if text:
        return f"I didn't quite catch that. Could you answer this part again: {text}"
    return "I didn't quite catch that. Could you tell me a little more about that?"


def _build_retry_prompt(step: dict, user_input: str, topic_history: Optional[list[dict[str, str]]] = None) -> str:
    if openai_client:
        result = run_clarification_writer_agent(
            step,
            user_input,
            topic_history=topic_history or [],
        )
        clarification = str(result.get("clarification_question") or "").strip()
        if clarification:
            return clarification
    return _fallback_clarifying_question(step)


def _auto_capture_following_answers(topic_key: str, state: dict, seed_text: str):
    # Disabled in normal operation: silent auto-filling made the conversation feel
    # presumptive and could create unrelated or repeated questions.
    return







def parse_multi_select_typed_input(step: dict, user_input: str):
    if not user_input.strip():
        return []

    lowered_map = {opt.lower(): opt for opt in step.get("opts", [])}
    parts = [p.strip() for p in re.split(r",|/|;|\n", user_input) if p.strip()]
    resolved = []
    has_other = "Other" in step.get("opts", [])
    for part in parts:
        match = lowered_map.get(part.lower())
        if match:
            resolved.append(match)
        else:
            interpreted = interpret_user_input_with_options(step, part)
            if interpreted in step.get("opts", []):
                resolved.append(interpreted)
            elif has_other and part.strip():
                resolved.append("Other")

    deduped = []
    for item in resolved:
        if item not in deduped:
            deduped.append(item)
    return deduped


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ChatReport — HNC Symptom Check-In",
    page_icon="🩺",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg1: #f7fafc;
    --bg2: #f3f7fa;
    --bg3: #eef4f8;
    --card: rgba(255,255,255,0.94);
    --card-solid: #ffffff;
    --border: #d7e4ef;
    --border-strong: #bfd3e4;
    --text: #17324a;
    --muted: #65788d;
    --primary: #0f6cbd;
    --primary-strong: #0a5a9f;
    --primary-soft: #eef7ff;
    --primary-ink: #11456d;
    --accent: #0d9488;
    --accent-soft: #ecfdf8;
    --success: #15803d;
    --success-soft: #ecfdf5;
    --warning-soft: #fff8e8;
    --shadow: 0 12px 32px rgba(23, 50, 74, 0.05);
    --shadow-sm: 0 6px 18px rgba(23, 50, 74, 0.04);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

html, body, .stApp {
    background:
        radial-gradient(circle at top left, rgba(15,108,189,0.05), transparent 22%),
        radial-gradient(circle at top right, rgba(13,148,136,0.04), transparent 24%),
        linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 58%, var(--bg3) 100%);
}

/* ── Layout ── */
.block-container {
    padding-top: 1.15rem;
    padding-bottom: 2.2rem;
    max-width: 1380px;
    padding-left: 1.1rem;
    padding-right: 1.1rem;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 0.65rem;
        padding-right: 0.65rem;
    }
}

/* ── Sidebar nav ── */
section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top, rgba(255,255,255,0.20), transparent 35%),
        linear-gradient(180deg, #123a5b 0%, #163f60 55%, #183b58 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
}

/* ── Buttons (main content area) ── */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 0.68rem 0.95rem;
    font-family: 'Manrope', sans-serif;
    font-size: 14px;
    font-weight: 700;
    border: 1px solid var(--border);
    background: linear-gradient(180deg, #ffffff 0%, #f8fbfe 100%);
    color: var(--text);
    transition: all 0.16s ease;
    text-align: center !important;
    box-shadow: none;
}
.stButton > button:hover {
    border-color: #91b7d7;
    background: #ffffff;
    color: #123664;
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(15, 108, 189, 0.08);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary) 0%, #2f88d5 100%);
    color: white;
    border: none;
    box-shadow: 0 14px 28px rgba(15, 108, 189, 0.22);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--primary-strong) 0%, var(--primary) 100%);
    color: white;
}

/* ── Sidebar nav buttons ── */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    padding: 12px 13px !important;
    font-size: 12.8px !important;
    font-weight: 700 !important;
    line-height: 1.45 !important;
    min-height: 0 !important;
    border-radius: 16px !important;
    margin-bottom: 6px !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    text-align: left !important;
    color: #eef7ff !important;
    border-color: rgba(255,255,255,0.10) !important;
    background: rgba(255,255,255,0.06) !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    border-color: rgba(255,255,255,0.16) !important;
    background: rgba(255,255,255,0.11) !important;
    color: #ffffff !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primaryFormSubmit"],
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
    font-size: 13.4px !important;
    font-weight: 700 !important;
    padding: 12px 14px !important;
    white-space: normal !important;
    color: white !important;
    margin-top: 4px !important;
    background: linear-gradient(135deg, #1184d1 0%, #0f6cbd 100%) !important;
}

/* ── Inputs ── */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
div[data-baseweb="select"] > div {
    border-radius: 14px !important;
    border: 1px solid #cfdeeb !important;
    background: rgba(255,255,255,0.98) !important;
    box-shadow: none !important;
}
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    padding: 0.9rem 1rem !important;
}
.stTextInput label,
.stTextArea label,
.stNumberInput label,
[data-testid="stAudioInput"] label {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    color: var(--primary-ink) !important;
}

/* ── Chat message wrappers ── */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    margin-bottom: 10px;
    padding: 0.05rem 0;
    background: transparent;
    display: flex;
    width: 100%;
}
[data-testid="stChatMessageContent"] {
    border-radius: 16px;
    padding: 0.8rem 0.95rem !important;
    border: 1px solid rgba(215, 228, 239, 0.9);
    box-shadow: none;
    background: #ffffff;
    width: fit-content;
    max-width: min(72%, 680px);
}
[data-testid="stChatMessageAvatar"] {
    display: none !important;
}
[data-testid="stChatMessage"]:has([aria-label="assistant"]) {
    justify-content: flex-start;
}
[data-testid="stChatMessage"]:has([aria-label="assistant"]) [data-testid="stChatMessageContent"] {
    border-left: 3px solid #b7d5eb;
    background: #ffffff;
}
[data-testid="stChatMessage"]:has([aria-label="user"]) {
    justify-content: flex-end;
}
[data-testid="stChatMessage"]:has([aria-label="user"]) [data-testid="stChatMessageContent"] {
    background: #f8fbfe;
    border-left: 3px solid #0f6cbd;
    border-right: 3px solid #0f6cbd;
    border-left: none;
}

.chat-shell {
    background:
        radial-gradient(circle at top right, rgba(15,108,189,0.07), transparent 32%),
        linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248,251,254,0.98) 100%);
    border: 1px solid #d9e6f0;
    border-radius: 26px;
    padding: 0;
    overflow: hidden;
    box-shadow: 0 24px 60px rgba(23, 50, 74, 0.09);
    margin-top: 6px;
}

.chat-shell-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 14px;
    border-bottom: 1px solid #e2ebf2;
    background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(245,249,252,0.9) 100%);
}

.chat-shell-title {
    display: block;
    width: 100%;
}

.chat-shell-name {
    font-family: 'Manrope', sans-serif;
    font-size: 15px;
    font-weight: 800;
    color: #143551;
    letter-spacing: -0.03em;
}

.chat-shell-inline-summary {
    flex: 0 0 auto;
}

.chat-shell-inline-summary details {
    display: inline-block;
}

.chat-shell-inline-summary summary {
    list-style: none;
    cursor: pointer;
    font-size: 12px;
    font-weight: 700;
    color: #607589;
    white-space: nowrap;
}

.chat-shell-inline-summary summary::-webkit-details-marker {
    display: none;
}

.chat-shell-inline-summary summary::before {
    content: "▸";
    display: inline-block;
    margin-right: 6px;
    color: #607589;
}

.chat-shell-inline-summary details[open] summary::before {
    content: "▾";
}

.chat-shell-inline-summary-body {
    padding: 10px 14px 8px 14px;
    border-bottom: 1px solid #e8eef4;
    background: rgba(247, 251, 254, 0.78);
}

.chat-history {
    padding: 14px 14px 8px 14px;
    min-height: 0;
    background:
        linear-gradient(180deg, rgba(250,252,254,0.88) 0%, rgba(244,248,252,0.92) 100%);
}

.composer-wrap {
    padding: 0 12px 12px 12px;
    background: transparent;
}

.chat-row {
    display: flex !important;
    width: 100% !important;
    margin-bottom: 12px;
    align-items: flex-start;
    justify-content: flex-start !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    gap: 10px;
}

.chat-avatar {
    width: 26px;
    height: 26px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 26px;
    font-size: 11px;
    font-weight: 800;
    color: white;
    margin-top: 2px;
}

.chat-row.assistant .chat-avatar {
    background: #ef476f;
}

.chat-row.user .chat-avatar {
    background: #20262d;
}

.chat-entry {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-width: min(86%, 860px);
    min-width: 0;
}

.chat-meta {
    display: flex;
    align-items: baseline;
    gap: 6px;
    padding: 0 2px;
}

.chat-role {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: none;
    color: #1d2b36;
}

.chat-time {
    font-size: 10px;
    color: #9aa9b6;
}

.chat-bubble {
    display: block;
    width: 100%;
    max-width: 100%;
    border-radius: 16px;
    padding: 0.62rem 0.75rem;
    border: 1px solid #f4f7fa;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 13.5px;
    box-shadow: none;
    background: rgba(255,255,255,0.72);
}

.chat-row.assistant .chat-bubble {
    color: #17324a;
    border-top-left-radius: 6px;
}

.chat-row.user .chat-bubble {
    color: #17324a;
    border-top-left-radius: 6px;
}

/* ── Topic status pills ── */
.status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    margin-left: 6px;
}
.pill-done   { background: #d1fae5; color: #065f46; }
.pill-active { background: #dbeafe; color: #1e40af; }
.pill-todo   { background: #f3f4f6; color: #6b7280; }

/* ── Modern shell cards ── */
.card {
    background: var(--card);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.7);
    border-radius: 24px;
    padding: 22px 24px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
}

.soft-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 18px 18px;
    box-shadow: var(--shadow-sm);
}

.assistant-chip {
    display: none;
}
.assistant-chip .avatar {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #d5ecfb, #edf8ff);
    font-size: 22px;
}
.assistant-chip .name {
    font-family: 'Manrope', sans-serif;
    font-size: 14px;
    font-weight: 800;
    color: #113553;
    margin-bottom: 2px;
}
.assistant-chip .role {
    font-size: 12px;
    color: #678196;
}

.memory-banner {
    background: linear-gradient(135deg, #edf7ff 0%, #f9fcff 100%);
    border: 1px solid #cadeef;
    border-radius: 18px;
    padding: 12px 14px;
    color: #23486f;
    font-size: 13px;
    margin-bottom: 12px;
}

.report-box {
    background: rgba(255,255,255,0.94);
    border: 1px solid #dbe5f1;
    border-radius: 22px;
    padding: 24px 28px;
    font-size: 14.5px;
    line-height: 1.72;
    white-space: pre-wrap;
    box-shadow: var(--shadow);
}

.report-dashboard {
    background:
        linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(247,250,253,0.98) 100%);
    border: 1px solid #d8e4ee;
    border-radius: 28px;
    padding: 24px 24px 20px 24px;
    box-shadow: var(--shadow);
}

.report-summary-banner {
    background: linear-gradient(180deg, #fff4cf 0%, #fff8e7 100%);
    border: 1px solid #ebd28a;
    border-radius: 22px;
    padding: 18px 20px;
    margin: 14px 0 18px 0;
    display: grid;
    grid-template-columns: 88px 1fr;
    gap: 16px;
    align-items: center;
}

.report-summary-avatar {
    width: 76px;
    height: 76px;
    border-radius: 18px;
    border: 1px solid #dfc26b;
    background: linear-gradient(180deg, #fffdf7 0%, #fff4d2 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 34px;
}

.report-summary-copy {
    color: #5e4a12;
    font-size: 13px;
    line-height: 1.65;
}

.report-summary-title {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b6c1b;
    margin-bottom: 4px;
}

.report-topic-card {
    border-radius: 18px;
    padding: 0;
    overflow: hidden;
    border: 1px solid #d7e1eb;
    background: #ffffff;
    min-height: 164px;
    box-shadow: 0 8px 18px rgba(23, 50, 74, 0.05);
    margin-bottom: 10px;
}

.report-topic-card.red {
    border-color: #ef9c9c;
    background: linear-gradient(180deg, #fff7f7 0%, #fff1f1 100%);
}

.report-topic-card.green {
    border-color: #9fd1a8;
    background: linear-gradient(180deg, #f6fff7 0%, #effbf1 100%);
}

.report-topic-strip {
    padding: 8px 12px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: white;
    background: #9aa8b7;
}

.report-topic-card.red .report-topic-strip {
    background: #d84c43;
}

.report-topic-card.green .report-topic-strip {
    background: #3f8f49;
}

.report-topic-body {
    padding: 12px;
}

.report-topic-name {
    font-size: 13px;
    font-weight: 800;
    color: #16324b;
    margin-bottom: 8px;
}

.report-topic-compare {
    display: grid;
    grid-template-columns: 1fr;
    gap: 8px;
}

.report-topic-compare-row {
    border: 1px solid #e4ebf2;
    border-radius: 14px;
    padding: 8px 9px;
    background: rgba(255,255,255,0.86);
}

.report-topic-compare-label {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8295a9;
    margin-bottom: 2px;
}

.report-topic-compare-value {
    font-size: 12px;
    line-height: 1.45;
    color: #18344d;
}

.report-topic-open {
    margin-top: -4px;
    margin-bottom: 12px;
}

.report-detail-shell {
    margin-top: 10px;
    border: 1px solid #d7e2eb;
    border-radius: 24px;
    background: linear-gradient(180deg, #ffffff 0%, #f9fbfd 100%);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
}

.report-detail-shell.inline {
    margin-top: 10px;
    margin-bottom: 18px;
    animation: reportSlideDown 180ms ease-out;
}

.report-detail-header {
    padding: 16px 18px;
    border-bottom: 1px solid #e5edf4;
    background: rgba(255,255,255,0.88);
}

.report-detail-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    padding: 16px 18px 6px 18px;
}

.report-detail-panel {
    border: 1px solid #e2eaf1;
    border-radius: 18px;
    padding: 14px;
    background: #ffffff;
}

.report-detail-label {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7d90a3;
    margin-bottom: 6px;
}

.report-detail-text {
    font-size: 14px;
    line-height: 1.6;
    color: #17324a;
}

@media (max-width: 900px) {
    .report-summary-banner,
    .report-detail-grid {
        grid-template-columns: 1fr;
    }
}

@keyframes reportSlideDown {
    from {
        opacity: 0;
        transform: translateY(-6px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* ── Progress / completion ── */
.prog-label {
    font-size: 12px;
    color: #6b7280;
    margin-bottom: 4px;
    font-weight: 600;
}
.completion-badge {
    background: linear-gradient(135deg, #0f9f6e, #0f6cbd);
    border-radius: 20px;
    padding: 18px 20px;
    color: white;
    font-weight: 700;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 14px 28px rgba(15, 108, 189, 0.20);
}

/* ── Welcome / login ── */
.welcome-card {
    background:
        radial-gradient(circle at top right, rgba(15,108,189,0.12), transparent 30%),
        linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(246,250,254,0.99) 100%);
    border: 1px solid #d6e4f0;
    border-radius: 30px;
    padding: 32px 36px;
    max-width: 720px;
    margin: 56px auto 24px auto;
    box-shadow: var(--shadow);
}

.overview-card {
    background:
        radial-gradient(circle at top right, rgba(15,108,189,0.08), transparent 32%),
        linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,251,254,0.98) 100%);
    border: 1px solid #d6e4f0;
    border-radius: 28px;
    padding: 28px 30px;
    max-width: 960px;
    margin: 24px auto 18px auto;
    box-shadow: var(--shadow);
}

.overview-table-wrap {
    margin-top: 18px;
    border: 1px solid #d7e4ee;
    border-radius: 22px;
    overflow: hidden;
    background: #ffffff;
    box-shadow: 0 8px 20px rgba(23, 50, 74, 0.04);
}

.overview-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
}

.overview-table col.topic-col {
    width: 220px;
}

.overview-table col.summary-col {
    width: 250px;
}

.overview-table th,
.overview-table td {
    padding: 16px 18px;
    vertical-align: top;
    border-bottom: 1px solid #edf3f7;
}

.overview-table thead th {
    background: #f5f9fd;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6a8198;
    text-align: left;
    border-bottom: 1px solid #d7e4ee;
}

.overview-table tbody tr:last-child td {
    border-bottom: none;
}

.overview-topic-name {
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #16324b;
}

.overview-summary-main {
    font-size: 14px;
    line-height: 1.55;
    color: #16324b;
    font-weight: 700;
    margin-bottom: 8px;
}

.overview-summary-details {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

@media (max-width: 768px) {
    .overview-table,
    .overview-table thead,
    .overview-table tbody,
    .overview-table tr,
    .overview-table th,
    .overview-table td {
        display: block;
        width: 100%;
    }

    .overview-table thead {
        display: none;
    }

    .overview-table td {
        padding: 14px 15px;
    }

    .overview-table tbody tr {
        border-bottom: 1px solid #edf3f7;
    }

    .overview-table tbody tr:last-child {
        border-bottom: none;
    }

    .overview-topic-name {
        margin-bottom: 2px;
    }
}

.overview-note {
    margin-top: 16px;
    font-size: 13px;
    line-height: 1.65;
    color: #5f7287;
    background: #f7fbfe;
    border: 1px solid #dbe7f0;
    border-radius: 16px;
    padding: 12px 14px;
}

.subtle-note {
    background: #f7fbfe;
    border: 1px dashed #c6d8e7;
    border-radius: 16px;
    padding: 10px 12px;
    font-size: 12.5px;
    color: #6b7d92;
    margin-top: 8px;
}

.topic-panel {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
    box-shadow: none;
}

.section-kicker {
    display: none;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 6px 12px;
    background: #edf7ff;
    border: 1px solid #cfe2f2;
    color: #0f5d93;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.active-question {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
    margin: 0 0 10px 0;
    box-shadow: none;
}

.active-question .label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #7d92a7;
    margin-bottom: 6px;
}

.active-question .text {
    font-family: 'Manrope', sans-serif;
    font-size: 18px;
    line-height: 1.4;
    font-weight: 700;
    color: #153652;
}

.reply-shell {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
    box-shadow: none;
    margin-top: 4px;
}

.composer-shell {
    background: rgba(255,255,255,0.92);
    border: 1px solid #d9e4ed;
    border-radius: 26px;
    padding: 12px;
    box-shadow: 0 18px 36px rgba(23, 50, 74, 0.08);
    backdrop-filter: blur(10px);
}

.composer-shell.compact {
    padding: 12px;
}


.composer-row {
    display: flex;
    align-items: flex-end;
    gap: 10px;
}

[data-testid="stAudioInput"] {
    background: transparent;
    border: none;
    border-radius: 999px;
    padding: 0;
    margin: 0;
    box-shadow: none;
}

.stTextInput input {
    border-radius: 999px !important;
}

.stTextArea textarea {
    border-radius: 18px !important;
}

[data-testid="stAudioInput"] audio {
    border-radius: 14px;
}

.composer-shell div[data-testid="stButton"] > button {
    width: 100% !important;
    min-width: 0 !important;
    padding: 0.72rem 0.95rem !important;
    border-radius: 16px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 18px rgba(23, 50, 74, 0.06) !important;
    background: linear-gradient(180deg, #ffffff 0%, #f7fbfe 100%) !important;
    border: 1px solid #d9e4ed !important;
}

.composer-shell div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    border-color: #9fc1dd !important;
    color: #10375a !important;
}

.suggested-replies-note {
    margin: 8px 0 10px 2px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #6b7d92;
}

.composer-shell div[data-baseweb="select"] > div {
    border-radius: 16px !important;
    min-height: 46px !important;
    height: 46px !important;
    background: #f7fbfe !important;
    border: 1px solid #d7e4ee !important;
    display: flex !important;
    align-items: center !important;
}

.composer-shell [data-testid="stTextInput"] {
    margin-bottom: 0 !important;
}

.composer-shell [data-testid="stTextInput"] input {
    min-height: 52px !important;
    height: 52px !important;
    background: #f9fcff !important;
    border: 1px solid #d6e4ef !important;
    padding-left: 16px !important;
}

.composer-shell [data-testid="stSelectbox"] {
    margin-bottom: 0 !important;
}

.composer-shell [data-testid="stAudioInput"] {
    background: transparent;
    border: none;
    border-radius: 16px;
    min-height: 46px;
    width: 100%;
    min-width: 100%;
    max-width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    margin: 10px 0 0 0;
    box-shadow: none !important;
}

.composer-shell [data-testid="stAudioInput"] > div {
    width: 100%;
    min-width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 0 !important;
    margin: 0 !important;
}

.composer-shell [data-testid="stAudioInput"] button {
    border-radius: 16px !important;
    width: 100% !important;
    height: 48px !important;
    min-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 1px solid #d7e4ee !important;
    background: linear-gradient(180deg, #ffffff 0%, #f5f9fd 100%) !important;
    box-shadow: 0 8px 18px rgba(23, 50, 74, 0.08) !important;
}

@media (max-width: 768px) {
    .chat-shell {
        border-radius: 22px;
    }

    .chat-shell-header {
        padding: 9px 10px 9px 10px;
        align-items: flex-start;
        flex-direction: column;
    }

    .chat-history {
        padding: 12px 10px 8px 10px;
        min-height: 180px;
    }

    .composer-wrap {
        padding: 0 10px 10px 10px;
    }

    .chat-entry {
        max-width: 92%;
    }
}

.composer-shell [data-testid="stAudioInput"] button::before {
    display: none !important;
}

.composer-shell [data-testid="stAudioInput"] button:hover {
    border-color: #bed4e7 !important;
    background: #ffffff !important;
    box-shadow: 0 10px 20px rgba(15, 108, 189, 0.10) !important;
}

.composer-shell [data-testid="stAudioInput"] button svg {
    width: 17px !important;
    height: 17px !important;
    color: #0f6cbd !important;
}


[data-testid="stProgressBar"] > div {
    border-radius: 999px !important;
    background: rgba(15,108,189,0.12) !important;
}
[data-testid="stProgressBar"] div[role="progressbar"] {
    background: linear-gradient(90deg, #0d9488 0%, #0f6cbd 100%) !important;
}
</style>
""", unsafe_allow_html=True)




def render_memory_banner(message: str):
    st.markdown(f'<div class="memory-banner">📋 {message}</div>', unsafe_allow_html=True)


def render_section_kicker(text: str):
    st.markdown(f'<div class="section-kicker">{_html.escape(text)}</div>', unsafe_allow_html=True)


def render_active_question(question: str, label: str = "Current question"):
    st.markdown(
        '<div class="active-question">'
        f'<div class="label">{_html.escape(label)}</div>'
        f'<div class="text">{_html.escape(question)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _dynamic_step_text(topic_key: Optional[str], step: dict, state: Optional[dict] = None) -> str:
    question_text = step["text"]
    if not state or not topic_key or not openai_client:
        return question_text

    prompt_cache = state.setdefault("generated_prompts", {})
    cached = prompt_cache.get(step["id"])
    if cached:
        return cached

    last_visit_same_question_answer = st.session_state.get("last_checkin", {}).get(topic_key, {}).get(step.get("id"))
    result = run_question_writer_agent(
        step,
        topic_history=_recent_topic_history(state),
        recent_questions=_recent_topic_questions(state),
        last_visit_same_question_answer=last_visit_same_question_answer,
    )
    rewritten = str(result.get("question_text") or question_text).strip() or question_text
    prompt_cache[step["id"]] = rewritten
    return rewritten


def _step_prompt_text(step: dict, topic_key: Optional[str] = None, state: Optional[dict] = None) -> str:
    question_text = _dynamic_step_text(topic_key, step, state)
    if step.get("type") == "options" and step.get("id") != "med_adherence_issue":
        question_text += " (Choose an option below, or answer in your own words if that fits better.)"
    return question_text


def _append_assistant_message(state: dict, text: str):
    text = (text or "").strip()
    if not text:
        return
    if state["chat"] and state["chat"][-1]["role"] == "assistant" and state["chat"][-1]["content"].strip() == text:
        return
    state["chat"].append({"role": "assistant", "content": text})


def _remember_prompted_step(state: dict, step: Optional[dict], prompt_text: str = ""):
    state["last_prompted_step_id"] = step.get("id") if step else None
    state["last_prompted_text"] = (prompt_text or "").strip()


def _ensure_step_prompted(topic_key: str, state: dict, step: Optional[dict]):
    if not step:
        return
    prompt_text = _step_prompt_text(step, topic_key=topic_key, state=state)
    last_id = state.get("last_prompted_step_id")
    last_text = state.get("last_prompted_text", "")
    if last_id == step.get("id") and (
        last_text == prompt_text or _is_semantically_redundant_question(last_text, prompt_text)
    ):
        return
    _append_assistant_message(state, prompt_text)
    _remember_prompted_step(state, step, prompt_text)


def _recent_topic_history(state: dict, limit: int = 10) -> list[dict[str, str]]:
    history = []
    for msg in state.get("chat", [])[-limit:]:
        role = msg.get("role", "")
        content = str(msg.get("content", "")).strip()
        if role and content:
            history.append({"role": role, "content": content})
    return history


def _recent_topic_questions(state: dict, limit: int = 8) -> list[str]:
    questions = []
    for msg in state.get("chat", [])[-limit * 2:]:
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        parts = [part.strip() for part in content.split("\n\n") if part.strip()]
        for part in parts:
            if "?" in part:
                questions.append(part)
    return questions[-limit:]


def _question_already_asked(state: dict, question_text: str) -> bool:
    candidate = (question_text or "").strip()
    if not candidate:
        return False
    for asked in _recent_topic_questions(state, limit=10):
        if _is_semantically_redundant_question(asked, candidate):
            return True
    return False


def render_chat_bubble(role: str, content: str):
    safe = _html.escape(content or "").replace("\n", "<br>")
    role_cls = "user" if role == "user" else "assistant"
    role_label = "You" if role == "user" else "Care Assistant"
    avatar_label = "Y" if role == "user" else "I"
    timestamp = datetime.now().strftime("%H:%M")
    st.markdown(
        f'<div class="chat-row {role_cls}">'
        f'  <div class="chat-avatar">{avatar_label}</div>'
        f'  <div class="chat-entry">'
        f'    <div class="chat-meta"><div class="chat-role">{role_label}</div><div class="chat-time">{timestamp}</div></div>'
        f'    <div class="chat-bubble">{safe}</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# SECRETS / OPENAI
# ══════════════════════════════════════════════════════════════════

def _secret(*keys, default=None):
    for k in keys:
        if k in st.secrets:
            return st.secrets[k]
    return default


OPENAI_API_KEY = _secret("openai_api_key", "OPENAI_API_KEY", "openai_key")
openai_client: Optional[OpenAI] = None
_openai_error: Optional[str] = None

if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        _openai_error = str(e)
else:
    _openai_error = "OpenAI API key not configured."


# ══════════════════════════════════════════════════════════════════
# GOOGLE SHEETS
# ══════════════════════════════════════════════════════════════════

_sheet = None
_sheet_error: Optional[str] = None


def _init_sheets():
    global _sheet, _sheet_error
    if _sheet is not None or _sheet_error is not None:
        return
    try:
        creds = Credentials.from_service_account_info(
            _secret("gcp_service_account"),
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        book = gspread.authorize(creds).open_by_key(_secret("gsheet_id"))
        try:
            ws = book.worksheet("ChatReport")
        except Exception:
            ws = book.add_worksheet(title="ChatReport", rows=2000, cols=5)
            ws.append_row(["timestamp", "name", "all_data_json", "report"])
        _sheet = ws
    except Exception as e:
        _sheet_error = str(e)


def save_to_sheet(name: str, all_data: dict, report: str = "") -> bool:
    """
    Append one row to the Google Sheet.
    Columns: timestamp | name | all_data_json | report
    Returns True on success, False on failure.
    """
    _init_sheets()
    if _sheet is None:
        st.error(f"Could not connect to Google Sheets: {_sheet_error}")
        return False
    try:
        _sheet.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            name,
            json.dumps(all_data, ensure_ascii=False),
            report,
        ])
        return True
    except Exception as e:
        st.error(f"Failed to save to Google Sheets: {e}")
        return False


# ══════════════════════════════════════════════════════════════════
# VOICE / WHISPER
# ══════════════════════════════════════════════════════════════════

def _transcribe(audio_bytes: bytes) -> str:
    if not openai_client:
        return ""
    try:
        buf = io.BytesIO(audio_bytes)
        buf.name = "audio.wav"
        return openai_client.audio.transcriptions.create(
            model="whisper-1", file=buf
        ).text.strip()
    except Exception:
        return ""


def voice_widget(key_suffix: str, label: str = "Speak your answer") -> Optional[str]:
    """Renders voice recorder. Returns transcript string if new audio was processed."""
    transcript_key = f"_vt_{key_suffix}"
    hash_key = f"_vh_{key_suffix}"
    if hash_key not in st.session_state:
        st.session_state[hash_key] = None

    audio = st.audio_input(label, key=f"_vrec_{key_suffix}", label_visibility="collapsed")
    if not audio:
        return st.session_state.get(transcript_key)

    try:
        ab = audio.getvalue()
    except Exception:
        return st.session_state.get(transcript_key)

    if not ab:
        return st.session_state.get(transcript_key)

    ah = hashlib.sha1(ab).hexdigest()
    if ah == st.session_state[hash_key]:
        return st.session_state.get(transcript_key)

    st.session_state[hash_key] = ah
    with st.spinner("Transcribing…"):
        text = _transcribe(ab)

    if text:
        st.session_state[transcript_key] = text
        st.rerun()

    return st.session_state.get(transcript_key)


# ══════════════════════════════════════════════════════════════════
# TOPIC & FLOW DEFINITIONS
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# FLOW ENGINE
# ══════════════════════════════════════════════════════════════════

def _step_is_relevant(topic_key: str, step: dict, data: dict, raw_answers: Optional[dict] = None) -> bool:
    return True


def get_next_step(topic_key: str, data: dict, raw_answers: Optional[dict] = None) -> Optional[dict]:
    """Return the first unanswered applicable step for this topic."""
    for step in FLOWS.get(topic_key, []):
        when = step.get("when")
        if when and not when(data):
            continue
        if not _step_is_relevant(topic_key, step, data, raw_answers):
            continue
        if step["id"] not in data:
            return step
    return None


def get_upcoming_steps(topic_key: str, data: dict, raw_answers: Optional[dict] = None, limit: int = 5) -> list[dict]:
    upcoming = []
    for step in FLOWS.get(topic_key, []):
        when = step.get("when")
        if when and not when(data):
            continue
        if not _step_is_relevant(topic_key, step, data, raw_answers):
            continue
        if step["id"] in data:
            continue
        upcoming.append(step)
        if len(upcoming) >= limit:
            break
    return upcoming


def topic_is_complete(topic_key: str, data: dict, raw_answers: Optional[dict] = None) -> bool:
    return get_next_step(topic_key, data, raw_answers) is None


def get_topic_progress(topic_key: str, data: dict, raw_answers: Optional[dict] = None) -> tuple[int, int]:
    """Returns (answered, applicable) counts."""
    flow = FLOWS.get(topic_key, [])
    applicable = [
        s for s in flow
        if (not s.get("when") or s["when"](data)) and _step_is_relevant(topic_key, s, data, raw_answers)
    ]
    answered = [s for s in applicable if s["id"] in data]
    return len(answered), len(applicable)



# ══════════════════════════════════════════════════════════════════
# MULTI-AGENT CLINICAL SYSTEM
# ══════════════════════════════════════════════════════════════════
# Architecture based on the ChatReport agent design:
#   Agent 1 — Answer Interpreter   : classify patient free-text
#   Agent 2 — Prior Comparison     : delta from last check-in
#   Agent 3 — Urgency & Criticality: patient safety monitoring
#   Agent 4 — Sentiment Monitor    : engagement & emotional state
#   Agent 5 — Doctor-Relevance     : clinical sufficiency + follow-up
#   Agent 6 — Next-Move            : author the follow-up question
#   Orchestrator — coordinates all agents, resolves conflicts
# ══════════════════════════════════════════════════════════════════


# ── Shared clinical background for HNC patients ───────────────────
_HNC_CONTEXT = (
    "Patients are adults receiving treatment for head and neck cancer (HNC) — "
    "typically chemoradiation or surgery. Common issues: severe mucositis, "
    "dysphagia, pain, weight loss, fatigue, depression, and impaired communication. "
    "Many are immunocompromised. Underreporting of severity is common in this population."
)

_RED_FLAGS = (
    "- Pain ≥ 7/10, uncontrolled or worsening despite medication\n"
    "- Fever ≥ 100.4 °F / 38 °C or chills with possible infection\n"
    "- Significant unintentional weight loss (> 5 lbs since last visit)\n"
    "- Complete inability to swallow or take any oral intake\n"
    "- Feeding tube complications: leakage, blockage, site infection\n"
    "- Breathing difficulty at rest, wheezing, or worsening dyspnoea\n"
    "- Falls or near-falls, especially with dizziness\n"
    "- Suicidal ideation or self-harm intent\n"
    "- Severe depression / distress interfering with daily function\n"
    "- New neurological symptoms: sudden weakness, numbness, confusion\n"
    "- Medication non-adherence affecting symptom control"
)


def _call_agent(system_prompt: str, user_content: dict, max_tokens: int = 500) -> dict:
    """
    Call OpenAI with a system + user message pair. All agents use this.
    Returns parsed JSON dict; returns {} on any error.
    """
    if not openai_client:
        return {}
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": json.dumps(user_content, ensure_ascii=False)},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return _extract_json_object(resp.choices[0].message.content.strip())
    except Exception as exc:
        print(f"[_call_agent error] {exc}")
        return {}


# ── Also keep legacy helper for report generation ─────────────────
def _call_openai(prompt: str, max_tokens: int = 120, temp: float = 0.4) -> str:
    if not openai_client:
        return ""
    try:
        r = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temp,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return ""


_ROUTING_SUPPORT_SYS = """
You are a compact routing support agent for a clinical chatbot.

You will receive a mode plus the relevant inputs.

Modes:
  - question_relation
  - pain_location_focus

For mode=question_relation:
  Decide whether two patient-facing questions are asking essentially the same thing.
  Rules:
    - Treat paraphrases as the same intent.
    - If one question is only a softer or more natural wording of the other, mark true.
    - If one asks for a different detail, mark false.
    - Be especially sensitive to duplicate symptom questions in clinical chat.

For mode=pain_location_focus:
  Decide whether a patient-described pain location should trigger the focused
  head-and-neck follow-up branch.
  Rules:
    - Mark true only when the described location is specifically in or very near the
      ear, jaw, mouth, lips, gums, teeth, cheek, palate, tongue, or throat.
    - Mark false for broad or nonspecific areas like head, face, neck, arm, back, chest,
      or anything outside that focused region.

Return ONLY valid JSON with the fields needed for the mode:
{
  "same_intent": true/false,
  "head_neck_focused": true/false
}
"""


def _run_routing_support_agent(mode: str, payload: dict, max_tokens: int = 80) -> dict:
    if not openai_client:
        return {}
    result = _call_agent(_ROUTING_SUPPORT_SYS, {"mode": mode, **payload}, max_tokens=max_tokens)
    return result or {}


def run_question_relation_agent(question_a: str, question_b: str) -> dict:
    default = {"same_intent": False}
    result = _run_routing_support_agent("question_relation", {
        "question_a": question_a,
        "question_b": question_b,
    })
    return {**default, **result}


def run_pain_location_focus_agent(location_text: str) -> dict:
    default = {"head_neck_focused": False}
    if not str(location_text or "").strip():
        return default
    result = _run_routing_support_agent("pain_location_focus", {
        "location_text": location_text,
    })
    return {**default, **result}


_CONVERSATION_COPY_SYS = """
You are a compact conversation copy agent for a clinical chatbot.

You will receive a mode plus the relevant inputs.

Modes:
  - question_rewrite
  - clarification
  - quick_replies

For mode=question_rewrite:
  Rewrite the next patient-facing question naturally.
  Rules:
    - The form question is a question bank hint, not fixed wording.
    - Write one natural nurse-like question that asks for the same clinical detail.
    - Use topic history so the question feels like a direct continuation of the conversation.
    - Avoid repeating a recent question from this topic.
    - Do not rewrite a narrower follow-up detail question back into the broader parent question.
    - If the patient already named a specific location, medication, symptom, or person, do not turn the next question back into a generic chooser they already answered.
    - Mention last-visit history only when it genuinely helps orient the patient.
    - Never prepend awkward phrases like "Last visit you reported yes."
    - Do not imply symptoms the patient has not endorsed.
    - Keep the question concise and conversational.

For mode=clarification:
  Write a short clarification question when the patient's reply did not clearly answer the current question.
  Rules:
    - Ask for the same missing information more clearly, not a different detail.
    - Be warm and brief.
    - If options exist, you may gently restate the kind of answer needed.
    - Do not sound robotic or blame the patient.
    - Do not repeat the original question verbatim.

For mode=quick_replies:
  Suggest clickable quick-reply buttons.
  Rules:
    - Return 2 to 4 short patient-style reply options when helpful.
    - These are suggestion buttons, not the only valid answers. Free text will still be available.
    - Suggestions must fit the current question naturally and must not contradict the recent topic history.
    - Do not repeat a detail the patient already gave as if it still needs to be answered.
    - If predefined_options already exist, return an empty list.
    - For number questions, suggestions must be numeric strings within range.
    - For free-text questions, keep suggestions short and natural, like something a patient would actually tap.
    - Prefer suggestions that help the patient answer quickly, not a full exhaustive list.
    - If no helpful suggestions are obvious, return an empty list.

Return ONLY valid JSON with the fields needed for the mode:
{
  "question_text": "...",
  "clarification_question": "...",
  "suggestions": ["...", "..."]
}
"""


def _run_conversation_copy_agent(mode: str, payload: dict, max_tokens: int) -> dict:
    if not openai_client:
        return {}
    result = _call_agent(_CONVERSATION_COPY_SYS, {"mode": mode, **payload}, max_tokens=max_tokens)
    return result or {}


def run_question_writer_agent(
    step: dict,
    topic_history: list[dict[str, str]],
    recent_questions: list[str],
    last_visit_same_question_answer: Any = None,
) -> dict:
    default = {"question_text": step.get("text", "")}
    result = _run_conversation_copy_agent("question_rewrite", {
        "base_question_text": step.get("text", ""),
        "question_type": step.get("type", "options"),
        "options": step.get("opts", []),
        "recent_topic_history": topic_history,
        "recent_question_texts": recent_questions,
        "last_visit_same_question_answer": last_visit_same_question_answer,
    }, max_tokens=140)
    return {**default, **result}


def run_clarification_writer_agent(step: dict, patient_reply: str, topic_history: list[dict[str, str]]) -> dict:
    default = {"clarification_question": "Could you tell me a little more about that?"}
    result = _run_conversation_copy_agent("clarification", {
        "original_question": step.get("text", ""),
        "question_type": step.get("type", "options"),
        "options": step.get("opts", []),
        "patient_reply": patient_reply,
        "recent_topic_history": topic_history,
    }, max_tokens=120)
    return {**default, **result}


def run_quick_reply_suggester_agent(step: dict, topic_history: list[dict[str, str]], recent_questions: list[str]) -> dict:
    default = {"suggestions": []}
    result = _run_conversation_copy_agent("quick_replies", {
        "question_text": step.get("text", ""),
        "question_type": step.get("type", "free_text"),
        "predefined_options": step.get("opts", []),
        "placeholder": step.get("placeholder"),
        "min_value": step.get("min_v"),
        "max_value": step.get("max_v"),
        "recent_topic_history": topic_history,
        "recent_question_texts": recent_questions,
    }, max_tokens=140)
    if not result:
        return default

    suggestions = result.get("suggestions", [])
    if not isinstance(suggestions, list):
        return default

    cleaned = []
    seen = set()
    for item in suggestions:
        text = str(item or "").strip()
        if not text:
            continue
        key = _norm_text(text)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= 4:
            break
    return {"suggestions": cleaned}


_REPORT_SUPPORT_SYS = """
You are a compact report support agent for a clinical symptom dashboard.

You will receive a mode plus the relevant inputs.

Modes:
  - topic_summary
  - topic_insight
  - overview

For mode=topic_summary:
  Write a one-sentence human summary of the patient's last check-in for a single topic.
  Rules:
    - Write one natural sentence, not bullet points.
    - Focus on the most clinically relevant details.
    - Prefer plain language.
    - If the data is sparse, write a modest summary instead of inventing detail.
    - Do not mention internal field names.
    - Keep it under 20 words.

For mode=topic_insight:
  Compare one topic from a patient's current check-in against their last check-in.
  Rules:
    - worsened: same issue but clearly more severe, broader, or more disruptive now
    - new_issue: a clinically meaningful issue is present now but was absent or not reported last visit
    - improved: clearly better now than last visit
    - stable: same overall, unchanged, or no meaningful difference
    - unanswered: current topic has essentially no usable answer
    - Keep summaries patient/clinician-readable, plain language, and concise.
    - detail_lines should be short factual lines that help a clinician compare last vs now.
    - attention_lines should include only the most clinically relevant items.

For mode=overview:
  Write a short patient-summary banner for a clinical topic dashboard.
  Rules:
    - Use the provided topic insight summaries.
    - Keep each item short.
    - Mention only the most meaningful changes.
    - If a category has nothing meaningful, return an empty list or null.

Return ONLY valid JSON with the fields needed for the mode:
{
  "summary": "...",
  "status": "worsened|new_issue|improved|stable|unanswered",
  "status_label": "...",
  "last_summary": "..." or null,
  "current_summary": "..." or null,
  "detail_lines": ["...", "..."],
  "attention_lines": ["...", "..."],
  "main_issue": "..." or null,
  "new_issues": ["...", "..."],
  "improvements": ["...", "..."],
  "needs_attention": ["...", "..."]
}
"""


def _run_report_support_agent(mode: str, payload: dict, max_tokens: int) -> dict:
    if not openai_client:
        return {}
    result = _call_agent(_REPORT_SUPPORT_SYS, {"mode": mode, **payload}, max_tokens=max_tokens)
    return result or {}


def run_topic_summary_agent(topic_label: str, topic_data: dict) -> dict:
    default = {"summary": "Information was recorded for this topic."}
    result = _run_report_support_agent("topic_summary", {
        "topic_label": topic_label,
        "topic_data": topic_data,
    }, max_tokens=100)
    return {**default, **result}


def run_report_topic_insight_agent(topic_label: str, last_topic_data: dict, current_topic_data: dict) -> dict:
    default = {
        "status": "unanswered" if not current_topic_data else "stable",
        "status_label": "Not answered" if not current_topic_data else "Stable",
        "last_summary": None,
        "current_summary": None,
        "detail_lines": [],
        "attention_lines": [],
    }
    result = _run_report_support_agent("topic_insight", {
        "topic_label": topic_label,
        "last_topic_data": last_topic_data,
        "current_topic_data": current_topic_data,
    }, max_tokens=220)
    return {**default, **result}


def run_report_overview_agent(topic_insights: list[dict]) -> dict:
    default = {
        "main_issue": None,
        "new_issues": [],
        "improvements": [],
        "needs_attention": [],
    }
    result = _run_report_support_agent("overview", {
        "topic_insights": topic_insights,
    }, max_tokens=180)
    return {**default, **result}


# ══════════════════════════════════════════════════════════════════
# AGENT 1 — ANSWER INTERPRETER
# ══════════════════════════════════════════════════════════════════

_ANSWER_INTERPRETER_SYS = f"""
You are the Answer Interpreter Agent for a clinical chatbot serving head and neck
cancer patients. {_HNC_CONTEXT}

Your ONLY job: classify a patient's free-text answer for the current question.

MATCHING RULES:
1. EXACT MATCH — case-insensitive, ignore minor punctuation → match_type "exact"

2. IMPLICIT/SEMANTIC MATCH — use natural language understanding to determine if the
   answer clearly and unambiguously implies one specific option.
   Exception for severity questions: map numbers to ranges:
     0 → "0 — No pain/None", 1-3 → Mild, 4-6 → Moderate, 7-9 → Severe/High, 10 → Worst
   - Treat natural conversational yes/no language as valid yes/no:
     "yeah", "yep", "yup", "sure" → Yes
     "nope", "nah", "not really" → No
   - When a question asks for a body location, any real body part is a meaningful answer.
     Do NOT reject "hand", "head", "jaw", "neck", "arm", etc. just because it is not
     one of the named specific options.
   - For location questions with a catch-all option such as "Somewhere else" or "Other",
     any concrete anatomical location that is not one of the named specific options MUST
     map to the catch-all option.
   - This includes locations such as nose, face, cheek, scalp, lip, gums, ear, shoulder,
     chest, back, stomach, leg, foot, or any other specific body area.

3. CATCH-ALL OPTION RULE (CRITICAL) — if the options list contains a catch-all such
   as "Somewhere else", "Other", "None of these", or "Something else", AND the
   patient's answer does not match any specific option but IS a valid, meaningful
   response to the question, you MUST map it to the catch-all option.
   - match_type = "implicit", confidence = 0.85
   - Examples for options ["Throat", "Tongue", "Somewhere else"]:
       "headache" → "Somewhere else"   (it's a valid pain location, just not listed)
       "my hand"  → "Somewhere else"
       "jaw"      → "Somewhere else"
       "shoulder" → "Somewhere else"
       "ear"      → "Somewhere else"
       "neck"     → "Somewhere else"
       "nose"     → "Somewhere else"
       "my face"  → "Somewhere else"
   - Examples for options ["Gabapentin", "Oxycodone", "Other"]:
       "Tylenol"  → "Other"
       "ibuprofen"→ "Other"
   NEVER return no_match when a catch-all option exists and the answer is a
   recognisable, meaningful response to the question asked.

4. TYPE MISMATCH DETECTION — if the question asks for one type of information but
   the patient provides a different type, return match_type "no_match" even if a
   catch-all option exists. Do NOT accept a wrong-type answer via catch-all.
   Examples of type mismatches:
     - Question asks WHERE the pain is (a location) but patient says "comes and goes",
       "all the time", "only when I swallow", "sometimes" (these are timing/pattern, not location)
     - Question asks HOW BAD the pain is (severity) but patient names a body part
     - Question asks WHEN pain started but patient describes the type of pain
   In these cases: match_type = "no_match", matched_option = null,
   reasoning should explain the type mismatch so the chatbot can ask again clearly.

5. AMBIGUITY — two or more SPECIFIC (non-catch-all) options equally plausible
   → match_type "no_match", list candidates

6. NO MATCH — answer is completely unrelated/nonsensical AND no catch-all exists
   → match_type "no_match"

6. SPECIAL STATES:
   a) DISTRESS FLAG: any expression of being unable to cope, hopelessness, suicidal
      ideation → distress_flag true
   b) URGENCY FLAG: sudden severe pain, breathing difficulty, "worst of my life",
      fever with chills, bleeding, or any red flag symptom → urgency_flag true
   c) OFF-TOPIC: answer is entirely unrelated to the question (e.g. patient asks
      about appointment scheduling when asked about pain location) → match_type "off_topic"
   d) INVALID: empty or gibberish → match_type "invalid"

CONFIDENCE: 1.0 exact, 0.85-0.95 strong implicit, 0.85 catch-all, <0.7 → no_match.
matched_option MUST be copied VERBATIM from options list, or null.

TOPIC-HISTORY RULES:
  - You will receive recent conversation history for this topic only.
  - Use it to resolve conversational replies like "yes", "no", "only soup", "my sister helps some",
    or "every day almost" in context of the current question.
  - Do not treat the patient's answer as unrelated just because it is brief; use the immediate topic history.
  - Do not force a mapping if the answer is meaningful but clearly does not fit any option; prefer the catch-all option when available.
  - If the patient already gave a concrete location, medication, food type, support source, or other real-world example,
    preserve that meaning by mapping to the correct catch-all option instead of asking them to classify it themselves.

Return ONLY valid JSON:
{{
  "match_type": "exact|implicit|no_match|off_topic|invalid",
  "matched_option": "..." or null,
  "confidence": 0.0-1.0,
  "candidates": [],
  "distress_flag": false,
  "urgency_flag": false,
  "reasoning": "One sentence."
}}
"""


def run_answer_interpreter(step: dict, patient_answer: str, topic_history: Optional[list[dict[str, str]]] = None) -> dict:
    """
    Agent 1: Classify patient's free-text answer against predefined options.
    Returns interpreter output dict, or safe default on failure.
    """
    default = {
        "match_type": "no_match", "matched_option": None, "confidence": 0.0,
        "candidates": [], "distress_flag": False, "urgency_flag": False,
        "reasoning": "Agent unavailable."
    }
    if not patient_answer.strip():
        return {**default, "match_type": "invalid"}

    result = _call_agent(_ANSWER_INTERPRETER_SYS, {
        "question_text": step.get("text", ""),
        "options": step.get("opts", []),
        "patient_answer": patient_answer,
        "recent_topic_history": topic_history or [],
    }, max_tokens=200)

    if not result:
        return default

    # Validate matched_option is actually in the options list
    mo = result.get("matched_option")
    if mo and mo not in step.get("opts", []):
        result["matched_option"] = None
        result["match_type"] = "no_match"

    return {**default, **result}


# ══════════════════════════════════════════════════════════════════
# AGENT 2 — PRIOR CHECK-IN COMPARISON
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# AGENT 2 — URGENCY & CRITICALITY
# ══════════════════════════════════════════════════════════════════

_URGENCY_SYS = f"""
You are the Urgency & Criticality Agent for a clinical chatbot serving head and
neck cancer patients. {_HNC_CONTEXT}

You monitor patient safety. Read ALL raw answers across the session — urgency
signals often appear in free-text not captured by structured options.
When in doubt, flag. A false positive is far less harmful than a missed crisis.

RED FLAGS TO DETECT:
{_RED_FLAGS}

TIER DEFINITIONS:
  0 — NO URGENCY: Continue normally.
  1 — WATCH: Notable signal. Log for report, continue session normally.
  2 — URGENT: Care team must contact patient today. Continue session.
       Show one care team message to the patient.
  3 — EMERGENCY: Immediate threat. Terminate session. Patient to emergency services.

SIGNAL RULES:

  IMPORTANT CLINICAL CONTEXT FOR HNC PATIENTS:
  Pain scores of 7-8/10 are clinically expected during active chemoradiation for head
  and neck cancer — mucositis, dysphagia, and treatment toxicity routinely produce this
  level. A high pain score ALONE does not warrant Tier 2 escalation in this population.
  You MUST look for additional signals beyond the number itself.

  Medical signals:
    M1 — SEVERE UNCONTROLLED PAIN:
      Tier 1 (WATCH): Pain 7-9/10 alone, without other signals.
      Tier 2 (URGENT): Pain 7-9/10 AND at least one of:
        - Patient states nothing helps OR medication is not working
        - Patient cannot eat, sleep, or perform basic activities because of pain
        - Pain is new or suddenly much worse than their baseline
        - Fever/chills present alongside severe pain
      Tier 2 (URGENT): Pain reported as 10/10 or "worst of my life" or "unbearable"
    M2: Sudden new severe pain in head/neck — Tier 2
    M3: Fever with chills in an HNC patient (immunocompromised) — Tier 2 always
    M4: Complete inability to swallow ANY liquids — Tier 2
    M5: Breathing difficulty at rest or wheezing — Tier 2
    M6: Prescription medication suddenly stopped — Tier 2
    M7: Reported falls — Tier 1
    M8: Significant functional disruption (3+ nights no sleep, cannot eat for days) — Tier 2
  
  Psychological signals:
    P1: Explicit suicidal ideation or self-harm intent → TIER 3 ALWAYS
    P2: Passive death wish ("I don't care if I make it") → Tier 2
    P3: Crisis desperation ("I can't take this anymore" about life, not just pain) → Tier 2
    P4: Complete isolation ("no one to help me") with high pain → Tier 2

  Accumulation: 3+ Tier 1 signals in same session → escalate to Tier 2.

IMPORTANT NON-ESCALATION GUARDRAILS:
  - Do NOT escalate just because an answer is brief, partial, or missing one detail.
  - Do NOT escalate just because the patient does not remember a dose, timing, or exact amount.
  - Do NOT escalate just because a patient reports PRN or non-daily use without saying it is prescribed daily.
  - Do NOT treat "every 2 days", "sometimes", or similar medication-use frequency by itself as urgent.
  - Medication adherence becomes urgent only if the patient clearly reports they stopped an important prescribed medication,
    cannot access it, or their symptoms are uncontrolled because they are missing it.
  - If the patient gives usable but incomplete information, continue normally unless another red flag is clearly present.

PATIENT MESSAGES (verbatim — do not modify):
  Tier 2: "Thank you for sharing this with us. We can see you're having a really
    difficult time. A member of your care team will be reaching out to you today.
    Please keep your phone nearby. Your responses have been saved."
  Tier 3 (medical): "We're concerned about what you've shared. Please call 911 or
    go to your nearest emergency room immediately. Your care team has been notified."
  Tier 3 (P1/P2): "We hear you, and we want you to know your safety matters.
    Please call or text 988 right now — the Suicide & Crisis Lifeline is available
    24/7. If you are in immediate danger, call 911 or go to your nearest emergency
    room. Your care team has been notified."

Return ONLY valid JSON:
{{
  "session_tier": 0-3,
  "new_signals": ["M1", "P3"],
  "escalation_reason": "..." or null,
  "patient_message": "..." or null,
  "continue_session": true/false,
  "clinical_note": "..." or null
}}
"""


def run_urgency_agent(
    step: dict,
    current_answer_raw: str,
    current_answer_matched: Optional[str],
    session_answers: dict,
    prior_baseline: dict,
    active_signals: list,
    distress_flag: bool,
    urgency_flag: bool,
) -> dict:
    """
    Agent 3: Patient safety monitoring. Returns urgency assessment.
    """
    default = {
        "session_tier": 0, "new_signals": [], "escalation_reason": None,
        "patient_message": None, "continue_session": True, "clinical_note": None
    }
    result = _call_agent(_URGENCY_SYS, {
        "current_question": step.get("text", ""),
        "current_answer_raw": current_answer_raw,
        "current_answer_matched": current_answer_matched,
        "distress_flag_from_interpreter": distress_flag,
        "urgency_flag_from_interpreter": urgency_flag,
        "all_session_answers_so_far": session_answers,
        "active_signals_this_session": active_signals,
        "prior_baseline_summary": prior_baseline,
    }, max_tokens=300)

    if not result:
        return default

    # Safety: P1 (suicidal ideation) from the distress flag always forces Tier 3
    if distress_flag and "suicid" in current_answer_raw.lower():
        result["session_tier"] = 3
        result["continue_session"] = False

    return {**default, **result}


# ══════════════════════════════════════════════════════════════════
# AGENT 4 — PATIENT SENTIMENT & ENGAGEMENT MONITOR
# ══════════════════════════════════════════════════════════════════

_SENTIMENT_SYS = f"""
You are the Patient Sentiment & Engagement Monitor for a clinical chatbot serving
head and neck cancer patients. {_HNC_CONTEXT}

You track the patient's emotional state and engagement quality across the session.
You do NOT classify safety crises — that belongs to the Urgency Agent.

IMPORTANT POPULATION CONTEXT:
  - Brief answers ≠ disengagement (pain impairs fluency)
  - Stoicism is common — calibrate against reported pain level
  - Emotional flooding (long distressed answers) needs acknowledgment before next Q

DIMENSION SCORES:
  emotional_state: positive|neutral|fatigued|distressed|frustrated|anxious|overwhelmed|resigned
  engagement_level: high|moderate|low|resistant|confused
  engagement_trajectory: stable|improving|declining|insufficient_data

SIGNALS TO DETECT (set to true if present):
  E3_resistance: patient explicitly pushes back ("I already told you", "can we be done")
  E7_wants_to_stop: "I need to stop", "I'm done", "I can't do this right now"
  EM1_pain_frustration: venting about their pain situation
  EM2_sadness_grief: expressing loss or grief about what they can no longer do
  EM6_emotional_flooding: unusually long, distressed, emotionally dense answer

ADAPTATION SIGNALS:
  tone_profile: standard|warm|gentle|simplified
  acknowledgment_required: true if EM1/EM2/EM6 active in current answer
  acknowledgment_text: ≤25 words, first person chatbot voice, warm but not effusive,
    reflects what patient shared emotionally (NOT their clinical data back at them).
    Example: "That sounds really hard to carry. Thank you for sharing that."
  simplify_next_question: true if patient seems confused or cognitively fatigued
  reduce_follow_up_depth: true if E3/E7 active or engagement declining

Return ONLY valid JSON:
{{
  "emotional_state": "...",
  "engagement_level": "...",
  "engagement_trajectory": "...",
  "signals": {{
    "E3_resistance": false,
    "E7_wants_to_stop": false,
    "EM1_pain_frustration": false,
    "EM2_sadness_grief": false,
    "EM6_emotional_flooding": false
  }},
  "adaptation": {{
    "tone_profile": "standard",
    "acknowledgment_required": false,
    "acknowledgment_text": null,
    "simplify_next_question": false,
    "reduce_follow_up_depth": false
  }},
  "engagement_note_for_doctor": null
}}
"""


def run_sentiment_agent(
    step: dict,
    current_answer_raw: str,
    session_answers: dict,
    active_signals: list,
    question_count: int,
) -> dict:
    """
    Agent 4: Track patient sentiment and engagement. Returns adaptation signals.
    """
    default = {
        "emotional_state": "neutral", "engagement_level": "moderate",
        "engagement_trajectory": "insufficient_data",
        "signals": {
            "E3_resistance": False, "E7_wants_to_stop": False,
            "EM1_pain_frustration": False, "EM2_sadness_grief": False,
            "EM6_emotional_flooding": False,
        },
        "adaptation": {
            "tone_profile": "standard", "acknowledgment_required": False,
            "acknowledgment_text": None, "simplify_next_question": False,
            "reduce_follow_up_depth": False,
        },
        "engagement_note_for_doctor": None,
    }
    result = _call_agent(_SENTIMENT_SYS, {
        "current_question": step.get("text", ""),
        "current_answer": current_answer_raw,
        "questions_answered_so_far": question_count,
        "all_session_answers": session_answers,
        "active_sentiment_signals": active_signals,
    }, max_tokens=300)

    if not result:
        return default

    # Merge nested dicts carefully
    merged = {**default}
    merged["emotional_state"] = result.get("emotional_state", default["emotional_state"])
    merged["engagement_level"] = result.get("engagement_level", default["engagement_level"])
    merged["engagement_trajectory"] = result.get("engagement_trajectory", default["engagement_trajectory"])
    merged["engagement_note_for_doctor"] = result.get("engagement_note_for_doctor")

    if "signals" in result and isinstance(result["signals"], dict):
        merged["signals"] = {**default["signals"], **result["signals"]}
    if "adaptation" in result and isinstance(result["adaptation"], dict):
        merged["adaptation"] = {**default["adaptation"], **result["adaptation"]}

    return merged


# ══════════════════════════════════════════════════════════════════
# AGENT 5 — DOCTOR-RELEVANCE
# ══════════════════════════════════════════════════════════════════

_DOCTOR_RELEVANCE_SYS = f"""
You are the Doctor-Relevance Agent for a clinical chatbot serving head and neck
cancer patients. {_HNC_CONTEXT}

You evaluate patient answers from the physician's perspective in one pass. Your outputs:
  1. Clinical sufficiency verdict on the current answer
  2. Follow-up recommendation with a precise information GOAL (not the question itself)
  3. Comparison to the same question from the prior visit when prior data exists
  4. A compact doctor-facing note (≤35 words, third person)
  5. A short patient-facing comparison or acknowledgment note only when useful

FOLLOW-UP RULES:
  - The question list is a question bank, not a rigid script. Judge the current answer
    like a clinician deciding whether anything important is still missing.
  - A meaningful free-text answer in the patient's own words is clinically usable even if it does not match the option wording.
  - If a free-text question contains yes/no wording and the patient gives a simple "yes" or "no", treat that as minimally usable data unless the question clearly asked for a descriptive detail like where, when, how often, or what kind.
  - If the patient gave a broad but meaningful answer, break down what is missing conceptually; do NOT treat it as meaningless.
  - If the patient gives a meaningful negative screen ("no", "not really", "I am okay", "fine") to a broad symptom or emotional check-in question, treat that as a usable answer rather than forcing an unnecessary impact follow-up.
  - If the patient gives a negative screen to a broad opener, set screen_negative_signal=true when the next likely question would otherwise just ask about downstream impact of the same denied problem.
  - If the patient clearly indicates they do not have a problem in that domain, prefer skipping nonessential downstream questions rather than completing the whole branch mechanically.
  - Ask only questions that are still clinically necessary after the patient's actual answer.
  - If the patient already explained the reason in their own words, do NOT recommend a generic "what is making this difficult" follow-up.
  - If the patient supplies one detail and explicitly does not know another, accept the known detail and only ask for the missing one if it is truly necessary.
  - If the missing detail is something the patient reasonably may not know right now, prefer no follow-up over repetitive questioning.
  - Never imply the presence of a symptom the patient just denied.
  - You will receive recent conversation history for this topic only. Use it to avoid repeated questions.
  - Do NOT create a custom follow-up whose only purpose is to ask the same thing as the candidate next step in different words.
  - If the next formal step already covers the natural next question, prefer no custom follow-up and let that next step be asked once.
  - A patient should never have to answer a natural-language version of a question and then immediately answer the form version of the same question.
  - This is especially important after structured option answers like Yes/No or category selections: if the next formal step can ask the next needed detail directly, prefer no custom bridge follow-up.
  - If the current answer already addresses the candidate next step, set next_step_action to skip that step.
  - If several upcoming questions become unnecessary for the same reason, include them in next_step_action.plan.
  - If the patient's raw wording already fully answers the candidate next step, skip that step and carry the raw detail forward instead of asking it again.
  - If the candidate next step, or any proposed follow-up, would substantially repeat a recent question already asked in this topic, suppress it.
  - If a natural assistant acknowledgment has already effectively asked the next question, do not ask it again.
  - If the patient explicitly says they do not know a detail, treat that as usable uncertainty rather than pushing repeatedly.
  - If the patient provided some but not all of the detail, mark it as partial and describe the single missing detail in follow_up_goal.
  - ONLY recommend follow-up if information_completeness is "partial" or "none"
    AND follow_up_count is 0 AND the missing info is clinically meaningful
  - NEVER recommend follow-up if follow_up_count ≥ 1 (absolute limit: 1 per question)
  - NEVER recommend follow-up if patient showed resistance in their answer

PRIOR-COMPARISON RULES:
  - You will receive last_checkin_answer for this same question when available.
  - Compare current answer to the same question from the prior visit.
  - change_direction:
      improved       — current answer suggests less pain / better status
      worsened       — current answer suggests more pain / worse status
      neutral_change — changed but direction unclear
      no_change      — answers are the same or equivalent
      new_data       — no prior data available
  - change_magnitude:
      numeric severity: large=3+ points difference, moderate=2, small=1, none=0
      non-numeric: large if clinically major, moderate for meaningful change, small for minor wording difference
  - clinical_note:
      one short plain-English comparison sentence; if no prior data, say no prior data available
  - patient_facing_note:
      one short natural sentence only when the comparison adds value
      good uses: weight up/down, pain improved/worsened, symptom burden better/worse
      do NOT use for trivial yes/no comparisons like "Last time you said yes"
  - patient_acknowledgment:
      optional short acknowledgment when uncertainty itself should be accepted naturally
      example: "Thanks for sharing that. It's okay if you're not sure of the exact dose."

SPECIAL CLINICAL SIGNALS (set if present):
  trajectory_mismatch: patient says improving but comparison shows worsening (or vice versa)
  medication_stop_signal: patient stopped taking prescription medication without explanation
  aggravating_medication_signal: patient reports their medication makes symptoms worse
  severity_underreporting: patient rates low severity but describes severe functional impact
  screen_negative_signal: patient's answer functions as a meaningful negative screen for the symptom/concern being assessed

next_step_action:
  - Use this to suppress an immediate next question when it no longer makes clinical sense
  - This is the main mechanism for skipping downstream impact/management/change questions generically across topics
  - Only use it when the candidate next step would be unnecessary, redundant, or context-mismatched given the current answer and session answers
  - Prefer suggested_answer to be an exact option from the candidate next step when obvious, often "No"
  - plan is optional and may list additional upcoming steps that should also be auto-resolved to avoid unnecessary questioning
  - carry_forward_answer is optional and should be used when the patient's current raw answer already provides the value for a downstream step, especially a free-text detail step that would otherwise repeat the same question
  - Good examples:
    - Patient denies emotional distress and next step asks whether anxiety is affecting sleep/eating → skip with suggested_answer "No"
    - Patient denies depression or feeling down and the next questions only elaborate on mood burden or support needs → skip them unless there is another clear concern
    - Patient answers a location chooser with a specific body part like "nose" and the next step asks which body part hurts → skip that next step and carry forward "nose"
    - Patient says a sore is not painful and next step asks whether treatment for painful sores is helping → skip with suggested_answer "No"
    - Patient says IV fluids are helping and gives no sign they want changes, and next step asks about adjusting frequency → skip with suggested_answer "No"
    - Patient says medication is not causing drowsiness and next step asks whether drowsiness is affecting schedule → skip with suggested_answer "No"
  - Do not use this to skip structurally essential questions like a severity rating or a new symptom location unless the current answer already fully covers them

follow_up_goal: A statement of WHAT information is needed — NOT a question.
  Example: "Obtain a numeric pain severity score — patient described pain without rating it."

information_completeness:
  complete — answer fully satisfies the clinical information need
  partial  — has some value but key information missing
  none     — no clinically usable information

change_significance:
  critical — large worsening, urgency flag, new alarming symptom, medication stopped
  notable  — meaningful change worth highlighting
  stable   — no meaningful change
  no_baseline — first session or no prior data

clinical_priority: high | medium | low
doctor_note: ≤35 words, third person, factual only, include comparison if change is critical/notable.

Return ONLY valid JSON:
{{
  "information_completeness": "complete|partial|none",
  "clinical_value_score": 0.0-1.0,
  "follow_up_recommended": true/false,
  "follow_up_goal": "..." or null,
  "follow_up_urgency": "immediate|routine|none",
  "patient_acknowledgment": "..." or null,
  "answered_with_uncertainty": false,
  "has_prior_data": true/false,
  "last_answer": "..." or null,
  "change_detected": true/false,
  "change_direction": "improved|worsened|neutral_change|no_change|new_data",
  "change_magnitude": "large|moderate|small|none",
  "clinical_note": "..." or null,
  "patient_facing_note": "..." or null,
  "change_significance": "critical|notable|stable|no_baseline",
  "clinical_priority": "high|medium|low",
  "doctor_note": "..." or null,
  "next_step_action": {{
    "skip_immediate_next_step": false,
    "suggested_answer": "..." or null,
    "reason": "..." or null,
    "carry_forward_answer": "..." or null,
    "plan": [
      {{
        "step_id": "...",
        "suggested_answer": "..." or null,
        "carry_forward_answer": "..." or null,
        "reason": "..." or null
      }}
    ]
  }},
  "special_signals": {{
    "trajectory_mismatch": false,
    "medication_stop_signal": false,
    "aggravating_medication_signal": false,
    "severity_underreporting": false,
    "screen_negative_signal": false
  }}
}}
"""


def run_doctor_relevance(
    step: dict,
    current_answer_raw: str,
    current_answer_matched: Optional[str],
    last_topic_data: dict,
    session_answers: dict,
    followup_count: int,
    topic_history: list[dict[str, str]],
    recent_questions: list[str],
    candidate_next_step: Optional[dict] = None,
    upcoming_steps: Optional[list[dict]] = None,
) -> dict:
    """
    Clinical assessment agent: assess sufficiency, prior-visit comparison,
    and follow-up relevance in one pass.
    """
    last_answer = None
    if last_topic_data:
        raw_last = last_topic_data.get(step["id"])
        if raw_last is not None:
            last_answer = str(raw_last)
    default = {
        "information_completeness": "complete", "clinical_value_score": 0.7,
        "follow_up_recommended": False, "follow_up_goal": None,
        "follow_up_urgency": "none", "change_significance": "no_baseline",
        "patient_acknowledgment": None,
        "answered_with_uncertainty": False,
        "has_prior_data": bool(last_answer),
        "last_answer": last_answer,
        "change_detected": False,
        "change_direction": "new_data" if last_answer is None else "no_change",
        "change_magnitude": "none",
        "clinical_note": "No prior data available." if last_answer is None else "",
        "patient_facing_note": None,
        "clinical_priority": "medium", "doctor_note": None,
        "next_step_action": {
            "skip_immediate_next_step": False,
            "suggested_answer": None,
            "reason": None,
            "carry_forward_answer": None,
            "plan": [],
        },
        "special_signals": {
            "trajectory_mismatch": False, "medication_stop_signal": False,
            "aggravating_medication_signal": False, "severity_underreporting": False,
            "screen_negative_signal": False,
        },
    }
    result = _call_agent(_DOCTOR_RELEVANCE_SYS, {
        "question_text": step.get("text", ""),
        "question_type": step.get("type", "options"),
        "options": step.get("opts", []),
        "current_answer_raw": current_answer_raw,
        "current_answer_matched": current_answer_matched,
        "last_checkin_answer": last_answer,
        "session_answers_so_far": session_answers,
        "recent_topic_history": topic_history,
        "recent_question_texts": recent_questions,
        "follow_up_count_this_question": followup_count,
        "candidate_next_step": {
            "id": candidate_next_step.get("id"),
            "text": candidate_next_step.get("text"),
            "type": candidate_next_step.get("type"),
            "options": candidate_next_step.get("opts", []),
        } if candidate_next_step else None,
        "upcoming_steps": [
            {
                "id": s.get("id"),
                "text": s.get("text"),
                "type": s.get("type"),
                "options": s.get("opts", []),
            }
            for s in (upcoming_steps or [])
        ],
    }, max_tokens=400)

    if not result:
        return default

    # Hard-enforce the follow-up limit
    if followup_count >= 1:
        result["follow_up_recommended"] = False
        result["follow_up_goal"] = None

    merged = {**default, **result}
    if "next_step_action" in result and isinstance(result["next_step_action"], dict):
        merged["next_step_action"] = {**default["next_step_action"], **result["next_step_action"]}
    if "special_signals" in result and isinstance(result["special_signals"], dict):
        merged["special_signals"] = {**default["special_signals"], **result["special_signals"]}

    return merged


# ══════════════════════════════════════════════════════════════════
# AGENT 6 — NEXT-MOVE (FOLLOW-UP QUESTION AUTHOR)
# ══════════════════════════════════════════════════════════════════

_NEXT_MOVE_SYS = f"""
You are the Next-Move Agent for a clinical chatbot serving head and neck cancer patients.
{_HNC_CONTEXT}

You receive a follow-up GOAL and write the actual follow-up question the patient sees.
The decision to follow up has already been made. Your job is HOW to ask it.

TONE PROFILES:
  standard  — professional, warm, clear
  warm      — add genuine warmth; soften clinical phrasing; sincere, not effusive
  gentle    — softest possible; avoid anything demanding or clinical
  simplified — short sentences, very simple words, one idea only

RULES:
  - Treat the original form question as background only; you are not tied to its exact wording
  - Ask the most clinically useful next single question, as a doctor or nurse naturally would
  - Stay anchored to the patient's last answer; the follow-up should feel like a direct continuation of what they just said
  - If the patient used plain-language wording, mirror that wording naturally instead of switching back to rigid form language
  - Ask only for the single missing detail; never restate details the patient already provided
  - If the patient said they do not know a detail, do not challenge that or sound repetitive
  - Never contradict an explicit "no" or "yes" the patient just gave
  - Never write a follow-up like "Besides anxiety..." or otherwise imply a symptom exists after the patient denied it
  - You will receive recent topic history and recent question texts from this topic only
  - Do not write a question that substantially repeats any recent question in that history
  - If the candidate next step already asks the same thing, return null instead of paraphrasing it
  - After a structured option answer, do not pre-ask the next formal step in different words just to sound conversational.
  - Never ask the patient to translate their own concrete answer into the form's categories. For example, after a patient says "nose", do not ask "throat, tongue, or somewhere else?" because that classification should happen internally.
  - If the patient's answer already gives a concrete real-world example, assume the system can preserve it and ask only the next clinically meaningful question.
  - If prior-comparison context is clinically useful, you may briefly reflect it in a natural way, but only as conversational context, never as a rigid template
  - Write in second person, conversational language
  - Never use medical jargon without immediate plain explanation
  - NEVER ask a multi-part question
  - NEVER repeat the original question verbatim
  - Keep the question to ≤25 words
  - An acknowledgment may be shown BEFORE your question — do not repeat it
  - If simplify=true: use the shortest phrasing possible

Return ONLY valid JSON:
{{
  "follow_up_question": "..." or null,
  "preamble": "..." or null
}}
preamble: ≤10 words transitional phrase if naturally needed, else null.
"""


def run_next_move_agent(
    step: dict,
    current_answer_raw: str,
    followup_goal: str,
    tone_profile: str,
    simplify: bool,
    topic_history: list[dict[str, str]],
    recent_questions: list[str],
    candidate_next_step: Optional[dict],
) -> dict:
    """
    Agent 6: Author the follow-up question in natural language.
    """
    result = _call_agent(_NEXT_MOVE_SYS, {
        "original_question": step.get("text", ""),
        "patient_answer": current_answer_raw,
        "follow_up_goal": followup_goal,
        "tone_profile": tone_profile,
        "simplify": simplify,
        "recent_topic_history": topic_history,
        "recent_question_texts": recent_questions,
        "candidate_next_step": {
            "id": candidate_next_step.get("id"),
            "text": candidate_next_step.get("text"),
            "type": candidate_next_step.get("type"),
            "options": candidate_next_step.get("opts", []),
        } if candidate_next_step else None,
    }, max_tokens=120)

    if result and "follow_up_question" in result:
        return result
    # Fallback: derive a question from the goal
    return {
        "follow_up_question": "Could you tell me a bit more about that?",
        "preamble": None,
    }


# ══════════════════════════════════════════════════════════════════
# ORCHESTRATOR — coordinates all agents
# ══════════════════════════════════════════════════════════════════

def _build_session_answers(topic_key: str) -> dict:
    """Build {question_id: raw_answer} from current session state for the topic."""
    state = st.session_state.topic_states.get(topic_key, {})
    data = state.get("data", {})
    raw_answers = state.get("raw_answers", {})
    payload = {}
    for k, v in data.items():
        if v is None:
            continue
        payload[k] = str(raw_answers.get(k, v))
    return payload


def _build_topic_history(topic_key: str) -> list[dict[str, str]]:
    state = st.session_state.topic_states.get(topic_key, {})
    return _recent_topic_history(state)


def _build_recent_question_texts(topic_key: str) -> list[str]:
    state = st.session_state.topic_states.get(topic_key, {})
    return _recent_topic_questions(state)


def _build_prior_baseline(topic_key: str) -> dict:
    """Return a compact prior check-in summary for agent context."""
    last = st.session_state.last_checkin.get(topic_key, {})
    if not last:
        return {}
    # Return key fields only to keep the payload small
    keys = list(last.keys())[:10]
    return {k: str(last[k]) for k in keys}


def _build_all_topic_data() -> dict:
    payload = {}
    for _, key in TOPICS:
        topic_state = st.session_state.topic_states[key]
        topic_data = dict(topic_state.get("data", {}))
        raw_answers = topic_state.get("raw_answers", {})
        if raw_answers:
            topic_data["_verbatim_answers"] = dict(raw_answers)
        payload[key] = topic_data
    return payload


def run_agent_pipeline(
    topic_key: str,
    step: dict,
    answer: str,
    raw_answer: Optional[str],
    state: dict,
    last_topic_data: dict,
) -> dict:
    """
    Orchestrator: runs all agents in sequence (with parallelism where safe)
    and returns a unified decision dict consumed by handle_answer.

    Returns:
    {
        "matched_option": str|None,
        "follow_up": bool,
        "follow_up_question": str,
        "acknowledgment": str,
        "assistant_message": str,
        "urgency_tier": int,
        "urgency_message": str|None,
        "reduce_follow_up": bool,
        "wants_to_stop": bool,
        "doctor_note": str|None,
        "clinical_priority": str,
        "change_significance": str,
        "change_clinical_note": str,
        "special_signals": dict,
        "sentiment_note": str|None,
    }
    """
    if not openai_client:
        return _pipeline_default()

    question_count = len(state.get("data", {}))
    current_raw_answer = str(raw_answer if raw_answer is not None else answer)
    session_answers = _build_session_answers(topic_key)
    prior_baseline  = _build_prior_baseline(topic_key)
    followup_count  = state.get("followup_counts", {}).get(step["id"], 0)
    topic_history = _build_topic_history(topic_key)
    recent_questions = _build_recent_question_texts(topic_key)
    candidate_next_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
    upcoming_steps = get_upcoming_steps(topic_key, state["data"], state.get("raw_answers"), limit=5)

    # ── STEP 1: Answer Interpreter (must run first) ────────────────
    interp = run_answer_interpreter(step, current_raw_answer, topic_history=topic_history)
    matched = interp.get("matched_option")
    distress = interp.get("distress_flag", False)
    urgency_flag = interp.get("urgency_flag", False)
    # ── STEP 2: Run urgency and sentiment in parallel ──────────────
    urgency_out = {}
    sentiment_out = {}

    active_urgency_signals = st.session_state.get("urgency_state", {}).get("all_signals", [])
    active_sentiment_signals = st.session_state.get("sentiment_state", {}).get("all_signals", [])

    def _run_urgency():
        return run_urgency_agent(
            step, current_raw_answer, matched, session_answers, prior_baseline,
            active_urgency_signals, distress, urgency_flag
        )

    def _run_sentiment():
        return run_sentiment_agent(
            step, current_raw_answer, session_answers, active_sentiment_signals, question_count
        )

    with _futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_urgency   = pool.submit(_run_urgency)
        f_sentiment = pool.submit(_run_sentiment)
        urgency_out  = f_urgency.result()
        sentiment_out = f_sentiment.result()

    # ── STEP 3: Urgency interrupt check ───────────────────────────
    tier = urgency_out.get("session_tier", 0)

    # Update session-level urgency state
    _merge_urgency_state(tier, urgency_out)

    if tier == 3:
        # Emergency — bypass all other agents
        return {
            **_pipeline_default(),
            "matched_option": matched,
            "urgency_tier": 3,
            "urgency_message": urgency_out.get("patient_message"),
            "wants_to_stop": True,
        }

    # ── STEP 4: Doctor-Relevance ───────────────────────────────────
    dr_out = run_doctor_relevance(
        step, current_raw_answer, matched, last_topic_data, session_answers, followup_count,
        topic_history=topic_history, recent_questions=recent_questions,
        candidate_next_step=candidate_next_step,
        upcoming_steps=upcoming_steps,
    )

    # ── STEP 5: Apply follow-up decision logic ─────────────────────
    adapt = sentiment_out.get("adaptation", {})
    sigs  = sentiment_out.get("signals", {})
    reduce = adapt.get("reduce_follow_up_depth", False)
    wants_to_stop = sigs.get("E7_wants_to_stop", False)

    dr_recommends  = dr_out.get("follow_up_recommended", False)
    followup_goal  = dr_out.get("follow_up_goal", "")
    priority       = dr_out.get("clinical_priority", "medium")

    # Override rules (clinical necessity > patient experience)
    force_followup = False
    if step.get("type") == "number" and dr_out.get("information_completeness") != "complete":
        force_followup = True  # Numeric severity is always high priority
    if dr_out.get("special_signals", {}).get("medication_stop_signal") and followup_count == 0:
        force_followup = True

    # Suppression rules
    suppress = False
    if followup_count >= 1:
        suppress = True  # Absolute limit
    if wants_to_stop:
        suppress = True
    if interp.get("match_type") in ("off_topic", "invalid"):
        suppress = True
    if reduce and priority != "high":
        suppress = True
    if sigs.get("E3_resistance") and priority != "high":
        suppress = True
    if (
        step.get("type") == "options"
        and candidate_next_step
        and matched in (step.get("opts") or [])
        and dr_out.get("information_completeness") == "complete"
        and not force_followup
    ):
        suppress = True

    do_follow_up = (force_followup or dr_recommends) and not suppress

    # ── STEP 6: Compose follow-up question if needed ───────────────
    follow_up_question = ""
    if do_follow_up and followup_goal:
        tone = adapt.get("tone_profile", "standard")
        simplify = adapt.get("simplify_next_question", False)
        nm_out = run_next_move_agent(
            step, current_raw_answer, followup_goal, tone, simplify,
            topic_history=topic_history, recent_questions=recent_questions,
            candidate_next_step=candidate_next_step,
        )
        preamble = nm_out.get("preamble") or ""
        fq = nm_out.get("follow_up_question", "")
        follow_up_question = f"{preamble} {fq}".strip() if preamble and fq else fq
        if _question_already_asked(state, follow_up_question):
            follow_up_question = ""
            do_follow_up = False

    # ── STEP 7: Build assistant message for non-follow-up case ─────
    assistant_message = ""
    if not do_follow_up:
        comp_note    = dr_out.get("clinical_note", "")
        patient_change_note = (dr_out.get("patient_facing_note") or "").strip()
        change_dir   = dr_out.get("change_direction", "new_data")
        prev_answer  = dr_out.get("last_answer", "")
        emotional    = sentiment_out.get("emotional_state", "neutral")

        # Build a brief contextual acknowledgment
        if dr_out.get("patient_acknowledgment") and dr_out.get("answered_with_uncertainty"):
            assistant_message = dr_out["patient_acknowledgment"]
        elif patient_change_note:
            assistant_message = patient_change_note
        elif comp_note and change_dir in ("worsened", "improved") and prev_answer:
            assistant_message = comp_note
        elif emotional == "distressed":
            assistant_message = "That sounds really difficult. I've made a note of this for your care team."
        elif emotional in ("anxious", "overwhelmed"):
            assistant_message = "I hear you — I've made a note of that for your care team."
        elif change_dir == "worsened":
            assistant_message = "I've noted that, and I can see things have been harder than last time."
        elif change_dir == "improved":
            assistant_message = "That's helpful to know, and it sounds like there's been some improvement since last time."
        else:
            assistant_message = _default_chatty_reply(topic_key, answer, step, last_topic_data)

    # ── STEP 8: Compose acknowledgment if needed ───────────────────
    acknowledgment = ""
    if adapt.get("acknowledgment_required") and adapt.get("acknowledgment_text"):
        acknowledgment = adapt["acknowledgment_text"]

    # ── STEP 9: Merge urgency and sentiment state ──────────────────
    _merge_sentiment_state(sentiment_out)

    # ── STEP 10: Tier 2 notice ─────────────────────────────────────
    urgency_msg = None
    urg_state = st.session_state.get("urgency_state", {})
    if tier == 2 and not urg_state.get("escalation_shown", False):
        urgency_msg = urgency_out.get("patient_message")
        urg_state["escalation_shown"] = True
        st.session_state["urgency_state"] = urg_state

    return {
        "matched_option": matched,
        "follow_up": do_follow_up,
        "follow_up_question": follow_up_question,
        "acknowledgment": acknowledgment,
        "assistant_message": assistant_message,
        "urgency_tier": tier,
        "urgency_message": urgency_msg,
        "reduce_follow_up": reduce,
        "wants_to_stop": wants_to_stop,
        "doctor_note": dr_out.get("doctor_note"),
        "clinical_priority": priority,
        "change_significance": dr_out.get("change_significance", "no_baseline"),
        "change_clinical_note": dr_out.get("clinical_note", ""),
        "next_step_action": dr_out.get("next_step_action"),
        "special_signals": dr_out.get("special_signals", {}),
        "sentiment_note": sentiment_out.get("engagement_note_for_doctor"),
    }


def _pipeline_default() -> dict:
    """Safe default when agents are unavailable."""
    return {
        "matched_option": None, "follow_up": False, "follow_up_question": "",
        "acknowledgment": "", "assistant_message": "",
        "urgency_tier": 0, "urgency_message": None,
        "reduce_follow_up": False, "wants_to_stop": False,
        "doctor_note": None, "clinical_priority": "medium",
        "change_significance": "no_baseline", "change_clinical_note": "",
        "next_step_action": None,
        "special_signals": {}, "sentiment_note": None,
    }


def _merge_urgency_state(tier: int, urgency_out: dict):
    """Merge new urgency signals into session-level urgency state."""
    state = st.session_state.get("urgency_state", {
        "current_tier": 0, "all_signals": [], "escalation_shown": False,
        "emergency_shown": False,
    })
    state["current_tier"] = max(state.get("current_tier", 0), tier)
    new_sigs = urgency_out.get("new_signals", [])
    existing = state.get("all_signals", [])
    state["all_signals"] = list(set(existing + new_sigs))
    st.session_state["urgency_state"] = state


def _merge_sentiment_state(sentiment_out: dict):
    """Merge new sentiment signals into session-level sentiment state."""
    state = st.session_state.get("sentiment_state", {"all_signals": []})
    new_sigs = [k for k, v in sentiment_out.get("signals", {}).items() if v]
    existing = state.get("all_signals", [])
    state["all_signals"] = list(set(existing + new_sigs))
    state["engagement_trajectory"] = sentiment_out.get("engagement_trajectory", "insufficient_data")
    state["emotional_state"] = sentiment_out.get("emotional_state", "neutral")
    st.session_state["sentiment_state"] = state


# ══════════════════════════════════════════════════════════════════
# LEGACY SUPPORT: keep interpret_user_input_with_options working
# ══════════════════════════════════════════════════════════════════

def interpret_user_input_with_options(step, user_input, topic_history: Optional[list[dict[str, str]]] = None):
    """
    Use the Answer Interpreter Agent to classify free-text against question options.
    Falls back to the catch-all option (Somewhere else / Other) if the agent returns
    no_match but a catch-all exists and the answer is a real, meaningful response.
    Returns matched option string if found, else original input.
    """
    if not step.get("opts"):
        return user_input

    normalized = _norm_text(user_input)
    for opt in step.get("opts", []):
        if _norm_text(opt) == normalized:
            return opt

    opts = step.get("opts", [])
    if (
        "Somewhere else" in opts
        and _looks_like_body_location_phrase(user_input)
        and ("where" in _norm_text(step.get("text", "")) or "location" in _norm_text(step.get("text", "")))
    ):
        return "Somewhere else"

    if not openai_client:
        return user_input

    result = run_answer_interpreter(step, user_input, topic_history=topic_history)
    matched = result.get("matched_option")

    if matched and matched in step.get("opts", []):
        return matched

    return user_input


# ══════════════════════════════════════════════════════════════════
# URGENCY BANNER RENDERER
# ══════════════════════════════════════════════════════════════════

def render_urgency_banner():
    """
    Render a coloured urgency banner at the top of the main content area.
    Only shown if urgency_state tier >= 2.
    """
    urg = st.session_state.get("urgency_state", {})
    tier = urg.get("current_tier", 0)
    if tier == 0:
        return

    if tier == 1:
        st.markdown(
            '<div style="background:#fff8e8;border:1px solid #f9c846;border-radius:14px;'
            'padding:10px 14px;margin-bottom:12px;font-size:13px;color:#7a5a00;">'
            '⚠️ <strong>Note for your care team:</strong> Some of your responses have been '
            'flagged for additional review before your visit.'
            '</div>',
            unsafe_allow_html=True,
        )
    elif tier == 2:
        msg = urg.get("patient_message") or (
            "Thank you for sharing this with us. A member of your care team will be "
            "reaching out to you today. Please keep your phone nearby."
        )
        st.markdown(
            f'<div style="background:#fff3f3;border:1.5px solid #e87a7a;border-radius:14px;'
            f'padding:12px 16px;margin-bottom:12px;font-size:13.5px;color:#7a1010;">'
            f'🔴 <strong>Care team notice:</strong> {msg}'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif tier == 3:
        msg = urg.get("patient_message") or (
            "Please call 911 or go to your nearest emergency room immediately. "
            "Your care team has been notified."
        )
        st.markdown(
            f'<div style="background:#ff0000;border-radius:14px;padding:16px 18px;'
            f'margin-bottom:16px;font-size:14px;color:white;font-weight:700;">'
            f'🚨 URGENT: {msg}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_urgency_indicator_html() -> str:
    """Return a small coloured dot HTML for the sidebar."""
    tier = st.session_state.get("urgency_state", {}).get("current_tier", 0)
    colours = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444", 3: "#991b1b"}
    labels  = {0: "All clear", 1: "Monitoring", 2: "Urgent", 3: "Emergency"}
    c = colours.get(tier, "#22c55e")
    l = labels.get(tier, "")
    return (
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{c};margin-right:5px;vertical-align:middle;"></span>'
        f'<span style="font-size:11px;color:{c};">{l}</span>'
    )


# ══════════════════════════════════════════════════════════════════
# REPORT GENERATION — Doctor-Facing Report Agent
# ══════════════════════════════════════════════════════════════════

_REPORT_AGENT_SYS = f"""
You are the Doctor-Facing Report Agent for ChatReport, a clinical symptom check-in
chatbot for head and neck cancer (HNC) patients. {_HNC_CONTEXT}

You synthesise all collected session data into a structured clinical pre-visit report
for the treating physician.

THE DOCTOR'S READING CONTEXT:
An oncologist or NP reviewing reports before clinic appointments.
May read multiple reports. They need to:
  • Know in 10 seconds: is action required today?
  • Understand patient's current status in 30 seconds.
  • Have full symptom detail available if needed.
  • Know what to address or follow up at the visit.

REPORT FORMAT — use this exact structure:
---
CHATREPORT — PRE-VISIT CLINICAL SUMMARY
Patient: [name]  |  Date: [date]
═══════════════════════════════════════════════════

🔴 FLAGS FOR PROVIDER ATTENTION
[List ONLY items matching red flag criteria — each as a concise bullet.
 Include: urgency signals detected, medication stops, crisis signals.
 If none: "No urgent flags identified."]

📋 CLINICAL OVERVIEW
[2-3 sentences: current status, most prominent issues, notable changes since last visit.
 Written for a clinician who has 10 seconds to orient.]

📊 SYMPTOM DETAILS BY DOMAIN
[One bold subsection per completed topic. Include:
 - Symptom presence/severity in clinical language
 - Patient-reported management strategies and medications
 - Functional impact where reported
 - Comparison to last visit where available
 - Clinically meaningful direct quotes in quotation marks]

💊 MEDICATION SUMMARY
[List all medications mentioned, doses/frequencies if reported, adherence status,
 and whether they are helping. Note any stops or side effects reported.]

🗣️ PATIENT ENGAGEMENT NOTES
[Only include if engagement was notable — e.g., declining engagement, distress,
 stoic underreporting, or patient explicitly shared emotional content.
 Omit this section if engagement was unremarkable.]

✅ SUGGESTED DISCUSSION POINTS
[2-4 bullets: items for provider to address or follow up — medication adjustment,
 referral, patient education need, unresolved concern.
 Do NOT repeat red flags already listed above.]
---

CLINICAL LANGUAGE RULES:
- Convert patient language to clinical terms where appropriate
  (e.g., "sore in my mouth" → "oral mucositis", "can't swallow" → "dysphagia")
- Include patient's own words in quotes only when clinically meaningful
- Omit topics with no data — do not write "N/A"
- Third person throughout ("Patient reports…")
- Never write "Unfortunately" or emotional commentary

RED FLAGS TO SCREEN FOR:
{_RED_FLAGS}

Urgency state will be provided. If emergency_tier >= 2, open with the flags section prominently.
Write only the completed report. No AI disclaimers or generation notes.
"""


def generate_report(name: str, all_data: dict) -> str:
    """
    Doctor-Facing Report Agent: synthesise all session data into a clinical report.
    Falls back to plain-text summary if OpenAI is unavailable.
    """
    topic_summaries = {}
    for label, key in TOPICS:
        d = all_data.get(key, {})
        if d:
            topic_summaries[label] = d

    if not openai_client:
        lines = [
            "CHATREPORT — PRE-VISIT CLINICAL SUMMARY",
            f"Patient: {name}  |  Date: {datetime.now().strftime('%B %d, %Y')}",
            "=" * 56, "",
        ]
        for label, data in topic_summaries.items():
            lines.append(f"[ {label.upper()} ]")
            for k, v in data.items():
                val = ", ".join(v) if isinstance(v, list) else str(v)
                lines.append(f"  - {k.replace('_', ' ').title()}: {val}")
            lines.append("")
        return "\n".join(lines)

    today    = datetime.now().strftime("%B %d, %Y")
    urg_tier = st.session_state.get("urgency_state", {}).get("current_tier", 0)
    urg_sigs = st.session_state.get("urgency_state", {}).get("all_signals", [])

    # Collect doctor notes from topic states
    doctor_notes = {}
    for _, key in TOPICS:
        topic_state = st.session_state.topic_states.get(key, {})
        notes = [
            v for k, v in topic_state.items()
            if k.endswith("_doctor_note") and v
        ]
        if notes:
            doctor_notes[key] = notes

    sentiment_notes = []
    for _, key in TOPICS:
        state = st.session_state.topic_states.get(key, {})
        note = state.get("_sentiment_note")
        if note:
            sentiment_notes.append(note)

    # Build prompt payload — note notes already collected above
    data_json = json.dumps({
        "patient_name": name,
        "report_date": today,
        "symptom_data_by_topic": topic_summaries,
        "freeform_notes": all_data.get("freeform_notes", []),
        "urgency_tier": urg_tier,
        "urgency_signals_active": urg_sigs,
        "last_checkin_data": st.session_state.get("last_checkin", {}),
    }, indent=2, ensure_ascii=False)

    report_prompt = f"DATA:\n{data_json}\n\nGenerate the clinical report following the format in your instructions."

    if not openai_client:
        return "Report generation unavailable — OpenAI API not configured."

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": _REPORT_AGENT_SYS},
                {"role": "user",   "content": report_prompt},
            ],
            max_tokens=2500,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Report generation failed: {e}"


def _default_chatty_reply(
    topic_key: str,
    answer: str,
    step: dict,
    last_topic_data: dict,
) -> str:
    """Fallback acknowledgment when agents are unavailable."""
    return "I've noted that for your care team."



# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "app_stage":           "login",
        "patient_name":        "",
        "selected_topic":      None,
        "topic_states": {
            key: {
                "status": "not_started",
                "data": {},
                "chat": [],
                "followup_counts": {},
                "raw_answers": {},
                "last_prompted_step_id": None,
                "last_prompted_text": "",
                "generated_prompts": {},
            }
            for _, key in TOPICS
        },
        "report":              "",
        "report_saved":        False,
        "last_checkin":        {},
        "has_prev_checkin":    False,
        "freeform_chat":       [],
        "urgency_state": {
            "current_tier": 0,
            "all_signals": [],
            "escalation_shown": False,
            "emergency_shown": False,
            "patient_message": None,
        },
        "sentiment_state": {
            "all_signals": [],
            "engagement_trajectory": "insufficient_data",
            "emotional_state": "neutral",
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ══════════════════════════════════════════════════════════════════
# LOAD PREVIOUS CHECK-IN
# ══════════════════════════════════════════════════════════════════

def load_last_checkin(name: str) -> dict:
    """
    Fetch the most recent saved check-in for this patient from Google Sheets.
    Returns a dict keyed by topic_key -> {q_id: answer}, or {} if none found.
    """
    _init_sheets()
    if _sheet is None:
        return {}
    try:
        rows = _sheet.get_all_values()
        last_row = None
        for row in rows[1:]:
            if len(row) >= 3 and row[1].strip().lower() == name.strip().lower():
                last_row = row          # keep iterating — last match wins
        if last_row:
            raw = json.loads(last_row[2])
            # raw is {topic_key: {q_id: answer}}
            return raw
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════════════════════
# TOPIC SUMMARY FORMATTER  (rule-based, no LLM)
# ══════════════════════════════════════════════════════════════════

# Key fields to surface per topic — (field_id, short_label)
_SUMMARY_FIELDS = {
    "pain": [
        ("has_pain",            "Pain today"),
        ("pain_location",       "Location"),
        ("throat_severity",     "Throat severity"),
        ("tongue_severity",     "Tongue severity"),
        ("pain_medications",    "Medications"),
        ("taking_as_prescribed","Adherence"),
    ],
    "nutrition": [
        ("eating_ability",        "Eating"),
        ("weight",                "Weight (lbs)"),
        ("swallowing_difficulty", "Swallowing"),
        ("feeding_tube",          "Feeding tube"),
        ("iv_fluids",             "IV fluids"),
        ("taste_changes",         "Taste changes"),
    ],
    "oral": [
        ("mouth_sores",    "Mouth sores"),
        ("dry_mouth",      "Dry mouth"),
        ("mucus_issues",   "Mucus"),
        ("oral_rinse_use", "Oral rinse"),
    ],
    "gi": [
        ("nausea_vomiting", "Nausea/vomiting/diarrhea"),
        ("constipation",    "Constipation"),
    ],
    "fatigue": [
        ("fatigue",           "Fatigue"),
        ("sleep_quality",     "Sleep"),
        ("medication_drowsy", "Medication drowsiness"),
    ],
    "activity": [
        ("activity_level",           "Activity level"),
        ("activity_limiting_factor", "Limiting factor"),
    ],
    "mood": [
        ("feeling_down",      "Feeling down"),
        ("support_adequate",  "Support"),
        ("anxiety_impact",    "Anxiety impact"),
    ],
    "other": [
        ("breathing_issues",  "Breathing"),
        ("hearing_changes",   "Hearing"),
        ("dizziness",         "Dizziness"),
        ("skin_issues",       "Skin"),
        ("voice_hoarseness",  "Voice"),
        ("fever_chills",      "Fever/chills"),
    ],
}


def _checkin_summary_html(topic_key: str, data: dict) -> str:
    """
    Build a chip-grid HTML block showing key facts from the previous check-in.
    Each field becomes a small pill: Label on top, value below.
    Returns an HTML string, or "" if no data to show.
    """
    fields = _SUMMARY_FIELDS.get(topic_key, [])
    chips  = []

    for field_id, label in fields:
        val = data.get(field_id)
        if val is None:
            continue
        field_type = QUESTION_TYPE_BY_ID.get(field_id, "options")
        if isinstance(val, list):
            if field_id == "pain_medications" and "Other" in val and data.get("pain_medications_other_detail"):
                val = [
                    data["pain_medications_other_detail"] if item == "Other" else item
                    for item in val
                ]
            val_str = ", ".join(str(v) for v in val)
        else:
            val_str = str(val)
        val_str = val_str.strip()
        if not val_str:
            continue
        if len(val_str) > 35:
            val_str = val_str[:32] + "…"

        is_option_value = field_type in {"options", "multi_select"}
        chip_bg = "#fff7ed" if is_option_value else "#f4f8ff"
        chip_border = "#fdba74" if is_option_value else "#d0e0f8"
        label_color = "#9a6a1a" if is_option_value else "#8fa8c8"
        value_color = "#c2410c" if is_option_value else "#1e3a5f"

        chips.append(
            f'<div style="display:inline-flex;flex-direction:column;'
            f'background:{chip_bg};border:1px solid {chip_border};'
            f'border-radius:10px;padding:5px 13px 6px 13px;'
            f'min-width:70px;max-width:200px;">'            f'<span style="font-size:10px;color:{label_color};font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.4px;'
            f'margin-bottom:2px;">{_html.escape(label)}</span>'            f'<span style="font-size:13px;color:{value_color};font-weight:700;'
            f'line-height:1.3;">{_html.escape(val_str)}</span>'            f'</div>'
        )

    if not chips:
        return ""

    return (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;padding:4px 0 6px 0;">'        + "".join(chips)        + '</div>'
    )


# ── Sidebar summary: natural-language sentence per topic ─────────

def _natural_summary(topic_key: str, data: dict) -> str:
    """Return a short natural-language sentence summarising a topic's previous answers."""
    if not data:
        return ""
    topic_label = next((label for label, key in TOPICS if key == topic_key), topic_key.replace("_", " ").title())
    result = run_topic_summary_agent(topic_label, data)
    summary = str(result.get("summary") or "").strip()
    if summary:
        return summary
    answered = len([k for k, v in data.items() if v not in (None, "", [], {}) and not str(k).endswith("_other_detail")])
    return f"{answered} details were recorded last visit." if answered else ""


def _report_topic_fallback(topic_key: str, topic_label: str, last_topic_data: dict, current_topic_data: dict) -> dict:
    last_summary = _natural_summary(topic_key, last_topic_data) if last_topic_data else "No prior details recorded."
    current_summary = _natural_summary(topic_key, current_topic_data) if current_topic_data else "Not answered this visit."
    if not current_topic_data:
        status = "unanswered"
    elif not last_topic_data:
        status = "new_issue"
    else:
        status = "stable"
    status_label = {
        "worsened": "Worsened",
        "new_issue": "New issue",
        "improved": "Improved",
        "stable": "Stable",
        "unanswered": "Not answered",
    }.get(status, "Stable")
    return {
        "topic_key": topic_key,
        "topic_label": topic_label,
        "status": status,
        "status_label": status_label,
        "last_summary": last_summary,
        "current_summary": current_summary,
        "detail_lines": [],
        "attention_lines": [],
    }


def _report_topic_insights(all_data: dict) -> list[dict]:
    insights = []
    last_ck = st.session_state.get("last_checkin", {})
    for label, key in TOPICS:
        current_topic_data = all_data.get(key, {}) or {}
        last_topic_data = last_ck.get(key, {}) or {}
        fallback = _report_topic_fallback(key, label, last_topic_data, current_topic_data)
        result = run_report_topic_insight_agent(label, last_topic_data, current_topic_data)
        merged = {**fallback, **result}
        merged["topic_key"] = key
        merged["topic_label"] = label
        if not merged.get("last_summary"):
            merged["last_summary"] = fallback["last_summary"]
        if not merged.get("current_summary"):
            merged["current_summary"] = fallback["current_summary"]
        merged["detail_lines"] = merged.get("detail_lines") or []
        merged["attention_lines"] = merged.get("attention_lines") or []
        insights.append(merged)
    return insights


def _report_status_class(status: str) -> str:
    if status in {"worsened", "new_issue"}:
        return "red"
    if status == "improved":
        return "green"
    return ""


def _report_status_text(status: str) -> str:
    mapping = {
        "improved": "Improved",
        "worsened": "Worsened",
        "new_issue": "New issue",
        "stable": "No meaningful change",
        "unanswered": "No current information",
    }
    return mapping.get(status, "No meaningful change")


def _render_report_summary_banner(topic_insights: list[dict]):
    overview = run_report_overview_agent(topic_insights)
    main_issue = overview.get("main_issue") or "Clinical check-in summary ready for review."
    new_issues = overview.get("new_issues") or []
    improvements = overview.get("improvements") or []
    needs_attention = overview.get("needs_attention") or []

    parts = [
        '<div class="report-summary-banner">',
        '<div class="report-summary-avatar">🧑</div>',
        '<div class="report-summary-copy">',
        '<div class="report-summary-title">Patient Summary</div>',
        f'<div><strong>Main issue:</strong> {_html.escape(str(main_issue))}</div>',
    ]
    if new_issues:
        parts.append(f'<div><strong>New issues:</strong> {_html.escape("; ".join(str(x) for x in new_issues[:3]))}</div>')
    if improvements:
        parts.append(f'<div><strong>Improvement:</strong> {_html.escape("; ".join(str(x) for x in improvements[:3]))}</div>')
    if needs_attention:
        parts.append(f'<div><strong>Needs attention:</strong> {_html.escape("; ".join(str(x) for x in needs_attention[:4]))}</div>')
    parts.extend(['</div>', '</div>'])
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_report_topic_card(insight: dict):
    status_class = _report_status_class(insight.get("status", "stable"))
    status_label = _report_status_text(insight.get("status", "stable"))
    topic_name = insight.get("topic_label", "").split(" ", 1)[1] if " " in insight.get("topic_label", "") else insight.get("topic_label", "")
    st.markdown(
        f'<div class="report-topic-card {status_class}">'
        f'  <div class="report-topic-strip">{_html.escape(status_label)}</div>'
        f'  <div class="report-topic-body">'
        f'    <div class="report-topic-name">{_html.escape(topic_name)}</div>'
        f'    <div class="report-topic-compare">'
        f'      <div class="report-topic-compare-row"><div class="report-topic-compare-label">Last</div><div class="report-topic-compare-value">{_html.escape(str(insight.get("last_summary") or "No prior details recorded."))}</div></div>'
        f'      <div class="report-topic-compare-row"><div class="report-topic-compare-label">Now</div><div class="report-topic-compare-value">{_html.escape(str(insight.get("current_summary") or "Not answered this visit."))}</div></div>'
        f'    </div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_report_topic_detail(insight: dict, all_data: dict):
    topic_key = insight["topic_key"]
    last_topic_data = st.session_state.get("last_checkin", {}).get(topic_key, {}) or {}
    current_topic_data = all_data.get(topic_key, {}) or {}

    with st.expander("More details", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Last check-in**")
            st.markdown(str(insight.get("last_summary") or "No prior details recorded."))
        with col2:
            st.markdown("**Current check-in**")
            st.markdown(str(insight.get("current_summary") or "Not answered this visit."))

        if insight.get("attention_lines"):
            st.markdown("**Key points**")
            for line in insight["attention_lines"]:
                st.markdown(f"- {line}")
        if insight.get("detail_lines"):
            st.markdown("**Comparison details**")
            for line in insight["detail_lines"]:
                st.markdown(f"- {line}")

        last_html = _checkin_summary_html(topic_key, last_topic_data)
        now_html = _checkin_summary_html(topic_key, current_topic_data)
        if last_html or now_html:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Last visit details**")
                st.markdown(last_html or '<div style="color:#7a8ea4;">No prior details recorded.</div>', unsafe_allow_html=True)
            with col2:
                st.markdown("**Current visit details**")
                st.markdown(now_html or '<div style="color:#7a8ea4;">No current details recorded.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# FREE-FORM CHAT LLM
# ══════════════════════════════════════════════════════════════════

def _freeform_llm_response(messages: list) -> str:
    """
    Generate a nurse reply in the free-form "Anything else?" chat.
    messages is the full conversation history: [{role, content}, ...].
    """
    if not openai_client:
        return "I'm not able to respond right now — please let your care team know directly."

    structured_context = {
        key: st.session_state.topic_states[key]["data"]
        for _, key in TOPICS
        if st.session_state.topic_states[key]["data"]
    }
    prior_context = st.session_state.get("last_checkin", {})

    hnc_context = (
        "You are a compassionate, clinically trained nurse at a head and neck cancer "
        "(HNC) center conducting a structured symptom check-in with a patient currently "
        "receiving chemoradiation or surgery for head and neck cancer. "
        "This patient population frequently experiences: severe mucositis, dysphagia, "
        "pain, significant weight loss, fatigue, depression, and impaired communication. "
        "Many patients have low health literacy or face barriers to care. "
        "Your tone is always warm, clear, and non-alarming. Never use medical jargon "
        "without explaining it simply. Never minimize a patient's reported symptom."
    )
    system = (
        f"{hnc_context}\n\n"
        "You are now in an open conversation with the patient. They may raise anything not "
        "covered by the structured check-in — a new symptom, a question about their treatment, "
        "a concern about a medication, or just something they want their provider to know.\n\n"
        f"CURRENT STRUCTURED CHECK-IN DATA:\n{json.dumps(structured_context, indent=2)}\n\n"
        f"MOST RECENT PRIOR CHECK-IN DATA:\n{json.dumps(prior_context, indent=2)}\n\n"
        "Guidelines:\n"
        "- Listen carefully and respond with warmth and clinical awareness.\n"
        "- If helpful, briefly notice whether this sounds better, worse, or different than last visit.\n"
        "- If they mention a symptom that sounds urgent (e.g., chest pain, breathing difficulty, "
        "  high fever, blood, suicidal thoughts), acknowledge it calmly and tell them it will be "
        "  flagged for their care team.\n"
        "- Do NOT diagnose or prescribe. You are gathering information, not treating.\n"
        "- Be conversational and natural, not stiff or repetitive.\n"
        "- Keep responses short — 2-4 sentences maximum.\n"
        "- If the patient seems to be done, gently close: 'Is there anything else you'd like "
        "  to share with your team before your visit?'"
    )

    api_messages = [{"role": "system", "content": system}]
    for m in messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    try:
        r = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=api_messages,
            max_tokens=200,
            temperature=0.5,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return "I'm having trouble responding right now — please share this with your care team directly."


# ══════════════════════════════════════════════════════════════════
# Conversational answer handling
# ══════════════════════════════════════════════════════════════════

def _append_next_question(
    topic_key: str,
    state: dict,
    next_step: Optional[dict],
    assistant_message: str = "",
):
    message = assistant_message.strip()
    next_text = _step_prompt_text(next_step, topic_key=topic_key, state=state) if next_step else ""
    prompt_consumed = False
    if message and next_text and _is_semantically_redundant_question(message, next_text):
        prompt_consumed = True
    if message:
        _append_assistant_message(state, message)
    if next_text:
        if not prompt_consumed:
            _append_assistant_message(state, next_text)
        _remember_prompted_step(state, next_step, next_text)
    elif not next_step:
        _remember_prompted_step(state, None, "")


def _maybe_skip_next_impact_question(topic_key: str, state: dict):
    return


def _apply_agent_next_step_action(topic_key: str, state: dict, action: Optional[dict]):
    if not action:
        return

    plan = []
    if action.get("skip_immediate_next_step"):
        next_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
        if next_step:
            plan.append({
                "step_id": next_step.get("id"),
                "suggested_answer": action.get("suggested_answer"),
                "carry_forward_answer": action.get("carry_forward_answer"),
                "reason": action.get("reason"),
            })
    for item in action.get("plan", []) or []:
        if isinstance(item, dict) and item.get("step_id"):
            plan.append(item)

    for item in plan:
        step_id = item.get("step_id")
        step = STEP_BY_ID.get(step_id)
        if not step or step_id in state["data"]:
            continue
        if step.get("type") == "free_text":
            carry = item.get("carry_forward_answer")
            if isinstance(carry, str) and carry.strip():
                value = carry.strip()
                state["data"][step_id] = value
                state["raw_answers"][step_id] = value
            continue
        if step.get("type") == "options":
            opts = step.get("opts", [])
            suggested = item.get("suggested_answer")
            chosen = None
            if suggested in opts:
                chosen = suggested
            else:
                normalized_opts = {_norm_text(opt): opt for opt in opts}
                if "no" in normalized_opts:
                    chosen = normalized_opts["no"]
                else:
                    for opt in opts:
                        if _norm_text(opt).startswith("no"):
                            chosen = opt
                            break
            if chosen:
                state["data"][step_id] = chosen
                state["raw_answers"][step_id] = chosen


def _apply_generic_fallback_next_step_action(topic_key: str, state: dict):
    return


def _maybe_apply_prompt_driven_skip(topic_key: str, state: dict, pipeline: dict):
    return


def _capture_rich_answer_into_next_step(
    topic_key: str,
    state: dict,
    current_step: dict,
    resolved_answer: Any,
    raw_answer: Any,
):
    """
    If a patient answers an option question with richer free text that already
    answers the immediate next step, capture that detail now so the app does not
    ask for it again in a different form.
    """
    if current_step.get("type") != "options":
        return

    raw_text = str(raw_answer or "").strip()
    if not raw_text:
        return
    if isinstance(resolved_answer, str) and _norm_text(raw_text) == _norm_text(resolved_answer):
        return

    next_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
    if not next_step or next_step.get("id") in state["data"]:
        return
    if next_step.get("type") != "options":
        return

    interpreted = interpret_user_input_with_options(
        next_step,
        raw_text,
        topic_history=_recent_topic_history(state),
    )
    if interpreted not in next_step.get("opts", []):
        return

    coerced = _coerce_structured_answer(
        topic_key,
        next_step,
        interpreted,
        state["data"],
        raw_answer=raw_text,
    )
    state["data"][next_step["id"]] = coerced
    state["raw_answers"][next_step["id"]] = raw_text


def _backfill_next_step_from_topic_history(topic_key: str, state: dict, next_step: Optional[dict]):
    """
    Safety net: if the next unresolved step is a location chooser with a catch-all
    and the patient already named a concrete body part earlier in this topic,
    resolve it internally instead of surfacing the chooser again.
    """
    if not next_step or next_step.get("id") in state["data"]:
        return
    if next_step.get("type") != "options":
        return
    if "Somewhere else" not in (next_step.get("opts") or []):
        return
    step_text = _norm_text(next_step.get("text", ""))
    if "where" not in step_text and "location" not in step_text:
        return

    raw_answers = state.get("raw_answers", {})
    for step_id, raw_text in reversed(list(raw_answers.items())):
        if step_id == next_step.get("id"):
            continue
        text = str(raw_text or "").strip()
        if not _looks_like_body_location_phrase(text):
            continue
        interpreted = interpret_user_input_with_options(
            next_step,
            text,
            topic_history=_recent_topic_history(state),
        )
        if interpreted not in next_step.get("opts", []):
            continue
        coerced = _coerce_structured_answer(
            topic_key,
            next_step,
            interpreted,
            state["data"],
            raw_answer=text,
        )
        state["data"][next_step["id"]] = coerced
        state["raw_answers"][next_step["id"]] = text
        break


def _resolve_next_step(topic_key: str, state: dict) -> Optional[dict]:
    next_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
    _backfill_next_step_from_topic_history(topic_key, state, next_step)
    return get_next_step(topic_key, state["data"], state.get("raw_answers"))


def _quick_reply_suggestions(topic_key: str, state: dict, step: dict) -> list[str]:
    if step.get("opts"):
        return []
    if step.get("type") not in {"free_text", "number"}:
        return []
    cache = state.setdefault("generated_quick_replies", {})
    cached = cache.get(step["id"])
    if isinstance(cached, list):
        return cached

    result = run_quick_reply_suggester_agent(
        step,
        topic_history=_recent_topic_history(state),
        recent_questions=_recent_topic_questions(state),
    )
    suggestions = result.get("suggestions", []) if isinstance(result, dict) else []
    if not isinstance(suggestions, list):
        suggestions = []
    cache[step["id"]] = suggestions
    return suggestions


def _mark_submission_once(submitted_key: str, candidate: str) -> bool:
    if not candidate or st.session_state.get(submitted_key) == candidate:
        return False
    st.session_state[submitted_key] = candidate
    return True


def _render_choice_button_grid(options: list[str], key_prefix: str) -> Optional[str]:
    if not options:
        return None
    cols_per_row = 2 if len(options) > 1 else 1
    for idx in range(0, len(options), cols_per_row):
        row = st.columns(cols_per_row)
        for offset, option in enumerate(options[idx:idx + cols_per_row]):
            with row[offset]:
                if st.button(option, key=f"{key_prefix}_{idx + offset}", use_container_width=True):
                    return option
    return None


def _process_option_submission(
    topic_key: str,
    step: dict,
    candidate: str,
    source: str,
    submitted_key: str,
    topic_history: list[dict[str, str]],
) -> bool:
    if not _mark_submission_once(submitted_key, candidate):
        return False
    interpreted = interpret_user_input_with_options(step, candidate, topic_history=topic_history)
    if interpreted in step.get("opts", []):
        handle_answer(
            topic_key,
            step,
            interpreted,
            source="structured",
            display_override=candidate,
            raw_answer=candidate,
        )
    else:
        _request_retry_for_step(topic_key, step, candidate, source=source)
    return True


def _process_multiselect_submission(
    topic_key: str,
    step: dict,
    candidate: str,
    source: str,
    submitted_key: str,
) -> bool:
    if not _mark_submission_once(submitted_key, candidate):
        return False
    parsed = parse_multi_select_typed_input(step, candidate)
    if parsed:
        handle_answer(
            topic_key,
            step,
            parsed,
            source="structured",
            display_override=candidate,
            raw_answer=candidate,
        )
    else:
        _request_retry_for_step(topic_key, step, candidate, source=source)
    return True


def _process_number_submission(topic_key: str, step: dict, candidate: str, submitted_key: str) -> bool:
    if not _mark_submission_once(submitted_key, candidate):
        return False
    try:
        value = int(float(candidate))
    except ValueError:
        st.warning("Please enter a number.")
        return True
    if value < step["min_v"] or value > step["max_v"]:
        st.warning(f"Please enter a value between {int(step['min_v'])} and {int(step['max_v'])}.")
        return True
    handle_answer(topic_key, step, value, source="typed")
    return True


def _store_followup_prompt(
    topic_key: str,
    state: dict,
    step: dict,
    question: str,
    assistant_message: str = "",
    retry_current_step: bool = False,
    allow_other_detail: bool = False,
    target_step: Optional[dict] = None,
):
    state["waiting_for_followup"] = True
    state["pending_followup"] = {
        "source_step_id": step["id"],
        "question": question,
        "answer_key": f"{step['id']}_llm_followup",
        "assistant_message": assistant_message.strip(),
        "retry_current_step": retry_current_step,
        "allow_other_detail": allow_other_detail,
        "target_step_id": target_step.get("id") if target_step else None,
    }
    combined_prompt = "\n\n".join([part for part in [assistant_message.strip(), question.strip()] if part])
    _append_assistant_message(state, combined_prompt)


def _request_retry_for_step(topic_key: str, step: dict, raw_input: str, source: str = "typed"):
    state = st.session_state.topic_states[topic_key]
    text = (raw_input or "").strip()
    if text:
        state["chat"].append({"role": "user", "content": text})
    retry_question = _build_retry_prompt(step, text, topic_history=_recent_topic_history(state))
    _store_followup_prompt(
        topic_key,
        state,
        step,
        retry_question,
        retry_current_step=True,
        allow_other_detail=("Other" in step.get("opts", [])),
    )
    st.rerun()


def _clear_step_inputs(topic_key: str, step: dict):
    sid = step["id"]
    stype = step["type"]

    keys_to_clear = []
    if stype == "options":
        keys_to_clear.extend([
            f"text_{topic_key}_{sid}",
            f"text_{topic_key}_{sid}_submitted",
            f"dropdown_{topic_key}_{sid}",
            f"dropdown_{topic_key}_{sid}_submitted",
            f"voice_{topic_key}_{sid}_submitted",
            f"_vt_{topic_key}_{sid}_opt",
            f"_vh_{topic_key}_{sid}_opt",
        ])
    elif stype == "multi_select":
        keys_to_clear.extend([
            f"text_{topic_key}_{sid}",
            f"text_{topic_key}_{sid}_submitted",
            f"dropdown_{topic_key}_{sid}",
            f"dropdown_{topic_key}_{sid}_submitted",
            f"voice_{topic_key}_{sid}_submitted",
            f"_vt_{topic_key}_{sid}_multi",
            f"_vh_{topic_key}_{sid}_multi",
        ])
    elif stype == "number":
        keys_to_clear.extend([
            f"text_{topic_key}_{sid}",
            f"text_{topic_key}_{sid}_submitted",
            f"suggested_{topic_key}_{sid}",
            f"suggested_{topic_key}_{sid}_submitted",
            f"_vt_{topic_key}_{sid}_num",
            f"_vh_{topic_key}_{sid}_num",
        ])
    elif stype == "free_text":
        keys_to_clear.extend([
            f"ft_{topic_key}_{sid}",
            f"ft_{topic_key}_{sid}_submitted",
            f"suggested_{topic_key}_{sid}",
            f"suggested_{topic_key}_{sid}_submitted",
            f"ft_{topic_key}_{sid}_voice_sync",
            f"_vt_{topic_key}_{sid}",
            f"_vh_{topic_key}_{sid}",
        ])

    for key in keys_to_clear:
        st.session_state.pop(key, None)


def handle_pending_followup(topic_key: str, answer: str, source: str = "typed"):
    state = st.session_state.topic_states[topic_key]
    pending = state.get("pending_followup") or {}
    answer_key = pending.get("answer_key")
    if not answer_key:
        state["waiting_for_followup"] = False
        state.pop("pending_followup", None)
        st.rerun()
        return

    if pending.get("retry_current_step"):
        source_step_id = pending.get("source_step_id")
        source_step = STEP_BY_ID.get(source_step_id)
        state["waiting_for_followup"] = False
        state.pop("pending_followup", None)
        if not source_step:
            st.rerun()
            return

        retry_text = (answer or "").strip()
        if source_step["type"] == "options":
            interpreted = interpret_user_input_with_options(source_step, retry_text, topic_history=_recent_topic_history(state))
            if interpreted in source_step.get("opts", []):
                handle_answer(
                    topic_key,
                    source_step,
                    interpreted,
                    source="structured",
                    raw_answer=retry_text,
                    display_override=retry_text,
                )
                return
            _request_retry_for_step(topic_key, source_step, retry_text, source=source)
            return

        if source_step["type"] == "multi_select":
            parsed = parse_multi_select_typed_input(source_step, retry_text)
            if parsed:
                handle_answer(
                    topic_key,
                    source_step,
                    parsed,
                    source="structured",
                    raw_answer=retry_text,
                    display_override=retry_text,
                )
                return
            if pending.get("allow_other_detail") and retry_text:
                state["data"][f"{source_step['id']}_other_detail"] = retry_text
                handle_answer(
                    topic_key,
                    source_step,
                    ["Other"],
                    source="structured",
                    display_override=retry_text,
                    raw_answer=retry_text,
                )
                return
            _request_retry_for_step(topic_key, source_step, retry_text, source=source)
            return

    target_step_id = pending.get("target_step_id")
    target_step = STEP_BY_ID.get(target_step_id) if target_step_id else None
    if target_step:
        state["waiting_for_followup"] = False
        state.pop("pending_followup", None)

        followup_text = (answer or "").strip()
        if target_step["type"] == "options":
            interpreted = interpret_user_input_with_options(
                target_step,
                followup_text,
                topic_history=_recent_topic_history(state),
            )
            if interpreted in target_step.get("opts", []):
                handle_answer(
                    topic_key,
                    target_step,
                    interpreted,
                    source="structured",
                    raw_answer=followup_text,
                    display_override=followup_text,
                )
                return
            _request_retry_for_step(topic_key, target_step, followup_text, source=source)
            return

        if target_step["type"] == "multi_select":
            parsed = parse_multi_select_typed_input(target_step, followup_text)
            if parsed:
                handle_answer(
                    topic_key,
                    target_step,
                    parsed,
                    source="structured",
                    raw_answer=followup_text,
                    display_override=followup_text,
                )
                return
            if "Other" in target_step.get("opts", []) and followup_text:
                state["data"][f"{target_step['id']}_other_detail"] = followup_text
                handle_answer(
                    topic_key,
                    target_step,
                    ["Other"],
                    source="structured",
                    display_override=followup_text,
                    raw_answer=followup_text,
                )
                return
            _request_retry_for_step(topic_key, target_step, followup_text, source=source)
            return

        if target_step["type"] == "number":
            try:
                numeric_value = int(float(followup_text))
            except (TypeError, ValueError):
                _request_retry_for_step(topic_key, target_step, followup_text, source=source)
                return
            handle_answer(
                topic_key,
                target_step,
                numeric_value,
                source="structured",
                display_override=followup_text,
                raw_answer=followup_text,
            )
            return

        handle_answer(
            topic_key,
            target_step,
            followup_text,
            source="free_text",
            raw_answer=followup_text,
            display_override=followup_text,
        )
        return

    state["chat"].append({"role": "user", "content": answer})
    state["data"][answer_key] = answer
    pending_key = f"pending_followup_{topic_key}_{pending.get('answer_key', 'pending')}"
    st.session_state.pop(pending_key, None)
    submitted_pending_key = f"{pending_key}_submitted"
    st.session_state.pop(submitted_pending_key, None)
    st.session_state.pop(f"{pending_key}_voice_sync", None)
    state["waiting_for_followup"] = False
    state.pop("pending_followup", None)

    last_topic_data = st.session_state.last_checkin.get(topic_key, {})
    closing = _default_chatty_reply(
        topic_key,
        answer,
        {"id": answer_key, "text": pending.get("question", "")},
        last_topic_data,
    )

    next_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
    state["status"] = "in_progress"

    if topic_is_complete(topic_key, state["data"], state.get("raw_answers")):
        state["status"] = "completed"
        state["chat"].append({
            "role": "assistant",
            "content": f"{closing}\n\n✅ Thank you — I have everything I need for this topic."
        })
    else:
        _append_next_question(topic_key, state, next_step, closing)

    st.rerun()
    return


def handle_answer(
    topic_key: str,
    step: dict,
    answer,
    source: str = "structured",
    display_override: Optional[str] = None,
    raw_answer: Any = None,
):
    """
    Core answer handler — orchestrates all agents and determines next action.
    For structured button clicks (source='structured') we skip the full agent
    pipeline and just do a quick comparison + vague-check to keep latency low.
    For free-text, voice, and typed answers we run the full multi-agent pipeline.
    """
    state = st.session_state.topic_states[topic_key]

    # ── Ensure followup_counts dict exists (backward compat) ──────
    if "followup_counts" not in state:
        state["followup_counts"] = {}
    if "raw_answers" not in state:
        state["raw_answers"] = {}
    if "last_prompted_step_id" not in state:
        state["last_prompted_step_id"] = None
    if "last_prompted_text" not in state:
        state["last_prompted_text"] = ""
    if "generated_prompts" not in state:
        state["generated_prompts"] = {}
    if "generated_quick_replies" not in state:
        state["generated_quick_replies"] = {}
    if state.get("last_prompted_step_id") == step.get("id"):
        _remember_prompted_step(state, None, "")
    _clear_step_inputs(topic_key, step)

    display = display_override if display_override is not None else (
        ", ".join(answer) if isinstance(answer, list) else str(answer)
    )
    state["chat"].append({"role": "user", "content": display})
    verbatim = raw_answer if raw_answer is not None else display
    if isinstance(verbatim, str) and verbatim.strip():
        state["raw_answers"][step["id"]] = verbatim.strip()
    if (
        step.get("type") == "multi_select"
        and isinstance(answer, list)
        and "Other" in answer
        and isinstance(verbatim, str)
        and verbatim.strip()
    ):
        state["data"][f"{step['id']}_other_detail"] = verbatim.strip()
    answer = _coerce_structured_answer(topic_key, step, answer, state["data"], raw_answer=raw_answer)
    state["data"][step["id"]] = answer
    _capture_rich_answer_into_next_step(topic_key, state, step, answer, verbatim)
    if topic_key == "pain" and step.get("id") == "pain_medications":
        meds = answer if isinstance(answer, list) else [answer]
        if "No pain medication" in meds:
            for stale_id in ("med_dose_freq", "taking_as_prescribed", "med_adherence_issue", "med_side_effects"):
                state["data"].pop(stale_id, None)
                state["raw_answers"].pop(stale_id, None)
    if step.get("id") == "other_pain_desc":
        focus = run_pain_location_focus_agent(verbatim if isinstance(verbatim, str) else str(answer))
        state["data"]["other_pain_head_neck_focused"] = bool(focus.get("head_neck_focused"))
    if isinstance(verbatim, str) and not openai_client:
        _auto_capture_following_answers(topic_key, state, verbatim)
    next_step = _resolve_next_step(topic_key, state)
    state["status"] = "in_progress"

    last_topic_data = st.session_state.last_checkin.get(topic_key, {})

    # ══════════════════════════════════════════════════════════════
    # BRANCH A — Structured non-string answers (fast path)
    # Lists/numbers do not benefit much from the language pipeline.
    # String answers, including button/option replies like "Yes" or "No",
    # still go through the agents so the app can skip irrelevant follow-ups.
    # ══════════════════════════════════════════════════════════════
    if source == "structured" and not isinstance(answer, str):
        if topic_is_complete(topic_key, state["data"], state.get("raw_answers")):
            state["status"] = "completed"
            state["chat"].append({
                "role": "assistant",
                "content": "✅ Thank you — I have everything I need for this topic.",
            })
            st.rerun()
            return
        _append_next_question(topic_key, state, next_step)
        st.rerun()
        return

    # ══════════════════════════════════════════════════════════════
    # BRANCH B — String answers — run full agent pipeline
    # ══════════════════════════════════════════════════════════════
    if isinstance(answer, str):
        if source in {"typed", "voice", "free_text"} and not openai_client and not answer.strip():
            _store_followup_prompt(
                topic_key, state, step, _fallback_clarifying_question(step),
            )
            st.rerun()
            return

        if openai_client:
            with st.spinner("Thinking…"):
                pipeline = run_agent_pipeline(
                    topic_key=topic_key,
                    step=step,
                    answer=answer,
                    raw_answer=verbatim if isinstance(verbatim, str) else str(verbatim),
                    state=state,
                    last_topic_data=last_topic_data,
                )

            # ── Emergency: terminate session ──────────────────────
            if pipeline.get("urgency_tier", 0) == 3:
                emergency_msg = pipeline.get("urgency_message") or (
                    "We are concerned about what you've shared. Please call 911 or "
                    "go to your nearest emergency room immediately. "
                    "Your care team has been notified."
                )
                state["chat"].append({
                    "role": "assistant",
                    "content": f"🚨 {emergency_msg}",
                })
                state["status"] = "completed"   # Lock this topic
                st.session_state["urgency_state"]["emergency_shown"] = True
                st.rerun()
                return

            # ── Tier 2: insert care team notice into chat ─────────
            tier2_msg = pipeline.get("urgency_message")
            if tier2_msg:
                state["chat"].append({"role": "assistant", "content": f"🔴 {tier2_msg}"})

            # ── Acknowledgment before follow-up / next question ───
            ack = pipeline.get("acknowledgment", "").strip()

            # ── Patient wants to stop ─────────────────────────────
            if pipeline.get("wants_to_stop"):
                closing = "Of course — we'll pause here. The answers you've shared have been saved for your care team."
                if ack:
                    closing = f"{ack}\n\n{closing}"
                state["chat"].append({"role": "assistant", "content": closing})
                state["status"] = "completed"
                st.rerun()
                return

            # ── Tier 2: avoid detached follow-ups in the same turn ─
            if tier2_msg:
                state["chat"].append({
                    "role": "assistant",
                    "content": "We'll pause this topic here for now so your care team can follow up directly.",
                })
                state["status"] = "completed"
                st.rerun()
                return

            # ── Follow-up question ────────────────────────────────
            if pipeline.get("follow_up") and pipeline.get("follow_up_question"):
                fq = pipeline["follow_up_question"]
                # Reject if semantically redundant with original question
                if _is_redundant_followup(step["text"], answer, fq):
                    pass   # Fall through to assistant_message + next question
                else:
                    next_step_action = pipeline.get("next_step_action")
                    if next_step_action:
                        _apply_agent_next_step_action(topic_key, state, next_step_action)
                        next_step = _resolve_next_step(topic_key, state)
                    # Increment follow-up counter
                    fc = state["followup_counts"]
                    fc[step["id"]] = fc.get(step["id"], 0) + 1
                    _store_followup_prompt(
                        topic_key, state, step, fq, ack,
                        target_step=next_step,
                    )
                    st.rerun()
                    return

            # ── Store doctor note and signals for report ──────────
            if pipeline.get("doctor_note"):
                state[f"{step['id']}_doctor_note"] = pipeline["doctor_note"]
            if pipeline.get("sentiment_note"):
                state["_sentiment_note"] = pipeline["sentiment_note"]

            assistant_message = pipeline.get("assistant_message", "").strip()
            if ack and assistant_message:
                assistant_message = f"{ack}\n\n{assistant_message}"
            elif ack:
                assistant_message = ack

            _apply_agent_next_step_action(topic_key, state, pipeline.get("next_step_action"))
            next_step = _resolve_next_step(topic_key, state)

        else:
            # No OpenAI — use fallback reply
            assistant_message = _default_chatty_reply(
                topic_key, answer, step, last_topic_data
            )
            _apply_generic_fallback_next_step_action(topic_key, state)
            next_step = _resolve_next_step(topic_key, state)

    else:
        # Non-string answer (numeric, list from multi_select on structured path)
        assistant_message = ""

    # ── Topic complete check ──────────────────────────────────────
    if topic_is_complete(topic_key, state["data"], state.get("raw_answers")):
        state["status"] = "completed"
        final_message = "✅ Thank you — I have everything I need for this topic."
        if assistant_message:
            final_message = f"{assistant_message}\n\n{final_message}"
        state["chat"].append({"role": "assistant", "content": final_message})
        st.rerun()
        return

    _append_next_question(topic_key, state, next_step, assistant_message)
    st.rerun()
    return


# ══════════════════════════════════════════════════════════════════
# INPUT RENDERING
# ══════════════════════════════════════════════════════════════════


def render_input(topic_key: str, step: dict):
    """Render the appropriate input widget for the current question."""
    stype = step["type"]
    sid   = step["id"]

    state = st.session_state.topic_states[topic_key]
    topic_history = _recent_topic_history(state)

    def render_option_buttons(button_topic_key: str, button_step: dict, multi: bool = False):
        opts = button_step.get("opts", [])
        selected = _render_choice_button_grid(opts, f"btn_{button_topic_key}_{button_step['id']}")
        if selected:
            payload = [selected] if multi else selected
            handle_answer(button_topic_key, button_step, payload, source="structured")
            return

    def render_suggested_buttons(button_topic_key: str, button_step: dict):
        if button_step.get("opts") or button_step.get("type") not in {"free_text", "number"}:
            return
        suggestions = _quick_reply_suggestions(button_topic_key, state, button_step)
        if not suggestions:
            return
        st.markdown('<div class="suggested-replies-note">Suggested replies</div>', unsafe_allow_html=True)
        pills_key = f"suggested_{button_topic_key}_{button_step['id']}"
        submitted_key = f"{pills_key}_submitted"

        selected = None
        if hasattr(st, "pills"):
            selected = st.pills(
                "Suggested replies",
                suggestions,
                key=pills_key,
                label_visibility="collapsed",
            )
        else:
            selected = _render_choice_button_grid(
                suggestions,
                f"suggest_{button_topic_key}_{button_step['id']}",
            )

        if not selected or not _mark_submission_once(submitted_key, selected):
            return
        if button_step["type"] == "number":
            try:
                numeric_value = int(float(selected))
            except (TypeError, ValueError):
                handle_answer(
                    button_topic_key,
                    button_step,
                    selected,
                    source="typed",
                    display_override=selected,
                    raw_answer=selected,
                )
                return
            handle_answer(
                button_topic_key,
                button_step,
                numeric_value,
                source="typed",
                display_override=selected,
                raw_answer=selected,
            )
            return
        handle_answer(
            button_topic_key,
            button_step,
            selected,
            source="free_text",
            display_override=selected,
            raw_answer=selected,
        )
        return

    # ── Options ─────────────────────────────────────────────────
    if stype == "options":
        st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
        user_text = st.text_input(
            "Message",
            key=f"text_{topic_key}_{sid}",
            label_visibility="collapsed",
            placeholder="Type a reply..."
        )
        render_option_buttons(topic_key, step, multi=False)

        with st.container():
            voice_text = voice_widget(f"{topic_key}_{sid}_opt", label="Mic")

        submitted_key = f"text_{topic_key}_{sid}_submitted"
        if _process_option_submission(topic_key, step, user_text, "typed", submitted_key, topic_history):
            return

        voice_submitted_key = f"voice_{topic_key}_{sid}_submitted"
        if _process_option_submission(topic_key, step, voice_text, "voice", voice_submitted_key, topic_history):
            return
        st.markdown('</div>', unsafe_allow_html=True)
                

    # ── Multi-select ─────────────────────────────────────────────
    elif stype == "multi_select":
        st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
        text_key = f"text_{topic_key}_{sid}"
        submit_key = f"{text_key}_submitted"
        user_text = st.text_input(
            "Reply",
            key=text_key,
            label_visibility="collapsed",
            placeholder="Type one or more answers, separated by commas..."
        )
        render_option_buttons(topic_key, step, multi=True)
        with st.container():
            voice_text = voice_widget(f"{topic_key}_{sid}_multi", label="Mic")

        if _process_multiselect_submission(topic_key, step, user_text, "typed", submit_key):
            return

        voice_submit_key = f"voice_{topic_key}_{sid}_submitted"
        if _process_multiselect_submission(topic_key, step, voice_text, "voice", voice_submit_key):
            return
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Number ───────────────────────────────────────────────────
    elif stype == "number":
        st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
        text_key = f"text_{topic_key}_{sid}"
        submit_key = f"{text_key}_submitted"
        if text_key not in st.session_state:
            st.session_state[text_key] = ""
        user_text = st.text_input(
            "Reply",
            key=text_key,
            label_visibility="collapsed",
            placeholder=f"Enter a number ({int(step['min_v'])}-{int(step['max_v'])})"
        )
        render_suggested_buttons(topic_key, step)
        with st.container():
            voice_text = voice_widget(f"{topic_key}_{sid}_num", label="Mic")

        candidate = user_text or voice_text or ""
        if _process_number_submission(topic_key, step, candidate, submit_key):
            return
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Free text ────────────────────────────────────────────────
    elif stype == "free_text":
        transcript_key = f"_vt_{topic_key}_{sid}"
        widget_key     = f"ft_{topic_key}_{sid}"
        submit_key     = f"{widget_key}_submitted"

        # Priority for pre-fill: voice transcript > empty
        transcript = st.session_state.get(transcript_key, "")
        if widget_key not in st.session_state:
            st.session_state[widget_key] = transcript or ""
        elif transcript and transcript != st.session_state.get(f"{widget_key}_voice_sync"):
            st.session_state[widget_key] = transcript
            st.session_state[f"{widget_key}_voice_sync"] = transcript

        st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
        with st.container():
            free_text = st.text_input(
                "Reply",
                placeholder=step.get("placeholder", "Please describe…"),
                key=widget_key,
                label_visibility="collapsed",
            )
        render_suggested_buttons(topic_key, step)
        with st.container():
            voice_text = voice_widget(f"{topic_key}_{sid}", label="Mic")
        if voice_text and voice_text != st.session_state.get(f"{widget_key}_voice_sync"):
            st.session_state[f"{widget_key}_voice_sync"] = voice_text
            st.session_state[submit_key] = voice_text
            handle_answer(topic_key, step, voice_text, source="voice")
            return

        if free_text and st.session_state.get(submit_key) != free_text:
            st.session_state[submit_key] = free_text
            handle_answer(topic_key, step, free_text, source="free_text")

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# FREE-FORM CHAT PANEL
# ══════════════════════════════════════════════════════════════════

def render_freeform_chat():
    """Render the open-ended 'Anything else?' chatbot panel."""
    _stc.html("""<script>
    (function(){
        var s=['section[data-testid=\"stMain\"]',
               'div[data-testid=\"stAppViewContainer\"]','.main'];
        for(var i=0;i<s.length;i++){
            var e=window.parent.document.querySelector(s[i]);
            if(e){e.scrollTop=0;break;}
        }
    })();
    </script>""", height=0)
    st.markdown(
        '<div class="chat-shell">'
        '  <div class="chat-shell-header">'
        '    <div class="chat-shell-title">'
        '      <div class="chat-shell-name">Anything else you’d like to share?</div>'
        '    </div>'
        '  </div>'
        '  <div class="chat-history">',
        unsafe_allow_html=True,
    )

    # ── Initialise conversation ──────────────────────────────────
    if not st.session_state.freeform_chat:
        opening = (
            "Is there anything else you'd like your care team to know before your visit? "
            "Feel free to share any concerns, questions, or symptoms we haven't covered yet."
        )
        st.session_state.freeform_chat = [{"role": "assistant", "content": opening}]

    # ── Show history ─────────────────────────────────────────────
    chat_container = st.container(border=False)
    with chat_container:
        for msg in st.session_state.freeform_chat:
            render_chat_bubble(msg["role"], msg["content"])

    st.markdown('</div><div class="composer-wrap">', unsafe_allow_html=True)

    # ── Input ────────────────────────────────────────────────────
    user_input = st.chat_input("Type here, or use the voice button below…",
                                key="freeform_chat_input")

    with st.container():
        vt = voice_widget("freeform")
        if vt and not user_input:
            user_input = vt
            # Clear the transcript so it doesn't re-fire
            st.session_state.pop("_vt_freeform", None)

    if user_input and user_input.strip():
        # Avoid re-appending if already in history (Streamlit reruns)
        last_user = next(
            (m["content"] for m in reversed(st.session_state.freeform_chat)
             if m["role"] == "user"), None
        )
        if user_input.strip() != last_user:
            st.session_state.freeform_chat.append(
                {"role": "user", "content": user_input.strip()}
            )
            if openai_client:
                with st.spinner("…"):
                    reply = _freeform_llm_response(st.session_state.freeform_chat)
            else:
                reply = (
                    "Got it — I've noted that for your care team. "
                    "Is there anything else you'd like to add?"
                )
            st.session_state.freeform_chat.append(
                {"role": "assistant", "content": reply}
            )
            st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TOPIC DETAIL PANEL
# ══════════════════════════════════════════════════════════════════

def render_topic_detail(topic_label: str, topic_key: str):
    """Render the chat + current question for the selected topic."""
    _stc.html("""<script>
    (function(){
        var s=['section[data-testid=\"stMain\"]',
               'div[data-testid=\"stAppViewContainer\"]','.main'];
        for(var i=0;i<s.length;i++){
            var e=window.parent.document.querySelector(s[i]);
            if(e){e.scrollTop=0;break;}
        }
    })();
    </script>""", height=0)
    state        = st.session_state.topic_states[topic_key]
    last_data    = st.session_state.last_checkin.get(topic_key, {})
    has_prev     = st.session_state.has_prev_checkin

    # ── Ensure followup_counts exists (backward compat) ────────────
    if "followup_counts" not in state:
        state["followup_counts"] = {}
    if "last_prompted_step_id" not in state:
        state["last_prompted_step_id"] = None
    if "last_prompted_text" not in state:
        state["last_prompted_text"] = ""

    # ── Urgency banner (Tier 1–3 from multi-agent system) ───────────
    render_urgency_banner()

    # ── Previous check-in summary content ────────────────────────
    chips_html = _checkin_summary_html(topic_key, last_data) if (has_prev and last_data) else ""
    inline_summary_html = ""
    if has_prev:
        summary_inner = (
            '<div class="chat-shell-inline-summary-body">'
            '<div style="font-size:12px;color:#5f7386;line-height:1.5;margin-bottom:6px;">'
            'These answers are from your last visit. You can change any of them for this visit.'
            '</div>'
            + (chips_html if chips_html else '<div style="font-size:12px;color:#6d7f90;">No prior summary recorded for this topic.</div>')
            + '</div>'
        )
        inline_summary_html = (
            '<div class="chat-shell-inline-summary">'
            '<details>'
            '<summary>Last visit summary</summary>'
            f'{summary_inner}'
            '</details>'
            '</div>'
        )

    # ── Initialize topic on first visit ─────────────────────────
    if state["status"] == "not_started":
        state["status"] = "in_progress"
        intro = TOPIC_INTROS.get(topic_key, "Let's go through this section together.")
        state["chat"] = [{"role": "assistant", "content": intro}]
        first_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
        if first_step:
            _ensure_step_prompted(topic_key, state, first_step)

    # ── Header with progress bar ─────────────────────────────────
    header_html = (
        '<div class="chat-shell">'
        '  <div class="chat-shell-header">'
        '    <div class="chat-shell-title">'
        f'      <div class="chat-shell-name">{_html.escape(topic_label)}</div>'
        '    </div>'
        f'    {inline_summary_html}'
        '  </div>'
        '  <div class="chat-history">'
    )
    st.markdown(
        header_html,
        unsafe_allow_html=True,
    )

    # ── Chat history ─────────────────────────────────────────────
    if state["chat"]:
        with st.container(border=False):
            for msg in state["chat"]:
                render_chat_bubble(msg["role"], msg["content"])

    # ── Completed ────────────────────────────────────────────────
    if state["status"] == "completed":
        st.markdown(
            '<div class="completion-badge">✅ This topic is complete</div>',
            unsafe_allow_html=True,
        )
        if st.button("✏️ Add a note or correction", key=f"reopen_{topic_key}"):
            state["status"] = "in_progress"
            state["chat"].append({
                "role": "assistant",
                "content": "Of course — please share any correction or additional detail.",
            })
            state["data"].pop("_correction_note", None)
            st.rerun()
        st.markdown('</div><div class="composer-wrap"></div></div>', unsafe_allow_html=True)
        return

    # ── Current question ─────────────────────────────────────────
    if state.get("waiting_for_followup"):
        pending = state.get("pending_followup") or {}
        pending_suffix = pending.get("answer_key", "pending")
        pending_key = f"pending_followup_{topic_key}_{pending_suffix}"
        pending_submit_key = f"{pending_key}_submitted"
        if pending_key not in st.session_state:
            st.session_state[pending_key] = ""
        st.markdown('</div><div class="composer-wrap">', unsafe_allow_html=True)

        st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
        with st.container():
            pending_text = st.text_input(
                "Reply",
                key=pending_key,
                placeholder="Type or speak your answer here...",
                label_visibility="collapsed",
            )
        with st.container():
            pending_voice = voice_widget(f"pending_{topic_key}_{pending_suffix}", label="Mic")
        if pending_voice and pending_voice != st.session_state.get(f"{pending_key}_voice_sync"):
            st.session_state[f"{pending_key}_voice_sync"] = pending_voice
            st.session_state[pending_submit_key] = pending_voice
            handle_pending_followup(topic_key, pending_voice, source="voice")
            return

        if pending_text and st.session_state.get(pending_submit_key) != pending_text:
            st.session_state[pending_submit_key] = pending_text
            handle_pending_followup(topic_key, pending_text, source="followup")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        return
    next_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
    if next_step:
        _ensure_step_prompted(topic_key, state, next_step)
        st.markdown('</div><div class="composer-wrap">', unsafe_allow_html=True)
        render_input(topic_key, next_step)
        st.markdown('</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR  (MASTER PANEL)
# ══════════════════════════════════════════════════════════════════



def render_sidebar():
    with st.sidebar:
        # ── Header ───────────────────────────────────────────────
        _urg_html = render_urgency_indicator_html()
        st.markdown(
            f'<div style="font-size:18px;font-weight:800;color:#10233d;margin:0 0 2px 0;letter-spacing:-0.03em;">🩺 ChatReport</div>'
            f'<div style="font-size:11px;color:#6b7b92;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.08em;">'
            f'Clinical symptom assistant &nbsp;{_urg_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.patient_name:
            st.markdown(
                f'<div style="font-size:11.5px;color:#6b7280;margin-bottom:8px;">'                f'Patient: <strong>{_html.escape(st.session_state.patient_name)}</strong></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<hr style="margin:6px 0 10px 0;border:none;border-top:1px solid #dde6f5;">',
            unsafe_allow_html=True,
        )

        # ── Overall progress ─────────────────────────────────────
        completed   = sum(1 for _, k in TOPICS
                          if st.session_state.topic_states[k]["status"] == "completed")
        in_progress = sum(1 for _, k in TOPICS
                          if st.session_state.topic_states[k]["status"] == "in_progress")
        total = len(TOPICS)

        st.markdown(
            f'<div class="prog-label">{completed}/{total} topics complete</div>',
            unsafe_allow_html=True,
        )
        st.progress(completed / total if total > 0 else 0)
        st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

        # ── Topic nav buttons ─────────────────────────────────────
        # Using st.button (not HTML anchors) so session state is preserved.
        # The label has two parts separated by \n:
        #   line 1: "{selected_marker}{icon} {topic_name}"
        #   line 2: summary snippet or "No prior data" (when prev check-in exists)
        # CSS white-space:pre-wrap renders the \n as a real line break.

        has_prev = st.session_state.get("has_prev_checkin", False)
        last_ck  = st.session_state.get("last_checkin", {})

        for label, key in TOPICS:
            status = st.session_state.topic_states[key]["status"]
            icon   = {"completed": "✅", "in_progress": "🔵"}.get(status, "⚪")
            dname  = label.split(" ", 1)[1] if " " in label else label
            marker = "▶ " if st.session_state.selected_topic == key else "   "

            # Build button label
            btn = f"{marker}{icon} {dname}"
            if has_prev:
                prev_data = last_ck.get(key, {})
                if prev_data:
                    snip = _natural_summary(key, prev_data)
                    btn += f"\n   {snip}" if snip else "\n   No data recorded"
                else:
                    btn += "\n   No prior data"

            if st.button(btn, key=f"nav_{key}", use_container_width=True):
                st.session_state.selected_topic = key
                st.rerun()

        # ── Anything else? ────────────────────────────────────────
        ff_msgs  = [m for m in st.session_state.freeform_chat if m["role"] == "user"]
        ff_badge = f" ({len(ff_msgs)})" if ff_msgs else ""
        ff_mark  = "▶ " if st.session_state.selected_topic == "freeform" else "   "
        if st.button(f"{ff_mark}💬 Anything else?{ff_badge}",
                     key="nav_freeform", use_container_width=True):
            st.session_state.selected_topic = "freeform"
            st.rerun()

        # ── Submit ────────────────────────────────────────────────
        st.markdown(
            '<hr style="margin:8px 0 8px 0;border:none;border-top:1px solid #dde6f5;">',
            unsafe_allow_html=True,
        )
        any_started = completed >= 1 or in_progress >= 1
        if any_started:
            if st.button("📤 Submit Check-In", use_container_width=True,
                         type="primary", key="sidebar_submit"):
                all_data = _build_all_topic_data()
                if ff_msgs:
                    all_data["freeform_notes"] = [
                        m["content"] for m in st.session_state.freeform_chat
                        if m["role"] == "user"
                    ]
                with st.spinner("Generating report…"):
                    report = generate_report(st.session_state.patient_name, all_data)
                st.session_state.report = report
                with st.spinner("Saving…"):
                    save_to_sheet(st.session_state.patient_name, all_data, report)
                st.session_state.report_saved = True
                st.session_state.app_stage = "report"
                st.rerun()

# ══════════════════════════════════════════════════════════════════
# SCREENS
# ══════════════════════════════════════════════════════════════════

TOPIC_LABELS = {key: label for label, key in TOPICS}
TOPIC_KEYS   = [k for _, k in TOPICS]


def screen_login():
    st.markdown("""
    <div class="welcome-card">
        <div style="font-size:12px;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;color:#6b7b92;margin-bottom:8px;">
            Pre-visit symptom check-in
        </div>
        <h1 style="margin:0 0 10px 0; color:#10233d; font-size:34px; letter-spacing:-0.04em;">🩺 ChatReport</h1>
        <p style="color:#56667d; margin-bottom:14px; font-size:15px; line-height:1.75;">
        A modern clinical check-in for patients receiving head and neck cancer treatment.
        Your answers help your care team review symptoms before the visit.
        </p>
        <div class="subtle-note">
            You can answer by typing or by voice. You may switch topics at any time, and your most recent prior check-in will be used to guide the conversation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        name = st.text_input("Please enter your name:", placeholder="First and last name…")
        if st.button("Begin Check-In →", type="primary", use_container_width=True):
            if name.strip():
                st.session_state.patient_name = name.strip()

                # ── Load previous check-in from Sheets ──────────
                with st.spinner("Loading your previous check-in…"):
                    prev = load_last_checkin(name.strip())

                if prev:
                    st.session_state.last_checkin     = prev
                    st.session_state.has_prev_checkin = True
                else:
                    st.session_state.last_checkin     = {}
                    st.session_state.has_prev_checkin = False

                st.session_state.selected_topic = TOPIC_KEYS[0] if TOPIC_KEYS else None
                st.session_state.app_stage      = "overview"
                st.rerun()
            else:
                st.warning("Please enter your name to continue.")


def screen_overview():
    has_prev = st.session_state.get("has_prev_checkin", False)
    last_ck  = st.session_state.get("last_checkin", {})
    patient  = st.session_state.get("patient_name", "")

    st.markdown(
        '<div class="overview-card">'
        '<div style="font-size:12px;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;color:#6b7b92;margin-bottom:8px;">'
        'Last visit overview'
        '</div>'
        f'<div style="font-size:30px;font-weight:800;letter-spacing:-0.04em;color:#10233d;">'
        f'{_html.escape(patient) if patient else "Your"} previous check-in summary'
        '</div>'
        '<div style="font-size:15px;line-height:1.75;color:#56667d;margin-top:10px;">'
        'Before you start this visit\'s symptom check-in, here is a quick summary of what was recorded last time.'
        '</div>',
        unsafe_allow_html=True,
    )

    if has_prev:
        rows = []
        for label, key in TOPICS:
            prev_data = last_ck.get(key, {})
            if not prev_data:
                continue
            topic_name = label.split(" ", 1)[1] if " " in label else label
            summary = _natural_summary(key, prev_data) or "Information was recorded for this topic."
            detail_html = _checkin_summary_html(key, prev_data) or '<span style="color:#7a8ea4;">No extra details recorded</span>'
            rows.append(
                "<tr>"
                f'<td><div class="overview-topic-name">{_html.escape(topic_name)}</div></td>'
                f'<td><div class="overview-summary-main">{_html.escape(summary)}</div></td>'
                f'<td><div class="overview-summary-details">{detail_html}</div></td>'
                "</tr>"
            )

        if rows:
            st.markdown(
                '<div class="overview-table-wrap">'
                '<table class="overview-table">'
                '<colgroup>'
                '<col class="topic-col">'
                '<col class="summary-col">'
                '<col>'
                '</colgroup>'
                '<thead><tr><th>Topic</th><th>Main Summary</th><th>Details From Last Visit</th></tr></thead>'
                f'<tbody>{"".join(rows)}</tbody>'
                '</table>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="overview-note">'
                'These details are from your last visit. You can change, update, or add anything as you go through today\'s topics.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="overview-note">'
                'A prior visit was found, but there were no summary details available to show here. You can continue to today\'s topics.'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="overview-note">'
            'No previous visit summary was found for you. You can start today\'s symptom check-in now.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        if st.button("Continue to Topics →", type="primary", use_container_width=True):
            st.session_state.app_stage = "main"
            st.rerun()


def screen_main():
    render_sidebar()

    selected = st.session_state.selected_topic

    if not selected:
        st.markdown('<div class="card"><div style="font-size:12px;font-weight:800;color:#6b7b92;text-transform:uppercase;letter-spacing:0.08em;">Getting started</div><div style="font-size:28px;font-weight:800;letter-spacing:-0.03em;margin-top:6px;">Choose a symptom area from the sidebar</div><div style="font-size:14px;color:#5f6f84;line-height:1.7;margin-top:8px;">Move through the check-in in any order you prefer. Your answers are saved into a provider-ready summary for the care team.</div></div>', unsafe_allow_html=True)
        return

    # Route to free-form chat or regular topic
    if selected == "freeform":
        render_freeform_chat()
    else:
        topic_label = TOPIC_LABELS.get(selected, selected)
        render_topic_detail(topic_label, selected)


def screen_report():
    render_sidebar()

    all_data = _build_all_topic_data()
    ff_msgs  = [m for m in st.session_state.freeform_chat if m["role"] == "user"]
    if ff_msgs:
        all_data["freeform_notes"] = [m["content"] for m in ff_msgs]

    if not st.session_state.report:
        with st.spinner("Generating clinical report…"):
            st.session_state.report = generate_report(
                st.session_state.patient_name, all_data
            )
    topic_insights = _report_topic_insights(all_data)

    st.markdown('<div class="report-dashboard">', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:28px;font-weight:800;letter-spacing:-0.04em;color:#10233d;">📄 Clinical Check-In Report</div>'
        f'<div style="font-size:13px;color:#627287;line-height:1.7;margin-top:6px;"><strong>Patient:</strong> {_html.escape(st.session_state.patient_name)} &nbsp;|&nbsp; <strong>Date:</strong> {datetime.now().strftime("%B %d, %Y")}</div>',
        unsafe_allow_html=True,
    )
    _render_report_summary_banner(topic_insights)

    for row_start in range(0, len(topic_insights), 4):
        row_items = topic_insights[row_start:row_start + 4]
        cols = st.columns(len(row_items))
        for col, insight in zip(cols, row_items):
            with col:
                _render_report_topic_card(insight)
                _render_report_topic_detail(insight, all_data)

    with st.expander("Full clinical narrative report", expanded=False):
        st.markdown('<div class="report-box">', unsafe_allow_html=True)
        st.markdown(st.session_state.report)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅️ Back to Check-In"):
            st.session_state.app_stage = "main"
            st.rerun()

    with col2:
        saved     = st.session_state.get("report_saved", False)
        btn_label = "✅ Saved" if saved else "💾 Save to Google Sheets"
        if st.button(btn_label, type="primary", disabled=saved):
            with st.spinner("Saving…"):
                _init_sheets()
                save_to_sheet(
                    st.session_state.patient_name,
                    all_data,
                    st.session_state.report,
                )
            st.session_state.report_saved = True
            st.success("Saved successfully!")
            st.rerun()

    with col3:
        if st.button("📋 Copy to Clipboard (manual)"):
            st.info("Select the report text above and copy (Ctrl+C / Cmd+C).")


# ══════════════════════════════════════════════════════════════════
# MAIN DISPATCH
# ══════════════════════════════════════════════════════════════════

_init_sheets()

stage = st.session_state.get("app_stage", "login")

if stage == "login":
    screen_login()
elif stage == "overview":
    screen_overview()
elif stage == "main":
    screen_main()
elif stage == "report":
    screen_report()
else:
    screen_login()
