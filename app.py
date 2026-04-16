import hashlib
import html as _html
import io
import json
import re
from datetime import datetime
from typing import Optional

import streamlit as st
import streamlit.components.v1 as _stc
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI


def _extract_json_object(text: str) -> dict:
    """Parse a JSON object from LLM output, tolerating surrounding text."""
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

.chat-row {
    display: flex !important;
    width: 100% !important;
    margin-bottom: 10px;
    align-items: flex-start;
}

.chat-row.assistant {
    justify-content: flex-start !important;
    padding-right: 24%;
}

.chat-row.user {
    justify-content: flex-end !important;
    padding-left: 24%;
}

.chat-bubble {
    display: inline-block;
    width: auto !important;
    max-width: min(72%, 680px);
    border-radius: 16px;
    padding: 0.8rem 0.95rem;
    border: 1px solid rgba(215, 228, 239, 0.9);
    background: #ffffff;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
}

.chat-row.assistant .chat-bubble {
    margin-left: 0 !important;
    margin-right: auto !important;
    border-left: 3px solid #b7d5eb;
}

.chat-row.user .chat-bubble {
    margin-left: auto !important;
    margin-right: 0 !important;
    background: #f8fbfe;
    border-left: none;
    border-right: 3px solid #0f6cbd;
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

.chat-shell {
    background: rgba(255,255,255,0.82);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 14px;
    box-shadow: var(--shadow);
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
    width: auto;
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


.reply-shell {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
    box-shadow: none;
    margin-top: 4px;
}

.composer-shell {
    background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
    border: 1px solid #d9e4ed;
    border-radius: 24px;
    padding: 10px;
    box-shadow: 0 12px 26px rgba(23, 50, 74, 0.05);
}

.composer-shell.compact {
    padding: 8px;
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
    width: auto !important;
    min-width: 0 !important;
    padding: 0.46rem 0.9rem !important;
    border-radius: 999px !important;
    font-size: 13px !important;
    box-shadow: none !important;
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
    min-height: 46px !important;
    height: 46px !important;
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
    margin: 6px 0 0 0;
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
    height: 46px !important;
    min-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 1px solid #d7e4ee !important;
    background: linear-gradient(180deg, #ffffff 0%, #f5f9fd 100%) !important;
    box-shadow: 0 8px 18px rgba(23, 50, 74, 0.08) !important;
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


def _step_prompt_text(step: dict) -> str:
    question_text = step["text"]
    if step.get("type") == "options":
        question_text += " (Choose an option below, or answer in your own words if that fits better.)"
    return question_text


def _append_assistant_message(state: dict, text: str):
    text = (text or "").strip()
    if not text:
        return
    if state["chat"] and state["chat"][-1]["role"] == "assistant" and state["chat"][-1]["content"].strip() == text:
        return
    state["chat"].append({"role": "assistant", "content": text})


def render_chat_bubble(role: str, content: str):
    safe = _html.escape(content or "").replace("\n", "<br>")
    role_cls = "user" if role == "user" else "assistant"
    st.markdown(
        f'<div class="chat-row {role_cls}"><div class="chat-bubble">{safe}</div></div>',
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


# Sheet column order: timestamp, name, one column per topic, freeform notes, data_json (hidden, for reloading)
_SHEET_TOPIC_HEADERS = [label.split(" ", 1)[1] if " " in label else label for label, _ in [
    ("🩹 Pain & Medications",   "pain"),
    ("🍽️  Nutrition & Fluids",   "nutrition"),
    ("👄 Oral Symptoms",         "oral"),
    ("🤢 GI Symptoms",           "gi"),
    ("😴 Fatigue & Sleep",       "fatigue"),
    ("🚶 Activity Level",        "activity"),
    ("🧠 Mood",                  "mood"),
    ("💊 Other Symptoms",        "other"),
]]
_SHEET_HEADERS = ["timestamp", "name"] + _SHEET_TOPIC_HEADERS + ["Freeform Notes", "data_json"]
_DATA_JSON_COL  = len(_SHEET_HEADERS) - 1   # 0-based index of the data_json column


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
            ws = book.add_worksheet(title="ChatReport", rows=2000, cols=len(_SHEET_HEADERS))
            ws.append_row(_SHEET_HEADERS)
        _sheet = ws
    except Exception as e:
        _sheet_error = str(e)


def format_topic_data(topic_key: str, data: dict) -> str:
    """
    Convert a topic's collected answers into a readable multi-line string
    for the Google Sheets column. Pure code — no LLM.
    Each answered question becomes one line: "Question label: Answer"
    """
    if not data:
        return ""
    lines = []
    for step in FLOWS.get(topic_key, []):
        answer = data.get(step["id"])
        if answer is None:
            continue
        # Format list answers as comma-separated
        val = ", ".join(str(v) for v in answer) if isinstance(answer, list) else str(answer)
        # Use question text as label, trimmed to 55 chars
        label = step["text"].rstrip("?").strip()
        if len(label) > 55:
            label = label[:52] + "..."
        lines.append(f"{label}: {val}")
        # Include any follow-up answer captured by LLM Role 2
        followup = data.get(f"{step['id']}_followup")
        if followup:
            lines.append(f"  Follow-up: {followup}")
        # Include free-text "other" detail if present
        other_detail = data.get(f"{step['id']}_other_detail")
        if other_detail:
            lines.append(f"  Detail: {other_detail}")
    return "\n".join(lines)


def save_to_sheet(name: str, all_data: dict) -> bool:
    """
    Append one row to the Google Sheet.
    Columns: timestamp | name | <one per topic> | freeform notes | data_json
    Returns True on success, False on failure.
    """
    _init_sheets()
    if _sheet is None:
        st.error(f"Could not connect to Google Sheets: {_sheet_error}")
        return False
    try:
        topic_keys = ["pain", "nutrition", "oral", "gi", "fatigue", "activity", "mood", "other"]
        topic_cols = [format_topic_data(key, all_data.get(key, {})) for key in topic_keys]
        freeform   = "\n".join(all_data.get("freeform_notes", []))
        row = (
            [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name]
            + topic_cols
            + [freeform, json.dumps(all_data, ensure_ascii=False)]
        )
        _sheet.append_row(row)
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

# Each entry: (display_label, internal_key)
TOPICS = [
    ("🩹 Pain & Medications",   "pain"),
    ("🍽️  Nutrition & Fluids",   "nutrition"),
    ("👄 Oral Symptoms",         "oral"),
    ("🤢 GI Symptoms",           "gi"),
    ("😴 Fatigue & Sleep",       "fatigue"),
    ("🚶 Activity Level",        "activity"),
    ("🧠 Mood",                  "mood"),
    ("💊 Other Symptoms",        "other"),
]

TOPIC_INTROS = {
    "pain":      "Let's talk about any pain you've been having, what you're taking for it, and whether that regimen is helping.",
    "nutrition": "I'd like to ask about your eating, drinking, and weight.",
    "oral":      "Let's go over any mouth and throat symptoms like sticky mucus, thrush, dryness, and what you're using to manage them.",
    "gi":        "I'll ask about nausea, vomiting, diarrhea, constipation, and how you're managing those symptoms.",
    "fatigue":   "Let's discuss how your energy and sleep have been.",
    "activity":  "Tell me about how your daily activities have been going.",
    "mood":      "This section covers how you've been feeling emotionally and your support system.",
    "other":     "Finally, let's cover any other symptoms — breathing, skin, hearing, and more.",
}


def _q(id, text, type="options", opts=None, when=None,
        placeholder="Please describe...", min_v=0, max_v=10, default_v=0):
    """Helper to build a question step dict."""
    return {
        "id": id, "text": text, "type": type,
        "opts": opts or [], "when": when,
        "placeholder": placeholder,
        "min_v": min_v, "max_v": max_v, "default_v": default_v,
    }


def _safe_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ── PAIN & MEDICATIONS (Main 2, 3, 12, 38) ────────────────────────
FLOW_PAIN = [
    # Main 2
    _q("has_pain", "Do you have any pain today?", opts=["Yes", "No"]),

    _q("pain_location", "Where exactly is the pain?",
       opts=["Throat", "Tongue", "Somewhere else"],
       when=lambda d: d.get("has_pain") == "Yes"),

    _q("throat_timing",
       "Is the throat pain there all the time, or only when you swallow or eat?",
       opts=["All the time", "Only when swallowing", "Only when eating",
             "Both swallowing and eating"],
       when=lambda d: d.get("pain_location") == "Throat"),

    _q("throat_severity",
       "On a scale of 0–10, how bad is the throat pain at its worst?",
       type="number", min_v=0, max_v=10, default_v=5,
       when=lambda d: d.get("pain_location") == "Throat"),

    _q("throat_med_helps",
       "Are you taking pain medication for this? Is it helping?",
       opts=["Yes, it helps", "Yes, but it's not enough", "No, I'm not taking anything"],
       when=lambda d: (d.get("pain_location") == "Throat"
                       and _safe_int(d.get("throat_severity", 0)) > 4)),

    _q("tongue_type",
       "Is it a sore or ulcer on the tongue, or a general painful feeling?",
       opts=["There's a sore/ulcer", "Just pain, no visible sore"],
       when=lambda d: d.get("pain_location") == "Tongue"),

    _q("tongue_spot",
       "Is the pain in one specific spot, or does it spread?",
       opts=["One spot", "Spreads across tongue", "Whole mouth"],
       when=lambda d: d.get("pain_location") == "Tongue"),

    _q("tongue_severity",
       "On a scale of 0–10, how bad is the tongue pain at its worst?",
       type="number", min_v=0, max_v=10, default_v=5,
       when=lambda d: d.get("pain_location") == "Tongue"),

    _q("other_pain_desc",
       "Can you describe where the pain is?",
       type="free_text", placeholder="e.g., near my jaw and ear…",
       when=lambda d: d.get("pain_location") == "Somewhere else"),

    _q("pain_medications",
       "Which medications are you currently taking for pain?",
       type="multi_select",
       opts=["Gabapentin", "Oxycodone", "Butrans patch", "Other", "No pain medication"]),

    _q("med_dose_freq",
       "How often are you taking your pain medication, and at what dose?",
       type="free_text", placeholder="e.g., Oxycodone 5mg every 6 hours…",
       when=lambda d: (bool(d.get("pain_medications"))
                       and "No pain medication" not in (d.get("pain_medications") or []))),

    _q("taking_as_prescribed",
       "Are you taking your medications as prescribed?",
       opts=["Yes", "No"]),

    _q("med_adherence_issue",
       "What is making it difficult to take your medications?",
       opts=["Side effects", "Schedule", "Access issues", "Other"],
       when=lambda d: d.get("taking_as_prescribed") == "No"),

    _q("med_side_effects",
       "Are you experiencing any side effects from your medications?",
       opts=["Yes", "No"],
       when=lambda d: d.get("taking_as_prescribed") == "Yes"),
]

# ── NUTRITION & FLUIDS (Main 5, 6, 8, 25, 26, 27, 34) ─────────────
FLOW_NUTRITION = [
    _q("eating_ability",
       "How has your eating been since your last visit?",
       opts=["Eating normally — no problems",
             "Eating less than usual, but managing",
             "Struggling — only liquids or very little",
             "Not eating — using a feeding tube only"]),

    _q("fluid_intake_managing",
       "Are you drinking enough fluids throughout the day — water, shakes, or other drinks?",
       opts=["Yes, drinking well", "A little less than usual", "Struggling to drink enough"],
       when=lambda d: d.get("eating_ability") == "Eating less than usual, but managing"),

    _q("food_type",
       "What are you able to eat right now?",
       opts=["Mostly normal food", "Soft foods only (yogurt, soup, pudding)",
             "Mix of soft and liquid", "Mainly liquids"],
       when=lambda d: d.get("eating_ability") == "Eating less than usual, but managing"),

    _q("nutritional_shakes",
       "How many nutritional shakes or Boost/Ensure drinks are you having per day?",
       opts=["None", "1–2", "3–4", "More than 4"],
       when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),

    _q("eating_barrier",
       "What is stopping you from eating more?",
       opts=["Pain when eating/swallowing", "Feel full very quickly",
             "No appetite", "Nausea", "Too tired to prepare food"],
       when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),

    _q("fluid_struggling",
       "Are you drinking enough fluids — water, juice, or anything?",
       opts=["Yes, drinking well", "A little", "Very little, hard to drink"],
       when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),

    _q("fluid_barrier",
       "What's making it hard to drink?",
       opts=["Pain when swallowing", "Dry mouth", "Nausea", "Just not thirsty"],
       when=lambda d: (d.get("eating_ability") == "Struggling — only liquids or very little"
                       and d.get("fluid_struggling") in ["A little", "Very little, hard to drink"])),

    _q("pain_med_timing",
       "Are you timing your pain medication before meals to make eating easier?",
       opts=["Yes, it helps", "I try, but it's not enough",
             "No, I didn't know to do this", "No, I don't take pain medication"],
       when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),

    _q("tube_issues",
       "Is the tube feeding going well — no blockages, leaks, or discomfort around the site?",
       opts=["Working fine", "Some issues — leaking or blockage",
             "Discomfort/soreness around the tube"],
       when=lambda d: d.get("eating_ability") == "Not eating — using a feeding tube only"),

    _q("tube_oral_sips",
       "Are you still able to take any sips of water or liquids by mouth at all?",
       opts=["Yes, small amounts", "Very occasionally for comfort", "No, nothing by mouth"],
       when=lambda d: d.get("eating_ability") == "Not eating — using a feeding tube only"),

    _q("weight",
       "What has your weight been recently? (Enter in pounds)",
       type="number", min_v=50, max_v=500, default_v=150),

    _q("weight_impact",
       "Has any weight change been affecting how you feel or your energy levels?",
       opts=["Yes, I've noticed a difference", "Not really"]),

    _q("swallowing_difficulty",
       "Are you having any difficulty swallowing — liquids, food, or pills?",
       opts=["Yes", "No"]),

    _q("swallowing_type",
       "Is it painful to swallow, or just mechanically difficult?",
       opts=["Painful to swallow", "Mechanically difficult"],
       when=lambda d: d.get("swallowing_difficulty") == "Yes"),

    _q("choking_with_eating",
       "Do you cough or choke when you eat?",
       opts=["Yes", "No"],
       when=lambda d: d.get("swallowing_difficulty") == "Yes"),

    _q("swallowing_method",
       "Are you still able to swallow liquids by mouth, or is everything through a feeding tube?",
       opts=["I swallow by mouth", "Everything through the feeding tube"],
       when=lambda d: d.get("swallowing_difficulty") == "Yes"),

    _q("choking_coughing",
       "Are you having any difficulty with choking or coughing when eating or drinking?",
       opts=["Yes", "No"]),

    _q("choking_type",
       "Does it happen with liquids, solids, or both?",
       opts=["Liquids", "Solids", "Both"],
       when=lambda d: d.get("choking_coughing") == "Yes"),

    _q("choking_frequency",
       "Does it happen every time you eat, or only occasionally?",
       opts=["Every time", "Occasionally"],
       when=lambda d: d.get("choking_coughing") == "Yes"),

    _q("choking_pills",
       "Does it also happen when you take pills?",
       opts=["Yes", "No"],
       when=lambda d: d.get("choking_coughing") == "Yes"),

    _q("iv_fluids",
       "Are you currently receiving IV fluids or hydration treatments?",
       opts=["Yes", "No"]),

    _q("iv_frequency",
       "How often are you receiving IV fluids?",
       type="free_text", placeholder="e.g., twice a week…",
       when=lambda d: d.get("iv_fluids") == "Yes"),

    _q("iv_helping",
       "Do you feel the IV fluids are helping?",
       opts=["Yes", "No"],
       when=lambda d: d.get("iv_fluids") == "Yes"),

    _q("iv_adjust",
       "Would you like to adjust the frequency of your hydration visits?",
       opts=["Yes", "No"],
       when=lambda d: d.get("iv_fluids") == "Yes"),

    _q("need_hydration",
       "Do you feel like you might need hydration support?",
       opts=["Yes", "No"],
       when=lambda d: d.get("iv_fluids") == "No"),

    _q("feeding_tube",
       "Are you currently using a feeding tube?",
       opts=["Yes", "No"],
       when=lambda d: d.get("eating_ability") != "Not eating — using a feeding tube only"),

    _q("tube_status",
       "Is the feeding tube working well or are there issues?",
       opts=["Working well", "Leakage", "Blockage", "Discomfort"],
       when=lambda d: (d.get("feeding_tube") == "Yes"
                       and d.get("eating_ability") != "Not eating — using a feeding tube only")),

    _q("tube_oral",
       "Are you able to take anything by mouth at all?",
       opts=["Yes, some", "No, nothing by mouth"],
       when=lambda d: (d.get("feeding_tube") == "Yes"
                       and d.get("eating_ability") != "Not eating — using a feeding tube only")),

    _q("taste_changes",
       "Have you noticed any changes in your sense of taste?",
       opts=["Yes", "No"]),

    _q("taste_type",
       "Does food taste different, bland, or unpleasant?",
       opts=["Different", "Bland", "Unpleasant"],
       when=lambda d: d.get("taste_changes") == "Yes"),

    _q("taste_eating_impact",
       "Is the taste change affecting your ability to eat?",
       opts=["Yes", "No"],
       when=lambda d: d.get("taste_changes") == "Yes"),
]

# ── ORAL SYMPTOMS (Main 4, 7, 10, 24, 33) ─────────────────────────
FLOW_ORAL = [
    _q("mouth_sores",
       "Do you have any mouth sores, ulcers, or white patches/thrush right now?",
       opts=["Yes", "No"]),

    _q("sore_new_or_old",
       "Is this new since your last visit, or have you had it for a while?",
       opts=["New", "Not sure", "Same one as before"],
       when=lambda d: d.get("mouth_sores") == "Yes"),

    _q("sore_location",
       "Where exactly is it?",
       opts=["Inside the mouth/cheek", "On the tongue", "Back of the throat",
             "Gums/lips", "Multiple spots"],
       when=lambda d: (d.get("mouth_sores") == "Yes"
                       and d.get("sore_new_or_old") in ["New", "Not sure"])),

    _q("sore_pain_impact",
       "Is the sore painful? Is it affecting your ability to eat or drink?",
       opts=["No pain, just noticed it", "A little, but manageable",
             "Yes, can't eat/drink comfortably"],
       when=lambda d: (d.get("mouth_sores") == "Yes"
                       and d.get("sore_new_or_old") in ["New", "Not sure"])),

    _q("magic_mouthwash",
       "Are you using anything for it, like magic mouthwash or thrush medicine? If yes, is it helping?",
       opts=["Yes, it helps", "Yes, but not enough",
             "No, I don't have it", "No, I don't use it"],
       when=lambda d: (d.get("mouth_sores") == "Yes"
                       and d.get("sore_new_or_old") in ["New", "Not sure"])),

    _q("sore_progression",
       "Is the sore getting better, staying the same, or getting worse?",
       opts=["Getting better", "About the same", "Getting worse", "Not sure"],
       when=lambda d: (d.get("mouth_sores") == "Yes"
                       and d.get("sore_new_or_old") == "Same one as before")),

    _q("sore_eating_impact_old",
       "Is it still preventing you from eating or drinking comfortably?",
       opts=["Yes", "A little", "No"],
       when=lambda d: (d.get("mouth_sores") == "Yes"
                       and d.get("sore_new_or_old") == "Same one as before"
                       and d.get("sore_progression") in ["About the same", "Getting worse"])),

    _q("dry_mouth",
       "Are you experiencing any dryness in your mouth?",
       opts=["Yes", "No"]),

    _q("dry_mouth_timing",
       "Is the dryness worse at night or all day?",
       opts=["Worse at night", "All day"],
       when=lambda d: d.get("dry_mouth") == "Yes"),

    _q("dry_mouth_med",
       "Are you using any medication like Biotene or a saliva substitute?",
       opts=["Yes", "No"],
       when=lambda d: d.get("dry_mouth") == "Yes"),

    _q("dry_mouth_impact",
       "Is the dryness making it harder to eat, talk, or sleep?",
       opts=["Yes", "No"],
       when=lambda d: d.get("dry_mouth") == "Yes"),

    _q("mucus_issues",
       "Are you having problems with mucus or thick secretions in your throat?",
       opts=["Yes", "No"]),

    _q("mucus_type",
       "Is the mucus thick and hard to clear, or more watery?",
       opts=["Thick", "More watery"],
       when=lambda d: d.get("mucus_issues") == "Yes"),

    _q("mucus_impact",
       "Is the mucus affecting your ability to swallow or sleep?",
       opts=["Yes", "No"],
       when=lambda d: d.get("mucus_issues") == "Yes"),

    _q("mucus_management",
       "Are you using anything to manage it — like Robitussin or saline rinses?",
       opts=["Yes", "No"],
       when=lambda d: d.get("mucus_issues") == "Yes"),

    _q("teeth_gum_issues",
       "Are you having any problems with your teeth or gums?",
       opts=["Yes", "No"]),

    _q("teeth_issue_type",
       "Is there pain, bleeding, or sores with your teeth or gums?",
       opts=["Pain", "Bleeding", "Sores", "Multiple issues"],
       when=lambda d: d.get("teeth_gum_issues") == "Yes"),

    _q("brushing_difficult",
       "Is it making brushing difficult?",
       opts=["Yes", "No"],
       when=lambda d: d.get("teeth_gum_issues") == "Yes"),

    _q("avoiding_brushing",
       "Are you avoiding brushing because of the discomfort?",
       opts=["Yes", "No"],
       when=lambda d: d.get("teeth_gum_issues") == "Yes"),

    _q("oral_rinse_use",
       "Are you using mouthwash or oral rinses regularly?",
       opts=["Yes", "No"]),

    _q("oral_rinse_type",
       "What type are you using?",
       type="free_text",
       placeholder="e.g., magic mouthwash, salt/baking soda rinse…",
       when=lambda d: d.get("oral_rinse_use") == "Yes"),

    _q("oral_rinse_helping",
       "Is it helping?",
       opts=["Yes", "No"],
       when=lambda d: d.get("oral_rinse_use") == "Yes"),

    _q("oral_rinse_open",
       "Would you be open to trying an oral rinse to help with symptoms?",
       opts=["Yes", "No"],
       when=lambda d: d.get("oral_rinse_use") == "No"),
]

# ── GI SYMPTOMS (Main 11, 18) ─────────────────────────────────────
FLOW_GI = [
    _q("nausea_vomiting",
       "Have you had any nausea, vomiting, or diarrhea since your last visit?",
       type="multi_select",
       opts=["Nausea", "Vomiting", "Diarrhea", "None of these"]),

    _q("nausea_frequency",
       "How often are you feeling nauseated?",
       type="free_text",
       placeholder="e.g., a few times a day, mostly in the mornings…",
       when=lambda d: "Nausea" in (d.get("nausea_vomiting") or [])),

    _q("nausea_management",
       "What are you using for nausea, and is it helping?",
       type="free_text",
       placeholder="e.g., Zofran twice a day and it helps a little…",
       when=lambda d: "Nausea" in (d.get("nausea_vomiting") or [])),

    _q("vomiting_frequency",
       "How often are you vomiting and how much?",
       type="free_text",
       placeholder="e.g., once or twice a day, small amounts…",
       when=lambda d: "Vomiting" in (d.get("nausea_vomiting") or [])),

    _q("vomiting_management",
       "What are you doing to manage the vomiting, and is it helping?",
       type="free_text",
       placeholder="e.g., anti-nausea medication, small sips, and it is helping some…",
       when=lambda d: "Vomiting" in (d.get("nausea_vomiting") or [])),

    _q("diarrhea_frequency",
       "How often are you having diarrhea?",
       type="free_text",
       placeholder="e.g., three loose stools a day…",
       when=lambda d: "Diarrhea" in (d.get("nausea_vomiting") or [])),

    _q("diarrhea_management",
       "Are you taking anything for the diarrhea, and is it helping?",
       type="free_text",
       placeholder="e.g., Imodium and it helps some…",
       when=lambda d: "Diarrhea" in (d.get("nausea_vomiting") or [])),

    _q("constipation",
       "Have you had any constipation or trouble moving your bowels?",
       opts=["Yes", "No"]),

    _q("bowel_frequency",
       "How often are you having bowel movements?",
       type="free_text",
       placeholder="e.g., once every 3 days…",
       when=lambda d: d.get("constipation") == "Yes"),

    _q("constipation_meds",
       "Are you taking anything like Senna, Miralax, or other medications for constipation?",
       opts=["Yes", "No"],
       when=lambda d: d.get("constipation") == "Yes"),

    _q("bloating",
       "Are you feeling bloated or uncomfortable?",
       opts=["Yes", "No"],
       when=lambda d: d.get("constipation") == "Yes"),
]

# ── FATIGUE & SLEEP (Main 13, 14) ─────────────────────────────────
FLOW_FATIGUE = [
    _q("fatigue",
       "Are you feeling more tired or weak than usual?",
       opts=["Yes", "No"]),

    _q("fatigue_type",
       "Is it a general tiredness, or weakness in specific parts of your body?",
       opts=["General tiredness", "Weakness in specific parts"],
       when=lambda d: d.get("fatigue") == "Yes"),

    _q("weakness_location",
       "In which parts of your body do you feel weakness?",
       type="free_text", placeholder="e.g., legs, arms…",
       when=lambda d: (d.get("fatigue") == "Yes"
                       and d.get("fatigue_type") == "Weakness in specific parts")),

    _q("fatigue_daily_impact",
       "Is the fatigue affecting your daily activities — getting dressed, moving around?",
       opts=["Yes", "No"],
       when=lambda d: d.get("fatigue") == "Yes"),

    _q("medication_drowsy",
       "Are your pain medications making you feel drowsy?",
       opts=["Yes", "No", "Sometimes"]),

    _q("sleep_quality",
       "Are you able to sleep through the night?",
       opts=["Yes", "No"]),

    _q("sleep_wake_reason",
       "Are you waking up at night due to pain, dry mouth, or coughing?",
       type="free_text",
       placeholder="e.g., pain wakes me up around 3am…",
       when=lambda d: d.get("sleep_quality") == "No"),

    _q("drowsy_schedule",
       "Is drowsiness from medication affecting your normal wake/sleep schedule?",
       opts=["Yes", "No"],
       when=lambda d: d.get("sleep_quality") == "No"),
]

# ── ACTIVITY LEVEL (Main 30) ───────────────────────────────────────
FLOW_ACTIVITY = [
    _q("activity_level",
       "How is your daily life — are you able to do your usual activities?",
       opts=["Doing everything normally", "Doing less than usual",
             "Struggling with daily tasks"]),

    _q("difficult_activities",
       "What activities are most difficult right now?",
       type="free_text",
       placeholder="e.g., climbing stairs, cooking, getting dressed…",
       when=lambda d: d.get("activity_level") in
             ["Doing less than usual", "Struggling with daily tasks"]),

    _q("activity_limiting_factor",
       "Is the difficulty mainly due to pain, fatigue, or something else?",
       opts=["Pain", "Fatigue", "Both", "Something else"],
       when=lambda d: d.get("activity_level") in
             ["Doing less than usual", "Struggling with daily tasks"]),

    _q("activity_other_desc",
       "Can you tell me more about what's limiting your activities?",
       type="free_text", placeholder="e.g., balance issues, weakness…",
       when=lambda d: d.get("activity_limiting_factor") == "Something else"),
]

# ── MOOD (Main 15, 35, 39) ─────────────────────────────────────────
FLOW_MOOD = [
    _q("emotional_state",
       "How are you feeling emotionally? Are you feeling anxious or worried about anything?",
       type="free_text",
       placeholder="Please share how you've been feeling — there are no wrong answers…"),

    _q("anxiety_impact",
       "Is anxiety or worry affecting your sleep, eating, or daily activities?",
       opts=["Yes", "No", "A little"]),

    _q("social_support_quality",
       "Do you have people around you who you can talk to about how you're feeling?",
       opts=["Yes, I have good support", "Some support", "Not really"]),

    _q("feeling_down",
       "Have you been feeling down or depressed?",
       opts=["Yes", "No"]),

    _q("depression_frequency",
       "How often have you been feeling this way?",
       type="free_text",
       placeholder="e.g., most days, occasionally, mostly in the evenings…",
       when=lambda d: d.get("feeling_down") == "Yes"),

    _q("depression_daily_impact",
       "Is it affecting your daily activities or motivation?",
       opts=["Yes", "No"],
       when=lambda d: d.get("feeling_down") == "Yes"),

    _q("support_adequate",
       "Do you feel you have enough support between visits?",
       opts=["Yes", "No"]),

    _q("who_supports",
       "Who is supporting you — family, friends, or caregivers?",
       type="free_text",
       placeholder="e.g., my wife and daughter…",
       when=lambda d: d.get("support_adequate") == "Yes"),

    _q("needed_support",
       "What kind of support would be most helpful right now?",
       type="free_text",
       placeholder="e.g., emotional support, help with transportation, more info about treatment…",
       when=lambda d: d.get("support_adequate") == "No"),
]

# ── OTHER SYMPTOMS (Main 9, 16, 17, 19, 20, 21, 22, 23, 36, 37) ───
FLOW_OTHER = [
    _q("breathing_issues",
       "Are you having any difficulty breathing or shortness of breath?",
       opts=["Yes", "No"]),

    _q("breathing_timing",
       "Is the breathing difficulty constant, or does it come on with activity?",
       opts=["It's constant", "It comes on with activity"],
       when=lambda d: d.get("breathing_issues") == "Yes"),

    _q("wheezing",
       "Are you wheezing or feeling like something is blocking your airway?",
       opts=["Yes", "No"],
       when=lambda d: d.get("breathing_issues") == "Yes"),

    _q("hearing_changes",
       "Do you have any hearing problems or changes recently?",
       opts=["Yes", "No"]),

    _q("hearing_type",
       "Is it ringing in your ears, hearing loss, or both?",
       opts=["Ringing in ears", "Hearing loss", "Both"],
       when=lambda d: d.get("hearing_changes") == "Yes"),

    _q("hearing_constant",
       "Is it constant or does it come and go?",
       opts=["Constant", "Comes and goes"],
       when=lambda d: d.get("hearing_changes") == "Yes"),

    _q("hearing_worsening",
       "Has it gotten worse compared to your last visit?",
       opts=["Yes", "No"],
       when=lambda d: d.get("hearing_changes") == "Yes"),

    _q("dizziness",
       "Have you been feeling dizzy or lightheaded?",
       opts=["Yes", "No"]),

    _q("dizziness_timing",
       "Is it constant or only when you stand up or change position?",
       opts=["Constant", "Only when standing or changing position"],
       when=lambda d: d.get("dizziness") == "Yes"),

    _q("dizziness_worsening",
       "Has the dizziness gotten worse recently?",
       opts=["Yes", "No"],
       when=lambda d: d.get("dizziness") == "Yes"),

    _q("falls",
       "Have you had any falls or felt like you might fall?",
       opts=["Yes", "No"],
       when=lambda d: d.get("dizziness") == "Yes"),

    _q("numbness",
       "Have you noticed any numbness or tingling in your hands or feet?",
       opts=["Yes", "No"]),

    _q("numbness_location",
       "Is it in your hands, feet, or both?",
       opts=["Hands", "Feet", "Both"],
       when=lambda d: d.get("numbness") == "Yes"),

    _q("numbness_new",
       "Is it new or getting worse?",
       opts=["New", "Getting worse", "Same as before"],
       when=lambda d: d.get("numbness") == "Yes"),

    _q("numbness_daily_impact",
       "Is it affecting your daily activities?",
       opts=["Yes", "No"],
       when=lambda d: d.get("numbness") == "Yes"),

    _q("fever_chills",
       "Have you had any fever or chills recently?",
       opts=["Yes", "No"]),

    _q("fever_start", "When did the fever or chills start?",
       type="free_text", placeholder="e.g., two days ago…",
       when=lambda d: d.get("fever_chills") == "Yes"),

    _q("fever_temp", "How high was the fever?",
       type="free_text", placeholder="e.g., 101.5°F…",
       when=lambda d: d.get("fever_chills") == "Yes"),

    _q("fever_other_symptoms",
       "Do you have any other symptoms like cough or signs of infection?",
       opts=["Yes", "No"],
       when=lambda d: d.get("fever_chills") == "Yes"),

    _q("bp_monitoring",
       "Are you checking your blood pressure at home?",
       opts=["Yes", "No"]),

    _q("bp_reading", "What has your blood pressure been recently?",
       type="free_text", placeholder="e.g., 130/85…",
       when=lambda d: d.get("bp_monitoring") == "Yes"),

    _q("bp_dizziness",
       "Have you felt dizzy or lightheaded with blood pressure changes?",
       opts=["Yes", "No"],
       when=lambda d: d.get("bp_monitoring") == "Yes"),

    _q("bp_home_monitor",
       "Do you have a way to check your blood pressure at home?",
       opts=["Yes", "No"],
       when=lambda d: d.get("bp_monitoring") == "No"),

    _q("skin_issues",
       "Have you had any skin problems — like irritation, wounds, or redness?",
       opts=["Yes", "No"]),

    _q("skin_location", "Where is the skin issue located?",
       type="free_text", placeholder="e.g., neck, shoulder, near jaw…",
       when=lambda d: d.get("skin_issues") == "Yes"),

    _q("skin_start",
       "When did it start?",
       type="free_text",
       placeholder="e.g., about a week ago, at the start of radiation…",
       when=lambda d: d.get("skin_issues") == "Yes"),

    _q("skin_progression",
       "Is it getting better, worse, or staying the same?",
       opts=["Getting better", "About the same", "Getting worse"],
       when=lambda d: d.get("skin_issues") == "Yes"),

    _q("skin_drainage",
       "Any drainage, bleeding, or open areas?",
       opts=["Yes", "No"],
       when=lambda d: d.get("skin_issues") == "Yes"),

    _q("voice_hoarseness",
       "How is your voice? Have you noticed any hoarseness or trouble speaking?",
       opts=["Yes, problems with my voice", "No, voice is fine"]),

    _q("voice_timing",
       "Is the hoarseness constant or only when you're talking?",
       opts=["Constant", "Only when talking"],
       when=lambda d: d.get("voice_hoarseness") == "Yes, problems with my voice"),

    _q("voice_progression",
       "Has your voice improved or worsened since your last visit?",
       opts=["Improved", "About the same", "Worse"],
       when=lambda d: d.get("voice_hoarseness") == "Yes, problems with my voice"),

    _q("voice_communication_impact",
       "Is it affecting your ability to communicate with others?",
       opts=["Yes", "No"],
       when=lambda d: d.get("voice_hoarseness") == "Yes, problems with my voice"),

    _q("concentration",
       "Have you had trouble concentrating or remembering things?",
       opts=["Yes", "No"]),

    _q("concentration_new",
       "Is it new or ongoing?",
       opts=["New", "Ongoing"],
       when=lambda d: d.get("concentration") == "Yes"),

    _q("concentration_daily_impact",
       "Is it affecting your daily tasks?",
       opts=["Yes", "No"],
       when=lambda d: d.get("concentration") == "Yes"),

    _q("sexual_health",
       "Have you had any sexual health concerns or changes?",
       opts=["Yes", "Prefer not to say", "No"]),

    _q("sexual_discuss",
       "Would you like to discuss this further with your provider?",
       opts=["Yes", "No"],
       when=lambda d: d.get("sexual_health") == "Yes"),

    _q("sexual_cause",
       "Is it related to treatment, energy levels, or something else?",
       opts=["Treatment side effects", "Energy levels", "Other"],
       when=lambda d: d.get("sexual_health") == "Yes"),
]

# Master flow registry
FLOWS = {
    "pain":      FLOW_PAIN,
    "nutrition": FLOW_NUTRITION,
    "oral":      FLOW_ORAL,
    "gi":        FLOW_GI,
    "fatigue":   FLOW_FATIGUE,
    "activity":  FLOW_ACTIVITY,
    "mood":      FLOW_MOOD,
    "other":     FLOW_OTHER,
}


# ══════════════════════════════════════════════════════════════════
# FLOW ENGINE
# ══════════════════════════════════════════════════════════════════

def get_next_step(topic_key: str, data: dict) -> Optional[dict]:
    """Return the first unanswered applicable step for this topic."""
    for step in FLOWS.get(topic_key, []):
        when = step.get("when")
        if when and not when(data):
            continue
        if step["id"] not in data:
            return step
    return None


def topic_is_complete(topic_key: str, data: dict) -> bool:
    return get_next_step(topic_key, data) is None


def get_topic_progress(topic_key: str, data: dict) -> tuple[int, int]:
    """Returns (answered, applicable) counts."""
    flow = FLOWS.get(topic_key, [])
    applicable = [s for s in flow if not s.get("when") or s["when"](data)]
    answered = [s for s in applicable if s["id"] in data]
    return len(answered), len(applicable)


# ══════════════════════════════════════════════════════════════════
# LLM FUNCTIONS  (3 clear roles + freeform chat)
# ══════════════════════════════════════════════════════════════════

# ── Shared clinical context injected into every LLM call ─────────
_SYSTEM_CONTEXT = (
    "You are a compassionate nurse assistant helping head and neck cancer patients "
    "complete a symptom check-in before their visit. "
    "Always use simple, warm language. Never diagnose or prescribe."
)

_TOPIC_LABELS_LLM = {
    "pain":      "Pain & Medications",
    "nutrition": "Nutrition & Fluids",
    "oral":      "Oral Symptoms",
    "gi":        "GI Symptoms",
    "fatigue":   "Fatigue & Sleep",
    "activity":  "Daily Activity",
    "mood":      "Mood & Support",
    "other":     "Other Symptoms",
}

_RED_FLAGS = (
    "- Pain ≥ 7/10, uncontrolled\n"
    "- Fever ≥ 100.4°F\n"
    "- Weight loss > 5 lbs since last visit\n"
    "- Unable to swallow any liquids\n"
    "- Feeding tube complications\n"
    "- Breathing difficulty at rest\n"
    "- Falls or near-falls\n"
    "- Suicidal thoughts\n"
    "- Severe depression interfering with daily life\n"
    "- New neurological symptoms\n"
    "- Severe dehydration risk\n"
    "- No bowel movement > 3 days"
)


def _call_openai(prompt: str, max_tokens: int = 120, temp: float = 0.4,
                 system: str = "") -> str:
    if not openai_client:
        return ""
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return ""


# ── LLM Role 1: Map free-text input to a predefined option ───────
def match_to_option(step: dict, user_input: str) -> str:
    """
    LLM Role 1 — Option Matching.
    Map a patient's free-text answer to the closest predefined option.
    Returns the matched option, or the original input if no match is found.
    """
    if not openai_client:
        return user_input
    options = [o for o in step.get("opts", []) if o.lower() not in {"other", "somewhere else"}]
    if not options:
        return user_input
    prompt = (
        f"Map the patient's answer to one of the listed options.\n\n"
        f"Question: \"{step['text']}\"\n"
        f"Options: {options}\n"
        f"Patient answer: \"{user_input}\"\n\n"
        f"- Return the exact option if the answer clearly matches or implies it.\n"
        f"- For yes/no questions: return \"Yes\" if they describe the symptom, \"No\" if not.\n"
        f"- If the answer is a specific value that doesn't match any named option but "
        f"\"Other\" or \"Somewhere else\" is in the full options list, return that option.\n"
        f"- If no match at all, return the patient's answer exactly.\n"
        f"Return one line only."
    )
    result = _call_openai(prompt, max_tokens=40, temp=0, system=_SYSTEM_CONTEXT)
    return result if result in step.get("opts", []) else user_input


# ── LLM Role 2: Decide if a follow-up question is needed ─────────
def get_followup(topic_key: str, step: dict, answer: str,
                 history: list, next_step: Optional[dict] = None) -> dict:
    """
    LLM Role 2 — Follow-up Decision.
    After a patient's free-text answer, decide the next move:
    continue, send a supportive message, or ask one follow-up question.
    next_step is the next structured question in the flowchart (if any),
    passed so the LLM doesn't ask something the flow will already cover.
    """
    if not openai_client or not answer.strip():
        return {}

    topic_label = _TOPIC_LABELS_LLM.get(topic_key, topic_key)
    # Pass the full topic chat so the LLM can see everything the patient
    # has already said in this topic — not just the last 4 messages.
    full_history = "\n".join(
        f"{'Nurse' if m['role'] == 'assistant' else 'Patient'}: {m['content']}"
        for m in history
    ) or "(start of topic)"
    next_q = f'"{next_step["text"]}"' if next_step else "None — this is the last question in the topic."

    prompt = (
        f"Topic: {topic_label}\n"
        f"Question asked: {step['text']}\n"
        f"Patient answer: {answer}\n"
        f"Full topic conversation so far:\n{full_history}\n\n"
        f"Next structured question in the flowchart: {next_q}\n"
        f"Red flags to watch for:\n{_RED_FLAGS}\n\n"
        f"Decide the best next move. Choose mode=\"follow_up\" for either of these two goals:\n\n"
        f"Goal 1 — Clarify a vague answer:\n"
        f"  Ask if the answer is unclear, too short, or doesn't give useful clinical information.\n"
        f"  Example: patient says 'a lot' to a frequency question → ask 'Roughly how many times a day?'\n\n"
        f"Goal 2 — Gather one more useful clinical detail:\n"
        f"  Ask one short follow-up that would clearly help the care team, based on what the patient said.\n"
        f"  Example: patient says 'I take oxycodone' → ask 'Is it helping control the pain?'\n"
        f"  Example: patient says 'I feel very anxious' → ask 'Is the anxiety affecting your sleep or eating?'\n\n"
        f"Choose mode=\"continue\" when:\n"
        f"  - The answer is already clear and complete.\n"
        f"  - The follow-up you'd ask is already covered by the next structured question shown above.\n\n"
        f"assistant_message: optional 1-2 sentence warm acknowledgment or red flag note. Can accompany either mode.\n\n"
        f"Return JSON only — mode must be exactly \"continue\" or \"follow_up\":\n"
        f'{{"mode":"continue"|"follow_up","assistant_message":"...","follow_up_question":"..."}}'
    )
    parsed = _extract_json_object(
        _call_openai(prompt, max_tokens=200, temp=0.4, system=_SYSTEM_CONTEXT)
    )
    if parsed.get("mode") not in {"continue", "follow_up"}:
        return {}
    if parsed.get("mode") == "follow_up" and not parsed.get("follow_up_question"):
        return {}
    return parsed


# ── Freeform chat response ────────────────────────────────────────
def _freeform_llm_response(messages: list) -> str:
    """Respond to open-ended patient messages in the 'Anything else?' section."""
    if not openai_client:
        return "I've noted that for your care team. Is there anything else?"
    structured_context = {
        key: st.session_state.topic_states[key]["data"]
        for _, key in TOPICS
        if st.session_state.topic_states[key]["data"]
    }
    system = (
        f"{_SYSTEM_CONTEXT}\n\n"
        "The patient may now share anything not covered in the structured check-in.\n\n"
        f"Current check-in data:\n{json.dumps(structured_context, indent=2)}\n\n"
        f"Prior visit data:\n{json.dumps(st.session_state.get('last_checkin', {}), indent=2)}\n\n"
        "- Respond warmly in 2-4 sentences.\n"
        "- If an urgent symptom is mentioned (chest pain, high fever, breathing difficulty, suicidal thoughts), "
        "stay calm and say the care team will be notified.\n"
        "- Do not diagnose or prescribe.\n"
        "- If the patient seems done, ask: 'Is there anything else you'd like to share before your visit?'"
    )
    try:
        r = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": system}] + [
                {"role": m["role"], "content": m["content"]} for m in messages
            ],
            max_tokens=200,
            temperature=0.5,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return "I'm having trouble responding right now — please share this with your care team directly."


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "app_stage":           "login",
        "patient_name":        "",
        "selected_topic":      None,
        "topic_states": {
            key: {"status": "not_started", "data": {}, "chat": []}
            for _, key in TOPICS
        },
        "report":              "",
        "report_saved":        False,
        "last_checkin":        {},    # {topic_key: {q_id: answer}} from last session
        "has_prev_checkin":    False, # True if a previous session was found in Sheets
        "freeform_chat":       [],
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
    Reads the data_json column (last column) which stores the full structured data.
    Returns a dict keyed by topic_key -> {q_id: answer}, or {} if none found.
    """
    _init_sheets()
    if _sheet is None:
        return {}
    try:
        rows = _sheet.get_all_values()
        last_row = None
        for row in rows[1:]:
            if len(row) >= 2 and row[1].strip().lower() == name.strip().lower():
                last_row = row
        if last_row and len(last_row) > _DATA_JSON_COL:
            return json.loads(last_row[_DATA_JSON_COL])
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════════════════════
# TOPIC SUMMARY FORMATTER  (rule-based, no LLM)
# ══════════════════════════════════════════════════════════════════


# ── Sidebar summary: natural-language sentence per topic ─────────

def _topic_summary(topic_key: str, data: dict, patient_facing: bool = False) -> str:
    """
    Return a short summary of a topic's collected answers.
    patient_facing=False  → compact clinical phrase for the sidebar (e.g. "Throat pain, 6/10")
    patient_facing=True   → patient-friendly sentence for the overview screen
    """
    def v(f):
        val = data.get(f)
        return (val or None) if not isinstance(val, list) else (val if val else None)
    def yn(f): return v(f) == "Yes"

    if topic_key == "pain":
        if v("has_pain") == "No":
            return "You had no pain." if patient_facing else "No pain"
        loc = (v("pain_location") or "pain").lower()
        sev = v("throat_severity") or v("tongue_severity")
        if patient_facing:
            return (f"You reported {loc} pain, around {sev}/10 at its worst." if sev
                    else f"You reported pain in the {loc} area.")
        meds = v("pain_medications")
        other_med = v("pain_medications_other_detail")
        med_str = ""
        if isinstance(meds, list) and "No pain medication" not in meds:
            meds = [other_med if item == "Other" and other_med else item for item in meds]
            med_str = f", on {meds[0]}" if len(meds) == 1 else f", on {meds[0]} + {len(meds)-1} more"
        return (f"{loc.capitalize()} pain ({sev}/10){med_str}" if sev
                else f"{loc.capitalize()} pain{med_str}" if loc else f"Pain reported{med_str}")

    if topic_key == "nutrition":
        eating = v("eating_ability") or ""
        weight = v("weight")
        if patient_facing:
            if "normally" in eating: return "You said eating was going okay."
            if "less" in eating:     return "You said you were eating less than usual."
            if "Struggling" in eating: return "You said eating and drinking were hard."
            if "tube" in eating.lower(): return "You said you were using tube feeds."
            return f"Your weight was recorded as {weight} pounds." if weight else "Nutrition noted."
        w_str = f", {weight} lbs" if weight else ""
        if "normally" in eating: return f"Eating normally{w_str}"
        if "less" in eating:     return f"Eating less than usual{w_str}"
        if "Struggling" in eating: return f"Struggling, liquids only{w_str}"
        if "tube" in eating.lower(): return f"On feeding tube{w_str}"
        return f"Nutrition assessed{w_str}"

    if topic_key == "oral":
        syms = [s for s, check in [
            ("mouth sores or thrush", yn("mouth_sores")),
            ("dry mouth",             yn("dry_mouth")),
            ("sticky mucus",          yn("mucus_issues")),
            ("gum problems",          yn("teeth_gum_issues")),
        ] if check]
        if patient_facing:
            return f"You mentioned {', '.join(syms[:2])}." if syms else "No major mouth symptoms."
        return ", ".join(syms).capitalize() if syms else "No oral symptoms"

    if topic_key == "gi":
        nv = v("nausea_vomiting") or []
        syms = [s.lower() for s in ["Nausea", "Vomiting", "Diarrhea"] if s in nv]
        if yn("constipation"): syms.append("constipation")
        if patient_facing:
            return f"You mentioned {', '.join(syms[:2])}." if syms else "No major GI symptoms."
        return ", ".join(syms).capitalize() if syms else "No GI symptoms"

    if topic_key == "fatigue":
        fatigue, sleep = v("fatigue"), v("sleep_quality")
        if patient_facing:
            if fatigue == "Yes" and sleep == "No": return "You felt tired and slept poorly."
            if fatigue == "Yes": return "You felt more tired than usual."
            if sleep == "No":    return "You said sleep was difficult."
            return "You did not report major fatigue or sleep concerns."
        if fatigue == "No" and sleep == "Yes": return "No fatigue, sleeping well"
        if fatigue == "Yes" and sleep == "No": return "Fatigued, trouble sleeping"
        if fatigue == "Yes": return "Feeling fatigued"
        if sleep == "No":    return "Trouble sleeping"
        return "Fatigue assessed"

    if topic_key == "activity":
        level = v("activity_level") or ""
        if patient_facing:
            if "normally" in level:    return "You were doing your usual activities."
            if "less" in level:        return "You were doing less than usual."
            if "Struggling" in level:  return "You said daily tasks were hard."
            return "Activity noted."
        cause = (v("activity_limiting_factor") or "").lower()
        if "normally" in level:   return "Fully active"
        if "less" in level:       return f"Less active — {cause}" if cause else "Less active"
        if "Struggling" in level: return "Struggling with daily tasks"
        return "Activity assessed"

    if topic_key == "mood":
        if patient_facing:
            if v("feeling_down") == "Yes" and v("support_adequate") == "No":
                return "Your mood was low and support felt limited."
            if v("feeling_down") == "Yes":      return "You had been feeling down."
            if v("support_adequate") == "No":   return "You said you needed more support."
            if v("anxiety_impact") == "Yes":    return "Anxiety was affecting daily life."
            return ""
        parts = []
        if v("feeling_down") == "Yes":       parts.append("feeling depressed")
        if v("support_adequate") == "No":    parts.append("limited support")
        elif v("support_adequate") == "Yes": parts.append("good support")
        if v("anxiety_impact") == "Yes":     parts.append("anxiety affecting daily life")
        return ", ".join(parts[:2]).capitalize() if parts else "Mood assessed"

    if topic_key == "other":
        syms = [s for s, check in [
            ("breathing difficulty", yn("breathing_issues")),
            ("hearing changes",      yn("hearing_changes")),
            ("dizziness",            yn("dizziness")),
            ("fever/chills",         yn("fever_chills")),
            ("skin issues",          yn("skin_issues")),
            ("voice changes",        v("voice_hoarseness") == "Yes, problems with my voice"),
        ] if check]
        if patient_facing:
            return f"You mentioned {', '.join(syms[:2])}." if syms else "No other symptoms reported."
        return ", ".join(syms[:3]).capitalize() if syms else "No other symptoms"

    return ""


# Aliases so call sites that use old names still work
def _natural_summary(topic_key, data):        return _topic_summary(topic_key, data, patient_facing=False)
def _patient_overview_summary(topic_key, data): return _topic_summary(topic_key, data, patient_facing=True)


# ══════════════════════════════════════════════════════════════════
# ANSWER PIPELINE
# ══════════════════════════════════════════════════════════════════

def handle_answer(topic_key: str, step: dict, answer, display_override: Optional[str] = None):
    """
    Save the patient's answer, then:
    - If the answer is a matched structured option (from dropdown or Role 1): advance directly.
    - If the answer is an unmatched free-text string (from a free_text question, or typed
      on an option question with no match): call Role 2 to decide the next move.
    """
    state = st.session_state.topic_states[topic_key]
    display = display_override if display_override is not None else (
        ", ".join(answer) if isinstance(answer, list) else str(answer)
    )
    state["chat"].append({"role": "user", "content": display})
    state["data"][step["id"]] = answer
    state["status"] = "in_progress"

    # Role 2: called whenever the answer is a free-text string that was NOT matched
    # to a predefined option — covers both free_text questions and unmatched typed
    # answers on option questions.
    # next_step is computed first so Role 2 knows what the flowchart will ask next
    # and can avoid asking a redundant follow-up.
    next_step = get_next_step(topic_key, state["data"])
    is_unmatched_text = isinstance(answer, str) and answer not in step.get("opts", [])
    if is_unmatched_text and openai_client:
        turn = get_followup(topic_key, step, answer, state["chat"], next_step=next_step)
        if turn.get("mode") == "follow_up":
            state["waiting_for_followup"] = True
            state["pending_followup"] = {
                "step_id": step["id"],
                "question": turn["follow_up_question"],
            }
            msg_parts = [turn.get("assistant_message", ""), turn["follow_up_question"]]
            state["chat"].append({"role": "assistant", "content": "\n\n".join(p for p in msg_parts if p)})
            st.rerun()
            return
        if turn.get("assistant_message"):
            state["chat"].append({"role": "assistant", "content": turn["assistant_message"]})

    # Advance to next question or mark topic complete
    if topic_is_complete(topic_key, state["data"]):
        state["status"] = "completed"
        state["chat"].append({"role": "assistant", "content": "✅ Thank you — I have everything I need for this topic."})
    elif next_step:
        _append_assistant_message(state, _step_prompt_text(next_step))
    st.rerun()


def handle_pending_followup(topic_key: str, answer: str):
    """Save the follow-up answer and advance to the next structured question."""
    state = st.session_state.topic_states[topic_key]
    pending = state.get("pending_followup", {})

    state["chat"].append({"role": "user", "content": answer})
    state["data"][f"{pending.get('step_id', 'followup')}_followup"] = answer
    state["waiting_for_followup"] = False
    state.pop("pending_followup", None)

    next_step = get_next_step(topic_key, state["data"])
    if topic_is_complete(topic_key, state["data"]):
        state["status"] = "completed"
        state["chat"].append({"role": "assistant", "content": "\u2705 Thank you — I have everything I need for this topic."})
    elif next_step:
        _append_assistant_message(state, _step_prompt_text(next_step))
    st.rerun()


# ══════════════════════════════════════════════════════════════════
# INPUT RENDERING
# ══════════════════════════════════════════════════════════════════

def render_input(topic_key: str, step: dict):
    """
    Render the input widget for the current question.
    Supports four question types: options, multi_select, number, free_text.
    Each type accepts typed text and voice input.
    """
    stype = step["type"]
    sid   = step["id"]
    state = st.session_state.topic_states[topic_key]

    _, col = st.columns([1.05, 0.95])

    # ── Options: dropdown + free-text typed/voice ────────────────
    if stype == "options":
        with col:
            st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
            col_text, col_drop = st.columns([4.6, 1.7])
            with col_text:
                typed = st.text_input("", key=f"txt_{topic_key}_{sid}",
                                      label_visibility="collapsed", placeholder="Type a reply...")
            with col_drop:
                chosen = st.selectbox("", ["Select..."] + step["opts"],
                                      key=f"sel_{topic_key}_{sid}", label_visibility="collapsed")
            voice = voice_widget(f"{topic_key}_{sid}_opt")

            # Dropdown selection
            if chosen != "Select..." and st.session_state.get(f"sel_{topic_key}_{sid}_done") != chosen:
                st.session_state[f"sel_{topic_key}_{sid}_done"] = chosen
                handle_answer(topic_key, step, chosen)

            # Typed or voice: use LLM Role 1 to map to option
            raw = typed or voice or ""
            if raw and st.session_state.get(f"txt_{topic_key}_{sid}_done") != raw:
                st.session_state[f"txt_{topic_key}_{sid}_done"] = raw
                matched = match_to_option(step, raw)
                if matched in step.get("opts", []):
                    # Role 1 matched to a structured option.
                    # Special case: "Somewhere else" / "Other" means the patient already
                    # answered the next free-text question with their typed text — pre-fill it.
                    if matched.lower() in {"somewhere else", "other"}:
                        temp_data = {**state["data"], sid: matched}
                        next_st = get_next_step(topic_key, temp_data)
                        if next_st and next_st.get("type") == "free_text":
                            state["data"][next_st["id"]] = raw
                        handle_answer(topic_key, step, matched, display_override=raw)
                    else:
                        handle_answer(topic_key, step, matched)
                else:
                    # Role 1 found no match — pass raw text to handle_answer.
                    # Role 2 inside handle_answer will decide the next move.
                    handle_answer(topic_key, step, raw)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Multi-select: dropdown + free-text typed/voice ───────────
    elif stype == "multi_select":
        with col:
            st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
            col_text, col_drop = st.columns([4.6, 1.7])
            with col_text:
                typed = st.text_input("", key=f"txt_{topic_key}_{sid}",
                                      label_visibility="collapsed",
                                      placeholder="Type one or more, separated by commas...")
            with col_drop:
                chosen = st.selectbox("", ["Select..."] + step["opts"],
                                      key=f"sel_{topic_key}_{sid}", label_visibility="collapsed")
            voice = voice_widget(f"{topic_key}_{sid}_multi")

            if chosen != "Select..." and st.session_state.get(f"sel_{topic_key}_{sid}_done") != chosen:
                st.session_state[f"sel_{topic_key}_{sid}_done"] = chosen
                handle_answer(topic_key, step, [chosen])

            raw = typed or voice or ""
            if raw and st.session_state.get(f"txt_{topic_key}_{sid}_done") != raw:
                st.session_state[f"txt_{topic_key}_{sid}_done"] = raw
                # Parse comma-separated input, match each part to an option via LLM
                parts = [p.strip() for p in re.split(r"[,;/\n]", raw) if p.strip()]
                matched = list(dict.fromkeys(
                    m for p in parts
                    for m in [match_to_option(step, p)]
                    if m in step.get("opts", [])
                ))
                if matched:
                    handle_answer(topic_key, step, matched)
                elif "Other" in step.get("opts", []):
                    state["data"][f"{sid}_other_detail"] = raw
                    handle_answer(topic_key, step, ["Other"], display_override=raw)
                else:
                    # Nothing matched at all — pass raw text to handle_answer.
                    # Role 2 will decide the next move.
                    handle_answer(topic_key, step, raw)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Number: text input with range validation ─────────────────
    elif stype == "number":
        with col:
            st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
            typed = st.text_input("", key=f"txt_{topic_key}_{sid}",
                                  label_visibility="collapsed",
                                  placeholder=f"Enter a number ({int(step['min_v'])}–{int(step['max_v'])})")
            voice = voice_widget(f"{topic_key}_{sid}_num")
            raw = typed or voice or ""
            if raw and st.session_state.get(f"txt_{topic_key}_{sid}_done") != raw:
                st.session_state[f"txt_{topic_key}_{sid}_done"] = raw
                try:
                    val = int(float(raw))
                    if step["min_v"] <= val <= step["max_v"]:
                        handle_answer(topic_key, step, val)
                    else:
                        st.warning(f"Please enter a number between {int(step['min_v'])} and {int(step['max_v'])}.")
                except ValueError:
                    st.warning("Please enter a number.")
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Free text: text input + voice ────────────────────────────
    elif stype == "free_text":
        with col:
            st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
            widget_key = f"ft_{topic_key}_{sid}"
            voice = voice_widget(f"{topic_key}_{sid}")
            if voice and voice != st.session_state.get(f"{widget_key}_vsync"):
                st.session_state[widget_key] = voice
                st.session_state[f"{widget_key}_vsync"] = voice
            typed = st.text_input("", placeholder=step.get("placeholder", "Please describe…"),
                                  key=widget_key, label_visibility="collapsed")
            if typed and st.session_state.get(f"{widget_key}_done") != typed:
                st.session_state[f"{widget_key}_done"] = typed
                handle_answer(topic_key, step, typed)
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
    st.subheader("Anything else you’d like to share?")
    st.caption(
        "Mention any other symptoms, questions, or concerns you’d like your care team "
        "to know about before your visit."
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

    # ── Previous check-in summary card ────────────────────────────
    if has_prev:
        if last_data:
            prev_text = format_topic_data(topic_key, last_data)
            if prev_text:
                with st.expander("Last visit summary", expanded=False):
                    st.text(prev_text)
        else:
            st.caption("No information from your last visit was recorded for this section.")

    # ── Initialize topic on first visit ─────────────────────────
    if state["status"] == "not_started":
        state["status"] = "in_progress"
        intro = TOPIC_INTROS.get(topic_key, "Let's go through this section together.")
        state["chat"] = [{"role": "assistant", "content": intro}]

    # ── Header with progress bar ─────────────────────────────────
    answered, applicable = get_topic_progress(topic_key, state["data"])
    col_title, col_prog = st.columns([3, 1])
    with col_title:
        st.subheader(topic_label)
    with col_prog:
        if applicable > 0:
            st.caption(f"{answered}/{applicable} answered")

    next_step = get_next_step(topic_key, state["data"])
    if not state.get("waiting_for_followup") and next_step:
        _append_assistant_message(state, _step_prompt_text(next_step))

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
        return

    # ── Current question ─────────────────────────────────────────
    if state.get("waiting_for_followup"):
        pending = state.get("pending_followup") or {}
        pending_key = f"fu_{topic_key}_{pending.get('step_id', 'x')}"
        _, composer_col = st.columns([1.05, 0.95])
        with composer_col:
            st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
            fu_text = st.text_input("", key=pending_key,
                                    placeholder="Type your answer...",
                                    label_visibility="collapsed")
            fu_voice = voice_widget(f"fupd_{topic_key}_{pending.get('step_id', 'x')}")
            raw = fu_text or fu_voice or ""
            if raw and st.session_state.get(f"{pending_key}_done") != raw:
                st.session_state[f"{pending_key}_done"] = raw
                handle_pending_followup(topic_key, raw)
            st.markdown('</div>', unsafe_allow_html=True)
        return

    if next_step:
        render_input(topic_key, next_step)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR  (MASTER PANEL)
# ══════════════════════════════════════════════════════════════════


def render_sidebar():
    with st.sidebar:
        # ── Header ───────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:18px;font-weight:800;color:#10233d;margin:0 0 2px 0;letter-spacing:-0.03em;">🩺 ChatReport</div>'
            '<div style="font-size:11px;color:#6b7b92;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.08em;">Clinical symptom assistant</div>',
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
                all_data = {k: st.session_state.topic_states[k]["data"]
                            for _, k in TOPICS}
                ff_msgs = [m for m in st.session_state.freeform_chat if m["role"] == "user"]
                if ff_msgs:
                    all_data["freeform_notes"] = [m["content"] for m in ff_msgs]
                with st.spinner("Saving…"):
                    save_to_sheet(st.session_state.patient_name, all_data)
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
            summary = _patient_overview_summary(key, prev_data)
            if not summary:
                continue
            rows.append(
                "<tr>"
                f'<td><div class="overview-topic-name">{_html.escape(topic_name)}</div></td>'
                f'<td><div class="overview-summary-main">{_html.escape(summary)}</div></td>'
                "</tr>"
            )

        if rows:
            st.markdown(
                '<div class="overview-table-wrap">'
                '<table class="overview-table">'
                '<colgroup>'
                '<col class="topic-col">'
                '<col class="summary-col">'
                '</colgroup>'
                '<thead><tr><th>Topic</th><th>Last Visit Summary</th></tr></thead>'
                f'<tbody>{"".join(rows)}</tbody>'
                '</table>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="overview-note">'
                'This is a short summary from your last visit. You can update anything as you go through today\'s topics.'
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

    name = st.session_state.patient_name
    date = datetime.now().strftime("%B %d, %Y")

    st.markdown(
        f'<div class="card">'
        f'<div style="font-size:12px;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;color:#6b7b92;margin-bottom:8px;">Check-in complete</div>'
        f'<div style="font-size:28px;font-weight:800;letter-spacing:-0.03em;color:#10233d;">✅ Saved to Google Sheets</div>'
        f'<div style="font-size:14px;color:#56667d;margin-top:8px;">Patient: <strong>{_html.escape(name)}</strong> &nbsp;|&nbsp; Date: {date}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Show a per-topic summary table built entirely by code
    all_data = {key: st.session_state.topic_states[key]["data"] for _, key in TOPICS}
    ff_msgs  = [m["content"] for m in st.session_state.freeform_chat if m["role"] == "user"]

    for label, key in TOPICS:
        data = all_data.get(key, {})
        if not data:
            continue
        topic_name = label.split(" ", 1)[1] if " " in label else label
        formatted  = format_topic_data(key, data)
        if not formatted:
            continue
        with st.expander(f"**{topic_name}**", expanded=False):
            st.text(formatted)

    if ff_msgs:
        with st.expander("**Freeform Notes**", expanded=False):
            for msg in ff_msgs:
                st.markdown(f"- {msg}")

    st.markdown("---")
    if st.button("⬅️ Back to Check-In"):
        st.session_state.app_stage = "main"
        st.rerun()


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
