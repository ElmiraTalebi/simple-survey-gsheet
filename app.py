from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Dict, List, Optional

import gspread
import streamlit as st
from google.oauth2.service_account import Credentials
from streamlit.errors import StreamlitSecretNotFoundError

from llm_helper import LLMGuidance, get_llm_guidance
from questionnaire import QUESTIONS, Question

st.set_page_config(page_title="Cancer Symptom Check-In", layout="centered")

st.markdown(
    """
<style>
.block-container {
    padding-top: 2rem;
    max-width: 920px;
}
.card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e6e9f2;
    background: white;
    margin-bottom: 12px;
}
.stButton > button {
    width: 100%;
    border-radius: 10px;
}
.risk-high {
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #ffb3b3;
    background: #fff3f3;
}
.risk-emergency {
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #ff8080;
    background: #ffe5e5;
}
</style>
""",
    unsafe_allow_html=True,
)


def _secret(*keys: str, default=None):
    try:
        for k in keys:
            if k in st.secrets:
                return st.secrets[k]
    except StreamlitSecretNotFoundError:
        return default
    return default


def _require_secret(*keys: str):
    v = _secret(*keys)
    if v is None:
        raise KeyError(f"Missing secret. Tried: {', '.join(keys)}")
    return v


OPENAI_API_KEY = _secret("openai_api_key", "OPENAI_API_KEY", "openai_key")
OPENAI_MODEL = _secret("openai_model", "OPENAI_MODEL", default="gpt-5-mini")

sheet = None
sheets_init_error: Optional[str] = None


def _init_sheets() -> None:
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
            ws = book.add_worksheet(title="Form", rows=3000, cols=20)
            ws.append_row(["timestamp", "name", "json"])
        sheet = ws
    except Exception as e:
        sheets_init_error = str(e)


def load_past_checkins(name: str) -> List[Dict]:
    _init_sheets()
    if sheet is None or not name.strip():
        return []
    try:
        rows = sheet.get_all_values()
        past: List[Dict] = []
        for row in rows[1:]:
            if len(row) < 3:
                continue
            if row[1].strip().lower() != name.strip().lower():
                continue
            try:
                payload = json.loads(row[2])
                payload["timestamp"] = row[0]
                past.append(payload)
            except Exception:
                continue
        return past[-5:]
    except Exception:
        return []


def save_to_sheet(payload: Dict) -> None:
    _init_sheets()
    if sheet is None:
        raise RuntimeError(f"Sheets unavailable: {sheets_init_error}")
    sheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        payload.get("name", "Unknown"),
        json.dumps(payload),
    ])


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def is_yes(answer: str) -> bool:
    return normalize_text(answer) in {"yes", "y", "yeah", "yep", "true"}


def is_no(answer: str) -> bool:
    return normalize_text(answer) in {"no", "n", "nope", "false"}


def option_match(answer: str, options: List[str]) -> bool:
    a = normalize_text(answer)
    return any(a == normalize_text(opt) for opt in options)


def pick_keyword_followups(question: Question, answer: str) -> List[str]:
    lower = normalize_text(answer)
    out: List[str] = []
    for keyword, followups in question.followups_if_contains.items():
        if keyword in lower:
            out.extend(followups)
    return out


def build_followups(question: Question, answer: str) -> List[str]:
    out: List[str] = []
    out.extend(question.followups_for_any_answer)
    if is_yes(answer):
        out.extend(question.followups_if_yes)
    if is_no(answer):
        out.extend(question.followups_if_no)
    out.extend(pick_keyword_followups(question, answer))

    deduped: List[str] = []
    seen = set()
    for item in out:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def init_state() -> None:
    defaults = {
        "name": "",
        "name_confirmed": False,
        "past_checkins": [],
        "answers": {},
        "q_index": 0,
        "followup_queue": [],
        "chat_log": [],
        "risk_flags": [],
        "submitted": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def append_chat(role: str, content: str) -> None:
    st.session_state.chat_log.append({"role": role, "content": content})


def current_main_question() -> Optional[Question]:
    idx = st.session_state.q_index
    if idx >= len(QUESTIONS):
        return None
    return QUESTIONS[idx]


def ask_next_prompt() -> None:
    if st.session_state.followup_queue:
        append_chat("assistant", st.session_state.followup_queue[0])
        return

    q = current_main_question()
    if q is None:
        append_chat("assistant", "Check-in complete. Please review and submit below.")
        return

    prompt = f"Main {q.id}: {q.prompt}"
    if q.options:
        prompt += "\n\nOptions: " + " | ".join(q.options)
    append_chat("assistant", prompt)


def add_risk_flag(guidance: LLMGuidance) -> None:
    level = guidance.risk_level.lower().strip()
    if level not in {"low", "moderate", "high", "emergency"}:
        return
    if level in {"high", "emergency"}:
        st.session_state.risk_flags.append(
            {
                "risk": level,
                "message": guidance.supportive_reply,
                "time": datetime.now().strftime("%H:%M:%S"),
            }
        )


def apply_llm_support(user_text: str, candidates: List[str]) -> List[str]:
    guidance = get_llm_guidance(
        patient_text=user_text,
        candidate_followups=candidates,
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
    )
    if not guidance:
        return candidates

    append_chat("assistant", guidance.supportive_reply)
    add_risk_flag(guidance)
    if guidance.suggested_followup and guidance.suggested_followup not in candidates:
        return [guidance.suggested_followup] + candidates
    return candidates


def handle_user_input(user_text: str) -> None:
    append_chat("user", user_text)

    if st.session_state.followup_queue:
        st.session_state.followup_queue.pop(0)
        remaining = st.session_state.followup_queue
        if not option_match(user_text, ["yes", "no"]) or len(user_text.split()) > 6:
            st.session_state.followup_queue = apply_llm_support(user_text, remaining)
        ask_next_prompt()
        return

    q = current_main_question()
    if q is None:
        return

    st.session_state.answers[q.id] = user_text
    followups = build_followups(q, user_text)

    free_text = not option_match(user_text, q.options) if q.options else True
    if free_text or len(user_text.split()) > 8 or "?" in user_text:
        followups = apply_llm_support(user_text, followups)

    st.session_state.followup_queue.extend(followups)
    st.session_state.q_index += 1
    ask_next_prompt()


def render_past_checkins() -> None:
    if not st.session_state.past_checkins:
        return

    with st.expander("Recent check-ins", expanded=False):
        for item in reversed(st.session_state.past_checkins):
            when = item.get("timestamp", "Unknown time")
            st.markdown(f"- {when}")


def render_risk_flags() -> None:
    if not st.session_state.risk_flags:
        return

    st.markdown("### Risk flags")
    for flag in st.session_state.risk_flags:
        css = "risk-emergency" if flag["risk"] == "emergency" else "risk-high"
        st.markdown(
            f"<div class='{css}'><b>{flag['risk'].upper()}</b> at {flag['time']}: {flag['message']}</div>",
            unsafe_allow_html=True,
        )


def build_summary_payload() -> Dict:
    summary_answers = {}
    for q in QUESTIONS:
        if q.id in st.session_state.answers:
            summary_answers[f"Q{q.id}"] = {
                "question": q.prompt,
                "answer": st.session_state.answers[q.id],
            }

    return {
        "name": st.session_state.name,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "question_count": len(summary_answers),
        "risk_flags": st.session_state.risk_flags,
        "answers": summary_answers,
    }


def render_summary_actions() -> None:
    if st.session_state.q_index < len(QUESTIONS):
        return

    payload = build_summary_payload()
    summary_md = ["# Patient Check-In Summary", f"Patient: {st.session_state.name}", ""]
    for key, item in payload["answers"].items():
        summary_md.append(f"- {key}: {item['question']}")
        summary_md.append(f"  - Answer: {item['answer']}")

    st.download_button(
        "Download summary (.md)",
        data="\n".join(summary_md),
        file_name=f"checkin_{st.session_state.name.replace(' ', '_') or 'patient'}.md",
        mime="text/markdown",
    )

    submit_col, reset_col = st.columns(2)
    with submit_col:
        if st.button("Submit to Google Sheet", disabled=st.session_state.submitted):
            try:
                save_to_sheet(payload)
                st.session_state.submitted = True
                st.success("Submitted successfully.")
            except Exception as e:
                st.error(f"Submit failed: {e}")

    with reset_col:
        if st.button("Start new check-in"):
            for k in [
                "answers",
                "q_index",
                "followup_queue",
                "chat_log",
                "risk_flags",
                "submitted",
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            init_state()
            append_chat("assistant", "Welcome back. I will ask your check-in questions one at a time.")
            ask_next_prompt()
            st.rerun()


def main() -> None:
    init_state()

    st.title("Cancer Symptom Check-In")

    if not OPENAI_API_KEY:
        st.info("OpenAI key not found in secrets. Free-text LLM assistance is disabled.")

    with st.container(border=True):
        name = st.text_input("Patient name", value=st.session_state.name, placeholder="Enter full name")
        if st.button("Load patient and start"):
            st.session_state.name = name.strip()
            st.session_state.name_confirmed = bool(st.session_state.name)
            st.session_state.past_checkins = load_past_checkins(st.session_state.name)
            st.session_state.answers = {}
            st.session_state.q_index = 0
            st.session_state.followup_queue = []
            st.session_state.chat_log = []
            st.session_state.risk_flags = []
            st.session_state.submitted = False
            append_chat("assistant", f"Welcome {st.session_state.name or 'patient'}. I will ask your check-in questions one at a time.")
            ask_next_prompt()
            st.rerun()

    if not st.session_state.name_confirmed:
        st.stop()

    render_past_checkins()
    render_risk_flags()

    st.markdown("### Chat")
    for message in st.session_state.chat_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_text = st.chat_input("Type your answer")
    if user_text:
        handle_user_input(user_text)
        st.rerun()

    render_summary_actions()


if __name__ == "__main__":
    main()
