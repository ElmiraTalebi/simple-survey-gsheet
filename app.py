from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import gspread
import streamlit as st
from google.oauth2.service_account import Credentials
from openai import OpenAI
from streamlit.errors import StreamlitSecretNotFoundError


# --------------------------
# UI
# --------------------------
st.set_page_config(page_title="Cancer Symptom Check-In", layout="centered")
st.markdown(
    """
<style>
.block-container { padding-top: 2rem; max-width: 920px; }
.stButton > button { width: 100%; border-radius: 10px; }
.risk-high { padding: 10px; border-radius: 8px; border: 1px solid #ffb3b3; background: #fff3f3; margin-bottom: 8px; }
.risk-emergency { padding: 10px; border-radius: 8px; border: 1px solid #ff8080; background: #ffe5e5; margin-bottom: 8px; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------
# Data model
# --------------------------
@dataclass
class Question:
    id: int
    prompt: str
    options: List[str] = field(default_factory=list)
    followups_if_yes: List[str] = field(default_factory=list)
    followups_if_no: List[str] = field(default_factory=list)
    followups_if_contains: Dict[str, List[str]] = field(default_factory=dict)
    followups_for_any_answer: List[str] = field(default_factory=list)


QUESTIONS: List[Question] = [
    Question(1, "How has your overall feeling been since your last visit? Please rate from 0 to 10.", [str(i) for i in range(11)],
             followups_for_any_answer=["If score is 5 or above, what is contributing most to feeling worse today?"]),
    Question(2, "Do you have any pain today?", ["Yes", "No"], followups_if_yes=["Where exactly is the pain? (Throat, tongue, somewhere else)"]),
    Question(3, "Where exactly is the pain?", ["Throat", "Tongue", "Somewhere else", "No pain"],
             followups_if_contains={
                 "throat": [
                     "Is the throat pain all the time, only when swallowing, only when eating, or both?",
                     "On a scale of 0 to 10, how bad is it at worst?",
                     "Are you taking pain medication for this? Is it helping?",
                 ],
                 "tongue": [
                     "Is there a sore/ulcer on the tongue, or general painful feeling?",
                     "On a scale of 0 to 10, how bad is tongue pain at worst?",
                     "Is it one spot or spread across the tongue?",
                 ],
                 "somewhere else": [
                     "Please describe where you feel pain.",
                     "Any ear pain or hearing changes?",
                     "Any swelling near your jaw?",
                     "Does pain worsen when chewing or opening your mouth?",
                     "When did this pain start?",
                 ],
             }),
    Question(4, "Do you have any mouth sores or ulcers right now?", ["Yes", "No"],
             followups_if_yes=[
                 "Is this sore new since your last visit, or same as before?",
                 "Where exactly is it located (inside mouth/cheek, tongue, throat, gums/lips, multiple spots)?",
                 "Is the sore painful and affecting ability to eat/drink?",
                 "Are you using magic mouthwash? If yes, is it helping?",
                 "Compared with last visit: getting better, same, or worse?",
             ]),
    Question(5, "How has your eating been since your last visit? Are you able to eat and drink enough?",
             ["Eating normally", "Eating less but managing", "Struggling, mostly liquids", "Tube feeding only"],
             followups_for_any_answer=[
                 "Are you drinking enough fluids throughout the day?",
                 "What are you able to eat right now?",
                 "How many nutritional shakes (Boost/Ensure) per day?",
                 "What is making eating/drinking hard (pain, nausea, dry mouth, appetite, fatigue, other)?",
                 "Are you timing pain medication before meals to make eating easier?",
             ]),
    Question(6, "What has your weight at home been recently (lbs)?",
             followups_for_any_answer=["Has any weight change affected your energy or how you feel?"]),
    Question(7, "Are you experiencing dryness in your mouth?", ["Yes", "No"],
             followups_if_yes=["Is dryness worse at night or all day?", "Are you using Biotene or saliva substitute?", "Is dryness making eating, talking, or sleeping harder?"]),
    Question(8, "Are you having difficulty swallowing liquids, food, or pills?", ["Yes", "No"],
             followups_if_yes=["Is it painful to swallow or mechanically difficult?", "Do you cough or choke when you eat?", "Are you still swallowing liquids by mouth or using feeding tube only?"]),
    Question(9, "Are you having breathing difficulty or shortness of breath?", ["Yes", "No"],
             followups_if_yes=["Is it constant or with activity?", "Any wheezing or airway blockage feeling?"]),
    Question(10, "Are you having problems with mucus or thick secretions in your throat?", ["Yes", "No"],
             followups_if_yes=["Is mucus thick/hard to clear or watery?", "Is it affecting swallowing or sleep?", "Are you using anything to manage it (saline, Robitussin)?"]),
    Question(11, "Have you had nausea, vomiting, or blood when coughing?", ["Nausea", "Vomiting", "Blood when coughing", "None"]),
    Question(12, "Which medications are you currently taking for pain and symptoms?", ["Gabapentin", "Oxycodone", "Butrans patch", "Other", "No pain medication"],
             followups_for_any_answer=["How often are you taking each medication and at what dose?", "Do pain medications make you drowsy?"]),
    Question(13, "Are you feeling more tired or weak than usual?", ["Yes", "No"],
             followups_if_yes=["Is this general tiredness or weakness in specific body parts?", "If specific weakness, which body part(s)?", "Is fatigue affecting daily activities?"]),
    Question(14, "Are you able to sleep through the night?", ["Yes", "No"],
             followups_if_no=["Are you waking due to pain, dry mouth, or coughing?", "Is medication drowsiness affecting sleep/wake schedule?"]),
    Question(15, "How are you feeling emotionally? Are you anxious or worried about anything?",
             followups_for_any_answer=["Is anxiety affecting sleep, eating, or daily activities?", "Do you have people around you to talk to?"]),
    Question(16, "Any hearing problems or changes recently?", ["Yes", "No"],
             followups_if_yes=["Is it ringing, hearing loss, or both?", "Is it constant or comes/goes?", "Worse since last visit?"]),
    Question(17, "Have you been feeling dizzy or lightheaded?", ["Yes", "No"],
             followups_if_yes=["Is it constant or when standing/changing position?", "Has it worsened recently?", "Any falls or near-falls?"]),
    Question(18, "Have you had constipation or trouble moving bowels?", ["Yes", "No"],
             followups_if_yes=["How often are bowel movements?", "Are you taking Senna, Miralax, or other meds?", "Any bloating or discomfort?"]),
    Question(19, "Any numbness or tingling in hands or feet?", ["Yes", "No"], followups_if_yes=["Is it new, worse, or same?"]),
    Question(20, "Have you had fever or chills recently?", ["Yes", "No"], followups_if_yes=["When did it start?"]),
    Question(21, "Are you checking blood pressure at home?", ["Yes", "No"], followups_if_yes=["What has it been recently?"]),
    Question(22, "Any skin changes like irritation, wounds, or redness?", ["Yes", "No"],
             followups_if_yes=["Where is it located?", "Any drainage, bleeding, or open areas?"]),
    Question(23, "Any voice changes or hoarseness?", ["Yes", "No"], followups_if_yes=["Is it constant or only when talking?"]),
    Question(24, "Any problems with your teeth or gums?", ["Yes", "No"],
             followups_if_yes=["Is there pain, bleeding, sores, or multiple issues?", "Is brushing difficult?", "Are you avoiding brushing due to discomfort?"]),
    Question(25, "Are you currently receiving IV fluids or hydration treatments?", ["Yes", "No"]),
    Question(26, "How is your daily life? Are you able to do usual activities?",
             ["Doing everything normally", "Doing less than usual", "Struggling with daily tasks"],
             followups_for_any_answer=["If limited, is it mainly due to pain, fatigue, or something else?"]),
    Question(27, "Are you using mouthwash or oral rinses regularly?", ["Yes", "No"], followups_if_yes=["Is it helping?"]),
    Question(28, "Any changes in your sense of taste?", ["Yes", "No"]),
    Question(29, "Any trouble concentrating or remembering things?", ["Yes", "No"]),
    Question(30, "Any sexual health concerns or changes?", ["Yes", "No"]),
    Question(31, "Are you taking medications as prescribed?", ["Yes", "No"]),
    Question(32, "Do you feel you have enough support between visits?", ["Yes", "No"]),
    Question(33, "Have you been feeling down or depressed?", ["Yes", "No"]),
]


# --------------------------
# Secrets and integrations
# --------------------------
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
        out = []
        for row in rows[1:]:
            if len(row) >= 3 and row[1].strip().lower() == name.strip().lower():
                try:
                    payload = json.loads(row[2])
                    payload["timestamp"] = row[0]
                    out.append(payload)
                except Exception:
                    pass
        return out[-5:]
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


# --------------------------
# LLM helper
# --------------------------
@dataclass
class LLMGuidance:
    supportive_reply: str
    risk_level: str
    suggested_followup: Optional[str]


SYSTEM_PROMPT = """
You are an oncology symptom triage assistant helping with patient check-in.
Return strict JSON with keys:
- supportive_reply: brief supportive sentence
- risk_level: one of low, moderate, high, emergency
- suggested_followup: one concise follow-up question or null
Mark emergency for severe breathing difficulty, coughing blood, inability to swallow liquids, confusion, or chest pain.
""".strip()


def get_llm_guidance(patient_text: str, candidate_followups: List[str]) -> Optional[LLMGuidance]:
    if not OPENAI_API_KEY:
        return None
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        payload = {
            "patient_text": patient_text,
            "candidate_followups": candidate_followups[:10],
        }
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            text={"format": {"type": "json_object"}},
        )
        data = json.loads((resp.output_text or "{}").strip() or "{}")
        return LLMGuidance(
            supportive_reply=data.get("supportive_reply", "Thank you for sharing that."),
            risk_level=str(data.get("risk_level", "low")).lower(),
            suggested_followup=data.get("suggested_followup"),
        )
    except Exception:
        return None


# --------------------------
# Flow helpers
# --------------------------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def is_yes(answer: str) -> bool:
    return normalize_text(answer) in {"yes", "y", "yeah", "yep", "true"}


def is_no(answer: str) -> bool:
    return normalize_text(answer) in {"no", "n", "nope", "false"}


def option_match(answer: str, options: List[str]) -> bool:
    a = normalize_text(answer)
    return any(a == normalize_text(o) for o in options)


def pick_keyword_followups(question: Question, answer: str) -> List[str]:
    lower = normalize_text(answer)
    out: List[str] = []
    for keyword, fqs in question.followups_if_contains.items():
        if keyword in lower:
            out.extend(fqs)
    return out


def build_followups(question: Question, answer: str) -> List[str]:
    out: List[str] = []
    out.extend(question.followups_for_any_answer)
    if is_yes(answer):
        out.extend(question.followups_if_yes)
    if is_no(answer):
        out.extend(question.followups_if_no)
    out.extend(pick_keyword_followups(question, answer))
    dedup = []
    seen = set()
    for x in out:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup


# --------------------------
# Session state
# --------------------------
def init_state():
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


def append_chat(role: str, content: str):
    st.session_state.chat_log.append({"role": role, "content": content})


def current_main_question() -> Optional[Question]:
    i = st.session_state.q_index
    if i >= len(QUESTIONS):
        return None
    return QUESTIONS[i]


def ask_next_prompt():
    if st.session_state.followup_queue:
        append_chat("assistant", st.session_state.followup_queue[0])
        return

    q = current_main_question()
    if q is None:
        append_chat("assistant", "Check-in complete. Please review and submit below.")
        return

    msg = f"Main {q.id}: {q.prompt}"
    if q.options:
        msg += "\n\nOptions: " + " | ".join(q.options)
    append_chat("assistant", msg)


def add_risk_flag(g: LLMGuidance):
    level = (g.risk_level or "").lower().strip()
    if level in {"high", "emergency"}:
        st.session_state.risk_flags.append({
            "risk": level,
            "message": g.supportive_reply,
            "time": datetime.now().strftime("%H:%M:%S"),
        })


def apply_llm_support(user_text: str, candidates: List[str]) -> List[str]:
    g = get_llm_guidance(user_text, candidates)
    if not g:
        return candidates
    append_chat("assistant", g.supportive_reply)
    add_risk_flag(g)
    if g.suggested_followup and g.suggested_followup not in candidates:
        return [g.suggested_followup] + candidates
    return candidates


def handle_user_input(user_text: str):
    append_chat("user", user_text)

    if st.session_state.followup_queue:
        st.session_state.followup_queue.pop(0)
        rem = st.session_state.followup_queue
        if len(user_text.split()) > 6 or not option_match(user_text, ["yes", "no"]):
            st.session_state.followup_queue = apply_llm_support(user_text, rem)
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


def build_payload() -> Dict:
    ans = {}
    for q in QUESTIONS:
        if q.id in st.session_state.answers:
            ans[f"Q{q.id}"] = {"question": q.prompt, "answer": st.session_state.answers[q.id]}
    return {
        "name": st.session_state.name,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "question_count": len(ans),
        "risk_flags": st.session_state.risk_flags,
        "answers": ans,
    }


# --------------------------
# App
# --------------------------
def main():
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

    if st.session_state.past_checkins:
        with st.expander("Recent check-ins"):
            for x in reversed(st.session_state.past_checkins):
                st.markdown(f"- {x.get('timestamp', 'Unknown time')}")

    if st.session_state.risk_flags:
        st.markdown("### Risk flags")
        for f in st.session_state.risk_flags:
            cls = "risk-emergency" if f["risk"] == "emergency" else "risk-high"
            st.markdown(
                f"<div class='{cls}'><b>{f['risk'].upper()}</b> at {f['time']}: {f['message']}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Chat")
    for m in st.session_state.chat_log:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    text = st.chat_input("Type your answer")
    if text:
        handle_user_input(text)
        st.rerun()

    if st.session_state.q_index >= len(QUESTIONS):
        payload = build_payload()
        summary = ["# Patient Check-In Summary", f"Patient: {st.session_state.name}", ""]
        for k, v in payload["answers"].items():
            summary.append(f"- {k}: {v['question']}")
            summary.append(f"  - Answer: {v['answer']}")

        st.download_button(
            "Download summary (.md)",
            data="\n".join(summary),
            file_name=f"checkin_{st.session_state.name.replace(' ', '_') or 'patient'}.md",
            mime="text/markdown",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Submit to Google Sheet", disabled=st.session_state.submitted):
                try:
                    save_to_sheet(payload)
                    st.session_state.submitted = True
                    st.success("Submitted successfully.")
                except Exception as e:
                    st.error(f"Submit failed: {e}")

        with c2:
            if st.button("Start new check-in"):
                for k in ["answers", "q_index", "followup_queue", "chat_log", "risk_flags", "submitted"]:
                    if k in st.session_state:
                        del st.session_state[k]
                init_state()
                append_chat("assistant", "Welcome back. I will ask your check-in questions one at a time.")
                ask_next_prompt()
                st.rerun()


if __name__ == "__main__":
    main()
