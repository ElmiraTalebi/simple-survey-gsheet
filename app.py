# Full Cancer Symptom Check-In App with Trend Detection

import json
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ============================================================
# SECRET HELPERS
# ============================================================
def _secret(*keys: str, default=None):
    for k in keys:
        if k in st.secrets:
            return st.secrets[k]
    return default

def _require_secret(*keys: str):
    value = _secret(*keys)
    if value is None:
        raise KeyError(f"Missing required secret. Tried: {', '.join(keys)}")
    return value

# ============================================================
# OPENAI
# ============================================================
OPENAI_API_KEY = _secret("openai_api_key", "OPENAI_API_KEY", "openai_key")
openai_client: Optional[OpenAI] = None

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# GOOGLE SHEETS
# ============================================================
sheet = None
sheets_init_error: Optional[str] = None

def _init_sheets():
    global sheet, sheets_init_error
    if sheet is not None or sheets_init_error is not None:
        return
    try:
        gcp_sa = _require_secret("gcp_service_account")
        gsheet_id = _require_secret("gsheet_id")
        scope = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(gcp_sa, scopes=scope)
        client = gspread.authorize(creds)
        book = client.open_by_key(gsheet_id)
        try:
            sheet_local = book.worksheet("Form")
        except Exception:
            sheet_local = book.add_worksheet(title="Form", rows=2000, cols=20)
            sheet_local.append_row(["timestamp", "name", "json"])
        sheet = sheet_local
    except Exception as e:
        sheets_init_error = str(e)

# ============================================================
# TREND LOGIC
# ============================================================
def compute_trend(current_feeling: int, past_checkins: List[Dict]) -> Dict:
    if not past_checkins:
        return {"trend_text": "No previous visit available.", "delta": None}
    last = past_checkins[-1]
    last_feeling = last.get("feeling_level")
    if last_feeling is None:
        return {"trend_text": "Previous visit missing feeling data.", "delta": None}
    delta = current_feeling - last_feeling
    if delta > 0:
        direction = "increased"
    elif delta < 0:
        direction = "decreased"
    else:
        direction = "remained stable"
    return {
        "trend_text": f"Previous visit: {last_feeling}/10. Today: {current_feeling}/10. Feeling has {direction}" + (f" by {abs(delta)} point(s)." if delta != 0 else "."),
        "delta": delta,
    }

# ============================================================
# LOAD / SAVE
# ============================================================
def load_past_checkins(name: str) -> List[Dict]:
    _init_sheets()
    if sheet is None:
        return []
    rows = sheet.get_all_values()
    past = []
    for row in rows[1:]:
        if len(row) >= 3 and row[1].strip().lower() == name.strip().lower():
            try:
                data = json.loads(row[2])
                data["timestamp"] = row[0]
                past.append(data)
            except Exception:
                continue
    return past[-5:]

def save_to_sheet():
    _init_sheets()
    if sheet is None:
        raise RuntimeError(f"Google Sheets not available: {sheets_init_error}")
    trend_info = compute_trend(
        st.session_state.feeling_level,
        st.session_state.get("past_checkins", [])
    )
    chat_dict = {
        "feeling_level": st.session_state.feeling_level,
        "trend_delta": trend_info.get("delta"),
        "trend_summary": trend_info.get("trend_text"),
        "conversation": st.session_state.messages,
    }
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = st.session_state.get("patient_name", "Unknown")
    sheet.append_row([timestamp, name, json.dumps(chat_dict)])

# ============================================================
# GPT PROMPT
# ============================================================
def build_system_prompt():
    name = st.session_state.get("patient_name", "the patient")
    feeling = st.session_state.get("feeling_level")
    trend_info = compute_trend(feeling, st.session_state.get("past_checkins", [])) if feeling is not None else {"trend_text": "Not available"}
    return f"""You are a warm symptom intake assistant.
Patient: {name}
Today's feeling: {feeling}/10

Trend:
{trend_info['trend_text']}

Rules:
- Be warm.
- Ask one question at a time.
- No medical advice.
"""

def get_gpt_reply():
    if not openai_client:
        return "(LLM not configured.)"
    messages = [{"role": "system", "content": build_system_prompt()}]
    for m in st.session_state.messages[-20:]:
        role = "assistant" if m["role"] == "doctor" else "user"
        messages.append({"role": role, "content": m["content"]})
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=250,
        temperature=0.6,
    )
    return response.choices[0].message.content.strip()

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "messages": [],
    "stage": -1,
    "patient_name": "",
    "feeling_level": 5,
    "past_checkins": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# UI
# ============================================================
st.title("🩺 Cancer Symptom Check-In")

if st.session_state.stage == -1:
    name = st.text_input("Enter your name")
    if st.button("Start"):
        if name.strip():
            st.session_state.patient_name = name.strip()
            st.session_state.past_checkins = load_past_checkins(name.strip())
            st.session_state.stage = 0
            st.rerun()
    st.stop()

for msg in st.session_state.messages:
    if msg["role"] == "doctor":
        st.markdown(f"**🩺 Assistant:** {msg['content']}")
    else:
        st.markdown(f"**🙂 You:** {msg['content']}")

if st.session_state.stage == 0:
    level = st.slider("0 = worst, 10 = best", 0, 10, st.session_state.feeling_level)
    if st.button("Send feeling"):
        st.session_state.feeling_level = level
        st.session_state.messages.append({"role": "patient", "content": f"My feeling is {level}/10."})
        reply = get_gpt_reply()
        st.session_state.messages.append({"role": "doctor", "content": reply})
        st.session_state.stage = 1
        st.rerun()

elif st.session_state.stage == 1:
    if st.button("Submit Check-In"):
        try:
            save_to_sheet()
            st.success("Saved successfully.")
        except Exception as e:
            st.error(f"Failed to save to Google Sheets: {e}")
