
import json
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Smart Medical Chatbot", layout="wide")

OPENAI_API_KEY = st.secrets.get("openai_api_key", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if "chat" not in st.session_state:
    st.session_state.chat = []
if "data" not in st.session_state:
    st.session_state.data = {}
if "pending_followup" not in st.session_state:
    st.session_state.pending_followup = False
if "current_question" not in st.session_state:
    st.session_state.current_question = "Where is your pain?"

SYSTEM = "You are a compassionate clinical nurse."
RED_FLAGS = "severe pain, blood, fever, breathing issues"

def llm_dynamic_followup(question, answer, data):
    if not client:
        return {"action": "accept", "message": "Got it."}

    prompt = f'''
{SYSTEM}

QUESTION: {question}
PATIENT: {answer}

DATA: {json.dumps(data)}

RED FLAGS: {RED_FLAGS}

Decide next step:

- accept
- followup
- flag

Return JSON:
{{
 "action": "...",
 "message": "...",
 "question": "..."
}}
'''

    try:
        res = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {"action": "accept", "message": "Thanks."}

st.title("🩺 Smart Clinical Chatbot")

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.write(m["content"])

with st.chat_message("assistant"):
    st.write(st.session_state.current_question)

user_input = st.chat_input("Type your answer")

if user_input:
    st.session_state.chat.append({"role": "user", "content": user_input})

    decision = llm_dynamic_followup(
        st.session_state.current_question,
        user_input,
        st.session_state.data
    )

    if decision["action"] == "accept":
        st.session_state.data[st.session_state.current_question] = user_input
        st.session_state.chat.append({
            "role": "assistant",
            "content": decision["message"]
        })
        st.session_state.current_question = "Anything else about your symptoms?"

    elif decision["action"] == "flag":
        st.session_state.chat.append({
            "role": "assistant",
            "content": decision["message"]
        })

    elif decision["action"] == "followup":
        st.session_state.chat.append({
            "role": "assistant",
            "content": decision["message"]
        })
        st.session_state.chat.append({
            "role": "assistant",
            "content": decision.get("question", "Can you tell me more?")
        })
        st.session_state.pending_followup = True

    st.rerun()
