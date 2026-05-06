import json
import os
from typing import Dict, List, Any

import streamlit as st
from openai import OpenAI


# =========================
# Prompt / System Logic
# =========================

SYSTEM_PROMPT = """
Role:
You are a compassionate and professional nurse assistant conducting a conversational check-in with a head and neck cancer patient before their doctor visit. Your goal is to gather clinically relevant information and summarize it for the doctor.

Core Objectives:
- Collect all required clinical information across the specified topics.
- Adapt dynamically to the patient's responses.
- Minimize burden by avoiding unnecessary or repetitive questions.
- Maintain a natural, empathetic, human-like conversation, not a checklist or survey.
- Use prior patient history, if available, to personalize questions and avoid redundancy.



Response Format:
Always respond as valid JSON with exactly these keys:
{
  "reply": "Natural message to show the patient.",
  "is_complete": true or false,
  "doctor_summary": "Concise clinical summary for the doctor if complete, otherwise an empty string."
}


Opening:
If this is the first assistant message, start exactly with:
"How have you been doing compared to your last visit?"

Carefully analyze the patient's response:
- Extract any already-answered topics.
- Acknowledge the patient's concerns.
- Prioritize follow-ups based on what they mention.
- Do not lose track of the initial answer. Refer back to it when relevant.

Conversational Behavior Rules:
- Ask only one question at a time.
- Do not ask long lists of questions.
- If a patient answers multiple topics at once, do not repeat questions already answered.
- Expand only where details are missing.
- Use conditional logic:
  - If a patient says no to a topic, do not ask follow-ups for that topic.
- Occasionally offer guided options when helpful, especially for medications or symptoms patients may not recall precisely.
- Keep tone warm, reassuring, and professional.

Clinical Topics To Cover:
1. Pain
If pain is reported:
- Location (if the patient did not already mention the location of pain)
- Severity
- Onset
- Timing: constant vs intermittent
- Medications (please provide some options for patient ahead)
- Medication effectiveness
- Medication side effects such as constipation
- Factors that improve or worsen pain

Ask these factors separately and do not ask them in one question. If patient mentioned multiple pain or issues, ask question separately for each pain.

2. Nutrition
Assess eating status using categories:
- Eating normally
- Eating less but managing
- Liquids only / struggling
- Feeding tube only
Ask only relevant follow-ups based on category.
Also assess:
- Fluid intake
- Barriers to eating/drinking
- Use of nutritional supplements

Ask these factors separately and do not ask them in one question. 

3. Swallowing
Assess:
- Difficulty swallowing, choking, or coughing
If yes:
- Liquids vs solids
- Frequency
- Pills

4. Feeding Tube
If applicable:
- Functionality: leakage, blockage, discomfort
- Oral intake ability

5. Oral Symptoms
Assess:
- Mouth sores: new vs existing, location, pain and impact, treatments such as magic mouthwash
- Dry mouth: timing, treatments, functional impact
- Mucus: thick vs watery, impact, management
- Teeth/gums issues
- Oral hygiene practices

6. GI Symptoms
Assess:
- Nausea
- Vomiting
- Blood when coughing
- Constipation: frequency, medications, discomfort

7. Fatigue & Sleep
Assess:
- Fatigue: general vs localized weakness
- Impact on daily life
- Sleep: trouble falling/staying asleep
- Causes such as pain or other symptoms

8. Activity & Independence
Assess:
- Ability to perform daily activities
If limited:
- Which activities
- Cause: pain, fatigue, other

9. Mood & Support
Assess:
- Emotional state: anxiety, worry
- Impact on functioning
- Social support system

Use of Patient History:
If prior patient history is provided:
- Reference it naturally, for example:
  "Last time you mentioned some difficulty eating. Has that improved?"
- Avoid re-asking unchanged information unless clarification is needed.

Efficiency Rules:
Avoid asking:
- Questions already answered
- Irrelevant follow-ups

Prioritize:
- Symptoms impacting safety, such as weight loss, swallowing issues, bleeding
- Keeping the conversation concise but complete

Conversation Completion Rule:
After you are done with all the question, ask the patient if there is any other comment that patient would like to talk about and then end the conversation. 
Once all required information across all clinical topics has been gathered and you ask the other comment question:
- Clearly inform the patient that the check-in is complete.
- Use a warm and reassuring tone.
- Briefly summarize what will happen next, such as that the information will be shared with the doctor.
- Do not continue asking questions once complete.

Internal Output Requirement:
While chatting naturally, internally ensure you can produce a structured summary for the doctor including:
- Key symptoms
- Changes since last visit
- Red flags
- Medication effectiveness
- Functional status

Response Format:
Always respond as valid JSON with exactly these keys:
{
  "reply": "Natural message to show the patient.",
  "is_complete": true or false,
  "doctor_summary": "Concise clinical summary for the doctor if complete, otherwise an empty string."
}

Important:
- The patient should only see the value of "reply".
- Set "is_complete" to true only when the check-in is genuinely complete.
"""


def build_messages(
    chat_history: List[Dict[str, str]],
    prior_history: str = "",
) -> List[Dict[str, str]]:
    system_content = SYSTEM_PROMPT

    if prior_history.strip():
        system_content += f"""

Prior Patient History:
{prior_history.strip()}
"""

    messages = [{"role": "system", "content": system_content}]
    messages.extend(chat_history)
    return messages


def get_nurse_response(
    client: OpenAI,
    chat_history: List[Dict[str, str]],
    prior_history: str,
    model: str,
) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(chat_history, prior_history),
        temperature=0.4,
    )

    raw_content = response.choices[0].message.content or ""

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        parsed = {
            "reply": raw_content,
            "is_complete": False,
            "doctor_summary": "",
        }

    return {
        "reply": parsed.get("reply", "").strip(),
        "is_complete": bool(parsed.get("is_complete", False)),
        "doctor_summary": parsed.get("doctor_summary", "").strip(),
        "raw_response": raw_content.strip(),
    }


# =========================
# UI Helpers
# =========================

def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "is_complete" not in st.session_state:
        st.session_state.is_complete = False

    if "doctor_summary" not in st.session_state:
        st.session_state.doctor_summary = ""

    if "started" not in st.session_state:
        st.session_state.started = False

    if "raw_responses" not in st.session_state:
        st.session_state.raw_responses = []


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.is_complete = False
    st.session_state.doctor_summary = ""
    st.session_state.started = False
    st.session_state.raw_responses = []


def render_completion_banner() -> None:
    st.title("Thank you for submitting your check-in")
    st.success("Your responses have been received and are ready to be shared with your doctor.")

    st.subheader("Summary")
    with st.container(border=True):
        st.write(st.session_state.doctor_summary or "Summary is being prepared.")


def render_chat_history() -> None:
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(message["content"])
        elif message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])


def render_raw_responses() -> None:
    if not st.session_state.raw_responses:
        st.info("No GPT responses yet.")
        return

    for index, raw_response in enumerate(st.session_state.raw_responses, start=1):
        st.markdown(f"**Step {index}**")
        st.code(raw_response, language="json")


def add_assistant_message(content: str) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
        }
    )


def add_user_message(content: str) -> None:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": content,
        }
    )


# =========================
# Streamlit App
# =========================

def main() -> None:
    st.set_page_config(
        page_title="Nurse Assistant Check-In",
        page_icon="🩺",
        layout="centered",
    )

    initialize_state()

    st.title("Nurse Assistant Check-In")
    st.caption("A conversational pre-visit check-in for head and neck cancer care.")

    with st.sidebar:
        st.header("Settings")

        model = st.selectbox(
            "OpenAI model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
            index=0,
        )

        prior_history = st.text_area(
            "Prior patient history",
            placeholder=(
                "Optional example: Last visit, patient reported mild swallowing "
                "difficulty and reduced appetite."
            ),
            height=160,
        )

        if st.button("Start new check-in", use_container_width=True):
            reset_chat()
            st.rerun()

        st.divider()

        if st.session_state.is_complete:
            st.markdown("**Status:** Complete")
        else:
            st.markdown("**Status:** In progress")

        raw_tab = st.tabs(["Raw GPT response"])[0]
        with raw_tab:
            raw_response_placeholder = st.empty()
            with raw_response_placeholder.container():
                render_raw_responses()

    try:
      api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
      st.error("OPENAI_API_KEY is missing from Streamlit secrets.")
      st.stop()

    client = OpenAI(api_key=api_key)

    if not st.session_state.started:
        opening_message = "How have you been doing compared to your last visit?"
        add_assistant_message(opening_message)
        st.session_state.started = True

    if st.session_state.is_complete:
        render_completion_banner()
        with raw_response_placeholder.container():
            render_raw_responses()
        return

    render_chat_history()

    patient_input = st.chat_input("Type your response...")

    if patient_input:
        add_user_message(patient_input)

        with st.chat_message("user"):
            st.write(patient_input)

        with st.chat_message("assistant"):
            with st.spinner("Nurse assistant is reviewing your response..."):
                result = get_nurse_response(
                    client=client,
                    chat_history=st.session_state.messages,
                    prior_history=prior_history,
                    model=model,
                )

            assistant_reply = result["reply"]
            if not result["is_complete"]:
                st.write(assistant_reply)

        st.session_state.raw_responses.append(result["raw_response"])

        with raw_response_placeholder.container():
            render_raw_responses()

        if not result["is_complete"]:
            add_assistant_message(assistant_reply)

        st.session_state.is_complete = result["is_complete"]
        st.session_state.doctor_summary = result["doctor_summary"]

        if st.session_state.is_complete:
            st.rerun()


if __name__ == "__main__":
    main()
