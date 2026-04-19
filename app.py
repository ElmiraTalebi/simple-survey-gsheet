"""Streamlit multi-agent clinical chatbot for head and neck cancer symptom reporting."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import streamlit as st
from openai import OpenAI

APP_TITLE = "ChatReport"
DEFAULT_MODEL = "gpt-4o-mini"
MAX_FOLLOWUPS_PER_TOPIC = 2

KNOWLEDGE_BASE = {
    "general": {
        "main": [
            {
                "id": "M1",
                "question": "How has your overall feeling been since your last visit? Rate 0–10.",
                "type": "scale",
                "required_fields": ["overall_score"],
            }
        ]
    },
    "pain": {
        "priority": "high",
        "main": [{"id": "M2", "question": "Do you have any pain today?", "type": "yes_no"}],
        "followups": [
            "Where exactly is the pain?",
            "Is it throat, tongue, or somewhere else?",
            "Is the pain constant or only when swallowing or eating?",
            "On a scale of 0–10, how bad is it?",
            "Are you taking pain medication? Is it helping?",
            "Does the pain spread or stay in one spot?",
            "Does it worsen when chewing or opening your mouth?",
            "When did the pain start?",
        ],
        "required_fields": ["location", "severity", "timing"],
    },
    "oral_sores": {
        "priority": "high",
        "main": [
            {
                "id": "M4",
                "question": "Do you have any mouth sores or ulcers right now?",
                "type": "yes_no",
            }
        ],
        "followups": [
            "Is this sore new or from before?",
            "Where is it located?",
            "Is it painful?",
            "Is it affecting eating or drinking?",
            "Are you using mouthwash or medication?",
            "Is it getting better or worse?",
        ],
    },
    "nutrition": {
        "priority": "high",
        "main": [
            {
                "id": "M5",
                "question": "How has your eating been? Are you able to eat and drink enough?",
                "type": "multi_choice",
            }
        ],
        "followups": [
            "What are you able to eat right now?",
            "Are you drinking enough fluids?",
            "What is making it difficult to eat or drink?",
            "How many nutritional shakes per day?",
            "Are you timing pain medication before meals?",
        ],
        "required_fields": ["intake_level", "barriers"],
    },
    "weight": {
        "priority": "high",
        "main": [{"id": "M6", "question": "What has your weight been recently?", "type": "numeric"}],
        "followups": ["Is weight loss affecting your energy?"],
    },
    "dry_mouth": {
        "priority": "medium",
        "main": [{"id": "M7", "question": "Are you experiencing dryness in your mouth?", "type": "yes_no"}],
        "followups": [
            "Is it worse at night or all day?",
            "Are you using saliva substitutes?",
            "Is it affecting eating, talking, or sleeping?",
        ],
    },
    "swallowing": {
        "priority": "high",
        "main": [{"id": "M8", "question": "Are you having difficulty swallowing?", "type": "yes_no"}],
        "followups": [
            "Is it painful or mechanical difficulty?",
            "Do you cough or choke when eating?",
            "Can you still swallow liquids?",
        ],
    },
    "breathing": {
        "priority": "high",
        "main": [{"id": "M9", "question": "Are you having difficulty breathing?", "type": "yes_no"}],
        "followups": ["Is it constant or with activity?", "Are you wheezing?"],
    },
    "mucus": {
        "priority": "medium",
        "main": [{"id": "M10", "question": "Are you having problems with mucus or secretions?", "type": "yes_no"}],
        "followups": [
            "Is it thick or watery?",
            "Is it affecting swallowing or sleep?",
            "Are you using treatments?",
        ],
    },
    "gi": {
        "priority": "high",
        "main": [
            {
                "id": "M11",
                "question": "Have you had nausea, vomiting, or blood when coughing?",
                "type": "multi_choice",
            }
        ],
    },
    "medications": {
        "priority": "high",
        "main": [{"id": "M12", "question": "What medications are you taking?", "type": "multi_choice"}],
        "followups": ["How often and what dose?", "Are they making you drowsy?"],
    },
    "fatigue": {
        "priority": "medium",
        "main": [{"id": "M13", "question": "Are you feeling more tired or weak than usual?", "type": "yes_no"}],
        "followups": ["Is it general fatigue or specific weakness?", "Is it affecting daily activities?"],
    },
    "sleep": {
        "priority": "medium",
        "main": [{"id": "M14", "question": "Are you able to sleep through the night?", "type": "yes_no"}],
        "followups": ["Are you waking due to pain or dryness?", "Is medication affecting sleep?"],
    },
    "mood": {
        "priority": "medium",
        "main": [{"id": "M15", "question": "How are you feeling emotionally?", "type": "free_text"}],
        "followups": ["Is anxiety affecting daily activities?", "Do you have support?"],
    },
}

TOPIC_ORDER = [
    "pain",
    "general",
    "oral_sores",
    "nutrition",
    "weight",
    "dry_mouth",
    "swallowing",
    "breathing",
    "mucus",
    "gi",
    "medications",
    "fatigue",
    "sleep",
    "mood",
]


def get_openai_client() -> OpenAI:
    """Create the OpenAI client from Streamlit secrets or environment variables."""
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add it to Streamlit secrets or your environment."
        )
    return OpenAI(api_key=api_key)


def get_model_name() -> str:
    """Resolve the model name from secrets or environment variables."""
    return st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", DEFAULT_MODEL))


def sanitize_json_text(raw_text: str) -> str:
    """Strip Markdown code fences if the model returns fenced JSON."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned


def call_json_agent(agent_name: str, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call an OpenAI JSON agent and parse the response into a dictionary."""
    client = get_openai_client()
    model = get_model_name()
    print(f"[agent:{agent_name}] payload={json.dumps(payload, ensure_ascii=True)}")

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
        ],
    )

    raw_text = response.choices[0].message.content or "{}"
    parsed = json.loads(sanitize_json_text(raw_text))
    print(f"[agent:{agent_name}] response={json.dumps(parsed, ensure_ascii=True)}")
    return parsed


def get_topic_config(topic: str) -> dict[str, Any]:
    """Return the knowledge base entry for a topic."""
    return KNOWLEDGE_BASE[topic]


def get_main_question(topic: str) -> str:
    """Return the main screening question for a topic."""
    return get_topic_config(topic)["main"][0]["question"]


def find_next_unasked_topic(asked_topics: list[str]) -> str | None:
    """Return the next topic that has not had its main question asked yet."""
    for topic in TOPIC_ORDER:
        if topic not in asked_topics:
            return topic
    return None


def init_topic_state(topic: str) -> dict[str, Any]:
    """Create the storage shape for a topic in collected data."""
    return {
        "topic": topic,
        "main_question": get_main_question(topic),
        "main_answer": "",
        "field_updates": {},
        "response_summary": "",
        "symptoms": [],
        "concerning_features": [],
        "importance": "low",
        "urgent": False,
        "follow_up_needed": False,
        "missing_fields": [],
        "followups_asked": [],
        "asked_count": 0,
        "completed": False,
        "raw_messages": [],
    }


def initialize_session_state() -> None:
    """Initialize all Streamlit session variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "collected_data" not in st.session_state:
        st.session_state.collected_data = {}
    if "asked_topics" not in st.session_state:
        st.session_state.asked_topics = []
    if "current_topic" not in st.session_state:
        st.session_state.current_topic = "pain"
    if "current_question" not in st.session_state:
        st.session_state.current_question = get_main_question("pain")
    if "report" not in st.session_state:
        st.session_state.report = None
    if "finished" not in st.session_state:
        st.session_state.finished = False
    if "started" not in st.session_state:
        st.session_state.started = False


def append_message(role: str, content: str) -> None:
    """Add a message to the chat transcript."""
    st.session_state.messages.append({"role": role, "content": content})


def ensure_topic_record(topic: str) -> dict[str, Any]:
    """Create the topic data bucket if it does not exist."""
    if topic not in st.session_state.collected_data:
        st.session_state.collected_data[topic] = init_topic_state(topic)
    return st.session_state.collected_data[topic]


def start_conversation() -> None:
    """Seed the welcome message and first question."""
    if st.session_state.started:
        return

    welcome = (
        "Welcome to ChatReport. I will ask a few questions about symptoms and treatment effects "
        "to help generate a structured clinical summary for your care team."
    )
    first_question = get_main_question("pain")
    append_message("assistant", welcome)
    append_message("assistant", first_question)
    st.session_state.started = True
    st.session_state.current_topic = "pain"
    st.session_state.current_question = first_question
    if "pain" not in st.session_state.asked_topics:
        st.session_state.asked_topics.append("pain")


def symptom_extraction_agent(
    patient_message: str,
    current_topic: str,
    current_question: str,
    topic_state: dict[str, Any],
) -> dict[str, Any]:
    """Extract symptom data from the latest patient message."""
    system_prompt = """
You are the Symptom Extraction Agent for a clinical symptom-reporting chatbot.
Return only valid JSON.

Your job:
- Read the patient's latest message in the context of the current topic and question.
- Extract clinically relevant information.
- Infer field values only when they are strongly supported by the text.
- If the patient clearly denies the symptom, set symptom_present to false.
- If the answer is ambiguous, use null rather than inventing details.

Required JSON keys:
- topic: string
- symptom_present: true, false, or null
- response_summary: short plain-English summary
- symptoms: array of strings
- field_updates: object with any extracted structured values
- concerning_features: array of strings
- answered_question: boolean
- patient_message: string
"""
    payload = {
        "topic": current_topic,
        "current_question": current_question,
        "required_fields": get_topic_config(current_topic).get("required_fields", []),
        "known_topic_data": topic_state,
        "patient_message": patient_message,
    }
    return call_json_agent("symptom_extraction", system_prompt, payload)


def clinical_importance_agent(current_topic: str, extracted: dict[str, Any]) -> dict[str, Any]:
    """Assess the importance of the extracted symptoms and whether follow-up is required."""
    system_prompt = """
You are the Clinical Importance Agent for a head and neck cancer symptom chatbot.
Return only valid JSON.

Use the topic priority and extracted data to decide:
- importance_level: low, medium, or high
- urgent_flag: boolean
- follow_up_needed: boolean
- missing_fields: array of fields that should still be clarified
- rationale: one short sentence

Urgent should be true for severe breathing issues, bleeding, inability to swallow liquids,
uncontrolled pain, or other potentially dangerous symptoms clearly described by the patient.
If the patient clearly says a symptom is absent, follow_up_needed should usually be false.
"""
    topic_config = get_topic_config(current_topic)
    payload = {
        "topic": current_topic,
        "priority": topic_config.get("priority", "medium"),
        "required_fields": topic_config.get("required_fields", []),
        "extracted": extracted,
    }
    return call_json_agent("clinical_importance", system_prompt, payload)


def follow_up_agent(current_topic: str, missing_fields: list[str], topic_state: dict[str, Any]) -> dict[str, Any]:
    """Select one follow-up question from the topic knowledge base."""
    system_prompt = """
You are the Follow-up Agent for a clinical symptom-reporting chatbot.
Return only valid JSON.

Rules:
- Choose exactly one follow-up question from the provided options.
- Prefer the question that best addresses the missing fields.
- Do not repeat a question that has already been asked.
- If no follow-up is appropriate, return an empty string for question.

Required JSON keys:
- question: string
- targeted_fields: array of strings
"""
    topic_config = get_topic_config(current_topic)
    payload = {
        "topic": current_topic,
        "missing_fields": missing_fields,
        "followup_options": topic_config.get("followups", []),
        "already_asked": topic_state.get("followups_asked", []),
    }
    return call_json_agent("follow_up", system_prompt, payload)


def patient_experience_agent(conversation_history: list[dict[str, str]]) -> dict[str, Any]:
    """Estimate patient fatigue or burden based on the conversation so far."""
    system_prompt = """
You are the Patient Experience Agent.
Return only valid JSON.

Assess the patient burden from the conversation. Focus on whether the patient sounds tired,
frustrated, brief, disengaged, or overwhelmed.

Required JSON keys:
- fatigue_level: low, medium, or high
- signs: array of short phrases
- recommendation: one short sentence
"""
    payload = {"conversation_history": conversation_history}
    return call_json_agent("patient_experience", system_prompt, payload)


def orchestrator_agent(
    current_topic: str,
    extracted: dict[str, Any],
    importance: dict[str, Any],
    follow_up: dict[str, Any],
    experience: dict[str, Any],
    topic_state: dict[str, Any],
) -> dict[str, Any]:
    """Decide whether to ask follow-up, move on, or finish."""
    system_prompt = """
You are the Orchestrator Agent for a clinical chatbot.
Return only valid JSON.

Choose one action:
- ask_follow_up
- move_to_next_topic
- finish

Decision guidance:
- Ask follow-up if the symptom is present or unclear and clinically relevant details are missing.
- Move on if the topic has been adequately covered or clearly denied.
- Finish only if there are no remaining topics to ask.
- If patient fatigue is high, avoid unnecessary follow-ups.
- If urgent_flag is true, you may still ask one focused follow-up if needed, but keep it brief.

Required JSON keys:
- action: string
- reason: short string
"""
    remaining_topics = [topic for topic in TOPIC_ORDER if topic not in st.session_state.asked_topics]
    payload = {
        "current_topic": current_topic,
        "extracted": extracted,
        "importance": importance,
        "follow_up_candidate": follow_up,
        "patient_experience": experience,
        "topic_state": topic_state,
        "remaining_topics": remaining_topics,
        "max_followups_per_topic": MAX_FOLLOWUPS_PER_TOPIC,
    }
    return call_json_agent("orchestrator", system_prompt, payload)


def report_generator_agent(collected_data: dict[str, Any], conversation_history: list[dict[str, str]]) -> dict[str, Any]:
    """Build the final structured clinical report."""
    system_prompt = """
You are the Report Generator Agent for a clinical symptom-reporting chatbot.
Return only valid JSON.

Generate a concise but structured clinical summary for clinician review.

Required JSON keys:
- report_title: string
- generated_at: string
- chief_concerns: array of strings
- symptom_summary: array of objects with keys topic, summary, importance, urgent
- red_flags: array of strings
- supportive_notes: array of strings
- suggested_clinician_focus: array of strings
"""
    payload = {
        "collected_data": collected_data,
        "conversation_history": conversation_history,
        "generated_at": datetime.now().isoformat(),
    }
    return call_json_agent("report_generator", system_prompt, payload)


def update_topic_state(topic: str, patient_message: str, extracted: dict[str, Any], importance: dict[str, Any]) -> None:
    """Merge the latest agent outputs into the stored topic state."""
    topic_state = ensure_topic_record(topic)
    topic_state["raw_messages"].append(patient_message)

    if not topic_state["main_answer"]:
        topic_state["main_answer"] = patient_message

    field_updates = extracted.get("field_updates", {})
    if isinstance(field_updates, dict):
        topic_state["field_updates"].update(field_updates)

    topic_state["response_summary"] = extracted.get("response_summary", topic_state["response_summary"])
    topic_state["symptoms"] = list(
        dict.fromkeys(topic_state.get("symptoms", []) + extracted.get("symptoms", []))
    )
    topic_state["concerning_features"] = list(
        dict.fromkeys(topic_state.get("concerning_features", []) + extracted.get("concerning_features", []))
    )
    topic_state["importance"] = importance.get("importance_level", topic_state["importance"])
    topic_state["urgent"] = bool(importance.get("urgent_flag", topic_state["urgent"]))
    topic_state["follow_up_needed"] = bool(importance.get("follow_up_needed", False))
    topic_state["missing_fields"] = importance.get("missing_fields", [])

    required_fields = get_topic_config(topic).get("required_fields", [])
    if required_fields:
        still_missing = [field for field in required_fields if field not in topic_state["field_updates"]]
        topic_state["missing_fields"] = still_missing


def render_report(report: dict[str, Any]) -> None:
    """Render the final report in the Streamlit UI."""
    st.markdown("### Structured Clinical Report")
    with st.container(border=True):
        st.markdown(f"**{report.get('report_title', 'Clinical Symptom Report')}**")
        st.caption(report.get("generated_at", ""))

        chief_concerns = report.get("chief_concerns", [])
        if chief_concerns:
            st.markdown("**Chief concerns**")
            for item in chief_concerns:
                st.write(f"- {item}")

        symptom_summary = report.get("symptom_summary", [])
        if symptom_summary:
            st.markdown("**Symptom summary**")
            for item in symptom_summary:
                topic_name = str(item.get("topic", "")).replace("_", " ").title()
                summary = item.get("summary", "")
                importance = item.get("importance", "")
                urgent = "Yes" if item.get("urgent") else "No"
                st.write(f"- {topic_name}: {summary} | Importance: {importance} | Urgent: {urgent}")

        red_flags = report.get("red_flags", [])
        if red_flags:
            st.markdown("**Red flags**")
            for item in red_flags:
                st.write(f"- {item}")

        supportive_notes = report.get("supportive_notes", [])
        if supportive_notes:
            st.markdown("**Supportive notes**")
            for item in supportive_notes:
                st.write(f"- {item}")

        focus_items = report.get("suggested_clinician_focus", [])
        if focus_items:
            st.markdown("**Suggested clinician focus**")
            for item in focus_items:
                st.write(f"- {item}")

        with st.expander("Collected structured data"):
            st.json(st.session_state.collected_data)


def move_to_next_topic() -> None:
    """Advance to the next topic or finish if all topics are complete."""
    next_topic = find_next_unasked_topic(st.session_state.asked_topics)
    if next_topic is None:
        finish_conversation()
        return

    st.session_state.current_topic = next_topic
    st.session_state.current_question = get_main_question(next_topic)
    st.session_state.asked_topics.append(next_topic)
    append_message("assistant", st.session_state.current_question)
    print(f"[orchestrator] action=move_to_next_topic next_topic={next_topic}")


def finish_conversation() -> None:
    """Generate and display the final report."""
    report = report_generator_agent(st.session_state.collected_data, st.session_state.messages)
    st.session_state.report = report
    st.session_state.finished = True
    append_message(
        "assistant",
        "Thank you. I have enough information and generated a structured report for your clinical team.",
    )
    print("[orchestrator] action=finish")


def process_user_message(user_message: str) -> None:
    """Run the full multi-agent pipeline for one user turn."""
    if st.session_state.finished:
        return

    current_topic = st.session_state.current_topic
    topic_state = ensure_topic_record(current_topic)
    topic_state["asked_count"] += 1

    extracted = symptom_extraction_agent(
        patient_message=user_message,
        current_topic=current_topic,
        current_question=st.session_state.current_question,
        topic_state=topic_state,
    )
    importance = clinical_importance_agent(current_topic=current_topic, extracted=extracted)
    update_topic_state(current_topic, user_message, extracted, importance)
    topic_state = ensure_topic_record(current_topic)

    follow_up = {"question": "", "targeted_fields": []}
    if topic_state.get("missing_fields") and topic_state["asked_count"] <= MAX_FOLLOWUPS_PER_TOPIC:
        follow_up = follow_up_agent(current_topic, topic_state["missing_fields"], topic_state)

    experience = patient_experience_agent(st.session_state.messages)
    orchestration = orchestrator_agent(
        current_topic=current_topic,
        extracted=extracted,
        importance=importance,
        follow_up=follow_up,
        experience=experience,
        topic_state=topic_state,
    )

    action = orchestration.get("action", "move_to_next_topic")
    print(f"[orchestrator] topic={current_topic} action={action} reason={orchestration.get('reason', '')}")

    if action == "ask_follow_up" and follow_up.get("question"):
        topic_state["followups_asked"].append(follow_up["question"])
        st.session_state.current_question = follow_up["question"]
        append_message("assistant", follow_up["question"])
        return

    topic_state["completed"] = True

    if action == "finish":
        finish_conversation()
    else:
        move_to_next_topic()


def render_chat_messages() -> None:
    """Display the full chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="centered")
    st.title(APP_TITLE)
    st.caption("Multi-agent clinical chatbot for head and neck cancer symptom reporting")

    initialize_session_state()

    with st.sidebar:
        st.markdown("### Session Status")
        st.write(f"Current topic: `{st.session_state.current_topic}`")
        st.write(f"Topics asked: `{len(st.session_state.asked_topics)}` / `{len(TOPIC_ORDER)}`")
        if st.button("Restart conversation", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    try:
        start_conversation()
        render_chat_messages()

        if st.session_state.report:
            render_report(st.session_state.report)

        if not st.session_state.finished:
            user_message = st.chat_input("Type your response here...")
            if user_message:
                append_message("user", user_message)
                with st.chat_message("user"):
                    st.markdown(user_message)

                with st.spinner("Analyzing your symptoms and preparing the next step..."):
                    process_user_message(user_message)

                st.rerun()
    except Exception as exc:
        st.error(f"Application error: {exc}")
        st.info(
            "Make sure your OpenAI API key is configured as `OPENAI_API_KEY` in Streamlit secrets or environment variables."
        )


if __name__ == "__main__":
    main()
