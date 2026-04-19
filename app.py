import json
import os
import re
from copy import deepcopy
from typing import Any

import streamlit as st
from openai import OpenAI


st.set_page_config(page_title="ChatReport", page_icon="🩺", layout="centered")


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
        "main": [
            {
                "id": "M10",
                "question": "Are you having problems with mucus or secretions?",
                "type": "yes_no",
            }
        ],
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
    "general",
]


DEFAULT_MODEL = "gpt-4.1-mini"


def extract_json(text: str, fallback: Any) -> Any:
    if not text:
        return fallback
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}|\[.*\]", cleaned, re.DOTALL)
    if not match:
        return fallback
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return fallback


def get_openai_client() -> OpenAI | None:
    api_key = (
        st.session_state.get("api_key_input")
        or st.secrets.get("OPENAI_API_KEY", None)
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def call_json_agent(system_prompt: str, user_payload: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    client = get_openai_client()
    if client is None:
        return fallback

    try:
        response = client.responses.create(
            model=st.session_state.get("model_name", DEFAULT_MODEL),
            temperature=0,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ],
        )
        parsed = extract_json(response.output_text, fallback)
        return parsed if isinstance(parsed, dict) else fallback
    except Exception as exc:
        print(f"[agent-error] {exc}")
        return fallback


def normalize_presence(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"yes", "present", "positive", "true"}:
        return "yes"
    if text in {"no", "absent", "negative", "false"}:
        return "no"
    return "unknown"


def init_session_state() -> None:
    if "initialized" in st.session_state:
        return

    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.current_topic_index = 0
    st.session_state.current_topic = TOPIC_ORDER[0]
    st.session_state.collected_data = {}
    st.session_state.topic_states = {
        topic: {
            "main_asked": False,
            "main_answered": False,
            "asked_followups": [],
            "completed": False,
            "summary": "",
            "presence": "unknown",
        }
        for topic in TOPIC_ORDER
    }
    st.session_state.current_question = ""
    st.session_state.final_report = ""
    st.session_state.finished = False
    st.session_state.api_key_input = ""
    st.session_state.model_name = DEFAULT_MODEL


def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def get_topic_config(topic: str) -> dict[str, Any]:
    return KNOWLEDGE_BASE[topic]


def get_topic_data(topic: str) -> dict[str, Any]:
    return st.session_state.collected_data.setdefault(topic, {"notes": []})


def required_fields_for_topic(topic: str) -> list[str]:
    config = get_topic_config(topic)
    fields = list(config.get("required_fields", []))
    for item in config.get("main", []):
        for field in item.get("required_fields", []):
            if field not in fields:
                fields.append(field)
    return fields


def get_next_unfinished_topic() -> str | None:
    for index in range(st.session_state.current_topic_index, len(TOPIC_ORDER)):
        topic = TOPIC_ORDER[index]
        if not st.session_state.topic_states[topic]["completed"]:
            st.session_state.current_topic_index = index
            st.session_state.current_topic = topic
            return topic
    return None


def main_question_for(topic: str) -> str:
    return get_topic_config(topic)["main"][0]["question"]


def symptom_extraction_agent(
    patient_message: str,
    topic: str,
    question: str,
    topic_data: dict[str, Any],
) -> dict[str, Any]:
    config = get_topic_config(topic)
    fallback = {
        "topic": topic,
        "presence": "unknown",
        "details": {},
        "summary": patient_message,
    }
    system_prompt = f"""
You are the Symptom Extraction Agent for a head and neck cancer symptom reporting assistant.
Extract structured data from the patient's latest answer.

Return JSON only with this schema:
{{
  "topic": "{topic}",
  "presence": "yes" | "no" | "unknown",
  "details": {{}},
  "summary": "brief clinical summary"
}}

Rules:
- Focus on the current topic only.
- Use the question, topic configuration, and existing topic data for context.
- If a field is not stated, do not invent it.
- Keep detail values short and clinical.
"""
    payload = {
        "topic": topic,
        "question": question,
        "topic_config": config,
        "existing_topic_data": topic_data,
        "patient_message": patient_message,
    }
    result = call_json_agent(system_prompt, payload, fallback)
    result["presence"] = normalize_presence(result.get("presence"))
    if not isinstance(result.get("details"), dict):
        result["details"] = {}
    return result


def clinical_importance_agent(topic: str, extracted: dict[str, Any], topic_data: dict[str, Any]) -> dict[str, Any]:
    config = get_topic_config(topic)
    fallback = {
        "importance_level": config.get("priority", "medium"),
        "follow_up_needed": False,
        "missing_fields": [],
        "reason": "Fallback decision",
    }
    system_prompt = """
You are the Clinical Importance Agent.
Assess whether the current symptom answer needs follow-up.

Return JSON only:
{
  "importance_level": "low" | "medium" | "high" | "urgent",
  "follow_up_needed": true | false,
  "missing_fields": [],
  "reason": "short rationale"
}

Rules:
- Base the decision on extracted symptoms, topic requirements, and what is already collected.
- If the symptom is clearly absent, follow_up_needed should usually be false.
- If required clinical details are missing for a present high-priority symptom, follow_up_needed should be true.
"""
    payload = {
        "topic": topic,
        "topic_config": config,
        "extracted": extracted,
        "topic_data": topic_data,
        "required_fields": required_fields_for_topic(topic),
    }
    result = call_json_agent(system_prompt, payload, fallback)
    if not isinstance(result.get("missing_fields"), list):
        result["missing_fields"] = []
    return result


def follow_up_agent(topic: str, missing_fields: list[str], asked_followups: list[str], topic_data: dict[str, Any]) -> dict[str, Any]:
    config = get_topic_config(topic)
    candidate_followups = [q for q in config.get("followups", []) if q not in asked_followups]
    fallback_question = candidate_followups[0] if candidate_followups else ""
    fallback = {"question": fallback_question}
    system_prompt = """
You are the Follow-up Agent.
Choose exactly one follow-up question from the provided knowledge base options.

Return JSON only:
{
  "question": "one question"
}

Rules:
- You must choose from the candidate_followups list only.
- Prioritize questions that fill missing fields first.
- If no candidate follow-up is available, return an empty string.
"""
    payload = {
        "topic": topic,
        "missing_fields": missing_fields,
        "candidate_followups": candidate_followups,
        "topic_data": topic_data,
    }
    result = call_json_agent(system_prompt, payload, fallback)
    question = result.get("question", "")
    if question not in candidate_followups:
        question = fallback_question
    return {"question": question}


def patient_experience_agent(history: list[dict[str, str]]) -> dict[str, Any]:
    fallback = {"fatigue_level": "medium", "engagement_note": "Fallback estimate"}
    system_prompt = """
You are the Patient Experience Agent.
Estimate patient conversation fatigue from the conversation history.

Return JSON only:
{
  "fatigue_level": "low" | "medium" | "high",
  "engagement_note": "short note"
}

Rules:
- Focus on conversation burden, brevity, frustration, repetition, or signs of tiring.
- Do not infer medical fatigue; this is conversational fatigue only.
"""
    payload = {"conversation_history": history[-12:]}
    return call_json_agent(system_prompt, payload, fallback)


def topic_missing_fields(topic: str, merged_topic_data: dict[str, Any]) -> list[str]:
    required_fields = required_fields_for_topic(topic)
    missing = []
    for field in required_fields:
        value = merged_topic_data.get(field)
        if value in (None, "", [], {}):
            missing.append(field)
    return missing


def orchestrator_agent(
    topic: str,
    extracted: dict[str, Any],
    importance: dict[str, Any],
    experience: dict[str, Any],
    topic_state: dict[str, Any],
    topic_data: dict[str, Any],
) -> dict[str, Any]:
    config = get_topic_config(topic)
    fallback = {"action": "next_topic", "reason": "Fallback orchestration"}
    system_prompt = """
You are the Orchestrator Agent for a multi-agent clinical chatbot.
Decide the next step.

Return JSON only:
{
  "action": "ask_followup" | "next_topic" | "finish",
  "reason": "short rationale"
}

Rules:
- Ask a follow-up if clinically important details are still missing and a follow-up is appropriate.
- Move to the next topic when the current topic is sufficiently covered.
- Finish only when all topics have been completed.
- If patient conversational fatigue is high, prefer fewer follow-ups unless a high-priority issue still needs clarification.
"""
    payload = {
        "topic": topic,
        "topic_config": config,
        "extracted": extracted,
        "importance": importance,
        "experience": experience,
        "topic_state": topic_state,
        "topic_data": topic_data,
        "remaining_followups": [
            q for q in config.get("followups", []) if q not in topic_state.get("asked_followups", [])
        ],
    }
    result = call_json_agent(system_prompt, payload, fallback)
    action = result.get("action", "next_topic")
    if action not in {"ask_followup", "next_topic", "finish"}:
        action = "next_topic"
    return {"action": action, "reason": result.get("reason", fallback["reason"])}


def report_generator_agent(collected_data: dict[str, Any], history: list[dict[str, str]]) -> str:
    fallback_report = build_fallback_report(collected_data)
    client = get_openai_client()
    if client is None:
        return fallback_report

    system_prompt = """
You are the Report Generator Agent.
Convert collected symptom data into a structured clinical report for a head and neck cancer symptom check-in.

Return JSON only:
{
  "report_markdown": "markdown report"
}

Rules:
- Keep it concise and clinically useful.
- Include major positive symptoms, relevant negatives, functional impact, medications, and a short summary.
- Do not include billing or treatment recommendations.
"""
    payload = {
        "collected_data": collected_data,
        "conversation_history": history,
    }
    try:
        response = client.responses.create(
            model=st.session_state.get("model_name", DEFAULT_MODEL),
            temperature=0,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ],
        )
        parsed = extract_json(response.output_text, {"report_markdown": fallback_report})
        report = parsed.get("report_markdown", fallback_report)
        return report if isinstance(report, str) and report.strip() else fallback_report
    except Exception as exc:
        print(f"[report-error] {exc}")
        return fallback_report


def build_fallback_report(collected_data: dict[str, Any]) -> str:
    lines = ["## Structured Clinical Report", ""]
    for topic in TOPIC_ORDER:
        topic_data = collected_data.get(topic, {})
        if not topic_data:
            continue
        lines.append(f"### {topic.replace('_', ' ').title()}")
        for key, value in topic_data.items():
            if value in (None, "", [], {}):
                continue
            label = key.replace("_", " ").title()
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            lines.append(f"- **{label}:** {value}")
        lines.append("")
    return "\n".join(lines).strip()


def merge_topic_data(topic: str, extracted: dict[str, Any]) -> dict[str, Any]:
    topic_data = get_topic_data(topic)
    details = extracted.get("details", {})
    if extracted.get("presence") != "unknown":
        topic_data["presence"] = extracted["presence"]
    if extracted.get("summary"):
        topic_data["summary"] = extracted["summary"]
    for key, value in details.items():
        if value not in (None, "", [], {}):
            topic_data[key] = value
    return topic_data


def ask_assistant_question(question: str) -> None:
    st.session_state.current_question = question
    add_message("assistant", question)


def move_to_next_topic() -> None:
    current_topic = st.session_state.current_topic
    st.session_state.topic_states[current_topic]["completed"] = True
    next_topic = get_next_unfinished_topic()
    if next_topic is None:
        finish_chat()
        return
    ask_assistant_question(main_question_for(next_topic))
    st.session_state.topic_states[next_topic]["main_asked"] = True


def finish_chat() -> None:
    st.session_state.finished = True
    report = report_generator_agent(st.session_state.collected_data, st.session_state.messages)
    st.session_state.final_report = report
    add_message(
        "assistant",
        "Thank you. I have enough information now and generated your structured symptom report below.",
    )


def start_chat() -> None:
    if st.session_state.messages:
        return
    welcome = (
        "Hello, I’m ChatReport. I’ll ask a few symptom questions to help create a structured clinical report "
        "for your care team."
    )
    add_message("assistant", welcome)
    first_topic = st.session_state.current_topic
    first_question = main_question_for(first_topic)
    st.session_state.topic_states[first_topic]["main_asked"] = True
    ask_assistant_question(first_question)


def process_turn(patient_message: str) -> None:
    if st.session_state.finished:
        return

    topic = st.session_state.current_topic
    topic_state = st.session_state.topic_states[topic]
    current_question = st.session_state.current_question

    add_message("user", patient_message)

    extracted = symptom_extraction_agent(
        patient_message=patient_message,
        topic=topic,
        question=current_question,
        topic_data=deepcopy(get_topic_data(topic)),
    )
    merged_data = merge_topic_data(topic, extracted)
    topic_state["main_answered"] = True
    topic_state["presence"] = extracted.get("presence", "unknown")
    topic_state["summary"] = extracted.get("summary", "")

    inferred_missing = topic_missing_fields(topic, merged_data)
    importance = clinical_importance_agent(topic, extracted, deepcopy(merged_data))
    llm_missing = [item for item in importance.get("missing_fields", []) if isinstance(item, str)]
    if inferred_missing:
        importance["missing_fields"] = sorted(set(llm_missing + inferred_missing))

    experience = patient_experience_agent(st.session_state.messages)
    orchestration = orchestrator_agent(
        topic=topic,
        extracted=extracted,
        importance=importance,
        experience=experience,
        topic_state=deepcopy(topic_state),
        topic_data=deepcopy(merged_data),
    )

    print(
        "[decision]",
        json.dumps(
            {
                "topic": topic,
                "extracted": extracted,
                "importance": importance,
                "experience": experience,
                "orchestrator": orchestration,
            },
            ensure_ascii=True,
        ),
    )

    action = orchestration["action"]
    remaining_followups = [
        q for q in get_topic_config(topic).get("followups", []) if q not in topic_state["asked_followups"]
    ]
    high_conversation_fatigue = experience.get("fatigue_level") == "high"

    if action == "finish":
        finish_chat()
        return

    if action == "ask_followup" and remaining_followups:
        if high_conversation_fatigue and importance.get("importance_level") not in {"high", "urgent"}:
            move_to_next_topic()
            return

        followup = follow_up_agent(
            topic=topic,
            missing_fields=importance.get("missing_fields", []),
            asked_followups=topic_state["asked_followups"],
            topic_data=deepcopy(merged_data),
        )
        question = followup.get("question", "").strip()
        if question:
            topic_state["asked_followups"].append(question)
            ask_assistant_question(question)
            return

    move_to_next_topic()


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Configuration")
        st.session_state.api_key_input = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get("api_key_input", ""),
            type="password",
            help="Stored only in this Streamlit session. You can also set OPENAI_API_KEY.",
        )
        st.session_state.model_name = st.text_input(
            "Model",
            value=st.session_state.get("model_name", DEFAULT_MODEL),
            help="Responses API model name.",
        )
        st.caption("For safety, the app does not store an API key in code.")

        if st.button("Reset Chat"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def render_chat() -> None:
    st.title("ChatReport")
    st.caption("Multi-agent clinical chatbot for head and neck cancer symptom reporting")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.final_report:
        st.markdown("### Final Report")
        st.info(st.session_state.final_report)


def main() -> None:
    init_session_state()
    render_sidebar()
    start_chat()
    render_chat()

    if st.session_state.finished:
        return

    if prompt := st.chat_input("Type your answer here"):
        process_turn(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
