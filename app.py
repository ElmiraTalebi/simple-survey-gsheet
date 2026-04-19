import json
import os
import re
from copy import deepcopy
from datetime import datetime
from typing import Any

import streamlit as st
from openai import OpenAI

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None


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
TOPIC_LABELS = {
    "pain": "Pain",
    "oral_sores": "Oral Sores",
    "nutrition": "Nutrition",
    "weight": "Weight",
    "dry_mouth": "Dry Mouth",
    "swallowing": "Swallowing",
    "breathing": "Breathing",
    "mucus": "Mucus / Secretions",
    "gi": "GI Symptoms",
    "medications": "Medications",
    "fatigue": "Fatigue",
    "sleep": "Sleep",
    "mood": "Mood",
    "general": "General Well-Being",
}
SECTION_ORDER = [
    ("pain_medications", "Pain & Medications"),
    ("nutrition_fluids", "Nutrition & Fluids"),
    ("oral_symptoms", "Oral Symptoms"),
    ("gi_symptoms", "GI Symptoms"),
    ("fatigue_sleep", "Fatigue & Sleep"),
    ("swallow_breathe", "Swallowing & Breathing"),
    ("mood_support", "Mood"),
    ("general_wellbeing", "General Well-Being"),
]
SECTION_TOPICS = {
    "pain_medications": ["pain", "medications"],
    "nutrition_fluids": ["nutrition", "weight"],
    "oral_symptoms": ["oral_sores", "dry_mouth", "mucus"],
    "gi_symptoms": ["gi"],
    "fatigue_sleep": ["fatigue", "sleep"],
    "swallow_breathe": ["swallowing", "breathing"],
    "mood_support": ["mood"],
    "general_wellbeing": ["general"],
}
SECTION_INTROS = {
    "pain_medications": "Let's focus on pain symptoms and the medications you are using for relief.",
    "nutrition_fluids": "Let's go through eating, drinking, and weight changes.",
    "oral_symptoms": "This section covers mouth sores, dryness, and mucus or secretion concerns.",
    "gi_symptoms": "Let's review nausea, vomiting, or blood when coughing.",
    "fatigue_sleep": "This section focuses on energy level, weakness, and sleep.",
    "swallow_breathe": "Let's review swallowing and breathing symptoms.",
    "mood_support": "This section covers emotional well-being and support.",
    "general_wellbeing": "Let's finish with your overall well-being since the last visit.",
}

_sheet = None
_sheet_error: str | None = None


def extract_json(text: str, fallback: Any) -> Any:
    if not text:
        return fallback


def topic_label(topic: str) -> str:
    return TOPIC_LABELS.get(topic, topic.replace("_", " ").title())


def section_label(section_key: str) -> str:
    for key, label in SECTION_ORDER:
        if key == section_key:
            return label
    return section_key.replace("_", " ").title()


def section_for_topic(topic: str) -> str:
    for section_key, topics in SECTION_TOPICS.items():
        if topic in topics:
            return section_key
    return SECTION_ORDER[0][0]


def category_for_topic(topic: str) -> str:
    mapping = {
        "pain": "pain",
        "swallowing": "swallowing",
        "nutrition": "nutrition",
        "weight": "nutrition",
        "oral_sores": "oral_symptoms",
        "dry_mouth": "oral_symptoms",
        "mucus": "oral_symptoms",
        "gi": "gi_symptoms",
        "fatigue": "fatigue",
        "sleep": "fatigue",
        "mood": "mood",
        "breathing": "breathing",
        "medications": "other",
        "general": "other",
    }
    return mapping.get(topic, "other")
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
        st.secrets.get("OPENAI_API_KEY", None)
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


def question_type_for_topic(topic: str) -> str:
    return get_topic_config(topic)["main"][0].get("type", "free_text")


def effective_presence(topic_data: dict[str, Any], extracted: dict[str, Any]) -> str:
    latest = normalize_presence(extracted.get("presence"))
    if latest != "unknown":
        return latest
    stored = normalize_presence(topic_data.get("presence"))
    return stored


def init_session_state() -> None:
    if "initialized" in st.session_state:
        return

    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.current_topic_index = 0
    st.session_state.current_topic = TOPIC_ORDER[0]
    st.session_state.selected_section = SECTION_ORDER[0][0]
    st.session_state.collected_data = {}
    st.session_state.topic_states = {
        topic: {
            "main_asked": False,
            "main_answered": False,
            "asked_followups": [],
            "completed": False,
            "summary": "",
            "presence": "unknown",
            "current_question": "",
        }
        for topic in TOPIC_ORDER
    }
    st.session_state.section_states = {
        section_key: {"started": False}
        for section_key, _ in SECTION_ORDER
    }
    st.session_state.current_question = ""
    st.session_state.final_report = ""
    st.session_state.finished = False
    st.session_state.model_name = DEFAULT_MODEL
    st.session_state.previous_visit_history = ""
    st.session_state.previous_visit_topics = {}
    st.session_state.patient_name = ""
    st.session_state.sheet_status = ""
    st.session_state.pending_user_input = ""


def add_message(role: str, content: str, topic: str | None = None) -> None:
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "topic": topic,
            "section": section_for_topic(topic) if topic else None,
        }
    )


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


def get_next_unfinished_topic(section_key: str | None = None) -> str | None:
    topics_to_scan = SECTION_TOPICS.get(section_key, TOPIC_ORDER) if section_key else TOPIC_ORDER
    for topic in topics_to_scan:
        if not st.session_state.topic_states[topic]["completed"]:
            st.session_state.current_topic_index = TOPIC_ORDER.index(topic)
            st.session_state.current_topic = topic
            return topic
    return None


def set_active_topic(topic: str) -> None:
    st.session_state.current_topic = topic
    st.session_state.current_topic_index = TOPIC_ORDER.index(topic)
    st.session_state.selected_section = section_for_topic(topic)
    st.session_state.current_question = st.session_state.topic_states[topic].get("current_question", "")


def get_active_topic_for_section(section_key: str) -> str | None:
    current_topic = st.session_state.get("current_topic")
    if current_topic in SECTION_TOPICS.get(section_key, []) and not st.session_state.topic_states[current_topic]["completed"]:
        return current_topic
    return get_next_unfinished_topic(section_key)


def has_remaining_topics() -> bool:
    return any(not st.session_state.topic_states[topic]["completed"] for topic in TOPIC_ORDER)


def main_question_for(topic: str) -> str:
    return get_topic_config(topic)["main"][0]["question"]


def normalize_severity_value(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"low", "mild", "1", "2", "3"}:
        return "low"
    if text in {"medium", "moderate", "4", "5", "6"}:
        return "medium"
    if text in {"high", "severe", "7", "8", "9", "10"}:
        return "high"
    return None


def normalize_fatigue_value(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"none", "no", "normal"}:
        return "none"
    if text in {"mild", "low"}:
        return "mild"
    if text in {"moderate", "medium"}:
        return "moderate"
    if text in {"severe", "high"}:
        return "severe"
    return None


def topic_result_from_extraction(topic: str, extracted: dict[str, Any], patient_message: str) -> dict[str, Any]:
    result = {"topic": topic, "presence": "unknown", "details": {}, "summary": patient_message}
    category = category_for_topic(topic)
    details: dict[str, Any] = {}

    if category == "pain":
        pain = extracted.get("pain", {}) if isinstance(extracted.get("pain"), dict) else {}
        present = pain.get("present")
        result["presence"] = "yes" if present is True else "no" if present is False else "unknown"
        if pain.get("location"):
            details["location"] = pain["location"]
        severity = normalize_severity_value(pain.get("severity"))
        if severity:
            details["severity"] = severity
        if pain.get("timing"):
            details["timing"] = pain["timing"]

    elif category == "swallowing":
        swallowing = str(extracted.get("swallowing") or "").strip().lower()
        if swallowing == "normal":
            result["presence"] = "no"
        elif swallowing in {"difficulty", "painful"}:
            result["presence"] = "yes"
            details["swallowing_type"] = swallowing

    elif category == "nutrition":
        nutrition = str(extracted.get("nutrition") or "").strip().lower()
        if nutrition == "normal":
            result["presence"] = "no" if topic == "nutrition" else "unknown"
        elif nutrition:
            result["presence"] = "yes"
            if topic == "nutrition":
                details["intake_level"] = nutrition
            if topic == "weight" and nutrition:
                details["nutrition_status"] = nutrition

    elif category == "oral_symptoms":
        oral = str(extracted.get("oral_symptoms") or "").strip()
        if oral:
            result["presence"] = "yes"
            if topic == "oral_sores":
                details["oral_symptoms"] = oral
            elif topic == "dry_mouth":
                details["dry_mouth_details"] = oral
            elif topic == "mucus":
                details["mucus_details"] = oral

    elif category == "gi_symptoms":
        gi = str(extracted.get("gi_symptoms") or "").strip()
        if gi:
            result["presence"] = "yes"
            details["gi_symptoms"] = gi

    elif category == "fatigue":
        fatigue = normalize_fatigue_value(extracted.get("fatigue"))
        if topic == "fatigue":
            if fatigue == "none":
                result["presence"] = "no"
            elif fatigue:
                result["presence"] = "yes"
                details["fatigue_level"] = fatigue
        elif topic == "sleep":
            if fatigue:
                details["fatigue_context"] = fatigue

    elif category == "mood":
        mood = str(extracted.get("mood") or "").strip().lower()
        if mood == "normal":
            result["presence"] = "no"
        elif mood:
            result["presence"] = "yes"
            details["emotional_state"] = mood

    elif category == "breathing":
        breathing = str(extracted.get("breathing") or "").strip().lower()
        if breathing == "normal":
            result["presence"] = "no"
        elif breathing == "difficulty":
            result["presence"] = "yes"
            details["breathing"] = breathing

    else:
        other = str(extracted.get("other") or "").strip()
        if other:
            result["presence"] = "yes"
            if topic == "medications":
                details["medications_list"] = other
            elif topic == "general":
                details["overall_note"] = other

    if topic == "weight":
        match = re.search(r"\b(\d{2,3}(?:\.\d+)?)\b", patient_message)
        if match:
            details["weight"] = match.group(1)
            result["presence"] = "yes"

    if topic == "general":
        match = re.search(r"\b([0-9]|10)\b", patient_message)
        if match:
            details["overall_score"] = match.group(1)

    result["details"] = details
    return result


def asked_questions_for_topic(topic: str) -> list[str]:
    return [
        message["content"]
        for message in st.session_state.messages
        if message.get("role") == "assistant" and message.get("topic") == topic
    ]


def format_report_from_agent(report_data: dict[str, Any], collected_data: dict[str, Any]) -> str:
    lines = ["## Structured Clinical Report", ""]
    if st.session_state.get("previous_visit_history", "").strip():
        lines.extend(["### Previous Visit History", st.session_state["previous_visit_history"].strip(), ""])

    section_order = ["pain", "nutrition", "swallowing", "fatigue", "other"]
    labels = {
        "pain": "Pain",
        "nutrition": "Nutrition",
        "swallowing": "Swallowing",
        "fatigue": "Fatigue",
        "other": "Other",
    }
    for key in section_order:
        value = report_data.get(key)
        if value:
            lines.append(f"### {labels[key]}")
            lines.append(f"- {value}")
            lines.append("")

    priority = report_data.get("overall_priority")
    if priority:
        lines.append("### Overall Priority")
        lines.append(f"- {priority}")
        lines.append("")

    if len(lines) <= 3:
        return build_fallback_report(
            collected_data,
            st.session_state.get("previous_visit_history", ""),
            st.session_state.get("previous_visit_topics", {}),
        )
    return "\n".join(lines).strip()


def symptom_extraction_agent(
    patient_message: str,
    topic: str,
    question: str,
    topic_data: dict[str, Any],
) -> dict[str, Any]:
    fallback = {
        "pain": {"present": None, "location": None, "severity": None, "timing": None},
        "swallowing": None,
        "nutrition": None,
        "oral_symptoms": None,
        "gi_symptoms": None,
        "fatigue": None,
        "mood": None,
        "breathing": None,
        "other": None,
    }
    system_prompt = f"""
🟢 1. SYMPTOM EXTRACTION AGENT (STRONG)
You are a clinical information extraction agent.

TASK:
Convert patient natural language into structured clinical data.

INPUT:
Patient message:
"{patient_message}"

CONSTRAINTS:
- Extract ONLY what is explicitly stated
- Do NOT infer medical facts not mentioned
- Use null if unknown
- Normalize terms (e.g., "hurts a lot" → severity: "high")
- Map symptoms to categories in knowledge_base

CATEGORIES:
pain, swallowing, nutrition, oral_symptoms, gi_symptoms, fatigue, mood, breathing, other

OUTPUT FORMAT (STRICT JSON):
{{
  "pain": {{
    "present": true/false/null,
    "location": "string or null",
    "severity": "low|medium|high|null",
    "timing": "constant|intermittent|null"
  }},
  "swallowing": "normal|difficulty|painful|null",
  "nutrition": "normal|reduced|liquid_only|tube|null",
  "oral_symptoms": "string or null",
  "gi_symptoms": "string or null",
  "fatigue": "none|mild|moderate|severe|null",
  "mood": "normal|anxious|depressed|null",
  "breathing": "normal|difficulty|null",
  "other": "string or null"
}}
"""
    payload = {
        "topic": topic,
        "question": question,
        "knowledge_base": KNOWLEDGE_BASE,
        "existing_topic_data": topic_data,
        "previous_visit_topic_summary": st.session_state.previous_visit_topics.get(topic, ""),
        "previous_visit_history": st.session_state.get("previous_visit_history", ""),
        "user_input": patient_message,
    }
    extracted = call_json_agent(system_prompt, payload, fallback)
    if not isinstance(extracted, dict):
        extracted = fallback
    result = topic_result_from_extraction(topic, extracted, patient_message)
    result["summary"] = patient_message
    return result


def clinical_importance_agent(topic: str, extracted: dict[str, Any], topic_data: dict[str, Any]) -> dict[str, Any]:
    fallback = {
        "importance": get_topic_config(topic).get("priority", "medium"),
        "needs_followup": False,
        "missing_fields": [],
    }
    system_prompt = """
🟡 2. CLINICAL IMPORTANCE AGENT (STRONG)
You are a clinical prioritization agent.

TASK:
Assess how important the patient’s symptoms are for a doctor AND determine if follow-up is required.

INPUT:
- extracted_symptoms (JSON)
- knowledge_base (JSON)

CLINICAL PRIORITY RULES:
HIGH:
- swallowing difficulty
- reduced nutrition / not eating
- severe pain
- breathing issues

MEDIUM:
- moderate fatigue
- oral symptoms affecting function
- persistent symptoms

LOW:
- mild or vague symptoms

FOLLOW-UP RULES:
- required_fields missing → needs_followup = true
- severity unknown for high-priority symptom → needs_followup = true
- vague description → needs_followup = true

CONSTRAINTS:
- Be conservative (prefer follow-up if unsure)
- Do NOT hallucinate missing info

OUTPUT (STRICT JSON):
{
  "importance": "low|medium|high",
  "needs_followup": true|false,
  "missing_fields": ["field1", "field2"]
}
"""
    payload = {
        "current_topic": topic,
        "extracted_symptoms": extracted,
        "knowledge_base": KNOWLEDGE_BASE,
        "collected_data": topic_data,
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
    fallback = {"follow_up_question": fallback_question, "target_field": missing_fields[0] if missing_fields else None}
    system_prompt = """
🔵 3. FOLLOW-UP AGENT (STRONG, KB-GROUNDED)
You are a clinical follow-up question generator.

INPUT:
- current_topic
- missing_fields (list)
- knowledge_base (JSON)
- conversation_history

TASK:
Select EXACTLY ONE follow-up question from knowledge_base[current_topic].followups

RULES:
1. ONLY select from knowledge base (no new questions)
2. Prioritize questions that fill missing_fields
3. Avoid repeating any question in conversation_history
4. Keep question simple (low health literacy)
5. If no useful follow-up exists → return null

OUTPUT (STRICT JSON):
{
  "follow_up_question": "string or null",
  "target_field": "field name or null"
}
"""
    payload = {
        "current_topic": topic,
        "missing_fields": missing_fields,
        "knowledge_base": KNOWLEDGE_BASE,
        "conversation_history": asked_questions_for_topic(topic),
    }
    result = call_json_agent(system_prompt, payload, fallback)
    question = result.get("follow_up_question", "")
    if question not in candidate_followups:
        question = fallback_question
    return {"question": question, "target_field": result.get("target_field")}


def patient_experience_agent(history: list[dict[str, str]]) -> dict[str, Any]:
    fallback = {"fatigue_level": "medium", "should_limit_questions": False}
    system_prompt = """
🟣 4. PATIENT EXPERIENCE AGENT (STRONG)
You are a patient interaction monitoring agent.

TASK:
Assess patient burden and risk of fatigue.

INPUT:
- conversation_history (list of messages)
- last_user_message

RULES:
fatigue_level = high if:
- many short responses
- repeated "I don’t know"
- conversation > 10 turns

fatigue_level = medium if:
- answers getting shorter
- hesitation

fatigue_level = low otherwise

CONSTRAINT:
Be conservative; do NOT overestimate fatigue.

OUTPUT (STRICT JSON):
{
  "fatigue_level": "low|medium|high",
  "should_limit_questions": true|false
}
"""
    user_messages = [m for m in history if m["role"] == "user"]
    payload = {"conversation_history": history[-12:], "last_user_message": user_messages[-1]["content"] if user_messages else ""}
    return call_json_agent(system_prompt, payload, fallback)


def safety_agent(extracted: dict[str, Any]) -> dict[str, Any]:
    fallback = {"alert": False, "reason": "", "recommended_action": "continue"}
    system_prompt = """
⚫ 6. SAFETY AGENT (STRONG)
You are a clinical safety monitoring agent.

TASK:
Detect urgent or dangerous symptoms.

INPUT:
- extracted_symptoms

ALERT CONDITIONS:
- cannot eat or drink
- severe pain
- breathing difficulty
- rapid worsening

OUTPUT (STRICT JSON):
{
  "alert": true|false,
  "reason": "string",
  "recommended_action": "continue|flag_for_doctor|urgent_attention"
}
"""
    payload = {"extracted_symptoms": extracted}
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
    followup_candidate: dict[str, Any],
    experience: dict[str, Any],
    safety: dict[str, Any],
    topic_state: dict[str, Any],
    topic_data: dict[str, Any],
) -> dict[str, Any]:
    fallback = {"action": "next_topic", "next_topic": None, "question": None, "reason": "Fallback orchestration"}
    system_prompt = """
🧠 0. ORCHESTRATOR (STRONG VERSION)
You are the Orchestrator of a clinical symptom-reporting system for head and neck cancer patients.

Your role is to decide the NEXT ACTION using outputs from multiple agents.

INPUTS:
- extracted_symptoms (JSON)
- importance_assessment (JSON)
- followup_candidate (JSON)
- patient_state (JSON)
- current_topic (string)
- collected_data (JSON)
- knowledge_base (JSON)

PRIMARY GOAL:
Collect clinically useful, concise, and complete information for a doctor.

STRICT DECISION RULES:
1. If importance == "high" AND required_fields missing → action = "follow_up"
2. If importance == "medium" AND missing key detail → action = "follow_up"
3. If patient_state.should_limit_questions == true → avoid follow_up unless critical
4. If all required_fields for current_topic are filled → action = "next_topic"
5. If all high-priority topics are completed → action = "finish"

TOPIC PRIORITY ORDER:
Use knowledge_base priority (high → medium → low)

CONSTRAINTS:
- Ask at most ONE follow-up question
- Do NOT repeat previously asked questions
- Do NOT invent new questions (must come from KB)
- Prefer moving forward over over-questioning

OUTPUT FORMAT (STRICT JSON ONLY):
{
  "action": "follow_up" | "next_topic" | "finish",
  "next_topic": "topic_name or null",
  "question": "string or null",
  "reason": "short explanation"
}
"""
    payload = {
        "extracted_symptoms": extracted,
        "importance_assessment": importance,
        "followup_candidate": followup_candidate,
        "patient_state": experience,
        "current_topic": topic,
        "collected_data": topic_data,
        "knowledge_base": KNOWLEDGE_BASE,
        "required_fields": required_fields_for_topic(topic),
        "safety_assessment": safety,
    }
    result = call_json_agent(system_prompt, payload, fallback)
    action = result.get("action", "next_topic")
    if action not in {"follow_up", "next_topic", "finish"}:
        action = "next_topic"
    return {
        "action": action,
        "next_topic": result.get("next_topic"),
        "question": result.get("question"),
        "reason": result.get("reason", fallback["reason"]),
    }


def report_generator_agent(collected_data: dict[str, Any], history: list[dict[str, str]]) -> str:
    fallback_report = build_fallback_report(
        collected_data,
        st.session_state.get("previous_visit_history", ""),
        st.session_state.get("previous_visit_topics", {}),
    )
    client = get_openai_client()
    if client is None:
        return fallback_report

    fallback_data = {
        "pain": None,
        "nutrition": None,
        "swallowing": None,
        "fatigue": None,
        "other": None,
        "overall_priority": "medium",
    }
    system_prompt = """
🟠 5. REPORT AGENT (STRONG, DOCTOR-FOCUSED)
You are a clinical report generator.

TASK:
Convert collected patient data into a structured report for a physician.

INPUT:
- collected_data (JSON)
- knowledge_base (JSON)

GOALS:
- concise
- clinically relevant
- no unnecessary text

RULES:
- Summarize by category (pain, nutrition, swallowing, etc.)
- Include severity and functional impact
- Highlight high-priority issues

OUTPUT (STRICT JSON):
{
  "pain": "summary or null",
  "nutrition": "summary or null",
  "swallowing": "summary or null",
  "fatigue": "summary or null",
  "other": "summary or null",
  "overall_priority": "low|medium|high"
}
"""
    payload = {
        "collected_data": collected_data,
        "knowledge_base": KNOWLEDGE_BASE,
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
        parsed = extract_json(response.output_text, fallback_data)
        if isinstance(parsed, dict):
            return format_report_from_agent(parsed, collected_data)
        return fallback_report
    except Exception as exc:
        print(f"[report-error] {exc}")
        return fallback_report


def build_fallback_report(
    collected_data: dict[str, Any],
    previous_visit_history: str = "",
    previous_visit_topics: dict[str, str] | None = None,
) -> str:
    previous_visit_topics = previous_visit_topics or {}
    lines = ["## Structured Clinical Report", ""]
    if previous_visit_history.strip():
        lines.extend(
            [
                "### Previous Visit History",
                previous_visit_history.strip(),
                "",
            ]
        )
    for topic in TOPIC_ORDER:
        topic_data = collected_data.get(topic, {})
        previous_summary = previous_visit_topics.get(topic, "").strip()
        if not topic_data and not previous_summary:
            continue
        lines.append(f"### {topic_label(topic)}")
        if previous_summary:
            lines.append(f"- **Previous Visit:** {previous_summary}")
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


def previous_visit_agent(previous_visit_history: str) -> dict[str, str]:
    history = previous_visit_history.strip()
    if not history:
        return {}

    fallback = {topic: "" for topic in TOPIC_ORDER}
    client = get_openai_client()
    if client is None:
        return fallback

    system_prompt = f"""
You are a clinical summarization helper.
Summarize the previous visit history into short topic-specific snippets for this head and neck symptom interview.

Return JSON only with exactly these keys:
{json.dumps({topic: "" for topic in TOPIC_ORDER}, ensure_ascii=True)}

Rules:
- Each value should be a short phrase or sentence fragment.
- Leave a value as an empty string when the previous visit history does not mention that topic.
- Do not invent data.
- Use concise clinician-friendly wording.
"""
    try:
        response = client.responses.create(
            model=st.session_state.get("model_name", DEFAULT_MODEL),
            temperature=0,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": history},
            ],
        )
        parsed = extract_json(response.output_text, fallback)
        if isinstance(parsed, dict):
            return {topic: str(parsed.get(topic, "") or "") for topic in TOPIC_ORDER}
    except Exception as exc:
        print(f"[previous-visit-error] {exc}")
    return fallback


def refresh_previous_visit_topics() -> None:
    history = st.session_state.get("previous_visit_history", "").strip()
    cache_key = "_previous_visit_cache"
    if not history:
        st.session_state.previous_visit_topics = {}
        st.session_state[cache_key] = ""
        return
    if st.session_state.get(cache_key) == history:
        return
    st.session_state.previous_visit_topics = previous_visit_agent(history)
    st.session_state[cache_key] = history


def _secret(name: str, default: Any = None) -> Any:
    return st.secrets.get(name, default)


def _init_sheets() -> None:
    global _sheet, _sheet_error
    if _sheet is not None or _sheet_error is not None:
        return

    if gspread is None or Credentials is None:
        _sheet_error = "Google Sheets dependencies are not available."
        return

    try:
        service_account_info = _secret("gcp_service_account")
        gsheet_id = _secret("gsheet_id")
        if not service_account_info or not gsheet_id:
            _sheet_error = "Missing `gcp_service_account` or `gsheet_id` in Streamlit secrets."
            return

        creds = Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        book = gspread.authorize(creds).open_by_key(gsheet_id)
        try:
            ws = book.worksheet("ChatReport")
        except Exception:
            ws = book.add_worksheet(title="ChatReport", rows=2000, cols=6)
            ws.append_row(["timestamp", "patient_name", "selected_section", "all_data_json", "report", "previous_visit_history"])
        _sheet = ws
    except Exception as exc:
        _sheet_error = str(exc)


def save_to_sheet(patient_name: str, selected_section: str, all_data: dict[str, Any], report: str = "") -> bool:
    _init_sheets()
    if _sheet is None:
        st.session_state.sheet_status = f"Could not connect to Google Sheets: {_sheet_error}"
        return False

    try:
        _sheet.append_row(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                patient_name,
                selected_section,
                json.dumps(all_data, ensure_ascii=False),
                report,
                st.session_state.get("previous_visit_history", ""),
            ]
        )
        st.session_state.sheet_status = "Saved to Google Sheets."
        return True
    except Exception as exc:
        st.session_state.sheet_status = f"Failed to save to Google Sheets: {exc}"
        return False


def topic_status_text(topic: str) -> str:
    state = st.session_state.topic_states[topic]
    if state["completed"]:
        return "Complete"
    if state["main_asked"]:
        return "In progress"
    return "Not started"


def render_topic_sections() -> None:
    st.markdown("### Topic Sections")
    for topic in TOPIC_ORDER:
        state = st.session_state.topic_states[topic]
        current_data = st.session_state.collected_data.get(topic, {})
        current_summary = current_data.get("summary", state.get("summary", ""))
        previous_summary = st.session_state.previous_visit_topics.get(topic, "").strip()
        priority = get_topic_config(topic).get("priority", "routine").title()
        st.markdown(f"#### {topic_label(topic)} • {topic_status_text(topic)}")
        st.markdown(f"*Priority: {priority}*")
        if previous_summary:
            st.markdown(f"**Previous visit:** {previous_summary}")
        else:
            st.markdown("**Previous visit:** No prior summary recorded for this topic.")
        if current_summary:
            st.markdown(f"**Current visit:** {current_summary}")
        else:
            st.markdown("**Current visit:** No details captured yet.")
        required = required_fields_for_topic(topic)
        if required:
            present = [field for field in required if current_data.get(field)]
            missing = [field for field in required if not current_data.get(field)]
            st.markdown(f"**Required details captured:** {', '.join(present) if present else 'None yet'}")
            st.markdown(f"**Still missing:** {', '.join(missing) if missing else 'None'}")
        st.markdown("---")


def render_selected_section() -> None:
    section_key = st.session_state.selected_section
    st.markdown(f"### {section_label(section_key)}")
    st.markdown(SECTION_INTROS[section_key])

    for topic in SECTION_TOPICS[section_key]:
        state = st.session_state.topic_states[topic]
        current_data = st.session_state.collected_data.get(topic, {})
        current_summary = current_data.get("summary", state.get("summary", ""))
        previous_summary = st.session_state.previous_visit_topics.get(topic, "").strip()
        subtitle = f"{topic_label(topic)} • {topic_status_text(topic)}"
        st.markdown(f"#### {subtitle}")
        if previous_summary:
            st.markdown(f"**Previous visit:** {previous_summary}")
        else:
            st.markdown("**Previous visit:** No prior summary recorded.")
        if current_summary:
            st.markdown(f"**Current visit:** {current_summary}")
        else:
            st.markdown("**Current visit:** No details captured yet.")
        required = required_fields_for_topic(topic)
        if required:
            present = [field for field in required if current_data.get(field)]
            missing = [field for field in required if not current_data.get(field)]
            st.markdown(f"**Captured:** {', '.join(present) if present else 'None yet'}")
            st.markdown(f"**Missing:** {', '.join(missing) if missing else 'None'}")
        st.markdown("---")


def ask_assistant_question(question: str) -> None:
    st.session_state.current_question = question
    current_topic = st.session_state.current_topic
    st.session_state.topic_states[current_topic]["current_question"] = question
    add_message("assistant", question, topic=current_topic)


def section_status(section_key: str) -> str:
    topics = SECTION_TOPICS[section_key]
    completed_count = sum(1 for topic in topics if st.session_state.topic_states[topic]["completed"])
    started_count = sum(1 for topic in topics if st.session_state.topic_states[topic]["main_asked"])
    if completed_count == len(topics):
        return "Complete"
    if started_count > 0:
        return "In progress"
    return "Not started"


def ensure_section_started(section_key: str) -> None:
    active_topic = get_active_topic_for_section(section_key)
    if active_topic is None:
        return

    set_active_topic(active_topic)
    if not st.session_state.section_states[section_key]["started"]:
        add_message("assistant", SECTION_INTROS[section_key], topic=active_topic)
        st.session_state.section_states[section_key]["started"] = True

    if not st.session_state.topic_states[active_topic]["main_asked"]:
        st.session_state.topic_states[active_topic]["main_asked"] = True
        ask_assistant_question(main_question_for(active_topic))


def move_to_next_topic() -> None:
    current_topic = st.session_state.current_topic
    st.session_state.topic_states[current_topic]["completed"] = True
    next_topic = get_next_unfinished_topic()
    if next_topic is None:
        finish_chat()
        return
    set_active_topic(next_topic)
    st.session_state.topic_states[next_topic]["main_asked"] = True
    ask_assistant_question(main_question_for(next_topic))


def should_force_followup(topic: str, merged_data: dict[str, Any], missing_fields: list[str]) -> bool:
    if not missing_fields:
        return False

    presence = normalize_presence(merged_data.get("presence"))
    question_type = question_type_for_topic(topic)

    if presence == "yes":
        return True

    if question_type in {"scale", "numeric"}:
        return True

    return False


def finish_chat() -> None:
    st.session_state.finished = True
    report = report_generator_agent(st.session_state.collected_data, st.session_state.messages)
    st.session_state.final_report = report
    add_message(
        "assistant",
        "Thank you. I have enough information now and generated your structured symptom report below.",
        topic=st.session_state.current_topic,
    )


def start_chat() -> None:
    if st.session_state.messages:
        return
    welcome = (
        "Hello, I’m ChatReport. I’ll ask a sequence of symptom questions to build a concise clinical report for your doctor."
    )
    add_message("assistant", welcome)
    first_topic = TOPIC_ORDER[0]
    set_active_topic(first_topic)
    st.session_state.topic_states[first_topic]["main_asked"] = True
    ask_assistant_question(main_question_for(first_topic))


def process_turn(patient_message: str) -> None:
    if st.session_state.finished:
        return

    topic = st.session_state.current_topic
    topic_state = st.session_state.topic_states[topic]
    current_question = st.session_state.current_question

    add_message("user", patient_message, topic=topic)

    extracted = symptom_extraction_agent(
        patient_message=patient_message,
        topic=topic,
        question=current_question,
        topic_data=deepcopy(get_topic_data(topic)),
    )
    merged_data = merge_topic_data(topic, extracted)
    topic_state["main_answered"] = True
    topic_state["presence"] = effective_presence(merged_data, extracted)
    topic_state["summary"] = extracted.get("summary", "")
    if topic_state["presence"] != "unknown":
        merged_data["presence"] = topic_state["presence"]

    inferred_missing = topic_missing_fields(topic, merged_data)
    importance = clinical_importance_agent(topic, extracted, deepcopy(merged_data))
    llm_missing = [item for item in importance.get("missing_fields", []) if isinstance(item, str)]
    if inferred_missing:
        importance["missing_fields"] = sorted(set(llm_missing + inferred_missing))
    else:
        importance["missing_fields"] = llm_missing

    if should_force_followup(topic, merged_data, importance["missing_fields"]):
        importance["needs_followup"] = True
    elif topic_state["presence"] == "no":
        importance["needs_followup"] = False

    experience = patient_experience_agent(st.session_state.messages)
    safety = safety_agent(extracted)
    followup = follow_up_agent(
        topic=topic,
        missing_fields=importance.get("missing_fields", []),
        asked_followups=topic_state["asked_followups"],
        topic_data=deepcopy(merged_data),
    )
    orchestration = orchestrator_agent(
        topic=topic,
        extracted=extracted,
        importance=importance,
        followup_candidate=followup,
        experience=experience,
        safety=safety,
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
                "safety": safety,
                "followup": followup,
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
    should_limit_questions = bool(experience.get("should_limit_questions"))

    if action == "finish":
        finish_chat()
        return

    if action == "follow_up" and remaining_followups:
        if should_limit_questions and safety.get("recommended_action") == "continue" and importance.get("importance") != "high":
            move_to_next_topic()
            return

        question = (orchestration.get("question") or followup.get("question") or "").strip()
        if question:
            topic_state["asked_followups"].append(question)
            ask_assistant_question(question)
            return

    if should_force_followup(topic, merged_data, importance.get("missing_fields", [])) and remaining_followups:
        question = (followup.get("question") or "").strip()
        if question:
            topic_state["asked_followups"].append(question)
            ask_assistant_question(question)
            return

    move_to_next_topic()


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## ChatReport")
        st.markdown("Clinical symptom assistant")
        st.session_state.patient_name = st.text_input(
            "Patient Name",
            value=st.session_state.get("patient_name", ""),
        )
        st.session_state.previous_visit_history = st.text_area(
            "Previous Visit History",
            value=st.session_state.get("previous_visit_history", ""),
            height=180,
            help="Optional. Paste a summary from the previous visit so the chatbot can show topic-by-topic context.",
        )

        st.markdown("---")
        st.markdown("### Topics")
        completed_sections = sum(1 for section_key, _ in SECTION_ORDER if section_status(section_key) == "Complete")
        st.markdown(f"{completed_sections}/{len(SECTION_ORDER)} sections complete")

        for section_key, label in SECTION_ORDER:
            status = section_status(section_key)
            marker = "▶ " if st.session_state.selected_section == section_key else ""
            icon = {"Complete": "✅", "In progress": "🔵", "Not started": "⚪"}[status]
            button_label = f"{marker}{icon} {label}"
            if st.button(button_label, key=f"section_{section_key}", use_container_width=True):
                st.session_state.selected_section = section_key
                ensure_section_started(section_key)
                st.rerun()

        st.markdown("---")
        st.markdown("### Spreadsheet")
        if st.button("Save To Google Sheets", use_container_width=True):
            report = st.session_state.final_report or build_fallback_report(
                st.session_state.collected_data,
                st.session_state.get("previous_visit_history", ""),
                st.session_state.get("previous_visit_topics", {}),
            )
            save_to_sheet(
                patient_name=st.session_state.get("patient_name", ""),
                selected_section=st.session_state.get("selected_section", ""),
                all_data=st.session_state.collected_data,
                report=report,
            )
        if st.session_state.get("sheet_status"):
            st.markdown(st.session_state.sheet_status)

        if st.button("Reset Chat"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def render_chat() -> None:
    st.markdown("# ChatReport")
    st.markdown("Multi-agent clinical chatbot for head and neck cancer symptom reporting")
    st.markdown(f"**Current topic:** {topic_label(st.session_state.current_topic)}")
    st.markdown("---")

    if st.session_state.get("previous_visit_history", "").strip():
        st.markdown("### Previous Visit History")
        st.markdown(st.session_state.previous_visit_history)
        st.markdown("---")
    for message in st.session_state.messages:
        speaker = "Assistant" if message["role"] == "assistant" else "Patient"
        st.markdown(f"**{speaker}:** {message['content']}")

    if st.session_state.final_report:
        st.markdown("### Final Report")
        st.info(st.session_state.final_report)


def main() -> None:
    init_session_state()
    st.markdown("### Visit Information")
    st.session_state.patient_name = st.text_input(
        "Patient Name",
        value=st.session_state.get("patient_name", ""),
    )
    st.session_state.previous_visit_history = st.text_area(
        "Previous Visit History",
        value=st.session_state.get("previous_visit_history", ""),
        height=140,
    )
    refresh_previous_visit_topics()
    start_chat()
    render_chat()

    if st.session_state.finished:
        return

    st.markdown("### Your Response")
    prompt = st.text_area(
        f"Type your answer for {topic_label(st.session_state.current_topic)}",
        value=st.session_state.get("pending_user_input", ""),
        key="pending_user_input",
        height=100,
    )
    if st.button("Send Response") and prompt.strip():
        process_turn(prompt)
        st.session_state.pending_user_input = ""
        st.rerun()

    if st.button("Reset Chat"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    main()
