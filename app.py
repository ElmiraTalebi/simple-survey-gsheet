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
You are the Symptom Extraction Agent inside ChatReport, a clinical intake assistant for head and neck cancer symptom reporting.
Your job is to read the patient's latest reply and convert only the information relevant to the CURRENT TOPIC into structured JSON.

CURRENT TOPIC: {topic}

Return JSON only. No prose, no markdown, no code fences.
Use exactly this schema:
{{
  "topic": "{topic}",
  "presence": "yes" | "no" | "unknown",
  "details": {{
    "field_name": "short extracted value"
  }},
  "summary": "1 short clinical summary sentence"
}}

Extraction guidance:
- Focus only on the current topic, even if the patient mentions other symptoms.
- Use the current question, topic configuration, and existing topic data as context.
- Capture explicit facts only. Do not guess, infer, or fill missing values.
- If the patient denies the symptom, set "presence" to "no".
- If the patient confirms the symptom, set "presence" to "yes".
- If the answer is ambiguous, incomplete, or not clearly related, set "presence" to "unknown".
- Put clinically useful details into "details" using concise field names such as "location", "severity", "timing", "duration", "impact", "medication_help", "intake_level", "barriers", "weight", "emotional_state", or similar when supported by the patient message.
- Preserve numbers exactly when given, especially 0-10 ratings, counts, doses, and weights.
- If the patient gives multiple details in one answer, capture all of them.
- The "summary" should be brief, factual, and suitable for a clinician.

Important:
- Do not output null values, placeholders, explanations, or extra keys beyond the schema.
- If no concrete detail is provided, leave "details" as an empty object.
"""
    payload = {
        "topic": topic,
        "question": question,
        "topic_config": config,
        "existing_topic_data": topic_data,
        "previous_visit_topic_summary": st.session_state.previous_visit_topics.get(topic, ""),
        "previous_visit_history": st.session_state.get("previous_visit_history", ""),
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
You are the Clinical Importance Agent for a symptom-reporting workflow in head and neck oncology.
Your task is to assess how clinically important the CURRENT TOPIC is right now and whether more clarification is needed before moving on.

Return JSON only. No prose outside JSON.
Use exactly this schema:
{
  "importance_level": "low" | "medium" | "high" | "urgent",
  "follow_up_needed": true | false,
  "missing_fields": ["field1", "field2"],
  "reason": "short rationale"
}

Decision framework:
- Use the topic priority, extracted symptom status, existing topic data, and required fields.
- If the symptom is absent or clearly denied, importance usually follows the baseline topic priority but follow_up_needed is usually false.
- If the symptom is present and required clinical fields are still missing, follow_up_needed should usually be true.
- Escalate to "urgent" only for potentially dangerous airway, breathing, bleeding, severe swallowing, or otherwise alarming information explicitly stated by the patient.
- If enough information is already present for the current topic, set follow_up_needed to false.
- missing_fields must contain only fields that are actually missing and clinically useful for the current topic.
- Do not request missing_fields that are not relevant to the current topic.
- The reason should be short, specific, and operational, such as "Pain present but severity missing" or "Symptom denied clearly".

Important:
- Be conservative and factual.
- Do not invent urgency.
- Do not recommend treatment or emergency action in this step.
"""
    payload = {
        "topic": topic,
        "topic_config": config,
        "extracted": extracted,
        "topic_data": topic_data,
        "required_fields": required_fields_for_topic(topic),
        "previous_visit_topic_summary": st.session_state.previous_visit_topics.get(topic, ""),
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
Your task is to choose the SINGLE best next follow-up question for the current topic from the approved knowledge-base questions.

Return JSON only:
{
  "question": "one question"
}

Rules:
- You must choose exactly one question from candidate_followups, or return an empty string if none is appropriate.
- Never write a new question. Never paraphrase. Never combine two questions.
- Prioritize the question most likely to fill a missing required field first.
- If required fields are already covered, choose the follow-up that best clarifies severity, function, timing, or patient impact.
- Avoid repeating what appears already answered in topic_data.
- Avoid choosing a question that is semantically redundant with a previously asked follow-up.
- If candidate_followups is empty, return an empty string.
- Output only the JSON object.
"""
    payload = {
        "topic": topic,
        "missing_fields": missing_fields,
        "candidate_followups": candidate_followups,
        "topic_data": topic_data,
        "previous_visit_topic_summary": st.session_state.previous_visit_topics.get(topic, ""),
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
Estimate conversational fatigue from the patient's interaction style, not medical fatigue.

Return JSON only:
{
  "fatigue_level": "low" | "medium" | "high",
  "engagement_note": "short note"
}

Interpretation rules:
- "low": patient is engaged, answering normally, and not showing friction.
- "medium": answers are shorter, somewhat repetitive, delayed, or the conversation is getting longer.
- "high": the patient appears frustrated, exhausted by questioning, minimally responsive, or repeatedly gives very brief answers.
- Consider brevity, repetition, confusion, tone, and total conversation burden.
- Do not infer physical tiredness or cancer-related fatigue unless it affects conversational burden directly.
- Keep the engagement_note short and specific, such as "Answers remain detailed" or "Responses are increasingly brief".
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
You are the Orchestrator Agent for ChatReport.
You decide whether the chatbot should ask one more follow-up on the current topic, move to the next topic, or finish the interview.

Return JSON only:
{
  "action": "ask_followup" | "next_topic" | "finish",
  "reason": "short rationale"
}

Decision rules:
- Choose "ask_followup" when the symptom is present or unclear AND clinically useful information is still missing AND a follow-up question is available.
- Choose "next_topic" when the topic is adequately covered, clearly denied, or no useful follow-up remains.
- Choose "finish" only when the interview is complete and no meaningful topics remain.
- Respect conversation burden: if patient conversational fatigue is high, prefer moving on unless the current topic is high priority and still insufficiently characterized.
- High-priority active symptoms should usually get at least one clarifying follow-up if important fields are missing.
- Do not ask repeated or low-yield follow-ups.
- Use the reason field to explain the operational logic briefly, such as "Pain present and severity missing" or "Symptom denied; proceed".
"""
    payload = {
        "topic": topic,
        "topic_config": config,
        "extracted": extracted,
        "importance": importance,
        "experience": experience,
        "topic_state": topic_state,
        "topic_data": topic_data,
        "previous_visit_topic_summary": st.session_state.previous_visit_topics.get(topic, ""),
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
    fallback_report = build_fallback_report(
        collected_data,
        st.session_state.get("previous_visit_history", ""),
        st.session_state.get("previous_visit_topics", {}),
    )
    client = get_openai_client()
    if client is None:
        return fallback_report

    system_prompt = """
You are the Report Generator Agent for ChatReport.
Transform the collected structured interview data into a concise, clinician-friendly report for head and neck cancer symptom follow-up.

Return JSON only:
{
  "report_markdown": "markdown report"
}

Report requirements:
- Produce clean markdown.
- Include a short title and clearly separated sections.
- Summarize the most important positive symptoms first.
- Include relevant negatives when the patient clearly denied symptoms.
- Mention functional impact such as eating, drinking, swallowing, breathing, sleep, energy, and emotional state when available.
- Include medications and any helpful details about benefit or side effects if available.
- End with a brief overall clinical summary.
- Keep the tone factual, concise, and suitable for clinical review.

Do not:
- Invent data.
- Add treatment plans, diagnoses, or medical advice.
- Mention that an AI generated the report.
"""
    payload = {
        "collected_data": collected_data,
        "conversation_history": history,
        "previous_visit_history": st.session_state.get("previous_visit_history", ""),
        "previous_visit_topics": st.session_state.get("previous_visit_topics", {}),
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
        with st.expander(f"{topic_label(topic)} • {topic_status_text(topic)}", expanded=topic == st.session_state.current_topic):
            st.caption(f"Priority: {priority}")
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


def render_selected_section() -> None:
    section_key = st.session_state.selected_section
    st.markdown(f"### {section_label(section_key)}")
    st.caption(SECTION_INTROS[section_key])

    for topic in SECTION_TOPICS[section_key]:
        state = st.session_state.topic_states[topic]
        current_data = st.session_state.collected_data.get(topic, {})
        current_summary = current_data.get("summary", state.get("summary", ""))
        previous_summary = st.session_state.previous_visit_topics.get(topic, "").strip()
        subtitle = f"{topic_label(topic)} • {topic_status_text(topic)}"
        with st.expander(subtitle, expanded=topic == st.session_state.current_topic):
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
    current_section = section_for_topic(current_topic)
    next_topic = get_next_unfinished_topic(current_section)
    if next_topic is not None:
        set_active_topic(next_topic)
        st.session_state.topic_states[next_topic]["main_asked"] = True
        ask_assistant_question(main_question_for(next_topic))
        return

    add_message(
        "assistant",
        f"{section_label(current_section)} is complete. You can choose another topic from the sidebar.",
        topic=current_topic,
    )

    if not has_remaining_topics():
        finish_chat()


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
        "Hello, I’m ChatReport. Please choose one of the symptom topics from the sidebar and answer the questions for that section."
    )
    add_message("assistant", welcome)
    ensure_section_started(st.session_state.selected_section)


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
        importance["follow_up_needed"] = True
    elif topic_state["presence"] == "no":
        importance["follow_up_needed"] = False

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

    if should_force_followup(topic, merged_data, importance.get("missing_fields", [])) and remaining_followups:
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
        st.header("ChatReport")
        st.caption("Clinical symptom assistant")
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
        st.subheader("Topics")
        completed_sections = sum(1 for section_key, _ in SECTION_ORDER if section_status(section_key) == "Complete")
        st.progress(completed_sections / len(SECTION_ORDER))
        st.caption(f"{completed_sections}/{len(SECTION_ORDER)} sections complete")

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
        st.subheader("Spreadsheet")
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
            st.caption(st.session_state.sheet_status)

        if st.button("Reset Chat"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def render_chat() -> None:
    st.title("ChatReport")
    st.caption("Multi-agent clinical chatbot for head and neck cancer symptom reporting")

    if st.session_state.get("previous_visit_history", "").strip():
        with st.expander("Previous Visit History", expanded=False):
            st.markdown(st.session_state.previous_visit_history)

    render_selected_section()

    selected_section = st.session_state.selected_section
    for message in st.session_state.messages:
        if message.get("section") not in {None, selected_section}:
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.final_report:
        st.markdown("### Final Report")
        st.info(st.session_state.final_report)


def main() -> None:
    init_session_state()
    render_sidebar()
    refresh_previous_visit_topics()
    start_chat()
    ensure_section_started(st.session_state.selected_section)
    render_chat()

    if st.session_state.finished:
        return

    active_topic = get_active_topic_for_section(st.session_state.selected_section)
    if active_topic is None:
        return

    if prompt := st.chat_input(f"Type your answer for {section_label(st.session_state.selected_section)}"):
        process_turn(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
