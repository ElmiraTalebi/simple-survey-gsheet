"""Single-file Streamlit application for ChatReport."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any
from uuid import uuid4

import streamlit as st
from openai import OpenAI

PAGE_TITLE = "ChatReport"
PAGE_ICON = "🏥"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 1500
MAX_WORKERS = 3
QUESTION_SEQUENCE = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"]
QUESTION_IDS_TO_SKIP_ON_Q1_NO = {"Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9"}
TERMINATION_SIGNAL = "E7_termination_intent"
RESISTANCE_SIGNAL = "E3_explicit_resistance"
STATUS_COLORS = {0: "#2e8b57", 1: "#d8a500", 2: "#ff8c00", 3: "#d62828"}

PAIN_QUESTIONS = [
    {
        "id": "Q1",
        "text": "Do you have any pain today?",
        "options": ["Yes", "No", "Sometimes / It comes and goes"],
        "domain": "presence_and_pattern",
        "input_type": "buttons_with_text",
    },
    {
        "id": "Q2",
        "text": "Where is your pain located?",
        "options": [
            "Mouth or throat",
            "Chest",
            "Abdomen / Stomach",
            "Head or neck",
            "Limbs / Arms or legs",
            "Back",
            "Multiple locations",
            "Somewhere else",
        ],
        "domain": "location_and_character",
        "input_type": "buttons_with_text",
    },
    {
        "id": "Q3",
        "text": "How would you describe your pain?",
        "options": [
            "Sharp or stabbing",
            "Dull or achy",
            "Burning",
            "Throbbing or pulsing",
            "Cramping",
            "Pressure or tightness",
            "I cannot describe it",
        ],
        "domain": "location_and_character",
        "input_type": "buttons_with_text",
    },
    {
        "id": "Q4",
        "text": "How severe is your pain right now? (0 = no pain, 10 = worst imaginable)",
        "options": [
            "0 — No pain",
            "1–3 — Mild",
            "4–6 — Moderate",
            "7–9 — Severe",
            "10 — Worst imaginable",
        ],
        "domain": "severity_and_trajectory",
        "input_type": "buttons_with_text",
    },
    {
        "id": "Q5",
        "text": "How long have you had this pain?",
        "options": [
            "Less than 24 hours",
            "1 to 3 days",
            "4 to 7 days",
            "More than a week",
            "It has been ongoing for months",
        ],
        "domain": "presence_and_pattern",
        "input_type": "buttons_with_text",
    },
    {
        "id": "Q6",
        "text": "Is the pain constant or does it come and go?",
        "options": [
            "Constant — it never fully goes away",
            "Intermittent — it comes and goes",
            "Only at specific times (e.g., when eating, moving)",
        ],
        "domain": "presence_and_pattern",
        "input_type": "buttons_with_text",
    },
    {
        "id": "Q7",
        "text": "Does anything make your pain better?",
        "options": [
            "Rest",
            "Pain medication",
            "Heat or warm compress",
            "Cold or ice",
            "Eating or drinking",
            "Changing position",
            "Nothing helps",
            "I don't know",
        ],
        "domain": "aggravating_and_relieving",
        "input_type": "buttons_with_text",
    },
    {
        "id": "Q8",
        "text": "Does anything make your pain worse?",
        "options": [
            "Movement or physical activity",
            "Eating or drinking",
            "Specific foods",
            "Stress or anxiety",
            "Touching the area",
            "Time of day",
            "Nothing makes it worse",
            "I don't know",
        ],
        "domain": "aggravating_and_relieving",
        "input_type": "buttons_with_text",
    },
    {
        "id": "Q9",
        "text": "Are you currently taking any medication for the pain?",
        "options": [
            "Yes, prescription medication",
            "Yes, over-the-counter medication",
            "No medication",
            "I tried medication but stopped",
        ],
        "domain": "medication",
        "input_type": "buttons_with_text",
    },
    {
        "id": "Q10",
        "text": "Compared to your last check-in, how has your pain changed?",
        "options": [
            "Much better",
            "Slightly better",
            "About the same",
            "Slightly worse",
            "Much worse",
            "This is a new pain — I did not have it before",
        ],
        "domain": "severity_and_trajectory",
        "input_type": "buttons_with_text",
    },
]

SKIP_LOGIC = {"Q1_no_skip_to": "Q10"}

MOCK_PATIENT = {
    "patient_id": "P-00441",
    "demographics": {
        "age": 58,
        "sex": "female",
        "preferred_name": "Maria",
        "preferred_language": "English",
    },
    "oncology_profile": {
        "primary_diagnosis": "Stage IV breast cancer",
        "stage": "IV",
        "diagnosis_date": "2024-10-01",
        "months_since_diagnosis": 18,
        "metastatic_sites": ["bone", "liver"],
        "primary_tumor_location": "right breast",
        "disease_status": "active",
    },
    "current_treatment": {
        "active_treatments": [
            {
                "type": "chemotherapy",
                "name": "capecitabine",
                "start_date": "2025-02-01",
                "cycle_number": 3,
                "status": "ongoing",
            }
        ],
        "treatment_phase": "palliative",
        "recent_procedures": [],
    },
    "comorbidities": [
        {
            "condition": "Type 2 diabetes",
            "relevance_to_pain": "high",
            "notes": "May compound chemotherapy-induced neuropathy",
        }
    ],
    "pain_medication_history": [
        {
            "medication_name": "oxycodone",
            "medication_type": "prescription_opioid",
            "start_date": "2024-08-01",
            "end_date": None,
            "status": "active",
            "discontinuation_reason": None,
            "tenure_days": 261,
            "notes": "Long-term use — tolerance possible",
        }
    ],
    "known_allergies": [],
    "prior_checkin_sessions": [
        {
            "session_id": "SES-20260405",
            "session_date": "2026-04-05",
            "days_ago": 14,
            "treatment_context_at_session": {
                "active_treatment_names": ["capecitabine"],
                "cycle_number": 2,
            },
            "answers": {
                "Q1": {"matched_option": "Yes", "raw_answer": "yeah I do"},
                "Q2": {
                    "matched_option": "Abdomen / Stomach",
                    "raw_answer": "my stomach",
                },
                "Q3": {
                    "matched_option": "Dull or achy",
                    "raw_answer": "dull and heavy",
                },
                "Q4": {
                    "matched_option": "4–6 — Moderate",
                    "raw_answer": "about a 5",
                    "severity_numeric": 5,
                },
                "Q5": {
                    "matched_option": "More than a week",
                    "raw_answer": "been going on a while",
                },
                "Q6": {
                    "matched_option": "Intermittent — it comes and goes",
                    "raw_answer": "it comes and goes",
                },
                "Q7": {
                    "matched_option": "Pain medication",
                    "raw_answer": "I take my pills",
                },
                "Q8": {
                    "matched_option": "Eating or drinking",
                    "raw_answer": "after eating",
                },
                "Q9": {
                    "matched_option": "Yes, prescription medication",
                    "raw_answer": "yes doctor gave me something",
                },
                "Q10": {
                    "matched_option": "About the same",
                    "raw_answer": "same as before",
                },
            },
            "session_flags": {
                "urgency_tier_reached": 1,
                "distress_flagged": True,
                "urgency_flagged": False,
                "engagement_trajectory": "declining",
                "signals_active": ["EM7_minimization", "E1_length_decline"],
            },
            "free_text_notes": None,
        }
    ],
}

MOCK_ANSWERS = {
    "Q1": "Yes",
    "Q2": "Abdomen / Stomach",
    "Q3": "Dull or achy",
    "Q4": "4–6 — Moderate",
    "Q5": "More than a week",
    "Q6": "Intermittent — it comes and goes",
    "Q7": "Pain medication",
    "Q8": "Eating or drinking",
    "Q9": "Yes, prescription medication",
    "Q10": "Slightly worse",
}

ANSWER_INTERPRETER_PROMPT = """
You are the Answer Interpreter Agent for a medical symptom-management chatbot used by cancer patients.
Your only job is to classify a patient's answer into the provided options and return JSON.
Apply exact matching first, then semantic matching. For Q4 use severity numeric rules. For Q9 use medication knowledge.
Special cases:
- distress_flag = true for inability to cope, wanting to die, no will to live
- urgency_flag = true for sudden severe pain, chest pain with symptoms, worst pain, or pain rated 10
- off_topic when unrelated
- invalid when empty or gibberish
Return only valid JSON with:
question_id, patient_answer_raw, match_type, matched_option, confidence, candidates, distress_flag, urgency_flag, reasoning
"""

PRIOR_CHECKIN_PROMPT = """
You are the Prior Check-in Context Agent for a cancer pain chatbot.
Mode session_open:
- Return patient_summary in 40-70 warm neutral words using "At your last check-in..."
- Return agent_baseline with had_pain, pain_location, pain_severity_label, pain_severity_numeric, pain_character, pain_timing, on_medication, medication_type, urgency_was_flagged, distress_was_flagged, days_since_last_checkin, notable_free_text
Mode comparison:
- Compare only the requested question_id against the last check-in
- Return question_id, last_checkin_answer, current_answer, has_prior_data, change_detected, change_direction, change_magnitude, clinical_note
Return only JSON.
"""

PATIENT_HISTORY_PROMPT = """
You are the Patient History & Context Agent for a cancer pain chatbot.
Mode session_start:
- Analyze the full patient record and return the six top-level keys:
patient_profile, longitudinal_pain_trend, risk_profile, anomaly_baseline, historical_flags, treatment_context
Mode on_demand:
- Return query_type and response for the requested targeted historical lookup
Return only JSON.
"""

DOCTOR_RELEVANCE_PROMPT = """
You are the Doctor-Relevance Agent for a cancer pain chatbot.
Assess whether each answer is clinically complete, whether follow-up is needed, what the follow-up goal is, and what note belongs in the doctor report.
Return only JSON with:
question_id, information_completeness, clinical_value_score, follow_up_recommended, follow_up_goal, follow_up_urgency, change_significance, clinical_priority, doctor_note, special_signals, reasoning
"""

URGENCY_PROMPT = """
You are the Urgency & Criticality Agent for a cancer pain chatbot.
Scan all current-session answers plus baseline context for medical and psychological safety signals.
Use urgency tiers 0-3. Tier 2 means same-day care-team contact. Tier 3 means emergency interruption.
Return only JSON with:
question_id, signals_detected, accumulation_triggered, session_tier, escalation_reason, new_signals_this_call, all_active_signals, clinical_escalation_summary, patient_message, continue_session, reasoning
"""

SENTIMENT_PROMPT = """
You are the Patient Sentiment & Engagement Monitor for a cancer pain chatbot.
Assess emotional state, engagement, cognitive accessibility, trust/openness, and adaptation signals.
Return only JSON with:
question_id, dimension_scores, signals_detected, new_signals_this_call, all_active_signals, adaptation_signals, urgency_handoff, engagement_summary_for_doctor, reasoning
"""

NEXT_MOVE_PROMPT = """
You are the Next-Move Agent for a cancer pain chatbot.
Write one brief follow-up question based on the follow_up_goal and tone profile.
Return only JSON with follow_up_question and preamble.
"""

REPORT_PROMPT = """
You are the Doctor-Facing Report Agent for a cancer pain chatbot.
Synthesize the full session package into eight JSON sections:
report_header, priority_flags, executive_summary, detailed_clinical_findings, change_from_last_visit, historical_and_treatment_context, behavioral_and_engagement_notes, data_quality_and_caveats
Return only JSON.
"""


def get_config_value(key: str, default: str | None = None) -> str | None:
    """
    Read configuration from Streamlit secrets first, then environment variables.

    Args:
        key: Configuration key name.
        default: Fallback value if the key is missing.

    Returns:
        Resolved configuration value or the provided default.
    """
    try:
        if key in st.secrets:
            value = st.secrets[key]
            return str(value) if value is not None else default
    except Exception:
        pass
    return os.getenv(key, default)

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="centered",
    initial_sidebar_state="collapsed",
)


def call_agent(
    system_prompt: str,
    user_content: str | dict,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """
    Call the OpenAI Chat Completions API and return parsed JSON.

    Args:
        system_prompt: System prompt string for the agent.
        user_content: User payload string or dictionary.
        temperature: Temperature for the model call.
        max_tokens: Maximum completion tokens.

    Returns:
        Parsed response as a dictionary.

    Raises:
        ValueError: If the model response is not valid JSON.
        RuntimeError: If the API call fails.
    """
    api_key = get_config_value("OPENAI_API_KEY")
    model = get_config_value("OPENAI_MODEL", DEFAULT_MODEL)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in Streamlit secrets or environment variables.")

    payload_text = json.dumps(user_content, indent=2) if isinstance(user_content, dict) else str(user_content)
    agent_name = "unknown_agent"
    question_id = "n/a"
    if isinstance(user_content, dict):
        agent_name = str(user_content.get("agent_name", agent_name))
        question_id = str(user_content.get("question_id", question_id))
    print(f"[call_agent] agent={agent_name} question_id={question_id}")

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload_text},
            ],
        )
        raw_content = response.choices[0].message.content or ""
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Agent response was not valid JSON: {raw_content}") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"OpenAI agent call failed: {exc}") from exc


def initialize_session_state(patient_record: dict) -> dict:
    """
    Create the base session-state dictionary.

    Args:
        patient_record: Full patient record.

    Returns:
        Initialized session-state dictionary.
    """
    now = datetime.now()
    return {
        "session_id": str(uuid4()),
        "patient_id": patient_record["patient_id"],
        "session_date": now.strftime("%Y-%m-%d"),
        "session_start_time": now.strftime("%H:%M"),
        "session_end_time": None,
        "topic": "Pain",
        "patient_record": patient_record,
        "patient_context": None,
        "prior_checkin_baseline": None,
        "prior_checkin_patient_summary": None,
        "question_sequence": QUESTION_SEQUENCE.copy(),
        "questions_answered": [],
        "questions_skipped": [],
        "questions_remaining": QUESTION_SEQUENCE.copy(),
        "current_question_id": "Q1",
        "current_question_index": 0,
        "awaiting_follow_up": False,
        "answer_records": {},
        "follow_up_counts": {qid: 0 for qid in QUESTION_SEQUENCE},
        "redirect_counts": {qid: 0 for qid in QUESTION_SEQUENCE},
        "urgency_state": {
            "current_tier": 0,
            "all_active_signals": [],
            "tier_history": [],
            "escalation_triggered": False,
            "emergency_triggered": False,
            "patient_message_displayed": None,
        },
        "sentiment_state": {
            "all_active_signals": [],
            "engagement_trajectory": "insufficient_data",
            "current_emotional_state": "neutral",
            "reduce_follow_up_depth_active": False,
            "simplify_questions_active": False,
            "acknowledgment_pending": False,
            "acknowledgment_text": None,
        },
        "session_completion_status": "in_progress",
        "turn_number": 0,
        "on_demand_query_responses": [],
        "doctor_report": None,
    }


def get_current_question(state: dict) -> dict | None:
    """
    Return the currently active question dictionary.

    Args:
        state: Session state dictionary.

    Returns:
        Current question or None.
    """
    question_id = state.get("current_question_id")
    if not question_id:
        return None
    return next((question for question in PAIN_QUESTIONS if question["id"] == question_id), None)


def advance_question(state: dict) -> dict:
    """
    Advance the session to the next remaining question.

    Args:
        state: Session state dictionary.

    Returns:
        Updated state.
    """
    current_qid = state.get("current_question_id")
    if current_qid in state["questions_remaining"]:
        state["questions_remaining"].remove(current_qid)
    if current_qid and current_qid not in state["questions_answered"]:
        state["questions_answered"].append(current_qid)

    if not state["questions_remaining"]:
        state["current_question_id"] = None
        state["current_question_index"] = len(state["question_sequence"])
        state["awaiting_follow_up"] = False
        return state

    next_qid = state["questions_remaining"][0]
    state["current_question_id"] = next_qid
    state["current_question_index"] = state["question_sequence"].index(next_qid)
    state["awaiting_follow_up"] = False
    return state


def get_all_raw_answers(state: dict) -> dict:
    """
    Return raw answers keyed by question ID.

    Args:
        state: Session state dictionary.

    Returns:
        Raw answer mapping.
    """
    raw_answers: dict[str, str] = {}
    for question_id, record in state.get("answer_records", {}).items():
        interpreter = record.get("answer_interpreter", {})
        raw_answer = interpreter.get("patient_answer_raw")
        if raw_answer:
            raw_answers[question_id] = raw_answer
    return raw_answers


def run_answer_interpreter(
    question_id: str,
    question_text: str,
    options: list[str],
    patient_answer: str,
) -> dict:
    """
    Run the answer interpreter agent.

    Args:
        question_id: Current question ID.
        question_text: Current question text.
        options: Allowed options.
        patient_answer: Raw patient answer.

    Returns:
        Parsed agent output or safe default.
    """
    max_tokens = int(get_config_value("MAX_TOKENS_AGENT", "1000") or "1000")
    try:
        return call_agent(
            ANSWER_INTERPRETER_PROMPT,
            {
                "agent_name": "answer_interpreter",
                "question_id": question_id,
                "question_text": question_text,
                "options": options,
                "patient_answer": patient_answer,
            },
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception:
        return {
            "question_id": question_id,
            "patient_answer_raw": patient_answer,
            "match_type": "invalid",
            "matched_option": None,
            "confidence": 0.0,
            "candidates": [],
            "distress_flag": False,
            "urgency_flag": False,
            "reasoning": "Agent error.",
        }


def run_prior_checkin_session_open(last_checkin: dict | None, current_session_answers: dict) -> dict:
    """
    Run the prior-checkin agent in session-open mode.

    Args:
        last_checkin: Most recent prior check-in.
        current_session_answers: Current answers so far.

    Returns:
        Prior-checkin summary package.
    """
    max_tokens = int(get_config_value("MAX_TOKENS_AGENT", "1000") or "1000")
    try:
        return call_agent(
            PRIOR_CHECKIN_PROMPT,
            {
                "agent_name": "prior_checkin",
                "question_id": None,
                "mode": "session_open",
                "last_checkin": last_checkin,
                "current_session_answers": current_session_answers,
            },
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception:
        return {
            "patient_summary": "This is your first check-in. There is no previous pain information on file."
            if last_checkin is None
            else "At your last check-in, you reported pain information that has been loaded for today's review.",
            "agent_baseline": {
                "had_pain": None,
                "pain_location": None,
                "pain_severity_label": None,
                "pain_severity_numeric": None,
                "pain_character": None,
                "pain_timing": None,
                "on_medication": None,
                "medication_type": None,
                "urgency_was_flagged": False,
                "distress_was_flagged": False,
                "days_since_last_checkin": None,
                "notable_free_text": None,
            },
        }


def run_prior_checkin_comparison(
    question_id: str,
    last_checkin: dict | None,
    current_session_answers: dict,
) -> dict:
    """
    Run the prior-checkin agent in comparison mode.

    Args:
        question_id: Current question ID.
        last_checkin: Most recent prior check-in.
        current_session_answers: Current answers so far.

    Returns:
        Structured comparison output.
    """
    max_tokens = int(get_config_value("MAX_TOKENS_AGENT", "1000") or "1000")
    current_answer = current_session_answers.get(question_id, {}).get("matched_option")
    try:
        return call_agent(
            PRIOR_CHECKIN_PROMPT,
            {
                "agent_name": "prior_checkin",
                "question_id": question_id,
                "mode": "comparison",
                "last_checkin": last_checkin,
                "current_session_answers": current_session_answers,
            },
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception:
        last_answer = None
        if last_checkin:
            answer_obj = (last_checkin.get("answers") or {}).get(question_id)
            if isinstance(answer_obj, dict):
                last_answer = answer_obj.get("matched_option")
        return {
            "question_id": question_id,
            "last_checkin_answer": last_answer,
            "current_answer": current_answer,
            "has_prior_data": bool(last_answer),
            "change_detected": False,
            "change_direction": "new_data" if not last_answer else "no_change",
            "change_magnitude": None if not last_answer else "none",
            "clinical_note": "No prior data available for this question."
            if not last_answer
            else "No change in this dimension since last check-in.",
        }


def run_history_session_start(patient_record: dict) -> dict:
    """
    Run the patient-history agent at session start.

    Args:
        patient_record: Full patient record.

    Returns:
        Context package or safe default.
    """
    max_tokens = int(get_config_value("MAX_TOKENS_AGENT", "1000") or "1000")
    try:
        return call_agent(
            PATIENT_HISTORY_PROMPT,
            {
                "agent_name": "patient_history",
                "question_id": None,
                "mode": "session_start",
                "patient_record": patient_record,
            },
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception:
        demographics = patient_record.get("demographics", {})
        oncology = patient_record.get("oncology_profile", {})
        treatment = patient_record.get("current_treatment", {})
        prior_sessions = patient_record.get("prior_checkin_sessions", [])
        medications = patient_record.get("pain_medication_history", [])
        return {
            "patient_profile": {
                "preferred_name": demographics.get("preferred_name"),
                "age": demographics.get("age"),
                "diagnosis_summary": oncology.get("primary_diagnosis"),
                "disease_status": oncology.get("disease_status"),
                "current_treatment_summary": ", ".join(
                    tx.get("name", tx.get("type", "treatment"))
                    for tx in treatment.get("active_treatments", [])
                )
                or "No active treatment listed.",
                "treatment_phase": treatment.get("treatment_phase"),
                "metastatic_sites": oncology.get("metastatic_sites", []),
                "primary_tumor_location": oncology.get("primary_tumor_location"),
                "comorbidities_relevant_to_pain": [
                    {"condition": item.get("condition"), "clinical_note": item.get("notes")}
                    for item in patient_record.get("comorbidities", [])
                    if item.get("relevance_to_pain") in {"high", "moderate"}
                ],
                "active_pain_medications": [
                    {
                        "name": med.get("medication_name"),
                        "type": med.get("medication_type"),
                        "tenure_days": med.get("tenure_days"),
                        "note": med.get("notes"),
                    }
                    for med in medications
                    if med.get("status") == "active"
                ],
                "known_allergies_to_pain_medications": patient_record.get("known_allergies", []),
                "is_first_session": len(prior_sessions) == 0,
            },
            "longitudinal_pain_trend": {
                "sessions_analyzed": len(prior_sessions),
                "date_range": {
                    "oldest_session": prior_sessions[-1]["session_date"] if prior_sessions else None,
                    "most_recent_session": prior_sessions[0]["session_date"] if prior_sessions else None,
                },
                "severity_trend": {
                    "overall_direction": "insufficient_data" if len(prior_sessions) < 2 else "stable",
                    "trend_pattern": "single prior check-in available" if prior_sessions else "no prior sessions",
                    "historical_severity_range": {"minimum": 5 if prior_sessions else None, "maximum": 5 if prior_sessions else None},
                    "patient_baseline_severity": 5 if prior_sessions else None,
                    "trend_note": "Limited prior pain history is available."
                    if prior_sessions
                    else "No prior pain history available.",
                },
                "location_history": {
                    "all_reported_locations": ["Abdomen / Stomach"] if prior_sessions else [],
                    "location_stability": "insufficient_data" if len(prior_sessions) < 2 else "stable",
                    "new_since_last_session": [],
                    "note": "Limited location history available.",
                },
                "character_history": {
                    "all_reported_characters": ["Dull or achy"] if prior_sessions else [],
                    "character_stability": "insufficient_data" if len(prior_sessions) < 2 else "stable",
                    "neuropathic_pattern_emerging": False,
                    "note": "Limited character history available.",
                },
                "medication_effectiveness_signal": {
                    "declining_effectiveness_detected": False,
                    "medication_stop_restart_pattern": False,
                    "note": "No clear medication effectiveness trend available.",
                },
            },
            "risk_profile": {
                "risk_flags": {
                    "PAIN_CRISIS_RISK": {"elevated": False, "basis": "No crisis trend available."},
                    "BONE_PAIN_RISK": {
                        "elevated": "bone" in oncology.get("metastatic_sites", []),
                        "basis": "Bone metastases present."
                        if "bone" in oncology.get("metastatic_sites", [])
                        else "No bone-specific risk signal recorded.",
                    },
                    "NEUROPATHIC_PAIN_RISK": {
                        "elevated": True,
                        "basis": "Diabetes and chemotherapy may contribute to neuropathic pain risk.",
                    },
                    "SPINAL_CORD_COMPRESSION_RISK": {
                        "elevated": False,
                        "basis": "No spinal compression history in available record.",
                    },
                    "OPIOID_RELATED_RISK": {
                        "elevated": bool(medications),
                        "basis": "Long-term opioid exposure may increase tolerance and management complexity."
                        if medications
                        else "No opioid exposure documented.",
                    },
                    "PSYCHOLOGICAL_RISK": {
                        "elevated": True,
                        "basis": "Cancer population carries elevated psychological burden.",
                    },
                    "IMMUNOCOMPROMISE_RISK": {
                        "elevated": True,
                        "basis": "Active chemotherapy increases infection risk.",
                    },
                    "UNDERREPORTING_RISK": {
                        "elevated": True,
                        "basis": "Prior minimization signal was present." if prior_sessions else "Cancer patients may underreport symptoms.",
                    },
                },
                "elevated_risk_count": 5,
                "overall_risk_level": "elevated",
                "risk_summary": "The patient has several pain-related risk factors tied to metastatic disease, chemotherapy, and long-term opioid exposure. Monitoring for underreporting and treatment-related complications remains important.",
            },
            "anomaly_baseline": {
                "typical_severity_range": {"low": 4, "high": 6},
                "typical_pain_location": "Abdomen / Stomach" if prior_sessions else None,
                "typical_pain_character": "Dull or achy" if prior_sessions else None,
                "typical_medication_status": "prescription" if medications else "none",
                "baseline_engagement": "declining" if prior_sessions else "insufficient_data",
                "anomaly_thresholds": {
                    "severity_spike_threshold": 8,
                    "severity_drop_threshold": 2,
                    "unexpected_location": "Any location not previously reported",
                    "unexpected_character": "Any pain character not previously reported",
                },
                "anomaly_notes": "Use caution when interpreting change because only limited prior history is available.",
            },
            "historical_flags": {
                "total_prior_sessions": len(prior_sessions),
                "distress_history": {
                    "sessions_with_distress_flag": 1 if prior_sessions else 0,
                    "distress_rate": 1.0 if prior_sessions else 0.0,
                    "most_recent_distress_session": prior_sessions[0]["session_date"] if prior_sessions else None,
                    "recurring_distress": True if prior_sessions else False,
                    "highest_tier_reached": prior_sessions[0].get("session_flags", {}).get("urgency_tier_reached", 0)
                    if prior_sessions
                    else 0,
                },
                "engagement_history": {
                    "sessions_with_declining_trajectory": 1 if prior_sessions else 0,
                    "recurring_engagement_decline": True if prior_sessions else False,
                    "typical_engagement_change": "declining" if prior_sessions else "insufficient_data",
                    "early_termination_history": False,
                },
                "signal_frequency": {
                    "recurrent_signals": prior_sessions[0].get("session_flags", {}).get("signals_active", [])
                    if prior_sessions
                    else [],
                },
                "notable_session_events": ["Prior session included distress and declining engagement."] if prior_sessions else [],
            },
            "treatment_context": {
                "expected_pain_sources": [
                    {
                        "source": "Metastatic disease",
                        "expected_location": "Bone or abdominal regions",
                        "expected_character": "Aching or pressure-type pain",
                        "clinical_note": "Active metastatic cancer may contribute to ongoing pain symptoms.",
                    },
                    {
                        "source": "Capecitabine treatment",
                        "expected_location": "Hands, feet, or diffuse discomfort",
                        "expected_character": "Burning, soreness, or neuropathic features",
                        "clinical_note": "Chemotherapy can contribute to treatment-related discomfort and neuropathy.",
                    },
                ],
                "treatment_timeline_context": {
                    "current_phase_description": treatment.get("treatment_phase", "unknown"),
                    "pain_expectation": "elevated",
                    "days_since_last_treatment": None,
                },
                "recent_treatment_changes": {
                    "change_detected": False,
                    "change_description": "No recent treatment changes documented.",
                    "pain_implication": "Pain pattern may still reflect active treatment and disease burden.",
                },
                "treatment_context_note": "Active metastatic breast cancer and ongoing chemotherapy both increase the likelihood of clinically meaningful pain symptoms. Changes in pain should be interpreted in the context of palliative treatment and metastatic disease burden.",
            },
        }


def run_history_on_demand(patient_record: dict, query_type: str, query_context: dict) -> dict:
    """
    Run the history agent for a targeted lookup.

    Args:
        patient_record: Full patient record.
        query_type: Lookup type.
        query_context: Extra query context.

    Returns:
        Targeted history response or safe default.
    """
    max_tokens = int(get_config_value("MAX_TOKENS_AGENT", "1000") or "1000")
    try:
        return call_agent(
            PATIENT_HISTORY_PROMPT,
            {
                "agent_name": "patient_history",
                "question_id": query_context.get("question_id"),
                "mode": "on_demand",
                "patient_record": patient_record,
                "query_type": query_type,
                "query_context": query_context,
            },
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception:
        return {
            "query_type": query_type,
            "response": {
                "clinical_interpretation": "Targeted history lookup unavailable; rely on current-session data.",
                "query_context": query_context,
            },
        }


def run_doctor_relevance(
    question_id: str,
    question_text: str,
    current_answer: dict,
    prior_checkin_baseline: dict,
    prior_checkin_comparison: dict,
    current_session_answers: dict,
    follow_up_count: int,
    on_demand_context: dict | None = None,
) -> dict:
    """
    Run the doctor-relevance agent.

    Args:
        question_id: Current question ID.
        question_text: Current question text.
        current_answer: Answer interpreter output.
        prior_checkin_baseline: Prior-checkin baseline.
        prior_checkin_comparison: Current comparison output.
        current_session_answers: Current session answers.
        follow_up_count: Follow-ups already used for this question.
        on_demand_context: Optional on-demand history context.

    Returns:
        Doctor-relevance output or safe default.
    """
    max_tokens = int(get_config_value("MAX_TOKENS_AGENT", "1000") or "1000")
    try:
        return call_agent(
            DOCTOR_RELEVANCE_PROMPT,
            {
                "agent_name": "doctor_relevance",
                "question_id": question_id,
                "question_text": question_text,
                "current_answer": current_answer,
                "prior_checkin_baseline": prior_checkin_baseline,
                "prior_checkin_comparison": prior_checkin_comparison,
                "current_session_answers": current_session_answers,
                "follow_up_count_this_question": follow_up_count,
                "on_demand_context": on_demand_context,
            },
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception:
        return {
            "question_id": question_id,
            "information_completeness": "none",
            "clinical_value_score": 0.0,
            "follow_up_recommended": False,
            "follow_up_goal": None,
            "follow_up_urgency": "none",
            "change_significance": "no_baseline",
            "clinical_priority": "low",
            "doctor_note": None,
            "special_signals": {
                "trajectory_mismatch": False,
                "aggravating_medication_signal": False,
                "medication_stop_signal": False,
                "patient_resistance_detected": False,
                "multi_location_unenumerated": False,
            },
            "reasoning": "Agent error.",
        }


def run_urgency(
    question_id: str,
    current_answer: dict,
    current_session_answers: dict,
    prior_checkin_baseline: dict,
    prior_checkin_comparison: dict,
    active_signals_this_session: list[str],
) -> dict:
    """
    Run the urgency agent.

    Args:
        question_id: Current question ID.
        current_answer: Answer interpreter output.
        current_session_answers: Current session answers.
        prior_checkin_baseline: Prior baseline.
        prior_checkin_comparison: Comparison output.
        active_signals_this_session: Existing urgency signals.

    Returns:
        Urgency output or safe default.
    """
    max_tokens = int(get_config_value("MAX_TOKENS_AGENT", "1000") or "1000")
    try:
        return call_agent(
            URGENCY_PROMPT,
            {
                "agent_name": "urgency",
                "question_id": question_id,
                "current_answer": current_answer,
                "current_session_answers": current_session_answers,
                "prior_checkin_baseline": prior_checkin_baseline,
                "prior_checkin_comparison": prior_checkin_comparison,
                "active_signals_this_session": active_signals_this_session,
            },
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception:
        return {
            "question_id": question_id,
            "signals_detected": [],
            "accumulation_triggered": {"A1": False, "A2": False},
            "session_tier": 0,
            "escalation_reason": None,
            "new_signals_this_call": [],
            "all_active_signals": active_signals_this_session,
            "clinical_escalation_summary": None,
            "patient_message": None,
            "continue_session": True,
            "reasoning": "Agent error.",
        }


def run_sentiment(
    question_id: str,
    current_answer: dict,
    current_session_answers: dict,
    active_signals_this_session: list[str],
    session_question_count: int,
    doctor_relevance_output: dict | None = None,
) -> dict:
    """
    Run the sentiment and engagement agent.

    Args:
        question_id: Current question ID.
        current_answer: Answer interpreter output.
        current_session_answers: Current session answers.
        active_signals_this_session: Existing sentiment signals.
        session_question_count: Count of answered questions.
        doctor_relevance_output: Optional doctor-relevance output.

    Returns:
        Sentiment output or safe default.
    """
    max_tokens = int(get_config_value("MAX_TOKENS_AGENT", "1000") or "1000")
    try:
        return call_agent(
            SENTIMENT_PROMPT,
            {
                "agent_name": "sentiment",
                "question_id": question_id,
                "current_answer": current_answer,
                "current_session_answers": current_session_answers,
                "active_signals_this_session": active_signals_this_session,
                "session_question_count": session_question_count,
                "doctor_relevance_output": doctor_relevance_output,
            },
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception:
        return {
            "question_id": question_id,
            "dimension_scores": {
                "emotional_state": {"primary": "neutral", "secondary": None},
                "engagement_level": "moderate",
                "engagement_trajectory": "insufficient_data",
                "cognitive_accessibility": "unclear",
                "trust_openness": "cooperative",
            },
            "signals_detected": [],
            "new_signals_this_call": [],
            "all_active_signals": active_signals_this_session,
            "adaptation_signals": {
                "tone_profile": "standard",
                "acknowledgment_required": False,
                "acknowledgment_text": None,
                "simplify_next_question": False,
                "suggest_pause_before_next_question": False,
                "reduce_follow_up_depth": False,
            },
            "urgency_handoff": {"flag": False, "quoted_text": None, "note": None},
            "engagement_summary_for_doctor": None,
            "reasoning": "Agent error.",
        }


def run_next_move(
    question_id: str,
    question_text: str,
    original_answer: dict,
    follow_up_goal: str,
    follow_up_urgency: str,
    tone_profile: str,
    acknowledgment_required: bool,
    acknowledgment_text: str | None,
    simplify: bool,
    current_session_answers: dict,
) -> dict:
    """
    Run the next-move agent to generate a follow-up question.

    Args:
        question_id: Current question ID.
        question_text: Current question text.
        original_answer: Original interpreted answer.
        follow_up_goal: Information goal to close.
        follow_up_urgency: Follow-up urgency.
        tone_profile: Tone profile from sentiment agent.
        acknowledgment_required: Whether acknowledgment is already shown.
        acknowledgment_text: Existing acknowledgment text.
        simplify: Whether to simplify wording.
        current_session_answers: Current session answers.

    Returns:
        Next-move output or safe default.
    """
    max_tokens = int(get_config_value("MAX_TOKENS_AGENT", "1000") or "1000")
    try:
        return call_agent(
            NEXT_MOVE_PROMPT,
            {
                "agent_name": "next_move",
                "question_id": question_id,
                "question_text": question_text,
                "original_answer": original_answer,
                "follow_up_goal": follow_up_goal,
                "follow_up_urgency": follow_up_urgency,
                "tone_profile": tone_profile,
                "acknowledgment_required": acknowledgment_required,
                "acknowledgment_text": acknowledgment_text,
                "simplify": simplify,
                "current_session_answers": current_session_answers,
            },
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception:
        return {"follow_up_question": "Could you tell me a little more about that?", "preamble": None}


def run_report(session_package: dict) -> dict:
    """
    Run the doctor-facing report agent.

    Args:
        session_package: Full session package.

    Returns:
        Report output or safe fallback report.
    """
    max_tokens = int(get_config_value("MAX_TOKENS_AGENT", "1000") or "1000")
    try:
        return call_agent(
            REPORT_PROMPT,
            {
                "agent_name": "report",
                "question_id": None,
                "session_package": session_package,
            },
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception:
        state = session_package.get("session_state", {})
        patient_context = state.get("patient_context", {})
        preferred_name = (
            patient_context.get("patient_profile", {}).get("preferred_name")
            or state.get("patient_record", {}).get("demographics", {}).get("preferred_name")
        )
        urgency_tier = state.get("urgency_state", {}).get("current_tier", 0)
        return {
            "report_header": {
                "report_id": f"RPT-{state.get('session_id', 'unknown')}",
                "patient_id": state.get("patient_id"),
                "patient_preferred_name": preferred_name,
                "patient_age": state.get("patient_record", {}).get("demographics", {}).get("age"),
                "diagnosis_summary": state.get("patient_record", {}).get("oncology_profile", {}).get("primary_diagnosis"),
                "session_date": state.get("session_date"),
                "session_duration_minutes": session_package.get("session_duration_minutes"),
                "session_completion_status": state.get("session_completion_status"),
                "questions_answered_count": len(state.get("questions_answered", [])),
                "questions_total": 10,
                "urgency_tier": urgency_tier,
                "report_generated_at": session_package.get("report_generated_at"),
                "topic": state.get("topic", "Pain"),
                "overall_risk_level": patient_context.get("risk_profile", {}).get("overall_risk_level", "unknown"),
            },
            "priority_flags": {
                "overall_action_required": "same_day_contact"
                if urgency_tier == 2
                else "emergency_services_notified"
                if urgency_tier == 3
                else "monitoring"
                if urgency_tier == 1
                else "none",
                "emergency": [],
                "urgent": [],
                "notable": [],
            },
            "executive_summary": {
                "narrative": "Session completed with fallback report generation after agent failure. Review structured answer records directly for the most reliable detail.",
                "one_line_status": "Fallback report generated.",
                "overall_pain_status": "unknown",
            },
            "detailed_clinical_findings": [],
            "change_from_last_visit": {
                "days_since_last_checkin": state.get("prior_checkin_baseline", {}).get("days_since_last_checkin"),
                "last_checkin_date": (
                    state.get("patient_record", {}).get("prior_checkin_sessions", [{}])[0].get("session_date")
                    if state.get("patient_record", {}).get("prior_checkin_sessions")
                    else None
                ),
                "trajectory_overview": "Trajectory could not be fully synthesized because the report agent was unavailable.",
                "changed_dimensions": [],
                "no_change_dimensions": [],
                "first_session_note": None,
            },
            "historical_and_treatment_context": {
                "longitudinal_severity_summary": patient_context.get("longitudinal_pain_trend", {}).get("severity_trend", {}).get("trend_note"),
                "pain_pattern_history": patient_context.get("treatment_context", {}).get("treatment_context_note"),
                "elevated_risk_flags": [],
                "anomaly_context": patient_context.get("anomaly_baseline", {}).get("anomaly_notes"),
                "treatment_context_note": patient_context.get("treatment_context", {}).get("treatment_context_note"),
                "expected_pain_sources": patient_context.get("treatment_context", {}).get("expected_pain_sources", []),
                "notable_prior_session_events": patient_context.get("historical_flags", {}).get("notable_session_events", []),
            },
            "behavioral_and_engagement_notes": {
                "session_emotional_state": state.get("sentiment_state", {}).get("current_emotional_state", "neutral"),
                "engagement_trajectory": state.get("sentiment_state", {}).get("engagement_trajectory", "insufficient_data"),
                "cognitive_accessibility": None,
                "engagement_summary": None,
                "active_behavioral_signals": state.get("sentiment_state", {}).get("all_active_signals", []),
                "data_reliability_note": "Interpret findings with caution because final synthesis used a fallback path.",
                "wellbeing_concern": urgency_tier >= 2,
                "wellbeing_note": state.get("urgency_state", {}).get("patient_message_displayed"),
            },
            "data_quality_and_caveats": {
                "overall_data_completeness_score": 50,
                "completeness_rating": "partial",
                "incomplete_answers": [],
                "match_quality_notes": [],
                "session_completion_note": "Fallback report generated due to report agent failure.",
                "follow_up_data_needs": [],
            },
        }


def initialize_session(patient_record: dict) -> dict:
    """
    Initialize the session and load patient context.

    Args:
        patient_record: Full patient record.

    Returns:
        Fully initialized session state.
    """
    state = initialize_session_state(patient_record)
    patient_context = run_history_session_start(patient_record)
    prior_sessions = patient_record.get("prior_checkin_sessions", [])
    last_checkin = prior_sessions[0] if prior_sessions else None
    prior_context = run_prior_checkin_session_open(last_checkin=last_checkin, current_session_answers={})
    state["patient_context"] = patient_context
    state["prior_checkin_baseline"] = prior_context.get("agent_baseline")
    state["prior_checkin_patient_summary"] = prior_context.get("patient_summary")
    return state


def process_answer(
    session_state: dict,
    patient_raw_answer: str,
    is_follow_up_answer: bool = False,
) -> tuple[dict, dict]:
    """
    Process one patient answer through the orchestrator.

    Args:
        session_state: Current session state.
        patient_raw_answer: Raw patient answer.
        is_follow_up_answer: Whether this answer is a follow-up response.

    Returns:
        Updated session state and orchestrator output.
    """
    session_state["turn_number"] += 1
    current_question = get_current_question(session_state)
    if current_question is None:
        return _complete_session(session_state)

    question_id = current_question["id"]
    answer_record = session_state["answer_records"].setdefault(question_id, {})

    answer_interpreter_output = run_answer_interpreter(
        question_id=question_id,
        question_text=current_question["text"],
        options=current_question.get("options") or [],
        patient_answer=patient_raw_answer,
    )
    answer_record["answer_interpreter"] = answer_interpreter_output
    answer_record.setdefault("question_text", current_question["text"])
    if is_follow_up_answer:
        answer_record["follow_up_answer_raw"] = patient_raw_answer
        answer_record["follow_up_interpreter"] = answer_interpreter_output

    current_session_answers = _build_current_session_answers(session_state)
    current_session_answers[question_id] = {
        "matched_option": answer_interpreter_output.get("matched_option"),
        "raw_answer": answer_interpreter_output.get("patient_answer_raw", patient_raw_answer),
        "match_type": answer_interpreter_output.get("match_type"),
        "confidence": answer_interpreter_output.get("confidence"),
    }

    last_checkin = _get_last_checkin(session_state)
    prior_checkin_comparison = {
        "question_id": question_id,
        "last_checkin_answer": None,
        "current_answer": answer_interpreter_output.get("matched_option"),
        "has_prior_data": False,
        "change_detected": False,
        "change_direction": "new_data",
        "change_magnitude": None,
        "clinical_note": "No prior data available for this question.",
    }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        comparison_future = None
        if not is_follow_up_answer:
            comparison_future = executor.submit(
                run_prior_checkin_comparison,
                question_id,
                last_checkin,
                current_session_answers,
            )
        if comparison_future is not None:
            prior_checkin_comparison = comparison_future.result()
            answer_record["prior_checkin_comparison"] = prior_checkin_comparison

        urgency_future = executor.submit(
            run_urgency,
            question_id,
            answer_interpreter_output,
            current_session_answers,
            session_state.get("prior_checkin_baseline") or {},
            prior_checkin_comparison,
            session_state["urgency_state"]["all_active_signals"],
        )
        sentiment_future = executor.submit(
            run_sentiment,
            question_id,
            answer_interpreter_output,
            current_session_answers,
            session_state["sentiment_state"]["all_active_signals"],
            len(session_state["questions_answered"]) + 1,
            None,
        )
        urgency_output = urgency_future.result()
        sentiment_output = sentiment_future.result()

    answer_record["urgency"] = urgency_output
    answer_record["sentiment"] = sentiment_output

    urgency_state = session_state["urgency_state"]
    sentiment_state = session_state["sentiment_state"]
    new_tier = urgency_output.get("session_tier", 0)
    urgency_state["current_tier"] = max(urgency_state["current_tier"], new_tier)
    urgency_state["tier_history"].append({"question_id": question_id, "tier": new_tier})
    urgency_state["all_active_signals"] = _merge_unique(
        urgency_state["all_active_signals"],
        urgency_output.get("all_active_signals", []),
    )

    sentiment_state["all_active_signals"] = _merge_unique(
        sentiment_state["all_active_signals"],
        sentiment_output.get("all_active_signals", []),
    )
    dimension_scores = sentiment_output.get("dimension_scores", {})
    emotional_state = dimension_scores.get("emotional_state", {})
    sentiment_state["engagement_trajectory"] = dimension_scores.get(
        "engagement_trajectory",
        sentiment_state["engagement_trajectory"],
    )
    sentiment_state["current_emotional_state"] = emotional_state.get(
        "primary",
        sentiment_state["current_emotional_state"],
    )
    adaptation_signals = sentiment_output.get("adaptation_signals", {})
    sentiment_state["reduce_follow_up_depth_active"] = adaptation_signals.get(
        "reduce_follow_up_depth",
        False,
    )
    sentiment_state["simplify_questions_active"] = adaptation_signals.get(
        "simplify_next_question",
        False,
    )
    sentiment_state["acknowledgment_pending"] = adaptation_signals.get(
        "acknowledgment_required",
        False,
    )
    sentiment_state["acknowledgment_text"] = adaptation_signals.get("acknowledgment_text")

    escalation_newly_triggered = False
    if urgency_state["current_tier"] == 3:
        return _handle_emergency(session_state, urgency_output)
    if urgency_state["current_tier"] == 2 and not urgency_state["escalation_triggered"]:
        urgency_state["escalation_triggered"] = True
        urgency_state["patient_message_displayed"] = urgency_output.get("patient_message")
        escalation_newly_triggered = True

    match_type = answer_interpreter_output.get("match_type")
    if match_type == "off_topic":
        session_state["redirect_counts"][question_id] += 1
        if session_state["redirect_counts"][question_id] >= 2:
            session_state["questions_skipped"].append({"question_id": question_id, "reason": "off_topic"})
            session_state = advance_question(session_state)
            if not session_state["questions_remaining"]:
                return _complete_session(session_state)
            return _next_question_output(session_state, None, escalation_newly_triggered)
        return (
            session_state,
            {
                "next_action": "redirect",
                "patient_facing_output": {
                    "acknowledgment_text": None,
                    "tier2_notice": urgency_output.get("patient_message") if escalation_newly_triggered else None,
                    "transition_bridge": None,
                    "question_text": current_question["text"],
                    "question_id": question_id,
                    "question_options": current_question.get("options"),
                    "message_type": "question",
                },
                "decision_log": {"reason": "off_topic_redirect", "question_id": question_id},
                "trigger_report": False,
                "doctor_report": None,
            },
        )
    if match_type == "invalid":
        return (
            session_state,
            {
                "next_action": "redisplay_question",
                "patient_facing_output": {
                    "acknowledgment_text": None,
                    "tier2_notice": urgency_output.get("patient_message") if escalation_newly_triggered else None,
                    "transition_bridge": None,
                    "question_text": current_question["text"],
                    "question_id": question_id,
                    "question_options": current_question.get("options"),
                    "message_type": "question",
                },
                "decision_log": {"reason": "invalid_answer", "question_id": question_id},
                "trigger_report": False,
                "doctor_report": None,
            },
        )

    on_demand_context = None
    query_type = _check_on_demand_conditions(
        session_state=session_state,
        answer_interpreter_output=answer_interpreter_output,
        prior_checkin_comparison=prior_checkin_comparison,
        urgency_output=urgency_output,
    )
    if query_type is not None:
        on_demand_context = run_history_on_demand(
            patient_record=session_state["patient_record"],
            query_type=query_type,
            query_context={
                "question_id": question_id,
                "answer": answer_interpreter_output,
                "prior_checkin_comparison": prior_checkin_comparison,
                "urgency_output": urgency_output,
            },
        )
        session_state["on_demand_query_responses"].append(on_demand_context)

    doctor_relevance_output = run_doctor_relevance(
        question_id=question_id,
        question_text=current_question["text"],
        current_answer=answer_interpreter_output,
        prior_checkin_baseline=session_state.get("prior_checkin_baseline") or {},
        prior_checkin_comparison=prior_checkin_comparison,
        current_session_answers=current_session_answers,
        follow_up_count=session_state["follow_up_counts"][question_id],
        on_demand_context=on_demand_context,
    )
    answer_record["doctor_relevance"] = doctor_relevance_output

    new_sentiment_signals = sentiment_output.get("new_signals_this_call", [])
    engagement_level = dimension_scores.get("engagement_level", "moderate")
    clinical_priority = doctor_relevance_output.get("clinical_priority", "low")
    follow_up_decision = doctor_relevance_output.get("follow_up_recommended", False)
    suppression_reasons: list[str] = []

    if session_state["follow_up_counts"][question_id] >= 1:
        follow_up_decision = False
        suppression_reasons.append("S1")
    if TERMINATION_SIGNAL in new_sentiment_signals:
        follow_up_decision = False
        suppression_reasons.append("S2")
    if match_type in {"invalid", "off_topic"}:
        follow_up_decision = False
        suppression_reasons.append("S3")
    if is_follow_up_answer:
        follow_up_decision = False
        suppression_reasons.append("S4")

    if not suppression_reasons:
        if question_id == "Q4" and doctor_relevance_output.get("information_completeness") != "complete":
            follow_up_decision = True
        if doctor_relevance_output.get("special_signals", {}).get("medication_stop_signal") and session_state["follow_up_counts"][question_id] == 0:
            follow_up_decision = True
        if (
            urgency_state["current_tier"] == 2
            and doctor_relevance_output.get("follow_up_goal")
            and doctor_relevance_output.get("follow_up_urgency") == "immediate"
        ):
            follow_up_decision = True

        if sentiment_state["reduce_follow_up_depth_active"] and clinical_priority != "high":
            follow_up_decision = False
            suppression_reasons.append("SS1")
        if RESISTANCE_SIGNAL in new_sentiment_signals and clinical_priority != "high":
            follow_up_decision = False
            suppression_reasons.append("SS2")
        if engagement_level == "resistant" and clinical_priority == "medium":
            follow_up_decision = False
            suppression_reasons.append("SS3")

    if TERMINATION_SIGNAL in new_sentiment_signals:
        return _handle_early_termination(session_state)

    if follow_up_decision:
        next_move_output = run_next_move(
            question_id=question_id,
            question_text=current_question["text"],
            original_answer=answer_interpreter_output,
            follow_up_goal=doctor_relevance_output.get("follow_up_goal"),
            follow_up_urgency=doctor_relevance_output.get("follow_up_urgency", "routine"),
            tone_profile=adaptation_signals.get("tone_profile", "standard"),
            acknowledgment_required=sentiment_state["acknowledgment_pending"],
            acknowledgment_text=sentiment_state["acknowledgment_text"],
            simplify=sentiment_state["simplify_questions_active"],
            current_session_answers=current_session_answers,
        )
        answer_record["next_move"] = next_move_output
        session_state["follow_up_counts"][question_id] += 1
        session_state["awaiting_follow_up"] = True
        return (
            session_state,
            {
                "next_action": "follow_up",
                "patient_facing_output": _compose_patient_output(
                    session_state=session_state,
                    question_text=next_move_output.get("follow_up_question"),
                    question_id=question_id,
                    question_options=None,
                    tier2_notice=urgency_output.get("patient_message") if escalation_newly_triggered else None,
                    transition_bridge=next_move_output.get("preamble"),
                    message_type="follow_up",
                ),
                "decision_log": {
                    "question_id": question_id,
                    "follow_up_decision": True,
                    "suppression_rules_triggered": suppression_reasons,
                },
                "trigger_report": False,
                "doctor_report": None,
            },
        )

    answer_record["final_matched_option"] = answer_interpreter_output.get("matched_option")
    if question_id == "Q1":
        session_state = _apply_q1_skip_logic(session_state)
    session_state = advance_question(session_state)
    if not session_state["questions_remaining"]:
        return _complete_session(session_state)
    return _next_question_output(session_state, None, escalation_newly_triggered)


def _handle_emergency(session_state: dict, urgency_output: dict) -> tuple[dict, dict]:
    """
    Handle Tier 3 emergency termination.

    Args:
        session_state: Current state.
        urgency_output: Urgency output.

    Returns:
        Updated state and emergency output.
    """
    session_state["session_completion_status"] = "emergency_terminated"
    session_state["urgency_state"]["emergency_triggered"] = True
    session_state["urgency_state"]["patient_message_displayed"] = urgency_output.get("patient_message")
    state, output = _complete_session(session_state)
    output["next_action"] = "emergency_terminated"
    output["patient_facing_output"] = {
        "acknowledgment_text": None,
        "tier2_notice": None,
        "transition_bridge": None,
        "question_text": urgency_output.get("patient_message"),
        "question_id": None,
        "question_options": None,
        "message_type": "emergency",
    }
    return state, output


def _handle_early_termination(session_state: dict) -> tuple[dict, dict]:
    """
    Handle patient early termination.

    Args:
        session_state: Current state.

    Returns:
        Updated state and closing output.
    """
    session_state["session_completion_status"] = "partial"
    state, output = _complete_session(session_state)
    output["next_action"] = "early_termination"
    output["patient_facing_output"] = {
        "acknowledgment_text": None,
        "tier2_notice": None,
        "transition_bridge": None,
        "question_text": "Of course — we'll stop here. The answers you've shared have been saved and will be reviewed by your care team.",
        "question_id": None,
        "question_options": None,
        "message_type": "closing",
    }
    return state, output


def _complete_session(session_state: dict) -> tuple[dict, dict]:
    """
    Complete the session and generate the doctor-facing report.

    Args:
        session_state: Current state.

    Returns:
        Updated state and completion output.
    """
    now = datetime.now()
    session_state["session_end_time"] = now.strftime("%H:%M")
    if session_state["session_completion_status"] == "in_progress":
        session_state["session_completion_status"] = "complete"

    session_duration_minutes = _compute_duration_minutes(
        session_state.get("session_start_time"),
        session_state.get("session_end_time"),
    )
    session_package = {
        "session_state": session_state,
        "raw_answers": get_all_raw_answers(session_state),
        "session_duration_minutes": session_duration_minutes,
        "report_generated_at": now.isoformat(timespec="minutes"),
    }
    doctor_report = run_report(session_package)
    session_state["doctor_report"] = doctor_report

    closing_message = "Thank you for completing today's check-in. Your responses have been saved and shared with your care team."
    if session_state["urgency_state"]["escalation_triggered"]:
        closing_message += " A member of your care team will be in touch with you today."

    return (
        session_state,
        {
            "next_action": "complete_session",
            "patient_facing_output": {
                "acknowledgment_text": None,
                "tier2_notice": None,
                "transition_bridge": None,
                "question_text": closing_message,
                "question_id": None,
                "question_options": None,
                "message_type": "closing",
            },
            "decision_log": {"session_completion_status": session_state["session_completion_status"]},
            "trigger_report": True,
            "doctor_report": doctor_report,
        },
    )


def _check_on_demand_conditions(
    session_state: dict,
    answer_interpreter_output: dict,
    prior_checkin_comparison: dict,
    urgency_output: dict,
) -> str | None:
    """
    Evaluate on-demand lookup conditions in priority order.

    Args:
        session_state: Current state.
        answer_interpreter_output: Interpreter output.
        prior_checkin_comparison: Comparison output.
        urgency_output: Urgency output.

    Returns:
        Query type string or None.
    """
    question_id = answer_interpreter_output.get("question_id")
    matched_option = answer_interpreter_output.get("matched_option")
    patient_context = session_state.get("patient_context") or {}
    anomaly = patient_context.get("anomaly_baseline", {})
    character_history = patient_context.get("longitudinal_pain_trend", {}).get("character_history", {})
    distress_history = patient_context.get("historical_flags", {}).get("distress_history", {})

    if question_id == "Q2" and prior_checkin_comparison.get("change_detected"):
        return "location_history"
    if question_id == "Q4":
        spike_threshold = anomaly.get("anomaly_thresholds", {}).get("severity_spike_threshold")
        severity_numeric = _severity_numeric_from_label(matched_option)
        if severity_numeric is not None and spike_threshold is not None and severity_numeric >= spike_threshold:
            return "severity_context"
    if question_id == "Q9" and (
        matched_option == "I tried medication but stopped"
        or "medication" in (answer_interpreter_output.get("patient_answer_raw", "").lower())
    ):
        return "medication_history"
    if answer_interpreter_output.get("distress_flag") and distress_history.get("recurring_distress"):
        return "psychological_history"
    if question_id == "Q3" and matched_option and matched_option not in character_history.get("all_reported_characters", []):
        return "character_history"
    if urgency_output.get("accumulation_triggered", {}).get("A1"):
        return "session_pattern"
    return None


def _apply_q1_skip_logic(session_state: dict) -> dict:
    """
    Apply the Q1 no-pain skip logic.

    Args:
        session_state: Current state.

    Returns:
        Updated state.
    """
    q1_record = session_state.get("answer_records", {}).get("Q1", {})
    matched_option = q1_record.get("final_matched_option") or q1_record.get("answer_interpreter", {}).get("matched_option")
    prior_baseline = session_state.get("prior_checkin_baseline") or {}
    longitudinal = session_state.get("patient_context", {}).get("longitudinal_pain_trend", {})
    presence_history = longitudinal.get("pain_presence_history", {})
    consistently_present = presence_history.get("consistently_present", False)

    if matched_option == "No" and prior_baseline.get("had_pain") is False and consistently_present is False:
        remaining = []
        for qid in session_state["questions_remaining"]:
            if qid in QUESTION_IDS_TO_SKIP_ON_Q1_NO:
                session_state["questions_skipped"].append({"question_id": qid, "reason": "Q1_no_pain"})
            else:
                remaining.append(qid)
        session_state["questions_remaining"] = remaining
    return session_state


def _next_question_output(
    session_state: dict,
    transition_bridge: str | None,
    escalation_newly_triggered: bool,
) -> tuple[dict, dict]:
    """
    Build the next-question output package.

    Args:
        session_state: Updated state.
        transition_bridge: Optional transition text.
        escalation_newly_triggered: Whether Tier 2 notice is newly triggered.

    Returns:
        Updated state and next-question output.
    """
    question = get_current_question(session_state)
    if question is None:
        return _complete_session(session_state)
    return (
        session_state,
        {
            "next_action": "ask_next_question",
            "patient_facing_output": _compose_patient_output(
                session_state=session_state,
                question_text=question["text"],
                question_id=question["id"],
                question_options=question.get("options"),
                tier2_notice=session_state["urgency_state"]["patient_message_displayed"] if escalation_newly_triggered else None,
                transition_bridge=transition_bridge or _build_transition_bridge(session_state, question["id"]),
                message_type="question",
            ),
            "decision_log": {"question_id": question["id"], "next_action": "ask_next_question"},
            "trigger_report": False,
            "doctor_report": None,
        },
    )


def _compose_patient_output(
    session_state: dict,
    question_text: str | None,
    question_id: str | None,
    question_options: list[str] | None,
    tier2_notice: str | None,
    transition_bridge: str | None,
    message_type: str,
) -> dict:
    """
    Compose patient-facing output in the required display order.

    Args:
        session_state: Current state.
        question_text: Main text to show.
        question_id: Question ID, if applicable.
        question_options: Options, if applicable.
        tier2_notice: Tier 2 notice text.
        transition_bridge: Transition bridge text.
        message_type: Output type label.

    Returns:
        Patient-facing output dictionary.
    """
    acknowledgment_text = None
    if session_state["sentiment_state"]["acknowledgment_pending"]:
        acknowledgment_text = session_state["sentiment_state"]["acknowledgment_text"]
        session_state["sentiment_state"]["acknowledgment_pending"] = False
    if acknowledgment_text:
        transition_bridge = None
    return {
        "acknowledgment_text": acknowledgment_text,
        "tier2_notice": tier2_notice,
        "transition_bridge": transition_bridge,
        "question_text": question_text,
        "question_id": question_id,
        "question_options": question_options,
        "message_type": message_type,
    }


def _build_current_session_answers(session_state: dict) -> dict:
    """
    Build the current-session answer mapping used by agents.

    Args:
        session_state: Current state.

    Returns:
        Current session answer mapping.
    """
    current_answers: dict[str, dict[str, Any]] = {}
    for question_id, record in session_state.get("answer_records", {}).items():
        interpreter = record.get("answer_interpreter", {})
        if interpreter:
            current_answers[question_id] = {
                "matched_option": record.get("final_matched_option") or interpreter.get("matched_option"),
                "raw_answer": interpreter.get("patient_answer_raw"),
                "match_type": interpreter.get("match_type"),
                "confidence": interpreter.get("confidence"),
            }
    return current_answers


def _get_last_checkin(session_state: dict) -> dict | None:
    """
    Return the most recent prior check-in if present.

    Args:
        session_state: Current state.

    Returns:
        Most recent prior check-in or None.
    """
    sessions = session_state.get("patient_record", {}).get("prior_checkin_sessions", [])
    return sessions[0] if sessions else None


def _merge_unique(existing: list[str], new_items: list[str]) -> list[str]:
    """
    Merge two string lists while preserving order and uniqueness.

    Args:
        existing: Existing list.
        new_items: New items to add.

    Returns:
        Merged unique list.
    """
    merged = list(existing)
    for item in new_items:
        if item not in merged:
            merged.append(item)
    return merged


def _severity_numeric_from_label(label: str | None) -> int | None:
    """
    Convert a severity label into a midpoint number.

    Args:
        label: Severity label.

    Returns:
        Midpoint numeric value or None.
    """
    mapping = {
        "0 — No pain": 0,
        "1–3 — Mild": 2,
        "4–6 — Moderate": 5,
        "7–9 — Severe": 8,
        "10 — Worst imaginable": 10,
    }
    return mapping.get(label)


def _compute_duration_minutes(start_time: str | None, end_time: str | None) -> int | None:
    """
    Compute duration in minutes from HH:MM timestamps.

    Args:
        start_time: Session start time.
        end_time: Session end time.

    Returns:
        Integer minutes or None.
    """
    if not start_time or not end_time:
        return None
    try:
        start_dt = datetime.strptime(start_time, "%H:%M")
        end_dt = datetime.strptime(end_time, "%H:%M")
        delta = end_dt - start_dt
        return max(int(delta.total_seconds() // 60), 0)
    except ValueError:
        return None


def _build_transition_bridge(session_state: dict, next_question_id: str) -> str | None:
    """
    Create a short bridge when the domain changes.

    Args:
        session_state: Current state.
        next_question_id: Upcoming question ID.

    Returns:
        Short bridge text or None.
    """
    questions_by_id = {question["id"]: question for question in PAIN_QUESTIONS}
    answered = session_state.get("questions_answered", [])
    if not answered:
        return None
    last_qid = answered[-1]
    last_domain = questions_by_id.get(last_qid, {}).get("domain")
    next_domain = questions_by_id.get(next_question_id, {}).get("domain")
    if last_domain and next_domain and last_domain != next_domain:
        return "A few more details about your pain."
    return None


def inject_css() -> None:
    """Inject custom CSS for the Streamlit UI."""
    st.markdown(
        """
        <style>
        .status-pill {display:flex; justify-content:flex-end; align-items:center; gap:8px; font-size:0.9rem; margin-bottom:0.5rem;}
        .status-dot {width:12px; height:12px; border-radius:50%; display:inline-block;}
        .amber-box {background:#fff3cd; border:1px solid #f1d48b; border-radius:10px; padding:0.85rem; margin:0.5rem 0;}
        .red-box {background:#fde2e2; border:1px solid #e07a7a; border-radius:10px; padding:1rem; margin:0.5rem 0;}
        .green-box {background:#e6f5ea; border:1px solid #90c49b; border-radius:10px; padding:1rem; margin:0.5rem 0;}
        div[data-testid="stChatMessage"] {border-radius:16px;}
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {background:#e8f1ff;}
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {background:#ffffff; border:1px solid #e5e7eb;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_app_state() -> None:
    """Initialize Streamlit session state on first load."""
    if st.session_state.get("initialized") and "session" in st.session_state and "messages" in st.session_state:
        return
    session = initialize_session(MOCK_PATIENT)
    st.session_state["session"] = session
    st.session_state["messages"] = []
    st.session_state["initialized"] = True
    st.session_state["doctor_report"] = None
    st.session_state["text_input_value"] = ""
    st.session_state["last_submitted_answer"] = None
    st.session_state["last_submission_follow_up"] = False
    st.session_state["last_error"] = None

    summary = session.get("prior_checkin_patient_summary")
    if summary:
        st.session_state["messages"].append({"role": "assistant", "content": summary, "type": "standard"})
    first_question = get_current_question(session)
    if first_question:
        st.session_state["messages"].append({"role": "assistant", "content": first_question["text"], "type": "question"})


def ensure_session_state() -> bool:
    """
    Ensure the minimal Streamlit state exists before rendering.

    Returns:
        True when session state is ready, otherwise False.
    """
    try:
        initialize_app_state()
        required_keys = {"session", "messages", "initialized"}
        missing = [key for key in required_keys if key not in st.session_state]
        if missing:
            session = initialize_session(MOCK_PATIENT)
            st.session_state["session"] = session
            st.session_state["messages"] = st.session_state.get("messages", [])
            st.session_state["initialized"] = True
            st.session_state.setdefault("doctor_report", None)
            st.session_state.setdefault("text_input_value", "")
            st.session_state.setdefault("last_submitted_answer", None)
            st.session_state.setdefault("last_submission_follow_up", False)
            st.session_state.setdefault("last_error", None)
        return True
    except Exception as exc:
        st.error(f"Unable to initialize the app session: {exc}")
        return False


def add_output_messages(patient_facing_output: dict) -> None:
    """Append orchestrator output to the Streamlit message list."""
    if patient_facing_output.get("acknowledgment_text"):
        st.session_state["messages"].append(
            {"role": "assistant", "content": patient_facing_output["acknowledgment_text"], "type": "standard"}
        )
    if patient_facing_output.get("tier2_notice"):
        st.session_state["messages"].append(
            {"role": "assistant", "content": patient_facing_output["tier2_notice"], "type": "tier2"}
        )
    text = patient_facing_output.get("question_text")
    bridge = patient_facing_output.get("transition_bridge")
    if text:
        content = f"{bridge}\n\n{text}" if bridge else text
        st.session_state["messages"].append(
            {"role": "assistant", "content": content, "type": patient_facing_output.get("message_type", "standard")}
        )


def submit_answer(answer: str) -> None:
    """Submit one patient answer and update the UI state."""
    if not answer.strip():
        return
    if not ensure_session_state():
        return
    session = st.session_state["session"]
    is_follow_up = session["awaiting_follow_up"]
    st.session_state["last_submitted_answer"] = answer
    st.session_state["last_submission_follow_up"] = is_follow_up
    st.session_state["messages"].append({"role": "user", "content": answer, "type": "standard"})

    try:
        with st.spinner("Processing your response..."):
            updated_state, orchestrator_output = process_answer(
                session_state=session,
                patient_raw_answer=answer,
                is_follow_up_answer=is_follow_up,
            )
        st.session_state["session"] = updated_state
        add_output_messages(orchestrator_output["patient_facing_output"])
        if orchestrator_output["trigger_report"]:
            st.session_state["doctor_report"] = orchestrator_output["doctor_report"]
        st.session_state["text_input_value"] = ""
        st.session_state["last_error"] = None
        st.rerun()
    except Exception as exc:
        st.session_state["last_error"] = str(exc)
        st.error(f"Something went wrong while processing this response: {exc}")


def run_mock_session() -> None:
    """Run the full mock session automatically for testing."""
    if not ensure_session_state():
        return
    session = initialize_session(MOCK_PATIENT)
    for _ in range(20):
        question = get_current_question(session)
        if not question:
            break
        answer = MOCK_ANSWERS.get(question["id"], "I don't know")
        updated_state, output = process_answer(
            session_state=session,
            patient_raw_answer=answer,
            is_follow_up_answer=session["awaiting_follow_up"],
        )
        print(f"[mock_session] qid={question['id']} output={json.dumps(output, indent=2)}")
        session = updated_state
        if output["trigger_report"]:
            break
    st.session_state["session"] = session
    st.session_state["doctor_report"] = session.get("doctor_report")


def render_sidebar() -> None:
    """Render sidebar metadata and developer controls."""
    if not ensure_session_state():
        return
    session = st.session_state["session"]
    context = session.get("patient_context") or {}
    patient_name = (
        context.get("patient_profile", {}).get("preferred_name")
        or session.get("patient_record", {}).get("demographics", {}).get("preferred_name", "Unknown")
    )
    urgency_tier = session["urgency_state"]["current_tier"]
    with st.sidebar:
        st.markdown(f"### {patient_name}")
        st.write(f"Session date: {session['session_date']}")
        st.write(f"Questions answered: {len(session['questions_answered'])} of 10")
        st.write(f"Current urgency tier: {urgency_tier}")
        st.write(f"Active signals: {len(session['urgency_state']['all_active_signals'])}")
        if st.checkbox("Developer: Run mock session"):
            run_mock_session()
            st.rerun()


def render_status_indicator() -> None:
    """Render the top-right urgency tier indicator."""
    if not ensure_session_state():
        return
    urgency_tier = st.session_state["session"]["urgency_state"]["current_tier"]
    color = STATUS_COLORS[urgency_tier]
    st.markdown(
        f"""
        <div class="status-pill">
          <span>Urgency Tier {urgency_tier}</span>
          <span class="status-dot" style="background:{color};"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_messages() -> None:
    """Render the chat thread."""
    if not ensure_session_state():
        return
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            message_type = message.get("type", "standard")
            if message_type == "tier2":
                st.markdown(f"<div class='amber-box'>{message['content']}</div>", unsafe_allow_html=True)
            elif message_type == "emergency":
                st.markdown(
                    f"<div class='red-box'><strong>{message['content']}</strong><br><br>Call or text <strong>988</strong> now. If you are in immediate danger, call <strong>911</strong>.</div>",
                    unsafe_allow_html=True,
                )
            elif message_type == "closing":
                st.markdown(f"<div class='green-box'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(message["content"])


def render_input_area() -> None:
    """Render button options and text input for the current question."""
    if not ensure_session_state():
        return
    session = st.session_state["session"]
    if session["session_completion_status"] != "in_progress":
        return
    if session["urgency_state"]["emergency_triggered"]:
        return

    current_question = get_current_question(session)
    if current_question is None:
        return

    options = None if session["awaiting_follow_up"] else current_question.get("options")
    if options:
        cols = st.columns(len(options))
        for index, option in enumerate(options):
            if cols[index].button(option, key=f"option_{current_question['id']}_{index}", use_container_width=True):
                submit_answer(option)

    st.text_input(
        "Type your answer",
        key="text_input_value",
        placeholder="Type here if you'd rather answer in your own words...",
    )
    if st.button("Send", use_container_width=True):
        submit_answer(st.session_state["text_input_value"])

    if st.session_state.get("last_error") and st.button("Retry", use_container_width=True):
        submit_answer(st.session_state.get("last_submitted_answer") or "")


def render_clinical_report() -> None:
    """Render the doctor-facing report after session completion."""
    if not ensure_session_state():
        return
    report = st.session_state.get("doctor_report")
    if not report:
        return
    with st.expander("📋 Clinical Report for Care Team", expanded=False):
        priority = report.get("priority_flags", {})
        for item in priority.get("emergency", []):
            st.error(item.get("summary"))
        for item in priority.get("urgent", []):
            st.warning(item.get("summary"))
        for item in priority.get("notable", []):
            st.info(item.get("summary"))

        executive_summary = report.get("executive_summary", {})
        st.markdown("### Executive Summary")
        st.markdown(executive_summary.get("narrative", ""))
        st.caption(executive_summary.get("one_line_status", ""))

        detailed = report.get("detailed_clinical_findings", [])
        for domain in detailed:
            with st.expander(domain.get("domain_label", "Clinical Findings")):
                for finding in domain.get("findings", []):
                    st.markdown(
                        f"**{finding.get('question_id')}**: {finding.get('matched_answer')}  \n{finding.get('clinical_note')}"
                    )

        change = report.get("change_from_last_visit", {})
        changed_dimensions = change.get("changed_dimensions", [])
        if changed_dimensions:
            st.markdown("### Change From Last Visit")
            st.table(changed_dimensions)

        quality = report.get("data_quality_and_caveats", {})
        st.metric("Data Quality Score", quality.get("overall_data_completeness_score", "N/A"))
        st.download_button(
            "Download Report JSON",
            data=json.dumps(report, indent=2),
            file_name="chatreport_clinical_report.json",
            mime="application/json",
        )


def main() -> None:
    """Run the Streamlit app."""
    inject_css()
    if not ensure_session_state():
        return
    render_sidebar()
    render_status_indicator()
    render_messages()
    render_input_area()
    render_clinical_report()


if __name__ == "__main__":
    main()
