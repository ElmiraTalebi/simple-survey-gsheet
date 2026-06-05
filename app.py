import json
import os
import html
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

import streamlit as st
from openai import OpenAI

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    gspread = None
    Credentials = None


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

Opening:
If this is the first assistant message, start exactly with:
"Are there any symptoms you would like to report to your medical team?"

Carefully analyze the patient's response:
- Extract any already-answered topics.
- Acknowledge the patient's concerns.
- Prioritize follow-ups based on what they mention.
- Do not lose track of the initial answer. Refer back to it when relevant.

Conversational Behavior Rules:
- Ask exactly one question at a time.
- One question means one clinical variable only. Do not combine questions with "and," "also," "as well," or commas that ask for multiple answers in the same turn.
- Do not ask long lists of questions.
- If a patient answers multiple topics at once, do not repeat questions already answered.
- If the patient names a symptom in the opening answer, address that symptom directly and move to the matching clinical topic first. For example, if the patient says "constipation," go directly to GI Symptoms instead of following the topic order.
- Expand only where details are missing.
- Use conditional logic:
- Focus deeply on the patient's main or most severe symptoms first. For other topics, use brief broad screening questions and ask detailed follow-ups only if the patient says yes or reports a concern.
- You should cover all the topics, but do not force every detailed sub-question for every patient. If a patient says no to a topic, do not ask follow-ups for that topic.
- After all topics are covered, follow the strict Final Closing Sequence below. Never combine the "anything else" question and the completion into a single turn.
- Occasionally offer guided options when helpful, especially for medications or symptoms patients may not recall precisely.
- Keep tone warm, reassuring, and professional.

Length Control:
- Keep each assistant reply to 1-2 short sentences.
- Do not collect full detail for mild, stable, or denied symptoms.
- If the patient reports several symptoms, triage them in this order: safety/red flags, symptoms the patient says are worst or worsening, symptoms affecting eating/drinking/swallowing/breathing/functioning, then broad screening of remaining topics.
- After the main concerns are addressed, ask broad screening questions such as: "Are you having any other issues with eating, swallowing, breathing, mood, or stomach/bowel symptoms?"

Memory and Redundancy Rules:
- Before every reply, silently review the full conversation and prior history.
- Treat the patient's current answers as already known facts.
- Never ask the patient to restate a fact they already gave, including whether something is worse, better, unchanged, present, absent, constant, intermittent, severe, or medication-related.
- If prior history says a symptom existed before and the patient already says it is worse, better, resolved, or unchanged, accept that comparison and ask only for the next missing clinically important detail.
- If a symptom was already screened in one topic, do not screen for the same symptom again in another topic. Refer back to it instead.



Clinical Topics To Cover:
1. Pain
If pain is reported:
- Location (if the patient did not already mention the location of pain)
- Severity
- Onset
- Timing: constant vs intermittent
- Medications (please provide some options for patient ahead)
- Medication effectiveness
- Medication side effects, without specifically asking about constipation here
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
- Weight change or unintentional weight loss
- Fluid intake
- Barriers to eating/drinking
- Use of nutritional supplements

If Feeding Tube:
- Functionality: leakage, blockage, discomfort
- Oral intake ability

Ask these factors separately and do not ask them in one question. 

3. Swallowing
Assess:
- Difficulty swallowing, choking, or coughing
If yes:
- Liquids vs solids
- Frequency
- Pills
- Blood when coughing



5. Oral Symptoms
Assess:
- Mouth sores: new vs existing, location, pain severity, effect on eating, drinking, swallowing, or speaking, treatments such as magic mouthwash
- Dry mouth: timing, treatments, functional impact
- Mucus: thick vs watery, impact, management
- Teeth/gums issues
- Oral hygiene practices

6. GI Symptoms
Assess:
- Nausea
- Vomiting
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
- Emotional state: anxiety, worry, sadness, depression, low mood, loss of interest, hopelessness
- Impact on functioning
- Social support system

If the patient reports mood concerns:
- Respond empathetically before asking the next question.
- Include depression/low mood whenever giving examples of mood symptoms; do not focus only on anxiety or worry.

10. Other
Assess:
- Fever
If fever is reported:
- When it started
- Highest temperature, if known
- Chills or feeling acutely unwell



Use of Patient History Rule:
- If prior patient history is provided, use it as memory, not as a checklist.
- Bring up past issues only when they are clinically relevant or not already addressed by the patient's current answer.
- Ask whether a past issue has resolved, improved, worsened, or stayed the same only if the patient has not already provided that comparison.
- If the patient has already provided the comparison, ask only one missing follow-up detail or move on.
- If patient context is provided, use the patient's name, doctor's name, and current week of therapy as clinical context.
- Consider the current week of therapy when deciding which symptoms are clinically relevant and how to interpret changes from prior history.
- If prior history is from an earlier week of therapy, compare the current check-in to that earlier point when the patient provides enough information.

Efficiency Rules:
Avoid asking:
- Questions already answered
- Irrelevant follow-ups

Prioritize:
- Symptoms impacting safety, such as weight loss, swallowing issues, bleeding
- Keeping the conversation concise but complete

Final Closing Sequence (STRICT - DO NOT SKIP):

Once all 9 clinical topics have been covered, ask one final open-ended question before completing the check-in.

Turn 1 - The Final Open-Ended Question:
- The message must be a single open-ended question, e.g.:
  "Before we wrap up, is there anything else you'd like to share with me - anything I haven't asked about?"
- On this turn, is_complete MUST be false.
- doctor_summary MUST be an empty string.
- Do NOT include closing language, "thank you," or any indication the check-in is ending.
- "topic" should be an empty string.

Turn 2 - The Patient Responds to the Final Open-Ended Question:
- Carefully determine whether the patient added a new symptom, concern, question, or care-related issue.
- If the patient did not add a new concern, briefly acknowledge their response, inform them the check-in is complete and the information will be shared with their doctor, and set is_complete to true.
- If the patient added a new symptom or concern, do not close the check-in yet.
- If the patient added a new symptom or concern, acknowledge it warmly and specifically, treat it like any other reported symptom, and ask the most relevant single follow-up question needed for that concern.
- If the patient added a new symptom or concern, is_complete MUST be false and doctor_summary MUST be an empty string.
- If the concern matches a listed clinical topic, set topic to the closest matching topic. If it does not fit any listed topic, set topic to an empty string.

Critical: is_complete must NEVER be true on the same turn where you ask "anything else." is_complete must also NEVER be true on any turn where you ask the patient a follow-up question.

Handling New Concerns at the Final "Anything Else" Question:
- If the patient answers the final "anything else" question with no new concern, proceed to the Closing Turn.
- If the patient mentions a new symptom or concern, do not close the check-in yet.
- The assistant must respond to the new concern before ending the check-in.
- Treat the new concern like any other reported symptom.
- Ask exactly one targeted follow-up question about the new concern.
- Do not ask multiple follow-up questions about a new concern raised at the final "anything else" question.
- After the patient answers that one follow-up question, return immediately to the final open-ended question.
- Do not use a generic follow-up such as "Can you tell me more about that symptom?" if a more specific clinical question is appropriate.
- Choose the single most useful follow-up question based on the concern. Prefer onset, severity/impact, or safety relevance.
- For new symptoms outside the listed topics, gather only the single most important missing detail needed for the doctor summary.
- After the patient answers the one follow-up question, return to the final open-ended question:
  "Before we wrap up, is there anything else you'd like to share with me - anything I haven't asked about?"
- Only set is_complete to true after the patient answers that final open-ended question with no additional new concern.

Example:
- If the patient answers the final open-ended question with "I also have ringing in my ears," do not close the check-in.
- Acknowledge it and ask a targeted follow-up such as: "I'm sorry you're dealing with that. When did the ringing in your ears start?"
- After the patient answers that one follow-up question, ask the final open-ended question again.

Internal Output Requirement:
While chatting naturally, internally ensure you can produce a structured summary for the doctor including:
- Key symptoms
- Changes since last visit
- Red flags
- Medication effectiveness
- Functional status
- Use **bold** markdown sparingly in the doctor_summary to highlight the most important clinical information doctors need to see quickly, such as red flags, new or worsening symptoms, severe symptoms, fever/chills, bleeding, choking, inability to take medications, major functional impact, or severe emotional distress.

Response Format:
Always respond as valid JSON with exactly these keys:
{
  "reply": "Natural message to show the patient.",
  "is_complete": true or false,
  "doctor_summary": "A concise 2-3 sentence clinical summary for the doctor if complete, otherwise an empty string.",
  "topic": "one of: Pain, Nutrition, Swallowing, Oral Symptoms, GI Symptoms, Fatigue & Sleep, Activity & Independence, Mood & Support, Other, or empty string"
}

Important:
- The patient should only see the value of "reply".
- Set "is_complete" to true only when the check-in is genuinely complete.
- For "topic", choose the single clinical topic that best matches the current assistant reply or question. If the reply is a general opening, closing, or administrative message, use an empty string.

"""


SUMMARY_SYSTEM_PROMPT = """
Role:
You are an expert clinical summarization assistant specializing in head and neck cancer. Your job is to produce a concise, doctor-facing pre-visit summary based on a conversational check-in already conducted between a patient and a nurse assistant.

Context:
- The check-in covers some or all of these topics: Pain, Nutrition, Swallowing, Oral Symptoms, GI Symptoms, Fatigue & Sleep, Activity & Independence, Mood & Support, Other.
- The full chat transcript (assistant questions and patient answers) will be provided.
- Prior patient history from a previous visit may also be provided. It may be missing.
- Patient context may include patient name, doctor name, and current week of therapy.
- The doctor has only a few minutes to review this summary before the visit, so clarity and brevity are critical.

Instructions:

1. Summarize only what the patient actually reported in the transcript. Do not invent, assume, or infer information that was not stated.

2. Present information the way clinicians are used to reading it: lead with red flags and key changes, use clinical shorthand where appropriate (severity, frequency, duration, location), and avoid conversational filler.

3. If prior patient history is provided, explicitly compare current findings to it (e.g., "improved since last visit," "new since last visit," "unchanged," "worsened"). If no prior history is provided, do not fabricate comparisons.

4. If patient context is provided, use the patient name, doctor name, and current week of therapy as clinical context. Include the week of therapy when it helps clarify symptom timing or changes.

5. For each topic, produce two summaries and a status:
   - "Main issues": a 1-2 sentence top-line for fast review. Include only red flags, significant symptoms, notable changes, and safety concerns. If the patient explicitly denied symptoms for the topic, write "No issues reported." If the topic was never covered in the chat, use an empty string.
   - "more details": a fuller but still concise breakdown. Include any of the following that apply: severity, location, onset, timing (constant vs intermittent), frequency, medications and their effectiveness, side effects, aggravating and alleviating factors, and functional impact. Use short labeled lines or bullets, not long paragraphs.
   - "status": one of "worse", "better", or "" (empty string).
     * "worse" - the topic represents a NEW symptom OR a WORSENING symptom compared to prior history. If no prior history is available, use "worse" only when the patient describes the symptom as new or recent.
     * "better" - the topic shows IMPROVEMENT compared to prior history.
     * "" (empty) - everything else, including unchanged/stable symptoms, topics with no prior history to compare against, and topics that were not discussed.

6. Always elevate the following to "Main issues" when present: unintentional weight loss, dehydration, choking on liquids, bleeding (including blood when coughing), severe or worsening pain, inability to take medications, severe emotional distress or suicidal ideation, feeding tube malfunction, and inability to perform basic daily activities.

7. Use **bold** markdown sparingly to highlight the most important clinical information doctors need to see quickly. Bold only short phrases or key findings, not entire paragraphs. Prioritize bolding red flags, new or worsening symptoms, severe symptoms, safety concerns, major functional impact, weight loss/dehydration, bleeding, choking, fever/chills or feeling acutely unwell, inability to take medications, feeding tube problems, and severe emotional distress or suicidal ideation.

8. Use the "Other" fields to capture clinically relevant content the patient raised that does not map to the listed topics (e.g., new symptoms outside scope, social or caregiving issues affecting care, specific questions the patient wants to ask the doctor). Leave empty if nothing applies.

9. Tone: neutral, factual, clinical. Do not reassure the patient, editorialize, or offer recommendations or treatment plans - just report what was said.

Response Format:
Always respond as valid JSON with exactly these keys, and no text outside the JSON object:
{
  "Pain_Main issues": "",
  "Pain_more details": "",
  "Pain_status": "",
  "Nutrition_Main issues": "",
  "Nutrition_more details": "",
  "Nutrition_status": "",
  "Swallowing_Main issues": "",
  "Swallowing_more details": "",
  "Swallowing_status": "",
  "Oral Symptoms_Main issues": "",
  "Oral Symptoms_more details": "",
  "Oral Symptoms_status": "",
  "GI Symptoms_Main issues": "",
  "GI Symptoms_more details": "",
  "GI Symptoms_status": "",
  "Fatigue & Sleep_Main issues": "",
  "Fatigue & Sleep_more details": "",
  "Fatigue & Sleep_status": "",
  "Activity & Independence_Main issues": "",
  "Activity & Independence_more details": "",
  "Activity & Independence_status": "",
  "Mood & Support_Main issues": "",
  "Mood & Support_more details": "",
  "Mood & Support_status": "",
  "Other_Main issues": "",
  "Other_more details": "",
  "Other_status": ""
}

Rules:
- Every key listed above must be present in your response.
- Use an empty string for any topic with no relevant information.
- Output must be valid JSON with no commentary, markdown, or text outside the object.
"""


CHAT_TOPICS = [
    "Pain",
    "Nutrition",
    "Swallowing",
    "Oral Symptoms",
    "GI Symptoms",
    "Fatigue & Sleep",
    "Activity & Independence",
    "Mood & Support",
    "Other",
]

SUMMARY_TOPICS = CHAT_TOPICS


def build_patient_context(
    patient_name: str = "",
    doctor_name: str = "",
    therapy_week: str = "",
) -> str:
    context_lines = []

    if patient_name.strip():
        context_lines.append(f"Patient name: {patient_name.strip()}")
    if doctor_name.strip():
        context_lines.append(f"Doctor name: {doctor_name.strip()}")
    if therapy_week.strip():
        context_lines.append(f"Current week of therapy: {therapy_week.strip()}")

    return "\n".join(context_lines)


def build_messages(
    chat_history: List[Dict[str, str]],
    prior_history: str = "",
    patient_context: str = "",
    system_prompt: str = SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    system_content = system_prompt

    if patient_context.strip():
        system_content += f"""

Patient Context:
{patient_context.strip()}
"""

    if prior_history.strip():
        system_content += f"""

Prior Patient History:
{prior_history.strip()}
"""

    messages = [{"role": "system", "content": system_content}]
    messages.extend(chat_history)
    return messages


def _is_anything_else_turn(chat_history: List[Dict[str, str]]) -> bool:
    if len(chat_history) < 2:
        return False

    previous_message = chat_history[-2]
    if previous_message.get("role") != "assistant":
        return False

    content = previous_message.get("content", "").lower()
    return "anything else" in content and ("wrap up" in content or "haven't asked" in content)


def _patient_added_new_concern(chat_history: List[Dict[str, str]]) -> bool:
    if not chat_history or chat_history[-1].get("role") != "user":
        return False

    text = chat_history[-1].get("content", "").strip().lower()
    if not text:
        return False

    no_patterns = [
        r"^no[\s.!?]*$",
        r"^nope[\s.!?]*$",
        r"^nothing[\s.!?]*$",
        r"^none[\s.!?]*$",
        r"^not really[\s.!?]*$",
        r"^that's all[\s.!?]*$",
        r"^that is all[\s.!?]*$",
        r"^i think that's all[\s.!?]*$",
        r"^i think that is all[\s.!?]*$",
    ]
    return not any(re.match(pattern, text) for pattern in no_patterns)


def _should_return_to_anything_else(chat_history: List[Dict[str, str]]) -> bool:
    if not chat_history or chat_history[-1].get("role") != "user":
        return False

    for index in range(len(chat_history) - 2, -1, -1):
        message = chat_history[index]
        if message.get("role") != "assistant":
            continue

        content = message.get("content", "").lower()
        is_anything_else = (
            "anything else" in content
            and ("wrap up" in content or "haven't asked" in content)
        )
        if not is_anything_else:
            continue

        if index + 1 >= len(chat_history):
            return False

        patient_reply = chat_history[index + 1]
        if patient_reply.get("role") != "user":
            return False

        messages_after_final_question = chat_history[index + 2:]
        assistant_followups = [
            later_message
            for later_message in messages_after_final_question
            if later_message.get("role") == "assistant"
        ]
        return (
            _patient_added_new_concern(chat_history[: index + 2])
            and len(assistant_followups) >= 1
        )

    return False


def _reply_has_multiple_questions(reply: str) -> bool:
    if reply.count("?") > 1:
        return True

    question = reply.lower()
    multi_part_patterns = [
        r"\b(where|when|how|what|which)\b.+\band\b.+\b(where|when|how|what|which)\b",
        r"\bcan you\b.+\band\b.+\b(where|when|how|what|which)\b",
        r"\bcould you\b.+\band\b.+\b(where|when|how|what|which)\b",
    ]
    return "?" in question and any(
        re.search(pattern, question) for pattern in multi_part_patterns
    )


def _parse_nurse_response(raw_content: str) -> Dict[str, Any]:
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        return {
            "reply": raw_content,
            "is_complete": False,
            "doctor_summary": "",
        }


def get_nurse_response(
    client: OpenAI,
    chat_history: List[Dict[str, str]],
    prior_history: str,
    patient_context: str,
    model: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> Dict[str, Any]:
    if _should_return_to_anything_else(chat_history):
        reply = "Before we wrap up, is there anything else you'd like to share with me - anything I haven't asked about?"
        raw_content = json.dumps(
            {
                "reply": reply,
                "is_complete": False,
                "doctor_summary": "",
                "topic": "",
            }
        )
        return {
            "reply": reply,
            "is_complete": False,
            "doctor_summary": "",
            "topic": "",
            "raw_response": raw_content,
        }

    response = client.chat.completions.create(
        model=model,
        messages=build_messages(chat_history, prior_history, patient_context, system_prompt),
        reasoning_effort="minimal",
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content or ""
    parsed = _parse_nurse_response(raw_content)

    if _reply_has_multiple_questions(parsed.get("reply", "")):
        retry_messages = build_messages(chat_history, prior_history, patient_context, system_prompt)
        retry_messages.append(
            {
                "role": "system",
                "content": (
                    "Internal quality check: Rewrite the next assistant response so it "
                    "asks exactly one question, asks for only one clinical variable, "
                    "does not repeat information the patient already gave, and still "
                    "returns the required JSON object."
                ),
            }
        )
        retry_response = client.chat.completions.create(
            model=model,
            messages=retry_messages,
            reasoning_effort="minimal",
            response_format={"type": "json_object"},
        )
        raw_content = retry_response.choices[0].message.content or ""
        parsed = _parse_nurse_response(raw_content)

    reply = parsed.get("reply", "").strip()
    is_complete = bool(parsed.get("is_complete", False))
    if _is_anything_else_turn(chat_history) and _patient_added_new_concern(chat_history):
        if not reply or "check-in is complete" in reply.lower() or "shared with your doctor" in reply.lower():
            final_concern_messages = build_messages(chat_history, prior_history, patient_context, system_prompt)
            final_concern_messages.append(
                {
                    "role": "system",
                    "content": (
                        "Internal quality check: The patient added a new concern in "
                        "response to the final anything-else question. Do not close the "
                        "check-in yet. Ask exactly one specific clinically relevant "
                        "follow-up question about the new concern. is_complete must be "
                        "false and doctor_summary must be an empty string. Return the "
                        "required JSON object."
                    ),
                }
            )
            final_concern_response = client.chat.completions.create(
                model=model,
                messages=final_concern_messages,
                reasoning_effort="minimal",
                response_format={"type": "json_object"},
            )
            raw_content = final_concern_response.choices[0].message.content or ""
            parsed = _parse_nurse_response(raw_content)
            reply = parsed.get("reply", "").strip()
            is_complete = bool(parsed.get("is_complete", False))

    return {
        "reply": reply,
        "is_complete": is_complete,
        "doctor_summary": parsed.get("doctor_summary", "").strip(),
        "topic": parsed.get("topic", "").strip(),
        "raw_response": raw_content.strip(),
    }


def get_doctor_summary(
    client: OpenAI,
    chat_history: List[Dict[str, str]],
    prior_history: str,
    patient_context: str,
    model: str,
) -> Dict[str, str]:
    """Run the summarizer agent and return a dict with all 18 topic keys."""

    transcript_lines: List[str] = []
    for msg in chat_history:
        role_label = "Nurse Assistant" if msg["role"] == "assistant" else "Patient"
        transcript_lines.append(f"{role_label}: {msg['content']}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(empty transcript)"

    user_content = (
        f"Chat transcript between nurse assistant and patient:\n\n{transcript}"
    )
    if patient_context.strip():
        user_content += f"\n\nPatient context:\n{patient_context.strip()}"
    else:
        user_content += "\n\nPatient context: (none provided)"

    if prior_history.strip():
        user_content += f"\n\nPrior patient history:\n{prior_history.strip()}"
    else:
        user_content += "\n\nPrior patient history: (none provided)"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        reasoning_effort="minimal",
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content or ""

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        parsed = {}

    result: Dict[str, str] = {}
    for topic in SUMMARY_TOPICS:
        for suffix in ["Main issues", "more details", "status"]:
            key = f"{topic}_{suffix}"
            value = parsed.get(key, "")
            if not isinstance(value, str):
                value = ""
            result[key] = value.strip()
    return result


# =========================
# Google Sheets
# =========================

_sheet = None
_sheet_error: Optional[str] = None


def _secret(name: str, default: Any = None) -> Any:
    try:
        return st.secrets[name]
    except KeyError:
        return default


def _extract_spreadsheet_id(value: str) -> str:
    value = value.strip()
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", value)
    if match:
        return match.group(1)
    return value


def _init_sheets() -> None:
    global _sheet, _sheet_error
    if _sheet is not None or _sheet_error is not None:
        return

    if gspread is None or Credentials is None:
        _sheet_error = (
            "Google Sheets libraries are not installed. Add gspread and "
            "google-auth to your app dependencies."
        )
        return

    try:
        spreadsheet_secret = (
            _secret("gsheet_id")
            or _secret("gsheet_url")
            or _secret("google_sheet_url")
            or _secret("google_sheet_link")
        )
        if not spreadsheet_secret:
            raise ValueError(
                "Missing Google Sheet secret. Add gsheet_id or gsheet_url to Streamlit secrets."
            )

        creds = Credentials.from_service_account_info(
            _secret("gcp_service_account"),
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        spreadsheet_id = _extract_spreadsheet_id(str(spreadsheet_secret))
        book = gspread.authorize(creds).open_by_key(spreadsheet_id)
        try:
            ws = book.worksheet("ChatReport")
        except Exception:
            ws = book.add_worksheet(title="ChatReport", rows=2000, cols=5)
            ws.append_row(["timestamp", "name", "all_data_json", "report"])
        _sheet = ws
    except Exception as exc:
        _sheet_error = str(exc)


def save_to_sheet(name: str, all_data: dict, report: str = "") -> bool:
    """
    Append one row to the Google Sheet.
    Columns: timestamp | name | all_data_json | report
    Returns True on success, False on failure.
    """
    _init_sheets()
    if _sheet is None:
        st.error(f"Could not connect to Google Sheets: {_sheet_error}")
        return False
    try:
        _sheet.append_row(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                name,
                json.dumps(all_data, ensure_ascii=False),
                report,
            ]
        )
        return True
    except Exception as exc:
        st.error(f"Failed to save to Google Sheets: {exc}")
        return False


def build_sheet_payload(
    patient_name: str,
    doctor_name: str,
    therapy_week: str,
    prior_history: str,
    messages: List[Dict[str, str]],
    doctor_summary: str,
    structured_summary: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "patient_name": patient_name.strip(),
        "doctor_name": doctor_name.strip(),
        "therapy_week": therapy_week.strip(),
        "prior_history": prior_history.strip(),
        "transcript": messages,
        "doctor_summary": doctor_summary.strip(),
        "structured_summary": structured_summary,
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

    if "check_in_started" not in st.session_state:
        st.session_state.check_in_started = False

    if "raw_responses" not in st.session_state:
        st.session_state.raw_responses = []

    if "current_topic" not in st.session_state:
        st.session_state.current_topic = ""

    if "completed_topics" not in st.session_state:
        st.session_state.completed_topics = []

    if "doctor_summary_structured" not in st.session_state:
        st.session_state.doctor_summary_structured = {}

    if "summary_generated" not in st.session_state:
        st.session_state.summary_generated = False

    if "sheet_saved" not in st.session_state:
        st.session_state.sheet_saved = False


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.is_complete = False
    st.session_state.doctor_summary = ""
    st.session_state.started = False
    st.session_state.check_in_started = False
    st.session_state.raw_responses = []
    st.session_state.current_topic = ""
    st.session_state.completed_topics = []
    st.session_state.doctor_summary_structured = {}
    st.session_state.summary_generated = False
    st.session_state.sheet_saved = False


def render_topic_boxes() -> None:
    topic_boxes = ""
    for topic in CHAT_TOPICS:
        topic_classes = ["topic-box"]
        if topic in st.session_state.completed_topics:
            topic_classes.append("topic-complete")
        if topic == st.session_state.current_topic:
            topic_classes.append("topic-active")
        topic_boxes += f'<div class="{" ".join(topic_classes)}">{topic}</div>'

    st.markdown(
        f"""
        <style>
            .topic-grid {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 0.5rem;
                margin: 0.75rem 0 1.25rem 0;
            }}
            .topic-box {{
                border: 1px solid rgba(49, 51, 63, 0.18);
                border-radius: 0.45rem;
                padding: 0.55rem 0.65rem;
                text-align: center;
                font-size: 0.86rem;
                font-weight: 600;
                background: rgba(250, 250, 250, 0.78);
            }}
            .topic-complete {{
                border-color: #16a34a;
                background: #f0fdf4;
                color: #166534;
            }}
            .topic-active {{
                border-color: #f59e0b;
                background: #fff7ed;
                color: #9a3412;
                box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.18);
            }}
        </style>
        <div class="topic-grid">{topic_boxes}</div>
        """,
        unsafe_allow_html=True,
    )


def render_completion_banner() -> None:
    st.title("Thank you for submitting your check-in")
    st.success("Your responses have been received and will be shared with your doctor.")
    st.caption(
        "Your check-in is complete. The doctor's summary has been prepared and is "
        "available to your care team."
    )


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


def _basic_md_to_html(text: str) -> str:
    """Minimal markdown -> HTML for content rendered inside our HTML cards.
    Handles HTML escaping, **bold**, *italic*, and bullet lines (- or *)."""
    if not text:
        return ""

    text = html.escape(text)
    text = re.sub(r"\*\*([^*\n]+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<em>\1</em>", text)

    out_lines: List[str] = []
    in_list = False
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        bullet_match = re.match(r"^[-*]\s+(.+)$", line)
        if bullet_match:
            if not in_list:
                out_lines.append('<ul style="margin:0.25rem 0 0.25rem 0; padding-left:1.25rem;">')
                in_list = True
            out_lines.append(f"<li>{bullet_match.group(1)}</li>")
        else:
            if in_list:
                out_lines.append("</ul>")
                in_list = False
            if line:
                out_lines.append(f'<div style="margin-bottom:0.25rem;">{line}</div>')
    if in_list:
        out_lines.append("</ul>")
    return "\n".join(out_lines)


def _limit_to_three_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentence for sentence in sentences[:3] if sentence).strip()


def render_topic_card(
    topic: str,
    main_issues: str,
    more_details: str,
    status: str = "",
) -> None:
    """Render one card per topic. Colored by status; muted when undiscussed."""
    main_text = main_issues.strip()
    detail_text = more_details.strip()
    is_muted = not (main_text or detail_text)

    if is_muted:
        st.markdown(
            f'<div style="border:1px solid rgba(49,51,63,0.18); border-radius:0.5rem; '
            f'padding:0.85rem 1rem; margin-bottom:0.6rem; opacity:0.5;">'
            f"<strong>{html.escape(topic)}</strong> &mdash; "
            f"<em>Not discussed in this check-in</em></div>",
            unsafe_allow_html=True,
        )
        return

    if status == "worse":
        border_color = "#dc2626"
        bg_color = "#fef2f2"
        badge_color = "#dc2626"
        badge_text = "NEW / WORSENING"
    elif status == "better":
        border_color = "#16a34a"
        bg_color = "#f0fdf4"
        badge_color = "#16a34a"
        badge_text = "IMPROVING"
    else:
        border_color = "rgba(49,51,63,0.22)"
        bg_color = "transparent"
        badge_color = ""
        badge_text = ""

    badge_html = ""
    if badge_text:
        badge_html = (
            f'<span style="float:right; background:{badge_color}; color:white; '
            f"padding:0.2rem 0.55rem; border-radius:0.3rem; font-size:0.7rem; "
            f'font-weight:700; letter-spacing:0.04em; margin-top:0.15rem;">'
            f"{badge_text}</span>"
        )

    if main_text:
        main_html = _basic_md_to_html(main_text)
    else:
        main_html = (
            '<div style="color:rgba(49,51,63,0.6); font-style:italic;">'
            "No main issues reported.</div>"
        )

    details_html = ""
    if detail_text:
        details_inner = _basic_md_to_html(detail_text)
        details_html = (
            '<details style="margin-top:0.7rem;">'
            '<summary style="cursor:pointer; font-size:0.88rem; font-weight:500; '
            'color:rgba(49,51,63,0.75);">More details</summary>'
            '<div style="margin-top:0.5rem; font-size:0.92rem;">'
            f"{details_inner}</div></details>"
        )

    st.markdown(
        f'<div style="border:1.5px solid {border_color}; border-radius:0.5rem; '
        f"background:{bg_color}; padding:0.85rem 1rem; margin-bottom:0.6rem;\">"
        f'<div style="font-size:1.05rem; font-weight:700; margin-bottom:0.5rem; '
        f'overflow:hidden;">{html.escape(topic)}{badge_html}</div>'
        f'<div style="font-size:0.95rem; line-height:1.55;">{main_html}</div>'
        f"{details_html}</div>",
        unsafe_allow_html=True,
    )


def render_doctor_summary_page() -> None:
    st.markdown(
        '<div style="color:#16a34a; font-size:0.85rem; font-weight:600; '
        'margin-bottom:0.3rem;">&#10003; Check-in submitted successfully</div>',
        unsafe_allow_html=True,
    )
    st.title("Doctor Summary")
    # st.caption(
    #     "Pre-visit check-in summary for head and neck cancer patient. "
    #     "Cards are listed in clinical-topic order; topics that were not discussed are muted."
    # )

    if not st.session_state.summary_generated:
        st.info(
            "The doctor summary will be available once the patient completes the check-in."
        )
        return

    summary = st.session_state.doctor_summary_structured
    if not summary:
        st.warning(
            "No summary available. Try clicking 'Regenerate doctor summary' in the sidebar."
        )
        return

    if st.session_state.doctor_summary:
        overall_html = _basic_md_to_html(
            _limit_to_three_sentences(st.session_state.doctor_summary)
        )
        st.markdown(
            '<div style="border:1.5px solid rgba(49,51,63,0.3); '
            "background:#f8fafc; padding:0.9rem 1.1rem; "
            "margin:0.75rem 0 1.1rem 0; border-radius:0.5rem; "
            'border-left:4px solid #475569;">'
            '<div style="font-size:0.75rem; font-weight:700; '
            "letter-spacing:0.06em; color:rgba(49,51,63,0.65); "
            'margin-bottom:0.45rem; text-transform:uppercase;">'
            "Overall Summary</div>"
            f'<div style="font-size:0.95rem; line-height:1.55;">{overall_html}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    discussed_count = sum(
        1
        for topic in SUMMARY_TOPICS
        if summary.get(f"{topic}_Main issues", "").strip()
        or summary.get(f"{topic}_more details", "").strip()
    )
    st.caption(
        f"{discussed_count} of {len(SUMMARY_TOPICS)} topics with reported information."
    )

    st.markdown(
        '<div style="display:flex; gap:1rem; font-size:0.78rem; '
        'color:rgba(49,51,63,0.7); margin:0.25rem 0 0.5rem 0;">'
        '<span><span style="display:inline-block; width:0.65rem; height:0.65rem; '
        'background:#dc2626; border-radius:0.15rem; margin-right:0.3rem;"></span>'
        "New / worsening</span>"
        '<span><span style="display:inline-block; width:0.65rem; height:0.65rem; '
        'background:#16a34a; border-radius:0.15rem; margin-right:0.3rem;"></span>'
        "Improving</span>"
        '<span><span style="display:inline-block; width:0.65rem; height:0.65rem; '
        'background:transparent; border:1px solid rgba(49,51,63,0.3); '
        'border-radius:0.15rem; margin-right:0.3rem;"></span>'
        "Unchanged or no comparison</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.write("")  # small spacer

    for topic in SUMMARY_TOPICS:
        main_issues = summary.get(f"{topic}_Main issues", "")
        more_details = summary.get(f"{topic}_more details", "")
        status = summary.get(f"{topic}_status", "")
        render_topic_card(topic, main_issues, more_details, status)


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

    with st.sidebar:
        st.markdown("**Clinical Topics To Cover**")
        topic_boxes_placeholder = st.empty()
        with topic_boxes_placeholder.container():
            render_topic_boxes()

        st.divider()

        st.header("Settings")

        model = "gpt-5-mini"

        system_prompt = st.text_area(
            "Editable chatbot instructions",
            value=SYSTEM_PROMPT,
            height=260,
        )

        patient_name = st.text_input(
            "Patient name *",
            placeholder="Required",
        )

        doctor_name = st.text_input(
            "Doctor name",
            placeholder="Optional",
        )

        therapy_week = st.text_input(
            "Week of therapy",
            placeholder="Example: Week 3",
        )

        prior_history = st.text_area(
            "Prior patient history",
            placeholder=(
                "Optional example: Last visit, patient reported mild swallowing "
                "difficulty and reduced appetite."
            ),
            height=160,
        )

        patient_context = build_patient_context(
            patient_name=patient_name,
            doctor_name=doctor_name,
            therapy_week=therapy_week,
        )

        if st.button("Start new check-in", use_container_width=True):
            if not patient_name.strip():
                st.error("Please enter the patient's name before starting the check-in.")
                st.stop()
            reset_chat()
            st.session_state.check_in_started = True
            st.rerun()

        if st.session_state.is_complete and st.session_state.summary_generated:
            if st.button("Regenerate doctor summary", use_container_width=True):
                st.session_state.summary_generated = False
                st.rerun()

        st.divider()

        if st.session_state.is_complete:
            st.markdown("**Status:** Complete")
        else:
            st.markdown("**Status:** In progress")

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        st.error("OPENAI_API_KEY is missing from Streamlit secrets.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # Generate the doctor summary exactly once after the chat completes.
    if st.session_state.is_complete and not st.session_state.summary_generated:
        with st.spinner("Preparing doctor summary..."):
            try:
                structured = get_doctor_summary(
                    client=client,
                    chat_history=st.session_state.messages,
                    prior_history=prior_history,
                    patient_context=patient_context,
                    model=model,
                )
                st.session_state.doctor_summary_structured = structured
                if not st.session_state.sheet_saved:
                    sheet_payload = build_sheet_payload(
                        patient_name=patient_name,
                        doctor_name=doctor_name,
                        therapy_week=therapy_week,
                        prior_history=prior_history,
                        messages=st.session_state.messages,
                        doctor_summary=st.session_state.doctor_summary,
                        structured_summary=structured,
                    )
                    st.session_state.sheet_saved = save_to_sheet(
                        name=patient_name.strip() or "Unknown patient",
                        all_data=sheet_payload,
                        report=st.session_state.doctor_summary,
                    )
            except Exception as exc:  # surface the error but don't crash the app
                st.warning(f"Could not generate doctor summary: {exc}")
                st.session_state.doctor_summary_structured = {}
            st.session_state.summary_generated = True
        st.rerun()

    # ---- Routing: doctor summary page after submission, otherwise patient chat ----
    if st.session_state.is_complete:
        if st.session_state.summary_generated:
            render_doctor_summary_page()
        else:
            st.info("Preparing your doctor summary...")
        return

    # ---- Patient view ----
    st.title("Nurse Assistant Check-In")
    st.caption("A conversational pre-visit check-in for head and neck cancer care.")

    if not st.session_state.check_in_started:
        st.info("Enter the patient name in the sidebar, add prior patient history if available, then click **Start new check-in** to begin.")
        return

    if not st.session_state.started:
        opening_message = "Are there any symptoms you would like to report to your medical team?"
        add_assistant_message(opening_message)
        st.session_state.started = True

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
                    patient_context=patient_context,
                    model=model,
                    system_prompt=system_prompt,
                )

            assistant_reply = result["reply"]
            st.write(assistant_reply)

        st.session_state.raw_responses.append(result["raw_response"])
        if (
            st.session_state.current_topic
            and st.session_state.current_topic != result["topic"]
            and st.session_state.current_topic not in st.session_state.completed_topics
        ):
            st.session_state.completed_topics.append(st.session_state.current_topic)
        st.session_state.current_topic = result["topic"]

        with topic_boxes_placeholder.container():
            render_topic_boxes()

        add_assistant_message(assistant_reply)

        st.session_state.is_complete = result["is_complete"]
        st.session_state.doctor_summary = result["doctor_summary"]

        if st.session_state.is_complete:
            st.rerun()


if __name__ == "__main__":
    main()
