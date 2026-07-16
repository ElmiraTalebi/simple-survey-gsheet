import json
import os
import html
import re
import hashlib
import csv
import textwrap
from copy import deepcopy
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

PROMPT_VERSION = "virtual-doctor-2026-07-16-v5-clinician-dashboard"

PATIENT_DISCLAIMER = (
    "🤖 You're chatting with an automated assistant — not a person. "
    "This check-in is not monitored in real time. If you have urgent symptoms, "
    "call your nurse triage line. In an emergency, call 911 or go to the nearest ER."
)

# --- Conversation length budget (tune freely) ---
# soft: start prioritizing; wrap: no new topics, move to close; hard: force-close
# so a doctor summary is ALWAYS generated, even for very long conversations.
QUESTION_BUDGET_SOFT = 12
QUESTION_BUDGET_WRAP = 16
QUESTION_BUDGET_HARD = 22
# Maximum number of times the final "anything else" question is asked before the
# check-in closes with a note that remaining items go to the care team.
ANYTHING_ELSE_MAX_ASKS = 2
DEFAULT_MODEL = "gpt-5-mini"
MODEL_PARAMETERS = {
    "reasoning_effort": "minimal",
    "response_format": {"type": "json_object"},
}

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
- Extract every symptom or concern the patient mentions, even if they mention several in one message.
- Acknowledge the patient's concerns.
- Prioritize follow-ups based on what they mention.
- Do not lose track of the initial answer. Refer back to it when relevant.
- Keep an internal pending list of all reported symptoms or concerns until each one has been addressed.

Conversational Behavior Rules:
- Ask exactly one question at a time.
- One question means one clinical variable only. Do not combine questions with "and," "also," "as well," or commas that ask for multiple answers in the same turn.
- Do not ask long lists of questions.
- If a patient answers multiple topics at once, do not repeat questions already answered.
- If a patient reports multiple symptoms or concerns at once, you must eventually address every reported symptom or concern.
- Ask about only one reported symptom or concern per assistant turn.
- Do not move to broad screening, the final "anything else" question, or completion while any reported symptom or concern still needs at least one clinically relevant follow-up or acknowledgement.
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

Question Budget and Scope (IMPORTANT):
- You are collecting information FOR the doctor, not performing a clinical workup. Do not pursue diagnostic lines of questioning (for example: orthostatic-testing patterns for dizziness, sleep-apnea style workups for night breathing, or extended medication-history interrogations).
- Ask at most 4 follow-up questions about the patient's worst or most safety-relevant symptom, and at most 2 follow-up questions about each other reported symptom.
- Aim to finish a typical check-in within about 12 questions in total.
- It is acceptable to leave details uncollected: anything missing will be shown to the doctor as an unresolved item to ask about during the visit. Prefer moving on over drilling down.
- One or two essential details per symptom (onset, severity, or functional impact) are usually enough.

Memory and Redundancy Rules:
- Before every reply, silently review the full conversation and prior history.
- Treat the patient's current answers as already known facts.
- Before every reply, identify any patient-reported symptoms or concerns from the current conversation that have not yet been addressed.
- If any reported symptom or concern has not been addressed, ask the next most clinically important single follow-up about one of those pending symptoms or concerns.
- Only answers from the current conversation count as answered for the current visit.
- Prior history does not count as an answered topic for the current visit.
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



Red Flags (PLACEHOLDER LIST - replace with the clinical team's official list when provided):
Treat the following as red flags when reported: fever of 100.4 F / 38 C or higher (or feeling feverish with chills), suicidal thoughts or thoughts of self-harm, choking on liquids or inability to swallow liquids, coughing or vomiting blood, breathing difficulty at rest, unintentional weight loss of more than 5 pounds since the last visit, severe uncontrolled pain (8 or more out of 10), inability to take medications, almost no food or fluid intake for a day or more, and falls or near-fainting.

When a red flag is reported:
- Acknowledge it calmly and warmly; do not alarm the patient.
- Explicitly say you are flagging it for the care team, for example: "Thank you for telling me - I've flagged this for your care team to see right away."
- Ask at most 1-2 essential follow-up details (such as onset or severity), then move on. Do not perform a workup.
- Do NOT end the conversation because of a red flag; continue the check-in so other symptoms are still collected.
- Never give medical advice or safety instructions beyond reminding the patient that this check-in is not monitored in real time and that urgent concerns should go to the nurse triage line, or emergency services for emergencies.
- Always feature reported red flags prominently in the doctor summary using **bold**.

Use of Patient History Rule:
- If prior patient history is provided, use it as memory, not as a checklist.
- Prior patient history is background context only. Do not treat prior history as the patient's current answer.
- Prior positive symptoms are mandatory to re-check in the current visit unless the patient has already discussed them in the current conversation.
- Prior negative findings do not count as current denials. Re-screen clinically relevant topics in the current conversation.
- A broad answer such as "no," "nothing," "no symptoms," or "no symptoms today" does not count as reassessing prior reported symptoms by name.
- If the patient gives a broad denial and prior history includes reported symptoms, still ask about one prior reported symptom by name.
- If a symptom was present in prior history, briefly check its current status unless the patient has already discussed it in the current conversation.
- Ask whether the prior symptom has improved, worsened, resolved, or stayed the same.
- Do not skip a clinical topic only because it appears in prior history. Prior history should personalize the question, not replace asking about the patient's current condition.
- When prior history lists specific symptoms, ask about those prior symptoms by name at an appropriate point in the conversation, for example: "Last time you mentioned mouth sores. Are those better, worse, resolved, or about the same now?"
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
  "suggested_answers": ["Exactly five brief answers when reply asks a question; otherwise an empty list."],
  "is_complete": true or false,
  "doctor_summary": "A concise 2-3 sentence clinical summary for the doctor if complete, otherwise an empty string.",
  "topic": "one of: Pain, Nutrition, Swallowing, Oral Symptoms, GI Symptoms, Fatigue & Sleep, Activity & Independence, Mood & Support, Other, or empty string"
}

Important:
- The patient should only see the value of "reply".
- Whenever reply asks the patient a question, return exactly five concise, realistic, directly relevant suggested_answers. Make them meaningfully different and, when appropriate, cover positive, negative, neutral, and uncertain responses. They are optional aids, not a questionnaire.
- For the final anything-else question, suggested_answers must be concrete options such as "No, that's all", "I have a question for my doctor", or specific common symptoms not yet discussed (for example "I also have ringing in my ears"). Never include a generic option like "I have another symptom to mention".
- When reply does not ask a question (including the completed closing message), suggested_answers must be an empty list.
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

If the transcript says "Patient selected Finish check-in.", treat it as a UI event rather than a symptom or clinical statement. Do not infer denials for topics that were not covered. Add significant reported concerns that still lacked follow-up to Unresolved_concerns.

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

10. For each topic, also produce:
   - "quote": a short verbatim quote from the PATIENT's own words (12 words or fewer) that best conveys the problem, ONLY for topics whose status is "worse". Copy the exact words from the transcript; never invent, merge, or paraphrase. Empty string otherwise.
   - "coverage": exactly one of "reported" (the patient gave information about this topic), "denied" (the patient was asked about this area and said no/none), or "not_assessed" (the topic never came up in the conversation). This distinction matters clinically: "denied" means the clinician can rely on the negative; "not_assessed" means they still need to ask.

11. Produce "Key_changes": a list of the clinically meaningful changes at this check-in, each as {"topic": one of the listed topics, "direction": "worse" | "new" | "improved", "detail": compact clinical shorthand}. When prior history provides a comparable value, express the change as prior -> current (for example "pain 6/10 -> 8/10" or "weight -5 lbs since last visit"). Keep each detail under 8 words. Use an empty list when nothing meaningful changed.

Response Format:
Always respond as valid JSON with exactly these keys, and no text outside the JSON object:
{
  "Overview": "A concise 1-3 sentence clinical overview without repetition.",
  "Urgent_flags": [
    {"label": "Short flag", "reason": "Patient-reported evidence", "topic": "Closest clinical topic"}
  ],
  "Unresolved_concerns": ["Concern mentioned but not adequately resolved or followed up"],
  "Key_changes": [
    {"topic": "Pain", "direction": "worse", "detail": "pain 6/10 -> 8/10"}
  ],
  "Pain_Main issues": "",
  "Pain_more details": "",
  "Pain_status": "",
  "Pain_quote": "",
  "Pain_coverage": "",
  "Nutrition_Main issues": "",
  "Nutrition_more details": "",
  "Nutrition_status": "",
  "Nutrition_quote": "",
  "Nutrition_coverage": "",
  "Swallowing_Main issues": "",
  "Swallowing_more details": "",
  "Swallowing_status": "",
  "Swallowing_quote": "",
  "Swallowing_coverage": "",
  "Oral Symptoms_Main issues": "",
  "Oral Symptoms_more details": "",
  "Oral Symptoms_status": "",
  "Oral Symptoms_quote": "",
  "Oral Symptoms_coverage": "",
  "GI Symptoms_Main issues": "",
  "GI Symptoms_more details": "",
  "GI Symptoms_status": "",
  "GI Symptoms_quote": "",
  "GI Symptoms_coverage": "",
  "Fatigue & Sleep_Main issues": "",
  "Fatigue & Sleep_more details": "",
  "Fatigue & Sleep_status": "",
  "Fatigue & Sleep_quote": "",
  "Fatigue & Sleep_coverage": "",
  "Activity & Independence_Main issues": "",
  "Activity & Independence_more details": "",
  "Activity & Independence_status": "",
  "Activity & Independence_quote": "",
  "Activity & Independence_coverage": "",
  "Mood & Support_Main issues": "",
  "Mood & Support_more details": "",
  "Mood & Support_status": "",
  "Mood & Support_quote": "",
  "Mood & Support_coverage": "",
  "Other_Main issues": "",
  "Other_more details": "",
  "Other_status": "",
  "Other_quote": "",
  "Other_coverage": ""
}

Rules:
- Every key listed above must be present in your response.
- Urgent_flags must be an empty list when there are no urgent concerns. Flag only patient-reported concerns; do not diagnose. Include fever, suicidal thoughts/severe distress, severe or rapidly worsening pain, breathing difficulty, inability to eat/drink, severe dehydration, bleeding/choking, feeding-tube malfunction, or inability to perform basic daily activities when reported.
- Unresolved_concerns must be an empty list when none can be identified from the transcript.
- Use an empty string for any topic with no relevant information.
- Every "coverage" value must be exactly "reported", "denied", or "not_assessed".
- Every "quote" must be verbatim patient words copied from the transcript, or an empty string. Never fabricate quotes.
- "Key_changes" must be an empty list when there are no meaningful changes.
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

Current Visit Reassessment Requirement:
Prior history is not the patient's current answer. Re-check prior reported symptoms by name during this current visit unless the patient already discussed them in this current conversation. Do not treat prior negative findings as current denials. A broad current denial such as "no symptoms today" does not count as reassessing prior reported symptoms by name.
"""

    system_content += """

Current Conversation Symptom Tracking Requirement:
Review the current conversation before every reply. If the patient has reported multiple symptoms or concerns, make sure every reported symptom or concern is eventually addressed. Do not move to broad screening, the final "anything else" question, or completion while any reported symptom or concern is still pending. Ask about only one pending symptom or concern per turn.
"""

    messages = [{"role": "system", "content": system_content}]
    # Session messages contain UI/evaluation metadata. Only API-supported fields
    # are sent to the model.
    messages.extend(
        {"role": message["role"], "content": message.get("content", "")}
        for message in chat_history
    )
    return messages


def _is_final_open_question(content: str) -> bool:
    """Detect the final open-ended 'anything else' question, tolerating paraphrases."""
    text = content.lower()
    if "anything else" not in text:
        return False
    context_markers = (
        "wrap up", "wrap-up", "wrapping up", "haven't asked", "havent asked",
        "before we finish", "before we end", "before your visit",
        "share with me", "share with your", "care team",
        "like to mention", "haven't covered", "havent covered",
    )
    return any(marker in text for marker in context_markers)


def _is_anything_else_turn(chat_history: List[Dict[str, str]]) -> bool:
    if len(chat_history) < 2:
        return False

    previous_message = chat_history[-2]
    if previous_message.get("role") != "assistant":
        return False

    return _is_final_open_question(previous_message.get("content", ""))


# A single closing clause ("no", "that's all", "I'm good", ...). A patient reply counts
# as a polite closing when EVERY comma/and-separated clause matches one of these.
_CLOSING_CLAUSE = re.compile(
    r"^(?:"
    r"no+|nope|nah|"
    r"nothing(?: else| more| new)?|none|not really|"
    r"that(?:'?s| is) (?:all|it|everything)|"
    r"(?:i'?m|i am) (?:good|fine|done|ok(?:ay)?|all set)|all set|"
    r"no thanks?|no thank you|thanks?|thank you|thanks a lot|"
    r"i don'?t think so|all good|"
    r"no more(?: symptoms| questions| concerns)?|"
    r"i think that'?s (?:all|it)|nothing comes to mind|nothing i can think of|"
    r"that covers everything|that covers it"
    r")$"
)


def _patient_added_new_concern(chat_history: List[Dict[str, str]]) -> bool:
    """True only when the patient's reply to the final open-ended question appears to
    contain new content rather than a polite closing (e.g. "No, that's all")."""
    if not chat_history or chat_history[-1].get("role") != "user":
        return False

    text = chat_history[-1].get("content", "").strip().lower()
    if not text:
        return False

    text = re.sub(r"[.!?\s]+$", "", text).strip()
    clauses = [c.strip() for c in re.split(r"[,;.!?]+|\band\b", text) if c.strip()]
    if clauses and all(_CLOSING_CLAUSE.match(clause) for clause in clauses):
        return False
    return True


def _should_return_to_anything_else(chat_history: List[Dict[str, str]]) -> bool:
    if not chat_history or chat_history[-1].get("role") != "user":
        return False

    for index in range(len(chat_history) - 2, -1, -1):
        message = chat_history[index]
        if message.get("role") != "assistant":
            continue

        if not _is_final_open_question(message.get("content", "")):
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


def _strip_quoted_segments(text: str) -> str:
    """Remove quoted fragments so question marks inside quotes are not counted."""
    text = re.sub(r'"[^"]*"|\u201c[^\u201d]*\u201d|\u2018[^\u2019]*\u2019', " ", text)
    # Straight single quotes only when they are not apostrophes inside words.
    text = re.sub(r"(?<![\w])'[^']*'(?![\w])", " ", text)
    return text


def _reply_has_multiple_questions(reply: str) -> bool:
    unquoted = _strip_quoted_segments(reply)
    if unquoted.count("?") > 1:
        return True
    if "?" not in unquoted:
        return False

    question = unquoted.lower()
    multi_part_patterns = [
        # "where ... and when ..." (two interrogative clauses)
        r"\b(where|when|how|what|which|why)\b[^?]*\band\s+(where|when|how|what|which|why)\b",
        # ", and is it ... / , and are you ..." (second clause led by an auxiliary verb)
        r",?\s+\band\s+(is|are|am|do|does|did|have|has|was|were|can|could|will|would|should)\s+(it|you|they|there|your|the)\b",
    ]
    return any(re.search(pattern, question) for pattern in multi_part_patterns)


def _parse_nurse_response(raw_content: str) -> Dict[str, Any]:
    """Parse and TYPE-COERCE the model output. Guarantees: dict with str reply,
    bool is_complete, str doctor_summary/topic, list suggested_answers."""
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        parsed = {"reply": raw_content}

    if not isinstance(parsed, dict):
        parsed = {}

    reply = parsed.get("reply")
    parsed["reply"] = reply.strip() if isinstance(reply, str) else ""

    raw_complete = parsed.get("is_complete")
    parsed["is_complete"] = raw_complete is True or (
        isinstance(raw_complete, str) and raw_complete.strip().lower() == "true"
    )

    for key in ("doctor_summary", "topic"):
        value = parsed.get(key)
        parsed[key] = value.strip() if isinstance(value, str) else ""

    if not isinstance(parsed.get("suggested_answers"), list):
        parsed["suggested_answers"] = []

    return parsed


def _fallback_suggested_answers(question: str) -> List[str]:
    """Safe, generic fallbacks used only after a malformed model response."""
    lower = question.lower()
    if "anything else" in lower:
        return [
            "No, that covers everything.",
            "No - I'm all set.",
            "I have a question for my doctor.",
            "I'd like to talk about how I'm coping.",
            "I forgot to mention one symptom.",
        ]
    if any(word in lower for word in ("better", "worse", "changed", "same")):
        return [
            "It has improved.",
            "It is about the same.",
            "It has become worse.",
            "It comes and goes.",
            "I am not sure how it has changed.",
        ]
    if lower.startswith(("are you", "do you", "have you", "is there", "can you")):
        return [
            "No, I am not experiencing that.",
            "Yes, but it is mild.",
            "Yes, and it is affecting me.",
            "It happens occasionally.",
            "I am not sure.",
        ]
    return [
        "It is mild.",
        "It is moderate.",
        "It is severe.",
        "It varies over time.",
        "I am not sure.",
    ]


def _validate_suggested_answers(value: Any) -> tuple[List[str], List[str]]:
    errors: List[str] = []
    if not isinstance(value, list):
        return [], ["suggested_answers is not a list"]
    answers = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(answers) != 5:
        errors.append(f"expected 5 suggested answers, received {len(answers)}")
    normalized = {re.sub(r"[^a-z0-9]+", " ", item.lower()).strip() for item in answers}
    if len(normalized) != len(answers):
        errors.append("suggested answers contain duplicates")
    if any(len(item) > 180 for item in answers):
        errors.append("one or more suggested answers are too long")
    return answers[:5], errors


def _normalize_nurse_response(parsed: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    normalized = dict(parsed) if isinstance(parsed, dict) else {}
    reply = normalized.get("reply", "")
    if not isinstance(reply, str):
        reply = ""
    normalized["reply"] = reply.strip()
    asks_question = "?" in normalized["reply"]
    answers, errors = _validate_suggested_answers(normalized.get("suggested_answers", []))
    if asks_question and errors:
        normalized["suggested_answers"] = _fallback_suggested_answers(normalized["reply"])
    elif asks_question:
        normalized["suggested_answers"] = answers
    else:
        normalized["suggested_answers"] = []
        errors = []
    normalized["is_complete"] = normalized.get("is_complete") is True
    for key in ("doctor_summary", "topic"):
        value = normalized.get(key, "")
        normalized[key] = value.strip() if isinstance(value, str) else ""
    return normalized, errors


def _create_chat_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
) -> str:
    """Single place that talks to the API. Only sends reasoning_effort to models
    that support it, so switching DEFAULT_MODEL cannot 400 every call."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": dict(MODEL_PARAMETERS["response_format"]),
    }
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        kwargs["reasoning_effort"] = MODEL_PARAMETERS["reasoning_effort"]
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _find_response_issues(
    parsed: Dict[str, Any],
    chat_history: List[Dict[str, str]],
) -> List[str]:
    """All validation problems for one model response, collected in one pass so a
    single corrective retry can address every issue at once."""
    issues: List[str] = []
    reply = parsed.get("reply", "")

    if not reply:
        issues.append("the reply was empty")
    if _reply_has_multiple_questions(reply):
        issues.append("the reply asked more than one question in a single turn")
    if reply and "?" in reply:
        _, suggestion_errors = _validate_suggested_answers(parsed.get("suggested_answers", []))
        issues.extend(suggestion_errors)

    if _is_anything_else_turn(chat_history) and _patient_added_new_concern(chat_history):
        closing_language = (
            "check-in is complete" in reply.lower()
            or "shared with your doctor" in reply.lower()
        )
        if parsed.get("is_complete") or closing_language:
            issues.append(
                "the check-in was closed although the patient raised a new concern at the "
                "final anything-else question; acknowledge the concern and ask exactly one "
                "targeted follow-up question instead"
            )
    return issues


def get_nurse_response(
    client: OpenAI,
    chat_history: List[Dict[str, str]],
    prior_history: str,
    patient_context: str,
    model: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> Dict[str, Any]:
    def call_model(extra_system: str = "") -> tuple[Dict[str, Any], str]:
        messages = build_messages(chat_history, prior_history, patient_context, system_prompt)
        if extra_system:
            messages.append({"role": "system", "content": extra_system})
        raw = _create_chat_completion(client, model, messages)
        return _parse_nurse_response(raw), raw

    questions_asked = sum(
        1
        for message in chat_history
        if message.get("role") == "assistant" and "?" in message.get("content", "")
    )
    final_question_asks = sum(
        1
        for message in chat_history
        if message.get("role") == "assistant"
        and _is_final_open_question(message.get("content", ""))
    )

    # ---- Hard stops (no API call): guarantee the check-in closes and a doctor
    # summary is generated even for very long or looping conversations. ----
    returning_to_final = _should_return_to_anything_else(chat_history)
    if questions_asked >= QUESTION_BUDGET_HARD or (
        returning_to_final and final_question_asks >= ANYTHING_ELSE_MAX_ASKS
    ):
        reply = (
            "Thank you - we've covered a lot today, and I want to be mindful of your "
            "time. I've noted everything you shared, including anything we didn't get "
            "to explore fully, so your care team can review it before your visit. "
            "Your check-in is now complete."
        )
        return {
            "reply": reply,
            "suggested_answers": [],
            "is_complete": True,
            "doctor_summary": "",
            "topic": "",
            "raw_response": "",
            "validation_errors": [],
            "completion_reason": (
                "question_limit_reached"
                if questions_asked >= QUESTION_BUDGET_HARD
                else "anything_else_limit"
            ),
        }

    # When the patient has answered the single follow-up spawned by the final
    # anything-else question, steer the model back to the final question — but still
    # let it react if the patient's answer contained ANOTHER new symptom. (This used
    # to be a hardcoded reply that skipped the model entirely.)
    steering = ""
    if _should_return_to_anything_else(chat_history):
        steering = (
            "Internal quality check: The patient already received one follow-up question "
            "about the concern they raised at the final anything-else question and has now "
            "answered it. If their last message does not raise another new symptom or "
            "concern, ask the final open-ended question again: \"Before we wrap up, is "
            "there anything else you'd like to share with me - anything I haven't asked "
            "about?\". If it does raise a new symptom or concern, acknowledge it warmly and "
            "ask exactly one targeted follow-up question about it. Either way, is_complete "
            "must be false and doctor_summary must be an empty string."
        )

    # Pacing steering: keep the conversation inside the question budget.
    if QUESTION_BUDGET_WRAP <= questions_asked:
        pacing_note = (
            f"Internal pacing check: You have already asked {questions_asked} questions. "
            "Do not open any new topics. If a reported symptom still lacks one essential "
            "detail, ask only that; otherwise move directly to the final anything-else "
            "question. Uncollected details will be listed for the doctor as unresolved."
        )
        steering = (steering + "\n\n" + pacing_note).strip()
    elif QUESTION_BUDGET_SOFT <= questions_asked:
        pacing_note = (
            f"Internal pacing check: You have already asked {questions_asked} questions. "
            "Be selective from here on: prioritize safety-relevant details, skip optional "
            "follow-ups, and begin moving toward broad screening and the final "
            "anything-else question."
        )
        steering = (steering + "\n\n" + pacing_note).strip()

    parsed, raw_content = call_model(steering)
    issues = _find_response_issues(parsed, chat_history)
    validation_errors: List[str] = list(issues)

    # At most ONE corrective retry per patient turn (latency budget). The retry
    # message lists every detected issue so a single call can fix all of them.
    if issues:
        quality_message = (
            "Internal quality check failed: "
            + "; ".join(issues)
            + ". Return a corrected response: a non-empty, warm reply that asks exactly one "
            "question about one clinical variable, does not repeat information the patient "
            "already gave, includes exactly five brief, distinct, directly relevant "
            "suggested_answers whenever the reply asks a question, and contains all required "
            "JSON keys with no text outside the JSON object."
        )
        if steering:
            quality_message += " " + steering
        retry_parsed, retry_raw = call_model(quality_message)
        if retry_parsed.get("reply"):
            parsed, raw_content = retry_parsed, retry_raw
            validation_errors.extend(
                f"after retry: {issue}"
                for issue in _find_response_issues(parsed, chat_history)
            )

    # ---- Local repairs: no further API calls. ----
    if not parsed.get("reply"):
        parsed["reply"] = (
            "Thank you for sharing that. What is the most important detail about that "
            "for your doctor to know?"
        )
        parsed["is_complete"] = False
        parsed["doctor_summary"] = ""

    # Never close while a newly raised concern is unaddressed, even if the retry
    # also tried to close.
    if (
        parsed.get("is_complete")
        and _is_anything_else_turn(chat_history)
        and _patient_added_new_concern(chat_history)
    ):
        parsed["is_complete"] = False
        parsed["doctor_summary"] = ""

    if not parsed.get("is_complete"):
        parsed["doctor_summary"] = ""

    parsed, normalize_errors = _normalize_nurse_response(parsed)
    validation_errors.extend(
        error for error in normalize_errors if error not in validation_errors
    )

    return {
        "reply": parsed["reply"],
        "suggested_answers": parsed.get("suggested_answers", []),
        "is_complete": bool(parsed.get("is_complete", False)),
        "doctor_summary": parsed.get("doctor_summary", "").strip(),
        "topic": parsed.get("topic", "").strip(),
        "raw_response": raw_content.strip(),
        "validation_errors": validation_errors,
        "completion_reason": "",
    }


def summary_is_degenerate(summary: Dict[str, Any]) -> bool:
    """True when a structured summary contains no usable content at all
    (JSON parse failure or an empty model response)."""
    if not summary:
        return True
    if str(summary.get("Overview", "")).strip():
        return False
    if summary.get("Urgent_flags") or summary.get("Unresolved_concerns"):
        return False
    for topic in SUMMARY_TOPICS:
        for suffix in ("Main issues", "more details", "status"):
            if str(summary.get(f"{topic}_{suffix}", "")).strip():
                return False
    return True


def get_doctor_summary(
    client: OpenAI,
    chat_history: List[Dict[str, str]],
    prior_history: str,
    patient_context: str,
    model: str,
) -> Dict[str, Any]:
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

    def _run_summarizer(extra_note: str = "") -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        if extra_note:
            messages.append({"role": "system", "content": extra_note})
        raw_content = _create_chat_completion(client, model, messages)
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}

        result: Dict[str, Any] = {
            "Overview": parsed.get("Overview", "") if isinstance(parsed.get("Overview", ""), str) else "",
            "Urgent_flags": parsed.get("Urgent_flags", []) if isinstance(parsed.get("Urgent_flags", []), list) else [],
            "Unresolved_concerns": parsed.get("Unresolved_concerns", []) if isinstance(parsed.get("Unresolved_concerns", []), list) else [],
        }
        raw_changes = parsed.get("Key_changes", [])
        result["Key_changes"] = raw_changes if isinstance(raw_changes, list) else []
        for topic in SUMMARY_TOPICS:
            for suffix in ["Main issues", "more details", "status", "quote", "coverage"]:
                key = f"{topic}_{suffix}"
                value = parsed.get(key, "")
                if not isinstance(value, str):
                    value = ""
                value = value.strip()
                if suffix == "coverage" and value not in ("reported", "denied", "not_assessed"):
                    value = ""
                result[key] = value
        return result

    result = _run_summarizer()

    # A degenerate summary (parse failure or entirely empty content) would render
    # an all-muted, colorless dashboard. Retry once automatically instead of
    # making the clinician click "Regenerate summary" by hand.
    if summary_is_degenerate(result):
        result = _run_summarizer(
            "Internal quality check: Your previous response was empty or invalid. "
            "Return the complete JSON object now, with every required key present, "
            "the topic statuses filled in where the transcript supports them, and "
            "no text outside the JSON object."
        )
    return result


# =========================
# Google Sheets
# =========================

_sheet = None
_sheet_error: Optional[str] = None
LOCAL_CHAT_REPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "SurveyResponses - ChatReport.csv",
)

LOCAL_CHAT_REPORT_COLUMNS = [
    "timestamp",
    "session_id",
    "prompt_version",
    "model_name",
    "patient_name",
    "doctor_name",
    "therapy_week",
    "prior_history",
    "completion_status",
    "completion_reason",
    "turn_count",
    "typed_response_count",
    "selected_response_count",
    "urgent_flags_json",
    "overview",
    "unresolved_concerns_json",
    "doctor_summary",
    "structured_summary_json",
    "transcript_json",
    "system_prompt",
    "validation_errors_json",
]


def _secret(name: str, default: Any = None) -> Any:
    try:
        return st.secrets[name]
    except Exception:
        # Missing secrets.toml raises FileNotFoundError, not KeyError.
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
            ws = book.add_worksheet(
                title="ChatReport", rows=2000, cols=len(LOCAL_CHAT_REPORT_COLUMNS)
            )
        current_header = ws.row_values(1)
        if current_header != LOCAL_CHAT_REPORT_COLUMNS:
            # One-time migration from the old synthetic-test layout.
            ws.clear()
            ws.append_row(LOCAL_CHAT_REPORT_COLUMNS)
        _sheet = ws
    except Exception as exc:
        _sheet_error = str(exc)


def _build_clean_report_row(
    name: str,
    all_data: dict,
    report: str = "",
    system_prompt: str = "",
) -> Dict[str, Any]:
    transcript = all_data.get("transcript", [])
    typed_count = sum(
        message.get("role") == "user" and message.get("response_mode") == "typed"
        for message in transcript
    )
    selected_count = sum(
        message.get("role") == "user" and message.get("response_mode") == "selected"
        for message in transcript
    )
    return {
        "timestamp": all_data.get("saved_at") or datetime.now().astimezone().isoformat(),
        "session_id": all_data.get("session_id", ""),
        "prompt_version": all_data.get("prompt_version", ""),
        "model_name": all_data.get("model_name", ""),
        "patient_name": all_data.get("patient_name") or name,
        "doctor_name": all_data.get("doctor_name", ""),
        "therapy_week": all_data.get("therapy_week", ""),
        "prior_history": all_data.get("prior_history", ""),
        "completion_status": (
            all_data.get("completion_reason", "natural_completion")
            in ("natural_completion", "question_limit_reached", "anything_else_limit")
        ),
        "completion_reason": all_data.get("completion_reason", "natural_completion"),
        "turn_count": sum(
            message.get("role") == "user"
            and message.get("response_mode") != "finish_button"
            for message in transcript
        ),
        "typed_response_count": typed_count,
        "selected_response_count": selected_count,
        "urgent_flags_json": json.dumps(all_data.get("urgent_flags", []), ensure_ascii=False),
        "overview": all_data.get("structured_summary", {}).get("Overview", ""),
        "unresolved_concerns_json": json.dumps(
            all_data.get("structured_summary", {}).get("Unresolved_concerns", []),
            ensure_ascii=False,
        ),
        "doctor_summary": report,
        "structured_summary_json": json.dumps(
            all_data.get("structured_summary", {}), ensure_ascii=False
        ),
        "transcript_json": json.dumps(transcript, ensure_ascii=False),
        "system_prompt": system_prompt,
        "validation_errors_json": json.dumps(
            all_data.get("validation_errors", []), ensure_ascii=False
        ),
    }


def save_to_sheet(
    name: str,
    all_data: dict,
    report: str = "",
    system_prompt: str = "",
) -> bool:
    """
    Append one row to the Google Sheet.
    Uses the same analysis-friendly columns as the local CSV.
    Returns True on success, False on failure.
    """
    _init_sheets()
    if _sheet is None:
        st.error(f"Could not connect to Google Sheets: {_sheet_error}")
        return False
    try:
        row = _build_clean_report_row(name, all_data, report, system_prompt)
        _sheet.append_row([row[column] for column in LOCAL_CHAT_REPORT_COLUMNS])
        return True
    except Exception as exc:
        st.error(f"Failed to save to Google Sheets: {exc}")
        return False


def save_to_local_csv(
    name: str,
    all_data: dict,
    report: str = "",
    system_prompt: str = "",
) -> bool:
    """Append one completed check-in using the analysis-friendly local schema."""
    try:
        row = _build_clean_report_row(name, all_data, report, system_prompt)
        file_needs_header = not os.path.exists(LOCAL_CHAT_REPORT_PATH) or os.path.getsize(
            LOCAL_CHAT_REPORT_PATH
        ) == 0
        with open(LOCAL_CHAT_REPORT_PATH, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=LOCAL_CHAT_REPORT_COLUMNS)
            if file_needs_header:
                writer.writeheader()
            writer.writerow(row)
        return True
    except Exception as exc:
        st.error(f"Failed to save to local CSV: {exc}")
        return False


def build_sheet_payload(
    patient_name: str,
    doctor_name: str,
    therapy_week: str,
    prior_history: str,
    messages: List[Dict[str, str]],
    doctor_summary: str,
    structured_summary: Dict[str, Any],
    system_prompt: str = SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    session_id: str = "",
    session_started_at: str = "",
    session_errors: Optional[List[str]] = None,
    completion_reason: str = "natural_completion",
) -> Dict[str, Any]:
    return {
        "schema_version": "check-in-session-v2",
        "session_id": session_id,
        "started_at": session_started_at,
        "saved_at": datetime.now().astimezone().isoformat(),
        "prompt_version": PROMPT_VERSION,
        "system_prompt": system_prompt,
        "model_name": model,
        "model_parameters": deepcopy(MODEL_PARAMETERS),
        "patient_profile": None,
        "patient_name": patient_name.strip(),
        "doctor_name": doctor_name.strip(),
        "therapy_week": therapy_week.strip(),
        "prior_history": prior_history.strip(),
        "transcript": messages,
        "doctor_summary": doctor_summary.strip(),
        "structured_summary": structured_summary,
        "urgent_flags": structured_summary.get("Urgent_flags", []),
        "validation_errors": session_errors or [],
        "completion_reason": completion_reason,
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

    if "show_provider_view" not in st.session_state:
        st.session_state.show_provider_view = False

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

    if "local_csv_saved" not in st.session_state:
        st.session_state.local_csv_saved = False

    if "show_suggestions" not in st.session_state:
        st.session_state.show_suggestions = False

    if "session_started_at" not in st.session_state:
        st.session_state.session_started_at = datetime.now().astimezone().isoformat()

    if "session_id" not in st.session_state:
        st.session_state.session_id = hashlib.sha256(
            st.session_state.session_started_at.encode("utf-8")
        ).hexdigest()[:16]

    if "session_errors" not in st.session_state:
        st.session_state.session_errors = []

    if "completion_reason" not in st.session_state:
        st.session_state.completion_reason = ""

    if "completed_at" not in st.session_state:
        st.session_state.completed_at = ""

    if "saved_prior_history" not in st.session_state:
        st.session_state.saved_prior_history = ""

    if "saved_system_prompt" not in st.session_state:
        st.session_state.saved_system_prompt = SYSTEM_PROMPT

    for key in ("saved_patient_name", "saved_doctor_name", "saved_therapy_week"):
        if key not in st.session_state:
            st.session_state[key] = ""


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.is_complete = False
    st.session_state.doctor_summary = ""
    st.session_state.started = False
    st.session_state.check_in_started = False
    st.session_state.show_provider_view = False
    st.session_state.raw_responses = []
    st.session_state.current_topic = ""
    st.session_state.completed_topics = []
    st.session_state.doctor_summary_structured = {}
    st.session_state.summary_generated = False
    st.session_state.sheet_saved = False
    st.session_state.local_csv_saved = False
    st.session_state.show_suggestions = False
    st.session_state.session_started_at = datetime.now().astimezone().isoformat()
    st.session_state.session_id = hashlib.sha256(
        st.session_state.session_started_at.encode("utf-8")
    ).hexdigest()[:16]
    st.session_state.session_errors = []
    st.session_state.completion_reason = ""
    st.session_state.completed_at = ""
    st.session_state.saved_prior_history = ""
    st.session_state.saved_system_prompt = SYSTEM_PROMPT
    st.session_state.saved_patient_name = ""
    st.session_state.saved_doctor_name = ""
    st.session_state.saved_therapy_week = ""


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
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 760px !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .stButton button {
            min-height: 3rem;
            font-size: 16px !important;
            line-height: 1.45 !important;
            white-space: normal;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Thank you — you're all done")
    st.success(
        "Your answers are saved. A summary will be shared with your care team "
        "before your visit."
    )
    st.info(PATIENT_DISCLAIMER)

    if st.button(
        "View provider summary (care team / research preview)",
        key="completion_provider_view",
        use_container_width=True,
    ):
        st.session_state.show_provider_view = True
        st.rerun()
    if st.button(
        "Start a new check-in",
        key="completion_new_checkin",
        use_container_width=True,
    ):
        reset_chat()
        st.rerun()


def render_chat_history() -> None:
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])
        elif message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(message["content"])


def render_current_suggestions() -> Optional[str]:
    """Render optional suggestions for only the current doctor question."""
    if not st.session_state.messages:
        return None
    current = st.session_state.messages[-1]
    if current.get("role") != "assistant":
        return None
    suggestions = current.get("suggested_answers", [])
    if len(suggestions) != 5:
        return None

    st.caption("Optional — choose one below or type your own answer.")
    for index, suggestion in enumerate(suggestions):
        if st.button(
            suggestion,
            key=f"suggestion_{len(st.session_state.messages)}_{index}",
            use_container_width=True,
        ):
            return suggestion
    return None


def render_raw_responses() -> None:
    if not st.session_state.raw_responses:
        st.info("No GPT responses yet.")
        return

    for index, raw_response in enumerate(st.session_state.raw_responses, start=1):
        st.markdown(f"**Step {index}**")
        st.code(raw_response, language="json")


def _basic_md_to_html(text: str, inline: bool = False) -> str:
    """Minimal markdown -> HTML for content rendered inside our HTML cards.
    Handles HTML escaping, **bold**, *italic*, and bullet lines (- or *).
    Inline mode is used for compact banners, badges, and chips."""
    if not text:
        return ""

    text = html.escape(text)
    text = re.sub(r"\*\*([^*\n]+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<em>\1</em>", text)

    if inline:
        return "<br>".join(line.strip() for line in text.splitlines() if line.strip())

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


def _inline_text(text: str) -> str:
    """HTML-escape plain text (names, dates) for inline display WITHOUT markdown
    conversion, so a patient named **Bob** does not render in bold."""
    if not text:
        return ""
    return "<br>".join(
        html.escape(line.strip()) for line in str(text).splitlines() if line.strip()
    )


_SENTENCE_ABBREVIATIONS = ("Dr.", "Mr.", "Mrs.", "Ms.", "St.", "vs.", "e.g.", "i.e.", "approx.")


def _limit_to_three_sentences(text: str) -> str:
    protected = text.strip()
    for abbreviation in _SENTENCE_ABBREVIATIONS:
        protected = protected.replace(abbreviation, abbreviation.replace(".", "\x00"))
    sentences = re.split(r"(?<=[.!?])\s+", protected)
    limited = " ".join(sentence for sentence in sentences[:3] if sentence).strip()
    return limited.replace("\x00", ".")


def resolve_topic_coverage(main_text: str, detail_text: str, coverage: str) -> str:
    """Backward-compatible coverage resolution: explicit value wins; otherwise
    derive it from the summarizer's text conventions."""
    coverage = (coverage or "").strip().lower()
    if coverage in ("reported", "denied", "not_assessed"):
        return coverage
    if main_text.strip().lower().rstrip(".") == "no issues reported":
        return "denied"
    if main_text.strip() or detail_text.strip():
        return "reported"
    return "not_assessed"


def render_topic_card(
    topic: str,
    main_issues: str,
    more_details: str,
    status: str = "",
    quote: str = "",
    coverage: str = "",
) -> str:
    """Build one fixed-size dashboard card and its non-reflowing detail overlay."""
    main_text = main_issues.strip()
    detail_text = more_details.strip()
    quote_text = quote.strip()
    topic_id = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    coverage = resolve_topic_coverage(main_text, detail_text, coverage)

    # "Not assessed" (never came up) - the clinician still needs to ask.
    if coverage == "not_assessed":
        return (
            f'<article class="topic-card muted" aria-label="{html.escape(topic)}: not assessed">'
            f'<div class="topic-heading"><span>{html.escape(topic)}</span></div>'
            f'<div class="main-issues"><span class="not-discussed">Not assessed this check-in</span></div>'
            f'</article>'
        )

    # "Denied" (screened, patient said none) - a reliable negative, shown quietly.
    if coverage == "denied" and status not in ("worse", "better"):
        return (
            f'<article class="topic-card denied" aria-label="{html.escape(topic)}: screened, none reported">'
            f'<div class="topic-heading"><span>{html.escape(topic)}</span>'
            f'<span class="denied-badge" aria-hidden="true">&#10003;</span></div>'
            f'<div class="main-issues"><span class="denied-note">Screened &mdash; none reported</span></div>'
            f'</article>'
        )

    if status == "worse":
        card_class = "topic-card worse"
        badge_text = "&#9650; NEW / WORSENING"
        badge_class = "worse"
    elif status == "better":
        card_class = "topic-card better"
        badge_text = "&#9660; IMPROVING"
        badge_class = "better"
    else:
        card_class = "topic-card neutral"
        badge_text = ""
        badge_class = ""

    badge_html = (
        f'<span class="status-badge {badge_class}">{badge_text}</span>' if badge_text else ""
    )

    if main_text:
        main_html = _basic_md_to_html(main_text, inline=True)
    else:
        main_html = '<span class="no-main">No main issues reported; details available.</span>'

    # Verbatim patient words for worsening items - clinicians trust the patient's
    # own phrasing over any paraphrase.
    quote_html = (
        f'<div class="pt-quote">&ldquo;{html.escape(quote_text)}&rdquo;</div>'
        if quote_text and status == "worse"
        else ""
    )
    overlay_quote_html = (
        f'<div class="overlay-section"><span class="overlay-label">Patient\'s words</span>'
        f'<em>&ldquo;{html.escape(quote_text)}&rdquo;</em></div>'
        if quote_text
        else ""
    )

    full_main_html = (
        _basic_md_to_html(main_text)
        if main_text
        else '<div class="empty-detail">No main issues supplied.</div>'
    )
    detail_html = (
        _basic_md_to_html(detail_text)
        if detail_text
        else '<div class="empty-detail">No additional details supplied.</div>'
    )
    return (
        f'<div class="topic-card-shell">'
        f'<input class="topic-toggle" type="checkbox" id="details-{topic_id}">'
        f'<article class="{card_class}">'
        f'<div class="topic-heading"><span>{html.escape(topic)}</span>{badge_html}</div>'
        f'<div class="main-issues">{main_html}</div>'
        f'{quote_html}'
        f'<label class="details-trigger" for="details-{topic_id}">More details</label>'
        f'</article>'
        f'<section class="topic-overlay" role="dialog" aria-modal="true" '
        f'aria-labelledby="overlay-title-{topic_id}">'
        f'<label class="overlay-backdrop" for="details-{topic_id}" '
        f'aria-label="Close details"></label>'
        f'<div class="overlay-panel">'
        f'<div class="overlay-heading"><strong id="overlay-title-{topic_id}">'
        f'{html.escape(topic)}</strong>{badge_html}'
        f'<label class="overlay-close" for="details-{topic_id}" aria-label="Close">&times;</label>'
        f'</div>'
        f'{overlay_quote_html}'
        f'<div class="overlay-section"><span class="overlay-label">Main issues</span>'
        f'{full_main_html}</div>'
        f'<div class="overlay-section"><span class="overlay-label">More details</span>'
        f'{detail_html}</div>'
        f'</div></section></div>'
    )

def render_doctor_summary_page() -> None:
    if not st.session_state.summary_generated:
        st.info(
            "The doctor summary will be available once the patient completes the check-in."
        )
        return

    summary = st.session_state.doctor_summary_structured
    if not summary:
        st.warning(
            "No summary is available - the generator hit an error. "
            "Click the button below to try again."
        )
        if st.button("Regenerate summary", key="empty_summary_regenerate"):
            st.session_state.summary_generated = False
            st.rerun()
        return

    if summary_is_degenerate(summary):
        st.warning(
            "The summary generator returned no content for this check-in, so the "
            "dashboard below is empty. Click 'Regenerate summary' at the bottom of "
            "the page (or below) to try again."
        )
        if st.button("Regenerate summary", key="degenerate_summary_regenerate"):
            st.session_state.summary_generated = False
            st.rerun()

    overview = str(summary.get("Overview", "")).strip()
    overview_text = overview or _limit_to_three_sentences(st.session_state.doctor_summary)
    overview_html = _basic_md_to_html(overview_text, inline=True)

    discussed_count = sum(
        1
        for topic in SUMMARY_TOPICS
        if summary.get(f"{topic}_Main issues", "").strip()
        or summary.get(f"{topic}_more details", "").strip()
    )

    started_at = st.session_state.get("session_started_at", "")
    try:
        check_in_datetime = datetime.fromisoformat(started_at).strftime(
            "%b %d, %Y · %I:%M %p"
        )
    except (TypeError, ValueError):
        check_in_datetime = str(started_at) or "Time unavailable"

    patient_name = st.session_state.get("saved_patient_name", "").strip() or "Unnamed patient"
    doctor_name = st.session_state.get("saved_doctor_name", "").strip()
    therapy_week = st.session_state.get("saved_therapy_week", "").strip() or "Week not specified"
    doctor_meta = (
        f'<span><b>Doctor:</b> {_inline_text(doctor_name)}</span>'
        if doctor_name
        else ""
    )

    duration_note = ""
    try:
        completed_at_dt = datetime.fromisoformat(st.session_state.get("completed_at", ""))
        started_at_dt = datetime.fromisoformat(started_at)
        duration_minutes = max(
            1, round((completed_at_dt - started_at_dt).total_seconds() / 60)
        )
        duration_note = f" &middot; took {duration_minutes} min"
    except (TypeError, ValueError):
        pass
    provenance_meta = f'<span class="provenance">Patient-reported{duration_note}</span>' 

    urgent_items: List[str] = []
    for flag in summary.get("Urgent_flags", []):
        if isinstance(flag, dict):
            label = _basic_md_to_html(str(flag.get("label", "Urgent concern")), inline=True)
            reason = _basic_md_to_html(str(flag.get("reason", "")), inline=True)
            topic = _basic_md_to_html(str(flag.get("topic", "")), inline=True)
            reason_html = f'<span class="urgent-reason">{reason}</span>' if reason else ""
            topic_html = f'<span class="urgent-topic">{topic}</span>' if topic else ""
            urgent_items.append(
                f'<span class="urgent-item"><strong>{label}</strong>{reason_html}{topic_html}</span>'
            )
        else:
            urgent_items.append(
                f'<span class="urgent-item"><strong>'
                f'{_basic_md_to_html(str(flag), inline=True)}</strong></span>'
            )

    if urgent_items:
        urgent_html = (
            '<section class="urgent-strip"><span class="urgent-title">URGENT</span>'
            f'<div class="urgent-list">{"".join(urgent_items)}</div></section>'
        )
    else:
        urgent_html = (
            '<section class="clear-strip"><span aria-hidden="true">&#10003;</span> '
            'No urgent concerns reported this check-in</section>'
        )

    unresolved_items = [
        _basic_md_to_html(str(item), inline=True)
        for item in summary.get("Unresolved_concerns", [])
        if str(item).strip()
    ]
    unresolved_html = ""
    if unresolved_items:
        unresolved_html = (
            '<section class="unresolved-strip"><strong>Unresolved:</strong>'
            + "".join(f'<span class="unresolved-chip">{item}</span>' for item in unresolved_items)
            + "</section>"
        )

    cards_html = "".join(
        render_topic_card(
            topic,
            str(summary.get(f"{topic}_Main issues", "")),
            str(summary.get(f"{topic}_more details", "")),
            str(summary.get(f"{topic}_status", "")),
            str(summary.get(f"{topic}_quote", "")),
            str(summary.get(f"{topic}_coverage", "")),
        )
        for topic in SUMMARY_TOPICS
    )

    # ---- "What changed" delta strip: the clinician's 5-second read. ----
    key_changes = summary.get("Key_changes", [])
    if not isinstance(key_changes, list):
        key_changes = []

    def _change_detail(topic_name: str) -> str:
        for change in key_changes:
            if (
                isinstance(change, dict)
                and str(change.get("topic", "")).strip().lower() == topic_name.lower()
            ):
                detail = str(change.get("detail", "")).strip()
                if detail:
                    return detail
        return topic_name

    worse_items: List[str] = []
    better_items: List[str] = []
    unchanged_count = denied_count = not_assessed_count = 0
    for topic in SUMMARY_TOPICS:
        topic_main = str(summary.get(f"{topic}_Main issues", ""))
        topic_details = str(summary.get(f"{topic}_more details", ""))
        topic_status = str(summary.get(f"{topic}_status", "")).strip()
        topic_coverage = resolve_topic_coverage(
            topic_main, topic_details, str(summary.get(f"{topic}_coverage", ""))
        )
        if topic_status == "worse":
            worse_items.append(_basic_md_to_html(_change_detail(topic), inline=True))
        elif topic_status == "better":
            better_items.append(_basic_md_to_html(_change_detail(topic), inline=True))
        elif topic_coverage == "reported":
            unchanged_count += 1
        elif topic_coverage == "denied":
            denied_count += 1
        else:
            not_assessed_count += 1

    delta_parts: List[str] = []
    if worse_items:
        delta_parts.append(
            '<span class="d-item"><b class="worse">&#9650; Worse / new:</b> '
            + ", ".join(worse_items) + "</span>"
        )
    if better_items:
        delta_parts.append(
            '<span class="d-item"><b class="better">&#9660; Improved:</b> '
            + ", ".join(better_items) + "</span>"
        )
    if unchanged_count:
        delta_parts.append(
            f'<span class="d-item">&ndash; Reported, no major change: {unchanged_count}</span>'
        )
    if denied_count:
        delta_parts.append(
            f'<span class="d-item">&#10003; None reported: {denied_count}</span>'
        )
    if not_assessed_count:
        delta_parts.append(
            f'<span class="d-item">&#9675; Not assessed: {not_assessed_count}</span>'
        )
    delta_html = (
        f'<section class="delta-strip" aria-label="What changed">{"".join(delta_parts)}</section>'
        if delta_parts
        else ""
    )

    dashboard_html = textwrap.dedent(f"""
    <style>
      header[data-testid="stHeader"], [data-testid="stSidebar"],
      [data-testid="collapsedControl"] {{ display:none !important; }}
      .block-container {{ max-width:none !important; padding:0.45rem 0.75rem 0.35rem !important; }}
      [data-testid="stAppViewContainer"] {{ background:#ffffff; }}
      .clinical-dashboard {{ color:#1f2937; font-family:inherit; width:100%; }}
      .dashboard-header {{ min-height:28px; display:flex; align-items:center; gap:0.85rem;
        margin-top:0.24rem; padding:0.28rem 0.55rem; border:1px solid #dbe2ea; border-radius:7px;
        background:#f8fafc; font-size:0.78rem; white-space:nowrap; }}
      .dashboard-header > span {{ min-width:0; overflow:hidden; text-overflow:ellipsis; }}
      .dashboard-header .patient {{ flex:1 1 auto; font-size:0.94rem; font-weight:800; color:#111827; }}
      .dashboard-header > span:not(.patient) {{ flex:0 1 auto; }}
      .dashboard-header span:not(.patient) {{ color:#475569; }}
      .urgent-strip {{ display:flex; align-items:flex-start; gap:0.55rem; padding:0.38rem 0.55rem;
        margin-top:0.3rem; border:2px solid #dc2626; border-radius:7px; background:#fef2f2;
        color:#7f1d1d; font-size:0.8rem; line-height:1.25; }}
      .urgent-title {{ flex:0 0 auto; border-radius:4px; background:#dc2626; color:#fff;
        font-size:0.67rem; font-weight:900; letter-spacing:0.08em; padding:0.2rem 0.35rem; }}
      .urgent-list {{ display:flex; flex-wrap:wrap; gap:0.24rem 0.8rem; }}
      .urgent-item {{ display:inline-flex; flex-wrap:wrap; align-items:baseline; gap:0.22rem; }}
      .urgent-item + .urgent-item::before {{ content:"•"; margin-right:0.35rem; color:#dc2626; }}
      .urgent-reason::before {{ content:"— "; }}
      .urgent-topic {{ border:1px solid #fca5a5; border-radius:999px; padding:0 0.28rem;
        background:#fff; font-size:0.68rem; font-weight:700; }}
      .clear-strip {{ margin-top:0.22rem; padding:0.16rem 0.5rem; border-left:3px solid #16a34a;
        border-radius:4px; background:#f0fdf4; color:#166534; font-size:0.72rem; line-height:1.15; }}
      .overview-slot {{ margin-top:0.28rem; padding:0.38rem 0.55rem; border-left:3px solid #475569;
        background:#f8fafc; border-radius:5px; font-size:0.84rem; line-height:1.28; }}
      .overview-slot .slot-label {{ font-size:0.65rem; font-weight:800; letter-spacing:0.06em;
        color:#64748b; text-transform:uppercase; margin-right:0.45rem; }}
      .unresolved-strip {{ display:flex; align-items:center; gap:0.32rem;
        margin-top:0.24rem; min-height:24px; padding:0.18rem 0.45rem; border:1px solid #f59e0b;
        border-radius:5px; background:#fffbeb; color:#92400e; font-size:0.72rem; white-space:nowrap; }}
      .unresolved-chip {{ display:inline-block; padding:0.04rem 0.3rem; border-radius:999px;
        background:#fef3c7; }}
      .legend {{ height:18px; display:flex; align-items:center; justify-content:flex-end; gap:0.65rem;
        color:#64748b; font-size:0.65rem; white-space:nowrap; }}
      .legend i {{ display:inline-block; width:8px; height:8px; margin-right:0.2rem;
        vertical-align:-1px; border-radius:2px; }}
      .legend .red {{ background:#dc2626; }} .legend .green {{ background:#16a34a; }}
      .legend .gray {{ border:1px solid #94a3b8; background:#fff; }}
      .legend .muted-dot {{ border:1px solid #cbd5e1; background:#e2e8f0; }}
      .topic-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
        grid-template-rows:repeat(3,minmax(0,1fr)); gap:0.36rem; height:clamp(315px,44vh,405px); }}
      .topic-card-shell {{ min-width:0; min-height:0; }}
      .topic-card {{ box-sizing:border-box; height:100%; min-height:0; position:relative;
        border-radius:7px; padding:0.42rem 0.5rem 1.55rem; overflow:hidden; }}
      .topic-card.worse {{ border:1.5px solid #dc2626; background:#fef2f2; }}
      .topic-card.better {{ border:1.5px solid #16a34a; background:#f0fdf4; }}
      .topic-card.neutral {{ border:1px solid #cbd5e1; background:#fff; }}
      .topic-card.muted {{ border:1px solid #dbe2ea; background:#f8fafc; opacity:0.55; padding-bottom:0.42rem; }}
      .topic-heading {{ display:flex; align-items:center; gap:0.35rem; min-width:0; margin-bottom:0.24rem;
        font-size:0.82rem; font-weight:800; line-height:1.1; color:#111827; }}
      .topic-heading > span:first-child {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
      .status-badge {{ margin-left:auto; flex:0 0 auto; border-radius:4px; padding:0.12rem 0.28rem;
        color:white; font-size:0.55rem; line-height:1; font-weight:900; letter-spacing:0.035em; }}
      .status-badge.worse {{ background:#dc2626; }}
      .status-badge.better {{ background:#16a34a; }}
      .main-issues {{ display:-webkit-box; -webkit-box-orient:vertical; -webkit-line-clamp:3;
        overflow:hidden; font-size:0.76rem; line-height:1.25; color:#334155; }}
      .main-issues strong {{ color:#111827; }}
      .not-discussed, .no-main {{ color:#64748b; font-style:italic; }}
      .details-trigger {{ position:absolute; left:0.5rem; bottom:0.34rem; cursor:pointer;
        color:#334155; font-size:0.67rem; font-weight:750; text-decoration:underline;
        text-underline-offset:2px; }}
      .topic-toggle {{ position:absolute; opacity:0; pointer-events:none; }}
      .topic-overlay {{ display:none; position:fixed; inset:0; z-index:999999; align-items:center;
        justify-content:center; padding:1rem; }}
      .topic-toggle:checked ~ .topic-overlay {{ display:flex; }}
      .overlay-backdrop {{ position:absolute; inset:0; cursor:pointer; background:rgba(15,23,42,0.48); }}
      .overlay-panel {{ position:relative; z-index:1; width:min(680px,88vw); max-height:72vh;
        overflow-y:auto; border:1px solid #94a3b8; border-radius:10px; background:white;
        padding:0.8rem 0.95rem; box-shadow:0 18px 55px rgba(15,23,42,0.3);
        font-size:0.86rem; line-height:1.35; }}
      .overlay-heading {{ display:flex; align-items:center; gap:0.5rem; padding-bottom:0.45rem;
        margin-bottom:0.45rem; border-bottom:1px solid #e2e8f0; font-size:1rem; }}
      .overlay-heading .status-badge {{ margin-left:0; }}
      .overlay-close {{ margin-left:auto; cursor:pointer; border-radius:4px; color:#475569;
        font-size:1.5rem; line-height:1; padding:0 0.2rem; }}
      .overlay-close:hover {{ background:#f1f5f9; color:#111827; }}
      .overlay-section + .overlay-section {{ margin-top:0.75rem; }}
      .overlay-label {{ display:block; margin-bottom:0.25rem; color:#64748b; font-size:0.65rem;
        font-weight:900; letter-spacing:0.06em; text-transform:uppercase; }}
      .overlay-panel ul {{ margin:0.2rem 0; padding-left:1.2rem; }}
      .empty-detail {{ color:#64748b; font-style:italic; }}
      .delta-strip {{ display:flex; flex-wrap:wrap; gap:0.25rem 1rem; margin-top:0.26rem;
        padding:0.32rem 0.55rem; border:1px solid #dbe2ea; border-left:3px solid #0f172a;
        border-radius:5px; background:#fff; font-size:0.79rem; color:#1f2937; }}
      .delta-strip .d-item {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:100%; }}
      .delta-strip b.worse {{ color:#b91c1c; }}
      .delta-strip b.better {{ color:#166534; }}
      .topic-card.denied {{ border:1px solid #cde8d4; background:#fbfefb; padding-bottom:0.42rem; }}
      .denied-badge {{ margin-left:auto; color:#16a34a; font-weight:900; font-size:0.85rem; }}
      .denied-note {{ color:#4d7c0f; font-size:0.74rem; }}
      .pt-quote {{ margin-top:0.18rem; font-style:italic; color:#7f1d1d; font-size:0.71rem;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
      .provenance {{ font-style:italic; }}
      @media (max-height:800px) {{
        .dashboard-header {{ min-height:25px; padding-top:0.2rem; padding-bottom:0.2rem; }}
        .overview-slot {{ padding-top:0.28rem; padding-bottom:0.28rem; }}
        .topic-grid {{ gap:0.3rem; }}
      }}
    </style>
    <main class="clinical-dashboard">
      {urgent_html}
      {delta_html}
      <header class="dashboard-header">
        <span class="patient">{_inline_text(patient_name)}</span>
        <span><b>Therapy:</b> {_inline_text(therapy_week)}</span>
        <span><b>Check-in:</b> {_inline_text(check_in_datetime)}</span>
        <span><b>Topics:</b> {discussed_count} of {len(SUMMARY_TOPICS)}</span>
        {provenance_meta}
        {doctor_meta}
      </header>
      <section class="overview-slot"><span class="slot-label">At a glance</span>{overview_html or '<em>No overview available.</em>'}</section>
      {unresolved_html}
      <div class="legend" aria-label="Status legend">
        <span><i class="red"></i>&#9650; New / worsening</span><span><i class="green"></i>&#9660; Improving</span>
        <span><i class="gray"></i>Reported</span><span>&#10003; None reported</span><span><i class="muted-dot"></i>&#9675; Not assessed</span>
      </div>
      <section class="topic-grid">{cards_html}</section>
    </main>
    """).strip()

    # The complete primary dashboard is intentionally emitted as one HTML block.
    # Streamlit's markdown parser treats indented lines as code blocks and blank
    # lines as HTML-block terminators. When interpolated content (e.g. bulleted
    # "more details") spans multiple lines, dedent() finds no common indent and
    # the template's 4-space indentation survives - so the raw HTML gets printed
    # as text. Collapsing to a single line makes rendering unconditional.
    dashboard_html = re.sub(r"\s*\n\s*", " ", dashboard_html)
    st.markdown(dashboard_html, unsafe_allow_html=True)

    secondary_columns = st.columns([1, 1, 1.25], gap="small")
    with secondary_columns[0]:
        with st.expander("Prior history"):
            st.write(
                st.session_state.get("saved_prior_history", "")
                or "No prior history provided."
            )
    with secondary_columns[1]:
        with st.expander("Full conversation"):
            for message in st.session_state.messages:
                label = "Virtual doctor" if message["role"] == "assistant" else "Patient"
                mode = message.get("response_mode")
                st.markdown(f"**{label}**" + (f" ({mode})" if mode else ""))
                st.write(message.get("content", ""))
                if message.get("suggested_answers"):
                    st.caption(
                        "Suggestions generated: "
                        + " | ".join(message["suggested_answers"])
                    )

    export_payload = build_sheet_payload(
        patient_name=st.session_state.saved_patient_name,
        doctor_name=st.session_state.saved_doctor_name,
        therapy_week=st.session_state.saved_therapy_week,
        prior_history=st.session_state.saved_prior_history,
        messages=st.session_state.messages,
        doctor_summary=st.session_state.doctor_summary,
        structured_summary=summary,
        system_prompt=st.session_state.saved_system_prompt,
        model=DEFAULT_MODEL,
        session_id=st.session_state.session_id,
        session_started_at=st.session_state.session_started_at,
        session_errors=st.session_state.session_errors,
        completion_reason=st.session_state.completion_reason or "natural_completion",
    )
    with secondary_columns[2]:
        st.download_button(
            "Download reproducibility record (JSON)",
            data=json.dumps(export_payload, ensure_ascii=False, indent=2),
            file_name=f"check_in_{st.session_state.session_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    # The dashboard hides the sidebar for the one-screen layout, so the controls
    # that normally live there must be reachable from this page.
    action_columns = st.columns([1, 1, 2], gap="small")
    with action_columns[0]:
        if st.button("Start new check-in", key="dashboard_new_checkin", use_container_width=True):
            reset_chat()
            st.rerun()
    with action_columns[1]:
        if st.button("Regenerate summary", key="dashboard_regenerate", use_container_width=True):
            st.session_state.summary_generated = False
            st.rerun()


def add_assistant_message(content: str, suggested_answers: Optional[List[str]] = None) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "suggested_answers": suggested_answers or [],
        }
    )


def add_user_message(content: str, response_mode: str = "typed") -> None:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": content,
            "response_mode": response_mode,
        }
    )


# =========================
# Streamlit App
# =========================

def main() -> None:
    st.set_page_config(
        page_title="Nurse Assistant Check-In",
        page_icon="🩺",
        layout="wide",
    )

    initialize_state()

    with st.sidebar:
        st.markdown("**What we'll cover**")
        topic_boxes_placeholder = st.empty()
        with topic_boxes_placeholder.container():
            render_topic_boxes()

        model = DEFAULT_MODEL

        if st.session_state.check_in_started and not st.session_state.is_complete:
            st.markdown(
                """
                <style>
                div.st-key-finish_checkin_sidebar button {
                    background-color: white !important;
                    border: 2px solid #475569 !important;
                    color: #1e293b !important;
                    font-weight: 700 !important;
                    min-height: 3.25rem !important;
                    white-space: normal !important;
                }
                div.st-key-finish_checkin_sidebar button:hover {
                    background-color: #f1f5f9 !important;
                    border-color: #334155 !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "I'm done for now — send my answers to my care team",
                key="finish_checkin_sidebar",
                use_container_width=True,
                help=(
                    "End the check-in now. Responses so far will still be sent "
                    "to the clinician summary."
                ),
            ):
                add_user_message(
                    "Patient selected Finish check-in.",
                    response_mode="finish_button",
                )
                add_assistant_message(
                    "You chose to finish the check-in now. Your responses so far "
                    "will be shared with your doctor."
                )
                st.session_state.completion_reason = "patient_finish_button"
                st.session_state.completed_at = datetime.now().astimezone().isoformat()
                st.session_state.current_topic = ""
                st.session_state.is_complete = True
                st.session_state.doctor_summary = ""
                st.session_state.show_suggestions = False
                st.rerun()

        st.divider()
        with st.expander("🔬 Research team settings", expanded=False):
            system_prompt = st.text_area(
                "Editable chatbot instructions",
                value=SYSTEM_PROMPT,
                height=260,
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
            if st.session_state.is_complete and st.session_state.summary_generated:
                if st.button("Regenerate doctor summary", use_container_width=True):
                    st.session_state.summary_generated = False
                    st.rerun()

    if st.session_state.check_in_started:
        patient_name = st.session_state.saved_patient_name
        doctor_name = st.session_state.saved_doctor_name
        therapy_week = st.session_state.saved_therapy_week
        prior_history = st.session_state.saved_prior_history
        system_prompt = st.session_state.saved_system_prompt
        patient_context = build_patient_context(patient_name, doctor_name, therapy_week)

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        # KeyError when the key is absent; FileNotFoundError when secrets.toml
        # does not exist at all (e.g., fresh local checkout).
        st.error("OPENAI_API_KEY is missing from Streamlit secrets.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # Generate the doctor summary exactly once after the chat completes.
    if st.session_state.is_complete and not st.session_state.summary_generated:
        with st.spinner("Preparing doctor summary..."):
            structured: Dict[str, Any] = {}
            try:
                structured = get_doctor_summary(
                    client=client,
                    chat_history=st.session_state.messages,
                    prior_history=prior_history,
                    patient_context=patient_context,
                    model=model,
                )
                st.session_state.doctor_summary_structured = structured
            except Exception as exc:  # surface the error but don't crash the app
                st.warning(f"Could not generate doctor summary: {exc}")
                st.session_state.doctor_summary_structured = {}
            # Saving is intentionally OUTSIDE the generation try-block: a storage
            # failure must never blank out an already-generated summary.
            try:
                if structured and (
                    not st.session_state.sheet_saved or not st.session_state.local_csv_saved
                ):
                    sheet_payload = build_sheet_payload(
                        patient_name=patient_name,
                        doctor_name=doctor_name,
                        therapy_week=therapy_week,
                        prior_history=prior_history,
                        messages=st.session_state.messages,
                        doctor_summary=st.session_state.doctor_summary,
                        structured_summary=structured,
                        system_prompt=st.session_state.saved_system_prompt,
                        model=model,
                        session_id=st.session_state.session_id,
                        session_started_at=st.session_state.session_started_at,
                        session_errors=st.session_state.session_errors,
                        completion_reason=st.session_state.completion_reason or "natural_completion",
                    )
                    saved_name = patient_name.strip() or "Unknown patient"
                    if not st.session_state.sheet_saved:
                        st.session_state.sheet_saved = save_to_sheet(
                            name=saved_name,
                            all_data=sheet_payload,
                            report=st.session_state.doctor_summary,
                            system_prompt=st.session_state.saved_system_prompt,
                        )
                    if not st.session_state.local_csv_saved:
                        st.session_state.local_csv_saved = save_to_local_csv(
                            name=saved_name,
                            all_data=sheet_payload,
                            report=st.session_state.doctor_summary,
                            system_prompt=st.session_state.saved_system_prompt,
                        )
            except Exception as exc:
                st.warning(f"Could not save the check-in record: {exc}")
            st.session_state.summary_generated = True
        st.rerun()

    # ---- Routing: patient completion screen, with an explicit provider preview ----
    if st.session_state.is_complete:
        if st.session_state.summary_generated:
            if st.session_state.show_provider_view:
                render_doctor_summary_page()
            else:
                render_completion_banner()
        else:
            st.info("Preparing a summary for your care team...")
        return

    # ---- Patient view ----
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 760px !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .patient-card {
            border: 1px solid #cbd5e1;
            border-radius: 1rem;
            padding: 1.25rem;
            background: #ffffff;
        }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        .stButton button,
        .stTextInput input,
        .stChatInput textarea {
            font-size: 16px !important;
            line-height: 1.55 !important;
        }
        .stButton button {
            min-height: 3rem;
            white-space: normal;
        }
        .patient-safety-strip {
            border-left: 4px solid #2563eb;
            background: #eff6ff;
            color: #172554;
            border-radius: 0.45rem;
            padding: 0.65rem 0.8rem;
            margin: 0.25rem 0 1rem;
            font-size: 16px;
            line-height: 1.45;
            font-weight: 600;
        }
        @media (max-width: 480px) {
            .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
            .patient-card { padding: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.check_in_started:
        st.markdown('<div class="patient-card">', unsafe_allow_html=True)
        st.title("Let's check in before your visit")
        st.write(
            "This will take just a few minutes. Your answers will be summed up for "
            "your doctor to review before your visit. You can stop at any time."
        )
        st.warning(PATIENT_DISCLAIMER)
        patient_name = st.text_input(
            "Your name *",
            placeholder="Enter your name",
            key="welcome_patient_name",
        )
        if st.button(
            "Start my check-in",
            key="welcome_start_checkin",
            type="primary",
            use_container_width=True,
        ):
            if not patient_name.strip():
                st.error("Please enter your name to start.")
            else:
                reset_chat()
                st.session_state.saved_prior_history = prior_history
                st.session_state.saved_system_prompt = system_prompt
                st.session_state.saved_patient_name = patient_name
                st.session_state.saved_doctor_name = doctor_name
                st.session_state.saved_therapy_week = therapy_week
                st.session_state.check_in_started = True
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.title("Your check-in")
    st.markdown(
        f'<div class="patient-safety-strip">{html.escape(PATIENT_DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.started:
        opening_message = "Are there any symptoms you would like to report to your medical team?"
        add_assistant_message(
            opening_message,
            [
                "No, I do not have any symptoms to report.",
                "I have pain I would like to discuss.",
                "I am having trouble eating or drinking.",
                "I have several symptoms to report.",
                "I am not sure where to start.",
            ],
        )
        st.session_state.started = True

    render_chat_history()
    selected_answer = render_current_suggestions()

    patient_input = st.chat_input("Type your response...")
    submitted_answer = selected_answer or patient_input

    if submitted_answer:
        response_mode = "selected" if selected_answer else "typed"
        add_user_message(submitted_answer, response_mode=response_mode)
        st.session_state.show_suggestions = False

        with st.chat_message("user", avatar="👤"):
            st.write(submitted_answer)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("The automated assistant is reviewing your response..."):
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
        if result.get("validation_errors"):
            st.session_state.session_errors.extend(result["validation_errors"])
        if (
            st.session_state.current_topic
            and st.session_state.current_topic != result["topic"]
            and st.session_state.current_topic not in st.session_state.completed_topics
        ):
            st.session_state.completed_topics.append(st.session_state.current_topic)
        # A topic the model returns to is active again, not completed.
        if result["topic"] and result["topic"] in st.session_state.completed_topics:
            st.session_state.completed_topics.remove(result["topic"])
        st.session_state.current_topic = result["topic"]

        with topic_boxes_placeholder.container():
            render_topic_boxes()

        add_assistant_message(assistant_reply, result.get("suggested_answers", []))

        st.session_state.is_complete = result["is_complete"]
        if result.get("completion_reason"):
            st.session_state.completion_reason = result["completion_reason"]
        elif st.session_state.is_complete and not st.session_state.completion_reason:
            st.session_state.completion_reason = "natural_completion"
        if st.session_state.is_complete and not st.session_state.completed_at:
            st.session_state.completed_at = datetime.now().astimezone().isoformat()
        st.session_state.doctor_summary = result["doctor_summary"]

        st.rerun()


if __name__ == "__main__":
    main()
