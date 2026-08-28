import json
import os
import html
import re
import hashlib
import csv
import textwrap
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
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

PROMPT_VERSION = "virtual-doctor-2026-07-21-v7-agency-adaptive"

# --- Checkbox-first opening (per the June 5 clinical-team decision) ---
# Wording is a starting point; the clinicians asked to wordsmith the question.
CHECKLIST_QUESTION_FIRST = (
    "Please check everything you are experiencing today."
)
CHECKLIST_QUESTION_RETURNING = (
    "Which of these are new or worse since your last check-in? Check all that apply."
)
CHECKLIST_NONE_LABEL = "None of these — I'm doing okay today"
CHECKLIST_ITEMS = [
    ("Pain", "Pain"),
    ("Mouth sores", "Oral Symptoms"),
    ("Difficulty swallowing", "Swallowing"),
    ("Trouble eating or drinking", "Nutrition"),
    ("Weight loss", "Nutrition"),
    ("Nausea, vomiting, or constipation", "GI Symptoms"),
    ("Fatigue or poor sleep", "Fatigue & Sleep"),
    ("Trouble with daily activities", "Activity & Independence"),
    ("Feeling down, anxious, or depressed", "Mood & Support"),
    ("Breathing problems", "Other"),
    ("Fever or chills", "Other"),
    ("Something else", "Other"),
]
CHECKLIST_PREFIX = "I selected these symptoms on the checklist:"

# --- Patient-facing disclosure (June 5 clinical-team requirement) ---
# The patient must never believe a human is on the other side of the chat, or
# that anyone is watching the answers in real time.
TRIAGE_PHONE = "[TRIAGE PHONE NUMBER]"

WELCOME_TITLE = "Hi {patient_name} 👋"

WELCOME_BODY = (
    "Before your visit, your care team would like a quick check-in about how "
    "you're feeling. It takes about 3–5 minutes, and your answers are "
    "summarized for your doctor to review before you arrive."
)

DISCLAIMER_FULL = (
    "🤖 **Disclaimer: You are chatting with an automated assistant, not a person.** "
    "This check-in is **not monitored in real time**. If you have urgent "
    f"symptoms, call your nurse triage line at **{TRIAGE_PHONE}**. "
    "For emergencies, call 911 or go to the nearest ER."
)

DISCLAIMER_BANNER = (
    f"🤖 Automated assistant · not monitored in real time · "
    f"urgent symptoms → call nurse triage {TRIAGE_PHONE}"
)

WELCOME_BUTTON_LABEL = "I understand — start my check-in"

# --- Conversation length budget (tune freely) ---
# soft: start prioritizing; wrap: no new topics, move to close; hard: force-close
# so a doctor summary is ALWAYS generated, even for very long conversations.
QUESTION_BUDGET_SOFT = 12
QUESTION_BUDGET_WRAP = 16
QUESTION_BUDGET_HARD = 22
# At the wrap (16) and hard (22) thresholds the patient is OFFERED a choice - nothing
# closes silently. This absolute ceiling, a few questions beyond the hard threshold,
# is the final backstop that guarantees the check-in closes and a doctor summary is
# generated even if the patient keeps choosing to continue.
QUESTION_BUDGET_ABSOLUTE_MARGIN = 6

# --- Helper agents (advisor's architecture for the rule-forgetting problem) ---
# As the conversation grows, the single interview agent loses sight of the system
# prompt and over-questions (the fever/vomiting loops). Two lightweight helpers fix
# this without bloating the main prompt:
#   * JUDGE  - a parallel supervisor that reads the conversation with ONLY the pacing
#              rules as its prompt and, one turn later, injects a short "move on"
#              directive into the interview agent. Runs concurrently, so no latency.
#   * SUMMARY - a running per-topic summarizer that compresses each closed topic so the
#              interview agent carries "summary so far + current topic" instead of the
#              whole transcript. Distinct from (and much lighter than) the dashboard
#              summarizer, which is intentionally rich.
# Both are feature-flagged so they can be A/B'd during stress testing.
ENABLE_JUDGE_AGENT = True
ENABLE_ROLLING_SUMMARY = True
# The judge only needs the recent exchange to catch over-questioning; feeding it the
# whole transcript would make it slower than the (compressed) interview call and add
# latency as the chat grows. This caps how many recent messages it reads.
JUDGE_CONTEXT_TAIL = 14
# The judge is best-effort and one-turn-lagged: after the interview reply is ready we
# wait at most this long for the judge's nudge, then move on without it (it will be
# skipped for this turn). This guarantees the judge can never hold up a patient's turn.
# Usually it has already finished during the interview call, so the wait is ~0.
JUDGE_GRACE_SECONDS = 0.75
# Only compress once enough new turns have accumulated to be worth an extra API call,
# so short early topics don't add a blocking summarizer call to every close.
SUMMARY_MIN_NEW_MESSAGES = 12
# Deterministic backstop for the per-symptom follow-up limit the model tends to forget
# (the "billion fever questions" loop). After this many questions in a row on ONE
# topic, code - not the prompt or the judge - forces the interviewer to move on. Set to
# the worst-symptom allowance (4) so a legitimate deep-dive is never cut short, while
# true loops (5+) are stopped for certain.
PER_TOPIC_QUESTION_CAP = 4
# --- Deterministic question quotas (the QDA finding, enforced in code) ---
# The patient's worst symptom gets WORST_SYMPTOM_QUOTA questions, every other selected
# symptom gets OTHER_SYMPTOM_QUOTA. Code - not the model - decides which topic is asked
# next and when the check-in ends, so length is exact rather than emergent:
#   total = 4 + 3*(n-1)  ->  1 symptom: 4, 2: 7, 3: 10, 5: 16
# The patient designates the worst symptom by tapping it; the model never guesses.
# Set per the advisor's request that the check-in was "too short" - he asked for about
# 3 questions per topic (4 on the worst), with 4 as the ceiling.
WORST_SYMPTOM_QUOTA = 4
OTHER_SYMPTOM_QUOTA = 3
# Maximum number of times the final "anything else" question is asked before the
# check-in closes with a note that remaining items go to the care team.
ANYTHING_ELSE_MAX_ASKS = 2
# Used when the patient declines the final "anything else" question but the model
# does not close cleanly on its own.
FINAL_CLOSING_REPLY = (
    "Thank you for sharing all of that with me. Your check-in is complete, and "
    "everything you told me will be shared with your doctor before your visit."
)
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
The check-in normally begins with a symptom CHECKLIST that the patient fills in before the chat. When the first user message starts with "I selected these symptoms on the checklist:", the listed symptoms are the ONLY topics to cover:
- Ask follow-up questions one at a time about the selected symptoms, starting with the most severe or most safety-relevant.
- If the patient selected more than one symptom, do NOT assume on your own which one is worst. Ask them briefly which symptom is bothering them the most right now, and start with that one.
- Do NOT ask about, screen, or mention topics the patient did not select. No broad screening questions. Unselected areas are shown to the provider automatically, so skipping them is safe and expected.
- If the first user message says the patient checked "None of these", acknowledge warmly (do not question their answer) and go directly to the final anything-else question.
- If the patient MENTIONS a symptom they did not select (for example while answering a question about something else), do NOT open it as a new topic and do NOT ask any follow-up questions about it. Briefly acknowledge what they said and continue with the selected topics. Everything they mention is recorded for the doctor automatically. Only topics the patient explicitly selects - on the opening checklist or by adding one during the check-in - may be asked about.

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
- Cover ONLY the symptoms the patient selected on the checklist or raised themselves. Do not force every detailed sub-question. If a patient says no to something, do not ask follow-ups about it.
- After all topics are covered, follow the strict Final Closing Sequence below. Never combine the "anything else" question and the completion into a single turn.
- Occasionally offer guided options when helpful, especially for medications or symptoms patients may not recall precisely.
- Keep tone warm, reassuring, and professional.

Length Control:
- Keep each assistant reply to 1-2 short sentences.
- Do not collect full detail for mild, stable, or denied symptoms.
- If the patient reports several symptoms, triage them in this order: safety/red flags, symptoms the patient says are worst or worsening, symptoms affecting eating/drinking/swallowing/breathing/functioning, then broad screening of remaining topics.
- Once the selected symptoms are addressed, move directly to the final anything-else question - no broad screening.

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
- Do NOT re-ask about prior symptoms that the patient did not select at this check-in. They are shown to the provider automatically; re-discuss a prior symptom only if the patient selects or mentions it again.
- When a symptom the patient DID select also appears in prior history, personalize the follow-up with the prior value, for example: "Last time you rated your throat pain 6 out of 10 - what is it now at its worst?"
- Prior negative findings do not count as current denials.
- Never treat prior history as the patient's current answer.
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

Once every selected symptom (and anything else the patient raised) has been addressed, ask one final open-ended question before completing the check-in.

Turn 1 - The Final Open-Ended Question:
- The message must be a single open-ended question, e.g.:
  "Before we wrap up, is there anything else you'd like to share with me - anything I haven't asked about?"
- Begin the message by briefly naming, in just a few words, the main topics you have already covered together, then ask the open-ended question. For example: "We've talked about your pain and how you've been eating. Before we wrap up, is there anything else you'd like to share with me - anything I haven't asked about?"
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

If the transcript contains a wrap-up action such as "Patient asked to wrap up the check-in." or "Patient selected Finish check-in.", treat it as a UI event rather than a symptom or clinical statement. Do not infer denials for topics that were not covered. Add significant reported concerns that still lacked follow-up to Unresolved_concerns.

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


JUDGE_SYSTEM_PROMPT = """
You are a silent supervisor for a pre-visit nurse check-in chatbot. You do NOT talk to the patient and you never see your output shown to them. You read the recent conversation and decide whether the interviewer's NEXT question is worth asking, then issue a short directive that will be handed to the interviewer.

Do NOT try to count the questions yourself from the transcript, and do NOT enforce numeric limits - a separate system counts reliably and caps how many questions are asked per topic and overall. Instead, you are GIVEN the exact counts and remaining budget (see "Pacing status" at the end of the input). Trust those numbers and use them to prioritize: when little budget remains, be stricter - allow only the single most clinically essential follow-up and otherwise say to move on; when there is plenty of room, a genuinely useful follow-up is fine.

Your job is the judgment that counting cannot make: whether the interviewer is about to ask something that adds little value. Flag ONLY these problems:

- Redundancy: the interviewer is re-asking, or is about to re-ask, something the patient has already answered (including whether a symptom is worse/better, present/absent, constant/intermittent, or medication-related).
- Diminishing returns: the essential clinical detail for the current symptom (roughly onset, severity, and functional impact) is already captured, so a further follow-up would add little for the doctor - it is time to move to the next selected symptom or toward wrap-up.
- Scope drift: the interviewer is heading into diagnostic-workup territory that is not this tool's job (for example orthostatic-testing patterns for dizziness, sleep-apnea style workups, or extended medication-history interrogations).

The interviewer collects information FOR the doctor; it is not doing a clinical workup, and anything left uncollected is simply listed for the doctor as unresolved.

Respond ONLY as valid JSON:
{
  "intervene": true or false,
  "directive": "A short imperative addressed to the interviewer when intervene is true, e.g. 'You already have onset, severity, and how eating is affected for swallowing - that is enough for the doctor; move to the next selected symptom.' Empty string when intervene is false."
}

Intervene ONLY when there is a real quality problem right now. When the interviewer is asking a genuinely useful new question, return {"intervene": false, "directive": ""}. Keep directives specific, one or two sentences, and never invent clinical facts.
"""


ROLLING_SUMMARY_SYSTEM_PROMPT = """
You maintain a very short running summary of a nurse check-in so the interviewer can safely forget the older message-by-message detail. This is NOT a clinical dashboard summary - keep it minimal. The only goal is to let the interviewer know, in one compact line per topic, what has already been established so it does not re-ask.

You are given the previous running summary and the new conversation since then. Return an UPDATED running summary that folds the new conversation into the old one.

Rules:
- One short line per topic already discussed, in the form "Topic: key facts the patient stated".
- Use compact clinical shorthand (severity, onset, frequency, medication + effect). Only facts the patient actually stated.
- Do not add follow-up suggestions, red-flag analysis, formatting, or commentary.
- Do not drop facts that were already in the previous summary; carry them forward.
- Keep the whole thing short.

Respond ONLY as valid JSON: {"summary": "the updated running summary as plain text"}.
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
    rolling_summary: str = "",
    summary_tail_start: int = 0,
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

Prior History Usage Requirement:
Prior history is background context, not the patient's current answer. Do not re-ask prior symptoms the patient did not select or mention at this check-in - they are shown to the provider automatically. When a selected symptom also appears in prior history, compare to the prior value in your follow-up (for example "last time it was 6/10 - what is it now?").
"""

    # When a running summary is available, the earlier part of the transcript is
    # replaced by this compact block so the system prompt keeps its influence and the
    # context stays small. Only the messages from summary_tail_start onward are sent
    # verbatim (the current, not-yet-summarized topic).
    use_summary = bool(rolling_summary.strip()) and summary_tail_start > 0
    if use_summary:
        system_content += f"""

Conversation So Far (compressed):
The earlier part of this check-in has been summarized below to keep you focused. Treat every line as an already-known fact the patient reported - do NOT re-ask any of it. Continue naturally from the recent messages that follow.
{rolling_summary.strip()}
"""

    system_content += """

Current Conversation Symptom Tracking Requirement:
Review the current conversation before every reply. If the patient has reported multiple symptoms or concerns, make sure every reported symptom or concern is eventually addressed. Do not move to broad screening, the final "anything else" question, or completion while any reported symptom or concern is still pending. Ask about only one pending symptom or concern per turn.
"""

    messages = [{"role": "system", "content": system_content}]
    # Session messages contain UI/evaluation metadata. Only API-supported fields
    # are sent to the model. With a running summary, only the recent tail is sent
    # verbatim; the rest lives in the compressed block above.
    history_to_send = chat_history[summary_tail_start:] if use_summary else chat_history
    messages.extend(
        {"role": message["role"], "content": message.get("content", "")}
        for message in history_to_send
    )
    return messages


def _is_final_open_question(content: str) -> bool:
    """Detect the final open-ended 'anything else' question, tolerating paraphrases."""
    text = content.lower()
    if "anything else" not in text:
        return False
    # "anything else about your pain" scopes to one topic; it is a follow-up, not the
    # closing question. Treating it as final would end the check-in with topics unasked.
    if re.search(r"anything else\s+(?:about|regarding|on|with|for|related to)\b", text):
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
    r"(?:(?:i'?m|i am)\s+)?ready(?: to (?:finish|wrap up|be done|go|stop))?|"
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
    # Split on punctuation, "and", AND spaced dashes so "No - I'm ready to finish"
    # separates into closing clauses instead of reading as one unmatched concern.
    clauses = [c.strip() for c in re.split(r"[,;.!?]+|\s[-–—]\s|\band\b", text) if c.strip()]
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


# A second question tacked on with "and": "..., and is it helping?" / "... and when
# did it start?". Matched on the original reply so the cut point maps back to it.
_TRAILING_QUESTION = re.compile(
    r"[,;]?\s+\band\b\s+(?:where|when|how|what|which|why|is|are|am|do|does|did|have|"
    r"has|was|were|can|could|will|would|should)\b",
    re.IGNORECASE,
)


def _repair_multiple_questions(reply: str) -> str:
    """Keep only the first question when the model asks several in one turn. Used as
    a last resort after the corrective retry also came back with a compound question:
    one clinical variable per turn matters more than the dropped half."""
    first_mark = reply.find("?")
    if first_mark != -1 and "?" in reply[first_mark + 1:]:
        return reply[: first_mark + 1].strip()
    match = _TRAILING_QUESTION.search(reply)
    if not match:
        return reply
    kept = reply[: match.start()].rstrip(" ,;")
    if not kept:
        return reply
    return kept if kept.endswith("?") else kept + "?"


# Words that carry no topical meaning, so they must not make two questions look alike.
_QUESTION_STOPWORDS = {
    "what", "when", "where", "which", "would", "could", "should", "have", "has", "had",
    "does", "did", "do", "you", "your", "yours", "are", "is", "was", "were", "been",
    "the", "this", "that", "these", "those", "there", "here", "with", "without",
    "about", "from", "into", "than", "then", "them", "they", "and", "but", "for",
    "any", "some", "more", "most", "much", "many", "just", "also", "still", "been",
    "like", "feel", "feels", "felt", "tell", "know", "sorry", "thanks", "thank",
    "please", "right", "now", "today", "since", "both", "either", "over", "under",
    "can", "cannot", "able", "been", "will", "may", "might", "one", "other",
}


def _stem(word: str) -> str:
    """Crude suffix stripper so "swallow" and "swallowing" compare equal."""
    for suffix in ("ing", "ies", "ied", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _question_keywords(text: str) -> set:
    """Stemmed content words of the QUESTION itself. The empathy preamble ("I'm sorry
    that's bothering you - ...") is dropped, otherwise its words dilute the comparison
    and a genuine re-ask stops looking like one."""
    segments = re.split(r"(?<=[.!?])\s+|\s[—–-]\s", text)
    question_text = " ".join(seg for seg in segments if "?" in seg) or text
    return {
        _stem(word)
        for word in re.findall(r"[a-z]+", question_text.lower())
        if len(word) > 3 and word not in _QUESTION_STOPWORDS
    }


def _looks_redundant(reply: str, chat_history: List[Dict[str, str]], threshold: float = 0.5) -> bool:
    """True when the new question closely repeats one already asked. The judge cannot
    catch this - it runs in parallel and is a turn behind - so redundancy is detected
    here, in the same turn, where the corrective retry can still replace the question.
    Wasting a question matters now that each topic has a fixed quota."""
    if "?" not in reply:
        return False
    new_words = _question_keywords(reply)
    if len(new_words) < 3:
        return False
    for message in chat_history:
        if message.get("role") != "assistant":
            continue
        previous = message.get("content", "")
        if "?" not in previous:
            continue
        old_words = _question_keywords(previous)
        if len(old_words) < 3:
            continue
        # Jaccard (shared / total distinct). Using min() instead made short questions
        # look like repeats purely because they share the topic name.
        union = len(new_words | old_words)
        if union and len(new_words & old_words) / union >= threshold:
            return True
    return False


def _has_closing_language(reply: str) -> bool:
    lower = reply.lower()
    return "check-in is complete" in lower or "shared with your doctor" in lower


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


def _model_parameters(model: str) -> Dict[str, Any]:
    """API parameters a given model actually accepts. Only the reasoning models
    support reasoning_effort, so switching DEFAULT_MODEL cannot 400 every call."""
    parameters: Dict[str, Any] = {
        "response_format": dict(MODEL_PARAMETERS["response_format"]),
    }
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        parameters["reasoning_effort"] = MODEL_PARAMETERS["reasoning_effort"]
    return parameters


def _create_chat_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
) -> str:
    """Single place that talks to the API."""
    response = client.chat.completions.create(
        model=model, messages=messages, **_model_parameters(model)
    )
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
    if reply and _looks_redundant(reply, chat_history):
        issues.append(
            "the reply re-asks a question the patient has already answered; ask a DIFFERENT "
            "question that gathers new information about the assigned topic instead"
        )

    # The patient is told the check-in ended while the app keeps the chat open.
    if reply and _has_closing_language(reply) and not parsed.get("is_complete"):
        issues.append(
            "the reply told the patient the check-in was complete but is_complete was false"
        )

    if _is_anything_else_turn(chat_history) and _patient_added_new_concern(chat_history):
        if parsed.get("is_complete") or _has_closing_language(reply):
            issues.append(
                "the check-in was closed although the patient raised a new concern at the "
                "final anything-else question; acknowledge the concern and ask exactly one "
                "targeted follow-up question instead"
            )
    return issues


def _questions_asked(chat_history: List[Dict[str, str]]) -> int:
    """Number of questions the assistant has asked so far (used for pacing)."""
    return sum(
        1
        for message in chat_history
        if message.get("role") == "assistant" and "?" in message.get("content", "")
    )


def _questions_per_topic(chat_history: List[Dict[str, str]]) -> Dict[str, int]:
    """How many questions have been asked about each topic across the whole check-in.
    Used to decide, in code, when every selected topic has been covered."""
    counts: Dict[str, int] = {}
    for message in chat_history:
        if message.get("role") != "assistant":
            continue
        if "?" not in message.get("content", ""):
            continue
        topic = (message.get("topic") or "").strip()
        if topic:
            counts[topic] = counts.get(topic, 0) + 1
    return counts


def _labels_for_topic(topic: str, selected_labels: List[str]) -> List[str]:
    """The patient's selected symptom labels that map to this clinical topic. Several
    can share one topic (e.g. "Breathing problems" and "Fever or chills" are both
    "Other"), which is exactly why quotas must be counted per SYMPTOM, not per topic."""
    label_to_topic = dict(CHECKLIST_ITEMS)
    return [
        label
        for label in selected_labels
        if label != CHECKLIST_NONE_LABEL and label_to_topic.get(label) == topic
    ]


def _topic_quota(topic: str, worst_label: str, selected_labels: List[str]) -> int:
    """Questions allotted to a topic = the sum of its selected symptoms' quotas (QDA:
    4 for the patient's worst symptom, 2 for each other). So a topic covering two of
    the patient's symptoms gets enough questions for both."""
    labels = _labels_for_topic(topic, selected_labels)
    if not labels:
        return OTHER_SYMPTOM_QUOTA
    return sum(
        WORST_SYMPTOM_QUOTA if label == worst_label else OTHER_SYMPTOM_QUOTA
        for label in labels
    )


def _quota_state(
    chat_history: List[Dict[str, str]],
    selected_topics: List[str],
    worst_label: str,
    selected_labels: List[str],
) -> tuple[str, Dict[str, int], bool]:
    """(next_target_topic, per_topic_counts, all_quotas_met).

    The next target is the first selected topic whose quota is not yet filled, so code -
    not the model - controls topic order, depth, and when the interview is finished."""
    counts = _questions_per_topic(chat_history)
    target = ""
    for topic in selected_topics:
        if counts.get(topic, 0) < _topic_quota(topic, worst_label, selected_labels):
            target = topic
            break
    return target, counts, (target == "")


def _current_topic_question_run(chat_history: List[Dict[str, str]]) -> tuple[str, int]:
    """(topic, count) of the unbroken run of questions the interviewer has just asked
    about the same topic, walking backwards from the latest assistant question. Used to
    deterministically stop a single topic from being over-questioned, regardless of what
    the model remembers. A non-question assistant turn (e.g. a pure acknowledgement)
    does not break the run; a question on a different topic does.

    An UNLABELLED question (the model returned an empty topic, which the prompt permits)
    must NOT end the run - otherwise a single missing label silently switches the cap off
    and lets a topic be over-questioned. It is treated as a continuation of the run."""
    run_topic = None
    count = 0
    for message in reversed(chat_history):
        if message.get("role") != "assistant":
            continue
        if "?" not in message.get("content", ""):
            continue
        topic = (message.get("topic") or "").strip()
        if not topic:
            # Unlabelled question: counts toward whatever run we are in, and if we have
            # not identified the run's topic yet, keep looking further back for it.
            count += 1
            continue
        if run_topic is None:
            # First labelled question found - keep any unlabelled ones already counted.
            run_topic = topic
            count += 1
        elif topic == run_topic:
            count += 1
        else:
            break
    return (run_topic or "", count)


def _effective_budget(symptom_count: int) -> tuple[int, int, int, int]:
    """Adaptive (soft, wrap, hard, absolute) question budget based on how many
    symptoms the patient chose. Per the advisor's formula: about +2 questions per
    extra symptom, anchored so the reference case of 3 symptoms reproduces the fixed
    12/16/22. `soft` = silent speed-up, `wrap` = first agency offer, `hard` = second
    agency offer, `absolute` = the final backstop that force-closes. symptom_count <= 0
    ("none of these" selected, so no symptom count) falls back to the fixed defaults."""
    if not symptom_count or symptom_count <= 0:
        soft, wrap, hard = QUESTION_BUDGET_SOFT, QUESTION_BUDGET_WRAP, QUESTION_BUDGET_HARD
    else:
        soft = max(8, 6 + 2 * symptom_count)
        wrap, hard = soft + 4, soft + 10
    return soft, wrap, hard, hard + QUESTION_BUDGET_ABSOLUTE_MARGIN


def get_nurse_response(
    client: OpenAI,
    chat_history: List[Dict[str, str]],
    prior_history: str,
    patient_context: str,
    model: str,
    system_prompt: str = SYSTEM_PROMPT,
    symptom_count: int = 0,
    extra_steering: str = "",
    rolling_summary: str = "",
    summary_tail_start: int = 0,
) -> Dict[str, Any]:
    def call_model(extra_system: str = "") -> tuple[Dict[str, Any], str]:
        messages = build_messages(
            chat_history,
            prior_history,
            patient_context,
            system_prompt,
            rolling_summary,
            summary_tail_start,
        )
        if extra_system:
            messages.append({"role": "system", "content": extra_system})
        raw = _create_chat_completion(client, model, messages)
        return _parse_nurse_response(raw), raw

    questions_asked = _questions_asked(chat_history)
    soft_budget, wrap_budget, hard_budget, absolute_budget = _effective_budget(symptom_count)
    final_question_asks = sum(
        1
        for message in chat_history
        if message.get("role") == "assistant"
        and _is_final_open_question(message.get("content", ""))
    )

    # ---- Hard stops (no API call): guarantee the check-in closes and a doctor
    # summary is generated even for very long or looping conversations. ----
    returning_to_final = _should_return_to_anything_else(chat_history)
    if questions_asked >= absolute_budget or (
        returning_to_final and final_question_asks >= ANYTHING_ELSE_MAX_ASKS
    ):
        reply = (
            "Thank you - we've covered a lot together, and I have more than enough to "
            "share with your doctor. To be mindful of your time, I'll wrap up here. "
            "I've noted everything you told me, including anything we didn't get to "
            "explore fully, so your care team can review it before your visit. Your "
            "check-in is now complete."
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
                if questions_asked >= absolute_budget
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

    # Pacing steering: keep the conversation inside the (adaptive) question budget.
    if wrap_budget <= questions_asked:
        pacing_note = (
            f"Internal pacing check: You have already asked {questions_asked} questions. "
            "Do not introduce topics the patient did not select. You MAY still cover a "
            "selected topic that has not come up yet, but keep it to one or two essential "
            "questions. If a reported symptom still lacks one essential detail, ask only "
            "that; otherwise move toward the final anything-else question. Uncollected "
            "details will be listed for the doctor as unresolved."
        )
        steering = (steering + "\n\n" + pacing_note).strip()
    elif soft_budget <= questions_asked:
        pacing_note = (
            f"Internal pacing check: You have already asked {questions_asked} questions. "
            "Be selective from here on: prioritize safety-relevant details, skip optional "
            "follow-ups, and begin moving toward broad screening and the final "
            "anything-else question."
        )
        steering = (steering + "\n\n" + pacing_note).strip()

    # Caller-supplied steering (patient pace controls, one-time wrap check-in) is
    # appended last so it takes precedence over the generic pacing notes above.
    if extra_steering.strip():
        steering = (steering + "\n\n" + extra_steering.strip()).strip()

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

    # A compound question survived the retry: keep the first question only, and
    # regenerate suggestions so they match the question the patient is left with.
    if _reply_has_multiple_questions(parsed.get("reply", "")):
        repaired = _repair_multiple_questions(parsed["reply"])
        if repaired != parsed["reply"]:
            parsed["reply"] = repaired
            parsed["suggested_answers"] = _fallback_suggested_answers(repaired)

    # Never close while a newly raised concern is unaddressed, even if the retry
    # also tried to close.
    if (
        parsed.get("is_complete")
        and _is_anything_else_turn(chat_history)
        and _patient_added_new_concern(chat_history)
    ):
        parsed["is_complete"] = False
        parsed["doctor_summary"] = ""

    # The patient answered the final anything-else question with a plain "no": the
    # Closing Turn is mandatory, so close here rather than spend another turn on it.
    # _patient_added_new_concern is conservative, so this only fires on an unambiguous
    # decline - any hint of new content leaves the check-in open.
    if _is_anything_else_turn(chat_history) and not _patient_added_new_concern(chat_history):
        if not parsed.get("is_complete") or "?" in parsed.get("reply", ""):
            parsed["reply"] = FINAL_CLOSING_REPLY
            parsed["suggested_answers"] = []
        parsed["is_complete"] = True

    # "is_complete must NEVER be true on any turn where you ask the patient a
    # follow-up question" - otherwise the app closes on an unanswered question.
    if parsed.get("is_complete") and "?" in parsed.get("reply", ""):
        parsed["is_complete"] = False

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


def _transcript_lines(chat_history: List[Dict[str, str]]) -> str:
    """Plain Nurse/Patient transcript for the helper agents."""
    lines = [
        f"{'Nurse' if message['role'] == 'assistant' else 'Patient'}: {message.get('content', '')}"
        for message in chat_history
    ]
    return "\n".join(lines)


def get_judge_directive(
    client: OpenAI,
    chat_history: List[Dict[str, str]],
    model: str,
    pacing_note: str = "",
) -> str:
    """Parallel supervisor agent. Judges whether the interviewer's next question is
    worth asking and returns a short directive for its NEXT turn (empty string when no
    intervention is warranted). `pacing_note` carries the exact, code-computed counts
    and budgets so the judge can prioritize without having to count the transcript
    itself. Pure - never touches Streamlit state, so it is safe in a worker thread."""
    if not chat_history:
        return ""
    user_content = _transcript_lines(chat_history)
    if pacing_note.strip():
        user_content += f"\n\n{pacing_note.strip()}"
    try:
        raw = _create_chat_completion(
            client,
            model,
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        parsed = json.loads(raw)
    except Exception:
        return ""
    if not isinstance(parsed, dict) or parsed.get("intervene") is not True:
        return ""
    directive = parsed.get("directive", "")
    return directive.strip() if isinstance(directive, str) else ""


def update_rolling_summary(
    client: OpenAI,
    previous_summary: str,
    new_messages: List[Dict[str, str]],
    model: str,
) -> str:
    """Lightweight running summarizer. Folds the conversation since the last summary
    into the previous summary and returns the updated compact text. On any failure it
    returns the previous summary unchanged, so a bad call never loses context."""
    if not new_messages:
        return previous_summary
    user_content = (
        f"Previous running summary:\n{previous_summary or '(none yet)'}\n\n"
        f"New conversation since then:\n{_transcript_lines(new_messages)}"
    )
    try:
        raw = _create_chat_completion(
            client,
            model,
            [
                {"role": "system", "content": ROLLING_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        parsed = json.loads(raw)
    except Exception:
        return previous_summary
    if not isinstance(parsed, dict):
        return previous_summary
    summary = parsed.get("summary", "")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return previous_summary


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


def _load_prior_summary_from_sheet(patient_name: str) -> str:
    """Read this patient's most recent completed check-in from the Google Sheet and
    build a prior-history note from that visit - the overview, the urgent items, AND the
    non-major reported issues per topic - so the next check-in (and the doctor
    dashboard's Prior history) shows the full picture from last time.

    Returns an empty string when there is no prior record, no name, or the sheet is
    unavailable - so it can never break a check-in that has no history."""
    name = (patient_name or "").strip().lower()
    if not name:
        return ""
    _init_sheets()
    if _sheet is None:
        return ""
    try:
        records = _sheet.get_all_records()
    except Exception:
        return ""

    # Rows are appended chronologically, so the last matching row is the latest visit.
    match = None
    for row in records:
        if str(row.get("patient_name", "")).strip().lower() == name:
            match = row
    if not match:
        return ""

    # Prefer the full structured summary; fall back to the flat columns if missing.
    try:
        summary = json.loads(match.get("structured_summary_json") or "{}")
    except Exception:
        summary = {}
    if not isinstance(summary, dict):
        summary = {}

    date_text = ""
    stamp = str(match.get("timestamp", "")).strip()
    if stamp:
        try:
            date_text = datetime.fromisoformat(stamp).strftime("%b %d, %Y")
        except (TypeError, ValueError):
            date_text = stamp[:10]

    sections: List[str] = [
        "Summary of last check-in" + (f" ({date_text})" if date_text else "") + ":"
    ]

    overview = str(summary.get("Overview", "")).strip() or str(match.get("overview", "")).strip()
    if overview:
        sections.append("Overview: " + overview.replace("**", ""))

    # Urgent items (the safety-relevant flags).
    urgent_lines: List[str] = []
    flags = summary.get("Urgent_flags")
    if not isinstance(flags, list):
        try:
            flags = json.loads(match.get("urgent_flags_json") or "[]")
        except Exception:
            flags = []
    for flag in flags if isinstance(flags, list) else []:
        if isinstance(flag, dict):
            label = str(flag.get("label", "")).strip()
            reason = str(flag.get("reason", "")).strip()
            topic = str(flag.get("topic", "")).strip()
            piece = f"{label}: {reason}" if (label and reason) else (label or reason)
            if not piece:
                continue
            if topic:
                piece = f"{piece} ({topic})"
            urgent_lines.append(f"- {piece}")
        elif str(flag).strip():
            urgent_lines.append(f"- {str(flag).strip()}")
    if urgent_lines:
        sections.append("Urgent items:\n" + "\n".join(urgent_lines))

    # Non-major reported issues, per topic, from the structured summary.
    issue_lines: List[str] = []
    for topic in SUMMARY_TOPICS:
        main = str(summary.get(f"{topic}_Main issues", "")).strip()
        if not main or main.lower().rstrip(".") in ("no issues reported", "none reported"):
            continue
        status = str(summary.get(f"{topic}_status", "")).strip()
        tag = f" [{status}]" if status in ("worse", "better") else ""
        issue_lines.append(f"- {topic}{tag}: {main.replace('**', '')}")
    if issue_lines:
        sections.append("Reported issues by topic:\n" + "\n".join(issue_lines))

    if len(sections) == 1:  # header only - nothing was captured
        sections.append("No issues were reported at the last check-in.")
    return "\n".join(sections)


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
        "schema_version": "check-in-session-v4",
        "symptoms_selected": list(st.session_state.get("selected_symptom_labels", [])),
        "disclaimer_acknowledged_at": st.session_state.get("disclaimer_acknowledged_at"),
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

    if "composer_text" not in st.session_state:
        st.session_state.composer_text = ""

    if "clear_composer" not in st.session_state:
        st.session_state.clear_composer = False

    if "closing_review" not in st.session_state:
        st.session_state.closing_review = False

    if "pace_mode" not in st.session_state:
        st.session_state.pace_mode = "normal"

    if "wrap_choice_offered" not in st.session_state:
        st.session_state.wrap_choice_offered = False

    if "stop_choice_offered" not in st.session_state:
        st.session_state.stop_choice_offered = False

    # Which agency offer ("wrap" / "stop") was made last turn, so the next turn knows
    # how to react to the patient's answer. Cleared once handled.
    if "pending_offer" not in st.session_state:
        st.session_state.pending_offer = None

    # What the patient tapped in response ("continue" / "wrap"), so code injects one
    # unambiguous instruction instead of making the model branch.
    if "offer_choice" not in st.session_state:
        st.session_state.offer_choice = None

    # A symptom the patient just added mid-chat: acknowledged now, covered later - the
    # interviewer must not abandon the current topic to jump to it.
    if "addon_notice" not in st.session_state:
        st.session_state.addon_notice = None

    # The worst symptom, chosen by the PATIENT (never inferred). Drives the 4/2 quotas.
    if "worst_topic" not in st.session_state:
        st.session_state.worst_topic = ""

    # The worst SYMPTOM label the patient tapped. Quotas are counted per symptom, so
    # two symptoms sharing a topic (e.g. breathing + fever, both "Other") each get one.
    if "worst_label" not in st.session_state:
        st.session_state.worst_label = ""

    # True while we are waiting for the patient to tap which symptom is worst.
    if "worst_pick_pending" not in st.session_state:
        st.session_state.worst_pick_pending = False

    if "judge_directive" not in st.session_state:
        st.session_state.judge_directive = ""

    if "rolling_summary" not in st.session_state:
        st.session_state.rolling_summary = ""

    if "summary_tail_start" not in st.session_state:
        st.session_state.summary_tail_start = 0

    if "pending_addon" not in st.session_state:
        st.session_state.pending_addon = None

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

    if "selected_topics" not in st.session_state:
        st.session_state.selected_topics = []

    if "selected_symptom_labels" not in st.session_state:
        st.session_state.selected_symptom_labels = []

    if "disclaimer_acknowledged" not in st.session_state:
        st.session_state.disclaimer_acknowledged = False

    if "disclaimer_acknowledged_at" not in st.session_state:
        st.session_state.disclaimer_acknowledged_at = None

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
    st.session_state.raw_responses = []
    st.session_state.current_topic = ""
    st.session_state.completed_topics = []
    st.session_state.doctor_summary_structured = {}
    st.session_state.summary_generated = False
    st.session_state.sheet_saved = False
    st.session_state.local_csv_saved = False
    st.session_state.show_suggestions = False
    st.session_state.composer_text = ""
    st.session_state.clear_composer = False
    st.session_state.closing_review = False
    st.session_state.pace_mode = "normal"
    st.session_state.wrap_choice_offered = False
    st.session_state.stop_choice_offered = False
    st.session_state.pending_offer = None
    st.session_state.offer_choice = None
    st.session_state.addon_notice = None
    st.session_state.worst_topic = ""
    st.session_state.worst_label = ""
    st.session_state.worst_pick_pending = False
    st.session_state.judge_directive = ""
    st.session_state.rolling_summary = ""
    st.session_state.summary_tail_start = 0
    st.session_state.pending_addon = None
    st.session_state.session_started_at = datetime.now().astimezone().isoformat()
    st.session_state.session_id = hashlib.sha256(
        st.session_state.session_started_at.encode("utf-8")
    ).hexdigest()[:16]
    st.session_state.session_errors = []
    st.session_state.completion_reason = ""
    st.session_state.completed_at = ""
    st.session_state.selected_topics = []
    st.session_state.selected_symptom_labels = []
    st.session_state.disclaimer_acknowledged = False
    st.session_state.disclaimer_acknowledged_at = None
    st.session_state.saved_prior_history = ""
    st.session_state.saved_system_prompt = SYSTEM_PROMPT
    st.session_state.saved_patient_name = ""
    st.session_state.saved_doctor_name = ""
    st.session_state.saved_therapy_week = ""


def render_topic_boxes() -> None:
    # Show the SYMPTOMS the patient actually checked (labels), not the deduped clinical
    # topics - otherwise selections that share a topic (e.g. "Breathing problems" and
    # "Fever or chills", both "Other") collapse and "3 checked" reads as "2 topics".
    # Each label's status is derived from its mapped topic. This also matches the budget,
    # which counts labels.
    label_to_topic = dict(CHECKLIST_ITEMS)
    display: List[tuple] = []  # (display_name, topic)
    for label in st.session_state.get("selected_symptom_labels", []):
        if label == CHECKLIST_NONE_LABEL:
            continue
        display.append((label, label_to_topic.get(label, "Other")))
    # Topics that surfaced mid-chat with no checklist label of their own.
    shown_topics = {topic for _, topic in display}
    for topic in st.session_state.get("selected_topics", []):
        if topic not in shown_topics:
            display.append((topic, topic))
            shown_topics.add(topic)
    # Before a check-in (nothing selected yet), preview the full topic list.
    if not display:
        display = [(topic, topic) for topic in CHAT_TOPICS]

    # A topic counts as covered once its question QUOTA is filled - not when the model
    # happens to switch away from it, which left the final topic permanently "uncovered".
    counts = _questions_per_topic(st.session_state.messages)
    worst_label = st.session_state.get("worst_label", "")
    sel_labels = st.session_state.get("selected_symptom_labels", [])

    def _is_covered(topic: str) -> bool:
        return bool(topic) and counts.get(topic, 0) >= _topic_quota(topic, worst_label, sel_labels)

    topic_boxes = ""
    for name, topic in display:
        topic_classes = ["topic-box"]
        if _is_covered(topic):
            topic_classes.append("topic-complete")
        elif topic == st.session_state.current_topic:
            topic_classes.append("topic-active")
        topic_boxes += f'<div class="{" ".join(topic_classes)}">{html.escape(name)}</div>'

    total_topics = len(display)
    done_topics = sum(1 for _, topic in display if _is_covered(topic))
    progress_pct = int((done_topics / total_topics) * 100) if total_topics else 0

    st.markdown(
        f"""
        <style>
            .topic-progress {{
                margin: 0.35rem 0 0.6rem 0;
            }}
            .topic-progress-label {{
                font-size: 0.78rem;
                font-weight: 700;
                color: #334155;
                margin-bottom: 0.3rem;
            }}
            .topic-progress-track {{
                height: 7px;
                border-radius: 999px;
                background: rgba(148, 163, 184, 0.35);
                overflow: hidden;
            }}
            .topic-progress-track > span {{
                display: block;
                height: 100%;
                width: {progress_pct}%;
                background: #16a34a;
                border-radius: 999px;
            }}
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
        <div class="topic-progress">
            <div class="topic-progress-label">Progress: {done_topics} of {total_topics} covered</div>
            <div class="topic-progress-track"><span></span></div>
        </div>
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


def _append_to_composer(suggestion: str) -> None:
    """Add a tapped suggestion to whatever is already in the response box, so the
    patient can stack several options (and their own typed text) before sending."""
    current = st.session_state.composer_text.rstrip()
    if not current:
        st.session_state.composer_text = suggestion
    else:
        # Avoid doubled punctuation: join with a space after end punctuation,
        # otherwise separate the clauses with a comma.
        separator = " " if current[-1] in ".!?,;:" else ", "
        st.session_state.composer_text = current + separator + suggestion


def render_current_suggestions() -> None:
    """Render optional suggestions for only the current doctor question. Tapping a
    suggestion ADDS it to the response box (the patient can tap several and combine
    them with their own words) - it never submits on its own."""
    if not st.session_state.messages:
        return
    current = st.session_state.messages[-1]
    if current.get("role") != "assistant":
        return
    suggestions = current.get("suggested_answers", [])
    if len(suggestions) != 5:
        return

    # Suggestions stay hidden behind the "Suggestions" toggle on every question.
    if st.button("Suggestions", key="toggle_current_suggestions"):
        st.session_state.show_suggestions = not st.session_state.show_suggestions

    if not st.session_state.show_suggestions:
        return

    st.caption(
        "Optional — tap any that apply to add them to your response. You can pick "
        "several, edit the text, or type your own, then press Send."
    )
    for index, suggestion in enumerate(suggestions):
        if st.button(
            suggestion,
            key=f"suggestion_{len(st.session_state.messages)}_{index}",
            use_container_width=True,
        ):
            # Runs before the text box is created, so setting its keyed state is safe
            # and the appended text shows on THIS rerun. No explicit st.rerun() - the
            # button click already triggers one; a second would double the latency.
            _append_to_composer(suggestion)


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


def apply_nurse_result(result: Dict[str, Any], topic_boxes_placeholder=None) -> None:
    """Shared bookkeeping after every nurse-model turn (chat reply or checklist kickoff)."""
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

    # NOTE: topics are never auto-added here. A symptom the patient merely MENTIONS while
    # answering something else must not become a new interview topic - only symptoms the
    # patient explicitly selects (opening checklist, "Add a symptom", or the closing
    # review) are covered. Mentioned symptoms still reach the doctor via the summary.

    if topic_boxes_placeholder is not None:
        with topic_boxes_placeholder.container():
            render_topic_boxes()

    add_assistant_message(
        result["reply"], result.get("suggested_answers", []), result.get("topic", "")
    )

    st.session_state.is_complete = result["is_complete"]
    if result.get("completion_reason"):
        st.session_state.completion_reason = result["completion_reason"]
    elif st.session_state.is_complete and not st.session_state.completion_reason:
        st.session_state.completion_reason = "natural_completion"
    if st.session_state.is_complete and not st.session_state.completed_at:
        st.session_state.completed_at = datetime.now().astimezone().isoformat()
    st.session_state.doctor_summary = result["doctor_summary"]


def render_welcome_screen(patient_name: str) -> None:
    """The disclosure gate shown before any check-in opening (June 5 decision).
    Nothing else may render or run until the patient acknowledges it."""
    display_name = patient_name.strip()
    st.subheader(
        WELCOME_TITLE.format(patient_name=display_name) if display_name else "Hi there 👋"
    )
    st.write(WELCOME_BODY)
    st.warning(DISCLAIMER_FULL)
    if st.button(
        WELCOME_BUTTON_LABEL,
        type="primary",
        key="welcome_acknowledge",
        use_container_width=True,
    ):
        st.session_state.disclaimer_acknowledged = True
        st.session_state.disclaimer_acknowledged_at = datetime.now(timezone.utc).isoformat()
        st.rerun()


def inject_patient_theme() -> None:
    """Warm, calm visual theme for the patient-facing check-in only. Injected inside
    the patient view, so it never reaches the doctor dashboard (which renders on a
    separate path with its own layout)."""
    st.markdown(
        """
        <style>
        :root {
            --pt-bg: #f4f7f9;
            --pt-surface: #ffffff;
            --pt-accent: #0f766e;
            --pt-accent-hover: #0c5f58;
            --pt-accent-soft: #e6f4f2;
            --pt-accent-border: #bfe3dd;
            --pt-text: #1f2933;
            --pt-muted: #667085;
            --pt-border: #e4e9f0;
        }
        /* Hide Streamlit's white top toolbar so there is no empty band above the
           header, and let the soft background run to the very top. */
        header[data-testid="stHeader"] { display: none !important; }
        [data-testid="stAppViewContainer"] { background: var(--pt-bg); }
        .block-container {
            max-width: 760px !important;
            padding-top: 1.6rem !important;
            padding-left: 1.1rem !important;
            padding-right: 1.1rem !important;
        }
        /* Trim the sidebar's top gap that existed to clear the (now hidden) toolbar. */
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-top: 1.2rem;
        }

        /* Friendly header card */
        .pt-header {
            display: flex; align-items: center; gap: 0.85rem;
            background: linear-gradient(135deg, #0f766e 0%, #12a19a 100%);
            color: #ffffff;
            padding: 1rem 1.2rem;
            border-radius: 18px;
            margin-bottom: 1.15rem;
            box-shadow: 0 8px 22px rgba(15, 118, 110, 0.20);
        }
        .pt-header-icon {
            font-size: 1.7rem; line-height: 1;
            background: rgba(255,255,255,0.20);
            width: 46px; height: 46px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            flex: 0 0 auto;
        }
        .pt-header-title { font-size: 1.16rem; font-weight: 800; letter-spacing: -0.01em; }
        .pt-header-sub { font-size: 0.85rem; opacity: 0.93; margin-top: 0.12rem; }

        /* Chat bubbles */
        [data-testid="stChatMessage"] {
            border-radius: 18px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.55rem;
            box-shadow: 0 1px 2px rgba(15,23,42,0.05);
            border: 1px solid var(--pt-border);
            background: var(--pt-surface);
        }
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background: var(--pt-accent-soft);
            border-color: var(--pt-accent-border);
        }
        [data-testid="stChatMessage"] p {
            font-size: 1.02rem; line-height: 1.55; color: var(--pt-text);
        }

        /* Composer text box */
        .stTextArea textarea,
        [data-testid="stTextArea"] textarea {
            border-radius: 14px !important;
            border: 1.5px solid var(--pt-border) !important;
            font-size: 1rem !important;
            padding: 0.75rem 0.9rem !important;
            background: #ffffff !important;
            color: var(--pt-text) !important;
        }
        .stTextArea textarea:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: var(--pt-accent) !important;
            box-shadow: 0 0 0 3px var(--pt-accent-soft) !important;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.15s ease;
        }
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="stBaseButton-primary"] {
            background: var(--pt-accent);
            border: none;
            box-shadow: 0 2px 8px rgba(15,118,110,0.22);
        }
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="stBaseButton-primary"]:hover {
            background: var(--pt-accent-hover);
        }
        .stButton > button[kind="secondary"],
        .stButton > button[data-testid="stBaseButton-secondary"] {
            border-radius: 999px;
            border: 1px solid var(--pt-border);
            background: #ffffff;
            color: var(--pt-text);
        }
        .stButton > button[kind="secondary"]:hover,
        .stButton > button[data-testid="stBaseButton-secondary"]:hover {
            border-color: var(--pt-accent);
            color: var(--pt-accent);
            background: var(--pt-accent-soft);
        }

        /* Checklist + captions + alerts */
        [data-testid="stCheckbox"] label p { font-size: 1rem; }
        [data-testid="stAlert"] { border-radius: 14px; }
        [data-testid="stCaptionContainer"] { color: var(--pt-muted); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_patient_header() -> None:
    st.markdown(
        """
        <div class="pt-header">
          <div class="pt-header-icon">🩺</div>
          <div>
            <div class="pt-header-title">Your pre-visit check-in</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer_banner() -> None:
    """Slim persistent reminder above the chat. Quiet by design - it is a
    reminder, not an alert."""
    st.markdown(
        f"""
        <div style="
            background: rgba(148, 163, 184, 0.16);
            color: #475569;
            border-radius: 6px;
            padding: 6px 10px;
            margin-bottom: 14px;
            font-size: 0.78rem;
            line-height: 1.35;
        ">{html.escape(DISCLAIMER_BANNER)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_symptom_checklist(returning: bool) -> Optional[tuple]:
    """The checkbox-first opening screen. Returns (user_message, labels, topics)
    when the patient clicks Continue, else None."""
    question = CHECKLIST_QUESTION_RETURNING if returning else CHECKLIST_QUESTION_FIRST
    st.subheader(question)
    st.caption(
        "Check all that apply. We'll only ask follow-up questions about the things "
        "you select - everything else is noted for your care team automatically."
    )

    columns = st.columns(2)
    checked_labels: List[str] = []
    for index, (label, _topic) in enumerate(CHECKLIST_ITEMS):
        with columns[index % 2]:
            if st.checkbox(label, key=f"symcb_{index}"):
                checked_labels.append(label)

    other_description = ""
    if "Something else" in checked_labels:
        other_description = st.text_input(
            "Tell us briefly what else is bothering you",
            key="symcb_other_desc",
            max_chars=200,
        )

    none_checked = st.checkbox(
        CHECKLIST_NONE_LABEL, key="symcb_none", disabled=bool(checked_labels)
    )

    can_continue = bool(checked_labels) or none_checked
    if not st.button(
        "Continue",
        type="primary",
        disabled=not can_continue,
        key="checklist_continue",
        use_container_width=True,
    ):
        return None

    if checked_labels:
        detail = ""
        if other_description.strip():
            detail = f' (something else: "{other_description.strip()}")'
        user_message = (
            f"{CHECKLIST_PREFIX} " + ", ".join(checked_labels) + detail + "."
        )
        topics: List[str] = []
        label_to_topic = dict(CHECKLIST_ITEMS)
        for topic in CHAT_TOPICS:
            if any(label_to_topic[l] == topic for l in checked_labels):
                topics.append(topic)
        return question, user_message, checked_labels, topics
    user_message = 'I checked: "None of these - I\'m doing okay today."'
    return question, user_message, [CHECKLIST_NONE_LABEL], []


def add_assistant_message(
    content: str,
    suggested_answers: Optional[List[str]] = None,
    topic: str = "",
) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "suggested_answers": suggested_answers or [],
            "topic": topic,
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


def generate_and_apply_turn(
    client: OpenAI,
    prior_history: str,
    patient_context: str,
    model: str,
    system_prompt: str,
    topic_boxes_placeholder=None,
) -> None:
    """Produce one interview-agent reply for the latest patient message, render it,
    and apply it. Assumes st.session_state.messages already ends with the user message
    to respond to. Shared by the normal send flow and the mid-conversation
    "add a symptom" action so both get identical pacing / judge / summary behavior."""
    symptom_count = len(
        [l for l in st.session_state.selected_symptom_labels if l != CHECKLIST_NONE_LABEL]
    )
    soft_budget, wrap_budget, hard_budget, absolute_budget = _effective_budget(symptom_count)
    questions_so_far = _questions_asked(st.session_state.messages)
    # Current same-topic run, used by the per-topic cap and the add-a-symptom notice.
    run_topic, run_count = _current_topic_question_run(st.session_state.messages)

    # ---- Deterministic quota control ----
    # Code decides which topic is asked next and how deep to go, so the 4/2 split and the
    # overall length are guaranteed rather than left to the model.
    target_topic, topic_counts, quotas_met = _quota_state(
        st.session_state.messages,
        st.session_state.selected_topics,
        st.session_state.worst_label,
        st.session_state.selected_symptom_labels,
    )
    # ---- Deterministic close ----
    # Checked BEFORE generating: the patient has just answered, and every selected topic
    # has filled its quota, so there is nothing left to ask. Doing this after generating
    # would leave the final question asked-but-unanswered. Suppressed when the patient
    # has just asked to keep going.
    if (
        quotas_met
        and st.session_state.selected_topics
        and st.session_state.offer_choice != "continue"
        and not st.session_state.is_complete
    ):
        st.session_state.closing_review = True
        return

    quota_steering = ""
    if target_topic and not quotas_met:
        asked = topic_counts.get(target_topic, 0)
        quota = _topic_quota(
            target_topic, st.session_state.worst_label, st.session_state.selected_symptom_labels
        )
        # Name the patient's ACTUAL symptoms in this topic. Several can share one topic
        # (breathing + fever are both "Other"), and every one of them must be covered -
        # this is what stopped "Breathing problems" being silently skipped.
        topic_labels = _labels_for_topic(
            target_topic, st.session_state.selected_symptom_labels
        )
        symptom_text = ", ".join(topic_labels) if topic_labels else target_topic
        worst_note = (
            f' "{st.session_state.worst_label}" is the symptom the patient said bothers them '
            "most, so cover it most thoroughly."
            if st.session_state.worst_label in topic_labels
            else ""
        )
        multi_note = (
            f" This topic covers {len(topic_labels)} of the patient's symptoms and you must "
            "cover EVERY one of them - do not spend all the questions on just one."
            if len(topic_labels) > 1
            else ""
        )
        quota_steering = (
            f"Topic assignment (follow exactly): ask your next question about {symptom_text}"
            f" (clinical topic: {target_topic})." + worst_note + multi_note
            + f" You have asked {asked} of {quota} planned questions here. Ask ONE question "
            "about this only. Do not ask about any other topic, and do not ask the final "
            "anything-else question yet."
        )

    pace_steering = ""
    if st.session_state.pace_mode == "faster":
        pace_steering = (
            "The patient has asked to move faster or is getting tired. Keep this reply "
            "especially brief, ask only the single most safety-relevant remaining "
            "question, and steer toward wrapping up soon."
        )
    elif st.session_state.pace_mode == "slower":
        pace_steering = (
            "The patient has asked to slow down and share more. Give them room: warmly "
            "acknowledge what they said and invite them to add any detail before you move "
            "on; do not rush ahead."
        )

    # One-time agency check-ins: at the wrap threshold (are you tired? wrap up or keep
    # going) and again at the hard threshold (we have enough - stop, or continue).
    # Nothing closes silently; the patient always chooses. The higher threshold is
    # checked first so a big jump lands on the right message.
    # Covered / still-to-come selected topics, used by both the offers and the
    # follow-up that reacts to the patient's answer.
    covered_topics = list(st.session_state.completed_topics)
    if (
        st.session_state.current_topic
        and st.session_state.current_topic not in covered_topics
    ):
        covered_topics.append(st.session_state.current_topic)
    remaining_topics = [
        t for t in st.session_state.selected_topics if t not in covered_topics
    ]
    covered_text = ", ".join(covered_topics) if covered_topics else "your main concerns"
    remaining_text = ", ".join(remaining_topics)

    # The patient is answering an offer we made last turn - tell the interviewer how to
    # honour either choice, so "let's continue" actually delivers what was promised.
    # A symptom was just added mid-chat: acknowledge it, but finish what is already in
    # progress first. The interviewer must not abandon the current topic to jump to it.
    addon_steering = ""
    addon_notice = st.session_state.addon_notice
    if addon_notice:
        st.session_state.addon_notice = None
        addon_steering = (
            f'The patient has just added "{addon_notice}" to their list. It has ALREADY been '
            "acknowledged for you, so do not thank them for it or mention it again, and do NOT "
            "switch to it now. Simply continue with the topic you are currently on"
            + (f" ({run_topic})" if run_topic else "")
            + (
                f", then cover the selected topics not yet discussed ({remaining_text})"
                if remaining_text
                else ""
            )
            + f', and only after those ask about "{addon_notice}". This turn, continue with the '
            "current topic's next question."
        )

    offer_followup = ""
    pending_offer = st.session_state.pending_offer
    offer_choice = st.session_state.offer_choice
    if pending_offer:
        st.session_state.pending_offer = None
        st.session_state.offer_choice = None
        # One strong, UNCONDITIONAL instruction per (threshold, choice). The model is
        # never asked to work out which branch applies - code already knows.
        if pending_offer == "wrap" and offer_choice == "continue":
            offer_followup = (
                "The patient explicitly chose to KEEP GOING. Continue the check-in now: cover the "
                "selected topics that have not come up yet"
                + (f" ({remaining_text})" if remaining_text else "")
                + ". Ask brief, essential questions, one at a time, and do not introduce topics "
                "the patient did not select. Do not offer to wrap up again."
            )
        elif pending_offer == "wrap" and offer_choice == "wrap":
            offer_followup = (
                "The patient explicitly chose to WRAP UP. Do not open any further topics. Ask at "
                "most ONE remaining essential question if something critical is missing; "
                "otherwise go straight to the final anything-else question."
            )
        elif pending_offer == "stop" and offer_choice == "continue":
            offer_followup = (
                "The patient explicitly chose to KEEP GOING even though you already have more "
                "than enough for the doctor. Let THEM lead: warmly invite them to tell you what "
                "else is on their mind and follow what they raise. Do not open further topics on "
                "your own initiative, and keep it brief."
            )
        elif pending_offer == "stop" and offer_choice == "wrap":
            offer_followup = (
                "The patient explicitly chose to FINISH. Do not ask any further clinical "
                "questions - go straight to the final anything-else question."
            )
        elif pending_offer == "wrap":
            # No button tapped (they typed a free-text reply) - let the model read it.
            offer_followup = (
                "The patient has just answered your wrap-up-or-continue question in their own "
                "words. Follow what they said: if they want to keep going, cover the selected "
                "topics that have not come up yet"
                + (f" ({remaining_text})" if remaining_text else "")
                + " with brief, essential questions; if they want to wrap up, ask at most one "
                "essential question and then go to the final anything-else question."
            )
        else:
            offer_followup = (
                "The patient has just answered your finish-or-continue question in their own "
                "words. Follow what they said: if they want to keep going, let THEM lead and "
                "follow what they raise, briefly; if they want to finish, go to the final "
                "anything-else question."
            )

    # ---- The wrap-up questions, at Slobodan's counts ----
    # 12 = silent (internal speed-up only), 16 = "are you tired, wrap up or continue?",
    # 22 = "we have enough for the doctor - stop, or keep going?". Triggered on the
    # QUESTION COUNT exactly as he specified, not on quota progress.
    topics_still_to_cover = [
        t
        for t in st.session_state.selected_topics
        if topic_counts.get(t, 0)
        < _topic_quota(t, st.session_state.worst_label, st.session_state.selected_symptom_labels)
    ]
    agency_steering = ""
    if questions_so_far >= hard_budget and not st.session_state.stop_choice_offered:
        agency_steering = (
            "Wrap-up check-in (do this once, this turn only): The check-in has become quite "
            "long. This turn, do NOT ask another clinical question. Instead, warmly reassure "
            "the patient that you already have more than enough to share with their doctor, and "
            "offer a clear choice: you can stop here now, or keep going if they want to provide "
            "more. Make it entirely their choice, with no pressure either way. is_complete must "
            "be false and doctor_summary must be an empty string."
        )
        quota_steering = ""
        st.session_state.stop_choice_offered = True
        st.session_state.wrap_choice_offered = True
        st.session_state.pending_offer = "stop"
    elif questions_so_far >= wrap_budget and not st.session_state.wrap_choice_offered:
        done_text = ", ".join(
            t for t in st.session_state.selected_topics if t not in topics_still_to_cover
        )
        agency_steering = (
            "Wrap-up check-in (do this once, this turn only): This turn, do NOT ask another "
            "clinical question. Instead, warmly tell the patient what you have discussed so far"
            + (f" ({done_text})" if done_text else "")
            + (
                f", note that these were not discussed yet ({', '.join(topics_still_to_cover)})"
                if topics_still_to_cover
                else ""
            )
            + ", say you wonder whether they are already getting tired, and offer a clear "
            "choice: you can wrap up quickly, or keep going over the remaining topics. Make it "
            "entirely their choice, with no pressure. is_complete must be false and "
            "doctor_summary must be an empty string."
        )
        quota_steering = ""
        st.session_state.wrap_choice_offered = True
        st.session_state.pending_offer = "wrap"

    # Supervisor directive from the PREVIOUS turn's judge (one-turn lag, so the judge
    # never sits in the critical path). Applied as the highest-priority steering.
    directive_steering = ""
    if ENABLE_JUDGE_AGENT and st.session_state.judge_directive:
        directive_steering = (
            "Supervisor directive (a pacing check has flagged this - follow it now): "
            + st.session_state.judge_directive
        )
        st.session_state.judge_directive = ""

    # Deterministic per-topic cap: the hard backstop for the follow-up limit the model
    # forgets. If it has already asked PER_TOPIC_QUESTION_CAP questions in a row about
    # one topic, code (not the prompt) forbids another and forces a move.
    cap_steering = ""

    # Exact, code-computed counts handed to the judge so it can prioritize (it never
    # counts the transcript itself). Enforcement still lives entirely in code.
    pace_line = {
        "faster": "\n- The patient pressed Speed up (getting tired): favor wrapping up, and "
        "be strict - flag any question that is not clearly essential.",
        "slower": "\n- The patient pressed Slow down (wants to share more): be lenient - do "
        "not flag a useful follow-up just because the essentials are captured.",
    }.get(st.session_state.pace_mode, "")
    judge_pacing_note = (
        "Pacing status (computed by a reliable counter - trust these, do not recount):\n"
        f"- Questions asked so far: {questions_so_far} "
        f"(soft nudge at {soft_budget}, wrap at {wrap_budget}, hard offer at {hard_budget}, "
        f"final stop at {absolute_budget}).\n"
        f"- Current topic \"{run_topic or 'none'}\": {run_count} follow-up question(s) in a row "
        f"(hard cap {PER_TOPIC_QUESTION_CAP} per topic)."
        + pace_line
    )

    if run_topic and run_count >= PER_TOPIC_QUESTION_CAP:
        cap_steering = (
            f"Hard pacing limit reached: you have already asked {run_count} questions in a "
            f"row about {run_topic}. Do NOT ask anything else about {run_topic} - you have "
            "enough for the doctor. This turn, briefly acknowledge the patient's answer and "
            "move to the next selected symptom that still needs attention, or, if none "
            "remain, go to the final anything-else question. Anything still missing about "
            f"{run_topic} will be listed for the doctor as unresolved."
        )

    extra_steering = "\n\n".join(
        s
        for s in (
            cap_steering,
            addon_steering,
            offer_followup,
            quota_steering,
            directive_steering,
            pace_steering,
            agency_steering,
        )
        if s
    )

    # Snapshot the plain transcript and summary state up front; the worker threads must
    # not touch Streamlit session state.
    messages_snapshot = [
        {
            "role": m["role"],
            "content": m.get("content", ""),
            "suggested_answers": m.get("suggested_answers", []),
        }
        for m in st.session_state.messages
    ]
    rolling_summary = st.session_state.rolling_summary if ENABLE_ROLLING_SUMMARY else ""
    summary_tail_start = st.session_state.summary_tail_start if ENABLE_ROLLING_SUMMARY else 0

    with st.spinner("Nurse assistant is reviewing your response..."):
        if ENABLE_JUDGE_AGENT:
            # Interview agent and judge run concurrently. We block only on the
            # interview (its reply is shown to the patient); the judge is best-effort
            # and one-turn-lagged, so we take its nudge only if it is ready within a
            # short grace and otherwise skip it - it can never hold up the turn.
            pool = ThreadPoolExecutor(max_workers=2)
            interview_future = pool.submit(
                get_nurse_response,
                client,
                messages_snapshot,
                prior_history,
                patient_context,
                model,
                system_prompt,
                symptom_count,
                extra_steering,
                rolling_summary,
                summary_tail_start,
            )
            judge_future = pool.submit(
                # Recent tail only - keeps the judge fast and light. The pacing note
                # gives it the exact counts/budget so it can prioritize without
                # counting the transcript itself.
                get_judge_directive,
                client,
                messages_snapshot[-JUDGE_CONTEXT_TAIL:],
                model,
                judge_pacing_note,
            )
            result = interview_future.result()
            try:
                st.session_state.judge_directive = judge_future.result(
                    timeout=JUDGE_GRACE_SECONDS
                )
            except Exception:
                # Not ready in time (or errored) - skip the judge this turn. The
                # code caps still enforce every hard limit.
                st.session_state.judge_directive = ""
            # Do not wait for a still-running judge thread; let it finish detached.
            pool.shutdown(wait=False)
        else:
            result = get_nurse_response(
                client=client,
                chat_history=messages_snapshot,
                prior_history=prior_history,
                patient_context=patient_context,
                model=model,
                system_prompt=system_prompt,
                symptom_count=symptom_count,
                extra_steering=extra_steering,
                rolling_summary=rolling_summary,
                summary_tail_start=summary_tail_start,
            )

    # The model sometimes tries to end early with its own "anything else?" question.
    # Only honour that when the quotas really are filled - otherwise it would skip
    # symptoms the patient selected (this is what silently dropped "Breathing problems").
    # When topics remain, discard the premature close and ask the assigned topic instead.
    if _is_final_open_question(result.get("reply", "")) and not result.get("is_complete"):
        st.session_state.raw_responses.append(result.get("raw_response", ""))
        if quotas_met:
            st.session_state.closing_review = True
            return
        forced = get_nurse_response(
            client=client,
            chat_history=messages_snapshot,
            prior_history=prior_history,
            patient_context=patient_context,
            model=model,
            system_prompt=system_prompt,
            symptom_count=symptom_count,
            extra_steering=(
                extra_steering
                + "\n\nYou tried to end the check-in, but the patient still has symptoms "
                "that have not been asked about. Do NOT ask the anything-else question. "
                + (quota_steering or "Ask the next assigned question.")
            ),
            rolling_summary=rolling_summary,
            summary_tail_start=summary_tail_start,
        )
        if forced.get("reply") and not _is_final_open_question(forced["reply"]):
            result = forced

    # Guarantee the "we'll come to it" promise for a just-added symptom, rather than
    # relying on the model to remember to say it. The quota order already guarantees the
    # interviewer does not jump to it.
    if addon_notice:
        result["reply"] = (
            f"Thanks for telling me about {addon_notice.lower()} - I'll ask you about that "
            "shortly. " + result["reply"]
        )

    with st.chat_message("assistant"):
        st.write(result["reply"])

    # A topic closing is the trigger to compress: fold everything up to now into the
    # running summary so later turns carry "summary + current topic" only.
    topics_closed_before = len(st.session_state.completed_topics)
    apply_nurse_result(result, topic_boxes_placeholder)
    new_detail = st.session_state.messages[st.session_state.summary_tail_start:]
    if (
        ENABLE_ROLLING_SUMMARY
        and len(st.session_state.completed_topics) > topics_closed_before
        and len(new_detail) >= SUMMARY_MIN_NEW_MESSAGES
    ):
        st.session_state.rolling_summary = update_rolling_summary(
            client, st.session_state.rolling_summary, new_detail, model
        )
        # Keep the just-asked question (the last message) in the verbatim tail, so the
        # next turn's patient answer still has a visible question to belong to.
        st.session_state.summary_tail_start = len(st.session_state.messages) - 1



def render_worst_picker(
    client: OpenAI,
    prior_history: str,
    patient_context: str,
    model: str,
    system_prompt: str,
    topic_boxes_placeholder=None,
) -> None:
    """Ask the PATIENT which selected symptom is worst. Their tap sets the 4-question
    quota; the model never infers it. Shown once, before any clinical question."""
    st.markdown("#### Which one is bothering you the most?")
    st.caption(
        "Tap the symptom that is troubling you most - we'll spend the most time on that "
        "one, and still cover the others."
    )
    label_to_topic = dict(CHECKLIST_ITEMS)
    for label in st.session_state.selected_symptom_labels:
        if label == CHECKLIST_NONE_LABEL:
            continue
        if st.button(label, key=f"worst_{label}", use_container_width=True):
            st.session_state.worst_topic = label_to_topic.get(label, "Other")
            st.session_state.worst_label = label
            st.session_state.worst_pick_pending = False
            message = f"The {label.lower()} is bothering me the most."
            add_user_message(message, response_mode="worst_pick")
            with st.chat_message("user"):
                st.write(message)
            generate_and_apply_turn(
                client, prior_history, patient_context, model,
                system_prompt, topic_boxes_placeholder,
            )
            st.rerun()


def render_closing_review(
    client: OpenAI,
    prior_history: str,
    patient_context: str,
    model: str,
    system_prompt: str,
    topic_boxes_placeholder=None,
) -> None:
    """The wrap-up screen: re-show the opening checklist with the patient's picks
    already ticked, so they can see what was covered and add anything else before
    finishing - instead of a wall of recap text."""
    st.markdown("#### Before we finish — is there anything else?")
    st.caption(
        "Here's your check-in list, with the symptoms you told me about already ticked. "
        "Tick anything else you'd like to talk about, or write it in the box below - "
        "including anything I haven't asked about, or a question for your doctor. "
        "If not, press Finish."
    )

    chosen = set(st.session_state.selected_symptom_labels)
    review_items = [item for item in CHECKLIST_ITEMS if item[0] != "Something else"]
    columns = st.columns(2)
    new_picks = []
    for index, (label, topic) in enumerate(review_items):
        already = label in chosen
        with columns[index % 2]:
            checked = st.checkbox(
                label,
                value=already,
                key=f"closerev_{label}",
                disabled=already,
                help="Already discussed" if already else None,
            )
            if checked and not already:
                new_picks.append((label, topic))

    # The open-ended half of "is there anything else?" - catches everything the
    # checklist cannot: off-list symptoms, "oh by the way", a question for the doctor.
    other_text = st.text_area(
        "Anything else?",
        key="closerev_other",
        placeholder="Optional - anything I haven't asked about, or a question for your doctor",
        height=80,
        label_visibility="collapsed",
    )
    has_other = bool(other_text.strip())

    actions = st.columns(2)
    with actions[0]:
        if (new_picks or has_other) and st.button(
            "Add these & keep going", use_container_width=True, key="closerev_add"
        ):
            for label, topic in new_picks:
                if label not in st.session_state.selected_symptom_labels:
                    st.session_state.selected_symptom_labels.append(label)
                if topic not in st.session_state.selected_topics:
                    st.session_state.selected_topics.append(topic)
            parts = []
            if new_picks:
                parts.append(
                    "I'd also like to talk about: "
                    + ", ".join(label for label, _ in new_picks)
                    + "."
                )
            if has_other:
                parts.append(other_text.strip())
            message = " ".join(parts)
            add_user_message(message, response_mode="closing_addon")
            st.session_state.closing_review = False
            with st.chat_message("user"):
                st.write(message)
            generate_and_apply_turn(
                client, prior_history, patient_context, model, system_prompt, topic_boxes_placeholder
            )
            st.rerun()
    with actions[1]:
        if st.button(
            "Finish check-in", type="primary", use_container_width=True, key="closerev_finish"
        ):
            # Anything typed but not discussed must still reach the doctor.
            if has_other:
                add_user_message(other_text.strip(), response_mode="closing_other")
            add_user_message(
                "Patient finished the check-in from the closing review.",
                response_mode="finish_button",
            )
            add_assistant_message(FINAL_CLOSING_REPLY)
            st.session_state.completion_reason = "closing_review_finish"
            st.session_state.completed_at = datetime.now().astimezone().isoformat()
            st.session_state.current_topic = ""
            st.session_state.is_complete = True
            st.session_state.doctor_summary = ""
            st.session_state.closing_review = False
            st.rerun()


# =========================
# Streamlit App
# =========================

def main() -> None:
    st.set_page_config(
        page_title="Nurse Assistant Check-In",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_state()

    left_panel, main_panel = st.columns([1, 3], gap="large")

    with left_panel:
        # =================================================================
        # Patient controls - kept at the TOP so the patient never has to
        # scroll past developer settings to reach them.
        # =================================================================
        topic_boxes_placeholder = st.empty()
        if st.session_state.started and not st.session_state.is_complete:
            # Progress panel only once the patient has actually selected symptoms -
            # not while they are still filling in the checklist.
            if st.session_state.selected_topics:
                st.markdown("**Your check-in**")
                with topic_boxes_placeholder.container():
                    render_topic_boxes()

            # Live checklist: add a symptom remembered mid-conversation. Offer every
            # checklist LABEL the patient has not already picked - deduping by topic
            # would hide distinct symptoms that share a topic (e.g. "Weight loss" vs
            # "Trouble eating", both Nutrition; "Fever" vs "Breathing", both Other).
            if st.session_state.selected_topics:
                st.caption("Remembered another symptom? Add it and we'll make sure to cover it.")
                with st.expander("➕ Add a symptom"):
                    chosen_labels = set(st.session_state.selected_symptom_labels)
                    shown_any = False
                    for label, topic in CHECKLIST_ITEMS:
                        if label == "Something else" or label in chosen_labels:
                            continue
                        shown_any = True
                        if st.button(
                            f"➕ {label}", key=f"addon_{label}", use_container_width=True
                        ):
                            st.session_state.pending_addon = (label, topic)
                            st.rerun()
                    if not shown_any:
                        st.caption("Everything is already on your list.")

            st.markdown("**Pace**")
            pace_columns = st.columns(2)
            with pace_columns[0]:
                if st.button(
                    "🐢 Slow down",
                    key="pace_slower",
                    use_container_width=True,
                    help="Give me more room to explain - the assistant won't rush you.",
                ):
                    st.session_state.pace_mode = (
                        "normal" if st.session_state.pace_mode == "slower" else "slower"
                    )
                    st.rerun()
            with pace_columns[1]:
                if st.button(
                    "🐇 Speed up",
                    key="pace_faster",
                    use_container_width=True,
                    help="I'm getting tired - keep it brief and wrap up sooner.",
                ):
                    st.session_state.pace_mode = (
                        "normal" if st.session_state.pace_mode == "faster" else "faster"
                    )
                    st.rerun()
            pace_label = {
                "normal": "Normal pace",
                "faster": "Going faster · tap again for normal",
                "slower": "Taking it slower · tap again for normal",
            }[st.session_state.pace_mode]
            st.caption(f"Pace: {pace_label}")

            st.markdown(
                """
                <style>
                div.st-key-finish_checkin_sidebar button {
                    background-color: #dc2626 !important;
                    border-color: #dc2626 !important;
                    color: white !important;
                    font-weight: 700 !important;
                }
                div.st-key-finish_checkin_sidebar button:hover {
                    background-color: #b91c1c !important;
                    border-color: #b91c1c !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Wrap up check-in",
                key="finish_checkin_sidebar",
                type="primary",
                use_container_width=True,
                help=(
                    "Review your check-in list and confirm before finishing - your "
                    "responses are always saved."
                ),
            ):
                # Show the checklist review screen (opening checklist, pre-ticked)
                # instead of an instant close. Suppress the automatic threshold offers.
                st.session_state.closing_review = True
                st.session_state.wrap_choice_offered = True
                st.session_state.stop_choice_offered = True
                st.session_state.show_suggestions = False
                st.rerun()

        # =================================================================
        # Setup - patient/visit info stays plainly visible; only the long,
        # technical prompt and prior-history text go into a collapsible box.
        # =================================================================
        # Divider only when patient controls were rendered above it; during setup
        # there is nothing above, so a leading divider just adds an empty gap.
        if st.session_state.check_in_started and not st.session_state.is_complete:
            st.divider()
        model = DEFAULT_MODEL

        status = (
            "Complete"
            if st.session_state.is_complete
            else ("In progress" if st.session_state.check_in_started else "Not started")
        )
        st.caption(f"Status: {status}")

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

        with st.expander("⚙️ Chatbot prompt & prior history"):
            system_prompt = st.text_area(
                "Editable chatbot instructions",
                value=SYSTEM_PROMPT,
                height=260,
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
            st.session_state.saved_prior_history = prior_history
            # If the prior-history field was left blank, auto-load last visit's urgent
            # summary from the Google Sheet by patient name (added feature).
            if not prior_history.strip():
                st.session_state.saved_prior_history = _load_prior_summary_from_sheet(
                    patient_name
                )
            st.session_state.saved_system_prompt = system_prompt
            st.session_state.saved_patient_name = patient_name
            st.session_state.saved_doctor_name = doctor_name
            st.session_state.saved_therapy_week = therapy_week
            st.session_state.check_in_started = True
            st.rerun()

        if st.session_state.is_complete and st.session_state.summary_generated:
            if st.button("Regenerate doctor summary", use_container_width=True):
                st.session_state.summary_generated = False
                st.rerun()

    with main_panel:
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
    
        # ---- Routing: doctor summary page after submission, otherwise patient chat ----
        if st.session_state.is_complete:
            if st.session_state.summary_generated:
                render_doctor_summary_page()
            else:
                st.info("Preparing your doctor summary...")
            return
    
        # ---- Patient view ----
        inject_patient_theme()
        render_patient_header()
    
        if not st.session_state.check_in_started:
            st.info("Enter the patient name in the sidebar, add prior patient history if available, then click **Start new check-in** to begin.")
            return
    
        # The disclosure gate: no checklist, no chat input, and no model call may
        # happen until the patient acknowledges it.
        if not st.session_state.disclaimer_acknowledged:
            render_welcome_screen(patient_name)
            return
    
        render_disclaimer_banner()
    
        if not st.session_state.started:
            # Checkbox-first opening (June 5 clinical-team decision) - the only mode.
            checklist_result = render_symptom_checklist(
                returning=bool(prior_history.strip())
            )
            if checklist_result is None:
                return
            question, user_message, labels, topics = checklist_result
            st.session_state.selected_symptom_labels = labels
            st.session_state.selected_topics = topics
            add_assistant_message(question)
            add_user_message(user_message, response_mode="checklist")
            st.session_state.started = True
            if len(topics) > 1:
                # More than one symptom: the PATIENT designates the worst before any
                # questions, so code can apply the 4/2 quotas without ever inferring it.
                st.session_state.worst_pick_pending = True
                st.rerun()
            if topics:
                st.session_state.worst_topic = topics[0]
                first = [l for l in labels if l != CHECKLIST_NONE_LABEL]
                st.session_state.worst_label = first[0] if first else ""
            generate_and_apply_turn(
                client, prior_history, patient_context, model, system_prompt, topic_boxes_placeholder
            )
            st.rerun()
    
        render_chat_history()
    
        # The patient designates which symptom is worst (never inferred by the model).
        if st.session_state.worst_pick_pending and not st.session_state.is_complete:
            render_worst_picker(
                client, prior_history, patient_context, model, system_prompt, topic_boxes_placeholder
            )
            return
    
        # Wrap-up / natural close -> show the checklist review (opening checklist, pre-ticked
        # with the patient's picks) instead of the normal chat input.
        if st.session_state.closing_review and not st.session_state.is_complete:
            render_closing_review(
                client, prior_history, patient_context, model, system_prompt, topic_boxes_placeholder
            )
            return
    
        # The patient is answering a pacing offer. Explicit buttons let the app know their
        # choice for certain, so it can inject one unambiguous instruction instead of asking
        # the model to work out which branch applies. Typing a reply still works.
        if st.session_state.pending_offer and not st.session_state.is_complete:
            offering_wrap = st.session_state.pending_offer == "wrap"
            st.caption("Tap your choice — or just type your answer below.")
            choice_columns = st.columns(2)
            choices = [
                ("continue", "Keep going", "offer_continue", "Let's keep going."),
                (
                    "wrap",
                    "Wrap up now" if offering_wrap else "Finish now",
                    "offer_wrap",
                    "Let's wrap up now." if offering_wrap else "Let's finish now.",
                ),
            ]
            for column, (choice_value, label, key, message) in zip(choice_columns, choices):
                with column:
                    if st.button(label, key=key, use_container_width=True):
                        add_user_message(message, response_mode="offer_choice")
                        if choice_value == "wrap":
                            # Wrapping up skips the remaining topics entirely - no further
                            # questions. Deterministic: straight to the closing review, where
                            # the patient still sees what was not covered.
                            st.session_state.pending_offer = None
                            st.session_state.offer_choice = None
                            st.session_state.closing_review = True
                            st.rerun()
                        st.session_state.offer_choice = choice_value
                        with st.chat_message("user"):
                            st.write(message)
                        generate_and_apply_turn(
                            client, prior_history, patient_context, model,
                            system_prompt, topic_boxes_placeholder,
                        )
                        st.rerun()
    
        render_current_suggestions()
    
        # Mid-conversation "add a symptom": the patient checked a new topic in the live
        # side panel. Add it to the checklist and voice it as if the patient raised it, so
        # the interviewer acknowledges and covers it - "check it and we'll cover it."
        if st.session_state.get("pending_addon"):
            addon_label, addon_topic = st.session_state.pending_addon
            st.session_state.pending_addon = None
            if addon_topic and addon_topic not in st.session_state.selected_topics:
                st.session_state.selected_topics.append(addon_topic)
            # Count it as another selected symptom so the adaptive budget grows (+2), the
            # same as if it had been checked on the opening checklist.
            if addon_label not in st.session_state.selected_symptom_labels:
                st.session_state.selected_symptom_labels.append(addon_label)
            addon_message = f"I just remembered - I'd also like to talk about {addon_label.lower()}."
            add_user_message(addon_message, response_mode="checklist_addon")
            # Acknowledge now, cover later - do not abandon the current topic.
            st.session_state.addon_notice = addon_label
            with st.chat_message("user"):
                st.write(addon_message)
            generate_and_apply_turn(
                client, prior_history, patient_context, model, system_prompt, topic_boxes_placeholder
            )
            st.rerun()
    
        # Clear the composer after a completed send. This must happen BEFORE the text
        # box widget is created, because a widget-keyed session value cannot be changed
        # once the widget exists in the same run.
        if st.session_state.get("clear_composer"):
            st.session_state.composer_text = ""
            st.session_state.clear_composer = False
    
        composer_value = st.text_area(
            "Your response",
            key="composer_text",
            height=90,
            label_visibility="collapsed",
            placeholder="Type your response here, or tap “Suggestions” above to start from an option…",
        )
        send_clicked = st.button(
            "Send",
            key="composer_send",
            type="primary",
            use_container_width=True,
        )
    
        submitted_answer = composer_value.strip() if (send_clicked and composer_value.strip()) else None
    
        if submitted_answer:
            # "selected" only when the patient sent a suggestion verbatim; a suggestion
            # they edited or added to counts as "typed" (the transcript still stores the
            # offered suggestions, so chip-assisted edits remain recoverable).
            last_message = st.session_state.messages[-1] if st.session_state.messages else {}
            offered_suggestions = (
                last_message.get("suggested_answers", [])
                if last_message.get("role") == "assistant"
                else []
            )
            response_mode = "selected" if submitted_answer in offered_suggestions else "typed"
            add_user_message(submitted_answer, response_mode=response_mode)
            st.session_state.show_suggestions = False
            st.session_state.clear_composer = True
    
            with st.chat_message("user"):
                st.write(submitted_answer)
    
            generate_and_apply_turn(
                client, prior_history, patient_context, model, system_prompt, topic_boxes_placeholder
            )
            st.rerun()
    
    
if __name__ == "__main__":
    main()
