import json
import os
import html
import re
import hashlib
import csv
import textwrap
import logging
import secrets
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Dict, List, Any, Optional

# Calls OpenAI's API (DEFAULT_MODEL below): the conversation is sent to OpenAI.
from openai import OpenAI
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    gspread = None
    Credentials = None


# =========================
# Prompt / System Logic
# =========================

PROMPT_VERSION = "virtual-doctor-2026-09-29-v21-self-harm-same-rule"

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
    ("Nausea", "GI Symptoms"),
    ("Vomiting", "GI Symptoms"),
    ("Constipation", "GI Symptoms"),
    ("Diarrhea", "GI Symptoms"),
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
TRIAGE_PHONE = os.environ.get("TRIAGE_PHONE", "[TRIAGE PHONE NUMBER]")

# Shown to the patient whenever the interviewer reports a red flag. The wording is fixed
# here (not written by the model) so it is exactly what the clinical team approved.
RED_FLAG_NOTICE = (
    f"If this feels like an emergency, call 911 now. Otherwise please call your nurse "
    f"triage line at {TRIAGE_PHONE} - this check-in is not monitored in real time."
)
# Shown instead of RED_FLAG_NOTICE when the patient reports thoughts of self-harm.
# DRAFT wording (Sep 27) - needs clinical-team approval before patient use.
SELF_HARM_NOTICE = (
    "You don't have to go through this alone. If you are thinking about harming yourself, "
    "you can call or text 988 (Suicide & Crisis Lifeline) any time, day or night. If you "
    f"are in immediate danger, call 911. You can also call your nurse triage line at "
    f"{TRIAGE_PHONE} - this check-in is not monitored in real time."
)

# --- Red-flag kinds (clinician feedback, Sep 17) ---
# The model reports WHICH red flag the patient described ("red_flag_kind"); the app uses
# it to pick the notice (988 for self-harm), to describe it in the closing message, and
# to move the matching checklist symptom to the front so it is asked about next.
RED_FLAG_LABELS = {
    "self_harm": "Feeling down, anxious, or depressed",
    "breathing": "Breathing problems",
    "fever": "Fever or chills",
    "vomit_blood": "Vomiting",
    "diarrhea": "Diarrhea",
}
RED_FLAG_DESCRIPTIONS = {
    "self_harm": "thoughts of harming themselves",
    "breathing": "shortness of breath",
    "fever": "fever",
    "vomit_blood": "blood in vomit",
    "chest_pain": "chest pain",
    "headache": "a severe headache",
    "neuro": "possible new neurologic symptoms",
    "bleeding": "bleeding",
    "diarrhea": "very frequent diarrhea",
    "other": "a red-flag symptom",
}
# Queue item used for a red flag with no checklist box, when the model gives no name.
RED_FLAG_DEFAULT_NAMES = {
    "chest_pain": "Chest pain",
    "headache": "Severe headache",
    "neuro": "New neurologic symptoms",
    "bleeding": "Bleeding",
    "other": "Urgent symptom",
}
# Every newly reported red flag gets up to this many questions, asked first (the reply
# that follows it up counts as the first), before the queue continues (Sep 28 rule).
RED_FLAG_QUESTIONS = 3
# Patient messages written in their own words (checklist taps and button presses are not
# a reply to a question).
PATIENT_TEXT_MODES = ("typed", "selected")
# A reply that does not answer the question ("hi", or something unrelated): the question
# is not counted and is asked ONCE more. If that is not answered either, it IS counted,
# is shown to the doctor as unanswered, and the bot goes on to the next question - so a
# patient who will not engage is never stuck (Sep 28 rule).


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
# What the patient's "Speed up" button does: the quotas above are replaced by these, so
# the check-in really gets shorter instead of only asking the model to be brief.
FAST_WORST_SYMPTOM_QUOTA = 3
FAST_OTHER_SYMPTOM_QUOTA = 2
# Clinician feedback (Sep 17): patients did not know the suggested answers existed until
# they pressed "Suggestions", so they are now shown with every question. Set to False to
# go back to hiding them behind the button (e.g. to compare both in the interviews).
SHOW_SUGGESTIONS_BY_DEFAULT = True
# Shown when the patient finishes from the closing screen.
FINAL_CLOSING_REPLY = (
    "Thank you for sharing all of that with me. Your check-in is complete, and "
    "everything you told me will be shared with your care team before your visit."
)
# Shown when the patient ticks "None of these", right before the closing screen.
NONE_SELECTED_REPLY = (
    "Thank you for letting me know - I'm glad you're doing okay today."
)
# Last resort when the model replies without a question although the check-in continues.
NO_QUESTION_FALLBACK_QUESTION = "Is there anything more you'd like to tell me about that?"
NO_QUESTION_FALLBACK_ANSWERS = [
    "No, that covers it.",
    "Yes, there's a bit more to it.",
    "It has been getting worse.",
    "It has been getting better.",
    "I'm not sure.",
]
# Sent when every selected topic is covered: the interviewer replies once, without a
# question, before the closing review screen - so the patient's last answer (which may
# be a red flag) is acknowledged instead of the chat silently stopping.
ACKNOWLEDGE_BEFORE_REVIEW_STEERING = (
    "Every symptom the patient selected has now been covered, so this is your last chat "
    "message before the patient sees a final review screen where they can add anything "
    "else or finish. In one or two short sentences, warmly acknowledge the patient's last "
    "answer. If it reports a red flag, follow the Red Flags rules. Do NOT ask any question, "
    "do NOT ask the anything-else question, and do NOT say the check-in is complete. "
    "is_complete must be false, doctor_summary must be an empty string, suggested_answers "
    "must be an empty list, and topic must be an empty string."
)
# The patient TYPED an answer to the "are you tired? wrap up or keep going" offer instead
# of tapping a button: the model reports the choice in "offer_choice" and the app then
# does exactly what the matching button does.
TYPED_OFFER_REPLY_STEERING = (
    "The patient has just answered your wrap-up offer in their own words instead of tapping "
    "a button. Set \"offer_choice\" to \"wrap\" if they want to stop or wrap up now, to "
    "\"continue\" if they want to keep going, or to an empty string if it is unclear. If it "
    "is \"wrap\", reply with one short, warm acknowledgement and NO question - the app then "
    "shows the closing screen. Otherwise continue with the topic assignment. A red flag in "
    "their message still comes first."
)
# The two cases where that last message asks a question after all (Sep 27 redesign).
ACKNOWLEDGE_EXCEPTIONS_STEERING = (
    "Exceptions to \"do NOT ask any question\" - in these two cases ask ONE question "
    "instead, with five suggested_answers as usual: (1) the patient's latest message "
    "reports a red flag you have not followed up yet (follow the Red Flags rules); (2) it "
    "does not answer your previous question and you are allowed to ask it again (see "
    "below). is_complete must still be false."
)
# Appended by code to that last acknowledgement, so the patient knows the questions are
# done and where their answers go before the closing review appears.
QUESTIONS_DONE_NOTE = (
    "That's all my questions for today - thank you. Everything you've told me will be "
    "passed on to your care team."
)
# Used instead when a red flag came up near the end, so the check-in does not stop
# coldly right after, e.g., a disclosure of self-harm (Jessica's Sep 17 case).
RED_FLAG_DONE_NOTE = (
    "That's all my questions for today. I've flagged what you shared so your care team "
    "can follow up with you."
)
RED_FLAG_CLOSING_STEERING = (
    "Earlier in this check-in the patient reported {description}. In your acknowledgement, "
    "respond warmly and specifically to that - not with a generic thank-you - and make "
    "clear it has been flagged for their care team. Set red_flag to true."
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
- Ask follow-up questions one at a time. The app assigns the symptom for every question, starting with the one the patient chose as bothering them most.
- The patient has already confirmed every symptom they selected. Never ask whether they have it (for example, do not ask "Have you had any nausea?"). Invite them to describe it instead, for example: "Can you tell me about the nausea you've been having?"
- The patient chooses which symptom bothers them most before the chat, with buttons the app shows. Do not ask them about this yourself, and do not guess.
- Do NOT ask about, screen, or mention topics the patient did not select. No broad screening questions. Unselected areas are shown to the provider automatically, so skipping them is safe and expected.
- Patients who check "None of these" go straight to the closing screen; the app handles this.
- If the patient raises a symptom or concern that is not related to the current topic, do NOT ask about it now - the app adds it to the end of their list and it will be assigned to you later (see Non-Answers and New Issues). Only symptoms on the patient's list may be asked about, and the app tells you which one to ask about each turn.

Carefully analyze the patient's response:
- Extract any already-answered topics.
- Extract every symptom or concern the patient mentions, even if they mention several in one message.
- Acknowledge the patient's concerns.
- Do not lose track of the initial answer. Refer back to it when relevant.

Conversational Behavior Rules:
- Ask exactly one question at a time.
- One question means one clinical variable only. Do not combine questions with "and," "also," "as well," or commas that ask for multiple answers in the same turn.
- Do not ask long lists of questions.
- If a patient answers multiple topics at once, do not repeat questions already answered.
- Every symptom on the patient's list is assigned to you in turn. New ones the patient raises are added to the list by the app (see Non-Answers and New Issues).
- Ask about only one reported symptom or concern per assistant turn.
- Expand only where details are missing.
- Use conditional logic:
- Go deepest on the symptom the patient said bothers them most - the app gives it the most questions.
- Cover ONLY the symptoms on the patient's list, as the app assigns them. Do not force every detailed sub-question. If a patient says no to something, do not ask follow-ups about it.
- The app decides when every symptom is covered and then shows the patient a closing screen (see Closing).
- Occasionally offer guided options when helpful, especially for medications or symptoms patients may not recall precisely.
- Keep tone warm, reassuring, and professional.

Length Control:
- Keep each assistant reply to 1-2 short sentences.
- Do not collect full detail for mild, stable, or denied symptoms.
- Do not add broad screening questions about symptoms that are not on the patient's list.

Question Budget and Scope (IMPORTANT):
- You are collecting information FOR the doctor, not performing a clinical workup. Do not pursue diagnostic lines of questioning (for example: orthostatic-testing patterns for dizziness, sleep-apnea style workups for night breathing, or extended medication-history interrogations).
- The app decides how many questions each symptom gets and tells you each turn (for example "1 of 3"). Do not count questions yourself.
- It is acceptable to leave details uncollected: anything missing will be shown to the doctor as an unresolved item to ask about during the visit. Prefer moving on over drilling down.
- Make each question count: ask for the most essential detail still missing (for example onset, severity, functional impact, or what they are doing for it).

Memory and Redundancy Rules:
- Before every reply, silently review the full conversation and prior history.
- Treat the patient's current answers as already known facts.
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

Pain scores above the scale: patients often rate pain above 10 (for example "12 out of 10") to say it is as bad as it gets. Accept it without correcting them or asking them to re-rate: acknowledge that it sounds severe, treat it as the maximum (10/10), and record their exact words for the doctor (for example "10+/10 - patient said 12"). The same applies to any answer outside a scale you offered.

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



4. Oral Symptoms
Assess:
- Mouth sores: new vs existing, location, pain severity, effect on eating, drinking, swallowing, or speaking, treatments such as magic mouthwash
- Dry mouth: timing, treatments, functional impact
- Mucus: thick vs watery, impact, management
- Teeth/gums issues
- Oral hygiene practices

5. GI Symptoms
Assess:
- Nausea
- Vomiting
- Constipation: frequency, medications, discomfort
- Diarrhea: frequency (bowel movements per day), medications, discomfort

6. Fatigue & Sleep
Assess:
- Fatigue: general vs localized weakness
- Impact on daily life
- Sleep: trouble falling/staying asleep
- Causes such as pain or other symptoms

7. Activity & Independence
Assess:
- Ability to perform daily activities
If limited:
- Which activities
- Cause: pain, fatigue, other

8. Mood & Support
Assess:
- Emotional state: anxiety, worry, sadness, depression, low mood, loss of interest, hopelessness
- Impact on functioning
- Social support system

If the patient reports mood concerns:
- Respond empathetically before asking the next question.
- Include depression/low mood whenever giving examples of mood symptoms; do not focus only on anxiety or worry.
- Anxiety and depression are different - follow up on the one the patient actually describes. Patients often say "anxious" when they mean worry or distress about a symptom, a test result, or their treatment; ask what they are worried about and how it affects them.
- Do NOT ask about suicidal thoughts or self-harm just because the patient reports anxiety or worry. Ask about them only if the patient describes depression, low mood, or hopelessness, or raises such thoughts themselves.

9. Other
Assess:
- Fever
If fever is reported:
- When it started
- Highest temperature, if known
- Chills or feeling acutely unwell
- Anything taken for it (for example acetaminophen) and whether it brought the fever down
- Breathing problems
If breathing problems are reported:
- Whether they are new or worse than before
- At rest or only with activity
- Anything that helps (for example an inhaler or oxygen)
(New or significant shortness of breath is also a red flag - see Red Flags.)

What the patient is doing about it (ALL symptoms):
For every symptom you cover, one of its questions should ask what the patient is doing for it - medication, mouthwash, supplements, or anything else - and whether it is helping, unless they already told you or it clearly does not apply. Doctors need this for every symptom, not only for pain and constipation.



Red Flags (clinical team's list):
Treat the following as red flags when reported: suicidal ideation (suicidal thoughts or thoughts of self-harm), chest pain with exertion, intractable vomiting, blood in vomit, uncontrolled severe diarrhea (more than 10 bowel movements a day), fever, any new or significant shortness of breath (not only at rest - any breathlessness that is new or noticeably worse counts), severe headache, new neurologic symptoms (for example new weakness, numbness, confusion, trouble speaking, or vision changes), and uncontrolled bleeding.

When a red flag is reported:
- Follow it up in THIS reply, even if you were asked to cover a different symptom this turn. The app then asks about it first - up to 3 questions in total, this one included - before the rest of the list continues.
- Set "red_flag_kind" to the kind of red flag, and "red_flag_symptom" to its name: the checklist name if one matches (for example shortness of breath -> "Breathing problems", fever -> "Fever or chills"), otherwise the patient's own words for it in 1-4 words (for example "Chest pain"). Set "symptom" to the same name.
- A red flag you have already followed up earlier in this check-in does not need another follow-up; just acknowledge it.
- Acknowledge it calmly and warmly; do not alarm the patient.
- Explicitly say you are flagging it for the care team, for example: "Thank you for telling me - I've flagged this for your care team."
- Never say or suggest that anyone will see it right away, is reading along now, or will contact the patient soon: this check-in is not monitored in real time, and the flag is reviewed with the rest of the check-in. For anything urgent, the notice the app shows gives the patient the numbers to call.
- Ask one essential detail per question, most important first. Do not perform a workup. Choose the details that matter most for that red flag, for example:
  - Shortness of breath: whether it is new, and whether it happens at rest or only with activity (or comes with chest pain).
  - Headache or neurologic symptoms: when it started and whether it is the worst they have had or still getting worse.
  - Fever: the highest temperature, and whether anything taken for it brought it down.
  - Vomiting or bleeding: how much or how often, and whether they can keep fluids down.
- Do NOT end the conversation because of a red flag; continue the check-in so other symptoms are still collected.
- Never give safety instructions, phone numbers, or medical advice yourself. Set "red_flag" to true instead: the app then shows the patient an approved notice with the emergency and nurse triage numbers.
- Thoughts of self-harm or suicide (DRAFT wording - pending clinical-team approval): respond with warmth and without judgment. Thank them for trusting you with this, say you are sorry they are going through it, and tell them you have flagged it for their care team. Then ask ONE gentle question, such as whether these thoughts are new for them - this is the first of its up to 3 questions, exactly as for any other red flag. Do not ask about any other symptom in that reply, and never end the conversation right after such a disclosure.

Non-Answers and New Issues:
Before every reply, read the patient's latest message and decide what it did. Use your judgment about what the patient means, not particular words.
- Did it answer your previous question? A greeting ("hi"), "ok", a question back, or a reply about something else does not answer it. A short or partial answer ("not sure", "about the same", "5") does answer it. Set "answered_last_question" accordingly (true when there was no previous question).
- If it did not answer, and the app has not told you to stop re-asking, gently ask the same question again in simpler words. Do not treat the unanswered question as covered.
- Did the patient raise a symptom or concern that is NOT related to the current topic (whether they just mention it or ask to talk about it)? Then set "new_issue" to its name - the checklist name if one matches, otherwise the patient's own words for it in 1-4 words (for example "Knee pain"). Do NOT ask about it and do NOT mention it in your reply: the app acknowledges it and adds it to the end of their list. Stay on the current topic.
- Something related to the current topic (for example stomach pain while talking about diarrhea, or mouth pain while talking about mouth sores) is part of the current topic, not a new issue: leave "new_issue" empty. You may ask about it within the current topic's questions.
- A red flag is never a new issue: it always comes first (see Red Flags).

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

Closing:
The app decides when the check-in ends. When every symptom on the patient's list is covered, it shows a closing screen where the patient can add anything else or finish; anything they add there is put on their list and assigned to you like any other symptom. So never ask an "anything else" question yourself, and never set is_complete to true yourself.

Response Format:
Always respond as valid JSON with exactly these keys:
{
  "reply": "Natural message to show the patient.",
  "suggested_answers": ["Exactly five brief answers when reply asks a question; otherwise an empty list."],
  "is_complete": false,
  "doctor_summary": "always an empty string - the app writes the doctor summary separately",
  "topic": "one of: Pain, Nutrition, Swallowing, Oral Symptoms, GI Symptoms, Fatigue & Sleep, Activity & Independence, Mood & Support, Other, or empty string",
  "red_flag": true or false,
  "red_flag_kind": "one of: self_harm, breathing, fever, vomit_blood, chest_pain, headache, neuro, bleeding, diarrhea, other, or empty string",
  "red_flag_symptom": "name of the red-flag symptom (checklist name if one matches, otherwise the patient's words), or empty string",
  "symptom": "the symptom on the patient's list this question is about, copied exactly, or empty string",
  "answered_last_question": true or false,
  "new_issue": "a symptom or concern the patient raised that is not related to the current topic, or empty string",
  "offer_choice": "wrap, continue, or empty string - only when the app says the patient typed an answer to the wrap-up offer"
}

Important:
- The patient should only see the value of "reply".
- Whenever reply asks the patient a question, return exactly five concise, realistic, directly relevant suggested_answers. Make them meaningfully different and, when appropriate, cover positive, negative, neutral, and uncertain responses. They are optional aids, not a questionnaire.
- When reply does not ask a question (for example an acknowledgement), suggested_answers must be an empty list.
- "is_complete": always false - the app ends the check-in (see Closing).
- For "topic", choose the single clinical topic that best matches the current assistant reply or question. If the reply is a general opening, closing, or administrative message, use an empty string.
- Set "red_flag" to true when the patient's latest message reports one of the Red Flags listed above; otherwise false. Keep replying normally: the check-in continues as usual.
- "red_flag_kind": when red_flag is true, the kind of red flag: self_harm (thoughts of self-harm or suicide), breathing (new or significant shortness of breath), fever, vomit_blood (blood in vomit), chest_pain, headache (severe headache), neuro (new neurologic symptoms), bleeding (uncontrolled bleeding), diarrhea (more than 10 a day), or other. Empty string when red_flag is false.
- "red_flag_symptom": see Red Flags. Empty string when red_flag is false.
- "answered_last_question" and "new_issue": see Non-Answers and New Issues.
- For "symptom", name the one checklist symptom your question is about, copied exactly as the patient selected it (for example "Diarrhea" or "Trouble eating or drinking"). Use an empty string if the reply asks no question or the question is not about one of their checklist symptoms.

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

If the transcript contains a button press such as "Patient finished the check-in from the closing review.", "Let's wrap up now.", "Let's finish now." or "Let's keep going.", treat it as a UI event rather than a symptom or clinical statement. Do not infer denials for topics that were not covered. Add significant reported concerns that still lacked follow-up to Unresolved_concerns.

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
- When an urgent flag is new (the patient says it is new, or it is absent from prior history), start its "label" with "NEW:" (for example "NEW: shortness of breath").
- Urgent_flags must be an empty list when there are no urgent concerns. Flag only patient-reported concerns; do not diagnose. Include fever, suicidal thoughts/severe distress, chest pain with exertion, intractable vomiting, blood in vomit, uncontrolled severe diarrhea (more than 10 bowel movements a day), any new or significant shortness of breath, severe headache, new neurologic symptoms (new weakness, numbness, confusion, trouble speaking, vision changes), uncontrolled bleeding, severe or rapidly worsening pain, breathing difficulty, inability to eat/drink, severe dehydration, bleeding/choking, feeding-tube malfunction, or inability to perform basic daily activities when reported.
- Unresolved_concerns must be an empty list when none can be identified from the transcript.
- Use an empty string for any topic with no relevant information.
- Every "coverage" value must be exactly "reported", "denied", or "not_assessed".
- Every "quote" must be verbatim patient words copied from the transcript, or an empty string. Never fabricate quotes.
- "Key_changes" must be an empty list when there are no meaningful changes.
- Output must be valid JSON with no commentary, markdown, or text outside the object.
"""


# Shown to the PATIENT after the check-in (Sep 17 request from Slobodan, Jessica and
# Erin): "this is what I heard from you and will share with your care team", so the
# patient can check the chatbot understood them before the doctor's report is made.
SHOW_PATIENT_SUMMARY = True
PATIENT_SUMMARY_SYSTEM_PROMPT = """
You write a short summary FOR THE PATIENT at the end of a pre-visit check-in between a head and neck cancer patient and a nurse assistant chatbot. It shows the patient what will be shared with their care team, so they can check it was understood correctly.

Rules:
- 2 or 3 short sentences, addressed to the patient as "you", in plain everyday language (about a 6th-grade reading level). No medical jargon or abbreviations.
- Say only what the patient actually told you in the transcript. Do not add, guess, or interpret.
- Cover the main symptoms they described, with the one or two details that matter most (for example how bad it is, how long, or how it affects eating or sleep).
- If they reported something serious (for example thoughts of harming themselves, fever, trouble breathing, bleeding), mention it plainly and kindly, e.g. "You also told me you have been having thoughts of harming yourself."
- No medical advice, no reassurance about what it means, no instructions, no phone numbers.
- Do not start with a greeting. Start with "Here's what you told me today:" or similar.

Respond ONLY as valid JSON: {"summary": "the 2-3 sentence summary"}.
"""


JUDGE_SYSTEM_PROMPT = """
You are a silent supervisor for a pre-visit nurse check-in chatbot. You do NOT talk to the patient and you never see your output shown to them. You read the recent conversation and decide whether the interviewer's NEXT question is worth asking, then issue a short directive that will be handed to the interviewer.

Do NOT try to count the questions yourself from the transcript, and do NOT enforce numeric limits - a separate system counts reliably and caps how many questions are asked per topic and overall. That system also decides which symptom is asked about next and when to move on, so never tell the interviewer to switch to another symptom or topic. Instead, you are GIVEN the exact counts and remaining budget (see "Pacing status" at the end of the input). Trust those numbers and use them to prioritize: when little budget remains, be stricter - allow only the single most clinically essential follow-up and otherwise say to move on; when there is plenty of room, a genuinely useful follow-up is fine.

Your job is the judgment that counting cannot make: whether the interviewer is about to ask something that adds little value. Flag ONLY these problems:

- Redundancy: the interviewer is re-asking, or is about to re-ask, something the patient has already answered (including whether a symptom is worse/better, present/absent, constant/intermittent, or medication-related).
- Diminishing returns: the essential clinical detail for the current symptom (roughly onset, severity, and functional impact) is already captured, so the next question should cover a different, still-useful aspect of the same symptom (for example medication and its effect, what helps or worsens it, or impact on eating, sleep or daily life).
- Scope drift: the interviewer is heading into diagnostic-workup territory that is not this tool's job (for example orthostatic-testing patterns for dizziness, sleep-apnea style workups, or extended medication-history interrogations).

The interviewer collects information FOR the doctor; it is not doing a clinical workup, and anything left uncollected is simply listed for the doctor as unresolved.

Never flag these - they are intended:
- Asking a question again because the patient's reply did not answer it (a greeting, "ok", or something unrelated). The app allows one repeat per question and then moves on by itself.
- Follow-up questions about a red flag the patient reported (for example self-harm, shortness of breath, fever, chest pain). The app gives each red flag up to three questions.

Respond ONLY as valid JSON:
{
  "intervene": true or false,
  "directive": "A short imperative addressed to the interviewer when intervene is true, e.g. 'You already have onset, severity, and how eating is affected for swallowing - ask about a different aspect, such as what helps or makes it worse.' Empty string when intervene is false."
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
Review the current conversation before every reply. The app assigns the symptom for each question and adds new symptoms the patient raises to their list, so follow its assignment. Ask about only one symptom per turn, and never ask an "anything else" question yourself - the app shows a closing screen for that.
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


# "Have you had / noticed / been having ..." - a question asking whether the patient has
# something at all.
_HAS_SYMPTOM_QUESTION = (
    r"\b(?:have|has|had|do|does|did|are|is)\s+you\s+(?:noticed|had|have|experienced|"
    r"experience|been\s+(?:having|experiencing|feeling|getting)|having|experiencing|getting|"
    r"feeling|felt|get|got)\b"
)
# Label words too generic to identify a symptom ("Trouble eating" -> "eating").
_SYMPTOM_FILLER_WORDS = {"trouble", "with", "difficulty", "problems", "feeling", "poor", "something", "else"}


def _match_symptom(reported: str, selected_symptoms: List[str]) -> str:
    """The selected checklist symptom the model named (case-insensitive), or ""."""
    wanted = (reported or "").strip().lower()
    return next((label for label in selected_symptoms if label.lower() == wanted), "")


def _symptom_mentioned(label: str, text: str) -> bool:
    """True when the text talks about this checklist symptom ("Diarrhea" in "...12 loose
    stools, diarrhea since Monday"). Used to pick the last-visit line that belongs to
    one checkbox rather than to its whole topic."""
    lower = (text or "").lower()
    for word in re.findall(r"[a-z]+", label.lower()):
        if len(word) <= 3 or word in _SYMPTOM_FILLER_WORDS:
            continue
        if _stem(word) in lower:
            return True
    return False


def _asks_whether_patient_has(reply: str, confirmed_symptoms: List[str]) -> str:
    """The checklist symptom a question asks the patient whether they have at all (e.g.
    "Have you had any nausea?"), or "" - they already confirmed it by ticking it."""
    for sentence in re.split(r"(?<=[.!?])\s+", _strip_quoted_segments(reply)):
        if "?" not in sentence:
            continue
        lower = sentence.lower()
        for label in confirmed_symptoms:
            for word in re.findall(r"[a-z]+", label.lower()):
                if len(word) <= 3 or word in _SYMPTOM_FILLER_WORDS:
                    continue
                match = re.search(_HAS_SYMPTOM_QUESTION + r"\s+(?:\w+\s+){0,3}" + re.escape(word), lower)
                # "How often do you have diarrhea?" asks for a detail, not whether they have it.
                if match and not re.search(r"\b(?:how|when|where|what|which|why|who)\b", lower[: match.start()]):
                    return label
    return ""


def _remove_questions(reply: str) -> str:
    """Drop every sentence that asks something, for turns where no question may be asked."""
    sentences = re.split(r"(?<=[.!?])\s+", reply.strip())
    kept = " ".join(sentence for sentence in sentences if "?" not in sentence).strip()
    return kept or "Thank you for sharing that with me."


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

    raw_symptom = parsed.get("symptom")
    parsed["symptom"] = raw_symptom.strip() if isinstance(raw_symptom, str) else ""

    raw_red_flag = parsed.get("red_flag")
    parsed["red_flag"] = raw_red_flag is True or (
        isinstance(raw_red_flag, str) and raw_red_flag.strip().lower() == "true"
    )

    raw_kind = parsed.get("red_flag_kind")
    kind = raw_kind.strip().lower() if isinstance(raw_kind, str) else ""
    parsed["red_flag_kind"] = (
        (kind if kind in RED_FLAG_DESCRIPTIONS else "other") if parsed["red_flag"] else ""
    )

    # Missing or malformed -> treated as answered, so a model that omits the field never
    # holds up the check-in.
    raw_answered = parsed.get("answered_last_question")
    parsed["answered_last_question"] = not (
        raw_answered is False
        or (isinstance(raw_answered, str) and raw_answered.strip().lower() == "false")
    )

    for key in ("new_issue", "red_flag_symptom"):
        value = parsed.get(key)
        parsed[key] = " ".join(value.split())[:60] if isinstance(value, str) else ""
    raw_choice = parsed.get("offer_choice")
    choice = raw_choice.strip().lower() if isinstance(raw_choice, str) else ""
    parsed["offer_choice"] = choice if choice in ("wrap", "continue") else ""
    if not parsed["red_flag"]:
        parsed["red_flag_symptom"] = ""

    return parsed


def _validate_suggested_answers(value: Any) -> tuple[List[str], List[str]]:
    """(usable suggestions, problems). Usable = non-empty, distinct, not too long, at most
    5. The problems trigger the one corrective retry; whatever is usable afterwards is
    shown, even if fewer than 5 - never a canned set that may not fit the question
    (Sep 28: canned "mild / moderate / severe" answers appeared under a medication
    question)."""
    errors: List[str] = []
    if not isinstance(value, list):
        return [], ["suggested_answers is not a list"]
    answers = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(answers) != 5:
        errors.append(f"expected 5 suggested answers, received {len(answers)}")
    usable: List[str] = []
    seen = set()
    duplicates = too_long = False
    for item in answers:
        key = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
        if key in seen:
            duplicates = True
            continue
        if len(item) > 180:
            too_long = True
            continue
        seen.add(key)
        usable.append(item)
    if duplicates:
        errors.append("suggested answers contain duplicates")
    if too_long:
        errors.append("one or more suggested answers are too long")
    return usable[:5], errors


def _normalize_nurse_response(parsed: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    normalized = dict(parsed) if isinstance(parsed, dict) else {}
    reply = normalized.get("reply", "")
    if not isinstance(reply, str):
        reply = ""
    normalized["reply"] = reply.strip()
    asks_question = "?" in normalized["reply"]
    answers, errors = _validate_suggested_answers(normalized.get("suggested_answers", []))
    if asks_question:
        # Whatever the model gave that is usable, even if fewer than 5.
        normalized["suggested_answers"] = answers
    else:
        normalized["suggested_answers"] = []
        errors = []
    normalized["is_complete"] = normalized.get("is_complete") is True
    normalized["red_flag"] = normalized.get("red_flag") is True
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
    question_expected: Optional[bool] = True,
    confirmed_symptoms: Optional[List[str]] = None,
    assigned_symptom: str = "",
    selected_symptoms: Optional[List[str]] = None,
) -> List[str]:
    """All validation problems for one model response, collected in one pass so a
    single corrective retry can address every issue at once.

    confirmed_symptoms: checklist symptoms the patient ticked, which must not be asked
    about as if unconfirmed ("Have you had any nausea?").
    assigned_symptom / selected_symptoms: the checklist symptom code assigned this turn,
    and all the patient's checklist symptoms - the question must stay on the assigned one.

    question_expected: True = the reply must ask the patient a question unless it
    completes the check-in; False = it must not ask one; None = either is fine."""
    issues: List[str] = []
    reply = parsed.get("reply", "")

    if not reply:
        issues.append("the reply was empty")
    # A reply with no question leaves the patient with nothing to answer (they had to
    # type "next" to keep going).
    if question_expected is True and reply and "?" not in reply and not parsed.get("is_complete"):
        issues.append(
            "the reply did not ask the patient a question although the check-in is not "
            "complete; end the reply with exactly one question so the patient knows how to "
            "continue"
        )
    if question_expected is False and "?" in reply:
        issues.append(
            "the reply asked a question, but no question may be asked this turn; only "
            "acknowledge the patient's last answer"
        )
    if _reply_has_multiple_questions(reply):
        issues.append("the reply asked more than one question in a single turn")
    if reply and "?" in reply:
        _, suggestion_errors = _validate_suggested_answers(parsed.get("suggested_answers", []))
        issues.extend(suggestion_errors)
    # The patient reported a red flag, or did not answer the last question (so it is
    # being asked again): leaving the assigned symptom is expected. A new issue is not a
    # reason - it is queued, and the reply stays on the assigned symptom.
    leaving_assignment_ok = (
        parsed.get("red_flag")
        or parsed.get("answered_last_question") is False
    )
    if assigned_symptom and "?" in reply and not leaving_assignment_ok:
        selected = selected_symptoms or []
        reported = _match_symptom(parsed.get("symptom", ""), selected)
        if reported and reported != assigned_symptom:
            # Jumping ahead leaves the assigned symptom half-done and forces a jump back.
            issues.append(
                f'the question is about "{reported}", but this turn is assigned to '
                f'"{assigned_symptom}"; ask about "{assigned_symptom}" now - "{reported}" '
                "will be covered later"
            )
        topic = parsed.get("topic", "")
        allowed_topics = {_symptom_topic(label) for label in selected}
        if topic and topic not in allowed_topics:
            issues.append(
                f"the question is about {topic}, which the patient did not select; ask about "
                f'"{assigned_symptom}" instead'
            )
    already_confirmed = _asks_whether_patient_has(reply, confirmed_symptoms or [])
    if already_confirmed:
        issues.append(
            f'the reply asks whether the patient has "{already_confirmed}", but they already '
            "confirmed it on the checklist; ask them to describe it or ask for a missing "
            "detail instead"
        )
    # Asking an unanswered question again is intended, not a redundant re-ask.
    if (
        reply
        and parsed.get("answered_last_question") is not False
        and _looks_redundant(reply, chat_history)
    ):
        issues.append(
            "the reply re-asks a question the patient has already answered; ask a DIFFERENT "
            "question that gathers new information about the assigned topic instead"
        )

    # The patient is told the check-in ended while the app keeps the chat open.
    if reply and _has_closing_language(reply) and not parsed.get("is_complete"):
        issues.append(
            "the reply told the patient the check-in was complete but is_complete was false"
        )

    return issues


def _questions_asked(chat_history: List[Dict[str, str]]) -> int:
    """Number of questions the assistant has asked so far (used for pacing)."""
    return sum(
        1
        for message in chat_history
        if message.get("role") == "assistant" and "?" in message.get("content", "")
    )


def _symptom_topic(label: str) -> str:
    """The clinical topic a checklist symptom belongs to (used for the model's topic
    label and the doctor dashboard cards)."""
    return dict(CHECKLIST_ITEMS).get(label, "Other")


def _questions_per_symptom(chat_history: List[Dict[str, str]]) -> Dict[str, int]:
    """How many questions have been asked about each checklist symptom across the whole
    check-in. Each checkbox is tracked on its own - "Weight loss" and "Trouble eating or
    drinking" are counted separately even though both are Nutrition. A question is
    credited to the symptom code assigned for that turn (stored as "symptom")."""
    counts: Dict[str, int] = {}
    for message in chat_history:
        if message.get("role") != "assistant":
            continue
        if "?" not in message.get("content", ""):
            continue
        # The patient's reply did not answer it ("hi", or something unrelated): it is
        # asked again, so it does not use up the symptom's quota (Sep 27 finding).
        if message.get("unanswered"):
            continue
        symptom = (message.get("symptom") or "").strip()
        if symptom:
            counts[symptom] = counts.get(symptom, 0) + 1
    return counts


def _symptom_quota(
    label: str,
    worst_label: str,
    pace_mode: str = "normal",
    extra: Optional[Dict[str, int]] = None,
) -> int:
    """Questions allotted to one checklist symptom (QDA): 4 for the patient's worst
    symptom, 3 for each other - or 3 and 2 once the patient has pressed Speed up.
    `extra` (state.quota_extra) adds questions granted later, e.g. so a red flag reported
    on an already-covered symptom still gets its RED_FLAG_QUESTIONS."""
    if pace_mode == "faster":
        base = FAST_WORST_SYMPTOM_QUOTA if label == worst_label else FAST_OTHER_SYMPTOM_QUOTA
    else:
        base = WORST_SYMPTOM_QUOTA if label == worst_label else OTHER_SYMPTOM_QUOTA
    return base + (extra or {}).get(label, 0)


def _interview_order(labels: List[str]) -> List[str]:
    """Selected checklist symptoms in the order they are asked: clinical topic order
    (CHAT_TOPICS), then checklist order within a topic."""
    checklist_index = {label: index for index, (label, _topic) in enumerate(CHECKLIST_ITEMS)}
    return sorted(
        labels,
        key=lambda label: (
            CHAT_TOPICS.index(_symptom_topic(label)),
            checklist_index.get(label, len(checklist_index)),
        ),
    )


def _move_symptom_to_front(state: "SessionState", label: str) -> None:
    """Ask about this symptom next (adding it if the patient had not ticked it). The
    quota order picks the first unfilled symptom, so the front is asked next. Every item
    in the list gets its own box in the sidebar progress panel."""
    if label in state.selected_symptom_labels:
        state.selected_symptom_labels.remove(label)
    state.selected_symptom_labels.insert(0, label)
    topic = _symptom_topic(label)
    if topic not in state.selected_topics:
        state.selected_topics.append(topic)


def _add_symptom_to_queue(state: "SessionState", label: str) -> None:
    """Add this symptom to the END of the list, so it is asked after everything already
    queued (and shown as a box in the sidebar progress panel)."""
    if label not in state.selected_symptom_labels:
        state.selected_symptom_labels.append(label)
    topic = _symptom_topic(label)
    if topic not in state.selected_topics:
        state.selected_topics.append(topic)


def _sentence_name(label: str) -> str:
    """A queue item's name for use inside a sentence ("Knee pain" -> "knee pain")."""
    return label[:1].lower() + label[1:] if label else label


def _added_to_list_note(label: str, now: bool = False) -> str:
    """The one fixed sentence used whenever a symptom joins the patient's list (added in
    the sidebar or raised in chat), so every patient hears the same thing."""
    name = " and ".join(_sentence_name(part) for part in label.split(" and "))
    return f"I've added {name} to your list - " + (
        "let's talk about it now. " if now else "I'll ask you about it shortly. "
    )


def _queue_label(name: str) -> str:
    """The checklist box for a symptom the patient raised, or - when there is none - a
    new item named with the patient's own words (first letter capitalized)."""
    name = " ".join((name or "").split())[:60]
    if not name:
        return ""
    checklist = [label for label, _topic in CHECKLIST_ITEMS if label != "Something else"]
    return _match_symptom(name, checklist) or name[:1].upper() + name[1:]


def _quota_state(
    chat_history: List[Dict[str, str]],
    selected_labels: List[str],
    worst_label: str,
    pace_mode: str = "normal",
    extra: Optional[Dict[str, int]] = None,
) -> tuple[str, Dict[str, int], bool]:
    """(next_target_symptom, per_symptom_counts, all_quotas_met).

    The next target is the first selected checklist symptom whose quota is not yet
    filled, so code - not the model - controls symptom order, depth, and when the
    interview is finished."""
    counts = _questions_per_symptom(chat_history)
    target = ""
    for label in selected_labels:
        if label == CHECKLIST_NONE_LABEL:
            continue
        if counts.get(label, 0) < _symptom_quota(label, worst_label, pace_mode, extra):
            target = label
            break
    return target, counts, (target == "")


def _current_topic_question_run(chat_history: List[Dict[str, str]]) -> tuple[str, int]:
    """(symptom or topic, count) of the unbroken run of questions the interviewer has
    just asked about the same checklist symptom (or, for questions not assigned to a
    symptom, the same topic), walking backwards from the latest assistant question. Used
    to deterministically stop a single symptom from being over-questioned, regardless of
    what the model remembers. A non-question assistant turn (e.g. a pure acknowledgement)
    does not break the run; a question on a different symptom does.

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
        if message.get("unanswered"):
            # Not answered and asked again: only the repeat counts, as in the quotas.
            continue
        topic = (message.get("symptom") or message.get("topic") or "").strip()
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
    question_expected: Optional[bool] = True,
    confirmed_symptoms: Optional[List[str]] = None,
    assigned_symptom: str = "",
    selected_symptoms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """question_expected: True = the reply must ask a question; False = it must not ask one (e.g. the acknowledgement before the closing
    review screen); None = either is fine (e.g. a wrap-up offer answered with buttons).
    confirmed_symptoms: checklist symptoms the patient ticked (never ask whether they
    have them). assigned_symptom / selected_symptoms: the checklist symptom this turn must
    ask about, and all of the patient's checklist symptoms."""
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

    # ---- Hard stop (no API call): the final backstop that guarantees the check-in
    # closes and a doctor summary is generated even for a very long conversation. ----
    if questions_asked >= absolute_budget:
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
            "red_flag": False,
            "raw_response": "",
            "validation_errors": [],
            "completion_reason": "question_limit_reached",
        }

    steering = ""

    # Pacing steering: keep the conversation inside the (adaptive) question budget.
    if wrap_budget <= questions_asked:
        pacing_note = (
            f"Internal pacing check: You have already asked {questions_asked} questions. "
            "Keep each remaining question to the most essential detail still missing, and do "
            "not ask about symptoms that are not on the patient's list. Uncollected details "
            "will be listed for the doctor as unresolved."
        )
        steering = (steering + "\n\n" + pacing_note).strip()
    elif soft_budget <= questions_asked:
        pacing_note = (
            f"Internal pacing check: You have already asked {questions_asked} questions. "
            "Be selective from here on: prioritize safety-relevant details and skip optional "
            "follow-ups."
        )
        steering = (steering + "\n\n" + pacing_note).strip()

    # Caller-supplied steering (patient pace controls, one-time wrap check-in) is
    # appended last so it takes precedence over the generic pacing notes above.
    if extra_steering.strip():
        steering = (steering + "\n\n" + extra_steering.strip()).strip()

    parsed, raw_content = call_model(steering)
    issues = _find_response_issues(
        parsed, chat_history, question_expected, confirmed_symptoms,
        assigned_symptom, selected_symptoms,
    )
    validation_errors: List[str] = list(issues)

    # At most ONE corrective retry per patient turn (latency budget). The retry
    # message lists every detected issue so a single call can fix all of them.
    if issues:
        if question_expected is False:
            corrected = (
                ". Return a corrected response: a non-empty, warm reply that acknowledges the "
                "patient's last answer without asking any question, with an empty "
                "suggested_answers list, and all required JSON keys with no text outside the "
                "JSON object."
            )
        else:
            corrected = (
                ". Return a corrected response: a non-empty, warm reply that asks exactly one "
                "question about one clinical variable, does not repeat information the patient "
                "already gave, includes exactly five brief, distinct, directly relevant "
                "suggested_answers whenever the reply asks a question, and contains all required "
                "JSON keys with no text outside the JSON object."
            )
        quality_message = "Internal quality check failed: " + "; ".join(issues) + corrected
        if steering:
            quality_message += " " + steering
        retry_parsed, retry_raw = call_model(quality_message)
        if retry_parsed.get("reply"):
            parsed, raw_content = retry_parsed, retry_raw
            validation_errors.extend(
                f"after retry: {issue}"
                for issue in _find_response_issues(
                    parsed, chat_history, question_expected, confirmed_symptoms,
                    assigned_symptom, selected_symptoms,
                )
            )

    # ---- Local repairs: no further API calls. ----
    if not parsed.get("reply"):
        parsed["reply"] = (
            "Thank you for sharing that. What is the most important detail about that "
            "for your doctor to know?"
        )
        parsed["is_complete"] = False
        parsed["doctor_summary"] = ""

    # A compound question survived the retry: keep the first question only. The model's
    # own suggestions are kept - they were written for this question, unlike any canned set.
    if _reply_has_multiple_questions(parsed.get("reply", "")):
        repaired = _repair_multiple_questions(parsed["reply"])
        if repaired != parsed["reply"]:
            parsed["reply"] = repaired

    # The retry still asks whether the patient has a symptom they ticked: keep the
    # acknowledgement and replace the question with an invitation to describe it.
    already_confirmed = _asks_whether_patient_has(parsed.get("reply", ""), confirmed_symptoms or [])
    if already_confirmed:
        acknowledgement = _remove_questions(parsed["reply"]) if "?" in parsed["reply"] else ""
        if acknowledgement == "Thank you for sharing that with me.":
            acknowledgement = ""
        parsed["reply"] = (
            acknowledgement
            + f" You mentioned {already_confirmed.lower()} on your checklist - can you tell me "
            "more about that?"
        ).strip()
        parsed["suggested_answers"] = list(NO_QUESTION_FALLBACK_ANSWERS)
        parsed["symptom"] = already_confirmed

    # No question may be asked this turn, but the retry still asked one: keep only the
    # sentences that do not ask anything.
    if question_expected is False and "?" in parsed.get("reply", ""):
        parsed["reply"] = _remove_questions(parsed["reply"])
        parsed["suggested_answers"] = []

    # Only the app ends a check-in (closing screen, or the hard stop above); the model's
    # own is_complete is never trusted, and it never writes the doctor summary.
    parsed["is_complete"] = False

    # The check-in continues but the reply (even after the retry) asks nothing: add a
    # safe question so the patient is never left without something to answer.
    added_fallback_question = False
    if question_expected is True and "?" not in parsed.get("reply", ""):
        parsed["reply"] = (parsed["reply"].rstrip() + " " + NO_QUESTION_FALLBACK_QUESTION).strip()
        parsed["suggested_answers"] = list(NO_QUESTION_FALLBACK_ANSWERS)
        added_fallback_question = True

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
        "red_flag": bool(parsed.get("red_flag", False)),
        "red_flag_kind": parsed.get("red_flag_kind", ""),
        # What the patient's latest message did, as judged by the model.
        "answered_last_question": parsed.get("answered_last_question", True) is not False,
        "new_issue": parsed.get("new_issue", ""),
        "offer_choice": parsed.get("offer_choice", ""),
        "red_flag_symptom": parsed.get("red_flag_symptom", ""),
        # True when the app (not the model) added NO_QUESTION_FALLBACK_QUESTION.
        "fallback_question": added_fallback_question,
        # The checklist symptom the model says its question is about ("" if none).
        "reported_symptom": parsed.get("symptom", ""),
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


def _patient_summary_fallback(selected_labels: List[str]) -> str:
    labels = [label.lower() for label in selected_labels if label != CHECKLIST_NONE_LABEL]
    if not labels:
        return (
            "You told me you are doing okay today. This will be shared with your care team "
            "before your visit."
        )
    listed = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]
    return (
        f"Today we talked about your {listed}. Everything you told me will be shared with "
        "your care team before your visit."
    )


def get_patient_summary(
    client: OpenAI,
    chat_history: List[Dict[str, str]],
    model: str,
    selected_labels: Optional[List[str]] = None,
) -> str:
    """2-3 plain-language sentences for the patient: what will be shared with their care
    team. Falls back to a list of the symptoms discussed if the model call fails."""
    transcript = _transcript_lines(
        [
            m for m in chat_history
            if m.get("role") in ("assistant", "user")
            and m.get("response_mode") not in ("finish_button",)
        ]
    )
    try:
        raw = _create_chat_completion(
            client,
            model,
            [
                {"role": "system", "content": PATIENT_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
        )
        summary = json.loads(raw).get("summary", "")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
    except Exception as exc:
        logger.warning("Could not generate patient summary: %s", exc)
    return _patient_summary_fallback(selected_labels or [])


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
csv.field_size_limit(10_000_000)
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


SECRETS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".streamlit", "secrets.toml"
)
logger = logging.getLogger("checkin")


def _secret(name: str, default: Any = None) -> Any:
    # Same secrets file as before (only the Google Sheets entries are still used).
    try:
        with open(SECRETS_PATH, "rb") as secrets_file:
            return tomllib.load(secrets_file)[name]
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
            # A full check-in: every queued symptom covered before "Finish", or the
            # question-limit backstop. "wrapped_up_early" is not a full completion.
            all_data.get("completion_reason", "natural_completion")
            in ("closing_review_finish", "natural_completion", "question_limit_reached")
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
        logger.error("Could not connect to Google Sheets: %s", _sheet_error)
        return False
    try:
        row = _build_clean_report_row(name, all_data, report, system_prompt)
        _sheet.append_row([row[column] for column in LOCAL_CHAT_REPORT_COLUMNS])
        return True
    except Exception as exc:
        logger.error("Failed to save to Google Sheets: %s", exc)
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
        logger.error("Failed to save to local CSV: %s", exc)
        return False


def _load_prior_summary(patient_name: str) -> str:
    """Read this patient's most recent completed check-in from the LOCAL CSV records and
    build a prior-history note from that visit - the overview, the urgent items, AND the
    non-major reported issues per topic - so the next check-in (and the doctor
    dashboard's Prior history) shows the full picture from last time.

    The records live on this server (no cloud lookup, and nothing to fail silently on a
    credentials error). Returns an empty string when there is no prior record or no
    name - so it can never break a check-in that has no history."""
    name = (patient_name or "").strip().lower()
    if not name:
        return ""

    # Rows are appended chronologically, so the last matching row is the latest visit.
    match = None
    try:
        with open(LOCAL_CHAT_REPORT_PATH, newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                if str(row.get("patient_name", "")).strip().lower() == name:
                    match = row
    except FileNotFoundError:
        return ""
    except Exception as exc:
        logger.error("Could not read prior check-ins from %s: %s", LOCAL_CHAT_REPORT_PATH, exc)
        return ""
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


def _prior_topic_notes(prior_history: str) -> Dict[str, str]:
    """What the last check-in recorded per clinical topic, taken from the "Reported
    issues by topic" lines of the note _load_prior_summary builds ("- Pain [worse]:
    ..."). Empty for hand-typed history, which has no such lines."""
    notes: Dict[str, str] = {}
    for line in (prior_history or "").splitlines():
        match = re.match(r"^- (.+?)(?: \[(?:worse|better)\])?: (.+)$", line.strip())
        if match and match.group(1) in SUMMARY_TOPICS:
            notes[match.group(1)] = match.group(2).strip()
    return notes


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
    symptoms_selected: Optional[List[str]] = None,
    disclaimer_acknowledged_at: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "check-in-session-v4",
        "symptoms_selected": list(symptoms_selected or []),
        "disclaimer_acknowledged_at": disclaimer_acknowledged_at,
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

class SessionState(SimpleNamespace):
    """Per-patient session state (the web-app counterpart of st.session_state)."""

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def new_session_state() -> SessionState:
    state = SessionState()
    reset_chat(state)
    return state


def reset_chat(state: SessionState) -> None:
    state.messages = []
    state.is_complete = False
    state.doctor_summary = ""
    state.started = False
    state.check_in_started = False
    state.raw_responses = []
    state.current_topic = ""
    state.completed_topics = []
    state.doctor_summary_structured = {}
    state.summary_generated = False
    state.sheet_saved = False
    state.local_csv_saved = False
    state.show_suggestions = SHOW_SUGGESTIONS_BY_DEFAULT
    state.closing_review = False
    state.pace_mode = "normal"
    state.wrap_choice_offered = False
    state.stop_choice_offered = False
    state.pending_offer = None
    state.offer_choice = None
    state.addon_notice = None
    state.addon_toast = None
    # Patient-facing summary shown after the check-in, before the doctor's report.
    # Extra questions granted per symptom (a red flag on an already-covered symptom).
    state.quota_extra = {}
    # Queue items created for red flags, so each red flag triggers its questions once.
    state.red_flag_items = []
    state.patient_summary = ""
    state.patient_summary_generated = False
    state.patient_summary_viewed = False
    # id of the last processed "send", so a retried request is not answered twice.
    state.last_send_id = ""
    state.worst_topic = ""
    state.worst_label = ""
    state.worst_pick_pending = False
    state.judge_directive = ""
    state.rolling_summary = ""
    state.summary_tail_start = 0
    state.pending_addon = None
    state.session_started_at = datetime.now().astimezone().isoformat()
    state.session_id = hashlib.sha256(
        state.session_started_at.encode("utf-8")
    ).hexdigest()[:16]
    state.session_errors = []
    state.completion_reason = ""
    state.completed_at = ""
    state.selected_topics = []
    state.selected_symptom_labels = []
    state.disclaimer_acknowledged = False
    state.disclaimer_acknowledged_at = None
    state.saved_prior_history = ""
    state.prior_history_status = ""
    state.saved_system_prompt = SYSTEM_PROMPT
    state.saved_patient_name = ""
    state.saved_doctor_name = ""
    state.saved_therapy_week = ""


def render_topic_boxes(state: SessionState) -> str:
    # One box per checkbox the patient ticked. Each box tracks its own symptom: "Weight
    # loss" and "Trouble eating or drinking" are separate boxes with separate progress,
    # even though both belong to the Nutrition topic.
    display = [
        label
        for label in state.get("selected_symptom_labels", [])
        if label != CHECKLIST_NONE_LABEL
    ]

    # A symptom counts as covered once its question QUOTA is filled - not when the model
    # happens to switch away from it, which left the final one permanently "uncovered".
    counts = _questions_per_symptom(state.messages)
    worst_label = state.get("worst_label", "")
    active_symptom = next(
        (
            message.get("symptom", "")
            for message in reversed(state.messages)
            if message.get("role") == "assistant" and "?" in message.get("content", "")
        ),
        "",
    )

    def _is_covered(label: str) -> bool:
        return counts.get(label, 0) >= _symptom_quota(
            label, worst_label, state.pace_mode, state.get("quota_extra", {})
        )

    topic_boxes = ""
    for label in display:
        topic_classes = ["topic-box"]
        if _is_covered(label):
            topic_classes.append("topic-complete")
        elif label == active_symptom:
            topic_classes.append("topic-active")
        topic_boxes += f'<div class="{" ".join(topic_classes)}">{html.escape(label)}</div>'

    total_topics = len(display)
    done_topics = sum(1 for label in display if _is_covered(label))
    progress_pct = int((done_topics / total_topics) * 100) if total_topics else 0

    return f"""
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
        """


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

def build_doctor_summary_page(state: SessionState) -> Dict[str, Any]:
    """Everything the doctor dashboard screen shows. The page itself (buttons, the
    expanders, the JSON download) is rendered by static/index.html."""
    page: Dict[str, Any] = {"warning": "", "html": ""}

    summary = state.doctor_summary_structured
    if not summary:
        page["warning"] = (
            "No summary is available - the generator hit an error. "
            "Click the button below to try again."
        )
        return page

    if summary_is_degenerate(summary):
        page["warning"] = (
            "The summary generator returned no content for this check-in, so the "
            "dashboard below is empty. Click 'Regenerate summary' at the bottom of "
            "the page (or below) to try again."
        )

    overview = str(summary.get("Overview", "")).strip()
    overview_text = overview or _limit_to_three_sentences(state.doctor_summary)
    overview_html = _basic_md_to_html(overview_text, inline=True)

    discussed_count = sum(
        1
        for topic in SUMMARY_TOPICS
        if summary.get(f"{topic}_Main issues", "").strip()
        or summary.get(f"{topic}_more details", "").strip()
    )

    started_at = state.get("session_started_at", "")
    try:
        check_in_datetime = datetime.fromisoformat(started_at).strftime(
            "%b %d, %Y · %I:%M %p"
        )
    except (TypeError, ValueError):
        check_in_datetime = str(started_at) or "Time unavailable"

    patient_name = state.get("saved_patient_name", "").strip() or "Unnamed patient"
    doctor_name = state.get("saved_doctor_name", "").strip()
    therapy_week = state.get("saved_therapy_week", "").strip() or "Week not specified"
    doctor_meta = (
        f'<span><b>Doctor:</b> {_inline_text(doctor_name)}</span>'
        if doctor_name
        else ""
    )

    duration_note = ""
    try:
        completed_at_dt = datetime.fromisoformat(state.get("completed_at", ""))
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

    # Every red flag the patient was shown a warning for must be on the URGENT strip,
    # even if the summarizer left it out (the chat and the dashboard must agree).
    flagged_text = " ".join(
        f"{flag.get('label', '')} {flag.get('reason', '')}" if isinstance(flag, dict) else str(flag)
        for flag in summary.get("Urgent_flags", [])
    ).lower()
    for red_flag_label in state.get("red_flag_items", []):
        if red_flag_label.lower() not in flagged_text:
            urgent_items.append(
                f'<span class="urgent-item"><strong>{_inline_text(red_flag_label)}</strong>'
                '<span class="urgent-reason">Red flag reported during the check-in</span></span>'
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
    # Questions the patient did not answer even after they were repeated (the patient was
    # told these go to the doctor), listed by the code rather than left to the summarizer.
    for message in state.messages:
        if message.get("role") == "assistant" and message.get("unanswered_final"):
            question = " ".join(
                part for part in re.split(r"(?<=[.!?])\s+", message.get("content", "")) if "?" in part
            )
            if question:
                unresolved_items.append(f"Not answered: {_inline_text(question)}")
    # What the patient wrote on the closing "Anything else?" page, word for word, taken
    # straight from the conversation (not from the summarizer).
    patient_comments = [
        str(message.get("free_text", "")).strip()
        for message in state.messages
        if message.get("role") == "user"
        and message.get("response_mode") in ("closing_other", "closing_addon", "summary_correction")
        and str(message.get("free_text", "")).strip()
    ]
    patient_comments_html = ""
    if patient_comments:
        patient_comments_html = (
            '<section class="patient-comments-strip">'
            '<span class="slot-label">Patient added (own words)</span>'
            + "".join(
                f'<div class="patient-comment">&ldquo;{_inline_text(comment)}&rdquo;</div>'
                for comment in patient_comments
            )
            + "</section>"
        )

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
      .patient-comments-strip {{ margin-top:0.28rem; padding:0.38rem 0.55rem; border:2px solid #2563eb;
        border-radius:7px; background:#eff6ff; color:#1e3a8a; font-size:0.84rem; line-height:1.3; }}
      .patient-comments-strip .slot-label {{ display:block; font-size:0.65rem; font-weight:800;
        letter-spacing:0.06em; text-transform:uppercase; color:#1d4ed8; margin-bottom:0.15rem; }}
      .patient-comment {{ font-weight:600; }}
      .patient-comment + .patient-comment {{ margin-top:0.2rem; }}
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
      {patient_comments_html}
      {unresolved_html}
      <div class="legend" aria-label="Status legend">
        <span><i class="red"></i>&#9650; New / worsening</span><span><i class="green"></i>&#9660; Improving</span>
        <span><i class="gray"></i>Reported</span><span>&#10003; None reported</span><span><i class="muted-dot"></i>&#9675; Not assessed</span>
      </div>
      <section class="topic-grid">{cards_html}</section>
    </main>
    """).strip()

    page["html"] = dashboard_html
    page["prior_history"] = state.get("saved_prior_history", "") or "No prior history provided."
    page["conversation"] = [
        {
            "label": "Virtual doctor" if message["role"] == "assistant" else "Patient",
            "mode": message.get("response_mode") or "",
            "content": message.get("content", ""),
            "suggested_answers": message.get("suggested_answers") or [],
        }
        for message in state.messages
    ]
    return page


def build_export_payload(state: SessionState) -> Dict[str, Any]:
    """The dashboard's "Download reproducibility record (JSON)"."""
    return build_sheet_payload(
        patient_name=state.saved_patient_name,
        doctor_name=state.saved_doctor_name,
        therapy_week=state.saved_therapy_week,
        prior_history=state.saved_prior_history,
        messages=state.messages,
        doctor_summary=state.doctor_summary,
        structured_summary=state.doctor_summary_structured,
        system_prompt=state.saved_system_prompt,
        model=DEFAULT_MODEL,
        session_id=state.session_id,
        session_started_at=state.session_started_at,
        session_errors=state.session_errors,
        completion_reason=state.completion_reason or "natural_completion",
        symptoms_selected=state.selected_symptom_labels,
        disclaimer_acknowledged_at=state.disclaimer_acknowledged_at,
    ) | {"patient_summary": state.get("patient_summary", "")}


def apply_nurse_result(state: SessionState, result: Dict[str, Any]) -> None:
    """Shared bookkeeping after every nurse-model turn (chat reply or checklist kickoff)."""
    state.raw_responses.append(result["raw_response"])
    if result.get("validation_errors"):
        state.session_errors.extend(result["validation_errors"])
    if (
        state.current_topic
        and state.current_topic != result["topic"]
        and state.current_topic not in state.completed_topics
    ):
        state.completed_topics.append(state.current_topic)
    # A topic the model returns to is active again, not completed.
    if result["topic"] and result["topic"] in state.completed_topics:
        state.completed_topics.remove(result["topic"])
    state.current_topic = result["topic"]

    # NOTE: topics are never auto-added here. A symptom the patient merely MENTIONS while
    # answering something else must not become a new interview topic - only symptoms the
    # patient explicitly selects (opening checklist, "Add a symptom", or the closing
    # review) are covered. Mentioned symptoms still reach the doctor via the summary.

    add_assistant_message(
        state,
        result["reply"],
        result.get("suggested_answers", []),
        result.get("topic", ""),
        result.get("symptom", ""),
        result.get("red_flag", False),
        result.get("red_flag_kind", ""),
    )
    if result.get("is_reask"):
        # The repeat of an unanswered question: it is not asked a third time.
        state.messages[-1]["is_reask"] = True

    state.is_complete = result["is_complete"]
    if result.get("completion_reason"):
        state.completion_reason = result["completion_reason"]
    elif state.is_complete and not state.completion_reason:
        state.completion_reason = "natural_completion"
    if state.is_complete and not state.completed_at:
        state.completed_at = datetime.now().astimezone().isoformat()
    state.doctor_summary = result["doctor_summary"]


def add_assistant_message(
    state: SessionState,
    content: str,
    suggested_answers: Optional[List[str]] = None,
    topic: str = "",
    symptom: str = "",
    red_flag: bool = False,
    red_flag_kind: str = "",
) -> None:
    state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "suggested_answers": suggested_answers or [],
            "topic": topic,
            # The checklist symptom this question counts toward ("" if none).
            "symptom": symptom,
            # True when the patient just reported a red flag: the page shows RED_FLAG_NOTICE.
            "red_flag": red_flag,
            # Which red flag the app detected ("" if only the model flagged it); picks
            # the notice shown (SELF_HARM_NOTICE for "self_harm").
            "red_flag_kind": red_flag_kind,
        }
    )


def add_user_message(state: SessionState, content: str, response_mode: str = "typed") -> None:
    state.messages.append(
        {
            "role": "user",
            "content": content,
            "response_mode": response_mode,
        }
    )


def _previous_question(state: SessionState) -> Optional[Dict[str, Any]]:
    """The assistant question the patient's latest (typed or tapped) message replies to,
    or None - e.g. after a checklist tap, a button, or a reply that asked nothing."""
    if len(state.messages) < 2:
        return None
    latest, previous = state.messages[-1], state.messages[-2]
    if latest.get("role") != "user" or latest.get("response_mode") not in PATIENT_TEXT_MODES:
        return None
    if previous.get("role") != "assistant" or "?" not in previous.get("content", ""):
        return None
    if previous.get("is_offer"):
        # The wrap-up offer is answered with a choice, not a symptom answer to repeat.
        return None
    return previous


def _may_reask(previous_question: Optional[Dict[str, Any]]) -> bool:
    """A question may be asked again once: not if it already is the repeat."""
    return previous_question is not None and not previous_question.get("is_reask")


def _latest_message_steering(state: SessionState, previous_question: Optional[Dict[str, Any]]) -> str:
    """Tells the model what its previous question was, and whether it may still ask it
    again if the patient's reply does not answer it (once per question)."""
    if previous_question is None:
        return ""
    label = previous_question.get("symptom", "")
    lines = [
        f'Your previous question was: "{previous_question.get("content", "")}". Before '
        "following the topic assignment, decide whether the patient's latest message "
        "answers it (see Non-Answers and New Issues)."
    ]
    if _may_reask(previous_question):
        lines.append(
            "If it does not, set answered_last_question to false and gently ask that question "
            "again in simpler words instead of the assigned question"
            + (f', with "symptom" set to "{label}"' if label else "")
            + "."
        )
    else:
        lines.append(
            "If it does not, set answered_last_question to false but do NOT ask it again - it "
            "was already asked twice and will be listed for the doctor as unanswered. Go on "
            "to the next question of the assignment."
        )
    lines.append(
        "A red flag in their message still comes first, and a new unrelated issue is only "
        "noted (see Non-Answers and New Issues)."
    )
    return " ".join(lines)


def _symptom_has_quota_left(state: SessionState, label: str) -> bool:
    asked = _questions_per_symptom(state.messages).get(label, 0)
    return asked < _symptom_quota(label, state.worst_label, state.pace_mode, state.quota_extra)


def _red_flag_item(result: Dict[str, Any]) -> str:
    """The queue item for a red flag the model reported ("" if none). A kind with a
    checklist box always uses that box (e.g. breathing -> "Breathing problems"), whatever
    the model called it; otherwise the model's name in the patient's words ("Chest
    pain"), or a default name for the kind."""
    if not result.get("red_flag"):
        return ""
    kind = result.get("red_flag_kind") or "other"
    if kind in RED_FLAG_LABELS:
        return RED_FLAG_LABELS[kind]
    return _queue_label(result.get("red_flag_symptom", "")) or RED_FLAG_DEFAULT_NAMES.get(
        kind, RED_FLAG_DEFAULT_NAMES["other"]
    )


def _apply_patient_signals(
    state: SessionState,
    result: Dict[str, Any],
    previous_question: Optional[Dict[str, Any]],
    current_label: str = "",
) -> None:
    """Act on what the model says the patient's latest message did. The model judges;
    the code decides where things go, so the behaviour is the same every time:
    - did not answer -> the question is not counted and is asked again once; if the
      repeat is not answered either, it is counted, marked for the doctor, and the bot
      goes on to the next question;
    - new red flag -> its queue item (checklist box, or a new item in the patient's
      words) moves to the FRONT with up to RED_FLAG_QUESTIONS questions, this reply's
      follow-up being the first; then the list continues where it was;
    - new issue not related to the current topic -> added to the END of the list
      (checklist box, or a new item in the patient's words), and the reply says it will
      be asked about shortly.
    Every item added gets its own box in the sidebar progress panel."""
    latest = state.messages[-1] if state.messages else {}
    if previous_question is not None and not result.get("answered_last_question", True):
        latest["answered"] = False
        if _may_reask(previous_question):
            previous_question["unanswered"] = True
            # This reply asks it again (unless it follows up a red flag instead), so a
            # second non-answer moves on.
            if "?" in result.get("reply", "") and not result.get("red_flag"):
                result["is_reask"] = True
        else:
            previous_question["unanswered_final"] = True

    red_label = _red_flag_item(result)
    if red_label and red_label not in state.red_flag_items:
        state.red_flag_items.append(red_label)
        _move_symptom_to_front(state, red_label)
        asked = _questions_per_symptom(state.messages).get(red_label, 0)
        remaining = (
            _symptom_quota(red_label, state.worst_label, state.pace_mode, state.quota_extra) - asked
        )
        if remaining < RED_FLAG_QUESTIONS:
            state.quota_extra[red_label] = (
                state.quota_extra.get(red_label, 0) + RED_FLAG_QUESTIONS - remaining
            )
        latest["red_flag_item"] = red_label
        if "?" in result.get("reply", ""):
            # The follow-up in this reply is the red flag's first question.
            result["reported_symptom"] = red_label

    issue_label = _queue_label(result.get("new_issue", ""))
    if (
        issue_label
        and issue_label not in (red_label, current_label, CHECKLIST_NONE_LABEL)
    ):
        if issue_label not in state.selected_symptom_labels:
            _add_symptom_to_queue(state, issue_label)
            latest["queued_issue"] = issue_label
            result["queued_issue"] = issue_label
        elif _symptom_has_quota_left(state, issue_label):
            result["queued_issue"] = issue_label
        else:
            result["noted_issue"] = issue_label


def _credit_question(state: SessionState, result: Dict[str, Any], assigned_symptom: str = "") -> None:
    """Credit the question to the checklist symptom it is actually about. The model names
    it; if it names one of the patient's symptoms, that one is counted - so a question
    that jumped ahead to another symptom (despite the retry) still uses up THAT
    symptom's quota instead of being asked again later. Otherwise fall back to the
    assigned symptom when the model's topic matches it. A fallback question the app
    added itself ("anything more about that?") always counts for the assigned symptom."""
    selected = [label for label in state.selected_symptom_labels if label != CHECKLIST_NONE_LABEL]
    reported_symptom = _match_symptom(result.get("reported_symptom", ""), selected)
    if "?" not in result.get("reply", ""):
        result["symptom"] = ""
    elif reported_symptom and not result.get("fallback_question"):
        result["symptom"] = reported_symptom
    elif assigned_symptom and (
        result.get("topic") == _symptom_topic(assigned_symptom)
        or result.get("fallback_question")
    ):
        result["symptom"] = assigned_symptom
    else:
        result["symptom"] = ""


def _acknowledge_then_closing_review(
    state: SessionState,
    client: OpenAI,
    prior_history: str,
    patient_context: str,
    model: str,
    system_prompt: str,
    symptom_count: int,
    previous_question: Optional[Dict[str, Any]] = None,
    latest_steering: str = "",
) -> None:
    """Every selected topic is covered. Before the closing review screen replaces the
    chat input, the interviewer replies once - without a question - so the patient's
    last answer (possibly a red flag) is acknowledged.

    Two exceptions keep the chat open instead: the patient's last message reports a red
    flag (it is followed up with one question), or it did not answer the last question
    (it is asked again, once per question). Both are the model's judgment,
    reported in red_flag / answered_last_question."""
    messages_snapshot = [
        {
            "role": m["role"],
            "content": m.get("content", ""),
            "suggested_answers": m.get("suggested_answers", []),
        }
        for m in state.messages
    ]
    # A red flag in the last few messages: acknowledge it specifically and show the
    # notice again, rather than a generic "that's all my questions".
    recent_kind, recent_red_flag = "", False
    for message in state.messages[-4:]:
        if message.get("role") == "assistant" and message.get("red_flag"):
            recent_red_flag = True
            recent_kind = message.get("red_flag_kind") or recent_kind
    steering = ACKNOWLEDGE_BEFORE_REVIEW_STEERING + "\n\n" + ACKNOWLEDGE_EXCEPTIONS_STEERING
    if latest_steering:
        steering += "\n\n" + latest_steering
    if recent_red_flag:
        steering += "\n\n" + RED_FLAG_CLOSING_STEERING.format(
            description=RED_FLAG_DESCRIPTIONS.get(recent_kind, "a red-flag symptom")
        )
    result = get_nurse_response(
        client=client,
        chat_history=messages_snapshot,
        prior_history=prior_history,
        patient_context=patient_context,
        model=model,
        system_prompt=system_prompt,
        symptom_count=symptom_count,
        extra_steering=steering,
        rolling_summary=state.rolling_summary if ENABLE_ROLLING_SUMMARY else "",
        summary_tail_start=state.summary_tail_start if ENABLE_ROLLING_SUMMARY else 0,
        question_expected=None,
    )
    if result.get("completion_reason"):
        # The absolute question limit closed the check-in instead.
        apply_nurse_result(state, result)
        return

    # Keep the chat open only when the model reports a reason to ask: a new red flag, or
    # an unanswered question that may still be asked again. Anything else is closed as
    # usual, with any stray question removed.
    may_reask = _may_reask(previous_question)
    # A red flag counts only if it is new - one already flagged earlier in the check-in
    # got its questions then.
    red_label = _red_flag_item(result)
    new_red_flag = bool(red_label) and red_label not in state.red_flag_items
    keeps_chat_open = "?" in result.get("reply", "") and (
        new_red_flag
        or (not result.get("answered_last_question", True) and may_reask)
    )
    if keeps_chat_open:
        _apply_patient_signals(state, result, previous_question)
        _credit_question(state, result)
        result.update(is_complete=False, doctor_summary="")
        apply_nurse_result(state, result)
        return
    # A new issue raised in the last answer: queue it and ask about it now, the same way
    # as any queued item, instead of closing.
    issue_label = _queue_label(result.get("new_issue", ""))
    if (
        issue_label
        and issue_label != CHECKLIST_NONE_LABEL
        and issue_label not in state.selected_symptom_labels
    ):
        _add_symptom_to_queue(state, issue_label)
        state.messages[-1]["queued_issue"] = issue_label
        generate_and_apply_turn(state, client, prior_history, patient_context, model, system_prompt)
        return
    if previous_question is not None and not result.get("answered_last_question", True):
        # Closing anyway: the unanswered question is listed for the doctor.
        state.messages[-1]["answered"] = False
        previous_question["unanswered_final"] = True
    if "?" in result.get("reply", ""):
        result["reply"] = _remove_questions(result["reply"])

    result.update(is_complete=False, doctor_summary="", suggested_answers=[], topic="", symptom="")
    # Clinician feedback (Sep 17): going straight from the last answer to "anything
    # else?" felt abrupt. Fixed wording (not model-written) so it cannot trip the
    # closing-language checks or promise that the check-in is complete.
    recent_red_flag = recent_red_flag or bool(result.get("red_flag"))
    recent_kind = recent_kind or result.get("red_flag_kind", "")
    done_note = RED_FLAG_DONE_NOTE if recent_red_flag else QUESTIONS_DONE_NOTE
    result["reply"] = (result["reply"].rstrip() + " " + done_note).strip()
    if recent_red_flag:
        result["red_flag"] = True
        result["red_flag_kind"] = recent_kind
    apply_nurse_result(state, result)
    state.closing_review = True


def generate_and_apply_turn(
    state: SessionState,
    client: OpenAI,
    prior_history: str,
    patient_context: str,
    model: str,
    system_prompt: str,
) -> None:
    """Produce one interview-agent reply for the latest patient message and apply it.
    Assumes state.messages already ends with the user message to respond to. Shared by the normal send flow and the mid-conversation
    "add a symptom" action so both get identical pacing / judge / summary behavior."""
    symptom_count = len(
        [l for l in state.selected_symptom_labels if l != CHECKLIST_NONE_LABEL]
    )
    soft_budget, wrap_budget, hard_budget, absolute_budget = _effective_budget(symptom_count)
    questions_so_far = _questions_asked(state.messages)
    # Current same-symptom run, used by the per-symptom cap and the add-a-symptom notice.
    run_topic, run_count = _current_topic_question_run(state.messages)

    # ---- The patient's latest message ----
    # The model judges what it did (answered the question? asked to switch symptom?
    # reported a red flag?) in the same call as its reply; the code acts on that after
    # the reply (_apply_patient_signals). Here it only tells the model what the previous
    # question was and whether it may still be asked again.
    previous_question = _previous_question(state)
    latest_steering = _latest_message_steering(state, previous_question)

    # ---- Deterministic quota control ----
    # Code decides which checklist symptom is asked next and how deep to go, so the 4/3
    # split and the overall length are guaranteed rather than left to the model.
    target_symptom, symptom_counts, quotas_met = _quota_state(
        state.messages,
        state.selected_symptom_labels,
        state.worst_label,
        state.pace_mode,
        state.quota_extra,
    )
    # ---- Deterministic close ----
    # Checked BEFORE generating: the patient has just answered, and every selected topic
    # has filled its quota, so there is nothing left to ask. Doing this after generating
    # would leave the final question asked-but-unanswered. The interviewer still
    # acknowledges the last answer
    # (without a question) so a red flag reported there is not met with silence.
    # The acknowledgement may still follow up a red flag or re-ask an unanswered
    # question instead, in which case the check-in stays open.
    # "Keep going" continues the list; once nothing is left, it ends the same way.
    if quotas_met and not state.is_complete:
        _acknowledge_then_closing_review(
            state, client, prior_history, patient_context, model, system_prompt, symptom_count,
            previous_question, latest_steering,
        )
        return

    quota_steering = ""
    if target_symptom and not quotas_met:
        asked = symptom_counts.get(target_symptom, 0)
        quota = _symptom_quota(target_symptom, state.worst_label, state.pace_mode, state.quota_extra)
        target_topic = _symptom_topic(target_symptom)
        symptom_text = (
            'the "Something else" concern the patient described on the checklist'
            if target_symptom == "Something else"
            else f'"{target_symptom}"'
        )
        worst_note = (
            " This is the symptom the patient said bothers them most, so cover it most "
            "thoroughly."
            if target_symptom == state.worst_label
            else ""
        )
        # The patient ticked it, so asking "have you had any nausea?" would be redundant.
        prior_note = _prior_topic_notes(prior_history).get(target_topic, "") if asked == 0 else ""
        if not _symptom_mentioned(target_symptom, prior_note):
            # e.g. last visit's GI line was about nausea - don't compare diarrhea with it.
            prior_note = ""
        confirmed_note = (
            " The patient already confirmed on the checklist that they have this symptom, so "
            "never ask whether they have it"
            + (
                "; invite them to describe it instead (for example \"Can you tell me about "
                "the nausea you've been having?\")."
                if asked == 0
                else "."
            )
        )
        if prior_note:
            # Tell the patient what was recorded last time and ask how it is now.
            confirmed_note = (
                " The patient already confirmed on the checklist that they have this symptom, so "
                "never ask whether they have it."
                f' Their last check-in recorded for {target_topic}: "{prior_note}". In this '
                "question, briefly tell the patient what was recorded last time and ask how it "
                "is today, for example \"I see your pain was 6 out of 10 at your last check-in - "
                "how severe is it today?\""
            )
        # Clinician feedback (Sep 17): the chatbot asked what the patient does for
        # constipation but never for pain or fever. From the second question on, remind
        # the model so one of this symptom's questions covers treatment.
        treatment_note = (
            " If the patient has not yet said what they are doing for this symptom "
            "(medication or anything else) and whether it helps, make sure one of your "
            "remaining questions about it asks that."
            if asked >= 1
            else ""
        )
        quota_steering = (
            f"Topic assignment (follow exactly): ask your next question about {symptom_text}"
            f" (clinical topic: {target_topic})." + worst_note + confirmed_note + treatment_note
            + " The patient's checklist symptoms are: "
            + "; ".join(f'"{label}"' for label in state.selected_symptom_labels if label != CHECKLIST_NONE_LABEL)
            + f'. Set "symptom" to "{target_symptom}" for this question.'
            + f" You have asked {asked} of {quota} planned questions about it. Ask ONE question "
            "about this symptom only. Do not ask about any other symptom or topic, and do not "
            "ask the final anything-else question yet. (The Red Flags and Non-Answers and New "
            "Issues rules still come first.)"
        )

    pace_steering = ""
    if state.pace_mode == "faster":
        pace_steering = (
            "The patient has asked to move faster or is getting tired. Keep this reply "
            "especially brief and ask for the single most important detail still missing. "
            "The app has already reduced the number of questions per symptom."
        )
    elif state.pace_mode == "slower":
        pace_steering = (
            "The patient has asked to slow down and share more. Give them room: warmly "
            "acknowledge what they said and invite them to add any detail before you move "
            "on; do not rush ahead."
        )

    # One-time agency check-ins: at the wrap threshold (are you tired? wrap up or keep
    # going) and again at the hard threshold (we have enough - stop, or continue).
    # Nothing closes silently; the patient always chooses. The higher threshold is
    # checked first so a big jump lands on the right message.
    # Selected symptoms not discussed yet, used by both the offers and the follow-up that
    # reacts to the patient's answer. A just-added symptom is named separately.
    remaining_topics = [
        label
        for label in state.selected_symptom_labels
        if label != CHECKLIST_NONE_LABEL
        and symptom_counts.get(label, 0) == 0
        and label not in (run_topic, state.addon_notice)
    ]
    remaining_text = ", ".join(remaining_topics)

    # The patient is answering an offer we made last turn - tell the interviewer how to
    # honour either choice, so "let's continue" actually delivers what was promised.
    # A symptom was just added mid-chat: acknowledge it, but finish what is already in
    # progress first. The interviewer must not abandon the current topic to jump to it.
    addon_steering = ""
    addon_notice = state.addon_notice
    # The added symptom can already be next in line (the current symptom's questions are
    # used up), in which case it is asked now rather than "shortly".
    addon_is_next = bool(addon_notice and target_symptom and target_symptom in addon_notice)
    if addon_notice:
        state.addon_notice = None
        state.addon_toast = None
        if addon_is_next:
            addon_steering = (
                f'The patient has just added "{addon_notice}" to their list. It has ALREADY been '
                "acknowledged for you, so do not thank them for it or mention it again."
            )
        else:
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
    typed_offer_reply = False
    pending_offer = state.pending_offer
    offer_choice = state.offer_choice
    if pending_offer:
        state.pending_offer = None
        state.offer_choice = None
        # One strong, UNCONDITIONAL instruction per (threshold, choice). The model is
        # never asked to work out which branch applies - code already knows.
        # Tapping "Wrap up now" / "Finish now" goes straight to the closing screen without
        # a model turn (action_offer), so only "Keep going" and typed replies reach here.
        if pending_offer == "wrap" and offer_choice == "continue":
            offer_followup = (
                "The patient explicitly chose to KEEP GOING. Continue the check-in now: cover the "
                "selected topics that have not come up yet"
                + (f" ({remaining_text})" if remaining_text else "")
                + ". Ask brief, essential questions, one at a time, and do not introduce topics "
                "the patient did not select. Do not offer to wrap up again."
            )
        elif pending_offer == "stop" and offer_choice == "continue":
            offer_followup = (
                "The patient explicitly chose to KEEP GOING even though you already have more "
                "than enough for the doctor. Continue with the topic assignment, keep each "
                "question brief, and do not offer to stop again."
            )
        else:
            # Typed instead of tapped: the model reads the choice, the code acts on it
            # exactly as the buttons do (Sep 28 fix - a typed "wrap up" used to be
            # overridden by the next queued question).
            typed_offer_reply = True
            offer_followup = TYPED_OFFER_REPLY_STEERING

    # ---- The wrap-up questions, at Slobodan's counts ----
    # 12 = silent (internal speed-up only), 16 = "are you tired, wrap up or continue?",
    # 22 = "we have enough for the doctor - stop, or keep going?". Triggered on the
    # QUESTION COUNT exactly as he specified, not on quota progress.
    topics_still_to_cover = [
        label
        for label in state.selected_symptom_labels
        if label != CHECKLIST_NONE_LABEL
        and symptom_counts.get(label, 0) < _symptom_quota(label, state.worst_label, state.pace_mode, state.quota_extra)
    ]
    agency_steering = ""
    if questions_so_far >= hard_budget and not state.stop_choice_offered:
        agency_steering = (
            "Wrap-up check-in (do this once, this turn only): The check-in has become quite "
            "long. This turn, do NOT ask another clinical question. Instead, warmly reassure "
            "the patient that you already have more than enough to share with their doctor, and "
            "offer a clear choice: you can stop here now, or keep going if they want to provide "
            "more. Make it entirely their choice, with no pressure either way. is_complete must "
            "be false and doctor_summary must be an empty string."
        )
        quota_steering = ""
        state.stop_choice_offered = True
        state.wrap_choice_offered = True
        state.pending_offer = "stop"
    elif questions_so_far >= wrap_budget and not state.wrap_choice_offered:
        done_text = ", ".join(
            label
            for label in state.selected_symptom_labels
            if label != CHECKLIST_NONE_LABEL and label not in topics_still_to_cover
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
        state.wrap_choice_offered = True
        state.pending_offer = "wrap"

    # Supervisor directive from the PREVIOUS turn's judge (one-turn lag, so the judge
    # never sits in the critical path). Applied as the highest-priority steering.
    directive_steering = ""
    if ENABLE_JUDGE_AGENT and state.judge_directive:
        directive_steering = (
            "Supervisor directive (a pacing check has flagged this - follow it now): "
            + state.judge_directive
            + " This directive does not apply if you are asking an unanswered question again "
            "or following up a red flag - do that instead."
            + (
                " The topic assignment above still decides which symptom you ask about: if this "
                "directive suggests moving to another symptom, instead ask about a different, "
                "not-yet-covered aspect of the assigned symptom."
                if quota_steering
                else ""
            )
        )
        state.judge_directive = ""

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
    }.get(state.pace_mode, "")
    judge_pacing_note = (
        "Pacing status (computed by a reliable counter - trust these, do not recount):\n"
        f"- Questions asked so far: {questions_so_far} "
        f"(soft nudge at {soft_budget}, wrap at {wrap_budget}, hard offer at {hard_budget}, "
        f"final stop at {absolute_budget}).\n"
        + (
            f"- Symptom the app has assigned for the next question: \"{target_symptom}\" "
            f"({symptom_counts.get(target_symptom, 0)} of "
            f"{_symptom_quota(target_symptom, state.worst_label, state.pace_mode, state.quota_extra)} planned questions "
            "asked). The app moves to the next symptom by itself once these are asked - do not "
            "direct the interviewer to switch symptoms."
            if quota_steering
            else f"- Current symptom \"{run_topic or 'none'}\": {run_count} follow-up question(s) in a row."
        )
        + pace_line
    )

    # The queue already limits each symptom exactly (3, 4 for the worst, plus red-flag
    # questions), so the cap only acts where no limit applies - e.g. after the patient
    # chose to keep going past their planned questions. It never contradicts an
    # assignment to ask about the same symptom.
    assigned_same_symptom = bool(quota_steering) and target_symptom == run_topic
    if run_topic and run_count >= PER_TOPIC_QUESTION_CAP and not assigned_same_symptom:
        cap_steering = (
            f"Hard pacing limit reached: you have already asked {run_count} questions in a "
            f"row about {run_topic}. Do NOT ask anything else about {run_topic} - you have "
            "enough for the doctor. This turn, briefly acknowledge the patient's answer and "
            "ask about the symptom the app assigns, or, if there is none, invite the patient "
            "to share anything else that is on their mind. Anything still missing about "
            f"{run_topic} will be listed for the doctor as unresolved."
        )

    extra_steering = "\n\n".join(
        s
        for s in (
            cap_steering,
            addon_steering,
            offer_followup,
            latest_steering,
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
        for m in state.messages
    ]
    rolling_summary = state.rolling_summary if ENABLE_ROLLING_SUMMARY else ""
    summary_tail_start = state.summary_tail_start if ENABLE_ROLLING_SUMMARY else 0
    # A wrap-up/stop offer is answered with buttons, so it need not be phrased as a
    # question; every other turn must leave the patient a question to answer.
    # A typed answer to the offer may be a plain "let's wrap up", which needs no question.
    question_expected = None if (agency_steering or typed_offer_reply) else True
    confirmed_symptoms = [
        label
        for label in state.selected_symptom_labels
        if label not in (CHECKLIST_NONE_LABEL, "Something else")
    ]
    selected_symptoms = [
        label for label in state.selected_symptom_labels if label != CHECKLIST_NONE_LABEL
    ]
    assigned_symptom = target_symptom if quota_steering else ""

    # (The web page shows "Nurse assistant is reviewing your response..." meanwhile.)
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
            question_expected,
            confirmed_symptoms,
            assigned_symptom,
            selected_symptoms,
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
            state.judge_directive = judge_future.result(
                timeout=JUDGE_GRACE_SECONDS
            )
        except Exception:
            # Not ready in time (or errored) - skip the judge this turn. The
            # code caps still enforce every hard limit.
            state.judge_directive = ""
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
            question_expected=question_expected,
            confirmed_symptoms=confirmed_symptoms,
            assigned_symptom=assigned_symptom,
            selected_symptoms=selected_symptoms,
        )

    # The model sometimes tries to end early with its own "anything else?" question.
    # Only honour that when the quotas really are filled - otherwise it would skip
    # symptoms the patient selected (this is what silently dropped "Breathing problems").
    # When topics remain, discard the premature close and ask the assigned topic instead.
    if _is_final_open_question(result.get("reply", "")) and not result.get("is_complete"):
        state.raw_responses.append(result.get("raw_response", ""))
        if quotas_met:
            _acknowledge_then_closing_review(
                state, client, prior_history, patient_context, model, system_prompt,
                symptom_count, previous_question, latest_steering,
            )
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
            confirmed_symptoms=confirmed_symptoms,
            assigned_symptom=assigned_symptom,
            selected_symptoms=selected_symptoms,
        )
        if forced.get("reply") and not _is_final_open_question(forced["reply"]):
            result = forced
        else:
            # Asked "anything else?" twice: never show it (only the closing screen ends a
            # check-in) - keep the acknowledgement and add the safe fallback question.
            result = forced if forced.get("reply") else result
            result["reply"] = (
                _remove_questions(result["reply"]) + " " + NO_QUESTION_FALLBACK_QUESTION
            ).strip()
            result["suggested_answers"] = list(NO_QUESTION_FALLBACK_ANSWERS)
            result["fallback_question"] = True
            result["is_complete"] = False

    # A typed answer to the wrap-up offer does exactly what the buttons do.
    if typed_offer_reply and result.get("offer_choice") == "wrap" and not result.get("red_flag"):
        # Same as tapping "Wrap up now": no more questions, straight to the closing screen.
        if "?" in result.get("reply", ""):
            result["reply"] = _remove_questions(result["reply"])
        result.update(suggested_answers=[], symptom="", topic="", is_complete=False, doctor_summary="")
        apply_nurse_result(state, result)
        state.closing_review = True
        state.show_suggestions = False
        return
    if typed_offer_reply and "?" not in result.get("reply", ""):
        # Kept going (or unclear), but the reply asks nothing: leave the patient a question.
        result["reply"] = (result["reply"].rstrip() + " " + NO_QUESTION_FALLBACK_QUESTION).strip()
        result["suggested_answers"] = list(NO_QUESTION_FALLBACK_ANSWERS)
        result["fallback_question"] = True

    # Guarantee the "we'll come to it" promise for a just-added symptom, rather than
    # relying on the model to remember to say it. The quota order already guarantees the
    # interviewer does not jump to it.
    # (Not in front of a red-flag reply: a cheerful "thanks for adding" would jar there,
    # and the page already confirmed the addition.)
    if addon_notice and not result.get("red_flag"):
        result["reply"] = _added_to_list_note(addon_notice, now=addon_is_next) + result["reply"]

    # Act on the model's reading of the patient's message (unanswered question, red flag,
    # new issue), then credit the question to the symptom it is about.
    _apply_patient_signals(state, result, previous_question, target_symptom)
    # A new issue is acknowledged with fixed wording (the model is told not to mention
    # it), so every patient hears the same thing. Not in front of a red-flag reply.
    queued, noted = result.get("queued_issue"), result.get("noted_issue")
    if (queued or noted) and not result.get("red_flag"):
        result["reply"] = (
            _added_to_list_note(queued)
            if queued
            else f"Thanks - I've noted {_sentence_name(noted)} for your care team. "
        ) + result["reply"]
    _credit_question(state, result, assigned_symptom)

    # A topic closing is the trigger to compress: fold everything up to now into the
    # running summary so later turns carry "summary + current topic" only.
    topics_closed_before = len(state.completed_topics)
    apply_nurse_result(state, result)
    if agency_steering:
        # The wrap-up offer: its reply is a choice, never repeated as an unanswered question.
        state.messages[-1]["is_offer"] = True
    new_detail = state.messages[state.summary_tail_start:]
    if (
        ENABLE_ROLLING_SUMMARY
        and len(state.completed_topics) > topics_closed_before
        and len(new_detail) >= SUMMARY_MIN_NEW_MESSAGES
    ):
        state.rolling_summary = update_rolling_summary(
            client, state.rolling_summary, new_detail, model
        )
        # Keep the just-asked question (the last message) in the verbatim tail, so the
        # next turn's patient answer still has a visible question to belong to.
        state.summary_tail_start = len(state.messages) - 1


# =========================
# Web application (replaces the Streamlit UI)
# =========================
#
# Streamlit re-ran main() on every click. Here, each click is one "action" that makes
# the same state changes main() made for that click, and static/index.html renders
# build_view(state). The LLM calls go to OpenAI's API.

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
SESSION_COOKIE = "checkin_session"
# In-memory sessions are dropped after this long without activity.
SESSION_IDLE_SECONDS = 12 * 60 * 60

# OpenAI API key from .streamlit/secrets.toml (as in the original GPT version), or the
# OPENAI_API_KEY environment variable.
# The library's default timeout is 10 minutes, so a stuck request left the patient
# waiting with no reply (Sep 17 "it didn't respond" reports). Fail fast instead: the
# action is rolled back and the page offers "Try again".
MODEL_TIMEOUT_SECONDS = 90
_client = OpenAI(
    api_key=_secret("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    timeout=MODEL_TIMEOUT_SECONDS,
    max_retries=1,
)

_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()


def _session_for(request: Request) -> tuple[str, Dict[str, Any]]:
    now = time.time()
    session_key = request.cookies.get(SESSION_COOKIE, "")
    with _sessions_lock:
        for key in [k for k, s in _sessions.items() if now - s["seen"] > SESSION_IDLE_SECONDS]:
            del _sessions[key]
        if session_key not in _sessions:
            session_key = secrets.token_urlsafe(32)
            _sessions[session_key] = {"state": new_session_state(), "lock": threading.Lock()}
        session = _sessions[session_key]
        session["seen"] = now
    return session_key, session


def _generate_turn(state: SessionState) -> None:
    patient_context = build_patient_context(
        state.saved_patient_name, state.saved_doctor_name, state.saved_therapy_week
    )
    started = time.monotonic()
    generate_and_apply_turn(
        state,
        _client,
        state.saved_prior_history,
        patient_context,
        DEFAULT_MODEL,
        state.saved_system_prompt,
    )
    # Per-turn latency, logged and kept on the reply, to track the slow-reply reports.
    elapsed = round(time.monotonic() - started, 2)
    if state.messages and state.messages[-1].get("role") == "assistant":
        state.messages[-1]["latency_seconds"] = elapsed
    logger.info("Session %s: turn generated in %.2fs", state.session_id, elapsed)


def _in_chat(state: SessionState) -> bool:
    return (
        state.check_in_started
        and state.disclaimer_acknowledged
        and state.started
        and not state.is_complete
    )


def _can_compose(state: SessionState) -> bool:
    """The chat input (and everything above it in main()) is showing."""
    return _in_chat(state) and not state.worst_pick_pending and not state.closing_review


def _current_suggestions(state: SessionState) -> List[str]:
    if not state.messages or state.messages[-1].get("role") != "assistant":
        return []
    suggestions = state.messages[-1].get("suggested_answers", [])
    return list(suggestions)


def action_start(state: SessionState, payload: Dict[str, Any]) -> str:
    patient_name = str(payload.get("patient_name") or "")
    doctor_name = str(payload.get("doctor_name") or "")
    therapy_week = str(payload.get("therapy_week") or "")
    prior_history = str(payload.get("prior_history") or "")
    system_prompt = payload.get("system_prompt")
    if not isinstance(system_prompt, str):
        system_prompt = SYSTEM_PROMPT
    if not patient_name.strip():
        return "Please enter the patient's name before starting the check-in."
    reset_chat(state)
    state.saved_prior_history = prior_history
    # If the prior-history field was left blank, auto-load last visit's urgent
    # summary from the Google Sheet by patient name (added feature).
    if not prior_history.strip():
        state.saved_prior_history = _load_prior_summary(patient_name)
    first_line = state.saved_prior_history.strip().splitlines()[0] if state.saved_prior_history.strip() else ""
    state.prior_history_status = (
        first_line.rstrip(":")
        if prior_history.strip() == "" and first_line
        else ("Prior history entered by hand" if prior_history.strip() else "No previous check-in found for this name")
    )
    state.saved_system_prompt = system_prompt
    state.saved_patient_name = patient_name
    state.saved_doctor_name = doctor_name
    state.saved_therapy_week = therapy_week
    state.check_in_started = True
    return ""


def action_acknowledge(state: SessionState, payload: Dict[str, Any]) -> str:
    if state.check_in_started and not state.is_complete and not state.disclaimer_acknowledged:
        state.disclaimer_acknowledged = True
        state.disclaimer_acknowledged_at = datetime.now(timezone.utc).isoformat()
    return ""


def action_checklist_continue(state: SessionState, payload: Dict[str, Any]) -> str:
    """Port of the checkbox-first opening (render_symptom_checklist + main)."""
    if not (
        state.check_in_started
        and state.disclaimer_acknowledged
        and not state.started
        and not state.is_complete
    ):
        return ""
    requested = set(payload.get("labels") or [])
    checked_labels = [label for label, _topic in CHECKLIST_ITEMS if label in requested]
    none_checked = bool(payload.get("none"))
    if not (checked_labels or none_checked):
        return ""

    returning = bool(state.saved_prior_history.strip())
    question = CHECKLIST_QUESTION_RETURNING if returning else CHECKLIST_QUESTION_FIRST
    if checked_labels:
        other_description = ""
        if "Something else" in checked_labels:
            other_description = str(payload.get("other_description") or "")[:200]
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
        # Symptoms are asked in this order (each checkbox is its own topic).
        labels = _interview_order(checked_labels)
    else:
        user_message = 'I checked: "None of these - I\'m doing okay today."'
        labels, topics = [CHECKLIST_NONE_LABEL], []

    state.selected_symptom_labels = labels
    state.selected_topics = topics
    add_assistant_message(state, question)
    add_user_message(state, user_message, response_mode="checklist")
    state.started = True
    if len([l for l in labels if l != CHECKLIST_NONE_LABEL]) > 1:
        # More than one symptom: the PATIENT designates the worst before any
        # questions, so code can apply the 4/2 quotas without ever inferring it.
        state.worst_pick_pending = True
        return ""
    if not topics:
        # "None of these": a fixed reply and straight to the closing screen, the same
        # screen every check-in ends on. Anything added there is queued and asked like
        # any other symptom (up to 3 questions). Used to depend on how the model worded
        # its own "anything else?" (Sep 28 fix).
        add_assistant_message(state, NONE_SELECTED_REPLY)
        state.closing_review = True
        state.show_suggestions = False
        return ""
    state.worst_topic = topics[0]
    first = [l for l in labels if l != CHECKLIST_NONE_LABEL]
    state.worst_label = first[0] if first else ""
    _generate_turn(state)
    return ""


def action_worst_pick(state: SessionState, payload: Dict[str, Any]) -> str:
    """Ask the PATIENT which selected symptom is worst. Their tap sets the 4-question
    quota; the model never infers it. Shown once, before any clinical question."""
    label = payload.get("label")
    if not (_in_chat(state) and state.worst_pick_pending):
        return ""
    if label == CHECKLIST_NONE_LABEL or label not in state.selected_symptom_labels:
        return ""
    label_to_topic = dict(CHECKLIST_ITEMS)
    state.worst_topic = label_to_topic.get(label, "Other")
    state.worst_label = label
    state.worst_pick_pending = False
    # Symptoms are asked in selected_symptom_labels order, so move the worst symptom to
    # the front - otherwise it waits behind symptoms that come earlier in the order.
    state.selected_symptom_labels.remove(label)
    state.selected_symptom_labels.insert(0, label)
    message = f"The {label.lower()} is bothering me the most."
    add_user_message(state, message, response_mode="worst_pick")
    _generate_turn(state)
    return ""


def action_closing_add(state: SessionState, payload: Dict[str, Any]) -> str:
    """Closing review: "Add these & keep going"."""
    if not (_in_chat(state) and state.closing_review and not state.worst_pick_pending):
        return ""
    chosen = set(state.selected_symptom_labels)
    requested = set(payload.get("labels") or [])
    review_items = [item for item in CHECKLIST_ITEMS if item[0] != "Something else"]
    new_picks = [
        (label, topic)
        for label, topic in review_items
        if label in requested and label not in chosen
    ]
    other_text = str(payload.get("other_text") or "")
    has_other = bool(other_text.strip())
    if not (new_picks or has_other):
        return ""

    for label, topic in new_picks:
        if label not in state.selected_symptom_labels:
            state.selected_symptom_labels.append(label)
        if topic not in state.selected_topics:
            state.selected_topics.append(topic)
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
    add_user_message(state, message, response_mode="closing_addon")
    # Kept verbatim so the doctor dashboard can show it word for word.
    state.messages[-1]["free_text"] = other_text.strip()
    state.closing_review = False
    state.show_suggestions = SHOW_SUGGESTIONS_BY_DEFAULT
    _generate_turn(state)
    return ""


def action_closing_finish(state: SessionState, payload: Dict[str, Any]) -> str:
    """Closing review: "Finish check-in"."""
    if not (_in_chat(state) and state.closing_review and not state.worst_pick_pending):
        return ""
    other_text = str(payload.get("other_text") or "")
    # Anything typed but not discussed must still reach the doctor.
    if other_text.strip():
        add_user_message(state, other_text.strip(), response_mode="closing_other")
        state.messages[-1]["free_text"] = other_text.strip()
    add_user_message(
        state,
        "Patient finished the check-in from the closing review.",
        response_mode="finish_button",
    )
    add_assistant_message(state, FINAL_CLOSING_REPLY)
    # Every queued symptom covered, or the patient wrapped up before the end?
    _target, _counts, all_covered = _quota_state(
        state.messages, state.selected_symptom_labels, state.worst_label, state.pace_mode,
        state.quota_extra,
    )
    state.completion_reason = "closing_review_finish" if all_covered else "wrapped_up_early"
    state.completed_at = datetime.now().astimezone().isoformat()
    state.current_topic = ""
    state.is_complete = True
    state.doctor_summary = ""
    state.closing_review = False
    return ""


def action_offer(state: SessionState, payload: Dict[str, Any]) -> str:
    """The patient taps an answer to a pacing offer. Explicit buttons let the app know
    their choice for certain, so it can inject one unambiguous instruction instead of
    asking the model to work out which branch applies. Typing a reply still works."""
    if not (_can_compose(state) and state.pending_offer):
        return ""
    offering_wrap = state.pending_offer == "wrap"
    choice = payload.get("choice")
    if choice == "wrap":
        add_user_message(
            state,
            "Let's wrap up now." if offering_wrap else "Let's finish now.",
            response_mode="offer_choice",
        )
        # Wrapping up skips the remaining topics entirely - no further
        # questions. Deterministic: straight to the closing review, where
        # the patient still sees what was not covered.
        state.pending_offer = None
        state.offer_choice = None
        state.closing_review = True
    elif choice == "continue":
        add_user_message(state, "Let's keep going.", response_mode="offer_choice")
        state.offer_choice = "continue"
        _generate_turn(state)
    return ""


def action_toggle_suggestions(state: SessionState, payload: Dict[str, Any]) -> str:
    # Suggestions start shown or hidden (SHOW_SUGGESTIONS_BY_DEFAULT); the button toggles them.
    if _can_compose(state) and _current_suggestions(state):
        state.show_suggestions = not state.show_suggestions
    return ""


def action_send(state: SessionState, payload: Dict[str, Any]) -> str:
    submitted_answer = str(payload.get("text") or "").strip()
    send_id = str(payload.get("id") or "")
    if send_id and send_id == state.last_send_id:
        # A retry of a send that was already answered (e.g. the page timed out while the
        # server was still working): return the current view instead of answering twice.
        return ""
    if not submitted_answer:
        return ""
    if not _can_compose(state):
        # Used to be dropped silently, which looked like the chatbot ignoring the patient.
        return (
            "Your message was not sent because the check-in had moved on. Please look at "
            "the screen and try again."
        )
    state.last_send_id = send_id
    # "selected" only when the patient sent a suggestion verbatim; a suggestion
    # they edited or added to counts as "typed" (the transcript still stores the
    # offered suggestions, so chip-assisted edits remain recoverable).
    last_message = state.messages[-1] if state.messages else {}
    offered_suggestions = (
        last_message.get("suggested_answers", [])
        if last_message.get("role") == "assistant"
        else []
    )
    response_mode = "selected" if submitted_answer in offered_suggestions else "typed"
    add_user_message(state, submitted_answer, response_mode=response_mode)
    state.show_suggestions = SHOW_SUGGESTIONS_BY_DEFAULT
    _generate_turn(state)
    return ""


def action_add_symptom(state: SessionState, payload: Dict[str, Any]) -> str:
    """Live checklist: add a symptom remembered mid-conversation."""
    label = payload.get("label")
    topic = dict(CHECKLIST_ITEMS).get(label)
    if not (state.started and not state.is_complete and state.selected_topics):
        return ""
    if topic is None or label == "Something else" or label in state.selected_symptom_labels:
        return ""
    state.pending_addon = (label, topic)
    return ""


def _process_pending_addon(state: SessionState) -> None:
    """Mid-conversation "add a symptom": the patient checked a new topic in the live
    side panel. Add it to the checklist and voice it as if the patient raised it, so
    the interviewer acknowledges and covers it - "check it and we'll cover it." As in
    the Streamlit flow, this waits while the worst-symptom picker or closing review
    is showing."""
    if not (state.get("pending_addon") and _can_compose(state)):
        return
    addon_label, addon_topic = state.pending_addon
    state.pending_addon = None
    if addon_topic and addon_topic not in state.selected_topics:
        state.selected_topics.append(addon_topic)
    # Count it as another selected symptom so the adaptive budget grows (+2), the
    # same as if it had been checked on the opening checklist.
    if addon_label not in state.selected_symptom_labels:
        state.selected_symptom_labels.append(addon_label)
    if state.messages and state.messages[-1].get("role") == "assistant":
        # A question is still waiting for the patient's answer (McKenna's Sep 17 bug):
        # generating a turn now skipped it and still counted it toward its symptom. Just
        # queue the symptom - no model call - and let the patient answer. The next reply
        # opens with "I'll ask you about that shortly" (see addon_notice).
        state.messages[-1].setdefault("symptoms_added_while_open", []).append(addon_label)
        state.addon_notice = (
            f"{state.addon_notice} and {addon_label}" if state.addon_notice else addon_label
        )
        state.addon_toast = addon_label
        return
    addon_message = f"I just remembered - I'd also like to talk about {addon_label.lower()}."
    add_user_message(state, addon_message, response_mode="checklist_addon")
    # Acknowledge now, cover later - do not abandon the current topic.
    state.addon_notice = addon_label
    _generate_turn(state)


def action_pace(state: SessionState, payload: Dict[str, Any]) -> str:
    mode = payload.get("mode")
    if state.started and not state.is_complete and mode in ("slower", "faster"):
        state.pace_mode = "normal" if state.pace_mode == mode else mode
    return ""


def action_wrap_up(state: SessionState, payload: Dict[str, Any]) -> str:
    if state.started and not state.is_complete:
        # Show the checklist review screen (opening checklist, pre-ticked)
        # instead of an instant close. Suppress the automatic threshold offers.
        state.closing_review = True
        state.wrap_choice_offered = True
        state.stop_choice_offered = True
        state.show_suggestions = False
    return ""


def _patient_summary_pending(state: SessionState) -> bool:
    """The patient has not yet been through the patient summary page."""
    return SHOW_PATIENT_SUMMARY and state.is_complete and not state.patient_summary_viewed


def action_generate_patient_summary(state: SessionState, payload: Dict[str, Any]) -> str:
    """Generate the patient-facing summary once, right after the chat completes."""
    if not (_patient_summary_pending(state) and not state.patient_summary_generated):
        return ""
    state.patient_summary = get_patient_summary(
        _client, state.messages, DEFAULT_MODEL, state.selected_symptom_labels
    )
    state.patient_summary_generated = True
    return ""


def action_patient_summary_continue(state: SessionState, payload: Dict[str, Any]) -> str:
    """The patient has read their summary: keep any correction they typed (so the doctor
    summary and the dashboard include it, word for word), then build the doctor report."""
    if not (_patient_summary_pending(state) and state.patient_summary_generated):
        return ""
    correction = str(payload.get("correction") or "").strip()[:1000]
    if correction:
        add_user_message(state, correction, response_mode="summary_correction")
        state.messages[-1]["free_text"] = correction
    state.patient_summary_viewed = True
    return ""


def action_generate_summary(state: SessionState, payload: Dict[str, Any]) -> str:
    """Generate the doctor summary exactly once after the chat completes."""
    if not (state.is_complete and not state.summary_generated):
        return ""
    if _patient_summary_pending(state):
        return ""
    prior_history = state.saved_prior_history
    patient_context = build_patient_context(
        state.saved_patient_name, state.saved_doctor_name, state.saved_therapy_week
    )
    structured: Dict[str, Any] = {}
    try:
        structured = get_doctor_summary(
            client=_client,
            chat_history=state.messages,
            prior_history=prior_history,
            patient_context=patient_context,
            model=DEFAULT_MODEL,
        )
        state.doctor_summary_structured = structured
    except Exception as exc:  # log the error but don't crash the app
        logger.warning("Could not generate doctor summary: %s", exc)
        state.doctor_summary_structured = {}
    # Saving is intentionally OUTSIDE the generation try-block: a storage
    # failure must never blank out an already-generated summary.
    try:
        if structured and (not state.sheet_saved or not state.local_csv_saved):
            sheet_payload = build_sheet_payload(
                patient_name=state.saved_patient_name,
                doctor_name=state.saved_doctor_name,
                therapy_week=state.saved_therapy_week,
                prior_history=prior_history,
                messages=state.messages,
                doctor_summary=state.doctor_summary,
                structured_summary=structured,
                system_prompt=state.saved_system_prompt,
                model=DEFAULT_MODEL,
                session_id=state.session_id,
                session_started_at=state.session_started_at,
                session_errors=state.session_errors,
                completion_reason=state.completion_reason or "natural_completion",
                symptoms_selected=state.selected_symptom_labels,
                disclaimer_acknowledged_at=state.disclaimer_acknowledged_at,
            ) | {"patient_summary": state.get("patient_summary", "")}
            saved_name = state.saved_patient_name.strip() or "Unknown patient"
            if not state.sheet_saved:
                state.sheet_saved = save_to_sheet(
                    name=saved_name,
                    all_data=sheet_payload,
                    report=state.doctor_summary,
                    system_prompt=state.saved_system_prompt,
                )
            if not state.local_csv_saved:
                state.local_csv_saved = save_to_local_csv(
                    name=saved_name,
                    all_data=sheet_payload,
                    report=state.doctor_summary,
                    system_prompt=state.saved_system_prompt,
                )
    except Exception as exc:
        logger.warning("Could not save the check-in record: %s", exc)
    state.summary_generated = True
    return ""


def action_regenerate_summary(state: SessionState, payload: Dict[str, Any]) -> str:
    if state.is_complete and state.summary_generated:
        state.summary_generated = False
    return ""


def action_new_checkin(state: SessionState, payload: Dict[str, Any]) -> str:
    if state.is_complete:
        reset_chat(state)
    return ""


ACTIONS = {
    "start": action_start,
    "acknowledge": action_acknowledge,
    "checklist_continue": action_checklist_continue,
    "worst_pick": action_worst_pick,
    "closing_add": action_closing_add,
    "closing_finish": action_closing_finish,
    "offer": action_offer,
    "toggle_suggestions": action_toggle_suggestions,
    "send": action_send,
    "add_symptom": action_add_symptom,
    "pace": action_pace,
    "wrap_up": action_wrap_up,
    "generate_patient_summary": action_generate_patient_summary,
    "patient_summary_continue": action_patient_summary_continue,
    "generate_summary": action_generate_summary,
    "regenerate_summary": action_regenerate_summary,
    "new_checkin": action_new_checkin,
}


def build_view(state: SessionState) -> Dict[str, Any]:
    """What the page should show for this session (the routing part of main())."""
    view: Dict[str, Any] = {
        "status": (
            "Complete"
            if state.is_complete
            else ("In progress" if state.check_in_started else "Not started")
        ),
        "check_in_started": state.check_in_started,
        "is_complete": state.is_complete,
        "default_system_prompt": SYSTEM_PROMPT,
        "prior_history_status": state.get("prior_history_status", ""),
        "saved_prior_history": state.saved_prior_history if state.check_in_started else "",
        "patient_panel": None,
    }

    if state.started and not state.is_complete:
        chosen_labels = set(state.selected_symptom_labels)
        view["patient_panel"] = {
            "topic_boxes_html": render_topic_boxes(state) if state.selected_topics else "",
            "addon_labels": (
                [
                    label
                    for label, _topic in CHECKLIST_ITEMS
                    if label != "Something else" and label not in chosen_labels
                ]
                if state.selected_topics
                else None
            ),
            "pace_mode": state.pace_mode,
        }

    # ---- Routing: patient summary, then doctor summary page after submission,
    # otherwise patient chat ----
    if _patient_summary_pending(state):
        if not state.patient_summary_generated:
            view["screen"] = "preparing_patient"
            return view
        any_red_flag = any(
            m.get("role") == "assistant" and m.get("red_flag") for m in state.messages
        )
        self_harm = any(m.get("red_flag_kind") == "self_harm" for m in state.messages)
        view["screen"] = "patient_summary"
        view["patient_summary"] = {
            "text": state.patient_summary,
            "notice": (
                SELF_HARM_NOTICE if self_harm else (RED_FLAG_NOTICE if any_red_flag else "")
            ),
        }
        return view
    if state.is_complete:
        if state.summary_generated:
            view["screen"] = "dashboard"
            view["dashboard"] = build_doctor_summary_page(state)
        else:
            view["screen"] = "preparing"
        return view

    if not state.check_in_started:
        view["screen"] = "setup"
        return view

    # The disclosure gate: no checklist, no chat input, and no model call may
    # happen until the patient acknowledges it.
    if not state.disclaimer_acknowledged:
        display_name = state.saved_patient_name.strip()
        view["screen"] = "welcome"
        view["welcome"] = {
            "title": (
                WELCOME_TITLE.format(patient_name=display_name) if display_name else "Hi there 👋"
            ),
            "body": WELCOME_BODY,
            "disclaimer_html": _basic_md_to_html(DISCLAIMER_FULL, inline=True),
            "button": WELCOME_BUTTON_LABEL,
        }
        return view

    view["disclaimer_banner"] = DISCLAIMER_BANNER
    view["red_flag_notice"] = RED_FLAG_NOTICE
    view["self_harm_notice"] = SELF_HARM_NOTICE

    if not state.started:
        # Checkbox-first opening (June 5 clinical-team decision) - the only mode.
        returning = bool(state.saved_prior_history.strip())
        view["screen"] = "checklist"
        view["checklist"] = {
            "question": CHECKLIST_QUESTION_RETURNING if returning else CHECKLIST_QUESTION_FIRST,
            "labels": [label for label, _topic in CHECKLIST_ITEMS],
            "none_label": CHECKLIST_NONE_LABEL,
        }
        return view

    view["screen"] = "chat"
    view["messages"] = [
        {
            "role": message["role"],
            "html": _basic_md_to_html(message.get("content", "")),
            "red_flag": bool(message.get("red_flag")),
            "self_harm": message.get("red_flag_kind") == "self_harm",
        }
        for message in state.messages
        if message["role"] in ("assistant", "user")
    ]

    if state.worst_pick_pending:
        view["mode"] = "worst_pick"
        view["worst_labels"] = [
            label for label in state.selected_symptom_labels if label != CHECKLIST_NONE_LABEL
        ]
    elif state.closing_review:
        chosen = set(state.selected_symptom_labels)
        view["mode"] = "closing_review"
        checklist_labels = [label for label, _topic in CHECKLIST_ITEMS]
        view["review_items"] = [
            {"label": label, "already": label in chosen}
            for label in checklist_labels
            if label != "Something else"
        ] + [
            # Items the patient raised in their own words ("Knee pain", "Chest pain") are
            # shown as already discussed, the same as in the sidebar.
            {"label": label, "already": True}
            for label in state.selected_symptom_labels
            if label not in checklist_labels and label != CHECKLIST_NONE_LABEL
        ]
    else:
        view["mode"] = "compose"
        view["pending_offer"] = state.pending_offer
        view["suggestions"] = _current_suggestions(state)
        view["show_suggestions"] = state.show_suggestions
        # A symptom added while a question was waiting: confirmed on the page right away
        # (no model call), acknowledged in the next reply.
        view["addon_toast"] = state.get("addon_toast")
    return view


app = FastAPI(title="Nurse Assistant Check-In", docs_url=None, redoc_url=None, openapi_url=None)


def _with_cookie(response: Response, request: Request, session_key: str) -> Response:
    response.set_cookie(
        SESSION_COOKIE,
        session_key,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
    )
    return response


@app.get("/")
def index(request: Request) -> Response:
    session_key, _session = _session_for(request)
    return _with_cookie(
        FileResponse(os.path.join(STATIC_DIR, "index.html")), request, session_key
    )


@app.get("/api/view")
def api_view(request: Request) -> Response:
    session_key, session = _session_for(request)
    with session["lock"]:
        view = build_view(session["state"])
    return _with_cookie(JSONResponse({"view": view}), request, session_key)


@app.post("/api/action")
def api_action(request: Request, body: Dict[str, Any]) -> Response:
    session_key, session = _session_for(request)
    action = ACTIONS.get(str(body.get("action")))
    payload = body.get("payload") if isinstance(body.get("payload"), dict) else {}
    error = ""
    # One action at a time per session, like Streamlit's one rerun at a time.
    with session["lock"]:
        state = session["state"]
        if action is not None:
            # If the action fails part-way (typically a model call timing out), put the
            # session back exactly as it was - otherwise the patient's message stays in
            # the chat with no reply, and re-sending it leaves two copies.
            snapshot = deepcopy(state)
            try:
                error = action(state, payload)
                _process_pending_addon(state)
            except Exception:
                logger.exception("Action %s failed", body.get("action"))
                state = session["state"] = snapshot
                error = (
                    "Sorry - the assistant couldn't respond just now. Nothing was lost; "
                    "please try again."
                )
        view = build_view(state)
    return _with_cookie(JSONResponse({"view": view, "error": error}), request, session_key)


@app.get("/api/export")
def api_export(request: Request) -> Response:
    """Download reproducibility record (JSON)."""
    session_key, session = _session_for(request)
    with session["lock"]:
        state = session["state"]
        if not (state.is_complete and state.summary_generated and state.doctor_summary_structured):
            return JSONResponse({"error": "No completed check-in."}, status_code=404)
        data = json.dumps(build_export_payload(state), ensure_ascii=False, indent=2)
        file_name = f"check_in_{state.session_id}.json"
    return _with_cookie(
        Response(
            data,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
        ),
        request,
        session_key,
    )


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Nurse assistant check-in web app")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
