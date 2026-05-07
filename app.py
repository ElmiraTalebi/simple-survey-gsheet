import json
import os
import html
import re
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
- You should cover all the topics and do not skip any of them. If a patient says no to a topic, do not ask follow-ups for that topic.
- After all topics are covered, follow the strict Final Closing Sequence below. Never combine the "anything else" question and the completion into a single turn.
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
- While chatting with the patient and gathering information, you may also have access to the patient’s history from previous visits. You should appropriately bring up past issues and ask whether they have resolved, improved, worsened, or stayed the same. However, the conversation should remain natural and human-like.

Efficiency Rules:
Avoid asking:
- Questions already answered
- Irrelevant follow-ups

Prioritize:
- Symptoms impacting safety, such as weight loss, swallowing issues, bleeding
- Keeping the conversation concise but complete

Final Closing Sequence (STRICT - TWO TURNS, NO EXCEPTIONS):

Once all 8 clinical topics have been covered, end the conversation with this exact two-turn sequence. Never collapse it into one turn.

Turn 1 - The "Anything Else" Turn:
- The message must be a single open-ended question, e.g.:
  "Before we wrap up, is there anything else you'd like to share with me - anything I haven't asked about?"
- On this turn, is_complete MUST be false.
- Do NOT include closing language, "thank you," or any indication the check-in is ending.
- "topic" should be an empty string.

Turn 2 - The Closing Turn (only after the patient has responded to Turn 1):
- Briefly acknowledge what the patient said. If they raised a new concern, note it warmly; if they said no, simply acknowledge.
- Inform them the check-in is complete and the information will be shared with their doctor.
- Use a warm and reassuring tone.
- On this turn, is_complete = true.

Critical: is_complete must NEVER be true on the same turn where you ask "anything else." That question and the completion must always be separated by the patient's response.

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
  "doctor_summary": "Concise clinical summary for the doctor if complete, otherwise an empty string.",
  "topic": "one of: Pain, Nutrition, Swallowing, Oral Symptoms, GI Symptoms, Fatigue & Sleep, Activity & Independence, Mood & Support, or empty string"
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
- The check-in covers some or all of these topics: Pain, Nutrition, Swallowing, Oral Symptoms, GI Symptoms, Fatigue & Sleep, Activity & Independence, Mood & Support.
- The full chat transcript (assistant questions and patient answers) will be provided.
- Prior patient history from a previous visit may also be provided. It may be missing.
- The doctor has only a few minutes to review this summary before the visit, so clarity and brevity are critical.

Instructions:

1. Summarize only what the patient actually reported in the transcript. Do not invent, assume, or infer information that was not stated.

2. Present information the way clinicians are used to reading it: lead with red flags and key changes, use clinical shorthand where appropriate (severity, frequency, duration, location), and avoid conversational filler.

3. If prior patient history is provided, explicitly compare current findings to it (e.g., "improved since last visit," "new since last visit," "unchanged," "worsened"). If no prior history is provided, do not fabricate comparisons.

4. For each topic, produce two summaries and a status:
   - "Main issues": a 1-3 line top-line for fast review. Include only red flags, significant symptoms, notable changes, and safety concerns. If the patient explicitly denied symptoms for the topic, write "No issues reported." If the topic was never covered in the chat, use an empty string.
   - "more details": a fuller but still concise breakdown. Include any of the following that apply: severity, location, onset, timing (constant vs intermittent), frequency, medications and their effectiveness, side effects, aggravating and alleviating factors, and functional impact. Use short labeled lines or bullets, not long paragraphs.
   - "status": one of "worse", "better", or "" (empty string).
     * "worse" - the topic represents a NEW symptom OR a WORSENING symptom compared to prior history. If no prior history is available, use "worse" only when the patient describes the symptom as new or recent.
     * "better" - the topic shows IMPROVEMENT compared to prior history.
     * "" (empty) - everything else, including unchanged/stable symptoms, topics with no prior history to compare against, and topics that were not discussed.

5. Always elevate the following to "Main issues" when present: unintentional weight loss, dehydration, choking on liquids, bleeding (including blood when coughing), severe or worsening pain, inability to take medications, severe emotional distress or suicidal ideation, feeding tube malfunction, and inability to perform basic daily activities.

6. Use the "Other" fields to capture clinically relevant content the patient raised that does not map to the eight topics (e.g., new symptoms outside scope, social or caregiving issues affecting care, specific questions the patient wants to ask the doctor). Leave empty if nothing applies.

7. Tone: neutral, factual, clinical. Do not reassure the patient, editorialize, or offer recommendations or treatment plans - just report what was said.

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
]

SUMMARY_TOPICS = CHAT_TOPICS + ["Other"]


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
        response_format={"type": "json_object"},
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
        "topic": parsed.get("topic", "").strip(),
        "raw_response": raw_content.strip(),
    }


def get_doctor_summary(
    client: OpenAI,
    chat_history: List[Dict[str, str]],
    prior_history: str,
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
        temperature=0.2,
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

    if "current_topic" not in st.session_state:
        st.session_state.current_topic = ""

    if "completed_topics" not in st.session_state:
        st.session_state.completed_topics = []

    if "doctor_summary_structured" not in st.session_state:
        st.session_state.doctor_summary_structured = {}

    if "summary_generated" not in st.session_state:
        st.session_state.summary_generated = False


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.is_complete = False
    st.session_state.doctor_summary = ""
    st.session_state.started = False
    st.session_state.raw_responses = []
    st.session_state.current_topic = ""
    st.session_state.completed_topics = []
    st.session_state.doctor_summary_structured = {}
    st.session_state.summary_generated = False


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
    st.title("Doctor Summary")
    st.caption(
        "Pre-visit check-in summary for head and neck cancer patient. "
        "Cards are listed in clinical-topic order; topics that were not discussed are muted."
    )

    if not st.session_state.summary_generated:
        st.info(
            "The doctor summary will be available once the patient completes the check-in."
        )
        return

    summary = st.session_state.doctor_summary_structured
    if not summary:
        st.warning("No summary available. Try regenerating from the patient view.")
        return

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

        model = "gpt-4.1"

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
                    model=model,
                )
                st.session_state.doctor_summary_structured = structured
            except Exception as exc:  # surface the error but don't crash the app
                st.warning(f"Could not generate doctor summary: {exc}")
                st.session_state.doctor_summary_structured = {}
            st.session_state.summary_generated = True
        st.rerun()

    # ---- Patient view ----
    st.title("Nurse Assistant Check-In")
    st.caption("A conversational pre-visit check-in for head and neck cancer care.")

    if not st.session_state.started:
        opening_message = "How have you been doing compared to your last visit?"
        add_assistant_message(opening_message)
        st.session_state.started = True

    if st.session_state.is_complete:
        render_chat_history()
        render_completion_banner()
        if st.session_state.summary_generated:
            st.divider()
            render_doctor_summary_page()
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
