import json
from typing import Any

import streamlit as st
from openai import OpenAI

try:
    import speech_recognition as sr
except Exception:
    sr = None


st.set_page_config(page_title="ChatReport", page_icon="🩺", layout="centered")


KNOWLEDGE_BASE = {
    "pain": {
        "priority": "high",
        "main": "Do you have any pain today?",
        "followups": [
            "Where exactly is the pain?",
            "How severe is the pain on a scale of 0–10?",
            "Does it affect eating or swallowing?",
        ],
        "required_fields": ["location", "severity"],
    },
    "nutrition": {
        "priority": "high",
        "main": "Are you able to eat and drink enough?",
        "followups": [
            "What are you able to eat?",
            "Are you drinking enough fluids?",
            "What makes eating difficult?",
        ],
        "required_fields": ["intake", "barriers"],
    },
    "swallowing": {
        "priority": "high",
        "main": "Are you having trouble swallowing?",
        "followups": [
            "Is swallowing painful or difficult?",
            "Can you swallow liquids?",
            "Does swallowing affect eating?",
        ],
        "required_fields": ["swallowing_status"],
    },
    "breathing": {
        "priority": "high",
        "main": "Are you having any difficulty breathing?",
        "followups": [
            "Is the breathing difficulty constant or only with activity?",
            "Can you breathe comfortably at rest?",
        ],
        "required_fields": ["breathing_status"],
    },
    "oral_symptoms": {
        "priority": "medium",
        "main": "Do you have any mouth sores, dryness, or mucus problems?",
        "followups": [
            "Which mouth or throat symptoms are bothering you most?",
            "Do these symptoms affect eating or speaking?",
        ],
        "required_fields": [],
    },
    "gi_symptoms": {
        "priority": "medium",
        "main": "Have you had nausea, vomiting, or any blood when coughing?",
        "followups": [
            "How often are these symptoms happening?",
            "Are they preventing you from eating or drinking?",
        ],
        "required_fields": [],
    },
    "fatigue": {
        "priority": "medium",
        "main": "Are you feeling more tired or weak than usual?",
        "followups": [
            "How much is fatigue affecting your daily activities?",
            "Is this fatigue mild, moderate, or severe?",
        ],
        "required_fields": [],
    },
    "mood": {
        "priority": "medium",
        "main": "How are you feeling emotionally?",
        "followups": [
            "Do you feel more anxious or low than usual?",
            "Do you have support at home?",
        ],
        "required_fields": [],
    },
    "other": {
        "priority": "low",
        "main": "Are there any other symptoms you want your doctor to know about?",
        "followups": [
            "What other symptoms are most important today?",
        ],
        "required_fields": [],
    },
}


TOPIC_ORDER = sorted(
    KNOWLEDGE_BASE.keys(),
    key=lambda topic: {"high": 0, "medium": 1, "low": 2}.get(KNOWLEDGE_BASE[topic]["priority"], 3),
)


ORCHESTRATOR_PROMPT = """
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


EXTRACTION_PROMPT = """
🟢 1. SYMPTOM EXTRACTION AGENT (STRONG)
You are a clinical information extraction agent.

TASK:
Convert patient natural language into structured clinical data.

CONSTRAINTS:
- Extract ONLY what is explicitly stated
- Do NOT infer medical facts not mentioned
- Use null if unknown
- Normalize terms (e.g., "hurts a lot" → severity: "high")
- Map symptoms to categories in knowledge_base

CATEGORIES:
pain, swallowing, nutrition, oral_symptoms, gi_symptoms, fatigue, mood, breathing, other

OUTPUT FORMAT (STRICT JSON):
{
  "pain": {
    "present": true,
    "location": "string or null",
    "severity": "low|medium|high|null",
    "timing": "constant|intermittent|null"
  },
  "swallowing": "normal|difficulty|painful|null",
  "nutrition": "normal|reduced|liquid_only|tube|null",
  "oral_symptoms": "string or null",
  "gi_symptoms": "string or null",
  "fatigue": "none|mild|moderate|severe|null",
  "mood": "normal|anxious|depressed|null",
  "breathing": "normal|difficulty|null",
  "other": "string or null"
}
"""


IMPORTANCE_PROMPT = """
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
  "needs_followup": true,
  "missing_fields": ["field1", "field2"]
}
"""


FOLLOWUP_PROMPT = """
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


EXPERIENCE_PROMPT = """
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
- repeated "I don't know"
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
  "should_limit_questions": true
}
"""


REPORT_PROMPT = """
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


SAFETY_PROMPT = """
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
  "alert": true,
  "reason": "string",
  "recommended_action": "continue|flag_for_doctor|urgent_attention"
}
"""


SUMMARY_PROMPT = """
Summarize the conversation so far into concise clinical notes.
Focus on symptoms, severity, function, red flags, and major changes.
Return short text only.
"""


def get_openai_client() -> OpenAI:
    api_key = st.secrets["OPENAI_API_KEY"]
    return OpenAI(api_key=api_key)


def extract_json_from_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0) -> dict[str, Any]:
    client = get_openai_client()
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return extract_json_from_text(response.output_text)


def call_text_llm(system_prompt: str, user_prompt: str, temperature: float = 0) -> str:
    client = get_openai_client()
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return (response.output_text or "").strip()


def extraction_agent(user_input: str) -> dict[str, Any]:
    return call_llm(EXTRACTION_PROMPT, f'Patient message: "{user_input}"')


def importance_agent(extracted: dict[str, Any], current_topic: str) -> dict[str, Any]:
    payload = {
        "current_topic": current_topic,
        "extracted_symptoms": extracted,
        "knowledge_base": KNOWLEDGE_BASE,
    }
    return call_llm(IMPORTANCE_PROMPT, json.dumps(payload))


def experience_agent(history: dict[str, Any], last_user_message: str) -> dict[str, Any]:
    payload = {
        "conversation_history": history,
        "last_user_message": last_user_message,
    }
    return call_llm(EXPERIENCE_PROMPT, json.dumps(payload))


def followup_agent(topic: str, missing_fields: list[str], history: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "current_topic": topic,
        "missing_fields": missing_fields,
        "conversation_history": history,
        "knowledge_base": KNOWLEDGE_BASE,
    }
    return call_llm(FOLLOWUP_PROMPT, json.dumps(payload))


def safety_agent(extracted: dict[str, Any]) -> dict[str, Any]:
    return call_llm(SAFETY_PROMPT, json.dumps({"extracted_symptoms": extracted}))


def report_agent(collected_data: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "collected_data": collected_data,
        "knowledge_base": KNOWLEDGE_BASE,
    }
    return call_llm(REPORT_PROMPT, json.dumps(payload))


def summarize_memory(history: list[dict[str, str]]) -> str:
    return call_text_llm(SUMMARY_PROMPT, json.dumps(history))


def get_next_topic(state: dict[str, Any]) -> str | None:
    current = state["current_topic"]
    if current is None:
        return TOPIC_ORDER[0]
    idx = TOPIC_ORDER.index(current)
    return TOPIC_ORDER[idx + 1] if idx + 1 < len(TOPIC_ORDER) else None


def category_to_topic_updates(extracted: dict[str, Any]) -> dict[str, dict[str, Any]]:
    updates: dict[str, dict[str, Any]] = {}

    pain = extracted.get("pain", {})
    if isinstance(pain, dict):
        pain_update = {k: v for k, v in pain.items() if v is not None}
        if pain_update:
            updates["pain"] = pain_update

    swallowing = extracted.get("swallowing")
    if swallowing is not None:
        updates["swallowing"] = {"swallowing_status": swallowing}

    nutrition = extracted.get("nutrition")
    if nutrition is not None:
        updates["nutrition"] = {"intake": nutrition}

    oral = extracted.get("oral_symptoms")
    if oral:
        updates["oral_symptoms"] = {"symptoms": oral}

    gi = extracted.get("gi_symptoms")
    if gi:
        updates["gi_symptoms"] = {"symptoms": gi}

    fatigue = extracted.get("fatigue")
    if fatigue is not None:
        updates["fatigue"] = {"fatigue_level": fatigue}

    mood = extracted.get("mood")
    if mood is not None:
        updates["mood"] = {"mood_status": mood}

    breathing = extracted.get("breathing")
    if breathing is not None:
        updates["breathing"] = {"breathing_status": breathing}

    other = extracted.get("other")
    if other:
        updates["other"] = {"details": other}

    return updates


def update_state(state: dict[str, Any], extracted: dict[str, Any], user_input: str) -> None:
    topic = state["current_topic"]
    if topic not in state["collected_data"]:
        state["collected_data"][topic] = {}

    topic_updates = category_to_topic_updates(extracted)
    current_update = topic_updates.get(topic, {})
    state["collected_data"][topic].update(current_update)
    state["collected_data"][topic]["last_patient_message"] = user_input

    if topic == "nutrition" and "pain" in topic_updates:
        pain_data = topic_updates["pain"]
        if pain_data.get("present") is True and "barriers" not in state["collected_data"][topic]:
            state["collected_data"][topic]["barriers"] = "pain"


def high_priority_topics_completed(state: dict[str, Any]) -> bool:
    high_priority_topics = [topic for topic, cfg in KNOWLEDGE_BASE.items() if cfg["priority"] == "high"]
    return all(topic in state["collected_data"] for topic in high_priority_topics)


def orchestrator(
    state: dict[str, Any],
    extracted: dict[str, Any],
    importance: dict[str, Any],
    followup: dict[str, Any],
    experience: dict[str, Any],
    safety: dict[str, Any],
) -> dict[str, Any]:
    current_topic = state["current_topic"]
    collected = state["collected_data"]
    required = KNOWLEDGE_BASE[current_topic].get("required_fields", [])
    missing = [field for field in required if field not in collected.get(current_topic, {})]

    if safety.get("alert"):
        return {
            "action": "follow_up",
            "question": "Your symptoms may need urgent attention. Can you confirm if you're unable to eat or breathe comfortably?",
            "next_topic": current_topic,
            "reason": safety.get("reason", "safety alert"),
        }

    if experience.get("should_limit_questions") and importance.get("importance") != "high":
        next_topic = get_next_topic(state)
        if next_topic is None:
            return {"action": "finish", "question": None, "next_topic": None, "reason": "fatigue and no topics left"}
        return {
            "action": "next_topic",
            "next_topic": next_topic,
            "question": KNOWLEDGE_BASE[next_topic]["main"],
            "reason": "limiting questions due to patient burden",
        }

    if importance.get("needs_followup") and missing:
        return {
            "action": "follow_up",
            "question": followup.get("follow_up_question"),
            "next_topic": current_topic,
            "reason": "required clinical detail still missing",
        }

    if not missing:
        next_topic = get_next_topic(state)
        if next_topic is None or high_priority_topics_completed(state):
            if next_topic is None:
                return {"action": "finish", "question": None, "next_topic": None, "reason": "all topics completed"}
        if next_topic is None:
            return {"action": "finish", "question": None, "next_topic": None, "reason": "no further topics"}
        return {
            "action": "next_topic",
            "question": KNOWLEDGE_BASE[next_topic]["main"],
            "next_topic": next_topic,
            "reason": "moving to next topic",
        }

    next_topic = get_next_topic(state)
    if next_topic is None:
        return {"action": "finish", "question": None, "next_topic": None, "reason": "interview complete"}
    return {
        "action": "next_topic",
        "question": KNOWLEDGE_BASE[next_topic]["main"],
        "next_topic": next_topic,
        "reason": "moving forward",
    }


def format_report(report: dict[str, Any]) -> str:
    lines = ["## Clinical Report", ""]
    for key in ["pain", "nutrition", "swallowing", "fatigue", "other"]:
        value = report.get(key)
        if value:
            lines.append(f"### {key.replace('_', ' ').title()}")
            lines.append(f"- {value}")
            lines.append("")
    if report.get("overall_priority"):
        lines.append("### Overall Priority")
        lines.append(f"- {report['overall_priority']}")
    return "\n".join(lines).strip()


def build_agent_history(state: dict[str, Any]) -> list[dict[str, Any]]:
    return {
        "summary": state.get("summary", ""),
        "short_memory": state.get("short_memory", []),
        "history": state.get("history", []),
    }


def update_memory(state: dict[str, Any], user_input: str, assistant_output: Any) -> None:
    assistant_text = assistant_output if isinstance(assistant_output, str) else json.dumps(assistant_output, ensure_ascii=False)
    turn = {"user": user_input, "assistant": assistant_text}
    state["short_memory"].append(turn)
    state["short_memory"] = state["short_memory"][-6:]
    state["history"].append(turn)

    if len(state["history"]) % 5 == 0:
        try:
            state["summary"] = summarize_memory(state["history"])
        except Exception:
            pass


def record_and_transcribe() -> str | None:
    if sr is None:
        st.session_state.voice_status = "Voice input is not available because SpeechRecognition is not installed."
        return None
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.session_state.voice_status = "Listening..."
            audio = recognizer.listen(source, timeout=5)
        text = recognizer.recognize_google(audio)
        st.session_state.voice_status = f"You said: {text}"
        return text
    except Exception as exc:
        st.session_state.voice_status = f"Voice input unavailable: {exc}"
        return None


def render_progress_bar(done: int, total: int) -> None:
    ratio = 0 if total == 0 else max(0.0, min(1.0, done / total))
    st.markdown(
        f"""
        <div style="background:#e5e7eb;border-radius:999px;height:10px;overflow:hidden;margin:8px 0 12px 0;">
            <div style="width:{ratio * 100:.0f}%;background:#111827;height:10px;border-radius:999px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_pipeline(state: dict[str, Any], user_input: str) -> tuple[dict[str, Any], Any, bool]:
    history_context = build_agent_history(state)
    extracted = extraction_agent(user_input)
    importance = importance_agent(extracted, state["current_topic"])
    experience = experience_agent(history_context, user_input)
    safety = safety_agent(extracted)

    update_state(state, extracted, user_input)

    topic = state["current_topic"]
    required = KNOWLEDGE_BASE.get(topic, {}).get("required_fields", [])
    missing = [field for field in required if field not in state["collected_data"].get(topic, {})]
    followup = followup_agent(topic, missing, history_context)

    decision = orchestrator(state, extracted, importance, followup, experience, safety)
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
                "decision": decision,
            },
            ensure_ascii=True,
        ),
    )

    if decision["action"] == "follow_up":
        return state, decision["question"], False

    if decision["action"] == "next_topic":
        state["current_topic"] = decision["next_topic"]
        return state, decision["question"], False

    report = report_agent(state["collected_data"])
    return state, report, True


def initialize_chat() -> None:
    if "state" not in st.session_state:
        st.session_state.state = {
            "current_topic": TOPIC_ORDER[0],
            "collected_data": {},
            "history": [],
            "short_memory": [],
            "summary": "",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Hello, I'm ChatReport. I'll ask focused symptom questions to create a concise report for your doctor.",
                },
                {
                    "role": "assistant",
                    "content": KNOWLEDGE_BASE[TOPIC_ORDER[0]]["main"],
                },
            ],
        }
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = ""
    if "voice_status" not in st.session_state:
        st.session_state.voice_status = ""


def main() -> None:
    initialize_chat()
    st.markdown(
        """
        <style>
        .chat-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 14px 16px;
            margin-bottom: 10px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        .chat-user {
            background: #111827;
            color: #ffffff;
        }
        .chat-assistant {
            background: #f9fafb;
            color: #111827;
        }
        .side-card {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🩺 ChatReport")
    st.markdown("Doctor-focused symptom reporting with memory, safety checks, and structured follow-up.")
    st.markdown(f"**Current topic:** {st.session_state.state['current_topic']}")

    col_chat, col_side = st.columns([2, 1])

    with col_chat:
        for message in st.session_state.state["messages"]:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-card chat-user"><strong>You</strong><br>{message["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-card chat-assistant"><strong>ChatReport</strong><br>{message["content"]}</div>',
                    unsafe_allow_html=True,
                )

    with col_side:
        st.markdown('<div class="side-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Progress")
        topics_done = len(st.session_state.state["collected_data"])
        total_topics = len(KNOWLEDGE_BASE)
        render_progress_bar(topics_done, total_topics)
        st.markdown(f"**Topics completed:** {topics_done}/{total_topics}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="side-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Extracted Info")
        st.json(st.session_state.state["collected_data"])
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.state.get("summary"):
            st.markdown('<div class="side-card">', unsafe_allow_html=True)
            st.markdown("### 📝 Summary")
            st.write(st.session_state.state["summary"])
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get("voice_status"):
            st.markdown('<div class="side-card">', unsafe_allow_html=True)
            st.markdown("### 🎙️ Voice")
            st.write(st.session_state.voice_status)
            st.markdown('</div>', unsafe_allow_html=True)

    col_input, col_voice = st.columns([4, 1])
    with col_input:
        prompt = st.text_area(
            "Your response",
            value=st.session_state.pending_input,
            key="pending_input",
            height=100,
        )
    with col_voice:
        voice_clicked = st.button("🎙️")

    if voice_clicked:
        voice_text = record_and_transcribe()
        if voice_text:
            st.session_state.pending_input = voice_text
            st.rerun()

    col_send, col_reset = st.columns(2)
    with col_send:
        send_clicked = st.button("Send")
    with col_reset:
        reset_clicked = st.button("🔄 Reset")

    if reset_clicked:
        st.session_state.clear()
        st.rerun()

    if send_clicked and prompt.strip():
        st.session_state.state["messages"].append({"role": "user", "content": prompt})
        with st.spinner("Analyzing..."):
            state, response, done = run_pipeline(st.session_state.state, prompt)
        update_memory(state, prompt, response)

        if done:
            st.session_state.state["messages"].append(
                {"role": "assistant", "content": "Thank you. I have generated your clinical report below."}
            )
            st.session_state.final_report = response
        else:
            st.session_state.state["messages"].append({"role": "assistant", "content": response})

        st.session_state.pending_input = ""
        st.rerun()

    if "final_report" in st.session_state:
        st.markdown("### 📋 Clinical Report")
        st.json(st.session_state.final_report)
        st.markdown(format_report(st.session_state.final_report))


if __name__ == "__main__":
    main()
