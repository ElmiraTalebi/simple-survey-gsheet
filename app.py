import json
import sys
import types
from pathlib import Path


APP_PATH = Path("/Users/elmiratalebianaraki/Documents/New project 2/app.py")


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def markdown(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def header(self, *args, **kwargs):
        return None

    def text_input(self, *args, **kwargs):
        return ""

    def text_area(self, *args, **kwargs):
        return ""

    def button(self, *args, **kwargs):
        return False


class FakeStreamlitModule(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SessionState()
        self.secrets = {}
        self.sidebar = DummyContext()

    def set_page_config(self, *args, **kwargs):
        return None

    def title(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def header(self, *args, **kwargs):
        return None

    def text_input(self, *args, **kwargs):
        return ""

    def text_area(self, *args, **kwargs):
        return ""

    def button(self, *args, **kwargs):
        return False

    def chat_input(self, *args, **kwargs):
        return None

    def chat_message(self, *args, **kwargs):
        return DummyContext()

    def rerun(self):
        return None


def normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def classify_presence(text: str) -> str:
    lowered = normalize_text(text)
    negative_markers = [
        "no",
        "none",
        "not",
        "don't",
        "dont",
        "denies",
        "without",
        "nope",
        "nah",
    ]
    positive_markers = [
        "yes",
        "have",
        "hurts",
        "pain",
        "sore",
        "ulcer",
        "dry",
        "difficulty",
        "trouble",
        "mucus",
        "nausea",
        "vomit",
        "blood",
        "tired",
        "weak",
        "anxious",
        "sad",
    ]
    if any(marker in lowered for marker in negative_markers):
        if any(phrase in lowered for phrase in ["no pain", "no sores", "no trouble", "no nausea", "no mucus", "no breathing"]):
            return "no"
    if lowered in {"yes", "y"}:
        return "yes"
    if lowered in {"no", "n"}:
        return "no"
    if any(marker in lowered for marker in positive_markers):
        return "yes"
    return "unknown"


def extract_details(topic: str, message: str) -> dict[str, str]:
    lowered = normalize_text(message)
    details = {}

    if topic == "pain":
        if "throat" in lowered:
            details["location"] = "throat"
        elif "tongue" in lowered:
            details["location"] = "tongue"
        elif "jaw" in lowered or "mouth" in lowered:
            details["location"] = "jaw/mouth"
        if "constant" in lowered:
            details["timing"] = "constant"
        elif "swallow" in lowered:
            details["timing"] = "with swallowing"
        elif "eat" in lowered:
            details["timing"] = "with eating"
        for token in lowered.replace("/", " ").split():
            if token.isdigit() and 0 <= int(token) <= 10:
                details["severity"] = token
                break
        if "tylenol" in lowered or "oxycodone" in lowered:
            details["medication_help"] = "using pain medication"

    if topic == "nutrition":
        if "soft" in lowered:
            details["intake_level"] = "soft foods only"
        elif "liquid" in lowered or "shakes" in lowered:
            details["intake_level"] = "liquids/shakes"
        elif "normal" in lowered:
            details["intake_level"] = "near normal"
        if "pain" in lowered:
            details["barriers"] = "pain with eating"
        elif "dry" in lowered:
            details["barriers"] = "dry mouth"
        elif "swallow" in lowered:
            details["barriers"] = "swallowing difficulty"

    if topic == "weight":
        for token in lowered.replace(",", " ").split():
            cleaned = token.replace("lb", "").replace("lbs", "")
            if cleaned.replace(".", "", 1).isdigit():
                details["weight"] = cleaned
                break

    if topic == "general":
        for token in lowered.replace("/", " ").split():
            if token.isdigit() and 0 <= int(token) <= 10:
                details["overall_score"] = token
                break

    if topic == "medications":
        if lowered:
            details["medications_list"] = message.strip()

    if topic == "mood" and lowered:
        details["emotional_state"] = message.strip()

    return details


def choose_followup(topic: str, candidate_followups: list[str], missing_fields: list[str], topic_data: dict) -> str:
    if not candidate_followups:
        return ""

    field_map = {
        "pain": {
            "location": "Where exactly is the pain?",
            "severity": "On a scale of 0–10, how bad is it?",
            "timing": "Is the pain constant or only when swallowing or eating?",
        },
        "nutrition": {
            "intake_level": "What are you able to eat right now?",
            "barriers": "What is making it difficult to eat or drink?",
        },
    }

    for field in missing_fields:
        mapped = field_map.get(topic, {}).get(field)
        if mapped in candidate_followups:
            return mapped

    for question in candidate_followups:
        lowered = question.lower()
        if "where" in lowered and "location" not in topic_data:
            return question
        if "0–10" in question or "0-10" in lowered:
            if "severity" not in topic_data:
                return question
        if "constant" in lowered or "swallowing" in lowered or "eating" in lowered:
            if "timing" not in topic_data:
                return question
    return candidate_followups[0]


def conversational_fatigue(history: list[dict[str, str]]) -> str:
    user_messages = [m["content"] for m in history if m["role"] == "user"]
    if not user_messages:
        return "low"
    short_count = sum(1 for msg in user_messages[-5:] if len(msg.strip()) <= 3)
    vague_count = sum(1 for msg in user_messages[-5:] if normalize_text(msg) in {"ok", "idk", "hmm", "?", "fine"})
    if short_count >= 3 or vague_count >= 2:
        return "high"
    if short_count >= 1 or len(user_messages) > 8:
        return "medium"
    return "low"


def fake_responses_create(model, temperature, input):
    system_prompt = input[0]["content"]
    payload = json.loads(input[1]["content"])

    if "SYMPTOM EXTRACTION AGENT" in system_prompt.upper():
        topic = payload["topic"]
        message = payload["user_input"]
        details = extract_details(topic, message)
        pain_block = {
            "present": True if classify_presence(message) == "yes" else False if classify_presence(message) == "no" else None,
            "location": details.get("location"),
            "severity": details.get("severity"),
            "timing": details.get("timing"),
        }
        response = {
            "pain": pain_block,
            "swallowing": "difficulty" if topic == "swallowing" and classify_presence(message) == "yes" else None,
            "nutrition": details.get("intake_level"),
            "oral_symptoms": details.get("oral_symptoms") or details.get("dry_mouth_details") or details.get("mucus_details"),
            "gi_symptoms": details.get("gi_symptoms"),
            "fatigue": details.get("fatigue_level"),
            "mood": details.get("emotional_state"),
            "breathing": "difficulty" if topic == "breathing" and classify_presence(message) == "yes" else "normal" if topic == "breathing" and classify_presence(message) == "no" else None,
            "other": message.strip()[:120],
        }
        return types.SimpleNamespace(output_text=json.dumps(response))

    if "CLINICAL IMPORTANCE AGENT" in system_prompt.upper():
        topic = payload["current_topic"]
        extracted = payload["extracted_symptoms"]
        required_fields = payload.get("required_fields", [])
        topic_data = payload.get("collected_data", {})
        presence = extracted.get("presence", "unknown")
        missing = [field for field in required_fields if not topic_data.get(field)]
        importance = payload["knowledge_base"][topic].get("priority", "medium")
        response = {
            "importance": importance,
            "needs_followup": presence == "yes" and bool(missing),
            "missing_fields": missing,
        }
        return types.SimpleNamespace(output_text=json.dumps(response))

    if "FOLLOW-UP AGENT" in system_prompt.upper():
        current_topic = payload["current_topic"]
        candidate_followups = payload["knowledge_base"].get(current_topic, {}).get("followups", [])
        response = {
            "follow_up_question": choose_followup(
                current_topic,
                candidate_followups,
                payload.get("missing_fields", []),
                {},
            ),
            "target_field": payload.get("missing_fields", [None])[0],
        }
        return types.SimpleNamespace(output_text=json.dumps(response))

    if "PATIENT EXPERIENCE AGENT" in system_prompt.upper():
        fatigue = conversational_fatigue(payload.get("conversation_history", []))
        response = {
            "fatigue_level": fatigue,
            "should_limit_questions": fatigue == "high",
        }
        return types.SimpleNamespace(output_text=json.dumps(response))

    if "SAFETY AGENT" in system_prompt.upper():
        summary = normalize_text(json.dumps(payload.get("extracted_symptoms", {})))
        alert = any(term in summary for term in ["severe", "difficulty", "cannot eat", "cannot drink", "high"])
        response = {
            "alert": alert,
            "reason": "Potential urgent symptom" if alert else "",
            "recommended_action": "flag_for_doctor" if alert else "continue",
        }
        return types.SimpleNamespace(output_text=json.dumps(response))

    if "ORCHESTRATOR" in system_prompt.upper():
        importance = payload["importance_assessment"]
        experience = payload["patient_state"]
        followup = payload.get("followup_candidate", {})
        action = "next_topic"
        if importance.get("needs_followup") and followup.get("question"):
            action = "follow_up"
            if experience.get("should_limit_questions") and importance.get("importance") != "high":
                action = "next_topic"
        response = {
            "action": action,
            "next_topic": None,
            "question": followup.get("question") if action == "follow_up" else None,
            "reason": "follow-up needed" if action == "follow_up" else "move on",
        }
        return types.SimpleNamespace(output_text=json.dumps(response))

    if "REPORT AGENT" in system_prompt.upper():
        collected_data = payload["collected_data"]
        response = {
            "pain": collected_data.get("pain", {}).get("summary"),
            "nutrition": collected_data.get("nutrition", {}).get("summary"),
            "swallowing": collected_data.get("swallowing", {}).get("summary"),
            "fatigue": collected_data.get("fatigue", {}).get("summary"),
            "other": collected_data.get("general", {}).get("summary"),
            "overall_priority": "high" if collected_data.get("pain", {}).get("severity") == "high" else "medium",
        }
        return types.SimpleNamespace(output_text=json.dumps(response))

    return types.SimpleNamespace(output_text="{}")


class FakeOpenAIClient:
    class Responses:
        @staticmethod
        def create(model, temperature, input):
            return fake_responses_create(model, temperature, input)

    def __init__(self, *args, **kwargs):
        self.responses = self.Responses()


def load_app_namespace():
    fake_streamlit = FakeStreamlitModule()
    sys.modules["streamlit"] = fake_streamlit
    openai_mod = types.ModuleType("openai")
    openai_mod.OpenAI = FakeOpenAIClient
    sys.modules["openai"] = openai_mod

    namespace = {"__name__": "__stress_test__"}
    exec(compile(APP_PATH.read_text(), str(APP_PATH), "exec"), namespace)

    def stub_call_json_agent(system_prompt: str, user_payload: dict, fallback: dict) -> dict:
        try:
            response = fake_responses_create(
                model="fake-model",
                temperature=0,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
            )
            parsed = json.loads(response.output_text)
            return parsed if isinstance(parsed, dict) else fallback
        except Exception:
            return fallback

    def stub_report_generator_agent(collected_data: dict, history: list[dict[str, str]]) -> str:
        previous_visit_history = fake_streamlit.session_state.get("previous_visit_history", "")
        previous_visit_topics = fake_streamlit.session_state.get("previous_visit_topics", {})
        return namespace["build_fallback_report"](collected_data, previous_visit_history, previous_visit_topics)

    namespace["call_json_agent"] = stub_call_json_agent
    namespace["report_generator_agent"] = stub_report_generator_agent
    return namespace, fake_streamlit


def fresh_state(ns, st):
    st.session_state.clear()
    ns["init_session_state"]()
    st.session_state.api_key_input = "test-key"
    ns["start_chat"]()


def move_to_next_section(ns, st) -> None:
    for section_key, _label in ns["SECTION_ORDER"]:
        if ns["section_status"](section_key) != "Complete":
            st.session_state.selected_section = section_key
            ns["ensure_section_started"](section_key)
            return


def run_scenario(ns, st, answers: list[str]) -> dict[str, object]:
    fresh_state(ns, st)
    for answer in answers:
        if st.session_state.finished:
            break
        if ns["section_status"](st.session_state.selected_section) == "Complete" and ns["has_remaining_topics"]():
            move_to_next_section(ns, st)
        ns["process_turn"](answer)

    while not st.session_state.finished and ns["has_remaining_topics"]():
        prior_section = st.session_state.selected_section
        move_to_next_section(ns, st)
        if st.session_state.selected_section == prior_section:
            break

    return {
        "finished": st.session_state.finished,
        "current_topic": st.session_state.current_topic,
        "message_count": len(st.session_state.messages),
        "final_report": st.session_state.final_report,
        "collected_data": st.session_state.collected_data,
        "topic_states": st.session_state.topic_states,
        "assistant_messages": [m["content"] for m in st.session_state.messages if m["role"] == "assistant"],
    }


def check(name: str, condition: bool, details: str, results: list[tuple[str, bool, str]]):
    results.append((name, condition, details))


def run_tests():
    ns, st = load_app_namespace()
    results: list[tuple[str, bool, str]] = []

    normal_answers = [
        "Yes, I have throat pain.",
        "It is in my throat.",
        "It is constant.",
        "7",
        "No mouth sores.",
        "Mostly soft foods and shakes because pain makes eating difficult.",
        "150 lbs",
        "Yes, my mouth is dry all day.",
        "Yes, swallowing is painful but I can drink liquids.",
        "No breathing trouble.",
        "Some thick mucus at night.",
        "No nausea, vomiting, or blood.",
        "Tylenol and oxycodone as needed.",
        "Yes, I feel more tired than usual.",
        "Not really, pain wakes me up.",
        "I feel anxious but supported.",
        "6",
    ]
    normal = run_scenario(ns, st, normal_answers)
    check(
        "normal scenario finishes and produces report",
        bool(normal["finished"] and normal["final_report"]),
        f"finished={normal['finished']}, report_present={bool(normal['final_report'])}",
        results,
    )
    pain_data = normal["collected_data"].get("pain", {})
    check(
        "pain follow-up fields are collected",
        all(pain_data.get(key) for key in ["location", "timing", "severity"]),
        f"pain_data={pain_data}",
        results,
    )

    denied_answers = ["No"] * 14
    denied = run_scenario(ns, st, denied_answers)
    check(
        "denied symptoms scenario still progresses deep into interview",
        denied["message_count"] >= 20 and not denied["topic_states"]["pain"]["asked_followups"],
        f"message_count={denied['message_count']}, pain_followups={denied['topic_states']['pain']['asked_followups']}",
        results,
    )

    fatigue_answers = [
        "Yes pain",
        "ok",
        "idk",
        "fine",
        "no",
        "no",
        "ok",
    ]
    fatigue = run_scenario(ns, st, fatigue_answers)
    nutrition_followups = fatigue["topic_states"]["nutrition"]["asked_followups"]
    check(
        "high conversational fatigue suppresses medium-value followups",
        len(nutrition_followups) == 0,
        f"nutrition_followups={nutrition_followups}",
        results,
    )

    ambiguous_answers = [
        "maybe",
        "not sure",
        "kind of",
        "unknown",
    ]
    ambiguous = run_scenario(ns, st, ambiguous_answers)
    check(
        "ambiguous answers do not crash the flow",
        ambiguous["message_count"] >= 6 and isinstance(ambiguous["collected_data"], dict),
        f"message_count={ambiguous['message_count']}, current_topic={ambiguous['current_topic']}",
        results,
    )

    report = normal["final_report"]
    check(
        "report includes structured markdown headings",
        "## Structured Clinical Report" in report and "### Pain" in report,
        f"report_preview={report[:200]!r}",
        results,
    )

    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed

    print(f"Stress test results: {passed} passed, {failed} failed")
    for name, ok, details in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        if not ok:
            print(f"  {details}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run_tests())
