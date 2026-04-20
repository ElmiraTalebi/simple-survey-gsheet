import hashlib
import html as _html
import io
import json
import re
import concurrent.futures as _futures
from difflib import get_close_matches
from datetime import datetime
from typing import Any, Optional

import streamlit as st
import streamlit.components.v1 as _stc
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI


# ══════════════════════════════════════════════════════════════════
# 🔥 NEW: Adaptive LLM Question Generator
# ══════════════════════════════════════════════════════════════════

def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _short_prev_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    text = str(value).strip()
    if len(text) > 160:
        text = text[:157] + "..."
    return text


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _is_redundant_followup(original_question: str, answer: str, followup_question: str) -> bool:
    oq = _norm_text(original_question)
    aq = _norm_text(answer)
    fq = _norm_text(followup_question)
    if not fq:
        return True
    if fq == oq or fq in oq or oq in fq:
        return True

    answer_words = set(aq.split())
    follow_words = set(fq.split())

    location_terms = {
        "hand", "hands", "jaw", "ear", "ears", "tongue", "throat", "mouth",
        "neck", "face", "lip", "lips", "gum", "gums", "shoulder", "chest",
        "arm", "arms", "leg", "legs", "back", "head",
    }
    if answer_words & location_terms and ("where" in follow_words or "located" in follow_words):
        return True

    return False


def _is_semantically_redundant_question(text_a: str, text_b: str) -> bool:
    a = _norm_text(text_a)
    b = _norm_text(text_b)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True

    stop = {
        "are", "you", "having", "have", "had", "any", "right", "now", "can",
        "could", "tell", "me", "about", "before", "please", "noticed", "notice",
        "your", "the", "do", "did", "is", "it", "feels", "feel",
    }
    a_words = {w for w in a.split() if w not in stop}
    b_words = {w for w in b.split() if w not in stop}
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words)
    smallest = min(len(a_words), len(b_words))
    return overlap >= 2 and overlap >= smallest - 1


def _coerce_structured_answer(
    topic_key: str,
    step: dict,
    answer: Any,
    current_data: dict,
    raw_answer: Any = None,
) -> Any:
    if not isinstance(answer, str):
        return answer

    raw = str(raw_answer if raw_answer is not None else answer).strip()
    if not raw:
        return answer

    if topic_key == "pain" and step["id"] == "pain_location":
        normalized = _norm_text(raw)
        # Specific HNC locations — match broadly (not just exact phrases)
        if any(w in normalized for w in ("throat", "pharynx", "larynx", "voice box")):
            return "Throat"
        if any(w in normalized for w in ("tongue", "lingual")):
            return "Tongue"
        # Already a clean option value — pass through unchanged
        if answer in ("Throat", "Tongue", "Somewhere else"):
            if answer == "Somewhere else" and raw not in ("Somewhere else", "somewhere else"):
                current_data["pain_location_raw"] = raw
                current_data["other_pain_desc"] = raw
            return answer
        # Everything else: it's a real location the patient named → preserve it
        # and classify as "Somewhere else" so the flowchart continues correctly
        current_data["pain_location_raw"] = raw
        current_data["other_pain_desc"] = raw
        return "Somewhere else"

    return answer


YES_SYNONYMS = {
    "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "of course", "i do",
    "i am", "it is", "there is", "correct", "right",
}

NO_SYNONYMS = {
    "no", "nope", "nah", "not really", "i dont", "i do not", "none", "negative",
}

BODY_LOCATION_TERMS = {
    "head", "scalp", "face", "jaw", "chin", "ear", "ears", "neck", "throat", "tongue",
    "mouth", "lip", "lips", "gum", "gums", "tooth", "teeth", "cheek", "palate",
    "shoulder", "chest", "arm", "arms", "elbow", "wrist", "hand", "hands", "finger",
    "fingers", "back", "side", "rib", "ribs", "stomach", "abdomen", "belly", "hip",
    "leg", "legs", "knee", "knees", "ankle", "ankles", "foot", "feet", "toe", "toes",
}

HEAD_NECK_LOCATION_TERMS = {
    "face", "jaw", "chin", "ear", "ears", "throat", "tongue",
    "mouth", "lip", "lips", "gum", "gums", "tooth", "teeth", "cheek", "palate",
}


def _looks_like_body_location(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    words = set(normalized.split())
    return bool(words & BODY_LOCATION_TERMS)


def _is_head_neck_location(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    words = set(normalized.split())
    return bool(words & HEAD_NECK_LOCATION_TERMS)


def _needs_head_neck_followup(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    words = set(normalized.split())
    focused_terms = {
        "ear", "ears", "jaw", "chin", "mouth", "lip", "lips", "gum", "gums",
        "tooth", "teeth", "cheek", "palate", "tongue", "throat",
    }
    return bool(words & focused_terms)


def _indicates_no_low_mood(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False

    explicit_phrases = {
        "no low mood",
        "not feeling down",
        "not depressed",
        "i am not depressed",
        "i m not depressed",
        "i dont feel down",
        "i do not feel down",
        "i dont have low mood",
        "i do not have low mood",
        "my mood is okay",
        "my mood is fine",
        "emotionally okay",
        "emotionally fine",
        "doing okay emotionally",
        "doing fine emotionally",
        "i feel okay",
        "i feel fine",
    }
    if normalized in {"okay", "ok", "fine", "good"}:
        return True
    if "low mood" in normalized and any(token in normalized for token in {"no", "not", "dont", "don t"}):
        return True
    if "depressed" in normalized and any(token in normalized for token in {"no", "not", "dont", "don t"}):
        return True
    if "feeling down" in normalized and any(token in normalized for token in {"no", "not", "dont", "don t"}):
        return True
    return any(phrase in normalized for phrase in explicit_phrases)


def _format_prior_answer_for_prompt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    text = str(value).strip()
    if not text:
        return ""
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return text
    if len(text) > 80:
        text = text[:77] + "..."
    return text


def _match_binary_option(step: dict, user_input: str) -> Optional[str]:
    opts = step.get("opts", [])
    if len(opts) != 2:
        return None
    normalized = _norm_text(user_input)
    option_map = {_norm_text(opt): opt for opt in opts}
    yes_opt = option_map.get("yes")
    no_opt = option_map.get("no")
    if yes_opt and normalized in YES_SYNONYMS:
        return yes_opt
    if no_opt and normalized in NO_SYNONYMS:
        return no_opt
    return None


_FREQUENCY_HINTS = {
    "daily", "everyday", "every day", "nightly", "weekly", "twice", "three times",
    "four times", "once", "morning", "evening", "bedtime", "hour", "hours",
    "every other day", "every other night", "per day", "per week", "a day", "a week",
    "as needed", "prn",
}

_NUMBER_WORD_PATTERN = r"(one|two|three|four|five|six|seven|eight|nine|ten|half|couple|few|several)"
_TIME_UNIT_PATTERN = r"(hour|hours|day|days|night|nights|week|weeks|month|months)"
_MED_FORM_PATTERN = r"(mg|mcg|g|ml|milligram|milligrams|tablet|tablets|pill|pills|capsule|capsules|teaspoon|teaspoons|tablespoon|tablespoons)"
_RELATION_TERMS = {
    "family", "friend", "friends", "caregiver", "caregivers", "wife", "husband",
    "daughter", "son", "mom", "dad", "mother", "father", "sister", "brother",
    "partner", "spouse", "neighbor", "roommate", "children", "child",
}
_MANAGEMENT_TERMS = {
    "zofran", "compazine", "anti nausea", "anti-nausea", "imodium", "miralax", "senna", "gabapentin", "oxycodone", "advil",
    "tylenol", "motrin", "ibuprofen", "rinse", "mouthwash", "patch", "tube feed",
    "salt water", "baking soda", "ensure", "boost", "water", "liquids", "soft foods",
}
_HELPING_TERMS = {
    "help", "helping", "helps", "working", "works", "not enough", "doesnt help",
    "does not help", "no relief", "better", "worse", "relief", "effective", "easier",
}
_REASON_TERMS = {
    "because", "due to", "from", "since", "hard", "difficult", "difficulty", "forget",
    "forgot", "pain", "fatigue", "tired", "nausea", "dry mouth", "not hungry",
    "appetite", "schedule", "cost", "insurance", "transportation", "side effect",
    "busy", "sleep", "swallow", "chew", "taste", "money", "refill",
}
_START_TIME_TERMS = {
    "today", "yesterday", "tonight", "this morning", "last night", "week", "weeks",
    "month", "months", "day", "days", "ago", "since", "started", "start", "begin",
    "began", "recently", "suddenly", "after", "before", "during", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february",
    "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "radiation", "chemo", "chemotherapy", "surgery",
}

_UNKNOWN_PHRASES = {
    "dont remember", "i dont remember", "do not remember", "not sure", "unsure",
    "no idea", "unknown", "cant remember", "cannot remember", "dont know",
    "i dont know", "not sure about", "not sure exactly", "i forget", "forgot",
}

_OPTION_ALIAS_HINTS = {
    "schedule": ["forget", "forgot", "late", "on time", "timing", "routine", "remember"],
    "side effects": ["side effect", "nausea", "drowsy", "sleepy", "constipated", "makes me sick", "dizzy"],
    "access issues": ["cost", "insurance", "refill", "pharmacy", "ran out", "couldnt get", "could not get"],
    "no appetite": ["no appetite", "not hungry", "appetite"],
    "nausea": ["nausea", "nauseous", "sick to my stomach"],
    "too tired to prepare food": ["too tired", "no energy", "too exhausted"],
    "pain when eating/swallowing": ["pain when swallowing", "hurts to swallow", "hurts to eat", "pain eating"],
    "dry mouth": ["dry mouth", "mouth is dry"],
    "pain": ["pain", "hurts", "ache", "aching", "sore"],
    "fatigue": ["fatigue", "tired", "exhausted", "worn out", "weak"],
    "treatment side effects": ["treatment", "radiation", "chemo", "chemotherapy", "side effect"],
    "energy levels": ["energy", "tired", "fatigue", "exhausted"],
    "family, friends, or caregivers": ["family", "friend", "caregiver", "wife", "husband", "daughter", "son", "mom", "dad"],
    "yes, it helps": ["helps", "working", "better"],
    "yes, but it's not enough": ["not enough", "barely helps", "helps a little", "still hurts", "still not enough"],
}


def _has_frequency_info(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    if any(phrase in normalized for phrase in _FREQUENCY_HINTS):
        return True
    if re.search(r"\b\d+\s*(x|times?)\b", normalized):
        return True
    if re.search(rf"\b{_NUMBER_WORD_PATTERN}\s+times?\b", normalized):
        return True
    if re.search(r"\bevery\s+\d+\s*(hour|hours|day|days)\b", normalized):
        return True
    if re.search(rf"\bevery\s+{_NUMBER_WORD_PATTERN}\s*{_TIME_UNIT_PATTERN}?\b", normalized):
        return True
    if re.search(r"\b(once|twice)\s+(a|per)?\s*(day|night|week|month)\b", normalized):
        return True
    if re.search(rf"\b{_NUMBER_WORD_PATTERN}\s+times?\s+(a|per)?\s*(day|night|week|month)\b", normalized):
        return True
    if re.search(r"\b(in the morning|in the evening|at night|before bed|at bedtime|with meals|before meals|after meals)\b", normalized):
        return True
    return False


def _is_unknown_answer(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    if normalized in _UNKNOWN_PHRASES:
        return True
    return any(phrase in normalized for phrase in _UNKNOWN_PHRASES)


def _has_dose_info(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    if re.search(rf"\b\d+\s*{_MED_FORM_PATTERN}\b", normalized):
        return True
    if re.search(rf"\b{_NUMBER_WORD_PATTERN}\s+{_MED_FORM_PATTERN}\b", normalized):
        return True
    if re.search(r"\b\d+\b", normalized) and any(term in normalized for term in {"dose", "tablet", "pill", "capsule"}):
        return True
    return False


def _has_amount_info(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    if re.search(r"\b\d+\b", normalized):
        return True
    if re.search(rf"\b{_NUMBER_WORD_PATTERN}\b", normalized):
        return True
    return any(token in normalized for token in {"small", "little", "a lot", "large", "amount", "few", "several", "half", "full"})


def _has_helping_info(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    return any(phrase in normalized for phrase in _HELPING_TERMS)


def _has_management_info(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    return bool(
        re.search(r"\b(take|taking|use|using|used|try|trying|on|take|drink|drinking)\b", normalized)
        or re.search(r"\b\d+\s*(mg|mcg|g|ml)\b", normalized)
        or any(word in normalized for word in _MANAGEMENT_TERMS)
    )


def _previous_step_in_flow(topic_key: str, step_id: str) -> Optional[dict]:
    flow = FLOWS.get(topic_key, [])
    prev = None
    for step in flow:
        if step["id"] == step_id:
            return prev
        prev = step
    return None


def _contextualize_raw_phrase(text: str, max_words: int = 8) -> str:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return ""
    words = normalized.split()
    if len(words) > max_words:
        normalized = " ".join(words[:max_words]) + "..."
    return normalized


def _infer_option_from_text(step: dict, user_input: str) -> Optional[str]:
    binary = _match_binary_option(step, user_input)
    if binary:
        return binary

    normalized = _norm_text(user_input)
    if not normalized or not step.get("opts"):
        return None

    for opt in step.get("opts", []):
        opt_norm = _norm_text(opt)
        if not opt_norm:
            continue
        if opt_norm in normalized:
            return opt
        aliases = _OPTION_ALIAS_HINTS.get(opt_norm, [])
        if any(alias in normalized for alias in aliases):
            return opt
    return None


def _has_location_info(text: str) -> bool:
    normalized = _norm_text(text)
    return bool(normalized) and (_looks_like_body_location(text) or any(
        token in normalized for token in {"inside", "outside", "left", "right", "back", "front", "near", "around"}
    ))


def _has_start_time_info(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    if any(token in normalized for token in _START_TIME_TERMS):
        return True
    if re.search(rf"\b\d+\s+{_TIME_UNIT_PATTERN}\s+ago\b", normalized):
        return True
    if re.search(rf"\b{_NUMBER_WORD_PATTERN}\s+{_TIME_UNIT_PATTERN}\s+ago\b", normalized):
        return True
    return bool(re.search(r"\bsince\s+\w+\b", normalized))


def _has_support_info(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    return any(token in normalized for token in _RELATION_TERMS)


def _has_reason_info(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    if len(normalized.split()) >= 3:
        return True
    return any(token in normalized for token in _REASON_TERMS)


def _has_specific_type_info(text: str) -> bool:
    normalized = _norm_text(text)
    return len(normalized.split()) >= 2 and not _is_unknown_answer(text)


def _has_plain_yes_no_signal(text: str) -> bool:
    return _match_binary_option({"opts": ["Yes", "No"]}, text) is not None


DETECTORS_BY_KIND = {
    "frequency": _has_frequency_info,
    "dose": _has_dose_info,
    "amount": _has_amount_info,
    "helping": _has_helping_info,
    "management": _has_management_info,
    "location": _has_location_info,
    "start_time": _has_start_time_info,
    "support": _has_support_info,
    "reason": _has_reason_info,
    "specific_type": _has_specific_type_info,
    "yes_no_signal": _has_plain_yes_no_signal,
}


def _best_known_pain_medication_label(state: dict) -> str:
    raw_answers = state.get("raw_answers", {})
    raw_text = str(raw_answers.get("pain_medications", "")).strip()
    if raw_text:
        cleaned = raw_text.strip(" .")
        cleaned = re.sub(r"\b(nothing else|and nothing else|only|just)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[,.]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            first_part = cleaned.split(" and ")[0].split(" but ")[0].strip()
            if first_part:
                return first_part.title()

    meds = state.get("data", {}).get("pain_medications") or []
    if isinstance(meds, list):
        meds = [m for m in meds if m not in {"Other", "No pain medication"}]
        if meds:
            return meds[0]
    return "that medication"


def _dose_is_unknown(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    return ("dose" in normalized and _is_unknown_answer(normalized)) or "no idea" in normalized


def _infer_components_from_question(step: dict) -> dict[str, dict[str, Any]]:
    question = _norm_text(step.get("text", ""))
    if not question or step.get("type") != "free_text":
        return {}

    if "how often" in question and ("dose" in question or "how much" in question):
        return {
            "frequency": {
                "detector": "frequency",
                "question": "How often is that usually?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if you're not sure about the timing right now. I've noted what you could tell me.",
            },
            "dose": {
                "detector": "dose",
                "question": "About how much is it each time?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if you don't remember the dose right now. I've noted that for your care team.",
            },
        }

    if "what are you using" in question and "helping" in question:
        return {
            "management": {"detector": "management", "question": "What have you been using for it?"},
            "helping": {"detector": "helping", "question": "Has that been helping at all?"},
        }

    if question.startswith("when did") or "when did it start" in question or "when did this" in question or "when did the" in question:
        return {
            "start_time": {
                "detector": "start_time",
                "question": "About when did it start?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if you're not sure exactly when it started. I've noted that for your care team.",
            }
        }

    if question.startswith("where") or "where is" in question or "where are" in question or "where exactly" in question or "which body part" in question:
        return {
            "location": {
                "detector": "location",
                "question": "Where are you feeling it?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if it's hard to describe exactly. I've noted what you could tell me.",
            }
        }

    if question.startswith("who ") and ("support" in question or "help" in question):
        return {
            "support": {
                "detector": "support",
                "question": "Who has been helping you most?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if you don't want to name anyone right now.",
            }
        }

    if question.startswith("what kind of support") or question.startswith("what is making") or "what s limiting" in question or "tell me more about what s limiting" in question:
        return {
            "reason": {
                "detector": "reason",
                "question": "What feels like the main thing making that harder right now?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if it's hard to pin down exactly. I've noted that this has been difficult.",
            }
        }

    if question.startswith("what kind") or question.startswith("what type"):
        return {
            "specific_type": {
                "detector": "specific_type",
                "question": "What kind is it?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if you don't know the exact type right now.",
            }
        }

    return {}


def _generic_missing_detail_override(step: dict, answer: str, state: dict) -> dict[str, Any]:
    schema = STEP_SCHEMAS.get(step.get("id"), {})
    components = schema.get("components") or _infer_components_from_question(step)
    if components:
        missing = []
        for name, comp in components.items():
            detector_name = comp.get("detector")
            detector = DETECTORS_BY_KIND.get(detector_name)
            if detector and not detector(answer):
                missing.append((name, comp))

        if missing:
            first_name, first_comp = missing[0]
            if _is_unknown_answer(answer) and first_comp.get("unknown_ok"):
                return {
                    "follow_up_recommended": False,
                    "follow_up_goal": None,
                    "assistant_message": first_comp.get("unknown_ack") or "That's okay if you're not sure right now. I've noted that for your care team.",
                    "information_completeness": "partial",
                    "clinical_priority": "low",
                }
            return {
                "follow_up_recommended": True,
                "follow_up_goal": f"Obtain the missing detail for {first_name.replace('_', ' ')} only.",
                "follow_up_question": first_comp.get("question") or "Could you tell me a bit more about that?",
                "information_completeness": "partial",
                "clinical_priority": "medium",
            }

    question = _norm_text(step.get("text", ""))
    if not question or not isinstance(answer, str):
        return {}

    if _is_unknown_answer(answer):
        if any(token in question for token in {"when did", "how often", "what has your weight", "how high", "blood pressure", "what has your blood pressure", "when did it start", "where is", "where exactly", "what type", "which body part"}):
            return {
                "follow_up_recommended": False,
                "follow_up_goal": None,
                "assistant_message": "That's okay if you're not sure right now. I've noted that for your care team.",
                "information_completeness": "partial",
                "clinical_priority": "low",
            }

    if "what are you using" in question and "is it helping" in question:
        has_management = _has_management_info(answer)
        has_helping = _has_helping_info(answer)
        if has_management and not has_helping:
            return {
                "follow_up_recommended": True,
                "follow_up_goal": "Obtain whether the management strategy is helping.",
                "follow_up_question": "Has that been helping at all?",
                "information_completeness": "partial",
                "clinical_priority": "medium",
            }
        if has_helping and not has_management:
            return {
                "follow_up_recommended": True,
                "follow_up_goal": "Obtain what the patient is using to manage the symptom.",
                "follow_up_question": "What have you been using for it?",
                "information_completeness": "partial",
                "clinical_priority": "medium",
            }

    if "how often" in question and "how much" in question:
        has_freq = _has_frequency_info(answer)
        has_amount = _has_amount_info(answer)
        if has_freq and not has_amount and not _is_unknown_answer(answer):
            return {
                "follow_up_recommended": True,
                "follow_up_goal": "Obtain the amount only; frequency was already provided.",
                "follow_up_question": "About how much is it each time?",
                "information_completeness": "partial",
                "clinical_priority": "medium",
            }
        if has_amount and not has_freq and not _is_unknown_answer(answer):
            return {
                "follow_up_recommended": True,
                "follow_up_goal": "Obtain the frequency only; amount was already provided.",
                "follow_up_question": "How often has that been happening?",
                "information_completeness": "partial",
                "clinical_priority": "medium",
            }

    return {}


def _targeted_followup_override(step: dict, answer: str, state: dict) -> dict[str, Any]:
    generic = _generic_missing_detail_override(step, answer, state)
    if generic:
        return generic

    if step.get("id") != "med_dose_freq":
        return {}

    has_freq = _has_frequency_info(answer)
    has_dose = _has_dose_info(answer)
    dose_unknown = _dose_is_unknown(answer)
    med_label = _best_known_pain_medication_label(state)

    if has_freq and dose_unknown:
        return {
            "follow_up_recommended": False,
            "follow_up_goal": None,
            "assistant_message": f"That's okay if you don't remember the dose right now. I've noted that you take {med_label} almost every day.",
            "information_completeness": "partial",
            "clinical_priority": "low",
        }

    if has_freq and not has_dose:
        return {
            "follow_up_recommended": True,
            "follow_up_goal": "Obtain the medication dose only; frequency was already provided.",
            "follow_up_question": f"About how much {med_label} do you usually take each time?",
            "information_completeness": "partial",
            "clinical_priority": "medium",
        }

    if has_dose and not has_freq:
        return {
            "follow_up_recommended": True,
            "follow_up_goal": "Obtain the medication frequency only; dose was already provided.",
            "follow_up_question": f"How often do you usually take {med_label}?",
            "information_completeness": "partial",
            "clinical_priority": "medium",
        }

    return {}


def _looks_vague_answer(answer: Any) -> bool:
    if not isinstance(answer, str):
        return False
    text = _norm_text(answer)
    if not text:
        return True

    vague_phrases = {
        "idk", "i dont know", "dont know", "not sure", "unsure", "maybe",
        "kinda", "kind of", "sort of", "ugh", "stuff", "things", "whatever",
    }
    if text in vague_phrases:
        return True

    words = text.split()
    if len(words) >= 1:
        unique_chars = set(text.replace(" ", ""))
        if len(unique_chars) <= 2 and len(text.replace(" ", "")) >= 4:
            return True
        if re.fullmatch(r"[a-zA-Z]{1,2,}", text):
            return True
    return False


def _fallback_clarifying_question(step: dict) -> str:
    text = step.get("text", "").strip()
    if not text:
        return "Could you tell me a little more about that so I can capture it accurately for your care team?"
    lower = _norm_text(text)
    if "where" in lower and "pain" in lower:
        return "Could you tell me where the pain is located?"
    if "pain" in lower:
        return "Could you tell me a bit more about the pain you're having right now?"
    return "Could you tell me a little more about that?"


def _suggest_step_options(step: dict, user_input: str, limit: int = 3) -> list[str]:
    raw = _norm_text(user_input)
    if not raw:
        return []

    candidate_options = [
        opt for opt in step.get("opts", [])
        if _norm_text(opt) not in {"other", "somewhere else"}
    ]
    if not candidate_options:
        return []

    normalized_to_option = {_norm_text(opt): opt for opt in candidate_options}
    matches = get_close_matches(raw, list(normalized_to_option.keys()), n=limit, cutoff=0.45)
    suggestions = [normalized_to_option[m] for m in matches]

    if suggestions:
        return suggestions

    raw_words = set(raw.split())
    token_matches = []
    for opt in candidate_options:
        opt_words = set(_norm_text(opt).split())
        if raw_words & opt_words:
            token_matches.append(opt)
    return token_matches[:limit]


def _build_retry_prompt(step: dict, user_input: str) -> str:
    schema = STEP_SCHEMAS.get(step.get("id"), {})
    if schema.get("unmatched_followup") and not _looks_vague_answer(user_input):
        return schema["unmatched_followup"]

    if _looks_vague_answer(user_input):
        return "I didn’t quite catch that. Could you please say it again?"

    suggestions = _suggest_step_options(step, user_input)
    if suggestions:
        if len(suggestions) == 1:
            return f"I want to make sure I record that correctly. Did you mean {suggestions[0]}?"
        if len(suggestions) == 2:
            return f"I want to make sure I record that correctly. Did you mean {suggestions[0]} or {suggestions[1]}?"
        return (
            f"I want to make sure I record that correctly. Did you mean "
            f"{suggestions[0]}, {suggestions[1]}, or {suggestions[2]}?"
        )

    if "Other" in step.get("opts", []):
        return (
            "I didn’t find a clear match in the quick options. "
            "Please type it again, and if it’s a different medication I’ll record it as another one."
        )

    return "I didn’t find a clear match there. Could you please say it again or choose the closest option?"


def _auto_capture_following_answers(topic_key: str, state: dict, seed_text: str):
    text = (seed_text or "").strip()
    if not text or _looks_vague_answer(text):
        return

    for _ in range(3):
        next_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
        if not next_step or next_step.get("type") not in {"options", "multi_select"}:
            return

        inferred = None
        if next_step["type"] == "options":
            inferred = _infer_option_from_text(next_step, text)
            if not inferred and len(_norm_text(text).split()) >= 3:
                maybe = interpret_user_input_with_options(next_step, text)
                if maybe in next_step.get("opts", []) and not _is_catchall_option_value(maybe):
                    inferred = maybe
            if inferred in next_step.get("opts", []):
                state["data"][next_step["id"]] = inferred
                state["raw_answers"][next_step["id"]] = text
                continue
            return

        if next_step["type"] == "multi_select":
            parsed = parse_multi_select_typed_input(next_step, text)
            if parsed:
                if all(item == "Other" for item in parsed):
                    return
                state["data"][next_step["id"]] = parsed
                state["raw_answers"][next_step["id"]] = text
                continue
            return







def parse_multi_select_typed_input(step: dict, user_input: str):
    if not user_input.strip():
        return []

    lowered_map = {opt.lower(): opt for opt in step.get("opts", [])}
    parts = [p.strip() for p in re.split(r",|/|;|\n", user_input) if p.strip()]
    resolved = []
    has_other = "Other" in step.get("opts", [])
    for part in parts:
        match = lowered_map.get(part.lower())
        if match:
            resolved.append(match)
        else:
            interpreted = interpret_user_input_with_options(step, part)
            if interpreted in step.get("opts", []):
                resolved.append(interpreted)
            elif has_other and not _looks_vague_answer(part):
                resolved.append("Other")

    deduped = []
    for item in resolved:
        if item not in deduped:
            deduped.append(item)
    return deduped


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ChatReport — HNC Symptom Check-In",
    page_icon="🩺",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg1: #f7fafc;
    --bg2: #f3f7fa;
    --bg3: #eef4f8;
    --card: rgba(255,255,255,0.94);
    --card-solid: #ffffff;
    --border: #d7e4ef;
    --border-strong: #bfd3e4;
    --text: #17324a;
    --muted: #65788d;
    --primary: #0f6cbd;
    --primary-strong: #0a5a9f;
    --primary-soft: #eef7ff;
    --primary-ink: #11456d;
    --accent: #0d9488;
    --accent-soft: #ecfdf8;
    --success: #15803d;
    --success-soft: #ecfdf5;
    --warning-soft: #fff8e8;
    --shadow: 0 12px 32px rgba(23, 50, 74, 0.05);
    --shadow-sm: 0 6px 18px rgba(23, 50, 74, 0.04);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

html, body, .stApp {
    background:
        radial-gradient(circle at top left, rgba(15,108,189,0.05), transparent 22%),
        radial-gradient(circle at top right, rgba(13,148,136,0.04), transparent 24%),
        linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 58%, var(--bg3) 100%);
}

/* ── Layout ── */
.block-container {
    padding-top: 1.15rem;
    padding-bottom: 2.2rem;
    max-width: 1380px;
    padding-left: 1.1rem;
    padding-right: 1.1rem;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 0.65rem;
        padding-right: 0.65rem;
    }
}

/* ── Sidebar nav ── */
section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top, rgba(255,255,255,0.20), transparent 35%),
        linear-gradient(180deg, #123a5b 0%, #163f60 55%, #183b58 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
}

/* ── Buttons (main content area) ── */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 0.68rem 0.95rem;
    font-family: 'Manrope', sans-serif;
    font-size: 14px;
    font-weight: 700;
    border: 1px solid var(--border);
    background: linear-gradient(180deg, #ffffff 0%, #f8fbfe 100%);
    color: var(--text);
    transition: all 0.16s ease;
    text-align: center !important;
    box-shadow: none;
}
.stButton > button:hover {
    border-color: #91b7d7;
    background: #ffffff;
    color: #123664;
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(15, 108, 189, 0.08);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary) 0%, #2f88d5 100%);
    color: white;
    border: none;
    box-shadow: 0 14px 28px rgba(15, 108, 189, 0.22);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--primary-strong) 0%, var(--primary) 100%);
    color: white;
}

/* ── Sidebar nav buttons ── */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    padding: 12px 13px !important;
    font-size: 12.8px !important;
    font-weight: 700 !important;
    line-height: 1.45 !important;
    min-height: 0 !important;
    border-radius: 16px !important;
    margin-bottom: 6px !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    text-align: left !important;
    color: #eef7ff !important;
    border-color: rgba(255,255,255,0.10) !important;
    background: rgba(255,255,255,0.06) !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    border-color: rgba(255,255,255,0.16) !important;
    background: rgba(255,255,255,0.11) !important;
    color: #ffffff !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primaryFormSubmit"],
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
    font-size: 13.4px !important;
    font-weight: 700 !important;
    padding: 12px 14px !important;
    white-space: normal !important;
    color: white !important;
    margin-top: 4px !important;
    background: linear-gradient(135deg, #1184d1 0%, #0f6cbd 100%) !important;
}

/* ── Inputs ── */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
div[data-baseweb="select"] > div {
    border-radius: 14px !important;
    border: 1px solid #cfdeeb !important;
    background: rgba(255,255,255,0.98) !important;
    box-shadow: none !important;
}
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    padding: 0.9rem 1rem !important;
}
.stTextInput label,
.stTextArea label,
.stNumberInput label,
[data-testid="stAudioInput"] label {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    color: var(--primary-ink) !important;
}

/* ── Chat message wrappers ── */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    margin-bottom: 10px;
    padding: 0.05rem 0;
    background: transparent;
    display: flex;
    width: 100%;
}
[data-testid="stChatMessageContent"] {
    border-radius: 16px;
    padding: 0.8rem 0.95rem !important;
    border: 1px solid rgba(215, 228, 239, 0.9);
    box-shadow: none;
    background: #ffffff;
    width: fit-content;
    max-width: min(72%, 680px);
}
[data-testid="stChatMessageAvatar"] {
    display: none !important;
}
[data-testid="stChatMessage"]:has([aria-label="assistant"]) {
    justify-content: flex-start;
}
[data-testid="stChatMessage"]:has([aria-label="assistant"]) [data-testid="stChatMessageContent"] {
    border-left: 3px solid #b7d5eb;
    background: #ffffff;
}
[data-testid="stChatMessage"]:has([aria-label="user"]) {
    justify-content: flex-end;
}
[data-testid="stChatMessage"]:has([aria-label="user"]) [data-testid="stChatMessageContent"] {
    background: #f8fbfe;
    border-left: 3px solid #0f6cbd;
    border-right: 3px solid #0f6cbd;
    border-left: none;
}

.chat-shell {
    background:
        radial-gradient(circle at top right, rgba(15,108,189,0.07), transparent 32%),
        linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248,251,254,0.98) 100%);
    border: 1px solid #d9e6f0;
    border-radius: 26px;
    padding: 0;
    overflow: hidden;
    box-shadow: 0 24px 60px rgba(23, 50, 74, 0.09);
    margin-top: 6px;
}

.chat-shell-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 10px 12px 10px 12px;
    border-bottom: 1px solid #e2ebf2;
    background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(245,249,252,0.9) 100%);
}

.chat-shell-title {
    display: flex;
    align-items: center;
    gap: 10px;
}

.chat-shell-avatar {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    background: linear-gradient(135deg, #0f6cbd 0%, #26a69a 100%);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    box-shadow: 0 8px 18px rgba(15,108,189,0.14);
}

.chat-shell-title-text {
    display: flex;
    flex-direction: column;
    gap: 1px;
}

.chat-shell-label {
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6e8497;
}

.chat-shell-name {
    font-family: 'Manrope', sans-serif;
    font-size: 14px;
    font-weight: 800;
    color: #143551;
    letter-spacing: -0.03em;
}

.chat-shell-note {
    font-size: 10px;
    color: #607589;
    background: #f3f8fb;
    border: 1px solid #e6eef5;
    border-radius: 999px;
    padding: 6px 9px;
}

.chat-shell-summary {
    padding: 10px 12px;
    border-bottom: 1px solid #e8eef4;
    background: rgba(247, 251, 254, 0.72);
    font-size: 12px;
    line-height: 1.55;
    color: #5a7085;
}

.chat-shell-summary strong {
    color: #17324a;
}

.chat-history {
    padding: 14px 14px 8px 14px;
    min-height: 220px;
    background:
        linear-gradient(180deg, rgba(250,252,254,0.88) 0%, rgba(244,248,252,0.92) 100%);
}

.composer-wrap {
    padding: 0 12px 12px 12px;
    background: transparent;
}

.chat-row {
    display: flex !important;
    width: 100% !important;
    margin-bottom: 12px;
    align-items: flex-start;
    justify-content: flex-start !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    gap: 10px;
}

.chat-avatar {
    width: 26px;
    height: 26px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 26px;
    font-size: 11px;
    font-weight: 800;
    color: white;
    margin-top: 2px;
}

.chat-row.assistant .chat-avatar {
    background: #ef476f;
}

.chat-row.user .chat-avatar {
    background: #20262d;
}

.chat-entry {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-width: min(86%, 860px);
    min-width: 0;
}

.chat-meta {
    display: flex;
    align-items: baseline;
    gap: 6px;
    padding: 0 2px;
}

.chat-role {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: none;
    color: #1d2b36;
}

.chat-time {
    font-size: 10px;
    color: #9aa9b6;
}

.chat-bubble {
    display: block;
    width: 100%;
    max-width: 100%;
    border-radius: 16px;
    padding: 0.62rem 0.75rem;
    border: 1px solid #f4f7fa;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 13.5px;
    box-shadow: none;
    background: rgba(255,255,255,0.72);
}

.chat-row.assistant .chat-bubble {
    color: #17324a;
    border-top-left-radius: 6px;
}

.chat-row.user .chat-bubble {
    color: #17324a;
    border-top-left-radius: 6px;
}

/* ── Topic status pills ── */
.status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    margin-left: 6px;
}
.pill-done   { background: #d1fae5; color: #065f46; }
.pill-active { background: #dbeafe; color: #1e40af; }
.pill-todo   { background: #f3f4f6; color: #6b7280; }

/* ── Modern shell cards ── */
.card {
    background: var(--card);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.7);
    border-radius: 24px;
    padding: 22px 24px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
}

.soft-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 18px 18px;
    box-shadow: var(--shadow-sm);
}

.assistant-chip {
    display: none;
}
.assistant-chip .avatar {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #d5ecfb, #edf8ff);
    font-size: 22px;
}
.assistant-chip .name {
    font-family: 'Manrope', sans-serif;
    font-size: 14px;
    font-weight: 800;
    color: #113553;
    margin-bottom: 2px;
}
.assistant-chip .role {
    font-size: 12px;
    color: #678196;
}

.memory-banner {
    background: linear-gradient(135deg, #edf7ff 0%, #f9fcff 100%);
    border: 1px solid #cadeef;
    border-radius: 18px;
    padding: 12px 14px;
    color: #23486f;
    font-size: 13px;
    margin-bottom: 12px;
}

.report-box {
    background: rgba(255,255,255,0.94);
    border: 1px solid #dbe5f1;
    border-radius: 22px;
    padding: 24px 28px;
    font-size: 14.5px;
    line-height: 1.72;
    white-space: pre-wrap;
    box-shadow: var(--shadow);
}

/* ── Progress / completion ── */
.prog-label {
    font-size: 12px;
    color: #6b7280;
    margin-bottom: 4px;
    font-weight: 600;
}
.completion-badge {
    background: linear-gradient(135deg, #0f9f6e, #0f6cbd);
    border-radius: 20px;
    padding: 18px 20px;
    color: white;
    font-weight: 700;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 14px 28px rgba(15, 108, 189, 0.20);
}

/* ── Welcome / login ── */
.welcome-card {
    background:
        radial-gradient(circle at top right, rgba(15,108,189,0.12), transparent 30%),
        linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(246,250,254,0.99) 100%);
    border: 1px solid #d6e4f0;
    border-radius: 30px;
    padding: 32px 36px;
    max-width: 720px;
    margin: 56px auto 24px auto;
    box-shadow: var(--shadow);
}

.overview-card {
    background:
        radial-gradient(circle at top right, rgba(15,108,189,0.08), transparent 32%),
        linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,251,254,0.98) 100%);
    border: 1px solid #d6e4f0;
    border-radius: 28px;
    padding: 28px 30px;
    max-width: 960px;
    margin: 24px auto 18px auto;
    box-shadow: var(--shadow);
}

.overview-table-wrap {
    margin-top: 18px;
    border: 1px solid #d7e4ee;
    border-radius: 22px;
    overflow: hidden;
    background: #ffffff;
    box-shadow: 0 8px 20px rgba(23, 50, 74, 0.04);
}

.overview-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
}

.overview-table col.topic-col {
    width: 220px;
}

.overview-table col.summary-col {
    width: 250px;
}

.overview-table th,
.overview-table td {
    padding: 16px 18px;
    vertical-align: top;
    border-bottom: 1px solid #edf3f7;
}

.overview-table thead th {
    background: #f5f9fd;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6a8198;
    text-align: left;
    border-bottom: 1px solid #d7e4ee;
}

.overview-table tbody tr:last-child td {
    border-bottom: none;
}

.overview-topic-name {
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #16324b;
}

.overview-summary-main {
    font-size: 14px;
    line-height: 1.55;
    color: #16324b;
    font-weight: 700;
    margin-bottom: 8px;
}

.overview-summary-details {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

@media (max-width: 768px) {
    .overview-table,
    .overview-table thead,
    .overview-table tbody,
    .overview-table tr,
    .overview-table th,
    .overview-table td {
        display: block;
        width: 100%;
    }

    .overview-table thead {
        display: none;
    }

    .overview-table td {
        padding: 14px 15px;
    }

    .overview-table tbody tr {
        border-bottom: 1px solid #edf3f7;
    }

    .overview-table tbody tr:last-child {
        border-bottom: none;
    }

    .overview-topic-name {
        margin-bottom: 2px;
    }
}

.overview-note {
    margin-top: 16px;
    font-size: 13px;
    line-height: 1.65;
    color: #5f7287;
    background: #f7fbfe;
    border: 1px solid #dbe7f0;
    border-radius: 16px;
    padding: 12px 14px;
}

.subtle-note {
    background: #f7fbfe;
    border: 1px dashed #c6d8e7;
    border-radius: 16px;
    padding: 10px 12px;
    font-size: 12.5px;
    color: #6b7d92;
    margin-top: 8px;
}

.topic-panel {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
    box-shadow: none;
}

.section-kicker {
    display: none;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 6px 12px;
    background: #edf7ff;
    border: 1px solid #cfe2f2;
    color: #0f5d93;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.active-question {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
    margin: 0 0 10px 0;
    box-shadow: none;
}

.active-question .label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #7d92a7;
    margin-bottom: 6px;
}

.active-question .text {
    font-family: 'Manrope', sans-serif;
    font-size: 18px;
    line-height: 1.4;
    font-weight: 700;
    color: #153652;
}

.reply-shell {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
    box-shadow: none;
    margin-top: 4px;
}

.composer-shell {
    background: rgba(255,255,255,0.92);
    border: 1px solid #d9e4ed;
    border-radius: 26px;
    padding: 12px;
    box-shadow: 0 18px 36px rgba(23, 50, 74, 0.08);
    backdrop-filter: blur(10px);
}

.composer-shell.compact {
    padding: 12px;
}


.composer-row {
    display: flex;
    align-items: flex-end;
    gap: 10px;
}

[data-testid="stAudioInput"] {
    background: transparent;
    border: none;
    border-radius: 999px;
    padding: 0;
    margin: 0;
    box-shadow: none;
}

.stTextInput input {
    border-radius: 999px !important;
}

.stTextArea textarea {
    border-radius: 18px !important;
}

[data-testid="stAudioInput"] audio {
    border-radius: 14px;
}

.composer-shell div[data-testid="stButton"] > button {
    width: 100% !important;
    min-width: 0 !important;
    padding: 0.72rem 0.95rem !important;
    border-radius: 16px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 18px rgba(23, 50, 74, 0.06) !important;
    background: linear-gradient(180deg, #ffffff 0%, #f7fbfe 100%) !important;
    border: 1px solid #d9e4ed !important;
}

.composer-shell div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    border-color: #9fc1dd !important;
    color: #10375a !important;
}

.composer-shell div[data-baseweb="select"] > div {
    border-radius: 16px !important;
    min-height: 46px !important;
    height: 46px !important;
    background: #f7fbfe !important;
    border: 1px solid #d7e4ee !important;
    display: flex !important;
    align-items: center !important;
}

.composer-shell [data-testid="stTextInput"] {
    margin-bottom: 0 !important;
}

.composer-shell [data-testid="stTextInput"] input {
    min-height: 52px !important;
    height: 52px !important;
    background: #f9fcff !important;
    border: 1px solid #d6e4ef !important;
    padding-left: 16px !important;
}

.composer-shell [data-testid="stSelectbox"] {
    margin-bottom: 0 !important;
}

.composer-shell [data-testid="stAudioInput"] {
    background: transparent;
    border: none;
    border-radius: 16px;
    min-height: 46px;
    width: 100%;
    min-width: 100%;
    max-width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    margin: 10px 0 0 0;
    box-shadow: none !important;
}

.composer-shell [data-testid="stAudioInput"] > div {
    width: 100%;
    min-width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 0 !important;
    margin: 0 !important;
}

.composer-shell [data-testid="stAudioInput"] button {
    border-radius: 16px !important;
    width: 100% !important;
    height: 48px !important;
    min-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 1px solid #d7e4ee !important;
    background: linear-gradient(180deg, #ffffff 0%, #f5f9fd 100%) !important;
    box-shadow: 0 8px 18px rgba(23, 50, 74, 0.08) !important;
}

@media (max-width: 768px) {
    .chat-shell {
        border-radius: 22px;
    }

    .chat-shell-header {
        padding: 9px 10px 9px 10px;
        align-items: flex-start;
        flex-direction: column;
    }

    .chat-history {
        padding: 12px 10px 8px 10px;
        min-height: 180px;
    }

    .composer-wrap {
        padding: 0 10px 10px 10px;
    }

    .chat-entry {
        max-width: 92%;
    }
}

.composer-shell [data-testid="stAudioInput"] button::before {
    display: none !important;
}

.composer-shell [data-testid="stAudioInput"] button:hover {
    border-color: #bed4e7 !important;
    background: #ffffff !important;
    box-shadow: 0 10px 20px rgba(15, 108, 189, 0.10) !important;
}

.composer-shell [data-testid="stAudioInput"] button svg {
    width: 17px !important;
    height: 17px !important;
    color: #0f6cbd !important;
}


[data-testid="stProgressBar"] > div {
    border-radius: 999px !important;
    background: rgba(15,108,189,0.12) !important;
}
[data-testid="stProgressBar"] div[role="progressbar"] {
    background: linear-gradient(90deg, #0d9488 0%, #0f6cbd 100%) !important;
}
</style>
""", unsafe_allow_html=True)




def render_memory_banner(message: str):
    st.markdown(f'<div class="memory-banner">📋 {message}</div>', unsafe_allow_html=True)


def render_section_kicker(text: str):
    st.markdown(f'<div class="section-kicker">{_html.escape(text)}</div>', unsafe_allow_html=True)


def render_active_question(question: str, label: str = "Current question"):
    st.markdown(
        '<div class="active-question">'
        f'<div class="label">{_html.escape(label)}</div>'
        f'<div class="text">{_html.escape(question)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


CONVERSATIONAL_QUESTION_BANK = {
    "has_pain": "Are you having any pain today?",
    "throat_timing": "Is that throat pain there all the time, or mainly when you swallow or eat?",
    "tongue_type": "Does it feel like a sore on your tongue, or more like general tongue pain?",
    "pain_medications": "What are you taking for pain right now?",
    "taking_as_prescribed": "Have you been able to take your medications the way they were prescribed?",
    "eating_ability": "How has eating been going since your last visit?",
    "swallowing_difficulty": "Any trouble swallowing food, liquids, or pills?",
    "mouth_sores": "Have you noticed any mouth sores, ulcers, or white patches lately?",
    "dry_mouth": "Has dry mouth been bothering you?",
    "fatigue": "Have you been feeling more tired or weaker than usual?",
    "activity_level": "How are your usual day-to-day activities going right now?",
    "emotional_state": "How have you been feeling emotionally lately?",
}


def _dynamic_step_text(topic_key: Optional[str], step: dict, state: Optional[dict] = None) -> str:
    question_text = CONVERSATIONAL_QUESTION_BANK.get(step["id"], step["text"])
    if not state:
        return question_text

    prior_topic_data = st.session_state.get("last_checkin", {}).get(topic_key or "", {})
    if step["id"] == "weight_impact":
        prior_weight = prior_topic_data.get("weight")
        current_weight = state.get("data", {}).get("weight")
        if prior_weight not in (None, "") and current_weight not in (None, ""):
            question_text = (
                f"Last visit your weight was {prior_weight} pounds, and today you entered "
                f"{current_weight} pounds. Has that weight change been affecting how you feel or your energy levels?"
            )

    prior_value = prior_topic_data.get(step["id"])
    prior_text = _format_prior_answer_for_prompt(prior_value)
    if prior_text:
        if step["id"] == "weight":
            question_text = f"Last visit your weight was {prior_text} pounds. {question_text}"
        else:
            question_text = f"Last visit you reported {prior_text}. {question_text}"

    raw_answers = state.get("raw_answers", {})
    prev_step = _previous_step_in_flow(topic_key or "", step.get("id", ""))
    prev_raw = str(raw_answers.get(prev_step["id"], "")).strip() if prev_step else ""
    prev_phrase = _contextualize_raw_phrase(prev_raw)
    schema = STEP_SCHEMAS.get(step.get("id"), {})

    if schema.get("components") and prev_phrase:
        first_component = next(iter(schema["components"].values()))
        detector_name = first_component.get("detector")
        if detector_name == "helping":
            return first_component.get("question", "Has that been helping at all?")
        if detector_name == "specific_type":
            return first_component.get("question", question_text)

    if topic_key == "pain" and step.get("id") == "med_adherence_issue":
        prior = _norm_text(str(raw_answers.get("taking_as_prescribed", "")))
        if any(token in prior for token in {"forget", "forgot", "late", "on time", "timing", "schedule"}):
            return "Is the main issue remembering to take them on time, or is something else getting in the way?"
        if any(token in prior for token in {"side effect", "makes me sick", "too sleepy", "drowsy", "nausea"}):
            return "Are the side effects the main reason it has been hard to take them, or is something else also part of it?"
        if prior:
            return "What has been getting in the way of taking them the way you planned?"

    lower = _norm_text(question_text)
    if prev_phrase:
        if "what is stopping you" in lower:
            return f"You mentioned {prev_phrase}. What feels like the biggest reason that's limiting you?"
        if "what s making it hard" in lower or "what's making it hard" in lower:
            return f"You mentioned {prev_phrase}. What's making that hardest right now?"
        if "what activities are most difficult" in lower:
            return f"From what you shared, what daily activities feel hardest right now?"
        if "what kind of support would be most helpful" in lower:
            return "What kind of help would feel most useful for you right now?"
        if lower.startswith("is it helping"):
            return "Has that been helping at all?"
        if "where is the skin issue located" in lower:
            return "Where on your body are you noticing that skin problem?"
        if "who is supporting you" in lower:
            return "Who has been helping support you between visits?"
        if "what type are you using" in lower:
            return "What kind have you been using?"
        if "what is making it difficult to take your medications" in lower:
            return "What has been making it hardest to take them regularly?"
        if "is it painful to swallow, or just mechanically difficult" in lower:
            return "Does swallowing feel painful, or does it feel like things just don't go down well?"

    return question_text


def _step_prompt_text(step: dict, topic_key: Optional[str] = None, state: Optional[dict] = None) -> str:
    question_text = _dynamic_step_text(topic_key, step, state)
    if step.get("type") == "options" and step.get("id") != "med_adherence_issue":
        question_text += " (Choose an option below, or answer in your own words if that fits better.)"
    return question_text


def _append_assistant_message(state: dict, text: str):
    text = (text or "").strip()
    if not text:
        return
    if state["chat"] and state["chat"][-1]["role"] == "assistant" and state["chat"][-1]["content"].strip() == text:
        return
    state["chat"].append({"role": "assistant", "content": text})


def render_chat_bubble(role: str, content: str):
    safe = _html.escape(content or "").replace("\n", "<br>")
    role_cls = "user" if role == "user" else "assistant"
    role_label = "You" if role == "user" else "Care Assistant"
    avatar_label = "Y" if role == "user" else "I"
    timestamp = datetime.now().strftime("%H:%M")
    st.markdown(
        f'<div class="chat-row {role_cls}">'
        f'  <div class="chat-avatar">{avatar_label}</div>'
        f'  <div class="chat-entry">'
        f'    <div class="chat-meta"><div class="chat-role">{role_label}</div><div class="chat-time">{timestamp}</div></div>'
        f'    <div class="chat-bubble">{safe}</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# SECRETS / OPENAI
# ══════════════════════════════════════════════════════════════════

def _secret(*keys, default=None):
    for k in keys:
        if k in st.secrets:
            return st.secrets[k]
    return default


OPENAI_API_KEY = _secret("openai_api_key", "OPENAI_API_KEY", "openai_key")
openai_client: Optional[OpenAI] = None
_openai_error: Optional[str] = None

if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        _openai_error = str(e)
else:
    _openai_error = "OpenAI API key not configured."


# ══════════════════════════════════════════════════════════════════
# GOOGLE SHEETS
# ══════════════════════════════════════════════════════════════════

_sheet = None
_sheet_error: Optional[str] = None


def _init_sheets():
    global _sheet, _sheet_error
    if _sheet is not None or _sheet_error is not None:
        return
    try:
        creds = Credentials.from_service_account_info(
            _secret("gcp_service_account"),
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        book = gspread.authorize(creds).open_by_key(_secret("gsheet_id"))
        try:
            ws = book.worksheet("ChatReport")
        except Exception:
            ws = book.add_worksheet(title="ChatReport", rows=2000, cols=5)
            ws.append_row(["timestamp", "name", "all_data_json", "report"])
        _sheet = ws
    except Exception as e:
        _sheet_error = str(e)


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
        _sheet.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            name,
            json.dumps(all_data, ensure_ascii=False),
            report,
        ])
        return True
    except Exception as e:
        st.error(f"Failed to save to Google Sheets: {e}")
        return False


# ══════════════════════════════════════════════════════════════════
# VOICE / WHISPER
# ══════════════════════════════════════════════════════════════════

def _transcribe(audio_bytes: bytes) -> str:
    if not openai_client:
        return ""
    try:
        buf = io.BytesIO(audio_bytes)
        buf.name = "audio.wav"
        return openai_client.audio.transcriptions.create(
            model="whisper-1", file=buf
        ).text.strip()
    except Exception:
        return ""


def voice_widget(key_suffix: str, label: str = "Speak your answer") -> Optional[str]:
    """Renders voice recorder. Returns transcript string if new audio was processed."""
    transcript_key = f"_vt_{key_suffix}"
    hash_key = f"_vh_{key_suffix}"
    if hash_key not in st.session_state:
        st.session_state[hash_key] = None

    audio = st.audio_input(label, key=f"_vrec_{key_suffix}", label_visibility="collapsed")
    if not audio:
        return st.session_state.get(transcript_key)

    try:
        ab = audio.getvalue()
    except Exception:
        return st.session_state.get(transcript_key)

    if not ab:
        return st.session_state.get(transcript_key)

    ah = hashlib.sha1(ab).hexdigest()
    if ah == st.session_state[hash_key]:
        return st.session_state.get(transcript_key)

    st.session_state[hash_key] = ah
    with st.spinner("Transcribing…"):
        text = _transcribe(ab)

    if text:
        st.session_state[transcript_key] = text
        st.rerun()

    return st.session_state.get(transcript_key)


# ══════════════════════════════════════════════════════════════════
# TOPIC & FLOW DEFINITIONS
# ══════════════════════════════════════════════════════════════════

# Each entry: (display_label, internal_key)
TOPICS = [
    ("🩹 Pain & Medications",   "pain"),
    ("🍽️  Nutrition & Fluids",   "nutrition"),
    ("👄 Oral Symptoms",         "oral"),
    ("🤢 GI Symptoms",           "gi"),
    ("😴 Fatigue & Sleep",       "fatigue"),
    ("🚶 Activity Level",        "activity"),
    ("🧠 Mood",                  "mood"),
    ("💊 Other Symptoms",        "other"),
]

TOPIC_INTROS = {
    "pain":      "Let's talk about any pain you've been having, what you're taking for it, and whether that regimen is helping.",
    "nutrition": "I'd like to ask about your eating, drinking, and weight.",
    "oral":      "Let's go over any mouth and throat symptoms like sticky mucus, thrush, dryness, and what you're using to manage them.",
    "gi":        "I'll ask about nausea, vomiting, diarrhea, constipation, and how you're managing those symptoms.",
    "fatigue":   "Let's discuss how your energy and sleep have been.",
    "activity":  "Tell me about how your daily activities have been going.",
    "mood":      "This section covers how you've been feeling emotionally and your support system.",
    "other":     "Finally, let's cover any other symptoms — breathing, skin, hearing, and more.",
}

TOPIC_SCENARIO_PASSES = {
    "pain": {
        "title": "Breakthrough throat pain with swallowing",
        "why_it_matters": "Shows free-text interpretation, targeted follow-up, and escalation when pain is worsening.",
        "conversation": [
            ("Assistant", "Do you have any pain today?"),
            ("Patient", "Yes, mostly in my throat when I swallow."),
            ("Assistant", "On a scale of 0-10, how bad is the throat pain at its worst?"),
            ("Patient", "About an 8. It's definitely worse than last week."),
            ("Assistant", "What pain medications are you using right now?"),
            ("Patient", "Oxycodone and gabapentin."),
            ("Assistant", "Are you taking them as prescribed, and are they helping enough?"),
            ("Patient", "I'm taking them, but the oxycodone only helps for a couple of hours."),
        ],
        "agent_pass": [
            "Answer Interpreter: maps 'mostly in my throat when I swallow' to the throat branch without forcing the patient to use menu wording.",
            "Prior Comparison: flags a likely worsening trend when the patient says the pain is worse than last week.",
            "Urgency Agent: treats 8/10 pain with incomplete relief as a higher-priority symptom for same-visit review.",
            "Doctor-Relevance Agent: confirms the note is clinically useful because it captures location, severity, medications, and response.",
            "Next-Move Agent: asks a focused follow-up about medication timing or swallowing impact instead of repeating the pain question.",
        ],
    },
    "nutrition": {
        "title": "Eating less, weight loss, and dehydration risk",
        "why_it_matters": "Shows nutrition decline with swallowing barriers and fluid support needs.",
        "conversation": [
            ("Assistant", "How has eating been going since your last visit?"),
            ("Patient", "I'm eating less than usual. Mostly soft foods and shakes."),
            ("Assistant", "What has your weight been recently?"),
            ("Patient", "I was 156 last visit and I'm 149 now."),
            ("Assistant", "Are you having trouble swallowing?"),
            ("Patient", "Yes, solids feel stuck and it hurts."),
            ("Assistant", "Are you getting any feeding tube support or IV fluids?"),
            ("Patient", "No feeding tube, but I had IV fluids twice this week because I got dehydrated."),
        ],
        "agent_pass": [
            "Answer Interpreter: preserves the patient's natural description while still mapping to structured fields like 'eating less than usual' and swallowing difficulty.",
            "Urgency Agent: marks weight loss plus dehydration support as clinically important even if the patient sounds calm.",
            "Doctor-Relevance Agent: recognizes that oral intake, weight change, and hydration support together are enough to surface in the report.",
            "Next-Move Agent: can ask whether pain medicine timing is affecting meals, which matches the existing nutrition logic.",
        ],
    },
    "oral": {
        "title": "Dry mouth, sores, and thick mucus",
        "why_it_matters": "Shows how the bot separates several oral symptoms without making the patient repeat themselves.",
        "conversation": [
            ("Assistant", "Do you have any mouth sores, ulcers, or white patches right now?"),
            ("Patient", "Yes, I have a sore on the side of my tongue."),
            ("Assistant", "Is it affecting your ability to eat or drink?"),
            ("Patient", "A little. Spicy food really burns."),
            ("Assistant", "Are you having dryness in your mouth?"),
            ("Patient", "Yes, all day, and it's waking me up at night."),
            ("Assistant", "Are you having problems with thick mucus in your throat too?"),
            ("Patient", "Yes, especially in the morning. I've been using salt and baking soda rinses and they help some."),
        ],
        "agent_pass": [
            "Answer Interpreter: captures tongue soreness, dry mouth timing, and mucus management from natural phrasing.",
            "Sentiment Agent: detects frustration and symptom burden even without explicit distress language.",
            "Doctor-Relevance Agent: keeps the detail because the conversation covers symptom type, location, daily impact, and what the patient is already trying.",
            "Next-Move Agent: avoids redundant questions if the patient already volunteered that rinses help some.",
        ],
    },
    "gi": {
        "title": "Nausea with partial medication relief",
        "why_it_matters": "Shows targeted follow-up when a free-text answer contains some, but not all, clinically needed details.",
        "conversation": [
            ("Assistant", "Have you had any nausea, vomiting, or diarrhea since your last visit?"),
            ("Patient", "Mostly nausea, and I threw up once yesterday."),
            ("Assistant", "How often are you feeling nauseated?"),
            ("Patient", "A few times a day, usually after I try to eat."),
            ("Assistant", "What are you using for nausea, and is it helping?"),
            ("Patient", "Zofran twice a day. It helps a little but not enough."),
            ("Assistant", "How often are you vomiting and how much?"),
            ("Patient", "Just once yesterday, small amount."),
        ],
        "agent_pass": [
            "Answer Interpreter: correctly splits the patient's first answer into nausea and vomiting rather than forcing one label.",
            "Doctor-Relevance Agent: notices when management is present but relief is incomplete, which makes the symptom more actionable.",
            "Next-Move Agent: asks only for the missing vomiting frequency and amount because the nausea management answer was already sufficient.",
        ],
    },
    "fatigue": {
        "title": "Fatigue affecting function and sleep",
        "why_it_matters": "Shows how symptom burden can span weakness, daytime function, and medication-related drowsiness.",
        "conversation": [
            ("Assistant", "Are you feeling more tired or weak than usual?"),
            ("Patient", "Yes, very tired. My legs also feel weak."),
            ("Assistant", "Is the fatigue affecting your daily activities?"),
            ("Patient", "Yes, I need help showering and I rest most of the day."),
            ("Assistant", "Are your pain medications making you feel drowsy?"),
            ("Patient", "Sometimes, especially after the afternoon dose."),
            ("Assistant", "Are you able to sleep through the night?"),
            ("Patient", "No, I wake up because of dry mouth and coughing."),
        ],
        "agent_pass": [
            "Answer Interpreter: handles mixed fatigue plus focal weakness in one reply.",
            "Urgency Agent: increases concern when weakness is affecting basic daily function.",
            "Doctor-Relevance Agent: links drowsiness, poor sleep, and daytime impairment into a clinically coherent summary.",
            "Next-Move Agent: asks what is waking the patient at night, which lines up with the sleep follow-up rule.",
        ],
    },
    "activity": {
        "title": "Reduced activity from pain and fatigue",
        "why_it_matters": "Shows a functional-status conversation rather than a symptom-only conversation.",
        "conversation": [
            ("Assistant", "How is your daily life? Are you able to do your usual activities?"),
            ("Patient", "I'm doing less than usual."),
            ("Assistant", "What activities are most difficult right now?"),
            ("Patient", "Cooking, grocery shopping, and climbing stairs."),
            ("Assistant", "Is the difficulty mainly due to pain, fatigue, or something else?"),
            ("Patient", "Both. I get tired fast and my throat pain makes meals a project."),
        ],
        "agent_pass": [
            "Answer Interpreter: maps the patient's free-text answer to the activity branch and keeps the concrete examples.",
            "Prior Comparison: can highlight a decline if the patient was previously doing normal activities.",
            "Doctor-Relevance Agent: preserves specific functional losses because they are more useful than a generic 'doing worse' label.",
        ],
    },
    "mood": {
        "title": "Anxiety, low mood, and limited support",
        "why_it_matters": "Shows empathic handling of emotional symptoms while still collecting actionable support needs.",
        "conversation": [
            ("Assistant", "How are you feeling emotionally? Are you feeling anxious or worried about anything?"),
            ("Patient", "I've been pretty overwhelmed and worried about treatment not working."),
            ("Assistant", "Is anxiety or worry affecting your sleep, eating, or daily activities?"),
            ("Patient", "Yes, mostly my sleep and appetite."),
            ("Assistant", "Have you been feeling down or depressed?"),
            ("Patient", "Yes, on and off most days."),
            ("Assistant", "Do you feel you have enough support between visits?"),
            ("Patient", "Not really. My daughter works and I'm alone a lot."),
        ],
        "agent_pass": [
            "Sentiment Agent: picks up emotional strain and possible disengagement risk from the patient's wording.",
            "Doctor-Relevance Agent: treats low support plus depressed mood as a meaningful care-coordination issue.",
            "Next-Move Agent: follows with a supportive question about what kind of help would be most useful right now.",
        ],
    },
    "other": {
        "title": "Mixed red-flag symptoms outside the core tracks",
        "why_it_matters": "Shows how the catch-all topic still supports fast safety escalation.",
        "conversation": [
            ("Assistant", "Are you having any difficulty breathing or shortness of breath?"),
            ("Patient", "A little short of breath when I walk across the room."),
            ("Assistant", "Have you had any fever or chills recently?"),
            ("Patient", "Yes, I had chills last night and my temperature was 100.8."),
            ("Assistant", "Have you been feeling dizzy or lightheaded?"),
            ("Patient", "Yes, mostly when I stand up, and I almost fell this morning."),
            ("Assistant", "Have you had any skin problems like irritation, wounds, or redness?"),
            ("Patient", "The skin on my neck is raw from radiation and looks worse this week."),
        ],
        "agent_pass": [
            "Urgency Agent: identifies fever, near-fall, and exertional shortness of breath as same-day review signals.",
            "Doctor-Relevance Agent: keeps the temperature, timing, near-fall, and skin progression because those specifics drive triage.",
            "Orchestrator: prioritizes the safety-sensitive follow-up path over lower-priority symptom details.",
        ],
    },
}

TOPIC_MAIN_RULES = {
    "pain":      ["Main2", "Main3", "Main12", "Main38"],
    "nutrition": ["Main5", "Main6", "Main8", "Main25", "Main26", "Main27", "Main34"],
    "oral":      ["Main4", "Main7", "Main10", "Main24", "Main33"],
    "gi":        ["Main11", "Main18"],
    "fatigue":   ["Main13", "Main14"],
    "activity":  ["Main30"],
    "mood":      ["Main15", "Main35", "Main39"],
    "other":     ["Main9", "Main16", "Main17", "Main19", "Main20", "Main21", "Main22", "Main23", "Main36", "Main37"],
}


def _q(id, text, type="options", opts=None, when=None,
        placeholder="Please describe...", min_v=0, max_v=10, default_v=0):
    """Helper to build a question step dict."""
    return {
        "id": id, "text": text, "type": type,
        "opts": opts or [], "when": when,
        "placeholder": placeholder,
        "min_v": min_v, "max_v": max_v, "default_v": default_v,
    }


def _safe_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ── PAIN & MEDICATIONS (Main 2, 3, 12, 38) ────────────────────────
FLOW_PAIN = [
    # Main 2
    _q("has_pain", "Do you have any pain today?", opts=["Yes", "No"]),

    # Main 3 — location
    _q("pain_location", "Where are you feeling the pain?",
       opts=["Throat", "Tongue", "Somewhere else"],
       when=lambda d: d.get("has_pain") == "Yes"),

    # ── Throat branch ──
    _q("throat_timing",
       "Is the throat pain there all the time, or only when you swallow or eat?",
       opts=["All the time", "Only when swallowing", "Only when eating",
             "Both swallowing and eating"],
       when=lambda d: d.get("pain_location") == "Throat"),

    _q("throat_severity",
       "On a scale of 0–10, how bad is the throat pain at its worst?",
       type="number", min_v=0, max_v=10, default_v=5,
       when=lambda d: d.get("pain_location") == "Throat"),

    _q("throat_med_helps",
       "Are you taking pain medication for this? Is it helping?",
       opts=["Yes, it helps", "Yes, but it's not enough", "No, I'm not taking anything"],
       when=lambda d: (d.get("pain_location") == "Throat"
                       and _safe_int(d.get("throat_severity", 0)) > 4)),

    # ── Tongue branch ──
    _q("tongue_type",
       "Is it a sore or ulcer on the tongue, or a general painful feeling?",
       opts=["There's a sore/ulcer", "Just pain, no visible sore"],
       when=lambda d: d.get("pain_location") == "Tongue"),

    _q("tongue_spot",
       "Is the pain in one specific spot, or does it spread?",
       opts=["One spot", "Spreads across tongue", "Whole mouth"],
       when=lambda d: d.get("pain_location") == "Tongue"),

    _q("tongue_severity",
       "On a scale of 0–10, how bad is the tongue pain at its worst?",
       type="number", min_v=0, max_v=10, default_v=5,
       when=lambda d: d.get("pain_location") == "Tongue"),

    # ── Somewhere else branch ──
    _q("other_pain_desc",
       "Which body part is hurting?",
       type="free_text", placeholder="e.g., near my jaw and ear…",
       when=lambda d: d.get("pain_location") == "Somewhere else"),

    _q("other_pain_severity",
       "How bad is that pain at its worst on a 0 to 10 scale?",
       type="number", min_v=0, max_v=10, default_v=5,
       when=lambda d: d.get("pain_location") == "Somewhere else"),

    _q("ear_pain", "Do you have ear pain or hearing changes?",
       opts=["Yes", "No"],
       when=lambda d: (
           d.get("pain_location") == "Somewhere else"
           and _needs_head_neck_followup(d.get("other_pain_desc", ""))
       )),

    _q("jaw_swelling", "Do you feel any swelling near your jaw?",
       opts=["Yes", "No"],
       when=lambda d: (
           d.get("pain_location") == "Somewhere else"
           and _needs_head_neck_followup(d.get("other_pain_desc", ""))
       )),

    _q("pain_with_chewing",
       "Does the pain worsen when chewing or opening your mouth?",
       opts=["Yes", "No"],
       when=lambda d: (
           d.get("pain_location") == "Somewhere else"
           and _needs_head_neck_followup(d.get("other_pain_desc", ""))
       )),

    _q("pain_start",                        # ← added (Main 3, Somewhere else branch)
       "When did this pain start?",
       type="free_text",
       placeholder="e.g., about a week ago, since I started radiation…",
       when=lambda d: d.get("pain_location") == "Somewhere else"),

    # Main 12 — Medications
    _q("pain_medications",
       "Which medications are you currently taking for pain?",
       type="multi_select",
       opts=["Gabapentin", "Oxycodone", "Butrans patch", "Other", "No pain medication"]),

    _q("med_dose_freq",
       "How often are you taking your pain medication, and at what dose?",
       type="free_text", placeholder="e.g., Oxycodone 5mg every 6 hours…",
       when=lambda d: (bool(d.get("pain_medications"))
                       and "No pain medication" not in (d.get("pain_medications") or []))),

    # Main 38 — Adherence
    _q("taking_as_prescribed",
       "Are you taking your medications as prescribed?",
       opts=["Yes", "No"]),

    _q("med_adherence_issue",
       "What is making it difficult to take your medications?",
       opts=["Side effects", "Schedule", "Access issues", "Other"],
       when=lambda d: d.get("taking_as_prescribed") == "No"),

    _q("med_side_effects",
       "Are you experiencing any side effects from your medications?",
       opts=["Yes", "No"],
       when=lambda d: d.get("taking_as_prescribed") == "Yes"),
]

# ── NUTRITION & FLUIDS (Main 5, 6, 8, 25, 26, 27, 34) ─────────────
FLOW_NUTRITION = [
    # Main 5 — Eating ability
    _q("eating_ability",
       "How has your eating been since your last visit?",
       opts=["Eating normally — no problems",
             "Eating less than usual, but managing",
             "Struggling — only liquids or very little",
             "Not eating — using a feeding tube only"]),

    # Branch: Eating less
    _q("fluid_intake_managing",
       "Are you drinking enough fluids throughout the day — water, shakes, or other drinks?",
       opts=["Yes, drinking well", "A little less than usual", "Struggling to drink enough"],
       when=lambda d: d.get("eating_ability") == "Eating less than usual, but managing"),

    _q("food_type",
       "What are you able to eat right now?",
       opts=["Mostly normal food", "Soft foods only (yogurt, soup, pudding)",
             "Mix of soft and liquid", "Mainly liquids"],
       when=lambda d: d.get("eating_ability") == "Eating less than usual, but managing"),

    # Branch: Struggling
    _q("nutritional_shakes",
       "How many nutritional shakes or Boost/Ensure drinks are you having per day?",
       opts=["None", "1–2", "3–4", "More than 4"],
       when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),

    _q("eating_barrier",
       "What is stopping you from eating more?",
       opts=["Pain when eating/swallowing", "Feel full very quickly",
             "No appetite", "Nausea", "Too tired to prepare food"],
       when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),

    _q("fluid_struggling",
       "Are you drinking enough fluids — water, juice, or anything?",
       opts=["Yes, drinking well", "A little", "Very little, hard to drink"],
       when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),

    _q("fluid_barrier",
       "What's making it hard to drink?",
       opts=["Pain when swallowing", "Dry mouth", "Nausea", "Just not thirsty"],
       when=lambda d: (d.get("eating_ability") == "Struggling — only liquids or very little"
                       and d.get("fluid_struggling") in ["A little", "Very little, hard to drink"])),

    _q("pain_med_timing",
       "Are you timing your pain medication before meals to make eating easier?",
       opts=["Yes, it helps", "I try, but it's not enough",
             "No, I didn't know to do this", "No, I don't take pain medication"],
       when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),

    # Branch: Tube only
    _q("tube_issues",
       "Is the tube feeding going well — no blockages, leaks, or discomfort around the site?",
       opts=["Working fine", "Some issues — leaking or blockage",
             "Discomfort/soreness around the tube"],
       when=lambda d: d.get("eating_ability") == "Not eating — using a feeding tube only"),

    _q("tube_oral_sips",
       "Are you still able to take any sips of water or liquids by mouth at all?",
       opts=["Yes, small amounts", "Very occasionally for comfort", "No, nothing by mouth"],
       when=lambda d: d.get("eating_ability") == "Not eating — using a feeding tube only"),

    # Main 6 — Weight
    _q("weight",
       "What has your weight been recently? (Enter in pounds)",
       type="number", min_v=50, max_v=500, default_v=150),

    _q("weight_impact",
       "Has any weight change been affecting how you feel or your energy levels?",
       opts=["Yes, I've noticed a difference", "Not really"]),

    # Main 8 — Swallowing
    _q("swallowing_difficulty",
       "Are you having any difficulty swallowing — liquids, food, or pills?",
       opts=["Yes", "No"]),

    _q("swallowing_type",
       "Is it painful to swallow, or just mechanically difficult?",
       opts=["Painful to swallow", "Mechanically difficult"],
       when=lambda d: d.get("swallowing_difficulty") == "Yes"),

    _q("choking_with_eating",
       "Do you cough or choke when you eat?",
       opts=["Yes", "No"],
       when=lambda d: d.get("swallowing_difficulty") == "Yes"),

    _q("swallowing_method",
       "Are you still able to swallow liquids by mouth, or is everything through a feeding tube?",
       opts=["I swallow by mouth", "Everything through the feeding tube"],
       when=lambda d: d.get("swallowing_difficulty") == "Yes"),

    # Main 25 — Choking/Coughing (standalone — separate from Main 8)
    _q("choking_coughing",
       "Are you having any difficulty with choking or coughing when eating or drinking?",
       opts=["Yes", "No"]),

    _q("choking_type",
       "Does it happen with liquids, solids, or both?",
       opts=["Liquids", "Solids", "Both"],
       when=lambda d: d.get("choking_coughing") == "Yes"),

    _q("choking_frequency",
       "Does it happen every time you eat, or only occasionally?",
       opts=["Every time", "Occasionally"],
       when=lambda d: d.get("choking_coughing") == "Yes"),

    _q("choking_pills",                     # ← added (Main 25)
       "Does it also happen when you take pills?",
       opts=["Yes", "No"],
       when=lambda d: d.get("choking_coughing") == "Yes"),

    # Main 26 — IV Fluids
    _q("iv_fluids",
       "Are you currently receiving IV fluids or hydration treatments?",
       opts=["Yes", "No"]),

    _q("iv_frequency",
       "How often are you receiving IV fluids?",
       type="free_text", placeholder="e.g., twice a week…",
       when=lambda d: d.get("iv_fluids") == "Yes"),

    _q("iv_helping",
       "Do you feel the IV fluids are helping?",
       opts=["Yes", "No"],
       when=lambda d: d.get("iv_fluids") == "Yes"),

    _q("iv_adjust",                         # ← added (Main 26)
       "Would you like to adjust the frequency of your hydration visits?",
       opts=["Yes", "No"],
       when=lambda d: d.get("iv_fluids") == "Yes"),

    _q("need_hydration",
       "Do you feel like you might need hydration support?",
       opts=["Yes", "No"],
       when=lambda d: d.get("iv_fluids") == "No"),

    # Main 27 — Feeding tube (for patients not already exclusively tube-fed)
    _q("feeding_tube",
       "Are you currently using a feeding tube?",
       opts=["Yes", "No"],
       when=lambda d: d.get("eating_ability") != "Not eating — using a feeding tube only"),

    _q("tube_status",
       "Is the feeding tube working well or are there issues?",
       opts=["Working well", "Leakage", "Blockage", "Discomfort"],
       when=lambda d: (d.get("feeding_tube") == "Yes"
                       and d.get("eating_ability") != "Not eating — using a feeding tube only")),

    _q("tube_oral",
       "Are you able to take anything by mouth at all?",
       opts=["Yes, some", "No, nothing by mouth"],
       when=lambda d: (d.get("feeding_tube") == "Yes"
                       and d.get("eating_ability") != "Not eating — using a feeding tube only")),

    # Main 34 — Taste
    _q("taste_changes",
       "Have you noticed any changes in your sense of taste?",
       opts=["Yes", "No"]),

    _q("taste_type",
       "Does food taste different, bland, or unpleasant?",
       opts=["Different", "Bland", "Unpleasant"],
       when=lambda d: d.get("taste_changes") == "Yes"),

    _q("taste_eating_impact",
       "Is the taste change affecting your ability to eat?",
       opts=["Yes", "No"],
       when=lambda d: d.get("taste_changes") == "Yes"),
]

# ── ORAL SYMPTOMS (Main 4, 7, 10, 24, 33) ─────────────────────────
FLOW_ORAL = [
    # Main 4 — Mouth sores / thrush
    _q("mouth_sores",
       "Do you have any mouth sores, ulcers, or white patches/thrush right now?",
       opts=["Yes", "No"]),

    _q("sore_new_or_old",
       "Is this new since your last visit, or have you had it for a while?",
       opts=["New", "Not sure", "Same one as before"],
       when=lambda d: d.get("mouth_sores") == "Yes"),

    _q("sore_location",
       "Where exactly is it?",
       opts=["Inside the mouth/cheek", "On the tongue", "Back of the throat",
             "Gums/lips", "Multiple spots"],
       when=lambda d: (d.get("mouth_sores") == "Yes"
                       and d.get("sore_new_or_old") in ["New", "Not sure"])),

    _q("sore_pain_impact",
       "Is the sore painful? Is it affecting your ability to eat or drink?",
       opts=["No pain, just noticed it", "A little, but manageable",
             "Yes, can't eat/drink comfortably"],
       when=lambda d: (d.get("mouth_sores") == "Yes"
                       and d.get("sore_new_or_old") in ["New", "Not sure"])),

    _q("magic_mouthwash",
       "Are you using anything for it, like magic mouthwash or thrush medicine? If yes, is it helping?",
       opts=["Yes, it helps", "Yes, but not enough",
             "No, I don't have it", "No, I don't use it"],
       when=lambda d: (d.get("mouth_sores") == "Yes"
                       and d.get("sore_new_or_old") in ["New", "Not sure"])),

    _q("sore_progression",
       "Is the sore getting better, staying the same, or getting worse?",
       opts=["Getting better", "About the same", "Getting worse", "Not sure"],
       when=lambda d: (d.get("mouth_sores") == "Yes"
                       and d.get("sore_new_or_old") == "Same one as before")),

    _q("sore_eating_impact_old",
       "Is it still preventing you from eating or drinking comfortably?",
       opts=["Yes", "A little", "No"],
       when=lambda d: (d.get("mouth_sores") == "Yes"
                       and d.get("sore_new_or_old") == "Same one as before"
                       and d.get("sore_progression") in ["About the same", "Getting worse"])),

    # Main 7 — Dry mouth
    _q("dry_mouth",
       "Are you experiencing any dryness in your mouth?",
       opts=["Yes", "No"]),

    _q("dry_mouth_timing",
       "Is the dryness worse at night or all day?",
       opts=["Worse at night", "All day"],
       when=lambda d: d.get("dry_mouth") == "Yes"),

    _q("dry_mouth_med",
       "Are you using any medication like Biotene or a saliva substitute?",
       opts=["Yes", "No"],
       when=lambda d: d.get("dry_mouth") == "Yes"),

    _q("dry_mouth_impact",
       "Is the dryness making it harder to eat, talk, or sleep?",
       opts=["Yes", "No"],
       when=lambda d: d.get("dry_mouth") == "Yes"),

    # Main 10 — Mucus / thick secretions
    _q("mucus_issues",
       "Are you having problems with mucus or thick secretions in your throat?",
       opts=["Yes", "No"]),

    _q("mucus_type",
       "Is the mucus thick and hard to clear, or more watery?",
       opts=["Thick", "More watery"],
       when=lambda d: d.get("mucus_issues") == "Yes"),

    _q("mucus_impact",
       "Is the mucus affecting your ability to swallow or sleep?",
       opts=["Yes", "No"],
       when=lambda d: d.get("mucus_issues") == "Yes"),

    _q("mucus_management",
       "Are you using anything to manage it — like Robitussin or saline rinses?",
       opts=["Yes", "No"],
       when=lambda d: d.get("mucus_issues") == "Yes"),

    # Main 24 — Teeth / Gums
    _q("teeth_gum_issues",
       "Are you having any problems with your teeth or gums?",
       opts=["Yes", "No"]),

    _q("teeth_issue_type",
       "Is there pain, bleeding, or sores with your teeth or gums?",
       opts=["Pain", "Bleeding", "Sores", "Multiple issues"],
       when=lambda d: d.get("teeth_gum_issues") == "Yes"),

    _q("brushing_difficult",
       "Is it making brushing difficult?",
       opts=["Yes", "No"],
       when=lambda d: d.get("teeth_gum_issues") == "Yes"),

    _q("avoiding_brushing",                 # ← added (Main 24)
       "Are you avoiding brushing because of the discomfort?",
       opts=["Yes", "No"],
       when=lambda d: d.get("teeth_gum_issues") == "Yes"),

    # Main 33 — Oral rinses
    _q("oral_rinse_use",
       "Are you using mouthwash or oral rinses regularly?",
       opts=["Yes", "No"]),

    _q("oral_rinse_type",
       "What type are you using?",
       type="free_text",
       placeholder="e.g., magic mouthwash, salt/baking soda rinse…",
       when=lambda d: d.get("oral_rinse_use") == "Yes"),

    _q("oral_rinse_helping",
       "Is it helping?",
       opts=["Yes", "No"],
       when=lambda d: d.get("oral_rinse_use") == "Yes"),

    _q("oral_rinse_open",
       "Would you be open to trying an oral rinse to help with symptoms?",
       opts=["Yes", "No"],
       when=lambda d: d.get("oral_rinse_use") == "No"),
]

# ── GI SYMPTOMS (Main 11, 18) ─────────────────────────────────────
FLOW_GI = [
    # Main 11 — Nausea / Vomiting / Diarrhea
    _q("nausea_vomiting",
       "Have you had any nausea, vomiting, or diarrhea since your last visit?",
       type="multi_select",
       opts=["Nausea", "Vomiting", "Diarrhea", "None of these"]),

    _q("nausea_frequency",
       "How often are you feeling nauseated?",
       type="free_text",
       placeholder="e.g., a few times a day, mostly in the mornings…",
       when=lambda d: "Nausea" in (d.get("nausea_vomiting") or [])),

    _q("nausea_management",
       "What are you using for nausea, and is it helping?",
       type="free_text",
       placeholder="e.g., Zofran twice a day and it helps a little…",
       when=lambda d: "Nausea" in (d.get("nausea_vomiting") or [])),

    _q("vomiting_frequency",
       "How often are you vomiting and how much?",
       type="free_text",
       placeholder="e.g., once or twice a day, small amounts…",
       when=lambda d: "Vomiting" in (d.get("nausea_vomiting") or [])),

    _q("vomiting_management",
       "What are you doing to manage the vomiting, and is it helping?",
       type="free_text",
       placeholder="e.g., anti-nausea medication, small sips, and it is helping some…",
       when=lambda d: "Vomiting" in (d.get("nausea_vomiting") or [])),

    _q("diarrhea_frequency",
       "How often are you having diarrhea?",
       type="free_text",
       placeholder="e.g., three loose stools a day…",
       when=lambda d: "Diarrhea" in (d.get("nausea_vomiting") or [])),

    _q("diarrhea_management",
       "Are you taking anything for the diarrhea, and is it helping?",
       type="free_text",
       placeholder="e.g., Imodium and it helps some…",
       when=lambda d: "Diarrhea" in (d.get("nausea_vomiting") or [])),

    # Main 18 — Constipation
    _q("constipation",
       "Have you had any constipation or trouble moving your bowels?",
       opts=["Yes", "No"]),

    _q("bowel_frequency",
       "How often are you having bowel movements?",
       type="free_text",
       placeholder="e.g., once every 3 days…",
       when=lambda d: d.get("constipation") == "Yes"),

    _q("constipation_meds",
       "Are you taking anything like Senna, Miralax, or other medications for constipation?",
       opts=["Yes", "No"],
       when=lambda d: d.get("constipation") == "Yes"),

    _q("bloating",
       "Are you feeling bloated or uncomfortable?",
       opts=["Yes", "No"],
       when=lambda d: d.get("constipation") == "Yes"),
]

# ── FATIGUE & SLEEP (Main 13, 14) ─────────────────────────────────
FLOW_FATIGUE = [
    # Main 13 — Fatigue / Weakness
    _q("fatigue",
       "Are you feeling more tired or weak than usual?",
       opts=["Yes", "No"]),

    _q("fatigue_type",
       "Is it a general tiredness, or weakness in specific parts of your body?",
       opts=["General tiredness", "Weakness in specific parts"],
       when=lambda d: d.get("fatigue") == "Yes"),

    _q("weakness_location",
       "In which parts of your body do you feel weakness?",
       type="free_text", placeholder="e.g., legs, arms…",
       when=lambda d: (d.get("fatigue") == "Yes"
                       and d.get("fatigue_type") == "Weakness in specific parts")),

    _q("fatigue_daily_impact",
       "Is the fatigue affecting your daily activities — getting dressed, moving around?",
       opts=["Yes", "No"],
       when=lambda d: d.get("fatigue") == "Yes"),

    # Main 14 — Drowsiness + Sleep
    _q("medication_drowsy",
       "Are your pain medications making you feel drowsy?",
       opts=["Yes", "No", "Sometimes"]),

    _q("sleep_quality",
       "Are you able to sleep through the night?",
       opts=["Yes", "No"]),

    _q("sleep_wake_reason",
       "Are you waking up at night due to pain, dry mouth, or coughing?",
       type="free_text",
       placeholder="e.g., pain wakes me up around 3am…",
       when=lambda d: d.get("sleep_quality") == "No"),

    _q("drowsy_schedule",                   # ← fixed: now conditional on sleep_quality, not drowsiness
       "Is drowsiness from medication affecting your normal wake/sleep schedule?",
       opts=["Yes", "No"],
       when=lambda d: d.get("sleep_quality") == "No"),
]

# ── ACTIVITY LEVEL (Main 30) ───────────────────────────────────────
FLOW_ACTIVITY = [
    _q("activity_level",
       "How is your daily life — are you able to do your usual activities?",
       opts=["Doing everything normally", "Doing less than usual",
             "Struggling with daily tasks"]),

    _q("difficult_activities",
       "What activities are most difficult right now?",
       type="free_text",
       placeholder="e.g., climbing stairs, cooking, getting dressed…",
       when=lambda d: d.get("activity_level") in
             ["Doing less than usual", "Struggling with daily tasks"]),

    _q("activity_limiting_factor",
       "Is the difficulty mainly due to pain, fatigue, or something else?",
       opts=["Pain", "Fatigue", "Both", "Something else"],
       when=lambda d: d.get("activity_level") in
             ["Doing less than usual", "Struggling with daily tasks"]),

    _q("activity_other_desc",
       "Can you tell me more about what's limiting your activities?",
       type="free_text", placeholder="e.g., balance issues, weakness…",
       when=lambda d: d.get("activity_limiting_factor") == "Something else"),
]

# ── MOOD (Main 15, 35, 39) ─────────────────────────────────────────
FLOW_MOOD = [
    # Main 15 — Emotional state / Anxiety
    _q("emotional_state",
       "How are you feeling emotionally? Are you feeling anxious or worried about anything?",
       type="free_text",
       placeholder="Please share how you've been feeling — there are no wrong answers…"),

    _q("anxiety_impact",
       "Is anxiety or worry affecting your sleep, eating, or daily activities?",
       opts=["Yes", "No", "A little"]),

    _q("social_support_quality",
       "Do you have people around you who you can talk to about how you're feeling?",
       opts=["Yes, I have good support", "Some support", "Not really"]),

    # Main 35 — Depression
    _q("feeling_down",
       "Have you been feeling down or depressed?",
       opts=["Yes", "No"]),

    _q("depression_frequency",
       "How often have you been feeling this way?",
       type="free_text",
       placeholder="e.g., most days, occasionally, mostly in the evenings…",
       when=lambda d: d.get("feeling_down") == "Yes"),

    _q("depression_daily_impact",
       "Is it affecting your daily activities or motivation?",
       opts=["Yes", "No"],
       when=lambda d: d.get("feeling_down") == "Yes"),

    # Main 39 — Support between visits
    _q("support_adequate",
       "Do you feel you have enough support between visits?",
       opts=["Yes", "No"]),

    _q("who_supports",
       "Who is supporting you — family, friends, or caregivers?",
       type="free_text",
       placeholder="e.g., my wife and daughter…",
       when=lambda d: d.get("support_adequate") == "Yes"),

    _q("needed_support",
       "What kind of support would be most helpful right now?",
       type="free_text",
       placeholder="e.g., emotional support, help with transportation, more info about treatment…",
       when=lambda d: d.get("support_adequate") == "No"),
]

# ── OTHER SYMPTOMS (Main 9, 16, 17, 19, 20, 21, 22, 23, 36, 37) ───
FLOW_OTHER = [
    # Main 9 — Breathing
    _q("breathing_issues",
       "Are you having any difficulty breathing or shortness of breath?",
       opts=["Yes", "No"]),

    _q("breathing_timing",
       "Is the breathing difficulty constant, or does it come on with activity?",
       opts=["It's constant", "It comes on with activity"],
       when=lambda d: d.get("breathing_issues") == "Yes"),

    _q("wheezing",
       "Are you wheezing or feeling like something is blocking your airway?",
       opts=["Yes", "No"],
       when=lambda d: d.get("breathing_issues") == "Yes"),

    # Main 16 — Hearing
    _q("hearing_changes",
       "Do you have any hearing problems or changes recently?",
       opts=["Yes", "No"]),

    _q("hearing_type",
       "Is it ringing in your ears, hearing loss, or both?",
       opts=["Ringing in ears", "Hearing loss", "Both"],
       when=lambda d: d.get("hearing_changes") == "Yes"),

    _q("hearing_constant",
       "Is it constant or does it come and go?",
       opts=["Constant", "Comes and goes"],
       when=lambda d: d.get("hearing_changes") == "Yes"),

    _q("hearing_worsening",
       "Has it gotten worse compared to your last visit?",
       opts=["Yes", "No"],
       when=lambda d: d.get("hearing_changes") == "Yes"),

    # Main 17 — Dizziness
    _q("dizziness",
       "Have you been feeling dizzy or lightheaded?",
       opts=["Yes", "No"]),

    _q("dizziness_timing",
       "Is it constant or only when you stand up or change position?",
       opts=["Constant", "Only when standing or changing position"],
       when=lambda d: d.get("dizziness") == "Yes"),

    _q("dizziness_worsening",               # ← added (Main 17)
       "Has the dizziness gotten worse recently?",
       opts=["Yes", "No"],
       when=lambda d: d.get("dizziness") == "Yes"),

    _q("falls",
       "Have you had any falls or felt like you might fall?",
       opts=["Yes", "No"],
       when=lambda d: d.get("dizziness") == "Yes"),

    # Main 19 — Numbness / Tingling
    _q("numbness",
       "Have you noticed any numbness or tingling in your hands or feet?",
       opts=["Yes", "No"]),

    _q("numbness_location",
       "Is it in your hands, feet, or both?",
       opts=["Hands", "Feet", "Both"],
       when=lambda d: d.get("numbness") == "Yes"),

    _q("numbness_new",
       "Is it new or getting worse?",
       opts=["New", "Getting worse", "Same as before"],
       when=lambda d: d.get("numbness") == "Yes"),

    _q("numbness_daily_impact",
       "Is it affecting your daily activities?",
       opts=["Yes", "No"],
       when=lambda d: d.get("numbness") == "Yes"),

    # Main 20 — Fever / Chills
    _q("fever_chills",
       "Have you had any fever or chills recently?",
       opts=["Yes", "No"]),

    _q("fever_start", "When did the fever or chills start?",
       type="free_text", placeholder="e.g., two days ago…",
       when=lambda d: d.get("fever_chills") == "Yes"),

    _q("fever_temp", "How high was the fever?",
       type="free_text", placeholder="e.g., 101.5°F…",
       when=lambda d: d.get("fever_chills") == "Yes"),

    _q("fever_other_symptoms",
       "Do you have any other symptoms like cough or signs of infection?",
       opts=["Yes", "No"],
       when=lambda d: d.get("fever_chills") == "Yes"),

    # Main 21 — Blood pressure
    _q("bp_monitoring",
       "Are you checking your blood pressure at home?",
       opts=["Yes", "No"]),

    _q("bp_reading", "What has your blood pressure been recently?",
       type="free_text", placeholder="e.g., 130/85…",
       when=lambda d: d.get("bp_monitoring") == "Yes"),

    _q("bp_dizziness",
       "Have you felt dizzy or lightheaded with blood pressure changes?",
       opts=["Yes", "No"],
       when=lambda d: d.get("bp_monitoring") == "Yes"),

    _q("bp_home_monitor",
       "Do you have a way to check your blood pressure at home?",
       opts=["Yes", "No"],
       when=lambda d: d.get("bp_monitoring") == "No"),

    # Main 22 — Skin
    _q("skin_issues",
       "Have you had any skin problems — like irritation, wounds, or redness?",
       opts=["Yes", "No"]),

    _q("skin_location", "Where is the skin issue located?",
       type="free_text", placeholder="e.g., neck, shoulder, near jaw…",
       when=lambda d: d.get("skin_issues") == "Yes"),

    _q("skin_start",                        # ← added (Main 22)
       "When did it start?",
       type="free_text",
       placeholder="e.g., about a week ago, at the start of radiation…",
       when=lambda d: d.get("skin_issues") == "Yes"),

    _q("skin_progression",
       "Is it getting better, worse, or staying the same?",
       opts=["Getting better", "About the same", "Getting worse"],
       when=lambda d: d.get("skin_issues") == "Yes"),

    _q("skin_drainage",
       "Any drainage, bleeding, or open areas?",
       opts=["Yes", "No"],
       when=lambda d: d.get("skin_issues") == "Yes"),

    # Main 23 — Voice / Hoarseness
    _q("voice_hoarseness",
       "How is your voice? Have you noticed any hoarseness or trouble speaking?",
       opts=["Yes, problems with my voice", "No, voice is fine"]),

    _q("voice_timing",
       "Is the hoarseness constant or only when you're talking?",
       opts=["Constant", "Only when talking"],
       when=lambda d: d.get("voice_hoarseness") == "Yes, problems with my voice"),

    _q("voice_progression",
       "Has your voice improved or worsened since your last visit?",
       opts=["Improved", "About the same", "Worse"],
       when=lambda d: d.get("voice_hoarseness") == "Yes, problems with my voice"),

    _q("voice_communication_impact",
       "Is it affecting your ability to communicate with others?",
       opts=["Yes", "No"],
       when=lambda d: d.get("voice_hoarseness") == "Yes, problems with my voice"),

    # Main 36 — Concentration / Memory
    _q("concentration",
       "Have you had trouble concentrating or remembering things?",
       opts=["Yes", "No"]),

    _q("concentration_new",
       "Is it new or ongoing?",
       opts=["New", "Ongoing"],
       when=lambda d: d.get("concentration") == "Yes"),

    _q("concentration_daily_impact",
       "Is it affecting your daily tasks?",
       opts=["Yes", "No"],
       when=lambda d: d.get("concentration") == "Yes"),

    # Main 37 — Sexual health
    _q("sexual_health",
       "Have you had any sexual health concerns or changes?",
       opts=["Yes", "Prefer not to say", "No"]),

    _q("sexual_discuss",
       "Would you like to discuss this further with your provider?",
       opts=["Yes", "No"],
       when=lambda d: d.get("sexual_health") == "Yes"),

    _q("sexual_cause",                      # ← added (Main 37)
       "Is it related to treatment, energy levels, or something else?",
       opts=["Treatment side effects", "Energy levels", "Other"],
       when=lambda d: d.get("sexual_health") == "Yes"),
]

# Master flow registry
FLOWS = {
    "pain":      FLOW_PAIN,
    "nutrition": FLOW_NUTRITION,
    "oral":      FLOW_ORAL,
    "gi":        FLOW_GI,
    "fatigue":   FLOW_FATIGUE,
    "activity":  FLOW_ACTIVITY,
    "mood":      FLOW_MOOD,
    "other":     FLOW_OTHER,
}

QUESTION_TYPE_BY_ID = {
    step["id"]: step.get("type", "options")
    for flow in FLOWS.values()
    for step in flow
}

STEP_BY_ID = {
    step["id"]: step
    for flow in FLOWS.values()
    for step in flow
}


STEP_SCHEMAS = {
    "eating_ability": {
        "unmatched_followup": "Could you tell me which sounds closest: eating normally, eating less than usual, mostly liquids, or tube feeds only?",
    },
    "activity_level": {
        "unmatched_followup": "Would you say you're doing your usual activities, doing less than usual, or really struggling with daily tasks?",
    },
    "sleep_quality": {
        "unmatched_followup": "Are you mostly sleeping through the night, or are you waking up a lot?",
    },
    "support_adequate": {
        "unmatched_followup": "Would you say you have enough support between visits, or not really?",
    },
    "social_support_quality": {
        "unmatched_followup": "Would you say you have good support, some support, or not much support right now?",
    },
    "med_dose_freq": {
        "components": {
            "frequency": {
                "detector": "frequency",
                "question": "How often do you usually take it?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if you're not sure about the timing right now. I've noted what you could tell me.",
            },
            "dose": {
                "detector": "dose",
                "question": "About how much do you usually take each time?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if you don't remember the dose right now. I've noted that for your care team.",
            },
        },
    },
    "nausea_management": {
        "components": {
            "management": {"detector": "management", "question": "What have you been using for the nausea?"},
            "helping": {"detector": "helping", "question": "Has that been helping at all?"},
        },
    },
    "vomiting_frequency": {
        "components": {
            "frequency": {
                "detector": "frequency",
                "question": "How often are you vomiting?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if you're not sure about the exact timing right now. I've noted what you could tell me.",
            },
            "amount": {
                "detector": "amount",
                "question": "About how much is it each time?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if the amount is hard to estimate right now. I've noted that for your care team.",
            },
        },
    },
    "vomiting_management": {
        "components": {
            "management": {"detector": "management", "question": "What have you been doing to manage the vomiting?"},
            "helping": {"detector": "helping", "question": "Has that been helping at all?"},
        },
    },
    "diarrhea_management": {
        "components": {
            "management": {"detector": "management", "question": "What have you been taking or doing for the diarrhea?"},
            "helping": {"detector": "helping", "question": "Has that been helping at all?"},
        },
    },
    "magic_mouthwash": {
        "components": {
            "management": {"detector": "management", "question": "What have you been using for that?"},
            "helping": {"detector": "helping", "question": "Has it been helping enough?"},
        },
    },
    "mucus_management": {
        "components": {
            "management": {"detector": "management", "question": "What have you been using for the mucus?"},
            "helping": {"detector": "helping", "question": "Has that been helping at all?"},
        },
    },
    "oral_rinse_type": {
        "components": {
            "specific_type": {
                "detector": "specific_type",
                "question": "What kind of rinse have you been using?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if you don't remember the exact name right now.",
            },
        },
    },
    "iv_frequency": {
        "components": {
            "frequency": {
                "detector": "frequency",
                "question": "How often have you been getting the IV fluids?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if you don't remember the exact schedule right now.",
            },
        },
    },
    "sleep_wake_reason": {
        "components": {
            "reason": {"detector": "reason", "question": "What tends to wake you up at night?"},
        },
    },
    "difficult_activities": {
        "components": {
            "reason": {"detector": "reason", "question": "Which daily activities feel hardest right now?"},
        },
    },
    "who_supports": {
        "components": {
            "support": {"detector": "support", "question": "Who has been helping support you?"},
        },
    },
    "needed_support": {
        "components": {
            "reason": {"detector": "reason", "question": "What kind of help would feel most useful right now?"},
        },
    },
    "other_pain_desc": {
        "components": {
            "location": {
                "detector": "location",
                "question": "Which body part is hurting?",
                "unknown_ok": False,
            },
        },
    },
    "pain_start": {
        "components": {
            "start_time": {
                "detector": "start_time",
                "question": "About when did that pain start?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if you're not sure exactly when it started.",
            },
        },
    },
    "skin_location": {
        "components": {
            "location": {
                "detector": "location",
                "question": "Where on your body are you noticing that skin problem?",
                "unknown_ok": False,
            },
        },
    },
    "skin_start": {
        "components": {
            "start_time": {
                "detector": "start_time",
                "question": "About when did you first notice it?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if you're not sure exactly when it started.",
            },
        },
    },
    "fever_start": {
        "components": {
            "start_time": {
                "detector": "start_time",
                "question": "About when did the fever or chills begin?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if you're not sure exactly when it started.",
            },
        },
    },
    "bp_reading": {
        "components": {
            "amount": {
                "detector": "amount",
                "question": "Do you remember roughly what the reading has been?",
                "unknown_ok": True,
                "unknown_ack": "That's okay if you don't remember the exact blood pressure reading right now.",
            },
        },
    },
}



# ══════════════════════════════════════════════════════════════════
# FLOW ENGINE
# ══════════════════════════════════════════════════════════════════

def _step_is_relevant(topic_key: str, step: dict, data: dict, raw_answers: Optional[dict] = None) -> bool:
    raw_answers = raw_answers or {}
    step_id = step.get("id")

    def raw(field: str) -> str:
        return _norm_text(str(raw_answers.get(field, "")))

    if topic_key == "pain":
        if step_id in {"ear_pain", "jaw_swelling", "pain_with_chewing"}:
            return _needs_head_neck_followup(data.get("other_pain_desc", "")) or _needs_head_neck_followup(raw_answers.get("other_pain_desc", ""))

        if step_id == "pain_med_timing":
            barrier = raw("eating_barrier")
            meds = data.get("pain_medications") or []
            has_pain_med = isinstance(meds, list) and "No pain medication" not in meds and len(meds) > 0
            return has_pain_med and any(token in barrier for token in {"pain", "swallow"})

    if topic_key == "oral":
        if step_id == "magic_mouthwash":
            impact = data.get("sore_pain_impact") or raw_answers.get("sore_pain_impact", "")
            return _norm_text(str(impact)) != _norm_text("No pain, just noticed it")

        if step_id == "avoiding_brushing":
            issue = _norm_text(str(data.get("teeth_issue_type") or raw_answers.get("teeth_issue_type", "")))
            brushing = data.get("brushing_difficult")
            return brushing == "Yes" or any(token in issue for token in {"pain", "sore", "multiple"})

    if topic_key == "fatigue":
        if step_id == "drowsy_schedule":
            return data.get("sleep_quality") == "No" and data.get("medication_drowsy") in {"Yes", "Sometimes"}

    if topic_key == "mood":
        if step_id == "feeling_down":
            emotional_state = str(data.get("emotional_state") or raw_answers.get("emotional_state", ""))
            if data.get("anxiety_impact") == "No" and _indicates_no_low_mood(emotional_state):
                return False

    if topic_key == "nutrition":
        if step_id == "pain_med_timing":
            barrier = raw("eating_barrier")
            return any(token in barrier for token in {"pain", "swallow"})

        if step_id == "iv_adjust":
            return data.get("iv_helping") == "No" or any(
                token in raw("iv_frequency") for token in {"want", "need", "more", "less", "change"}
            )

    if topic_key == "other":
        if step_id == "hearing_worsening":
            hearing = _norm_text(str(data.get("hearing_type") or raw_answers.get("hearing_type", "")))
            return bool(hearing)

        if step_id == "voice_communication_impact":
            progression = data.get("voice_progression")
            timing = data.get("voice_timing")
            return progression in {"About the same", "Worse"} or timing == "Constant"

    return True


def get_next_step(topic_key: str, data: dict, raw_answers: Optional[dict] = None) -> Optional[dict]:
    """Return the first unanswered applicable step for this topic."""
    for step in FLOWS.get(topic_key, []):
        when = step.get("when")
        if when and not when(data):
            continue
        if not _step_is_relevant(topic_key, step, data, raw_answers):
            continue
        if step["id"] not in data:
            return step
    return None


def topic_is_complete(topic_key: str, data: dict, raw_answers: Optional[dict] = None) -> bool:
    return get_next_step(topic_key, data, raw_answers) is None


def get_topic_progress(topic_key: str, data: dict, raw_answers: Optional[dict] = None) -> tuple[int, int]:
    """Returns (answered, applicable) counts."""
    flow = FLOWS.get(topic_key, [])
    applicable = [
        s for s in flow
        if (not s.get("when") or s["when"](data)) and _step_is_relevant(topic_key, s, data, raw_answers)
    ]
    answered = [s for s in applicable if s["id"] in data]
    return len(answered), len(applicable)



# ══════════════════════════════════════════════════════════════════
# MULTI-AGENT CLINICAL SYSTEM
# ══════════════════════════════════════════════════════════════════
# Architecture based on the ChatReport agent design:
#   Agent 1 — Answer Interpreter   : classify patient free-text
#   Agent 2 — Prior Comparison     : delta from last check-in
#   Agent 3 — Urgency & Criticality: patient safety monitoring
#   Agent 4 — Sentiment Monitor    : engagement & emotional state
#   Agent 5 — Doctor-Relevance     : clinical sufficiency + follow-up
#   Agent 6 — Next-Move            : author the follow-up question
#   Orchestrator — coordinates all agents, resolves conflicts
# ══════════════════════════════════════════════════════════════════


# ── Shared clinical background for HNC patients ───────────────────
_HNC_CONTEXT = (
    "Patients are adults receiving treatment for head and neck cancer (HNC) — "
    "typically chemoradiation or surgery. Common issues: severe mucositis, "
    "dysphagia, pain, weight loss, fatigue, depression, and impaired communication. "
    "Many are immunocompromised. Underreporting of severity is common in this population."
)

_RED_FLAGS = (
    "- Pain ≥ 7/10, uncontrolled or worsening despite medication\n"
    "- Fever ≥ 100.4 °F / 38 °C or chills with possible infection\n"
    "- Significant unintentional weight loss (> 5 lbs since last visit)\n"
    "- Complete inability to swallow or take any oral intake\n"
    "- Feeding tube complications: leakage, blockage, site infection\n"
    "- Breathing difficulty at rest, wheezing, or worsening dyspnoea\n"
    "- Falls or near-falls, especially with dizziness\n"
    "- Suicidal ideation or self-harm intent\n"
    "- Severe depression / distress interfering with daily function\n"
    "- New neurological symptoms: sudden weakness, numbness, confusion\n"
    "- Medication non-adherence affecting symptom control"
)


def _call_agent(system_prompt: str, user_content: dict, max_tokens: int = 500) -> dict:
    """
    Call OpenAI with a system + user message pair. All agents use this.
    Returns parsed JSON dict; returns {} on any error.
    """
    if not openai_client:
        return {}
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": json.dumps(user_content, ensure_ascii=False)},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return _extract_json_object(resp.choices[0].message.content.strip())
    except Exception as exc:
        print(f"[_call_agent error] {exc}")
        return {}


# ── Also keep legacy helper for report generation ─────────────────
def _call_openai(prompt: str, max_tokens: int = 120, temp: float = 0.4) -> str:
    if not openai_client:
        return ""
    try:
        r = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temp,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════
# AGENT 1 — ANSWER INTERPRETER
# ══════════════════════════════════════════════════════════════════

_ANSWER_INTERPRETER_SYS = f"""
You are the Answer Interpreter Agent for a clinical chatbot serving head and neck
cancer patients. {_HNC_CONTEXT}

Your ONLY job: classify a patient's free-text answer for the current question.

MATCHING RULES:
1. EXACT MATCH — case-insensitive, ignore minor punctuation → match_type "exact"

2. IMPLICIT/SEMANTIC MATCH — use natural language understanding to determine if the
   answer clearly and unambiguously implies one specific option.
   Exception for severity questions: map numbers to ranges:
     0 → "0 — No pain/None", 1-3 → Mild, 4-6 → Moderate, 7-9 → Severe/High, 10 → Worst
   - Treat natural conversational yes/no language as valid yes/no:
     "yeah", "yep", "yup", "sure" → Yes
     "nope", "nah", "not really" → No
   - When a question asks for a body location, any real body part is a meaningful answer.
     Do NOT reject "hand", "head", "jaw", "neck", "arm", etc. just because it is not
     one of the named specific options.

3. CATCH-ALL OPTION RULE (CRITICAL) — if the options list contains a catch-all such
   as "Somewhere else", "Other", "None of these", or "Something else", AND the
   patient's answer does not match any specific option but IS a valid, meaningful
   response to the question, you MUST map it to the catch-all option.
   - match_type = "implicit", confidence = 0.85
   - Examples for options ["Throat", "Tongue", "Somewhere else"]:
       "headache" → "Somewhere else"   (it's a valid pain location, just not listed)
       "my hand"  → "Somewhere else"
       "jaw"      → "Somewhere else"
       "shoulder" → "Somewhere else"
       "ear"      → "Somewhere else"
       "neck"     → "Somewhere else"
   - Examples for options ["Gabapentin", "Oxycodone", "Other"]:
       "Tylenol"  → "Other"
       "ibuprofen"→ "Other"
   NEVER return no_match when a catch-all option exists and the answer is a
   recognisable, meaningful response to the question asked.

4. TYPE MISMATCH DETECTION — if the question asks for one type of information but
   the patient provides a different type, return match_type "no_match" even if a
   catch-all option exists. Do NOT accept a wrong-type answer via catch-all.
   Examples of type mismatches:
     - Question asks WHERE the pain is (a location) but patient says "comes and goes",
       "all the time", "only when I swallow", "sometimes" (these are timing/pattern, not location)
     - Question asks HOW BAD the pain is (severity) but patient names a body part
     - Question asks WHEN pain started but patient describes the type of pain
   In these cases: match_type = "no_match", matched_option = null,
   reasoning should explain the type mismatch so the chatbot can ask again clearly.

5. AMBIGUITY — two or more SPECIFIC (non-catch-all) options equally plausible
   → match_type "no_match", list candidates

6. NO MATCH — answer is completely unrelated/nonsensical AND no catch-all exists
   → match_type "no_match"

6. SPECIAL STATES:
   a) DISTRESS FLAG: any expression of being unable to cope, hopelessness, suicidal
      ideation → distress_flag true
   b) URGENCY FLAG: sudden severe pain, breathing difficulty, "worst of my life",
      fever with chills, bleeding, or any red flag symptom → urgency_flag true
   c) OFF-TOPIC: answer is entirely unrelated to the question (e.g. patient asks
      about appointment scheduling when asked about pain location) → match_type "off_topic"
   d) INVALID: empty or gibberish → match_type "invalid"

CONFIDENCE: 1.0 exact, 0.85-0.95 strong implicit, 0.85 catch-all, <0.7 → no_match.
matched_option MUST be copied VERBATIM from options list, or null.

Return ONLY valid JSON:
{{
  "match_type": "exact|implicit|no_match|off_topic|invalid",
  "matched_option": "..." or null,
  "confidence": 0.0-1.0,
  "candidates": [],
  "distress_flag": false,
  "urgency_flag": false,
  "reasoning": "One sentence."
}}
"""


def run_answer_interpreter(step: dict, patient_answer: str) -> dict:
    """
    Agent 1: Classify patient's free-text answer against predefined options.
    Returns interpreter output dict, or safe default on failure.
    """
    default = {
        "match_type": "no_match", "matched_option": None, "confidence": 0.0,
        "candidates": [], "distress_flag": False, "urgency_flag": False,
        "reasoning": "Agent unavailable."
    }
    if not patient_answer.strip():
        return {**default, "match_type": "invalid"}

    result = _call_agent(_ANSWER_INTERPRETER_SYS, {
        "question_text": step.get("text", ""),
        "options": step.get("opts", []),
        "patient_answer": patient_answer,
    }, max_tokens=200)

    if not result:
        return default

    # Validate matched_option is actually in the options list
    mo = result.get("matched_option")
    if mo and mo not in step.get("opts", []):
        result["matched_option"] = None
        result["match_type"] = "no_match"

    return {**default, **result}


# ══════════════════════════════════════════════════════════════════
# AGENT 2 — PRIOR CHECK-IN COMPARISON
# ══════════════════════════════════════════════════════════════════

_PRIOR_COMPARISON_SYS = """
You are the Prior Check-in Comparison Agent for a clinical chatbot.
Compare a patient's current answer to the same question from their last check-in.

change_direction rules:
  improved       — current answer suggests less pain / better status
  worsened       — current answer suggests more pain / worse status
  neutral_change — changed but direction unclear (e.g., location shifted)
  no_change      — answers are the same or equivalent
  new_data       — no prior data available for comparison

change_magnitude rules:
  For numeric severity: large=3+ points difference, moderate=2, small=1, none=0
  For non-numeric: large if clinically major (e.g., no pain → severe pain),
    moderate for any other meaningful change, small for minor wording difference.

clinical_note: One sentence, plain English. Include change magnitude and direction.
If no prior data: "No prior data available for comparison."

Return ONLY valid JSON:
{
  "has_prior_data": true/false,
  "last_answer": "..." or null,
  "change_detected": true/false,
  "change_direction": "improved|worsened|neutral_change|no_change|new_data",
  "change_magnitude": "large|moderate|small|none",
  "clinical_note": "..."
}
"""


def run_prior_comparison(step: dict, current_answer: str, last_topic_data: dict) -> dict:
    """
    Agent 2: Compare current answer to last check-in answer for this question.
    Returns comparison dict, or no-prior-data default on failure.
    """
    default = {
        "has_prior_data": False, "last_answer": None,
        "change_detected": False, "change_direction": "new_data",
        "change_magnitude": "none", "clinical_note": "No prior data available."
    }
    if not last_topic_data:
        return default

    last_answer = last_topic_data.get(step["id"])
    if last_answer is None:
        return default

    result = _call_agent(_PRIOR_COMPARISON_SYS, {
        "question_text": step.get("text", ""),
        "last_check_in_answer": str(last_answer),
        "current_answer": current_answer,
    }, max_tokens=200)

    return {**default, **result} if result else default


# ══════════════════════════════════════════════════════════════════
# AGENT 3 — URGENCY & CRITICALITY
# ══════════════════════════════════════════════════════════════════

_URGENCY_SYS = f"""
You are the Urgency & Criticality Agent for a clinical chatbot serving head and
neck cancer patients. {_HNC_CONTEXT}

You monitor patient safety. Read ALL raw answers across the session — urgency
signals often appear in free-text not captured by structured options.
When in doubt, flag. A false positive is far less harmful than a missed crisis.

RED FLAGS TO DETECT:
{_RED_FLAGS}

TIER DEFINITIONS:
  0 — NO URGENCY: Continue normally.
  1 — WATCH: Notable signal. Log for report, continue session normally.
  2 — URGENT: Care team must contact patient today. Continue session.
       Show one care team message to the patient.
  3 — EMERGENCY: Immediate threat. Terminate session. Patient to emergency services.

SIGNAL RULES:

  IMPORTANT CLINICAL CONTEXT FOR HNC PATIENTS:
  Pain scores of 7-8/10 are clinically expected during active chemoradiation for head
  and neck cancer — mucositis, dysphagia, and treatment toxicity routinely produce this
  level. A high pain score ALONE does not warrant Tier 2 escalation in this population.
  You MUST look for additional signals beyond the number itself.

  Medical signals:
    M1 — SEVERE UNCONTROLLED PAIN:
      Tier 1 (WATCH): Pain 7-9/10 alone, without other signals.
      Tier 2 (URGENT): Pain 7-9/10 AND at least one of:
        - Patient states nothing helps OR medication is not working
        - Patient cannot eat, sleep, or perform basic activities because of pain
        - Pain is new or suddenly much worse than their baseline
        - Fever/chills present alongside severe pain
      Tier 2 (URGENT): Pain reported as 10/10 or "worst of my life" or "unbearable"
    M2: Sudden new severe pain in head/neck — Tier 2
    M3: Fever with chills in an HNC patient (immunocompromised) — Tier 2 always
    M4: Complete inability to swallow ANY liquids — Tier 2
    M5: Breathing difficulty at rest or wheezing — Tier 2
    M6: Prescription medication suddenly stopped — Tier 2
    M7: Reported falls — Tier 1
    M8: Significant functional disruption (3+ nights no sleep, cannot eat for days) — Tier 2
  
  Psychological signals:
    P1: Explicit suicidal ideation or self-harm intent → TIER 3 ALWAYS
    P2: Passive death wish ("I don't care if I make it") → Tier 2
    P3: Crisis desperation ("I can't take this anymore" about life, not just pain) → Tier 2
    P4: Complete isolation ("no one to help me") with high pain → Tier 2

  Accumulation: 3+ Tier 1 signals in same session → escalate to Tier 2.

IMPORTANT NON-ESCALATION GUARDRAILS:
  - Do NOT escalate just because an answer is brief, partial, or missing one detail.
  - Do NOT escalate just because the patient does not remember a dose, timing, or exact amount.
  - Do NOT escalate just because a patient reports PRN or non-daily use without saying it is prescribed daily.
  - Do NOT treat "every 2 days", "sometimes", or similar medication-use frequency by itself as urgent.
  - Medication adherence becomes urgent only if the patient clearly reports they stopped an important prescribed medication,
    cannot access it, or their symptoms are uncontrolled because they are missing it.
  - If the patient gives usable but incomplete information, continue normally unless another red flag is clearly present.

PATIENT MESSAGES (verbatim — do not modify):
  Tier 2: "Thank you for sharing this with us. We can see you're having a really
    difficult time. A member of your care team will be reaching out to you today.
    Please keep your phone nearby. Your responses have been saved."
  Tier 3 (medical): "We're concerned about what you've shared. Please call 911 or
    go to your nearest emergency room immediately. Your care team has been notified."
  Tier 3 (P1/P2): "We hear you, and we want you to know your safety matters.
    Please call or text 988 right now — the Suicide & Crisis Lifeline is available
    24/7. If you are in immediate danger, call 911 or go to your nearest emergency
    room. Your care team has been notified."

Return ONLY valid JSON:
{{
  "session_tier": 0-3,
  "new_signals": ["M1", "P3"],
  "escalation_reason": "..." or null,
  "patient_message": "..." or null,
  "continue_session": true/false,
  "clinical_note": "..." or null
}}
"""


def run_urgency_agent(
    step: dict,
    current_answer_raw: str,
    current_answer_matched: Optional[str],
    session_answers: dict,
    prior_baseline: dict,
    active_signals: list,
    distress_flag: bool,
    urgency_flag: bool,
) -> dict:
    """
    Agent 3: Patient safety monitoring. Returns urgency assessment.
    """
    default = {
        "session_tier": 0, "new_signals": [], "escalation_reason": None,
        "patient_message": None, "continue_session": True, "clinical_note": None
    }
    result = _call_agent(_URGENCY_SYS, {
        "current_question": step.get("text", ""),
        "current_answer_raw": current_answer_raw,
        "current_answer_matched": current_answer_matched,
        "distress_flag_from_interpreter": distress_flag,
        "urgency_flag_from_interpreter": urgency_flag,
        "all_session_answers_so_far": session_answers,
        "active_signals_this_session": active_signals,
        "prior_baseline_summary": prior_baseline,
    }, max_tokens=300)

    if not result:
        return default

    # Safety: P1 (suicidal ideation) from the distress flag always forces Tier 3
    if distress_flag and "suicid" in current_answer_raw.lower():
        result["session_tier"] = 3
        result["continue_session"] = False

    return {**default, **result}


# ══════════════════════════════════════════════════════════════════
# AGENT 4 — PATIENT SENTIMENT & ENGAGEMENT MONITOR
# ══════════════════════════════════════════════════════════════════

_SENTIMENT_SYS = f"""
You are the Patient Sentiment & Engagement Monitor for a clinical chatbot serving
head and neck cancer patients. {_HNC_CONTEXT}

You track the patient's emotional state and engagement quality across the session.
You do NOT classify safety crises — that belongs to the Urgency Agent.

IMPORTANT POPULATION CONTEXT:
  - Brief answers ≠ disengagement (pain impairs fluency)
  - Stoicism is common — calibrate against reported pain level
  - Emotional flooding (long distressed answers) needs acknowledgment before next Q

DIMENSION SCORES:
  emotional_state: positive|neutral|fatigued|distressed|frustrated|anxious|overwhelmed|resigned
  engagement_level: high|moderate|low|resistant|confused
  engagement_trajectory: stable|improving|declining|insufficient_data

SIGNALS TO DETECT (set to true if present):
  E3_resistance: patient explicitly pushes back ("I already told you", "can we be done")
  E7_wants_to_stop: "I need to stop", "I'm done", "I can't do this right now"
  EM1_pain_frustration: venting about their pain situation
  EM2_sadness_grief: expressing loss or grief about what they can no longer do
  EM6_emotional_flooding: unusually long, distressed, emotionally dense answer

ADAPTATION SIGNALS:
  tone_profile: standard|warm|gentle|simplified
  acknowledgment_required: true if EM1/EM2/EM6 active in current answer
  acknowledgment_text: ≤25 words, first person chatbot voice, warm but not effusive,
    reflects what patient shared emotionally (NOT their clinical data back at them).
    Example: "That sounds really hard to carry. Thank you for sharing that."
  simplify_next_question: true if patient seems confused or cognitively fatigued
  reduce_follow_up_depth: true if E3/E7 active or engagement declining

Return ONLY valid JSON:
{{
  "emotional_state": "...",
  "engagement_level": "...",
  "engagement_trajectory": "...",
  "signals": {{
    "E3_resistance": false,
    "E7_wants_to_stop": false,
    "EM1_pain_frustration": false,
    "EM2_sadness_grief": false,
    "EM6_emotional_flooding": false
  }},
  "adaptation": {{
    "tone_profile": "standard",
    "acknowledgment_required": false,
    "acknowledgment_text": null,
    "simplify_next_question": false,
    "reduce_follow_up_depth": false
  }},
  "engagement_note_for_doctor": null
}}
"""


def run_sentiment_agent(
    step: dict,
    current_answer_raw: str,
    session_answers: dict,
    active_signals: list,
    question_count: int,
) -> dict:
    """
    Agent 4: Track patient sentiment and engagement. Returns adaptation signals.
    """
    default = {
        "emotional_state": "neutral", "engagement_level": "moderate",
        "engagement_trajectory": "insufficient_data",
        "signals": {
            "E3_resistance": False, "E7_wants_to_stop": False,
            "EM1_pain_frustration": False, "EM2_sadness_grief": False,
            "EM6_emotional_flooding": False,
        },
        "adaptation": {
            "tone_profile": "standard", "acknowledgment_required": False,
            "acknowledgment_text": None, "simplify_next_question": False,
            "reduce_follow_up_depth": False,
        },
        "engagement_note_for_doctor": None,
    }
    result = _call_agent(_SENTIMENT_SYS, {
        "current_question": step.get("text", ""),
        "current_answer": current_answer_raw,
        "questions_answered_so_far": question_count,
        "all_session_answers": session_answers,
        "active_sentiment_signals": active_signals,
    }, max_tokens=300)

    if not result:
        return default

    # Merge nested dicts carefully
    merged = {**default}
    merged["emotional_state"] = result.get("emotional_state", default["emotional_state"])
    merged["engagement_level"] = result.get("engagement_level", default["engagement_level"])
    merged["engagement_trajectory"] = result.get("engagement_trajectory", default["engagement_trajectory"])
    merged["engagement_note_for_doctor"] = result.get("engagement_note_for_doctor")

    if "signals" in result and isinstance(result["signals"], dict):
        merged["signals"] = {**default["signals"], **result["signals"]}
    if "adaptation" in result and isinstance(result["adaptation"], dict):
        merged["adaptation"] = {**default["adaptation"], **result["adaptation"]}

    return merged


# ══════════════════════════════════════════════════════════════════
# AGENT 5 — DOCTOR-RELEVANCE
# ══════════════════════════════════════════════════════════════════

_DOCTOR_RELEVANCE_SYS = f"""
You are the Doctor-Relevance Agent for a clinical chatbot serving head and neck
cancer patients. {_HNC_CONTEXT}

You evaluate patient answers from the physician's perspective. Your outputs:
  1. Clinical sufficiency verdict on the current answer
  2. Follow-up recommendation with a precise information GOAL (not the question itself)
  3. A compact clinical note (≤35 words, third person) for the doctor's report

FOLLOW-UP RULES:
  - The question list is a question bank, not a rigid script. Judge the current answer
    like a clinician deciding whether anything important is still missing.
  - A meaningful free-text answer in the patient's own words is clinically usable even if it does not match the option wording.
  - If the patient gave a broad but meaningful answer, break down what is missing conceptually; do NOT treat it as meaningless.
  - If the patient already explained the reason in their own words, do NOT recommend a generic "what is making this difficult" follow-up.
  - If the patient supplies one detail and explicitly does not know another, accept the known detail and only ask for the missing one if it is truly necessary.
  - If the missing detail is something the patient reasonably may not know right now, prefer no follow-up over repetitive questioning.
  - ONLY recommend follow-up if information_completeness is "partial" or "none"
    AND follow_up_count is 0 AND the missing info is clinically meaningful
  - NEVER recommend follow-up if follow_up_count ≥ 1 (absolute limit: 1 per question)
  - NEVER recommend follow-up if patient showed resistance in their answer

SPECIAL CLINICAL SIGNALS (set if present):
  trajectory_mismatch: patient says improving but comparison shows worsening (or vice versa)
  medication_stop_signal: patient stopped taking prescription medication without explanation
  aggravating_medication_signal: patient reports their medication makes symptoms worse
  severity_underreporting: patient rates low severity but describes severe functional impact

follow_up_goal: A statement of WHAT information is needed — NOT a question.
  Example: "Obtain a numeric pain severity score — patient described pain without rating it."

information_completeness:
  complete — answer fully satisfies the clinical information need
  partial  — has some value but key information missing
  none     — no clinically usable information

change_significance:
  critical — large worsening, urgency flag, new alarming symptom, medication stopped
  notable  — meaningful change worth highlighting
  stable   — no meaningful change
  no_baseline — first session or no prior data

clinical_priority: high | medium | low
doctor_note: ≤35 words, third person, factual only, include comparison if change is critical/notable.

Return ONLY valid JSON:
{{
  "information_completeness": "complete|partial|none",
  "clinical_value_score": 0.0-1.0,
  "follow_up_recommended": true/false,
  "follow_up_goal": "..." or null,
  "follow_up_urgency": "immediate|routine|none",
  "change_significance": "critical|notable|stable|no_baseline",
  "clinical_priority": "high|medium|low",
  "doctor_note": "..." or null,
  "special_signals": {{
    "trajectory_mismatch": false,
    "medication_stop_signal": false,
    "aggravating_medication_signal": false,
    "severity_underreporting": false
  }}
}}
"""


def run_doctor_relevance(
    step: dict,
    current_answer_raw: str,
    current_answer_matched: Optional[str],
    prior_comparison: dict,
    session_answers: dict,
    followup_count: int,
) -> dict:
    """
    Agent 5: Assess clinical sufficiency and decide if follow-up is warranted.
    """
    default = {
        "information_completeness": "complete", "clinical_value_score": 0.7,
        "follow_up_recommended": False, "follow_up_goal": None,
        "follow_up_urgency": "none", "change_significance": "no_baseline",
        "clinical_priority": "medium", "doctor_note": None,
        "special_signals": {
            "trajectory_mismatch": False, "medication_stop_signal": False,
            "aggravating_medication_signal": False, "severity_underreporting": False,
        },
    }
    result = _call_agent(_DOCTOR_RELEVANCE_SYS, {
        "question_text": step.get("text", ""),
        "question_type": step.get("type", "options"),
        "options": step.get("opts", []),
        "current_answer_raw": current_answer_raw,
        "current_answer_matched": current_answer_matched,
        "prior_comparison": prior_comparison,
        "session_answers_so_far": session_answers,
        "follow_up_count_this_question": followup_count,
    }, max_tokens=400)

    if not result:
        return default

    # Hard-enforce the follow-up limit
    if followup_count >= 1:
        result["follow_up_recommended"] = False
        result["follow_up_goal"] = None

    merged = {**default, **result}
    if "special_signals" in result and isinstance(result["special_signals"], dict):
        merged["special_signals"] = {**default["special_signals"], **result["special_signals"]}

    return merged


# ══════════════════════════════════════════════════════════════════
# AGENT 6 — NEXT-MOVE (FOLLOW-UP QUESTION AUTHOR)
# ══════════════════════════════════════════════════════════════════

_NEXT_MOVE_SYS = f"""
You are the Next-Move Agent for a clinical chatbot serving head and neck cancer patients.
{_HNC_CONTEXT}

You receive a follow-up GOAL and write the actual follow-up question the patient sees.
The decision to follow up has already been made. Your job is HOW to ask it.

TONE PROFILES:
  standard  — professional, warm, clear
  warm      — add genuine warmth; soften clinical phrasing; sincere, not effusive
  gentle    — softest possible; avoid anything demanding or clinical
  simplified — short sentences, very simple words, one idea only

RULES:
  - Treat the original form question as background only; you are not tied to its exact wording
  - Ask the most clinically useful next single question, as a doctor or nurse naturally would
  - Stay anchored to the patient's last answer; the follow-up should feel like a direct continuation of what they just said
  - If the patient used plain-language wording, mirror that wording naturally instead of switching back to rigid form language
  - Ask only for the single missing detail; never restate details the patient already provided
  - If the patient said they do not know a detail, do not challenge that or sound repetitive
  - Write in second person, conversational language
  - Never use medical jargon without immediate plain explanation
  - NEVER ask a multi-part question
  - NEVER repeat the original question verbatim
  - Keep the question to ≤25 words
  - An acknowledgment may be shown BEFORE your question — do not repeat it
  - If simplify=true: use the shortest phrasing possible

Return ONLY valid JSON:
{{
  "follow_up_question": "...",
  "preamble": "..." or null
}}
preamble: ≤10 words transitional phrase if naturally needed, else null.
"""


def run_next_move_agent(
    step: dict,
    current_answer_raw: str,
    followup_goal: str,
    tone_profile: str,
    simplify: bool,
) -> dict:
    """
    Agent 6: Author the follow-up question in natural language.
    """
    result = _call_agent(_NEXT_MOVE_SYS, {
        "original_question": step.get("text", ""),
        "patient_answer": current_answer_raw,
        "follow_up_goal": followup_goal,
        "tone_profile": tone_profile,
        "simplify": simplify,
    }, max_tokens=120)

    if result and result.get("follow_up_question"):
        return result
    # Fallback: derive a question from the goal
    return {
        "follow_up_question": "Could you tell me a bit more about that?",
        "preamble": None,
    }


# ══════════════════════════════════════════════════════════════════
# ORCHESTRATOR — coordinates all agents
# ══════════════════════════════════════════════════════════════════

def _build_session_answers(topic_key: str) -> dict:
    """Build {question_id: raw_answer} from current session state for the topic."""
    state = st.session_state.topic_states.get(topic_key, {})
    data = state.get("data", {})
    raw_answers = state.get("raw_answers", {})
    payload = {}
    for k, v in data.items():
        if v is None:
            continue
        payload[k] = str(raw_answers.get(k, v))
    return payload


def _build_prior_baseline(topic_key: str) -> dict:
    """Return a compact prior check-in summary for agent context."""
    last = st.session_state.last_checkin.get(topic_key, {})
    if not last:
        return {}
    # Return key fields only to keep the payload small
    keys = list(last.keys())[:10]
    return {k: str(last[k]) for k in keys}


def _build_all_topic_data() -> dict:
    payload = {}
    for _, key in TOPICS:
        topic_state = st.session_state.topic_states[key]
        topic_data = dict(topic_state.get("data", {}))
        raw_answers = topic_state.get("raw_answers", {})
        if raw_answers:
            topic_data["_verbatim_answers"] = dict(raw_answers)
        payload[key] = topic_data
    return payload


def run_agent_pipeline(
    topic_key: str,
    step: dict,
    answer: str,
    state: dict,
    last_topic_data: dict,
) -> dict:
    """
    Orchestrator: runs all agents in sequence (with parallelism where safe)
    and returns a unified decision dict consumed by handle_answer.

    Returns:
    {
        "matched_option": str|None,
        "follow_up": bool,
        "follow_up_question": str,
        "acknowledgment": str,
        "assistant_message": str,
        "urgency_tier": int,
        "urgency_message": str|None,
        "reduce_follow_up": bool,
        "wants_to_stop": bool,
        "doctor_note": str|None,
        "clinical_priority": str,
        "change_significance": str,
        "change_clinical_note": str,
        "special_signals": dict,
        "sentiment_note": str|None,
    }
    """
    if not openai_client:
        return _pipeline_default()

    question_count = len(state.get("data", {}))
    session_answers = _build_session_answers(topic_key)
    prior_baseline  = _build_prior_baseline(topic_key)
    followup_count  = state.get("followup_counts", {}).get(step["id"], 0)
    targeted_followup = _targeted_followup_override(step, answer, state)

    # ── STEP 1: Answer Interpreter (must run first) ────────────────
    interp = run_answer_interpreter(step, answer)
    matched = interp.get("matched_option")
    distress = interp.get("distress_flag", False)
    urgency_flag = interp.get("urgency_flag", False)

    # ── STEP 2: Run three agents in parallel ───────────────────────
    # Prior Comparison, Urgency, and Sentiment can all run at once.
    prior_comp = {}
    urgency_out = {}
    sentiment_out = {}

    active_urgency_signals = st.session_state.get("urgency_state", {}).get("all_signals", [])
    active_sentiment_signals = st.session_state.get("sentiment_state", {}).get("all_signals", [])

    def _run_prior():
        return run_prior_comparison(step, answer, last_topic_data)

    def _run_urgency():
        return run_urgency_agent(
            step, answer, matched, session_answers, prior_baseline,
            active_urgency_signals, distress, urgency_flag
        )

    def _run_sentiment():
        return run_sentiment_agent(
            step, answer, session_answers, active_sentiment_signals, question_count
        )

    with _futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_prior     = pool.submit(_run_prior)
        f_urgency   = pool.submit(_run_urgency)
        f_sentiment = pool.submit(_run_sentiment)
        prior_comp   = f_prior.result()
        urgency_out  = f_urgency.result()
        sentiment_out = f_sentiment.result()

    # ── STEP 3: Urgency interrupt check ───────────────────────────
    tier = urgency_out.get("session_tier", 0)

    # Update session-level urgency state
    _merge_urgency_state(tier, urgency_out)

    if tier == 3:
        # Emergency — bypass all other agents
        return {
            **_pipeline_default(),
            "matched_option": matched,
            "urgency_tier": 3,
            "urgency_message": urgency_out.get("patient_message"),
            "wants_to_stop": True,
        }

    # ── STEP 4: Doctor-Relevance ───────────────────────────────────
    dr_out = run_doctor_relevance(
        step, answer, matched, prior_comp, session_answers, followup_count
    )
    if targeted_followup:
        dr_out = {
            **dr_out,
            "follow_up_recommended": targeted_followup.get("follow_up_recommended", dr_out.get("follow_up_recommended", False)),
            "follow_up_goal": targeted_followup.get("follow_up_goal", dr_out.get("follow_up_goal")),
            "information_completeness": targeted_followup.get("information_completeness", dr_out.get("information_completeness")),
            "clinical_priority": targeted_followup.get("clinical_priority", dr_out.get("clinical_priority", "medium")),
        }

    # ── STEP 5: Apply follow-up decision logic ─────────────────────
    adapt = sentiment_out.get("adaptation", {})
    sigs  = sentiment_out.get("signals", {})
    reduce = adapt.get("reduce_follow_up_depth", False)
    wants_to_stop = sigs.get("E7_wants_to_stop", False)

    dr_recommends  = dr_out.get("follow_up_recommended", False)
    followup_goal  = dr_out.get("follow_up_goal", "")
    priority       = dr_out.get("clinical_priority", "medium")

    # Override rules (clinical necessity > patient experience)
    force_followup = False
    if step.get("type") == "number" and dr_out.get("information_completeness") != "complete":
        force_followup = True  # Numeric severity is always high priority
    if dr_out.get("special_signals", {}).get("medication_stop_signal") and followup_count == 0:
        force_followup = True

    # Suppression rules
    suppress = False
    if followup_count >= 1:
        suppress = True  # Absolute limit
    if wants_to_stop:
        suppress = True
    if interp.get("match_type") in ("off_topic", "invalid"):
        suppress = True
    if reduce and priority != "high":
        suppress = True
    if sigs.get("E3_resistance") and priority != "high":
        suppress = True

    do_follow_up = (force_followup or dr_recommends) and not suppress

    # ── STEP 6: Compose follow-up question if needed ───────────────
    follow_up_question = ""
    if do_follow_up and followup_goal:
        if targeted_followup.get("follow_up_question"):
            follow_up_question = targeted_followup["follow_up_question"]
        else:
            tone = adapt.get("tone_profile", "standard")
            simplify = adapt.get("simplify_next_question", False)
            nm_out = run_next_move_agent(step, answer, followup_goal, tone, simplify)
            preamble = nm_out.get("preamble") or ""
            fq = nm_out.get("follow_up_question", "")
            follow_up_question = f"{preamble} {fq}".strip() if preamble else fq

    # ── STEP 7: Build assistant message for non-follow-up case ─────
    assistant_message = ""
    if not do_follow_up:
        comp_note    = prior_comp.get("clinical_note", "")
        change_dir   = prior_comp.get("change_direction", "new_data")
        prev_answer  = prior_comp.get("last_answer", "")
        emotional    = sentiment_out.get("emotional_state", "neutral")

        # Build a brief contextual acknowledgment
        if targeted_followup.get("assistant_message"):
            assistant_message = targeted_followup["assistant_message"]
        elif comp_note and change_dir in ("worsened", "improved") and prev_answer:
            assistant_message = comp_note
        elif emotional == "distressed":
            assistant_message = "That sounds really difficult. I've made a note of this for your care team."
        elif emotional in ("anxious", "overwhelmed"):
            assistant_message = "I hear you — I've made a note of that for your care team."
        elif change_dir == "worsened":
            assistant_message = "I've noted that, and I can see things have been harder than last time."
        elif change_dir == "improved":
            assistant_message = "That's helpful to know, and it sounds like there's been some improvement since last time."
        else:
            assistant_message = _default_chatty_reply(topic_key, answer, step, last_topic_data)

    # ── STEP 8: Compose acknowledgment if needed ───────────────────
    acknowledgment = ""
    if adapt.get("acknowledgment_required") and adapt.get("acknowledgment_text"):
        acknowledgment = adapt["acknowledgment_text"]

    # ── STEP 9: Merge urgency and sentiment state ──────────────────
    _merge_sentiment_state(sentiment_out)

    # ── STEP 10: Tier 2 notice ─────────────────────────────────────
    urgency_msg = None
    urg_state = st.session_state.get("urgency_state", {})
    if tier == 2 and not urg_state.get("escalation_shown", False):
        urgency_msg = urgency_out.get("patient_message")
        urg_state["escalation_shown"] = True
        st.session_state["urgency_state"] = urg_state

    return {
        "matched_option": matched,
        "follow_up": do_follow_up,
        "follow_up_question": follow_up_question,
        "acknowledgment": acknowledgment,
        "assistant_message": assistant_message,
        "urgency_tier": tier,
        "urgency_message": urgency_msg,
        "reduce_follow_up": reduce,
        "wants_to_stop": wants_to_stop,
        "doctor_note": dr_out.get("doctor_note"),
        "clinical_priority": priority,
        "change_significance": dr_out.get("change_significance", "no_baseline"),
        "change_clinical_note": prior_comp.get("clinical_note", ""),
        "special_signals": dr_out.get("special_signals", {}),
        "sentiment_note": sentiment_out.get("engagement_note_for_doctor"),
    }


def _pipeline_default() -> dict:
    """Safe default when agents are unavailable."""
    return {
        "matched_option": None, "follow_up": False, "follow_up_question": "",
        "acknowledgment": "", "assistant_message": "",
        "urgency_tier": 0, "urgency_message": None,
        "reduce_follow_up": False, "wants_to_stop": False,
        "doctor_note": None, "clinical_priority": "medium",
        "change_significance": "no_baseline", "change_clinical_note": "",
        "special_signals": {}, "sentiment_note": None,
    }


def _merge_urgency_state(tier: int, urgency_out: dict):
    """Merge new urgency signals into session-level urgency state."""
    state = st.session_state.get("urgency_state", {
        "current_tier": 0, "all_signals": [], "escalation_shown": False,
        "emergency_shown": False,
    })
    state["current_tier"] = max(state.get("current_tier", 0), tier)
    new_sigs = urgency_out.get("new_signals", [])
    existing = state.get("all_signals", [])
    state["all_signals"] = list(set(existing + new_sigs))
    st.session_state["urgency_state"] = state


def _merge_sentiment_state(sentiment_out: dict):
    """Merge new sentiment signals into session-level sentiment state."""
    state = st.session_state.get("sentiment_state", {"all_signals": []})
    new_sigs = [k for k, v in sentiment_out.get("signals", {}).items() if v]
    existing = state.get("all_signals", [])
    state["all_signals"] = list(set(existing + new_sigs))
    state["engagement_trajectory"] = sentiment_out.get("engagement_trajectory", "insufficient_data")
    state["emotional_state"] = sentiment_out.get("emotional_state", "neutral")
    st.session_state["sentiment_state"] = state


# ══════════════════════════════════════════════════════════════════
# LEGACY SUPPORT: keep interpret_user_input_with_options working
# ══════════════════════════════════════════════════════════════════

# Keywords that identify a catch-all option in any question
_CATCHALL_KEYWORDS = {"somewhere else", "other", "none of these", "something else"}


def _is_catchall_option_value(value: str) -> bool:
    return _norm_text(value) in _CATCHALL_KEYWORDS


# Timing/pattern words that are never valid body-location answers
_TIMING_WORDS = {
    "all the time", "comes and goes", "comes and go", "sometimes", "occasionally",
    "only when", "always", "never", "constant", "intermittent", "at night",
    "in the morning", "after eating", "when swallowing", "when i eat",
    "only at night", "mostly", "often", "rarely", "every day", "all day",
}


def _is_timing_answer(text: str) -> bool:
    """Return True if the text looks like a timing/frequency answer, not a location."""
    normalized = _norm_text(text)
    return any(phrase in normalized for phrase in _TIMING_WORDS)


def _catchall_fallback(step: dict, user_input: str) -> str:
    """
    If the step has a catch-all option (Somewhere else / Other / None of these),
    return it for any answer that looks like a genuine (non-vague) response.
    This is the last-resort safety net so patients never get stuck in a retry loop
    when they name something real that just isn't one of the specific listed options.

    Exception: if the question is asking for a location but the patient answered
    with a timing/pattern description, we do NOT use the catch-all — instead we
    return the raw input unchanged so the chatbot retries with a clearer prompt.
    """
    opts = step.get("opts", [])
    question_lower = _norm_text(step.get("text", ""))

    # Detect location questions by their wording
    is_location_question = any(
        w in question_lower
        for w in ("where", "location", "located", "which part", "which area")
    )

    for opt in opts:
        if any(kw in opt.lower() for kw in _CATCHALL_KEYWORDS):
            # Only use catch-all if the answer is not empty/gibberish
            if not user_input.strip() or _looks_vague_answer(user_input):
                break
            # Don't use catch-all if it's a location question and the answer
            # describes timing — that's a type mismatch, not a location
            if is_location_question and _is_timing_answer(user_input):
                break
            return opt
    return user_input


def interpret_user_input_with_options(step, user_input):
    """
    Use the Answer Interpreter Agent to classify free-text against question options.
    Falls back to the catch-all option (Somewhere else / Other) if the agent returns
    no_match but a catch-all exists and the answer is a real, meaningful response.
    Returns matched option string if found, else original input.
    """
    binary_match = _match_binary_option(step, user_input)
    if binary_match:
        return binary_match

    alias_match = _infer_option_from_text(step, user_input)
    if alias_match:
        return alias_match

    if step.get("id") == "pain_location":
        normalized = _norm_text(user_input)
        if any(w in normalized for w in ("throat", "pharynx", "larynx", "voice box")):
            return "Throat"
        if any(w in normalized for w in ("tongue", "lingual")):
            return "Tongue"
        if _looks_like_body_location(user_input) and not _is_timing_answer(user_input):
            return "Somewhere else"

    if not openai_client:
        # No LLM available — still try catch-all fallback for valid answers
        return _catchall_fallback(step, user_input)

    if not step.get("opts"):
        return user_input

    result = run_answer_interpreter(step, user_input)
    matched = result.get("matched_option")

    # Agent found a valid specific match
    if matched and matched in step.get("opts", []):
        return matched

    # Agent returned no_match — apply catch-all safety net
    return _catchall_fallback(step, user_input)


# ══════════════════════════════════════════════════════════════════
# URGENCY BANNER RENDERER
# ══════════════════════════════════════════════════════════════════

def render_urgency_banner():
    """
    Render a coloured urgency banner at the top of the main content area.
    Only shown if urgency_state tier >= 2.
    """
    urg = st.session_state.get("urgency_state", {})
    tier = urg.get("current_tier", 0)
    if tier == 0:
        return

    if tier == 1:
        st.markdown(
            '<div style="background:#fff8e8;border:1px solid #f9c846;border-radius:14px;'
            'padding:10px 14px;margin-bottom:12px;font-size:13px;color:#7a5a00;">'
            '⚠️ <strong>Note for your care team:</strong> Some of your responses have been '
            'flagged for additional review before your visit.'
            '</div>',
            unsafe_allow_html=True,
        )
    elif tier == 2:
        msg = urg.get("patient_message") or (
            "Thank you for sharing this with us. A member of your care team will be "
            "reaching out to you today. Please keep your phone nearby."
        )
        st.markdown(
            f'<div style="background:#fff3f3;border:1.5px solid #e87a7a;border-radius:14px;'
            f'padding:12px 16px;margin-bottom:12px;font-size:13.5px;color:#7a1010;">'
            f'🔴 <strong>Care team notice:</strong> {msg}'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif tier == 3:
        msg = urg.get("patient_message") or (
            "Please call 911 or go to your nearest emergency room immediately. "
            "Your care team has been notified."
        )
        st.markdown(
            f'<div style="background:#ff0000;border-radius:14px;padding:16px 18px;'
            f'margin-bottom:16px;font-size:14px;color:white;font-weight:700;">'
            f'🚨 URGENT: {msg}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_urgency_indicator_html() -> str:
    """Return a small coloured dot HTML for the sidebar."""
    tier = st.session_state.get("urgency_state", {}).get("current_tier", 0)
    colours = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444", 3: "#991b1b"}
    labels  = {0: "All clear", 1: "Monitoring", 2: "Urgent", 3: "Emergency"}
    c = colours.get(tier, "#22c55e")
    l = labels.get(tier, "")
    return (
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{c};margin-right:5px;vertical-align:middle;"></span>'
        f'<span style="font-size:11px;color:{c};">{l}</span>'
    )


# ══════════════════════════════════════════════════════════════════
# REPORT GENERATION — Doctor-Facing Report Agent
# ══════════════════════════════════════════════════════════════════

_REPORT_AGENT_SYS = f"""
You are the Doctor-Facing Report Agent for ChatReport, a clinical symptom check-in
chatbot for head and neck cancer (HNC) patients. {_HNC_CONTEXT}

You synthesise all collected session data into a structured clinical pre-visit report
for the treating physician.

THE DOCTOR'S READING CONTEXT:
An oncologist or NP reviewing reports before clinic appointments.
May read multiple reports. They need to:
  • Know in 10 seconds: is action required today?
  • Understand patient's current status in 30 seconds.
  • Have full symptom detail available if needed.
  • Know what to address or follow up at the visit.

REPORT FORMAT — use this exact structure:
---
CHATREPORT — PRE-VISIT CLINICAL SUMMARY
Patient: [name]  |  Date: [date]
═══════════════════════════════════════════════════

🔴 FLAGS FOR PROVIDER ATTENTION
[List ONLY items matching red flag criteria — each as a concise bullet.
 Include: urgency signals detected, medication stops, crisis signals.
 If none: "No urgent flags identified."]

📋 CLINICAL OVERVIEW
[2-3 sentences: current status, most prominent issues, notable changes since last visit.
 Written for a clinician who has 10 seconds to orient.]

📊 SYMPTOM DETAILS BY DOMAIN
[One bold subsection per completed topic. Include:
 - Symptom presence/severity in clinical language
 - Patient-reported management strategies and medications
 - Functional impact where reported
 - Comparison to last visit where available
 - Clinically meaningful direct quotes in quotation marks]

💊 MEDICATION SUMMARY
[List all medications mentioned, doses/frequencies if reported, adherence status,
 and whether they are helping. Note any stops or side effects reported.]

🗣️ PATIENT ENGAGEMENT NOTES
[Only include if engagement was notable — e.g., declining engagement, distress,
 stoic underreporting, or patient explicitly shared emotional content.
 Omit this section if engagement was unremarkable.]

✅ SUGGESTED DISCUSSION POINTS
[2-4 bullets: items for provider to address or follow up — medication adjustment,
 referral, patient education need, unresolved concern.
 Do NOT repeat red flags already listed above.]
---

CLINICAL LANGUAGE RULES:
- Convert patient language to clinical terms where appropriate
  (e.g., "sore in my mouth" → "oral mucositis", "can't swallow" → "dysphagia")
- Include patient's own words in quotes only when clinically meaningful
- Omit topics with no data — do not write "N/A"
- Third person throughout ("Patient reports…")
- Never write "Unfortunately" or emotional commentary

RED FLAGS TO SCREEN FOR:
{_RED_FLAGS}

Urgency state will be provided. If emergency_tier >= 2, open with the flags section prominently.
Write only the completed report. No AI disclaimers or generation notes.
"""


def generate_report(name: str, all_data: dict) -> str:
    """
    Doctor-Facing Report Agent: synthesise all session data into a clinical report.
    Falls back to plain-text summary if OpenAI is unavailable.
    """
    topic_summaries = {}
    for label, key in TOPICS:
        d = all_data.get(key, {})
        if d:
            topic_summaries[label] = d

    if not openai_client:
        lines = [
            "CHATREPORT — PRE-VISIT CLINICAL SUMMARY",
            f"Patient: {name}  |  Date: {datetime.now().strftime('%B %d, %Y')}",
            "=" * 56, "",
        ]
        for label, data in topic_summaries.items():
            lines.append(f"[ {label.upper()} ]")
            for k, v in data.items():
                val = ", ".join(v) if isinstance(v, list) else str(v)
                lines.append(f"  - {k.replace('_', ' ').title()}: {val}")
            lines.append("")
        return "\n".join(lines)

    today    = datetime.now().strftime("%B %d, %Y")
    urg_tier = st.session_state.get("urgency_state", {}).get("current_tier", 0)
    urg_sigs = st.session_state.get("urgency_state", {}).get("all_signals", [])

    # Collect doctor notes from topic states
    doctor_notes = {}
    for _, key in TOPICS:
        topic_state = st.session_state.topic_states.get(key, {})
        notes = [
            v for k, v in topic_state.items()
            if k.endswith("_doctor_note") and v
        ]
        if notes:
            doctor_notes[key] = notes

    sentiment_notes = []
    for _, key in TOPICS:
        state = st.session_state.topic_states.get(key, {})
        note = state.get("_sentiment_note")
        if note:
            sentiment_notes.append(note)

    # Build prompt payload — note notes already collected above
    data_json = json.dumps({
        "patient_name": name,
        "report_date": today,
        "symptom_data_by_topic": topic_summaries,
        "freeform_notes": all_data.get("freeform_notes", []),
        "urgency_tier": urg_tier,
        "urgency_signals_active": urg_sigs,
        "last_checkin_data": st.session_state.get("last_checkin", {}),
    }, indent=2, ensure_ascii=False)

    report_prompt = f"DATA:\n{data_json}\n\nGenerate the clinical report following the format in your instructions."

    if not openai_client:
        return "Report generation unavailable — OpenAI API not configured."

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": _REPORT_AGENT_SYS},
                {"role": "user",   "content": report_prompt},
            ],
            max_tokens=2500,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Report generation failed: {e}"


def _default_chatty_reply(
    topic_key: str,
    answer: str,
    step: dict,
    last_topic_data: dict,
) -> str:
    """Fallback acknowledgment when agents are unavailable."""
    prev_same = _short_prev_answer(last_topic_data.get(step["id"])) if last_topic_data else ""
    if prev_same and prev_same.lower() != answer.strip().lower():
        return "I've noted that — it sounds a bit different from last time, which is helpful for your team to know."
    if topic_key == "mood":
        return "That sounds like a lot to carry. I've made a note of it for your care team."
    if topic_key == "pain":
        return "I've noted those pain details so your team can see exactly how it's been feeling."
    return "I've noted that detail for your care team."



# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "app_stage":           "login",
        "patient_name":        "",
        "selected_topic":      None,
        "topic_states": {
            key: {
                "status": "not_started",
                "data": {},
                "chat": [],
                "followup_counts": {},
                "raw_answers": {},
            }
            for _, key in TOPICS
        },
        "report":              "",
        "report_saved":        False,
        "last_checkin":        {},
        "has_prev_checkin":    False,
        "freeform_chat":       [],
        "urgency_state": {
            "current_tier": 0,
            "all_signals": [],
            "escalation_shown": False,
            "emergency_shown": False,
            "patient_message": None,
        },
        "sentiment_state": {
            "all_signals": [],
            "engagement_trajectory": "insufficient_data",
            "emotional_state": "neutral",
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ══════════════════════════════════════════════════════════════════
# LOAD PREVIOUS CHECK-IN
# ══════════════════════════════════════════════════════════════════

def load_last_checkin(name: str) -> dict:
    """
    Fetch the most recent saved check-in for this patient from Google Sheets.
    Returns a dict keyed by topic_key -> {q_id: answer}, or {} if none found.
    """
    _init_sheets()
    if _sheet is None:
        return {}
    try:
        rows = _sheet.get_all_values()
        last_row = None
        for row in rows[1:]:
            if len(row) >= 3 and row[1].strip().lower() == name.strip().lower():
                last_row = row          # keep iterating — last match wins
        if last_row:
            raw = json.loads(last_row[2])
            # raw is {topic_key: {q_id: answer}}
            return raw
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════════════════════
# TOPIC SUMMARY FORMATTER  (rule-based, no LLM)
# ══════════════════════════════════════════════════════════════════

# Key fields to surface per topic — (field_id, short_label)
_SUMMARY_FIELDS = {
    "pain": [
        ("has_pain",            "Pain today"),
        ("pain_location",       "Location"),
        ("throat_severity",     "Throat severity"),
        ("tongue_severity",     "Tongue severity"),
        ("pain_medications",    "Medications"),
        ("taking_as_prescribed","Adherence"),
    ],
    "nutrition": [
        ("eating_ability",        "Eating"),
        ("weight",                "Weight (lbs)"),
        ("swallowing_difficulty", "Swallowing"),
        ("feeding_tube",          "Feeding tube"),
        ("iv_fluids",             "IV fluids"),
        ("taste_changes",         "Taste changes"),
    ],
    "oral": [
        ("mouth_sores",    "Mouth sores"),
        ("dry_mouth",      "Dry mouth"),
        ("mucus_issues",   "Mucus"),
        ("oral_rinse_use", "Oral rinse"),
    ],
    "gi": [
        ("nausea_vomiting", "Nausea/vomiting/diarrhea"),
        ("constipation",    "Constipation"),
    ],
    "fatigue": [
        ("fatigue",           "Fatigue"),
        ("sleep_quality",     "Sleep"),
        ("medication_drowsy", "Medication drowsiness"),
    ],
    "activity": [
        ("activity_level",           "Activity level"),
        ("activity_limiting_factor", "Limiting factor"),
    ],
    "mood": [
        ("feeling_down",      "Feeling down"),
        ("support_adequate",  "Support"),
        ("anxiety_impact",    "Anxiety impact"),
    ],
    "other": [
        ("breathing_issues",  "Breathing"),
        ("hearing_changes",   "Hearing"),
        ("dizziness",         "Dizziness"),
        ("skin_issues",       "Skin"),
        ("voice_hoarseness",  "Voice"),
        ("fever_chills",      "Fever/chills"),
    ],
}


def _checkin_summary_html(topic_key: str, data: dict) -> str:
    """
    Build a chip-grid HTML block showing key facts from the previous check-in.
    Each field becomes a small pill: Label on top, value below.
    Returns an HTML string, or "" if no data to show.
    """
    fields = _SUMMARY_FIELDS.get(topic_key, [])
    chips  = []

    for field_id, label in fields:
        val = data.get(field_id)
        if val is None:
            continue
        field_type = QUESTION_TYPE_BY_ID.get(field_id, "options")
        if isinstance(val, list):
            if field_id == "pain_medications" and "Other" in val and data.get("pain_medications_other_detail"):
                val = [
                    data["pain_medications_other_detail"] if item == "Other" else item
                    for item in val
                ]
            val_str = ", ".join(str(v) for v in val)
        else:
            val_str = str(val)
        val_str = val_str.strip()
        if not val_str:
            continue
        if len(val_str) > 35:
            val_str = val_str[:32] + "…"

        is_option_value = field_type in {"options", "multi_select"}
        chip_bg = "#fff7ed" if is_option_value else "#f4f8ff"
        chip_border = "#fdba74" if is_option_value else "#d0e0f8"
        label_color = "#9a6a1a" if is_option_value else "#8fa8c8"
        value_color = "#c2410c" if is_option_value else "#1e3a5f"

        chips.append(
            f'<div style="display:inline-flex;flex-direction:column;'
            f'background:{chip_bg};border:1px solid {chip_border};'
            f'border-radius:10px;padding:5px 13px 6px 13px;'
            f'min-width:70px;max-width:200px;">'            f'<span style="font-size:10px;color:{label_color};font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.4px;'
            f'margin-bottom:2px;">{_html.escape(label)}</span>'            f'<span style="font-size:13px;color:{value_color};font-weight:700;'
            f'line-height:1.3;">{_html.escape(val_str)}</span>'            f'</div>'
        )

    if not chips:
        return ""

    return (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;padding:4px 0 6px 0;">'        + "".join(chips)        + '</div>'
    )


# ── Sidebar summary: natural-language sentence per topic ─────────

def _natural_summary(topic_key: str, data: dict) -> str:
    """Return a short natural-language sentence summarising a topic's previous answers."""

    def v(field):
        val = data.get(field)
        if isinstance(val, list):
            return val if val else None
        return val if val is not None else None

    def yn(field):
        return v(field) == "Yes"

    if topic_key == "pain":
        if v("has_pain") == "No":
            return "No pain"
        loc  = (v("pain_location") or "").lower()
        sev  = v("throat_severity") or v("tongue_severity")
        meds = v("pain_medications")
        other_med = v("pain_medications_other_detail")
        med_str = ""
        if isinstance(meds, list) and "No pain medication" not in meds:
            meds = [other_med if item == "Other" and other_med else item for item in meds]
            med_str = f", on {meds[0]}" if len(meds) == 1 else f", on {meds[0]} + {len(meds)-1} more"
        if loc and sev is not None:
            return f"{loc.capitalize()} pain ({sev}/10){med_str}"
        elif loc:
            return f"{loc.capitalize()} pain{med_str}"
        return f"Pain reported{med_str}"

    elif topic_key == "nutrition":
        eating = v("eating_ability") or ""
        weight = v("weight")
        w_str  = f", {weight} lbs" if weight else ""
        if "normally" in eating:
            return f"Eating normally{w_str}"
        elif "less" in eating:
            return f"Eating less than usual{w_str}"
        elif "Struggling" in eating:
            return f"Struggling, liquids only{w_str}"
        elif "tube" in eating.lower():
            return f"On feeding tube{w_str}"
        return f"Nutrition assessed{w_str}"

    elif topic_key == "oral":
        syms = []
        if yn("mouth_sores"):      syms.append("mouth sores or thrush")
        if yn("dry_mouth"):        syms.append("dry mouth")
        if yn("mucus_issues"):     syms.append("sticky mucus")
        if yn("teeth_gum_issues"): syms.append("gum problems")
        return ", ".join(syms).capitalize() if syms else "No oral symptoms"

    elif topic_key == "gi":
        syms = []
        nv = v("nausea_vomiting") or []
        if "Nausea" in nv:              syms.append("nausea")
        if "Vomiting" in nv:            syms.append("vomiting")
        if "Diarrhea" in nv:            syms.append("diarrhea")
        if yn("constipation"):          syms.append("constipation")
        return ", ".join(syms).capitalize() if syms else "No GI symptoms"

    elif topic_key == "fatigue":
        fatigue = v("fatigue")
        sleep   = v("sleep_quality")
        if fatigue == "No" and sleep == "Yes":
            return "No fatigue, sleeping well"
        elif fatigue == "Yes" and sleep == "No":
            return "Fatigued, trouble sleeping"
        elif fatigue == "Yes":
            return "Feeling fatigued"
        elif sleep == "No":
            return "Trouble sleeping"
        return "Fatigue assessed"

    elif topic_key == "activity":
        level = v("activity_level") or ""
        if "normally" in level:
            return "Fully active"
        elif "less" in level:
            cause = (v("activity_limiting_factor") or "").lower()
            return f"Less active — {cause}" if cause else "Less active than usual"
        elif "Struggling" in level:
            return "Struggling with daily tasks"
        return "Activity assessed"

    elif topic_key == "mood":
        parts = []
        if v("feeling_down") == "Yes":       parts.append("feeling depressed")
        if v("support_adequate") == "No":    parts.append("limited support")
        elif v("support_adequate") == "Yes": parts.append("good support")
        if v("anxiety_impact") == "Yes":     parts.append("anxiety affecting daily life")
        return ", ".join(parts[:2]).capitalize() if parts else "Mood assessed"

    elif topic_key == "other":
        syms = []
        if yn("breathing_issues"): syms.append("breathing difficulty")
        if yn("hearing_changes"):  syms.append("hearing changes")
        if yn("dizziness"):        syms.append("dizziness")
        if yn("fever_chills"):     syms.append("fever/chills")
        if yn("skin_issues"):      syms.append("skin issues")
        if v("voice_hoarseness") == "Yes, problems with my voice":
            syms.append("voice changes")
        return ", ".join(syms[:3]).capitalize() if syms else "No other symptoms"

    return ""


def _scenario_conversation_markdown(lines: list[tuple[str, str]]) -> str:
    rendered = []
    for speaker, text in lines:
        rendered.append(f"**{speaker}:** {text}")
    return "\n\n".join(rendered)


def _render_topic_scenario_pass(topic_key: str):
    scenario = TOPIC_SCENARIO_PASSES.get(topic_key)
    if not scenario:
        st.caption("No scenario pass has been added for this topic yet.")
        return

    st.markdown(f"**Scenario:** {scenario['title']}")
    st.caption(scenario["why_it_matters"])
    st.markdown(_scenario_conversation_markdown(scenario["conversation"]))
    st.markdown("**Agent workflow pass**")
    for item in scenario["agent_pass"]:
        st.markdown(f"- {item}")


def _render_scenario_library():
    st.markdown("### Multi-Agent Scenario Pass")
    st.caption(
        "These realistic sample conversations show how each topic should behave across interpretation, triage, and next-step questioning."
    )
    for _, topic_key in TOPICS:
        label = TOPIC_LABELS.get(topic_key, topic_key)
        with st.expander(label, expanded=False):
            _render_topic_scenario_pass(topic_key)


# ══════════════════════════════════════════════════════════════════
# FREE-FORM CHAT LLM
# ══════════════════════════════════════════════════════════════════

def _freeform_llm_response(messages: list) -> str:
    """
    Generate a nurse reply in the free-form "Anything else?" chat.
    messages is the full conversation history: [{role, content}, ...].
    """
    if not openai_client:
        return "I'm not able to respond right now — please let your care team know directly."

    structured_context = {
        key: st.session_state.topic_states[key]["data"]
        for _, key in TOPICS
        if st.session_state.topic_states[key]["data"]
    }
    prior_context = st.session_state.get("last_checkin", {})

    hnc_context = (
        "You are a compassionate, clinically trained nurse at a head and neck cancer "
        "(HNC) center conducting a structured symptom check-in with a patient currently "
        "receiving chemoradiation or surgery for head and neck cancer. "
        "This patient population frequently experiences: severe mucositis, dysphagia, "
        "pain, significant weight loss, fatigue, depression, and impaired communication. "
        "Many patients have low health literacy or face barriers to care. "
        "Your tone is always warm, clear, and non-alarming. Never use medical jargon "
        "without explaining it simply. Never minimize a patient's reported symptom."
    )
    system = (
        f"{hnc_context}\n\n"
        "You are now in an open conversation with the patient. They may raise anything not "
        "covered by the structured check-in — a new symptom, a question about their treatment, "
        "a concern about a medication, or just something they want their provider to know.\n\n"
        f"CURRENT STRUCTURED CHECK-IN DATA:\n{json.dumps(structured_context, indent=2)}\n\n"
        f"MOST RECENT PRIOR CHECK-IN DATA:\n{json.dumps(prior_context, indent=2)}\n\n"
        "Guidelines:\n"
        "- Listen carefully and respond with warmth and clinical awareness.\n"
        "- If helpful, briefly notice whether this sounds better, worse, or different than last visit.\n"
        "- If they mention a symptom that sounds urgent (e.g., chest pain, breathing difficulty, "
        "  high fever, blood, suicidal thoughts), acknowledge it calmly and tell them it will be "
        "  flagged for their care team.\n"
        "- Do NOT diagnose or prescribe. You are gathering information, not treating.\n"
        "- Be conversational and natural, not stiff or repetitive.\n"
        "- Keep responses short — 2-4 sentences maximum.\n"
        "- If the patient seems to be done, gently close: 'Is there anything else you'd like "
        "  to share with your team before your visit?'"
    )

    api_messages = [{"role": "system", "content": system}]
    for m in messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    try:
        r = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=api_messages,
            max_tokens=200,
            temperature=0.5,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return "I'm having trouble responding right now — please share this with your care team directly."


# ══════════════════════════════════════════════════════════════════
# Conversational answer handling
# ══════════════════════════════════════════════════════════════════

def _append_next_question(
    topic_key: str,
    state: dict,
    next_step: Optional[dict],
    assistant_message: str = "",
):
    message = assistant_message.strip()
    next_text = _step_prompt_text(next_step, topic_key=topic_key, state=state) if next_step else ""
    if message and next_text and _is_semantically_redundant_question(message, next_text):
        message = ""
    if message:
        _append_assistant_message(state, message)
    if next_text:
        _append_assistant_message(state, next_text)


def _store_followup_prompt(
    topic_key: str,
    state: dict,
    step: dict,
    question: str,
    assistant_message: str = "",
    retry_current_step: bool = False,
    allow_other_detail: bool = False,
):
    if _looks_vague_answer(state["data"].get(step["id"], "")):
        assistant_message = ""
    state["waiting_for_followup"] = True
    state["pending_followup"] = {
        "source_step_id": step["id"],
        "question": question,
        "answer_key": f"{step['id']}_llm_followup",
        "assistant_message": assistant_message.strip(),
        "retry_current_step": retry_current_step,
        "allow_other_detail": allow_other_detail,
    }
    combined_prompt = "\n\n".join([part for part in [assistant_message.strip(), question.strip()] if part])
    _append_assistant_message(state, combined_prompt)


def _request_retry_for_step(topic_key: str, step: dict, raw_input: str, source: str = "typed"):
    state = st.session_state.topic_states[topic_key]
    text = (raw_input or "").strip()
    if text:
        state["chat"].append({"role": "user", "content": text})
    _store_followup_prompt(
        topic_key,
        state,
        step,
        _build_retry_prompt(step, text),
        retry_current_step=True,
        allow_other_detail=("Other" in step.get("opts", [])),
    )
    st.rerun()


def _clear_step_inputs(topic_key: str, step: dict):
    sid = step["id"]
    stype = step["type"]

    keys_to_clear = []
    if stype == "options":
        keys_to_clear.extend([
            f"text_{topic_key}_{sid}",
            f"text_{topic_key}_{sid}_submitted",
            f"dropdown_{topic_key}_{sid}",
            f"dropdown_{topic_key}_{sid}_submitted",
            f"voice_{topic_key}_{sid}_submitted",
            f"_vt_{topic_key}_{sid}_opt",
            f"_vh_{topic_key}_{sid}_opt",
        ])
    elif stype == "multi_select":
        keys_to_clear.extend([
            f"text_{topic_key}_{sid}",
            f"text_{topic_key}_{sid}_submitted",
            f"dropdown_{topic_key}_{sid}",
            f"dropdown_{topic_key}_{sid}_submitted",
            f"voice_{topic_key}_{sid}_submitted",
            f"_vt_{topic_key}_{sid}_multi",
            f"_vh_{topic_key}_{sid}_multi",
        ])
    elif stype == "number":
        keys_to_clear.extend([
            f"text_{topic_key}_{sid}",
            f"text_{topic_key}_{sid}_submitted",
            f"_vt_{topic_key}_{sid}_num",
            f"_vh_{topic_key}_{sid}_num",
        ])
    elif stype == "free_text":
        keys_to_clear.extend([
            f"ft_{topic_key}_{sid}",
            f"ft_{topic_key}_{sid}_submitted",
            f"ft_{topic_key}_{sid}_voice_sync",
            f"_vt_{topic_key}_{sid}",
            f"_vh_{topic_key}_{sid}",
        ])

    for key in keys_to_clear:
        st.session_state.pop(key, None)


def handle_pending_followup(topic_key: str, answer: str, source: str = "typed"):
    state = st.session_state.topic_states[topic_key]
    pending = state.get("pending_followup") or {}
    answer_key = pending.get("answer_key")
    if not answer_key:
        state["waiting_for_followup"] = False
        state.pop("pending_followup", None)
        st.rerun()
        return

    if pending.get("retry_current_step"):
        source_step_id = pending.get("source_step_id")
        source_step = STEP_BY_ID.get(source_step_id)
        state["waiting_for_followup"] = False
        state.pop("pending_followup", None)
        if not source_step:
            st.rerun()
            return

        retry_text = (answer or "").strip()
        if source_step["type"] == "options":
            interpreted = interpret_user_input_with_options(source_step, retry_text)
            if interpreted in source_step.get("opts", []):
                handle_answer(
                    topic_key,
                    source_step,
                    interpreted,
                    source=source,
                    raw_answer=retry_text,
                )
                return
            _request_retry_for_step(topic_key, source_step, retry_text, source=source)
            return

        if source_step["type"] == "multi_select":
            parsed = parse_multi_select_typed_input(source_step, retry_text)
            if parsed:
                handle_answer(topic_key, source_step, parsed, source=source)
                return
            if pending.get("allow_other_detail") and retry_text and not _looks_vague_answer(retry_text):
                state["data"][f"{source_step['id']}_other_detail"] = retry_text
                handle_answer(
                    topic_key,
                    source_step,
                    ["Other"],
                    source=source,
                    display_override=retry_text,
                )
                return
            _request_retry_for_step(topic_key, source_step, retry_text, source=source)
            return

    state["chat"].append({"role": "user", "content": answer})
    state["data"][answer_key] = answer
    pending_key = f"pending_followup_{topic_key}_{pending.get('answer_key', 'pending')}"
    st.session_state.pop(pending_key, None)
    submitted_pending_key = f"{pending_key}_submitted"
    st.session_state.pop(submitted_pending_key, None)
    st.session_state.pop(f"{pending_key}_voice_sync", None)
    state["waiting_for_followup"] = False
    state.pop("pending_followup", None)

    last_topic_data = st.session_state.last_checkin.get(topic_key, {})
    closing = _default_chatty_reply(
        topic_key,
        answer,
        {"id": answer_key, "text": pending.get("question", "")},
        last_topic_data,
    )

    next_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
    state["status"] = "in_progress"

    if topic_is_complete(topic_key, state["data"], state.get("raw_answers")):
        state["status"] = "completed"
        state["chat"].append({
            "role": "assistant",
            "content": f"{closing}\n\n✅ Thank you — I have everything I need for this topic."
        })
    else:
        _append_next_question(topic_key, state, next_step, closing)

    st.rerun()
    return


def handle_answer(
    topic_key: str,
    step: dict,
    answer,
    source: str = "structured",
    display_override: Optional[str] = None,
    raw_answer: Any = None,
):
    """
    Core answer handler — orchestrates all agents and determines next action.
    For structured button clicks (source='structured') we skip the full agent
    pipeline and just do a quick comparison + vague-check to keep latency low.
    For free-text, voice, and typed answers we run the full multi-agent pipeline.
    """
    state = st.session_state.topic_states[topic_key]

    # ── Ensure followup_counts dict exists (backward compat) ──────
    if "followup_counts" not in state:
        state["followup_counts"] = {}
    if "raw_answers" not in state:
        state["raw_answers"] = {}
    _clear_step_inputs(topic_key, step)

    display = display_override if display_override is not None else (
        ", ".join(answer) if isinstance(answer, list) else str(answer)
    )
    state["chat"].append({"role": "user", "content": display})
    verbatim = raw_answer if raw_answer is not None else display
    if isinstance(verbatim, str) and verbatim.strip():
        state["raw_answers"][step["id"]] = verbatim.strip()
    if (
        step.get("type") == "multi_select"
        and isinstance(answer, list)
        and "Other" in answer
        and isinstance(verbatim, str)
        and verbatim.strip()
    ):
        state["data"][f"{step['id']}_other_detail"] = verbatim.strip()
    answer = _coerce_structured_answer(topic_key, step, answer, state["data"], raw_answer=raw_answer)
    state["data"][step["id"]] = answer
    if isinstance(verbatim, str):
        _auto_capture_following_answers(topic_key, state, verbatim)
    next_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
    state["status"] = "in_progress"

    last_topic_data = st.session_state.last_checkin.get(topic_key, {})

    # ══════════════════════════════════════════════════════════════
    # BRANCH A — Structured button / dropdown click (fast path, no agents)
    # source="structured" means the patient clicked a predefined option —
    # it is already a clean matched answer, no LLM classification needed.
    # ══════════════════════════════════════════════════════════════
    if source == "structured":
        if topic_is_complete(topic_key, state["data"], state.get("raw_answers")):
            state["status"] = "completed"
            state["chat"].append({
                "role": "assistant",
                "content": "✅ Thank you — I have everything I need for this topic.",
            })
            st.rerun()
            return
        _append_next_question(topic_key, state, next_step)
        st.rerun()
        return

    # ══════════════════════════════════════════════════════════════
    # BRANCH B — Free text / voice / typed — run full agent pipeline
    # ══════════════════════════════════════════════════════════════
    if isinstance(answer, str):
        is_vague = _looks_vague_answer(answer)

        # Vague answer with no options to try → ask clarification (no LLM needed)
        if is_vague and source in {"typed", "voice", "free_text"}:
            _store_followup_prompt(
                topic_key, state, step, _fallback_clarifying_question(step),
            )
            st.rerun()
            return

        if openai_client:
            with st.spinner("Thinking…"):
                pipeline = run_agent_pipeline(
                    topic_key=topic_key,
                    step=step,
                    answer=answer,
                    state=state,
                    last_topic_data=last_topic_data,
                )

            # ── Emergency: terminate session ──────────────────────
            if pipeline.get("urgency_tier", 0) == 3:
                emergency_msg = pipeline.get("urgency_message") or (
                    "We are concerned about what you've shared. Please call 911 or "
                    "go to your nearest emergency room immediately. "
                    "Your care team has been notified."
                )
                state["chat"].append({
                    "role": "assistant",
                    "content": f"🚨 {emergency_msg}",
                })
                state["status"] = "completed"   # Lock this topic
                st.session_state["urgency_state"]["emergency_shown"] = True
                st.rerun()
                return

            # ── Tier 2: insert care team notice into chat ─────────
            tier2_msg = pipeline.get("urgency_message")
            if tier2_msg:
                state["chat"].append({"role": "assistant", "content": f"🔴 {tier2_msg}"})

            # ── Acknowledgment before follow-up / next question ───
            ack = pipeline.get("acknowledgment", "").strip()

            # ── Patient wants to stop ─────────────────────────────
            if pipeline.get("wants_to_stop"):
                closing = "Of course — we'll pause here. The answers you've shared have been saved for your care team."
                if ack:
                    closing = f"{ack}\n\n{closing}"
                state["chat"].append({"role": "assistant", "content": closing})
                state["status"] = "completed"
                st.rerun()
                return

            # ── Tier 2: avoid detached follow-ups in the same turn ─
            if tier2_msg:
                state["chat"].append({
                    "role": "assistant",
                    "content": "We'll pause this topic here for now so your care team can follow up directly.",
                })
                state["status"] = "completed"
                st.rerun()
                return

            # ── Follow-up question ────────────────────────────────
            if pipeline.get("follow_up") and pipeline.get("follow_up_question"):
                fq = pipeline["follow_up_question"]
                # Reject if semantically redundant with original question
                if _is_redundant_followup(step["text"], answer, fq):
                    pass   # Fall through to assistant_message + next question
                else:
                    # Increment follow-up counter
                    fc = state["followup_counts"]
                    fc[step["id"]] = fc.get(step["id"], 0) + 1
                    _store_followup_prompt(
                        topic_key, state, step, fq, ack,
                    )
                    st.rerun()
                    return

            # ── Store doctor note and signals for report ──────────
            if pipeline.get("doctor_note"):
                state[f"{step['id']}_doctor_note"] = pipeline["doctor_note"]
            if pipeline.get("sentiment_note"):
                state["_sentiment_note"] = pipeline["sentiment_note"]

            assistant_message = pipeline.get("assistant_message", "").strip()
            if ack and assistant_message:
                assistant_message = f"{ack}\n\n{assistant_message}"
            elif ack:
                assistant_message = ack

        else:
            # No OpenAI — use fallback reply
            assistant_message = _default_chatty_reply(
                topic_key, answer, step, last_topic_data
            )

    else:
        # Non-string answer (numeric, list from multi_select on structured path)
        assistant_message = ""

    # ── Topic complete check ──────────────────────────────────────
    if topic_is_complete(topic_key, state["data"], state.get("raw_answers")):
        state["status"] = "completed"
        final_message = "✅ Thank you — I have everything I need for this topic."
        if assistant_message:
            final_message = f"{assistant_message}\n\n{final_message}"
        state["chat"].append({"role": "assistant", "content": final_message})
        st.rerun()
        return

    _append_next_question(topic_key, state, next_step, assistant_message)
    st.rerun()
    return


# ══════════════════════════════════════════════════════════════════
# INPUT RENDERING
# ══════════════════════════════════════════════════════════════════


def render_input(topic_key: str, step: dict, prev_answer=None):
    """Render the appropriate input widget for the current question."""
    stype = step["type"]
    sid   = step["id"]


    state = st.session_state.topic_states[topic_key]
    prev = state["data"].get(step["id"])

    def render_option_buttons(button_topic_key: str, button_step: dict, multi: bool = False):
        opts = button_step.get("opts", [])
        if not opts:
            return
        cols_per_row = 2 if len(opts) > 1 else 1
        for idx in range(0, len(opts), cols_per_row):
            row = st.columns(cols_per_row)
            for offset, opt in enumerate(opts[idx:idx + cols_per_row]):
                with row[offset]:
                    if st.button(opt, key=f"btn_{button_topic_key}_{button_step['id']}_{idx + offset}", use_container_width=True):
                        payload = [opt] if multi else opt
                        handle_answer(button_topic_key, button_step, payload, source="structured")
                        return

    # ── Options ─────────────────────────────────────────────────
    if stype == "options":
        st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
        user_text = st.text_input(
            "Message",
            key=f"text_{topic_key}_{sid}",
            label_visibility="collapsed",
            placeholder="Type a reply..."
        )
        render_option_buttons(topic_key, step, multi=False)

        with st.container():
            voice_text = voice_widget(f"{topic_key}_{sid}_opt", label="Mic")

        submitted_key = f"text_{topic_key}_{sid}_submitted"

        if user_text and st.session_state.get(submitted_key) != user_text:
            st.session_state[submitted_key] = user_text
            interpreted = interpret_user_input_with_options(step, user_text)
            if interpreted in step.get("opts", []):
                handle_answer(topic_key, step, interpreted, source="structured",
                              display_override=user_text, raw_answer=user_text)
            else:
                _request_retry_for_step(topic_key, step, user_text, source="typed")
                return

        voice_submitted_key = f"voice_{topic_key}_{sid}_submitted"
        if voice_text and st.session_state.get(voice_submitted_key) != voice_text:
            st.session_state[voice_submitted_key] = voice_text
            interpreted = interpret_user_input_with_options(step, voice_text)
            if interpreted in step.get("opts", []):
                handle_answer(topic_key, step, interpreted, source="structured",
                              display_override=voice_text, raw_answer=voice_text)
            else:
                _request_retry_for_step(topic_key, step, voice_text, source="voice")
                return
        st.markdown('</div>', unsafe_allow_html=True)
                

    # ── Multi-select ─────────────────────────────────────────────
    elif stype == "multi_select":
        st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
        text_key = f"text_{topic_key}_{sid}"
        submit_key = f"{text_key}_submitted"
        user_text = st.text_input(
            "Reply",
            key=text_key,
            label_visibility="collapsed",
            placeholder="Type one or more answers, separated by commas..."
        )
        render_option_buttons(topic_key, step, multi=True)
        with st.container():
            voice_text = voice_widget(f"{topic_key}_{sid}_multi", label="Mic")

        if user_text and st.session_state.get(submit_key) != user_text:
            st.session_state[submit_key] = user_text
            parsed = parse_multi_select_typed_input(step, user_text)
            if parsed:
                handle_answer(topic_key, step, parsed, source="structured",
                              display_override=user_text, raw_answer=user_text)
            else:
                _request_retry_for_step(topic_key, step, user_text, source="typed")
                return

        voice_submit_key = f"voice_{topic_key}_{sid}_submitted"
        if voice_text and st.session_state.get(voice_submit_key) != voice_text:
            st.session_state[voice_submit_key] = voice_text
            parsed = parse_multi_select_typed_input(step, voice_text)
            if parsed:
                handle_answer(topic_key, step, parsed, source="structured",
                              display_override=voice_text, raw_answer=voice_text)
            else:
                _request_retry_for_step(topic_key, step, voice_text, source="voice")
                return
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Number ───────────────────────────────────────────────────
    elif stype == "number":
        st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
        text_key = f"text_{topic_key}_{sid}"
        submit_key = f"{text_key}_submitted"
        if text_key not in st.session_state:
            st.session_state[text_key] = ""
        user_text = st.text_input(
            "Reply",
            key=text_key,
            label_visibility="collapsed",
            placeholder=f"Enter a number ({int(step['min_v'])}-{int(step['max_v'])})"
        )
        with st.container():
            voice_text = voice_widget(f"{topic_key}_{sid}_num", label="Mic")

        candidate = user_text or voice_text or ""
        if candidate and st.session_state.get(submit_key) != candidate:
            st.session_state[submit_key] = candidate
            try:
                val = int(float(candidate))
                if val < step["min_v"] or val > step["max_v"]:
                    st.warning(f"Please enter a value between {int(step['min_v'])} and {int(step['max_v'])}.")
                else:
                    handle_answer(topic_key, step, val, source="typed")
            except ValueError:
                st.warning("Please enter a number.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Free text ────────────────────────────────────────────────
    elif stype == "free_text":
        transcript_key = f"_vt_{topic_key}_{sid}"
        widget_key     = f"ft_{topic_key}_{sid}"
        submit_key     = f"{widget_key}_submitted"

        # Priority for pre-fill: voice transcript > empty
        transcript = st.session_state.get(transcript_key, "")
        if widget_key not in st.session_state:
            st.session_state[widget_key] = transcript or ""
        elif transcript and transcript != st.session_state.get(f"{widget_key}_voice_sync"):
            st.session_state[widget_key] = transcript
            st.session_state[f"{widget_key}_voice_sync"] = transcript

        st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
        with st.container():
            free_text = st.text_input(
                "Reply",
                placeholder=step.get("placeholder", "Please describe…"),
                key=widget_key,
                label_visibility="collapsed",
            )
        with st.container():
            voice_text = voice_widget(f"{topic_key}_{sid}", label="Mic")
        if voice_text and voice_text != st.session_state.get(f"{widget_key}_voice_sync"):
            st.session_state[f"{widget_key}_voice_sync"] = voice_text
            st.session_state[submit_key] = voice_text
            handle_answer(topic_key, step, voice_text, source="voice")
            return

        if free_text and st.session_state.get(submit_key) != free_text:
            st.session_state[submit_key] = free_text
            handle_answer(topic_key, step, free_text, source="free_text")

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# FREE-FORM CHAT PANEL
# ══════════════════════════════════════════════════════════════════

def render_freeform_chat():
    """Render the open-ended 'Anything else?' chatbot panel."""
    _stc.html("""<script>
    (function(){
        var s=['section[data-testid=\"stMain\"]',
               'div[data-testid=\"stAppViewContainer\"]','.main'];
        for(var i=0;i<s.length;i++){
            var e=window.parent.document.querySelector(s[i]);
            if(e){e.scrollTop=0;break;}
        }
    })();
    </script>""", height=0)
    st.markdown(
        '<div class="chat-shell">'
        '  <div class="chat-shell-header">'
        '    <div class="chat-shell-title">'
        '      <div class="chat-shell-avatar">💬</div>'
        '      <div class="chat-shell-title-text">'
        '        <div class="chat-shell-label">Open Conversation</div>'
        '        <div class="chat-shell-name">Anything else you’d like to share?</div>'
        '      </div>'
        '    </div>'
        '    <div class="chat-shell-note">Share symptoms, concerns, or questions for your team</div>'
        '  </div>'
        '  <div class="chat-history">',
        unsafe_allow_html=True,
    )

    # ── Initialise conversation ──────────────────────────────────
    if not st.session_state.freeform_chat:
        opening = (
            "Is there anything else you'd like your care team to know before your visit? "
            "Feel free to share any concerns, questions, or symptoms we haven't covered yet."
        )
        st.session_state.freeform_chat = [{"role": "assistant", "content": opening}]

    # ── Show history ─────────────────────────────────────────────
    chat_container = st.container(border=False)
    with chat_container:
        for msg in st.session_state.freeform_chat:
            render_chat_bubble(msg["role"], msg["content"])

    st.markdown('</div><div class="composer-wrap">', unsafe_allow_html=True)

    # ── Input ────────────────────────────────────────────────────
    user_input = st.chat_input("Type here, or use the voice button below…",
                                key="freeform_chat_input")

    with st.container():
        vt = voice_widget("freeform")
        if vt and not user_input:
            user_input = vt
            # Clear the transcript so it doesn't re-fire
            st.session_state.pop("_vt_freeform", None)

    if user_input and user_input.strip():
        # Avoid re-appending if already in history (Streamlit reruns)
        last_user = next(
            (m["content"] for m in reversed(st.session_state.freeform_chat)
             if m["role"] == "user"), None
        )
        if user_input.strip() != last_user:
            st.session_state.freeform_chat.append(
                {"role": "user", "content": user_input.strip()}
            )
            if openai_client:
                with st.spinner("…"):
                    reply = _freeform_llm_response(st.session_state.freeform_chat)
            else:
                reply = (
                    "Got it — I've noted that for your care team. "
                    "Is there anything else you'd like to add?"
                )
            st.session_state.freeform_chat.append(
                {"role": "assistant", "content": reply}
            )
            st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TOPIC DETAIL PANEL
# ══════════════════════════════════════════════════════════════════

def render_topic_detail(topic_label: str, topic_key: str):
    """Render the chat + current question for the selected topic."""
    _stc.html("""<script>
    (function(){
        var s=['section[data-testid=\"stMain\"]',
               'div[data-testid=\"stAppViewContainer\"]','.main'];
        for(var i=0;i<s.length;i++){
            var e=window.parent.document.querySelector(s[i]);
            if(e){e.scrollTop=0;break;}
        }
    })();
    </script>""", height=0)
    state        = st.session_state.topic_states[topic_key]
    last_data    = st.session_state.last_checkin.get(topic_key, {})
    has_prev     = st.session_state.has_prev_checkin

    # ── Ensure followup_counts exists (backward compat) ────────────
    if "followup_counts" not in state:
        state["followup_counts"] = {}

    # ── Urgency banner (Tier 1–3 from multi-agent system) ───────────
    render_urgency_banner()

    # ── Previous check-in summary card ────────────────────────────
    if has_prev:
        if last_data:
            chips_html = _checkin_summary_html(topic_key, last_data)
            if chips_html:
                with st.expander("Last visit summary", expanded=False):
                    st.caption("These answers are from your last visit. You can change any of them for this visit.")
                    st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.caption("No information from your last visit was recorded for this section.")

    with st.expander("Sample conversation for this topic", expanded=False):
        _render_topic_scenario_pass(topic_key)

    # ── Initialize topic on first visit ─────────────────────────
    if state["status"] == "not_started":
        state["status"] = "in_progress"
        intro = TOPIC_INTROS.get(topic_key, "Let's go through this section together.")
        state["chat"] = [{"role": "assistant", "content": intro}]
        first_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
        if first_step:
            _append_assistant_message(state, _step_prompt_text(first_step, topic_key=topic_key, state=state))

    # ── Header with progress bar ─────────────────────────────────
    answered, applicable = get_topic_progress(topic_key, state["data"], state.get("raw_answers"))
    progress_note = f"{answered}/{applicable} answered" if applicable > 0 else "Getting started"
    last_visit_summary = _natural_summary(topic_key, last_data) if last_data else ""
    header_html = (
        '<div class="chat-shell">'
        '  <div class="chat-shell-header">'
        '    <div class="chat-shell-title">'
        '      <div class="chat-shell-avatar">🩺</div>'
        '      <div class="chat-shell-title-text">'
        f'        <div class="chat-shell-label">Topic Check-In</div>'
        f'        <div class="chat-shell-name">{_html.escape(topic_label)}</div>'
        '      </div>'
        '    </div>'
        f'    <div class="chat-shell-note">{_html.escape(progress_note)}</div>'
        '  </div>'
        + (
            f'<div class="chat-shell-summary"><strong>Last visit:</strong> {_html.escape(last_visit_summary)}</div>'
            if last_visit_summary else
            '<div class="chat-shell-summary"><strong>Last visit:</strong> No prior summary recorded for this topic.</div>'
        )
        + '  <div class="chat-history">'
    )
    st.markdown(
        header_html,
        unsafe_allow_html=True,
    )

    # ── Chat history ─────────────────────────────────────────────
    if state["chat"]:
        with st.container(border=False):
            for msg in state["chat"]:
                render_chat_bubble(msg["role"], msg["content"])

    # ── Completed ────────────────────────────────────────────────
    if state["status"] == "completed":
        st.markdown(
            '<div class="completion-badge">✅ This topic is complete</div>',
            unsafe_allow_html=True,
        )
        if st.button("✏️ Add a note or correction", key=f"reopen_{topic_key}"):
            state["status"] = "in_progress"
            state["chat"].append({
                "role": "assistant",
                "content": "Of course — please share any correction or additional detail.",
            })
            state["data"].pop("_correction_note", None)
            st.rerun()
        st.markdown('</div><div class="composer-wrap"></div></div>', unsafe_allow_html=True)
        return

    # ── Current question ─────────────────────────────────────────
    if state.get("waiting_for_followup"):
        pending = state.get("pending_followup") or {}
        pending_suffix = pending.get("answer_key", "pending")
        pending_key = f"pending_followup_{topic_key}_{pending_suffix}"
        pending_submit_key = f"{pending_key}_submitted"
        if pending_key not in st.session_state:
            st.session_state[pending_key] = ""
        st.markdown('</div><div class="composer-wrap">', unsafe_allow_html=True)

        st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
        with st.container():
            pending_text = st.text_input(
                "Reply",
                key=pending_key,
                placeholder="Type or speak your answer here...",
                label_visibility="collapsed",
            )
        with st.container():
            pending_voice = voice_widget(f"pending_{topic_key}_{pending_suffix}", label="Mic")
        if pending_voice and pending_voice != st.session_state.get(f"{pending_key}_voice_sync"):
            st.session_state[f"{pending_key}_voice_sync"] = pending_voice
            st.session_state[pending_submit_key] = pending_voice
            handle_pending_followup(topic_key, pending_voice, source="voice")
            return

        if pending_text and st.session_state.get(pending_submit_key) != pending_text:
            st.session_state[pending_submit_key] = pending_text
            handle_pending_followup(topic_key, pending_text, source="followup")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        return
    next_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
    if next_step:
        # Look up previous answer for this specific question
        prev_answer = last_data.get(next_step["id"]) if last_data else None
        _append_assistant_message(state, _step_prompt_text(next_step, topic_key=topic_key, state=state))
        st.markdown('</div><div class="composer-wrap">', unsafe_allow_html=True)
        render_input(topic_key, next_step, prev_answer=prev_answer)
        st.markdown('</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR  (MASTER PANEL)
# ══════════════════════════════════════════════════════════════════



def render_sidebar():
    with st.sidebar:
        # ── Header ───────────────────────────────────────────────
        _urg_html = render_urgency_indicator_html()
        st.markdown(
            f'<div style="font-size:18px;font-weight:800;color:#10233d;margin:0 0 2px 0;letter-spacing:-0.03em;">🩺 ChatReport</div>'
            f'<div style="font-size:11px;color:#6b7b92;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.08em;">'
            f'Clinical symptom assistant &nbsp;{_urg_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.patient_name:
            st.markdown(
                f'<div style="font-size:11.5px;color:#6b7280;margin-bottom:8px;">'                f'Patient: <strong>{_html.escape(st.session_state.patient_name)}</strong></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<hr style="margin:6px 0 10px 0;border:none;border-top:1px solid #dde6f5;">',
            unsafe_allow_html=True,
        )

        # ── Overall progress ─────────────────────────────────────
        completed   = sum(1 for _, k in TOPICS
                          if st.session_state.topic_states[k]["status"] == "completed")
        in_progress = sum(1 for _, k in TOPICS
                          if st.session_state.topic_states[k]["status"] == "in_progress")
        total = len(TOPICS)

        st.markdown(
            f'<div class="prog-label">{completed}/{total} topics complete</div>',
            unsafe_allow_html=True,
        )
        st.progress(completed / total if total > 0 else 0)
        st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

        # ── Topic nav buttons ─────────────────────────────────────
        # Using st.button (not HTML anchors) so session state is preserved.
        # The label has two parts separated by \n:
        #   line 1: "{selected_marker}{icon} {topic_name}"
        #   line 2: summary snippet or "No prior data" (when prev check-in exists)
        # CSS white-space:pre-wrap renders the \n as a real line break.

        has_prev = st.session_state.get("has_prev_checkin", False)
        last_ck  = st.session_state.get("last_checkin", {})

        for label, key in TOPICS:
            status = st.session_state.topic_states[key]["status"]
            icon   = {"completed": "✅", "in_progress": "🔵"}.get(status, "⚪")
            dname  = label.split(" ", 1)[1] if " " in label else label
            marker = "▶ " if st.session_state.selected_topic == key else "   "

            # Build button label
            btn = f"{marker}{icon} {dname}"
            if has_prev:
                prev_data = last_ck.get(key, {})
                if prev_data:
                    snip = _natural_summary(key, prev_data)
                    btn += f"\n   {snip}" if snip else "\n   No data recorded"
                else:
                    btn += "\n   No prior data"

            if st.button(btn, key=f"nav_{key}", use_container_width=True):
                st.session_state.selected_topic = key
                st.rerun()

        # ── Anything else? ────────────────────────────────────────
        ff_msgs  = [m for m in st.session_state.freeform_chat if m["role"] == "user"]
        ff_badge = f" ({len(ff_msgs)})" if ff_msgs else ""
        ff_mark  = "▶ " if st.session_state.selected_topic == "freeform" else "   "
        if st.button(f"{ff_mark}💬 Anything else?{ff_badge}",
                     key="nav_freeform", use_container_width=True):
            st.session_state.selected_topic = "freeform"
            st.rerun()

        # ── Submit ────────────────────────────────────────────────
        st.markdown(
            '<hr style="margin:8px 0 8px 0;border:none;border-top:1px solid #dde6f5;">',
            unsafe_allow_html=True,
        )
        any_started = completed >= 1 or in_progress >= 1
        if any_started:
            if st.button("📤 Submit Check-In", use_container_width=True,
                         type="primary", key="sidebar_submit"):
                all_data = _build_all_topic_data()
                if ff_msgs:
                    all_data["freeform_notes"] = [
                        m["content"] for m in st.session_state.freeform_chat
                        if m["role"] == "user"
                    ]
                with st.spinner("Generating report…"):
                    report = generate_report(st.session_state.patient_name, all_data)
                st.session_state.report = report
                with st.spinner("Saving…"):
                    save_to_sheet(st.session_state.patient_name, all_data, report)
                st.session_state.report_saved = True
                st.session_state.app_stage = "report"
                st.rerun()

# ══════════════════════════════════════════════════════════════════
# SCREENS
# ══════════════════════════════════════════════════════════════════

TOPIC_LABELS = {key: label for label, key in TOPICS}
TOPIC_KEYS   = [k for _, k in TOPICS]


def screen_login():
    st.markdown("""
    <div class="welcome-card">
        <div style="font-size:12px;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;color:#6b7b92;margin-bottom:8px;">
            Pre-visit symptom check-in
        </div>
        <h1 style="margin:0 0 10px 0; color:#10233d; font-size:34px; letter-spacing:-0.04em;">🩺 ChatReport</h1>
        <p style="color:#56667d; margin-bottom:14px; font-size:15px; line-height:1.75;">
        A modern clinical check-in for patients receiving head and neck cancer treatment.
        Your answers help your care team review symptoms before the visit.
        </p>
        <div class="subtle-note">
            You can answer by typing or by voice. You may switch topics at any time, and your most recent prior check-in will be used to guide the conversation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        name = st.text_input("Please enter your name:", placeholder="First and last name…")
        if st.button("Begin Check-In →", type="primary", use_container_width=True):
            if name.strip():
                st.session_state.patient_name = name.strip()

                # ── Load previous check-in from Sheets ──────────
                with st.spinner("Loading your previous check-in…"):
                    prev = load_last_checkin(name.strip())

                if prev:
                    st.session_state.last_checkin     = prev
                    st.session_state.has_prev_checkin = True
                else:
                    st.session_state.last_checkin     = {}
                    st.session_state.has_prev_checkin = False

                st.session_state.selected_topic = TOPIC_KEYS[0] if TOPIC_KEYS else None
                st.session_state.app_stage      = "overview"
                st.rerun()
            else:
                st.warning("Please enter your name to continue.")


def screen_overview():
    has_prev = st.session_state.get("has_prev_checkin", False)
    last_ck  = st.session_state.get("last_checkin", {})
    patient  = st.session_state.get("patient_name", "")

    st.markdown(
        '<div class="overview-card">'
        '<div style="font-size:12px;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;color:#6b7b92;margin-bottom:8px;">'
        'Last visit overview'
        '</div>'
        f'<div style="font-size:30px;font-weight:800;letter-spacing:-0.04em;color:#10233d;">'
        f'{_html.escape(patient) if patient else "Your"} previous check-in summary'
        '</div>'
        '<div style="font-size:15px;line-height:1.75;color:#56667d;margin-top:10px;">'
        'Before you start this visit\'s symptom check-in, here is a quick summary of what was recorded last time.'
        '</div>',
        unsafe_allow_html=True,
    )

    if has_prev:
        rows = []
        for label, key in TOPICS:
            prev_data = last_ck.get(key, {})
            if not prev_data:
                continue
            topic_name = label.split(" ", 1)[1] if " " in label else label
            summary = _natural_summary(key, prev_data) or "Information was recorded for this topic."
            detail_html = _checkin_summary_html(key, prev_data) or '<span style="color:#7a8ea4;">No extra details recorded</span>'
            rows.append(
                "<tr>"
                f'<td><div class="overview-topic-name">{_html.escape(topic_name)}</div></td>'
                f'<td><div class="overview-summary-main">{_html.escape(summary)}</div></td>'
                f'<td><div class="overview-summary-details">{detail_html}</div></td>'
                "</tr>"
            )

        if rows:
            st.markdown(
                '<div class="overview-table-wrap">'
                '<table class="overview-table">'
                '<colgroup>'
                '<col class="topic-col">'
                '<col class="summary-col">'
                '<col>'
                '</colgroup>'
                '<thead><tr><th>Topic</th><th>Main Summary</th><th>Details From Last Visit</th></tr></thead>'
                f'<tbody>{"".join(rows)}</tbody>'
                '</table>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="overview-note">'
                'These details are from your last visit. You can change, update, or add anything as you go through today\'s topics.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="overview-note">'
                'A prior visit was found, but there were no summary details available to show here. You can continue to today\'s topics.'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="overview-note">'
            'No previous visit summary was found for you. You can start today\'s symptom check-in now.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Review sample multi-agent scenario passes", expanded=False):
        _render_scenario_library()

    _, col, _ = st.columns([1, 2, 1])
    with col:
        if st.button("Continue to Topics →", type="primary", use_container_width=True):
            st.session_state.app_stage = "main"
            st.rerun()


def screen_main():
    render_sidebar()

    selected = st.session_state.selected_topic

    if not selected:
        st.markdown('<div class="card"><div style="font-size:12px;font-weight:800;color:#6b7b92;text-transform:uppercase;letter-spacing:0.08em;">Getting started</div><div style="font-size:28px;font-weight:800;letter-spacing:-0.03em;margin-top:6px;">Choose a symptom area from the sidebar</div><div style="font-size:14px;color:#5f6f84;line-height:1.7;margin-top:8px;">Move through the check-in in any order you prefer. Your answers are saved into a provider-ready summary for the care team.</div></div>', unsafe_allow_html=True)
        return

    # Route to free-form chat or regular topic
    if selected == "freeform":
        render_freeform_chat()
    else:
        topic_label = TOPIC_LABELS.get(selected, selected)
        render_topic_detail(topic_label, selected)


def screen_report():
    render_sidebar()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📄 Clinical Check-In Report")
    st.markdown(
        f"**Patient:** {st.session_state.patient_name} &nbsp;|&nbsp; "
        f"**Date:** {datetime.now().strftime('%B %d, %Y')}"
    )
    st.markdown(
        '<div style="font-size:13px;color:#627287;line-height:1.7;">'
        'This report is formatted for quick clinical review before the appointment.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    all_data = _build_all_topic_data()
    ff_msgs  = [m for m in st.session_state.freeform_chat if m["role"] == "user"]
    if ff_msgs:
        all_data["freeform_notes"] = [m["content"] for m in ff_msgs]

    if not st.session_state.report:
        with st.spinner("Generating clinical report…"):
            st.session_state.report = generate_report(
                st.session_state.patient_name, all_data
            )

    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown(st.session_state.report)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅️ Back to Check-In"):
            st.session_state.app_stage = "main"
            st.rerun()

    with col2:
        saved     = st.session_state.get("report_saved", False)
        btn_label = "✅ Saved" if saved else "💾 Save to Google Sheets"
        if st.button(btn_label, type="primary", disabled=saved):
            with st.spinner("Saving…"):
                _init_sheets()
                save_to_sheet(
                    st.session_state.patient_name,
                    all_data,
                    st.session_state.report,
                )
            st.session_state.report_saved = True
            st.success("Saved successfully!")
            st.rerun()

    with col3:
        if st.button("📋 Copy to Clipboard (manual)"):
            st.info("Select the report text above and copy (Ctrl+C / Cmd+C).")


# ══════════════════════════════════════════════════════════════════
# MAIN DISPATCH
# ══════════════════════════════════════════════════════════════════

_init_sheets()

stage = st.session_state.get("app_stage", "login")

if stage == "login":
    screen_login()
elif stage == "overview":
    screen_overview()
elif stage == "main":
    screen_main()
elif stage == "report":
    screen_report()
else:
    screen_login()
