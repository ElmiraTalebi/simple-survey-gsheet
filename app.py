#!/usr/bin/env python3
"""
ChatReport - Phase 1 CLI Prototype (Python 3.8+)

A rule-based conversational symptom reporting chatbot for head and neck cancer
patients undergoing chemoradiation. Produces a structured provider report.

IMPORTANT SAFETY NOTE:
- ChatReport is NOT providing medical advice.
- If emergency symptoms are reported (e.g., severe trouble breathing), it urges
  the patient to contact the care team / emergency services.
"""

import sys
import time
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Utility: ANSI (optional)
# -----------------------------
class ANSI:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"

def supports_color() -> bool:
    return sys.stdout.isatty()

def c(text: str, color: str) -> str:
    if not supports_color():
        return text
    return f"{color}{text}{ANSI.RESET}"


# -----------------------------
# Parsing helpers
# -----------------------------
YES_WORDS = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "affirmative"}
NO_WORDS  = {"no", "n", "nope", "nah", "not really"}

EXIT_WORDS = {"exit", "quit", "q", "stop"}

SEVERITY_WORDS_MAP = {
    "none": 0,
    "mild": 2,
    "moderate": 5,
    "severe": 8,
    "worst": 10,
}

FOOD_LEVELS = [
    "regular",
    "soft",
    "pureed",
    "liquid only",
]

def normalize(text: str) -> str:
    return (text or "").strip().lower()

def is_exit(text: str) -> bool:
    return normalize(text) in EXIT_WORDS

def parse_yes_no(text: str) -> Optional[bool]:
    t = normalize(text)
    if t in YES_WORDS:
        return True
    if t in NO_WORDS:
        return False
    # Handle longer answers like "yes, a little"
    for w in YES_WORDS:
        if f" {w} " in f" {t} " or t.startswith(w + ",") or t.startswith(w + " "):
            return True
    for w in NO_WORDS:
        if f" {w} " in f" {t} " or t.startswith(w + ",") or t.startswith(w + " "):
            return False
    return None

def extract_int_0_10(text: str) -> Optional[int]:
    """
    Try to extract a 0-10 severity score from text.
    Accepts:
      - exact number "7"
      - "7/10"
      - "about an 8"
      - words: mild/moderate/severe/none/worst
    """
    t = normalize(text)
    if not t:
        return None

    # Word mapping
    for k, v in SEVERITY_WORDS_MAP.items():
        if k in t:
            return v

    # Look for digits
    digits = []
    current = ""
    for ch in t:
        if ch.isdigit():
            current += ch
        else:
            if current:
                digits.append(current)
                current = ""
    if current:
        digits.append(current)

    # If "7/10" -> digits could be ["7", "10"]
    # Choose first digit that is 0-10
    for d in digits:
        try:
            num = int(d)
            if 0 <= num <= 10:
                return num
        except ValueError:
            pass

    return None

def contains_any(text: str, keywords: List[str]) -> bool:
    t = normalize(text)
    return any(k in t for k in keywords)

def safe_wrap(msg: str, width: int = 76) -> str:
    return "\n".join(textwrap.fill(line, width=width) for line in msg.splitlines())

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -----------------------------
# Data model helpers
# -----------------------------
@dataclass
class Exchange:
    timestamp: str
    speaker: str  # "bot" or "user"
    message: str


# -----------------------------
# Main ChatReport class
# -----------------------------
class ChatReport:
    """Main chatbot class"""

    def __init__(self, typing_speed: float = 0.015, wrap_width: int = 76):
        self.typing_speed = typing_speed
        self.wrap_width = wrap_width

        self.patient_name: str = ""
        self.appointment_date: str = ""
        self.treatment_week: Optional[int] = None

        self.conversation_history: List[Exchange] = []

        # Symptom data structure (extensible)
        self.symptoms: Dict[str, Dict[str, Any]] = {
            "pain": {
                "present": False,
                "details": {
                    "location": "",
                    "severity": None,     # int 0-10
                    "frequency": "",
                    "management": "",
                    "impact": "",
                },
                "patient_words": "",
            },
            "mouth": {
                "present": False,
                "details": {
                    "dry_mouth": None,     # yes/no
                    "sores": None,         # yes/no
                    "taste_changes": None, # yes/no
                    "severity": None,      # int 0-10 overall mouth discomfort
                    "impact_on_eating": "",
                },
                "patient_words": "",
            },
            "swallowing": {
                "present": False,
                "details": {
                    "difficulty_level": "",       # narrative or categorized
                    "pain_swallowing": None,      # yes/no
                    "food_types": "",             # regular/soft/pureed/liquid only
                    "choking_coughing": None,     # yes/no
                    "feeding_tube": None,         # yes/no
                },
                "patient_words": "",
            },
            "nutrition": {
                "present": False,
                "details": {
                    "appetite": "",
                    "weight_change": "",          # "lost 10 lbs in 2 weeks"
                    "intake_description": "",
                    "supplements": "",            # ensure/boost/etc
                    "nausea_vomiting": None,      # yes/no
                },
                "patient_words": "",
            },
            "breathing": {
                "present": False,
                "details": {
                    "shortness_of_breath": None,  # yes/no
                    "triggers_frequency": "",
                    "oxygen": None,               # yes/no
                    "severity": "",               # narrative
                },
                "patient_words": "",
            },
            "fatigue": {
                "present": False,
                "details": {
                    "fatigue_level": None,        # int 0-10
                    "impact": "",
                    "work_selfcare": "",
                    "daytime_rest": "",
                },
                "patient_words": "",
            },
            "mood": {
                "present": False,
                "details": {
                    "anxiety": None,              # yes/no
                    "depression": None,           # yes/no
                    "sleep": "",
                    "irritable": None,            # yes/no
                    "support_needed": "",
                },
                "patient_words": "",
            },
            "other": {
                "present": False,
                "details": {
                    "cough": None,                # yes/no
                    "constipation": None,         # yes/no
                    "skin_changes": None,         # yes/no
                    "concentration": None,        # yes/no
                    "sexual_problems": None,      # yes/no
                    "other_notes": "",
                },
                "patient_words": "",
            },
        }

        self.symptoms_not_reported: List[str] = []
        self.additional_notes: str = ""

        # Alerts (computed)
        self.alerts_high: List[str] = []
        self.alerts_monitor: List[str] = []

    # -----------------------------
    # Core I/O: typing + logging
    # -----------------------------
    def bot_say(self, message: str, emoji: str = "🤖") -> None:
        wrapped = safe_wrap(f"{emoji} ChatReport: {message}", width=self.wrap_width)
        self._type_out(wrapped + "\n")
        self._log("bot", message)

    def user_ask(self, prompt: str = "") -> str:
        if prompt:
            # Print prompt without typing effect for user readability
            print(prompt, end="", flush=True)
        try:
            msg = input().strip()
        except KeyboardInterrupt:
            print("\n")
            self.bot_say(
                "No problem — we can stop here. If anything feels urgent or unsafe, "
                "please contact your care team right away.",
                emoji="🛑"
            )
            raise
        self._log("user", msg)
        return msg

    def _type_out(self, text: str) -> None:
        # natural typing effect
        for ch in text:
            print(ch, end="", flush=True)
            time.sleep(self.typing_speed)
        # ensure flush
        print("", end="", flush=True)

    def _log(self, speaker: str, message: str) -> None:
        self.conversation_history.append(
            Exchange(timestamp=now_str(), speaker=speaker, message=message)
        )

    # -----------------------------
    # Generic question helpers
    # -----------------------------
    def ask_yes_no(self, question: str, allow_skip: bool = False) -> Optional[bool]:
        """
        Returns True/False or None (if allow_skip and user says 'skip').
        Re-prompts gently if unclear.
        """
        for _ in range(3):
            self.bot_say(question)
            ans = self.user_ask("You: ")
            if is_exit(ans):
                raise KeyboardInterrupt

            if allow_skip and normalize(ans) in {"skip", "prefer not to say", "n/a"}:
                return None

            yn = parse_yes_no(ans)
            if yn is not None:
                return yn

            self.bot_say(
                "Totally okay — just to make sure I understand, would you say yes or no?",
                emoji="💬"
            )

        # If still unclear, return None and move on
        self.bot_say("Thank you — we can move on.", emoji="💬")
        return None

    def ask_free_text(self, question: str, optional: bool = False) -> str:
        self.bot_say(question)
        ans = self.user_ask("You: ")
        if is_exit(ans):
            raise KeyboardInterrupt
        if not ans and optional:
            return ""
        # If empty but not optional, gently prompt once
        if not ans and not optional:
            self.bot_say("Take your time — even a short answer is helpful.", emoji="💬")
            ans = self.user_ask("You: ")
            if is_exit(ans):
                raise KeyboardInterrupt
        return ans.strip()

    def ask_severity_0_10(self, question: str) -> Optional[int]:
        for _ in range(3):
            self.bot_say(question)
            ans = self.user_ask("You: ")
            if is_exit(ans):
                raise KeyboardInterrupt
            sev = extract_int_0_10(ans)
            if sev is not None:
                return sev
            self.bot_say(
                "Thanks — if you can, please give a number from 0 to 10 "
                "(0 = none, 10 = worst).",
                emoji="💬"
            )
        return None

    # -----------------------------
    # Conversation stages
    # -----------------------------
    def greeting(self) -> None:
        self.bot_say(
            "Hello! I’m ChatReport, a symptom check-in tool to help you share how you’ve "
            "been feeling before your upcoming appointment."
        )
        self.bot_say(
            "This usually takes about 10–15 minutes. You can answer briefly or with more detail — "
            "whatever feels comfortable."
        )
        self.bot_say(
            "A quick note: I can’t provide medical advice. If you share something that sounds urgent, "
            "I may suggest contacting your care team right away."
        )
        self.bot_say(
            "For privacy, please avoid sharing highly identifying personal information. "
            "Your responses are meant to be shared only with your care team."
        )

        name = self.ask_free_text("What’s your first name?")
        # Basic cleanup
        name = name.split()[0].strip().title() if name else "there"
        self.patient_name = name

        self.bot_say(
            f"Nice to meet you, {self.patient_name}. Thank you for taking the time to do this."
        )

        # Appointment date (optional but useful for the report header)
        appt = self.ask_free_text(
            "If you’d like, what is your upcoming appointment date? (You can write something like "
            "'March 3' or type 'skip'.)",
            optional=True
        )
        if normalize(appt) in {"skip", ""}:
            self.appointment_date = "Not provided"
        else:
            self.appointment_date = appt

        # Treatment week (optional but useful)
        wk_text = self.ask_free_text(
            "What week of chemoradiation are you in right now? (1–7). If you’re not sure, type 'skip'.",
            optional=True
        )
        wk_norm = normalize(wk_text)
        if wk_norm in {"skip", ""}:
            self.treatment_week = None
        else:
            # try parse int
            try:
                wk = int("".join([ch for ch in wk_norm if ch.isdigit()]) or "0")
                if 1 <= wk <= 7:
                    self.treatment_week = wk
                else:
                    self.treatment_week = None
            except Exception:
                self.treatment_week = None

        self.bot_say(
            "Okay — let’s go step by step through a few common areas. "
            "If anything doesn’t apply, just say so."
        )

    def assess_pain(self) -> None:
        self.bot_say("First, let’s talk about pain.", emoji="➡️")
        yn = self.ask_yes_no("Have you had any pain since your last appointment?")
        if yn is False:
            self.symptoms_not_reported.append("Pain")
            return
        if yn is None:
            # unclear -> treat as not reported
            self.symptoms_not_reported.append("Pain (unclear)")
            return

        self.symptoms["pain"]["present"] = True

        patient_words = self.ask_free_text(
            "I’m sorry you’re dealing with that. In your own words, what kind of pain is it?"
        )
        self.symptoms["pain"]["patient_words"] = patient_words

        loc = self.ask_free_text(
            "Where are you feeling the pain? (For example: throat, mouth, neck, jaw, ear — "
            "or wherever it is.)",
            optional=True
        )
        sev = self.ask_severity_0_10(
            "On a scale of 0 to 10, how bad is the pain at its worst?"
        )
        freq = self.ask_free_text(
            "How often do you feel it? (constant, comes and goes, mostly when swallowing, etc.)",
            optional=True
        )
        mgmt = self.ask_free_text(
            "Are you doing anything to manage the pain right now? (medications, rinses, ice/heat, etc.)",
            optional=True
        )
        impact = self.ask_free_text(
            "How is the pain affecting your daily activities (eating, sleep, talking, moving around)?",
            optional=True
        )

        self.symptoms["pain"]["details"].update({
            "location": loc,
            "severity": sev,
            "frequency": freq,
            "management": mgmt,
            "impact": impact,
        })

        # Empathy + flag
        if sev is not None and sev >= 7:
            self.bot_say(
                "That sounds really difficult. Thank you for telling me — your care team will definitely "
                "want to review your pain control.",
                emoji="❤️"
            )
        else:
            self.bot_say("Thank you — that’s helpful.", emoji="❤️")

    def assess_mouth_symptoms(self) -> None:
        self.bot_say("Now I’d like to ask about your mouth and taste.", emoji="➡️")

        # Open-ended start
        open_text = self.ask_free_text(
            "Have you noticed any mouth-related symptoms — like dryness, sores, or taste changes?",
            optional=True
        )
        self.symptoms["mouth"]["patient_words"] = open_text

        # Determine presence by either explicit yes or content
        yn = parse_yes_no(open_text)
        likely_present = (yn is True) or contains_any(open_text, ["dry", "sores", "ulcer", "mouth", "taste", "metal", "burn", "mucositis"])
        if yn is False and not likely_present:
            self.symptoms_not_reported.append("Mouth symptoms (dry mouth/sores/taste changes)")
            return

        # Ask specifics
        dry = self.ask_yes_no("Have you been bothered by dry mouth (xerostomia)?")
        sores = self.ask_yes_no("Have you had mouth sores or ulcers?")
        taste = self.ask_yes_no("Have you noticed changes in taste (like food tasting different or metallic)?")

        # If all are clearly no -> treat as not reported
        if dry is False and sores is False and taste is False:
            self.symptoms_not_reported.append("Mouth symptoms (dry mouth/sores/taste changes)")
            return

        self.symptoms["mouth"]["present"] = True

        sev = self.ask_severity_0_10(
            "Overall, how severe are your mouth/taste symptoms on a 0–10 scale?"
        )
        impact = self.ask_free_text(
            "How are these mouth or taste issues affecting eating or drinking?",
            optional=True
        )

        self.symptoms["mouth"]["details"].update({
            "dry_mouth": dry,
            "sores": sores,
            "taste_changes": taste,
            "severity": sev,
            "impact_on_eating": impact,
        })

        if sev is not None and sev >= 7:
            self.bot_say(
                "I’m sorry — that level of discomfort can be exhausting. Thank you for sharing it.",
                emoji="❤️"
            )
        else:
            self.bot_say("Thank you — that helps your team understand what you’re dealing with.", emoji="❤️")

    def assess_swallowing(self) -> None:
        self.bot_say("Next, let’s talk about swallowing and eating textures.", emoji="➡️")
        yn = self.ask_yes_no("Have you had any trouble swallowing since your last appointment?")
        if yn is False:
            self.symptoms_not_reported.append("Swallowing difficulty (dysphagia)")
            return
        if yn is None:
            self.symptoms_not_reported.append("Swallowing difficulty (unclear)")
            return

        self.symptoms["swallowing"]["present"] = True

        words = self.ask_free_text("I’m sorry to hear that. Can you describe what swallowing has been like for you?")
        self.symptoms["swallowing"]["patient_words"] = words

        pain_swallow = self.ask_yes_no("Do you have pain when swallowing?")
        # Food types
        food = self.ask_free_text(
            "What kinds of foods can you tolerate right now? (regular / soft / pureed / liquid only — "
            "or describe in your own words.)",
            optional=True
        )
        choking = self.ask_yes_no("Have you had any choking or coughing when eating or drinking?")
        tube = self.ask_yes_no("Are you using a feeding tube right now?")

        self.symptoms["swallowing"]["details"].update({
            "difficulty_level": words,
            "pain_swallowing": pain_swallow,
            "food_types": food,
            "choking_coughing": choking,
            "feeding_tube": tube,
        })

        # Flag moderate concern for choking/coughing
        if choking is True:
            self.bot_say(
                "Thank you for telling me. Your care team will want to review that carefully.",
                emoji="❤️"
            )
        else:
            self.bot_say("Thank you — I’ve got that.", emoji="❤️")

    def assess_nutrition(self) -> None:
        self.bot_say("Now I’d like to ask about nutrition and weight changes.", emoji="➡️")

        appetite = self.ask_free_text(
            "How has your appetite been lately? (about the same, lower, much lower, etc.)",
            optional=True
        )
        weight = self.ask_free_text(
            "Have you noticed any weight change? If yes, about how much and over what timeframe? "
            "(For example: 'lost 6 pounds in 2 weeks')",
            optional=True
        )
        intake = self.ask_free_text(
            "What does a typical day of eating/drinking look like for you right now?",
            optional=True
        )
        supplements = self.ask_free_text(
            "Are you using any nutrition supplements (like Ensure/Boost) or high-calorie drinks?",
            optional=True
        )
        nv = self.ask_yes_no("Have you had nausea or vomiting recently?")

        # Determine if present: appetite low, weight loss, intake reduced, or nausea/vomiting
        present = False
        if contains_any(appetite, ["low", "poor", "bad", "none", "much lower", "decreased"]) and appetite.strip():
            present = True
        if contains_any(weight, ["lost", "loss", "down"]) and weight.strip():
            present = True
        if contains_any(intake, ["liquid", "barely", "hard", "can't", "not much", "small"]) and intake.strip():
            present = True
        if nv is True:
            present = True

        if not present:
            self.symptoms_not_reported.append("Nutrition concerns (appetite/weight/intake)")
        else:
            self.symptoms["nutrition"]["present"] = True

        self.symptoms["nutrition"]["patient_words"] = " | ".join([x for x in [appetite, weight, intake] if x])
        self.symptoms["nutrition"]["details"].update({
            "appetite": appetite,
            "weight_change": weight,
            "intake_description": intake,
            "supplements": supplements,
            "nausea_vomiting": nv,
        })

        self.bot_say("Thank you — nutrition changes are very common during treatment, and this helps your team prepare.", emoji="❤️")

    def assess_breathing(self) -> None:
        self.bot_say("Next, I want to check on breathing.", emoji="➡️")
        sob = self.ask_yes_no("Have you had shortness of breath or any difficulty breathing?")
        if sob is False:
            self.symptoms_not_reported.append("Breathing issues")
            return
        if sob is None:
            self.symptoms_not_reported.append("Breathing issues (unclear)")
            return

        self.symptoms["breathing"]["present"] = True

        severity = self.ask_free_text(
            "Can you describe it a little? (For example: mild vs severe, at rest vs with activity.)",
            optional=True
        )
        trig = self.ask_free_text(
            "How often does it happen, and do you notice any triggers?",
            optional=True
        )
        oxy = self.ask_yes_no("Do you need oxygen or any breathing support right now?")

        self.symptoms["breathing"]["patient_words"] = severity
        self.symptoms["breathing"]["details"].update({
            "shortness_of_breath": sob,
            "triggers_frequency": trig,
            "oxygen": oxy,
            "severity": severity,
        })

        # Emergency-ish string matching
        if contains_any(severity, ["can't breathe", "struggling", "severe", "choking", "blue", "faint"]) or oxy is True:
            self.bot_say(
                "I’m really glad you told me. If you’re having severe trouble breathing right now, "
                "please contact your care team immediately or call emergency services.",
                emoji="🛑"
            )

        self.bot_say("Thank you — I’ve noted that.", emoji="❤️")

    def assess_fatigue(self) -> None:
        self.bot_say("Now let’s talk about energy and fatigue.", emoji="➡️")
        sev = self.ask_severity_0_10(
            "On a 0–10 scale, how would you rate your fatigue lately? (0 = none, 10 = extreme)"
        )

        # If severity missing, still ask a couple of gentle questions
        impact = self.ask_free_text(
            "How is your energy level affecting your daily activities?",
            optional=True
        )
        work = self.ask_free_text(
            "Are you able to work or do self-care the way you usually do? (It’s okay if the answer is no.)",
            optional=True
        )
        rest = self.ask_free_text(
            "Do you find you need naps or extra rest during the day?",
            optional=True
        )

        self.symptoms["fatigue"]["patient_words"] = impact
        self.symptoms["fatigue"]["details"].update({
            "fatigue_level": sev,
            "impact": impact,
            "work_selfcare": work,
            "daytime_rest": rest,
        })

        if (sev is not None and sev >= 4) or any(x.strip() for x in [impact, work, rest]):
            self.symptoms["fatigue"]["present"] = True
        else:
            self.symptoms_not_reported.append("Fatigue/low energy (minimal)")

        if sev is not None and sev >= 7:
            self.bot_say("That’s a lot to carry. Thank you for sharing it.", emoji="❤️")
        else:
            self.bot_say("Thank you — that helps.", emoji="❤️")

    def assess_mood(self) -> None:
        self.bot_say("Now I’d like to check on emotional well-being.", emoji="➡️")
        self.bot_say(
            "Many people in treatment feel anxious, worried, or down at times. There’s no right or wrong answer here.",
            emoji="💬"
        )

        anxiety = self.ask_yes_no("Have you been feeling anxious or very worried lately?")
        depression = self.ask_yes_no("Have you been feeling down, sad, or depressed lately?")
        sleep = self.ask_free_text("How has your sleep been?", optional=True)
        irritable = self.ask_yes_no("Have you been feeling more irritable or on edge than usual?")
        support = self.ask_free_text(
            "Do you feel like you need more emotional support right now? (From family, friends, counselor, support group, etc.)",
            optional=True
        )

        self.symptoms["mood"]["details"].update({
            "anxiety": anxiety,
            "depression": depression,
            "sleep": sleep,
            "irritable": irritable,
            "support_needed": support,
        })
        self.symptoms["mood"]["patient_words"] = " | ".join([x for x in [sleep, support] if x])

        present = (anxiety is True) or (depression is True) or (irritable is True) or contains_any(sleep, ["bad", "poor", "can't", "insomnia"]) or bool(support.strip())
        self.symptoms["mood"]["present"] = bool(present)

        if not present:
            self.symptoms_not_reported.append("Emotional well-being concerns (none reported)")

        # Safety: self-harm ideation screening (gentle + optional)
        # NOTE: You did not explicitly request this, but it’s a common safety consideration.
        # Keep it minimal and non-invasive for Phase 1.
        if depression is True:
            self.bot_say(
                "Thank you for sharing that. If it’s okay, one quick safety check:",
                emoji="💬"
            )
            harm = self.ask_yes_no(
                "Have you had thoughts of hurting yourself?",
                allow_skip=True
            )
            if harm is True:
                self.bot_say(
                    "I’m really sorry you’re feeling this way. Please reach out to your care team right away. "
                    "If you feel in danger, call 988 (Suicide & Crisis Lifeline) or emergency services.",
                    emoji="🛑"
                )

        self.bot_say("Thank you — I appreciate you sharing this.", emoji="❤️")

    def assess_other_symptoms(self) -> None:
        self.bot_say("A few other common symptoms — we can keep this quick.", emoji="➡️")
        cough = self.ask_yes_no("Have you had a bothersome cough?")
        constipation = self.ask_yes_no("Have you had constipation?")
        skin = self.ask_yes_no("Any skin changes or irritation in the radiation area?")
        concentration = self.ask_yes_no("Any trouble concentrating?")
        sexual = self.ask_yes_no("Any sexual problems or concerns you’d like your team to know about?", allow_skip=True)

        other_notes = self.ask_free_text(
            "Anything else you want to mention — even if it doesn’t fit the categories above?",
            optional=True
        )

        self.symptoms["other"]["details"].update({
            "cough": cough,
            "constipation": constipation,
            "skin_changes": skin,
            "concentration": concentration,
            "sexual_problems": sexual,
            "other_notes": other_notes,
        })
        self.symptoms["other"]["patient_words"] = other_notes

        present = any(v is True for v in [cough, constipation, skin, concentration]) or (sexual is True) or bool(other_notes.strip())
        self.symptoms["other"]["present"] = bool(present)
        if not present:
            self.symptoms_not_reported.append("Other common symptoms (none reported)")

        self.bot_say("Thanks — we’re almost done.", emoji="❤️")

    def closing(self) -> None:
        self.bot_say(
            f"Before we wrap up, {self.patient_name}, is there anything important you want your doctor or team "
            "to know that we didn’t cover?",
            emoji="💬"
        )
        extra = self.user_ask("You: ").strip()
        if not is_exit(extra):
            self.additional_notes = extra

        self.bot_say(
            "Thank you. I’ll generate a summary that can be reviewed by your care team before your visit.",
            emoji="✅"
        )
        self.bot_say(
            "If anything feels urgent (especially breathing trouble, severe dehydration, or you feel unsafe), "
            "please contact your care team right away.",
            emoji="🛑"
        )
        self.bot_say("Thank you again for your time.", emoji="❤️")

    # -----------------------------
    # Report generation
    # -----------------------------
    def _compute_alerts(self) -> None:
        self.alerts_high.clear()
        self.alerts_monitor.clear()

        # High priority: pain >= 7
        pain_sev = self.symptoms["pain"]["details"].get("severity")
        if isinstance(pain_sev, int) and pain_sev >= 7:
            self.alerts_high.append(f"Severe pain reported (worst {pain_sev}/10).")

        # Swallowing: choking/coughing yes
        if self.symptoms["swallowing"]["details"].get("choking_coughing") is True:
            self.alerts_monitor.append("Choking/coughing with eating/drinking reported (aspiration risk).")

        # Swallowing: liquid only or feeding tube
        food = normalize(self.symptoms["swallowing"]["details"].get("food_types", ""))
        if "liquid" in food:
            self.alerts_monitor.append("Diet limited to liquids (significant dysphagia impact).")
        if self.symptoms["swallowing"]["details"].get("feeding_tube") is True:
            self.alerts_monitor.append("Feeding tube in use.")

        # Mouth severity >= 7
        mouth_sev = self.symptoms["mouth"]["details"].get("severity")
        if isinstance(mouth_sev, int) and mouth_sev >= 7:
            self.alerts_monitor.append(f"Severe mouth/taste symptoms (reported {mouth_sev}/10).")

        # Nutrition: detect weight loss magnitude keywords
        wt = normalize(self.symptoms["nutrition"]["details"].get("weight_change", ""))
        # Very rough heuristic: if mentions "lost" and a number >= 10 (lbs) => high priority
        if "lost" in wt or "loss" in wt or "down" in wt:
            # Extract any integer and interpret as lbs if not specified
            nums = []
            current = ""
            for ch in wt:
                if ch.isdigit():
                    current += ch
                else:
                    if current:
                        nums.append(int(current))
                        current = ""
            if current:
                nums.append(int(current))
            if any(n >= 10 for n in nums):
                self.alerts_high.append("Significant weight loss mentioned (≥10 units reported).")
            else:
                self.alerts_monitor.append("Weight loss mentioned.")

        # Breathing: any SOB present = monitor; severe descriptors => high
        if self.symptoms["breathing"]["present"]:
            sev_txt = normalize(self.symptoms["breathing"]["details"].get("severity", ""))
            if contains_any(sev_txt, ["can't breathe", "struggling", "severe", "at rest"]):
                self.alerts_high.append("Concerning breathing difficulty described.")
            else:
                self.alerts_monitor.append("Shortness of breath/difficulty breathing reported.")

        # Mood: depression/anxiety true => monitor
        if self.symptoms["mood"]["details"].get("depression") is True:
            self.alerts_monitor.append("Depressed mood reported.")
        if self.symptoms["mood"]["details"].get("anxiety") is True:
            self.alerts_monitor.append("Anxiety/worry reported.")

        # Fatigue high
        fat = self.symptoms["fatigue"]["details"].get("fatigue_level")
        if isinstance(fat, int) and fat >= 8:
            self.alerts_monitor.append(f"Severe fatigue reported ({fat}/10).")

    def _format_domain(self, title: str, key: str, fields: List[Tuple[str, str]]) -> str:
        """
        fields: list of (label, path_key) where path_key is in details dict
        """
        block = []
        block.append(f"{title} - PRESENT")
        details = self.symptoms[key]["details"]
        for label, field_key in fields:
            val = details.get(field_key)
            if val is None or val == "":
                continue
            # Format booleans as Yes/No
            if isinstance(val, bool):
                val = "Yes" if val else "No"
            block.append(f"  {label}: {val}")
        # Include patient words if meaningful
        pw = (self.symptoms[key].get("patient_words") or "").strip()
        if pw:
            block.append(f"  Patient words: {pw}")
        return "\n".join(block)

    def generate_report(self) -> str:
        self._compute_alerts()

        dt = now_str()
        week = str(self.treatment_week) if self.treatment_week else "Not provided"

        lines = []
        lines.append("=" * 62)
        lines.append("CHATREPORT SYMPTOM SUMMARY")
        lines.append("=" * 62)
        lines.append(f"Patient Name: {self.patient_name or 'Not provided'}")
        lines.append(f"Report Generated: {dt}")
        lines.append(f"Appointment: {self.appointment_date or 'Not provided'}")
        lines.append(f"Treatment Week: {week}")
        lines.append("=" * 62)
        lines.append("")
        lines.append("🔴 SYMPTOMS PRESENT (requires attention)")
        lines.append("")

        present_blocks = []

        if self.symptoms["pain"]["present"]:
            present_blocks.append(self._format_domain(
                "PAIN",
                "pain",
                [
                    ("Location", "location"),
                    ("Severity", "severity"),
                    ("Frequency", "frequency"),
                    ("Management", "management"),
                    ("Impact", "impact"),
                ],
            ))

        if self.symptoms["mouth"]["present"]:
            present_blocks.append(self._format_domain(
                "MOUTH / TASTE SYMPTOMS",
                "mouth",
                [
                    ("Dry mouth", "dry_mouth"),
                    ("Sores/ulcers", "sores"),
                    ("Taste changes", "taste_changes"),
                    ("Severity (overall)", "severity"),
                    ("Impact on eating", "impact_on_eating"),
                ],
            ))

        if self.symptoms["swallowing"]["present"]:
            present_blocks.append(self._format_domain(
                "SWALLOWING DIFFICULTY",
                "swallowing",
                [
                    ("Pain with swallowing", "pain_swallowing"),
                    ("Food types tolerated", "food_types"),
                    ("Choking/coughing episodes", "choking_coughing"),
                    ("Feeding tube in use", "feeding_tube"),
                ],
            ))

        if self.symptoms["nutrition"]["present"]:
            present_blocks.append(self._format_domain(
                "NUTRITIONAL CONCERNS",
                "nutrition",
                [
                    ("Appetite", "appetite"),
                    ("Weight changes", "weight_change"),
                    ("Food intake", "intake_description"),
                    ("Supplements", "supplements"),
                    ("Nausea/vomiting", "nausea_vomiting"),
                ],
            ))

        if self.symptoms["breathing"]["present"]:
            present_blocks.append(self._format_domain(
                "BREATHING",
                "breathing",
                [
                    ("Shortness of breath", "shortness_of_breath"),
                    ("Description", "severity"),
                    ("Triggers/frequency", "triggers_frequency"),
                    ("Needs oxygen", "oxygen"),
                ],
            ))

        if self.symptoms["fatigue"]["present"]:
            present_blocks.append(self._format_domain(
                "ENERGY / FATIGUE",
                "fatigue",
                [
                    ("Fatigue level", "fatigue_level"),
                    ("Impact", "impact"),
                    ("Work/self-care", "work_selfcare"),
                    ("Daytime rest", "daytime_rest"),
                ],
            ))

        if self.symptoms["mood"]["present"]:
            present_blocks.append(self._format_domain(
                "EMOTIONAL WELL-BEING",
                "mood",
                [
                    ("Anxiety/worry", "anxiety"),
                    ("Depressed mood", "depression"),
                    ("Sleep", "sleep"),
                    ("Irritable/on edge", "irritable"),
                    ("Support needed", "support_needed"),
                ],
            ))

        if self.symptoms["other"]["present"]:
            present_blocks.append(self._format_domain(
                "OTHER SYMPTOMS",
                "other",
                [
                    ("Cough", "cough"),
                    ("Constipation", "constipation"),
                    ("Skin changes (radiation area)", "skin_changes"),
                    ("Difficulty concentrating", "concentration"),
                    ("Sexual problems", "sexual_problems"),
                    ("Other notes", "other_notes"),
                ],
            ))

        if present_blocks:
            lines.extend(present_blocks)
        else:
            lines.append("No symptoms were clearly reported as present.")

        lines.append("")
        lines.append("✅ SYMPTOMS NOT REPORTED")
        if self.symptoms_not_reported:
            for s in self.symptoms_not_reported:
                lines.append(f"- {s}")
        else:
            lines.append("- Not specified")
        lines.append("")

        # Nutritional status section (even if not present, summarize what was said)
        lines.append("📊 NUTRITIONAL STATUS")
        nut = self.symptoms["nutrition"]["details"]
        lines.append(f"  Appetite: {nut.get('appetite') or 'Not provided'}")
        lines.append(f"  Weight changes: {nut.get('weight_change') or 'Not provided'}")
        lines.append(f"  Food intake: {nut.get('intake_description') or 'Not provided'}")
        lines.append("")

        lines.append("💭 EMOTIONAL WELLBEING")
        mood = self.symptoms["mood"]["details"]
        mood_desc = []
        if mood.get("anxiety") is True:
            mood_desc.append("anxiety/worry reported")
        if mood.get("depression") is True:
            mood_desc.append("depressed mood reported")
        if not mood_desc:
            mood_desc.append("no major concerns reported (or not specified)")
        lines.append(f"  Mood: {', '.join(mood_desc)}")
        lines.append(f"  Sleep: {mood.get('sleep') or 'Not provided'}")
        lines.append("")

        lines.append("📝 ADDITIONAL NOTES")
        lines.append(self.additional_notes or "None")
        lines.append("")
        lines.append("=" * 62)
        lines.append("CLINICAL ALERTS")
        high = self.alerts_high or ["None flagged by prototype rules."]
        mon = self.alerts_monitor or ["None flagged by prototype rules."]
        lines.append("🔴 High Priority: " + (" | ".join(high)))
        lines.append("⚠️  Monitor: " + (" | ".join(mon)))
        lines.append("=" * 62)

        return "\n".join(lines)

    # -----------------------------
    # Orchestration
    # -----------------------------
    def run(self) -> None:
        try:
            self.greeting()
            self.assess_pain()
            self.assess_mouth_symptoms()
            self.assess_swallowing()
            self.assess_nutrition()
            self.assess_breathing()
            self.assess_fatigue()
            self.assess_mood()
            self.assess_other_symptoms()
            self.closing()

            report = self.generate_report()
            print("\n" + c(report, ANSI.CYAN) + "\n")

        except KeyboardInterrupt:
            # already handled with a graceful message in user_ask
            print(c("Session ended.", ANSI.DIM))
            return


def main():
    # You can adjust typing speed here:
    # - smaller is faster, larger is slower
    bot = ChatReport(typing_speed=0, wrap_width=76)
    bot.run()


if __name__ == "__main__":
    main()
