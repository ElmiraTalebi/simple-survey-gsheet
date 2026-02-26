#!/usr/bin/env python3
"""
ChatReport - SAFE FAST CLI Prototype (Python 3.8+)

Designed to run quickly and reliably in GitHub repos / IDEs / remote terminals:
- No typing animation (instant output)
- Flushes output aggressively
- Simple, robust input parsing
- Graceful Ctrl+C exit
- Rule-based only (NO APIs)
"""

import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


# -----------------------------
# Helpers
# -----------------------------
YES_WORDS = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "affirmative"}
NO_WORDS = {"no", "n", "nope", "nah", "not really"}
EXIT_WORDS = {"exit", "quit", "q", "stop"}

SEVERITY_WORDS_MAP = {
    "none": 0,
    "mild": 2,
    "moderate": 5,
    "severe": 8,
    "worst": 10,
}

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def normalize(text: str) -> str:
    return (text or "").strip().lower()

def is_exit(text: str) -> bool:
    return normalize(text) in EXIT_WORDS

def safe_wrap(msg: str, width: int = 76) -> str:
    return "\n".join(textwrap.fill(line, width=width) for line in msg.splitlines())

def parse_yes_no(text: str) -> Optional[bool]:
    t = normalize(text)
    if t in YES_WORDS:
        return True
    if t in NO_WORDS:
        return False
    # Handle longer phrases
    for w in YES_WORDS:
        if t.startswith(w + " ") or t.startswith(w + ",") or f" {w} " in f" {t} ":
            return True
    for w in NO_WORDS:
        if t.startswith(w + " ") or t.startswith(w + ",") or f" {w} " in f" {t} ":
            return False
    return None

def extract_int_0_10(text: str) -> Optional[int]:
    t = normalize(text)
    if not t:
        return None

    for k, v in SEVERITY_WORDS_MAP.items():
        if k in t:
            return v

    digits = []
    cur = ""
    for ch in t:
        if ch.isdigit():
            cur += ch
        else:
            if cur:
                digits.append(cur)
                cur = ""
    if cur:
        digits.append(cur)

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


@dataclass
class Exchange:
    timestamp: str
    speaker: str  # "bot" or "user"
    message: str


class ChatReportSafe:
    """Fast, robust CLI prototype"""

    def __init__(self, wrap_width: int = 76):
        self.wrap_width = wrap_width
        self.patient_name: str = ""
        self.appointment_date: str = "Not provided"
        self.treatment_week: Optional[int] = None

        self.conversation_history: List[Exchange] = []
        self.symptoms_not_reported: List[str] = []
        self.additional_notes: str = ""

        self.symptoms: Dict[str, Dict[str, Any]] = {
            "pain": {"present": False, "details": {"location": "", "severity": None, "frequency": "", "management": "", "impact": ""}, "patient_words": ""},
            "mouth": {"present": False, "details": {"dry_mouth": None, "sores": None, "taste_changes": None, "severity": None, "impact_on_eating": ""}, "patient_words": ""},
            "swallowing": {"present": False, "details": {"description": "", "pain_swallowing": None, "food_types": "", "choking_coughing": None, "feeding_tube": None}, "patient_words": ""},
            "nutrition": {"present": False, "details": {"appetite": "", "weight_change": "", "intake_description": "", "supplements": "", "nausea_vomiting": None}, "patient_words": ""},
            "breathing": {"present": False, "details": {"shortness_of_breath": None, "description": "", "triggers_frequency": "", "oxygen": None}, "patient_words": ""},
            "fatigue": {"present": False, "details": {"fatigue_level": None, "impact": "", "work_selfcare": "", "daytime_rest": ""}, "patient_words": ""},
            "mood": {"present": False, "details": {"anxiety": None, "depression": None, "sleep": "", "irritable": None, "support_needed": ""}, "patient_words": ""},
            "other": {"present": False, "details": {"cough": None, "constipation": None, "skin_changes": None, "concentration": None, "sexual_problems": None, "other_notes": ""}, "patient_words": ""},
        }

        self.alerts_high: List[str] = []
        self.alerts_monitor: List[str] = []

    # ---------- I/O ----------
    def bot_say(self, message: str, emoji: str = "🤖") -> None:
        out = safe_wrap(f"{emoji} ChatReport: {message}", width=self.wrap_width)
        print(out, flush=True)
        self._log("bot", message)

    def user_in(self) -> str:
        try:
            msg = input("You: ").strip()
        except KeyboardInterrupt:
            print("\n", flush=True)
            raise
        self._log("user", msg)
        return msg

    def _log(self, speaker: str, message: str) -> None:
        self.conversation_history.append(Exchange(now_str(), speaker, message))

    # ---------- Question helpers ----------
    def ask_free_text(self, question: str, optional: bool = False) -> str:
        self.bot_say(question)
        ans = self.user_in()
        if is_exit(ans):
            raise KeyboardInterrupt
        if not ans and not optional:
            self.bot_say("Even a short answer is helpful. You can also type 'skip'.", emoji="💬")
            ans = self.user_in()
            if is_exit(ans) or normalize(ans) == "skip":
                return ""
        if normalize(ans) == "skip":
            return ""
        return ans

    def ask_yes_no(self, question: str, allow_skip: bool = False) -> Optional[bool]:
        for _ in range(3):
            self.bot_say(question)
            ans = self.user_in()
            if is_exit(ans):
                raise KeyboardInterrupt
            if allow_skip and normalize(ans) in {"skip", "prefer not to say", "n/a"}:
                return None
            yn = parse_yes_no(ans)
            if yn is not None:
                return yn
            self.bot_say("Please answer 'yes' or 'no' (or type 'skip').", emoji="💬")
        return None

    def ask_severity_0_10(self, question: str) -> Optional[int]:
        for _ in range(3):
            self.bot_say(question)
            ans = self.user_in()
            if is_exit(ans):
                raise KeyboardInterrupt
            sev = extract_int_0_10(ans)
            if sev is not None:
                return sev
            self.bot_say("Please enter a number from 0 to 10 (0 = none, 10 = worst).", emoji="💬")
        return None

    # ---------- Conversation ----------
    def greeting(self) -> None:
        self.bot_say("Hello! I’m ChatReport, a symptom check-in tool to help you share how you’ve been feeling before your appointment.")
        self.bot_say("This is not medical advice. If anything feels urgent (especially breathing trouble), contact your care team right away.")
        self.bot_say("For privacy, please avoid highly identifying personal details. Your responses are meant to be shared only with your care team.")

        name = self.ask_free_text("What’s your first name?")
        self.patient_name = (name.split()[0].title() if name else "Patient")
        self.bot_say(f"Thank you, {self.patient_name}.", emoji="❤️")

        appt = self.ask_free_text("Upcoming appointment date? (type 'skip' if you prefer)", optional=True)
        if appt:
            self.appointment_date = appt

        wk = self.ask_free_text("What week of treatment are you in (1–7)? (type 'skip' if not sure)", optional=True)
        try:
            if wk:
                w = int("".join(ch for ch in wk if ch.isdigit()))
                self.treatment_week = w if 1 <= w <= 7 else None
        except Exception:
            self.treatment_week = None

        self.bot_say("Okay — let’s go through a few areas.", emoji="➡️")

    def assess_pain(self) -> None:
        self.bot_say("First: pain.", emoji="➡️")
        yn = self.ask_yes_no("Have you had any pain since your last appointment?")
        if yn is False:
            self.symptoms_not_reported.append("Pain")
            return
        if yn is None:
            self.symptoms_not_reported.append("Pain (unclear)")
            return

        self.symptoms["pain"]["present"] = True
        words = self.ask_free_text("I’m sorry you’re dealing with that. In your own words, what kind of pain is it?")
        loc = self.ask_free_text("Where is the pain located? (throat, mouth, jaw, ear, neck, etc.)", optional=True)
        sev = self.ask_severity_0_10("On a scale of 0–10, how bad is the pain at its worst?")
        freq = self.ask_free_text("How often does it happen? (constant, comes and goes, worse when swallowing, etc.)", optional=True)
        mgmt = self.ask_free_text("What are you using to manage it? (medications, rinses, ice/heat, etc.)", optional=True)
        impact = self.ask_free_text("How is the pain affecting daily life (eating, sleep, talking, etc.)?", optional=True)

        self.symptoms["pain"]["patient_words"] = words
        self.symptoms["pain"]["details"].update({"location": loc, "severity": sev, "frequency": freq, "management": mgmt, "impact": impact})

        if isinstance(sev, int) and sev >= 7:
            self.bot_say("That sounds very difficult. Thank you for sharing — your team will want to review pain control.", emoji="❤️")
        else:
            self.bot_say("Thank you — I’ve noted that.", emoji="❤️")

    def assess_mouth(self) -> None:
        self.bot_say("Next: mouth symptoms and taste.", emoji="➡️")
        open_text = self.ask_free_text("Any mouth-related symptoms like dryness, sores, or taste changes? (you can type 'no')", optional=True)
        self.symptoms["mouth"]["patient_words"] = open_text

        yn = parse_yes_no(open_text)
        likely_present = (yn is True) or contains_any(open_text, ["dry", "sore", "ulcer", "taste", "metal", "burn", "mucositis"])
        if yn is False and not likely_present:
            self.symptoms_not_reported.append("Mouth symptoms (dry mouth/sores/taste)")
            return

        dry = self.ask_yes_no("Have you been bothered by dry mouth?")
        sores = self.ask_yes_no("Have you had mouth sores or ulcers?")
        taste = self.ask_yes_no("Have you noticed taste changes (food tastes different/metallic)?")

        if dry is False and sores is False and taste is False:
            self.symptoms_not_reported.append("Mouth symptoms (dry mouth/sores/taste)")
            return

        self.symptoms["mouth"]["present"] = True
        sev = self.ask_severity_0_10("Overall severity of mouth/taste symptoms (0–10)?")
        impact = self.ask_free_text("How are these affecting eating or drinking?", optional=True)

        self.symptoms["mouth"]["details"].update({"dry_mouth": dry, "sores": sores, "taste_changes": taste, "severity": sev, "impact_on_eating": impact})
        self.bot_say("Thank you — that helps your care team understand what you’re experiencing.", emoji="❤️")

    def assess_swallowing(self) -> None:
        self.bot_say("Next: swallowing.", emoji="➡️")
        yn = self.ask_yes_no("Any trouble swallowing?")
        if yn is False:
            self.symptoms_not_reported.append("Swallowing difficulty")
            return
        if yn is None:
            self.symptoms_not_reported.append("Swallowing difficulty (unclear)")
            return

        self.symptoms["swallowing"]["present"] = True
        desc = self.ask_free_text("Can you describe what swallowing is like?", optional=True)
        pain = self.ask_yes_no("Do you have pain when swallowing?")
        food = self.ask_free_text("What foods can you tolerate? (regular/soft/pureed/liquid only)", optional=True)
        choke = self.ask_yes_no("Any coughing or choking when eating/drinking?")
        tube = self.ask_yes_no("Are you using a feeding tube right now?")

        self.symptoms["swallowing"]["patient_words"] = desc
        self.symptoms["swallowing"]["details"].update({"description": desc, "pain_swallowing": pain, "food_types": food, "choking_coughing": choke, "feeding_tube": tube})
        self.bot_say("Thank you — I’ve captured that.", emoji="❤️")

    def assess_nutrition(self) -> None:
        self.bot_say("Next: nutrition and weight.", emoji="➡️")
        appetite = self.ask_free_text("How has your appetite been lately?", optional=True)
        weight = self.ask_free_text("Any weight changes? (how much and over what time, if known)", optional=True)
        intake = self.ask_free_text("What does a typical day of eating/drinking look like now?", optional=True)
        supp = self.ask_free_text("Any nutrition supplements (Ensure/Boost) or high-calorie drinks?", optional=True)
        nv = self.ask_yes_no("Any nausea or vomiting recently?")

        self.symptoms["nutrition"]["details"].update({"appetite": appetite, "weight_change": weight, "intake_description": intake, "supplements": supp, "nausea_vomiting": nv})
        self.symptoms["nutrition"]["patient_words"] = " | ".join([x for x in [appetite, weight, intake] if x])

        present = False
        if contains_any(appetite, ["low", "poor", "none", "decrease", "down"]) and appetite:
            present = True
        if contains_any(weight, ["lost", "loss", "down"]) and weight:
            present = True
        if contains_any(intake, ["liquid", "shake", "barely", "hard", "can't", "not much"]) and intake:
            present = True
        if nv is True:
            present = True

        self.symptoms["nutrition"]["present"] = present
        if not present:
            self.symptoms_not_reported.append("Nutrition concerns (appetite/weight/intake)")
        self.bot_say("Thank you — this helps your team plan for symptom management.", emoji="❤️")

    def assess_breathing(self) -> None:
        self.bot_say("Next: breathing.", emoji="➡️")
        sob = self.ask_yes_no("Any shortness of breath or difficulty breathing?")
        if sob is False:
            self.symptoms_not_reported.append("Breathing issues")
            return
        if sob is None:
            self.symptoms_not_reported.append("Breathing issues (unclear)")
            return

        self.symptoms["breathing"]["present"] = True
        desc = self.ask_free_text("Can you describe it? (mild vs severe, at rest vs activity)", optional=True)
        trig = self.ask_free_text("How often and any triggers?", optional=True)
        oxy = self.ask_yes_no("Do you need oxygen right now?")

        self.symptoms["breathing"]["details"].update({"shortness_of_breath": sob, "description": desc, "triggers_frequency": trig, "oxygen": oxy})
        self.symptoms["breathing"]["patient_words"] = desc

        if contains_any(desc, ["can't breathe", "struggling", "severe", "at rest"]) or oxy is True:
            self.bot_say("If you are having severe trouble breathing right now, please contact your care team immediately or call emergency services.", emoji="🛑")

        self.bot_say("Thank you — noted.", emoji="❤️")

    def assess_fatigue(self) -> None:
        self.bot_say("Next: energy and fatigue.", emoji="➡️")
        sev = self.ask_severity_0_10("Fatigue level (0–10)?")
        impact = self.ask_free_text("How is fatigue affecting daily activities?", optional=True)
        work = self.ask_free_text("Are you able to work or do self-care as usual?", optional=True)
        rest = self.ask_free_text("Do you need naps or extra rest during the day?", optional=True)

        self.symptoms["fatigue"]["details"].update({"fatigue_level": sev, "impact": impact, "work_selfcare": work, "daytime_rest": rest})
        self.symptoms["fatigue"]["patient_words"] = impact

        present = (isinstance(sev, int) and sev >= 4) or any(x.strip() for x in [impact, work, rest])
        self.symptoms["fatigue"]["present"] = bool(present)
        if not present:
            self.symptoms_not_reported.append("Fatigue (minimal/none reported)")
        self.bot_say("Thank you — I’ve captured that.", emoji="❤️")

    def assess_mood(self) -> None:
        self.bot_say("Next: emotional well-being.", emoji="➡️")
        self.bot_say("Many people feel anxious or down during treatment. Answer only what you’re comfortable sharing.", emoji="💬")

        anx = self.ask_yes_no("Have you been feeling anxious or very worried lately?")
        dep = self.ask_yes_no("Have you been feeling down, sad, or depressed lately?")
        sleep = self.ask_free_text("How has your sleep been?", optional=True)
        irr = self.ask_yes_no("More irritable or on edge than usual?")
        support = self.ask_free_text("Do you feel you need more emotional support right now?", optional=True)

        self.symptoms["mood"]["details"].update({"anxiety": anx, "depression": dep, "sleep": sleep, "irritable": irr, "support_needed": support})
        self.symptoms["mood"]["patient_words"] = " | ".join([x for x in [sleep, support] if x])

        present = (anx is True) or (dep is True) or (irr is True) or contains_any(sleep, ["poor", "bad", "can't", "insomnia"]) or bool(support.strip())
        self.symptoms["mood"]["present"] = bool(present)
        if not present:
            self.symptoms_not_reported.append("Emotional concerns (none reported)")

        # Optional safety check if depressed mood yes
        if dep is True:
            self.bot_say("One quick safety check (you can type 'skip'):", emoji="💬")
            harm = self.ask_yes_no("Have you had thoughts of hurting yourself?", allow_skip=True)
            if harm is True:
                self.bot_say("Please reach out to your care team right away. If you feel in danger, call 988 or emergency services.", emoji="🛑")

        self.bot_say("Thank you for sharing that.", emoji="❤️")

    def assess_other(self) -> None:
        self.bot_say("A few other common symptoms (quick):", emoji="➡️")
        cough = self.ask_yes_no("Bothersome cough?")
        const = self.ask_yes_no("Constipation?")
        skin = self.ask_yes_no("Skin irritation/changes in radiation area?")
        conc = self.ask_yes_no("Difficulty concentrating?")
        sex = self.ask_yes_no("Any sexual problems/concerns to share?", allow_skip=True)
        other_notes = self.ask_free_text("Anything else you want your team to know?", optional=True)

        self.symptoms["other"]["details"].update({"cough": cough, "constipation": const, "skin_changes": skin, "concentration": conc, "sexual_problems": sex, "other_notes": other_notes})
        self.symptoms["other"]["patient_words"] = other_notes

        present = any(v is True for v in [cough, const, skin, conc]) or (sex is True) or bool(other_notes.strip())
        self.symptoms["other"]["present"] = bool(present)
        if not present:
            self.symptoms_not_reported.append("Other symptoms (none reported)")
        self.bot_say("Thanks — almost done.", emoji="❤️")

    def closing(self) -> None:
        extra = self.ask_free_text("Before we finish: anything important we missed? (or type 'no')", optional=True)
        if extra and normalize(extra) not in {"no", "nope", "nah"}:
            self.additional_notes = extra
        self.bot_say("Thank you. I’ll generate a summary for your care team to review before your visit.", emoji="✅")
        self.bot_say("If anything feels urgent (especially breathing trouble), contact your care team right away.", emoji="🛑")
        self.bot_say("Thank you again.", emoji="❤️")

    # ---------- Report ----------
    def _compute_alerts(self) -> None:
        self.alerts_high.clear()
        self.alerts_monitor.clear()

        pain_sev = self.symptoms["pain"]["details"].get("severity")
        if isinstance(pain_sev, int) and pain_sev >= 7:
            self.alerts_high.append(f"Severe pain (worst {pain_sev}/10).")

        mouth_sev = self.symptoms["mouth"]["details"].get("severity")
        if isinstance(mouth_sev, int) and mouth_sev >= 7:
            self.alerts_monitor.append(f"Severe mouth/taste symptoms ({mouth_sev}/10).")

        if self.symptoms["swallowing"]["details"].get("choking_coughing") is True:
            self.alerts_monitor.append("Choking/coughing with eating/drinking reported.")

        food = normalize(self.symptoms["swallowing"]["details"].get("food_types", ""))
        if "liquid" in food:
            self.alerts_monitor.append("Diet limited to liquids (significant dysphagia impact).")

        if self.symptoms["swallowing"]["details"].get("feeding_tube") is True:
            self.alerts_monitor.append("Feeding tube in use.")

        wt = normalize(self.symptoms["nutrition"]["details"].get("weight_change", ""))
        if "lost" in wt or "loss" in wt or "down" in wt:
            self.alerts_monitor.append("Weight loss mentioned.")

        if self.symptoms["breathing"]["present"]:
            desc = normalize(self.symptoms["breathing"]["details"].get("description", ""))
            if contains_any(desc, ["can't breathe", "struggling", "severe", "at rest"]) or self.symptoms["breathing"]["details"].get("oxygen") is True:
                self.alerts_high.append("Concerning breathing difficulty described / oxygen needs.")
            else:
                self.alerts_monitor.append("Shortness of breath/difficulty breathing reported.")

        if self.symptoms["mood"]["details"].get("depression") is True:
            self.alerts_monitor.append("Depressed mood reported.")
        if self.symptoms["mood"]["details"].get("anxiety") is True:
            self.alerts_monitor.append("Anxiety/worry reported.")

        fat = self.symptoms["fatigue"]["details"].get("fatigue_level")
        if isinstance(fat, int) and fat >= 8:
            self.alerts_monitor.append(f"Severe fatigue ({fat}/10).")

        if not self.alerts_high:
            self.alerts_high.append("None flagged by prototype rules.")
        if not self.alerts_monitor:
            self.alerts_monitor.append("None flagged by prototype rules.")

    def _fmt_bool(self, v: Any) -> str:
        if v is True: return "Yes"
        if v is False: return "No"
        return "Not provided"

    def generate_report(self) -> str:
        self._compute_alerts()
        week = str(self.treatment_week) if self.treatment_week else "Not provided"

        lines: List[str] = []
        lines.append("=" * 62)
        lines.append("CHATREPORT SYMPTOM SUMMARY")
        lines.append("=" * 62)
        lines.append(f"Patient Name: {self.patient_name or 'Not provided'}")
        lines.append(f"Report Generated: {now_str()}")
        lines.append(f"Appointment: {self.appointment_date}")
        lines.append(f"Treatment Week: {week}")
        lines.append("=" * 62)
        lines.append("")
        lines.append("🔴 SYMPTOMS PRESENT (requires attention)")
        lines.append("")

        def add_domain(title: str, key: str, bullets: List[str]) -> None:
            lines.append(f"{title} - PRESENT")
            for b in bullets:
                if b:
                    lines.append(f"  {b}")
            pw = (self.symptoms[key].get("patient_words") or "").strip()
            if pw:
                lines.append(f"  Patient words: {pw}")

        if self.symptoms["pain"]["present"]:
            d = self.symptoms["pain"]["details"]
            add_domain("PAIN", "pain", [
                f"Location: {d.get('location') or 'Not provided'}",
                f"Severity: {d.get('severity') if d.get('severity') is not None else 'Not provided'} /10",
                f"Frequency: {d.get('frequency') or 'Not provided'}",
                f"Management: {d.get('management') or 'Not provided'}",
                f"Impact: {d.get('impact') or 'Not provided'}",
            ])

        if self.symptoms["mouth"]["present"]:
            d = self.symptoms["mouth"]["details"]
            add_domain("MOUTH / TASTE SYMPTOMS", "mouth", [
                f"Dry mouth: {self._fmt_bool(d.get('dry_mouth'))}",
                f"Sores/ulcers: {self._fmt_bool(d.get('sores'))}",
                f"Taste changes: {self._fmt_bool(d.get('taste_changes'))}",
                f"Severity: {d.get('severity') if d.get('severity') is not None else 'Not provided'} /10",
                f"Impact on eating: {d.get('impact_on_eating') or 'Not provided'}",
            ])

        if self.symptoms["swallowing"]["present"]:
            d = self.symptoms["swallowing"]["details"]
            add_domain("SWALLOWING DIFFICULTY", "swallowing", [
                f"Description: {d.get('description') or 'Not provided'}",
                f"Pain with swallowing: {self._fmt_bool(d.get('pain_swallowing'))}",
                f"Food types tolerated: {d.get('food_types') or 'Not provided'}",
                f"Choking/coughing: {self._fmt_bool(d.get('choking_coughing'))}",
                f"Feeding tube in use: {self._fmt_bool(d.get('feeding_tube'))}",
            ])

        if self.symptoms["nutrition"]["present"]:
            d = self.symptoms["nutrition"]["details"]
            add_domain("NUTRITION", "nutrition", [
                f"Appetite: {d.get('appetite') or 'Not provided'}",
                f"Weight changes: {d.get('weight_change') or 'Not provided'}",
                f"Food intake: {d.get('intake_description') or 'Not provided'}",
                f"Supplements: {d.get('supplements') or 'Not provided'}",
                f"Nausea/vomiting: {self._fmt_bool(d.get('nausea_vomiting'))}",
            ])

        if self.symptoms["breathing"]["present"]:
            d = self.symptoms["breathing"]["details"]
            add_domain("BREATHING", "breathing", [
                f"Shortness of breath: {self._fmt_bool(d.get('shortness_of_breath'))}",
                f"Description: {d.get('description') or 'Not provided'}",
                f"Triggers/frequency: {d.get('triggers_frequency') or 'Not provided'}",
                f"Needs oxygen: {self._fmt_bool(d.get('oxygen'))}",
            ])

        if self.symptoms["fatigue"]["present"]:
            d = self.symptoms["fatigue"]["details"]
            add_domain("ENERGY / FATIGUE", "fatigue", [
                f"Fatigue level: {d.get('fatigue_level') if d.get('fatigue_level') is not None else 'Not provided'} /10",
                f"Impact: {d.get('impact') or 'Not provided'}",
                f"Work/self-care: {d.get('work_selfcare') or 'Not provided'}",
                f"Daytime rest: {d.get('daytime_rest') or 'Not provided'}",
            ])

        if self.symptoms["mood"]["present"]:
            d = self.symptoms["mood"]["details"]
            add_domain("EMOTIONAL WELL-BEING", "mood", [
                f"Anxiety/worry: {self._fmt_bool(d.get('anxiety'))}",
                f"Depressed mood: {self._fmt_bool(d.get('depression'))}",
                f"Sleep: {d.get('sleep') or 'Not provided'}",
                f"Irritable/on edge: {self._fmt_bool(d.get('irritable'))}",
                f"Support needed: {d.get('support_needed') or 'Not provided'}",
            ])

        if self.symptoms["other"]["present"]:
            d = self.symptoms["other"]["details"]
            add_domain("OTHER SYMPTOMS", "other", [
                f"Cough: {self._fmt_bool(d.get('cough'))}",
                f"Constipation: {self._fmt_bool(d.get('constipation'))}",
                f"Skin changes: {self._fmt_bool(d.get('skin_changes'))}",
                f"Difficulty concentrating: {self._fmt_bool(d.get('concentration'))}",
                f"Sexual problems: {self._fmt_bool(d.get('sexual_problems'))}",
                f"Other notes: {d.get('other_notes') or 'Not provided'}",
            ])

        if not any(self.symptoms[k]["present"] for k in self.symptoms):
            lines.append("No symptoms were clearly reported as present.")

        lines.append("")
        lines.append("✅ SYMPTOMS NOT REPORTED")
        if self.symptoms_not_reported:
            for s in self.symptoms_not_reported:
                lines.append(f"- {s}")
        else:
            lines.append("- Not specified")
        lines.append("")

        lines.append("📝 ADDITIONAL NOTES")
        lines.append(self.additional_notes or "None")
        lines.append("")
        lines.append("=" * 62)
        lines.append("CLINICAL ALERTS")
        lines.append("🔴 High Priority: " + " | ".join(self.alerts_high))
        lines.append("⚠️  Monitor: " + " | ".join(self.alerts_monitor))
        lines.append("=" * 62)

        return "\n".join(lines)

    def run(self) -> None:
        try:
            self.greeting()
            self.assess_pain()
            self.assess_mouth()
            self.assess_swallowing()
            self.assess_nutrition()
            self.assess_breathing()
            self.assess_fatigue()
            self.assess_mood()
            self.assess_other()
            self.closing()

            print("\n" + self.generate_report() + "\n", flush=True)

        except KeyboardInterrupt:
            self.bot_say("Session ended. If anything feels urgent, please contact your care team right away.", emoji="🛑")
            print("Session ended.", flush=True)


def main() -> None:
    # Immediate startup print (helps debug “nothing shows” cases)
    print("Starting ChatReport SAFE mode...", flush=True)

    bot = ChatReportSafe(wrap_width=76)
    bot.run()


if __name__ == "__main__":
    # Extra safety: encourage unbuffered prints in some environments
    # (printing with flush=True already helps, but this line makes it obvious it started)
    main()
