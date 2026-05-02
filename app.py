import hashlib
import html as _html
import io
import importlib.util
import json
import pathlib
import re
import concurrent.futures as _futures
from datetime import datetime
from typing import Any, Optional

import streamlit as st
import streamlit.components.v1 as _stc
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI


def _load_topic_flows():
    try:
        from chatbot_topic_flows import FLOWS, QUESTION_TYPE_BY_ID, STEP_BY_ID, TOPIC_INTROS, TOPICS
        return FLOWS, QUESTION_TYPE_BY_ID, STEP_BY_ID, TOPIC_INTROS, TOPICS
    except ModuleNotFoundError:
        pass

    flow_path = pathlib.Path(__file__).with_name("chatbot_topic_flows.py")
    if flow_path.exists():
        spec = importlib.util.spec_from_file_location("chatbot_topic_flows", flow_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return (
                module.FLOWS,
                module.QUESTION_TYPE_BY_ID,
                module.STEP_BY_ID,
                module.TOPIC_INTROS,
                module.TOPICS,
            )

    fallback_ns: dict[str, Any] = {}
    exec(
        '''
from typing import Optional

TOPICS = [
    ("🍽️  Nutrition & Fluids", "nutrition"),
    ("🩹 Pain & Medications", "pain"),
    ("👄 Oral Symptoms", "oral"),
    ("🤢 GI Symptoms", "gi"),
    ("😴 Fatigue & Sleep", "fatigue"),
    ("🚶 Activity Level", "activity"),
    ("🧠 Mood", "mood"),
    ("💊 Other Symptoms", "other"),
]

TOPIC_INTROS = {
    "pain": "Let's talk about any pain you've been having, what you're taking for it, and whether that regimen is helping.",
    "nutrition": "I'd like to ask about your eating, drinking, and weight.",
    "oral": "Let's go over any mouth and throat symptoms like sticky mucus, thrush, dryness, and what you're using to manage them.",
    "gi": "I'll ask about nausea, vomiting, diarrhea, constipation, and how you're managing those symptoms.",
    "fatigue": "Let's discuss how your energy and sleep have been.",
    "activity": "Tell me about how your daily activities have been going.",
    "mood": "This section covers how you've been feeling emotionally and your support system.",
    "other": "Finally, let's cover any other symptoms — breathing, skin, hearing, and more.",
}

def _q(id, text, type="options", opts=None, when=None,
       placeholder="Please describe...", min_v=0, max_v=10, default_v=0,
       suggestions=None):
    return {
        "id": id, "text": text, "type": type,
        "opts": opts or [], "when": when,
        "placeholder": placeholder,
        "min_v": min_v, "max_v": max_v, "default_v": default_v,
        "suggestions": suggestions or [],
    }

def _safe_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _has_additional_symptom(d, label):
    return label in (d.get("additional_symptoms") or [])

FLOW_PAIN = [
    _q("has_pain", "Do you have any pain today?", opts=["Yes", "No"]),
    _q("pain_location", "Where are you feeling the pain?", opts=["Throat", "Tongue", "Somewhere else"], when=lambda d: d.get("has_pain") == "Yes"),
    _q("pain_start", "When did the pain start?", type="free_text", placeholder="e.g., about a week ago, since I started radiation…", when=lambda d: d.get("has_pain") == "Yes"),
    _q("pain_timing", "Is the pain constant, intermittent, or only happening with swallowing or eating?", opts=["Constant", "Intermittent", "Only when swallowing", "Only when eating", "Both swallowing and eating"], when=lambda d: d.get("has_pain") == "Yes"),
    _q("pain_better_worse", "Is there anything that makes the pain better or worse?", type="free_text", placeholder="e.g., pain medicine helps, swallowing makes it worse…", when=lambda d: d.get("has_pain") == "Yes"),
    _q("pain_severity", "On a scale of 0–10, how bad is the pain at its worst?", type="number", min_v=0, max_v=10, default_v=5, when=lambda d: d.get("has_pain") == "Yes"),
    _q("tongue_type", "Is it a sore or ulcer on the tongue, or a general painful feeling?", opts=["There's a sore/ulcer", "Just pain, no visible sore"], when=lambda d: d.get("pain_location") == "Tongue"),
    _q("tongue_spot", "Is the pain in one specific spot, or does it spread?", opts=["One spot", "Spreads across tongue", "Whole mouth"], when=lambda d: d.get("pain_location") == "Tongue"),
    _q("other_pain_desc", "Which body part is hurting?", type="free_text", placeholder="e.g., near my jaw and ear…", when=lambda d: d.get("pain_location") == "Somewhere else"),
    _q("ear_pain", "Do you have ear pain or hearing changes?", opts=["Yes", "No"], when=lambda d: (d.get("pain_location") == "Somewhere else" and bool(d.get("other_pain_head_neck_focused")))),
    _q("jaw_swelling", "Do you feel any swelling near your jaw?", opts=["Yes", "No"], when=lambda d: (d.get("pain_location") == "Somewhere else" and bool(d.get("other_pain_head_neck_focused")))),
    _q("pain_with_chewing", "Does the pain worsen when chewing or opening your mouth?", opts=["Yes", "No"], when=lambda d: (d.get("pain_location") == "Somewhere else" and bool(d.get("other_pain_head_neck_focused")))),
    _q("pain_medications", "Which medications are you currently taking for pain?", type="multi_select", opts=["Gabapentin", "Oxycodone", "Butrans patch", "Other", "No pain medication"]),
    _q("med_dose_freq", "How often are you taking your pain medication, and at what dose?", type="free_text", placeholder="e.g., Oxycodone 5mg every 6 hours…", when=lambda d: (bool(d.get("pain_medications")) and "No pain medication" not in (d.get("pain_medications") or []))),
    _q("med_side_effects", "Are you experiencing any side effects from your pain medications, such as constipation or anything else?", opts=["Yes", "No"], when=lambda d: (bool(d.get("pain_medications")) and "No pain medication" not in (d.get("pain_medications") or []))),
]

FLOW_NUTRITION = [
    _q("eating_ability", "How has your eating been since your last visit?", opts=["Eating normally — no problems", "Eating less than usual, but managing", "Struggling — only liquids or very little", "Not eating — using a feeding tube only"]),
    _q("fluid_intake_managing", "Are you drinking enough fluids throughout the day — water, shakes, or other drinks?", opts=["Yes, drinking well", "A little less than usual", "Struggling to drink enough"], when=lambda d: d.get("eating_ability") == "Eating less than usual, but managing"),
    _q("food_type", "What are you able to eat right now?", opts=["Mostly normal food", "Soft foods only (yogurt, soup, pudding)", "Mix of soft and liquid", "Mainly liquids"], when=lambda d: d.get("eating_ability") == "Eating less than usual, but managing"),
    _q("nutritional_shakes", "How many nutritional shakes or Boost/Ensure drinks are you having per day?", opts=["None", "1–2", "3–4", "More than 4"], when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),
    _q("eating_barrier", "What is stopping you from eating more?", opts=["Pain when eating/swallowing", "Feel full very quickly", "No appetite", "Nausea", "Too tired to prepare food", "Other"], when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),
    _q("fluid_struggling", "Are you drinking enough fluids — water, juice, or anything?", opts=["Yes, drinking well", "A little", "Very little, hard to drink"], when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),
    _q("fluid_barrier", "What's making it hard to drink?", opts=["Pain when swallowing", "Dry mouth", "Nausea", "Just not thirsty"], when=lambda d: (d.get("eating_ability") == "Struggling — only liquids or very little" and d.get("fluid_struggling") in ["A little", "Very little, hard to drink"])),
    _q("pain_med_timing", "Are you timing your pain medication before meals to make eating easier?", opts=["Yes, it helps", "I try, but it's not enough", "No, I didn't know to do this", "No, I don't take pain medication"], when=lambda d: d.get("eating_ability") == "Struggling — only liquids or very little"),
    _q("tube_issues", "Is the tube feeding going well — no blockages, leaks, or discomfort around the site?", opts=["Working fine", "Some issues — leaking or blockage", "Discomfort/soreness around the tube"], when=lambda d: d.get("eating_ability") == "Not eating — using a feeding tube only"),
    _q("tube_oral_sips", "Are you still able to take any sips of water or liquids by mouth at all?", opts=["Yes, small amounts", "Very occasionally for comfort", "No, nothing by mouth"], when=lambda d: d.get("eating_ability") == "Not eating — using a feeding tube only"),
    _q("swallowing_difficulty", "Are you having any difficulty swallowing — liquids, food, or pills?", opts=["Yes", "No"]),
    _q("swallowing_type", "Is it painful to swallow, or just mechanically difficult?", opts=["Painful to swallow", "Mechanically difficult"], when=lambda d: d.get("swallowing_difficulty") == "Yes"),
    _q("choking_with_eating", "Do you cough or choke when you eat?", opts=["Yes", "No"], when=lambda d: d.get("swallowing_difficulty") == "Yes"),
    _q("swallowing_method", "Are you still able to swallow liquids by mouth, or is everything through a feeding tube?", opts=["I swallow by mouth", "Everything through the feeding tube"], when=lambda d: d.get("swallowing_difficulty") == "Yes"),
    _q("choking_coughing", "Are you having any difficulty with choking or coughing when eating or drinking?", opts=["Yes", "No"]),
    _q("choking_type", "Does it happen with liquids, solids, or both?", opts=["Liquids", "Solids", "Both"], when=lambda d: d.get("choking_coughing") == "Yes"),
    _q("choking_frequency", "Does it happen every time you eat, or only occasionally?", opts=["Every time", "Occasionally"], when=lambda d: d.get("choking_coughing") == "Yes"),
    _q("choking_pills", "Does it also happen when you take pills?", opts=["Yes", "No"], when=lambda d: d.get("choking_coughing") == "Yes"),
    _q("feeding_tube", "Are you currently using a feeding tube?", opts=["Yes", "No"], when=lambda d: d.get("eating_ability") != "Not eating — using a feeding tube only"),
    _q("tube_status", "Is the feeding tube working well or are there issues?", opts=["Working well", "Leakage", "Blockage", "Discomfort"], when=lambda d: (d.get("feeding_tube") == "Yes" and d.get("eating_ability") != "Not eating — using a feeding tube only")),
    _q("tube_oral", "Are you able to take anything by mouth at all?", opts=["Yes, some", "No, nothing by mouth"], when=lambda d: (d.get("feeding_tube") == "Yes" and d.get("eating_ability") != "Not eating — using a feeding tube only")),
    _q("taste_changes", "Have you noticed any changes in your sense of taste?", opts=["Yes", "No"]),
    _q("taste_type", "Does food taste different, bland, or unpleasant?", opts=["Different", "Bland", "Unpleasant"], when=lambda d: d.get("taste_changes") == "Yes"),
    _q("taste_eating_impact", "Is the taste change affecting your ability to eat?", opts=["Yes", "No"], when=lambda d: d.get("taste_changes") == "Yes"),
]

FLOW_ORAL = [
    _q("mouth_sores", "Do you have any mouth sores, ulcers, or white patches/thrush right now?", opts=["Yes", "No"]),
    _q("sore_new_or_old", "Is this new since your last visit, or have you had it for a while?", opts=["New", "Not sure", "Same one as before"], when=lambda d: d.get("mouth_sores") == "Yes"),
    _q("sore_location", "Where exactly is it?", opts=["Inside the mouth/cheek", "On the tongue", "Back of the throat", "Gums/lips", "Multiple spots"], when=lambda d: (d.get("mouth_sores") == "Yes" and d.get("sore_new_or_old") in ["New", "Not sure"])),
    _q("sore_pain_impact", "Is the sore painful? Is it affecting your ability to eat or drink?", opts=["No pain, just noticed it", "A little, but manageable", "Yes, can't eat/drink comfortably"], when=lambda d: (d.get("mouth_sores") == "Yes" and d.get("sore_new_or_old") in ["New", "Not sure"])),
    _q("magic_mouthwash", "Are you using magic mouthwash or any other supportive medications for it? If yes, is it helping?", opts=["Yes, it helps", "Yes, but not enough", "No, I don't have it", "No, I don't use it"], when=lambda d: (d.get("mouth_sores") == "Yes" and d.get("sore_new_or_old") in ["New", "Not sure"])),
    _q("sore_progression", "Is the sore getting better, staying the same, or getting worse?", opts=["Getting better", "About the same", "Getting worse", "Not sure"], when=lambda d: (d.get("mouth_sores") == "Yes" and d.get("sore_new_or_old") == "Same one as before")),
    _q("sore_eating_impact_old", "Is it still preventing you from eating or drinking comfortably?", opts=["Yes", "A little", "No"], when=lambda d: (d.get("mouth_sores") == "Yes" and d.get("sore_new_or_old") == "Same one as before" and d.get("sore_progression") in ["About the same", "Getting worse"])),
    _q("dry_mouth", "Are you experiencing any dryness in your mouth?", opts=["Yes", "No"]),
    _q("dry_mouth_timing", "Is the dryness worse at night or all day?", opts=["Worse at night", "All day"], when=lambda d: d.get("dry_mouth") == "Yes"),
    _q("dry_mouth_med", "Are you using any medication like Biotene or a saliva substitute?", opts=["Yes", "No"], when=lambda d: d.get("dry_mouth") == "Yes"),
    _q("dry_mouth_impact", "Is the dryness making it harder to eat, talk, or sleep?", opts=["Yes", "No"], when=lambda d: d.get("dry_mouth") == "Yes"),
    _q("mucus_issues", "Are you having problems with mucus or thick secretions in your throat?", opts=["Yes", "No"]),
    _q("mucus_type", "Is the mucus thick and hard to clear, or more watery?", opts=["Thick", "More watery"], when=lambda d: d.get("mucus_issues") == "Yes"),
    _q("mucus_impact", "Is the mucus affecting your ability to swallow or sleep?", opts=["Yes", "No"], when=lambda d: d.get("mucus_issues") == "Yes"),
    _q("mucus_management", "Are you using anything to manage it — like Robitussin or saline rinses?", opts=["Yes", "No"], when=lambda d: d.get("mucus_issues") == "Yes"),
    _q("teeth_gum_issues", "Are you having any problems with your teeth or gums?", opts=["Yes", "No"]),
    _q("teeth_issue_type", "Is there pain, bleeding, or sores with your teeth or gums?", opts=["Pain", "Bleeding", "Sores", "Multiple issues"], when=lambda d: d.get("teeth_gum_issues") == "Yes"),
    _q("brushing_difficult", "Is it making brushing difficult?", opts=["Yes", "No"], when=lambda d: d.get("teeth_gum_issues") == "Yes"),
    _q("avoiding_brushing", "Are you avoiding brushing because of the discomfort?", opts=["Yes", "No"], when=lambda d: d.get("teeth_gum_issues") == "Yes"),
    _q("oral_rinse_use", "Are you using mouthwash or oral rinses regularly?", opts=["Yes", "No"]),
    _q("oral_rinse_type", "What type are you using?", type="free_text", placeholder="e.g., magic mouthwash, salt/baking soda rinse…", when=lambda d: d.get("oral_rinse_use") == "Yes"),
    _q("oral_rinse_helping", "Is it helping?", opts=["Yes", "No"], when=lambda d: d.get("oral_rinse_use") == "Yes"),
    _q("oral_rinse_open", "Would you be open to trying an oral rinse to help with symptoms?", opts=["Yes", "No"], when=lambda d: d.get("oral_rinse_use") == "No"),
]

FLOW_GI = [
    _q("nausea_vomiting", "Have you had any nausea, vomiting, or diarrhea since your last visit?", type="multi_select", opts=["Nausea", "Vomiting", "Diarrhea", "None of these"]),
    _q("nausea_frequency", "How often are you feeling nauseated?", type="free_text", placeholder="e.g., a few times a day, mostly in the mornings…", when=lambda d: "Nausea" in (d.get("nausea_vomiting") or [])),
    _q("nausea_management", "What are you using for nausea, and is it helping?", type="free_text", placeholder="e.g., Zofran twice a day and it helps a little…", when=lambda d: "Nausea" in (d.get("nausea_vomiting") or [])),
    _q("vomiting_frequency", "How often are you vomiting and how much?", type="free_text", placeholder="e.g., once or twice a day, small amounts…", when=lambda d: "Vomiting" in (d.get("nausea_vomiting") or [])),
    _q("vomiting_management", "What are you doing to manage the vomiting, and is it helping?", type="free_text", placeholder="e.g., anti-nausea medication, small sips, and it is helping some…", when=lambda d: "Vomiting" in (d.get("nausea_vomiting") or [])),
    _q("diarrhea_frequency", "How often are you having diarrhea?", type="free_text", placeholder="e.g., three loose stools a day…", when=lambda d: "Diarrhea" in (d.get("nausea_vomiting") or [])),
    _q("diarrhea_management", "Are you taking anything for the diarrhea, and is it helping?", type="free_text", placeholder="e.g., Imodium and it helps some…", when=lambda d: "Diarrhea" in (d.get("nausea_vomiting") or [])),
    _q("constipation", "Have you had any constipation or trouble moving your bowels?", opts=["Yes", "No"]),
    _q("bowel_frequency", "How often are you having bowel movements?", type="free_text", placeholder="e.g., once every 3 days…", when=lambda d: d.get("constipation") == "Yes"),
    _q("constipation_meds", "Are you taking anything like Senna, Miralax, or other medications for constipation?", opts=["Yes", "No"], when=lambda d: d.get("constipation") == "Yes"),
    _q("bloating", "Are you feeling bloated or uncomfortable?", opts=["Yes", "No"], when=lambda d: d.get("constipation") == "Yes"),
]

FLOW_FATIGUE = [
    _q("fatigue", "Are you feeling more tired or weak than usual?", opts=["Yes", "No"]),
    _q("fatigue_daily_impact", "Is the fatigue affecting your daily activities — getting dressed, moving around?", opts=["Yes", "No"], when=lambda d: d.get("fatigue") == "Yes"),
    _q("sleep_quality", "Are you having trouble falling asleep or staying asleep?", opts=["Yes", "No"]),
    _q("sleep_wake_reason", "Are you waking up because of pain or other symptoms?", type="free_text", placeholder="e.g., pain wakes me up around 3am…", when=lambda d: d.get("sleep_quality") == "Yes"),
]

FLOW_ACTIVITY = [
    _q("activity_level", "How is your daily life — are you able to do your usual activities?", opts=["Doing everything normally", "Doing less than usual", "Struggling with daily tasks"]),
    _q("difficult_activities", "What activities are most difficult right now?", type="free_text", placeholder="e.g., climbing stairs, cooking, getting dressed…", when=lambda d: d.get("activity_level") in ["Doing less than usual", "Struggling with daily tasks"]),
    _q("activity_limiting_factor", "Is the difficulty mainly due to pain, fatigue, or something else?", opts=["Pain", "Fatigue", "Both", "Something else"], when=lambda d: d.get("activity_level") in ["Doing less than usual", "Struggling with daily tasks"]),
    _q("activity_other_desc", "Can you tell me more about what's limiting your activities?", type="free_text", placeholder="e.g., balance issues, weakness…", when=lambda d: d.get("activity_limiting_factor") == "Something else"),
]

FLOW_MOOD = [
    _q("emotional_state", "How are you feeling emotionally? Are you feeling anxious or worried about anything?", type="free_text", placeholder="Please share how you've been feeling — there are no wrong answers…"),
]

FLOW_OTHER = [
    _q("additional_symptoms", "Are there any additional symptoms you would like to report?", type="multi_select", opts=["Breathing / shortness of breath", "Hearing problems or changes", "Dizziness / lightheadedness", "Numbness or tingling", "Fever or chills", "Skin problems, wounds, or redness", "Voice or speaking changes", "Trouble concentrating or remembering things", "Sexual health concerns", "None of these"]),
    _q("breathing_issues", "Are you having any difficulty breathing or shortness of breath?", opts=["Yes", "No"], when=lambda d: _has_additional_symptom(d, "Breathing / shortness of breath")),
    _q("breathing_timing", "Is the breathing difficulty constant, or does it come on with activity?", opts=["It's constant", "It comes on with activity"], when=lambda d: d.get("breathing_issues") == "Yes"),
    _q("wheezing", "Are you wheezing or feeling like something is blocking your airway?", opts=["Yes", "No"], when=lambda d: d.get("breathing_issues") == "Yes"),
    _q("hearing_changes", "Do you have any hearing problems or changes recently?", opts=["Yes", "No"], when=lambda d: _has_additional_symptom(d, "Hearing problems or changes")),
    _q("hearing_type", "Is it ringing in your ears, hearing loss, or both?", opts=["Ringing in ears", "Hearing loss", "Both"], when=lambda d: d.get("hearing_changes") == "Yes"),
    _q("hearing_constant", "Is it constant or does it come and go?", opts=["Constant", "Comes and goes"], when=lambda d: d.get("hearing_changes") == "Yes"),
    _q("hearing_worsening", "Has it gotten worse compared to your last visit?", opts=["Yes", "No"], when=lambda d: d.get("hearing_changes") == "Yes"),
    _q("dizziness", "Have you been feeling dizzy or lightheaded?", opts=["Yes", "No"], when=lambda d: _has_additional_symptom(d, "Dizziness / lightheadedness")),
    _q("dizziness_timing", "Is it constant or only when you stand up or change position?", opts=["Constant", "Only when standing or changing position"], when=lambda d: d.get("dizziness") == "Yes"),
    _q("dizziness_worsening", "Has the dizziness gotten worse recently?", opts=["Yes", "No"], when=lambda d: d.get("dizziness") == "Yes"),
    _q("falls", "Have you had any falls or felt like you might fall?", opts=["Yes", "No"], when=lambda d: d.get("dizziness") == "Yes"),
    _q("numbness", "Have you noticed any numbness or tingling in your hands or feet?", opts=["Yes", "No"], when=lambda d: _has_additional_symptom(d, "Numbness or tingling")),
    _q("numbness_location", "Is it in your hands, feet, or both?", opts=["Hands", "Feet", "Both"], when=lambda d: d.get("numbness") == "Yes"),
    _q("numbness_new", "Is it new or getting worse?", opts=["New", "Getting worse", "Same as before"], when=lambda d: d.get("numbness") == "Yes"),
    _q("numbness_daily_impact", "Is it affecting your daily activities?", opts=["Yes", "No"], when=lambda d: d.get("numbness") == "Yes"),
    _q("fever_chills", "Have you had any fever or chills recently?", opts=["Yes", "No"], when=lambda d: _has_additional_symptom(d, "Fever or chills")),
    _q("fever_start", "When did the fever or chills start?", type="free_text", placeholder="e.g., two days ago…", when=lambda d: d.get("fever_chills") == "Yes"),
    _q("fever_temp", "How high was the fever?", type="free_text", placeholder="e.g., 101.5°F…", when=lambda d: d.get("fever_chills") == "Yes"),
    _q("fever_other_symptoms", "Do you have any other symptoms like cough or signs of infection?", opts=["Yes", "No"], when=lambda d: d.get("fever_chills") == "Yes"),
    _q("skin_issues", "Have you had any skin problems — like irritation, wounds, or redness?", opts=["Yes", "No"], when=lambda d: _has_additional_symptom(d, "Skin problems, wounds, or redness")),
    _q("skin_location", "Where is the skin issue located?", type="free_text", placeholder="e.g., neck, shoulder, near jaw…", when=lambda d: d.get("skin_issues") == "Yes"),
    _q("skin_start", "When did it start?", type="free_text", placeholder="e.g., about a week ago, at the start of radiation…", when=lambda d: d.get("skin_issues") == "Yes"),
    _q("skin_progression", "Is it getting better, worse, or staying the same?", opts=["Getting better", "About the same", "Getting worse"], when=lambda d: d.get("skin_issues") == "Yes"),
    _q("skin_drainage", "Any drainage, bleeding, or open areas?", opts=["Yes", "No"], when=lambda d: d.get("skin_issues") == "Yes"),
    _q("voice_hoarseness", "How is your voice? Have you noticed any hoarseness or trouble speaking?", opts=["Yes, problems with my voice", "No, voice is fine"], when=lambda d: _has_additional_symptom(d, "Voice or speaking changes")),
    _q("voice_timing", "Is the hoarseness constant or only when you're talking?", opts=["Constant", "Only when talking"], when=lambda d: d.get("voice_hoarseness") == "Yes, problems with my voice"),
    _q("voice_progression", "Has your voice improved or worsened since your last visit?", opts=["Improved", "About the same", "Worse"], when=lambda d: d.get("voice_hoarseness") == "Yes, problems with my voice"),
    _q("voice_communication_impact", "Is it affecting your ability to communicate with others?", opts=["Yes", "No"], when=lambda d: d.get("voice_hoarseness") == "Yes, problems with my voice"),
    _q("concentration", "Have you had trouble concentrating or remembering things?", opts=["Yes", "No"], when=lambda d: _has_additional_symptom(d, "Trouble concentrating or remembering things")),
    _q("concentration_new", "Is it new or ongoing?", opts=["New", "Ongoing"], when=lambda d: d.get("concentration") == "Yes"),
    _q("concentration_daily_impact", "Is it affecting your daily tasks?", opts=["Yes", "No"], when=lambda d: d.get("concentration") == "Yes"),
    _q("sexual_health", "Have you had any sexual health concerns or changes?", opts=["Yes", "Prefer not to say", "No"], when=lambda d: _has_additional_symptom(d, "Sexual health concerns")),
    _q("sexual_discuss", "Would you like to discuss this further with your provider?", opts=["Yes", "No"], when=lambda d: d.get("sexual_health") == "Yes"),
    _q("sexual_cause", "Is it related to treatment, energy levels, or something else?", opts=["Treatment side effects", "Energy levels", "Other"], when=lambda d: d.get("sexual_health") == "Yes"),
]

FLOWS = {
    "pain": FLOW_PAIN,
    "nutrition": FLOW_NUTRITION,
    "oral": FLOW_ORAL,
    "gi": FLOW_GI,
    "fatigue": FLOW_FATIGUE,
    "activity": FLOW_ACTIVITY,
    "mood": FLOW_MOOD,
    "other": FLOW_OTHER,
}

QUESTION_TYPE_BY_ID = {step["id"]: step.get("type", "options") for flow in FLOWS.values() for step in flow}
STEP_BY_ID = {step["id"]: step for flow in FLOWS.values() for step in flow}
''',
        fallback_ns,
    )
    return (
        fallback_ns["FLOWS"],
        fallback_ns["QUESTION_TYPE_BY_ID"],
        fallback_ns["STEP_BY_ID"],
        fallback_ns["TOPIC_INTROS"],
        fallback_ns["TOPICS"],
    )


FLOWS, QUESTION_TYPE_BY_ID, STEP_BY_ID, TOPIC_INTROS, TOPICS = _load_topic_flows()


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


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _safe_int_value(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


_BODY_LOCATION_PATTERN = re.compile(
    r"\b("
    r"head|face|nose|ear|ears|jaw|chin|mouth|tongue|throat|neck|shoulder|arm|elbow|wrist|hand|hands|finger|fingers|"
    r"chest|back|side|stomach|belly|abdomen|hip|leg|legs|knee|knees|ankle|ankles|foot|feet|toe|toes|rib|ribs|"
    r"cheek|lip|lips|gum|gums|tooth|teeth|palate|scalp"
    r")\b"
)


def _looks_like_body_location_phrase(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    return bool(_BODY_LOCATION_PATTERN.search(normalized))


def _extract_numeric_value(text: str) -> Optional[int]:
    match = re.search(r"-?\d+(?:\.\d+)?", str(text or ""))
    if not match:
        return None
    try:
        return int(float(match.group(0)))
    except ValueError:
        return None


def _local_option_match(step: dict, user_input: str) -> Optional[str]:
    """Deterministic fallback for common patient wording when no LLM is available."""
    opts = step.get("opts", [])
    normalized = _norm_text(user_input)
    if not normalized:
        return None

    if "pain" in normalized and "fatigue" in normalized and "Both" in opts:
        return "Both"
    if "ringing" in normalized and "hearing loss" in normalized and "Both" in opts:
        return "Both"
    if "liquid" in normalized and "solid" in normalized and "Both" in opts:
        return "Both"

    for opt in opts:
        opt_norm = _norm_text(opt)
        if opt_norm == normalized:
            return opt
        if opt_norm and (opt_norm in normalized or normalized in opt_norm):
            return opt

    yes_words = {"yes", "yeah", "yep", "yup", "sure", "correct", "right"}
    no_words = {"no", "nope", "nah", "none", "not really", "no issues", "fine", "doing okay"}
    if any(word in normalized.split() for word in yes_words) or normalized.startswith("yes "):
        for opt in opts:
            if _norm_text(opt).startswith("yes"):
                return opt
    if normalized in no_words or normalized.startswith("no ") or " no " in f" {normalized} ":
        for opt in opts:
            if _norm_text(opt).startswith("no") or _norm_text(opt) == "none of these":
                return opt

    synonym_groups = [
        (("throat", "swallow"), ("Throat", "Painful to swallow", "Only when swallowing")),
        (("tongue",), ("Tongue", "On the tongue")),
        (("cheek", "inside mouth", "inside my mouth"), ("Inside the mouth/cheek",)),
        (("gum", "gums", "lip", "lips"), ("Gums/lips",)),
        (("multiple", "both"), ("Both", "Multiple spots", "Multiple issues")),
        (("nauseous", "nausea", "sick to my stomach"), ("Nausea",)),
        (("throwing up", "vomit", "vomiting"), ("Vomiting",)),
        (("loose stool", "diarrhea"), ("Diarrhea",)),
        (("soft", "yogurt", "soup", "pudding"), ("Soft foods only (yogurt, soup, pudding)",)),
        (("liquid", "shake", "ensure", "boost"), ("Mainly liquids", "Mix of soft and liquid")),
        (("less than usual", "less than normal"), ("Eating less than usual, but managing", "Doing less than usual")),
        (("struggling", "very little"), ("Struggling — only liquids or very little", "Struggling with daily tasks")),
        (("pain and fatigue", "fatigue and pain"), ("Both",)),
        (("general tired", "general tiredness"), ("General tiredness",)),
        (("specific", "legs", "arms"), ("Weakness in specific parts",)),
        (("comes and goes", "come and go"), ("Comes and goes",)),
        (("ringing", "hearing loss"), ("Both", "Ringing in ears", "Hearing loss")),
        (("voice is fine", "voice fine"), ("No, voice is fine",)),
        (("prefer not", "private"), ("Prefer not to say",)),
        (("bland",), ("Bland",)),
        (("different",), ("Different",)),
        (("unpleasant", "bad taste"), ("Unpleasant",)),
        (("as prescribed",), ("Yes", "Yes, it helps")),
        (("not enough",), ("Yes, but it's not enough", "I try, but it's not enough")),
        (("working well", "working fine"), ("Working well", "Working fine")),
    ]
    for triggers, candidates in synonym_groups:
        if not any(trigger in normalized for trigger in triggers):
            continue
        for candidate in candidates:
            if candidate in opts:
                return candidate

    best_opt = None
    best_score = 0
    input_words = set(normalized.split())
    for opt in opts:
        opt_words = set(_norm_text(opt).split())
        if not opt_words:
            continue
        score = len(input_words & opt_words)
        if score > best_score:
            best_score = score
            best_opt = opt
    if best_opt and best_score >= 2:
        return best_opt
    return None


def _is_redundant_followup(original_question: str, answer: str, followup_question: str) -> bool:
    oq = _norm_text(original_question)
    fq = _norm_text(followup_question)
    if not fq:
        return True
    if fq == oq or fq in oq or oq in fq:
        return True
    return _is_semantically_redundant_question(original_question, followup_question)


def _is_semantically_redundant_question(text_a: str, text_b: str) -> bool:
    a = _norm_text(text_a)
    b = _norm_text(text_b)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    if not openai_client or not ENABLE_LLM_SEMANTIC_REDUNDANCY:
        return False
    relation = run_question_relation_agent(text_a, text_b)
    return bool(relation.get("same_intent"))


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

    if topic_key == "pain" and step["id"] == "pain_location" and answer == "Somewhere else":
        if raw not in ("Somewhere else", "somewhere else"):
            current_data["pain_location_raw"] = raw
            current_data["other_pain_desc"] = raw
        return answer

    return answer

def _fallback_clarifying_question(step: dict) -> str:
    text = step.get("text", "").strip()
    if text:
        return f"I didn't quite catch that. Could you answer this part again: {text}"
    return "I didn't quite catch that. Could you tell me a little more about that?"


def _build_retry_prompt(
    step: dict,
    user_input: str,
    topic_history: Optional[list[dict[str, str]]] = None,
    recent_questions: Optional[list[str]] = None,
) -> str:
    return _fallback_clarifying_question(step)


def _auto_capture_following_answers(topic_key: str, state: dict, seed_text: str):
    # Disabled in normal operation: silent auto-filling made the conversation feel
    # presumptive and could create unrelated or repeated questions.
    return







def parse_multi_select_typed_input(step: dict, user_input: str):
    if not user_input.strip():
        return []

    normalized_full = _norm_text(user_input)
    lowered_map = {opt.lower(): opt for opt in step.get("opts", [])}
    parts = [p.strip() for p in re.split(r",|/|;|\n|\band\b", user_input, flags=re.IGNORECASE) if p.strip()]
    resolved = []
    has_other = "Other" in step.get("opts", [])
    has_none = "None of these" in step.get("opts", [])

    for opt in step.get("opts", []):
        if opt in {"Other", "None of these"}:
            continue
        opt_norm = _norm_text(opt)
        if opt_norm and opt_norm in normalized_full:
            resolved.append(opt)

    if has_none and any(phrase in normalized_full for phrase in ("none", "no nausea", "no vomiting", "no diarrhea", "no issues")):
        resolved.append("None of these")

    for part in parts:
        match = lowered_map.get(part.lower())
        if match:
            resolved.append(match)
        else:
            interpreted = interpret_user_input_with_options(step, part)
            if interpreted in step.get("opts", []):
                resolved.append(interpreted)
            elif has_other and part.strip():
                resolved.append("Other")

    if has_other and resolved and any(
        other_word in normalized_full
        for other_word in ("tylenol", "advil", "ibuprofen", "aleve", "morphine", "hydromorphone")
    ):
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
    gap: 12px;
    padding: 10px 14px;
    border-bottom: 1px solid #e2ebf2;
    background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(245,249,252,0.9) 100%);
}

.chat-shell-title {
    display: block;
    width: 100%;
}

.chat-shell-name {
    font-family: 'Manrope', sans-serif;
    font-size: 15px;
    font-weight: 800;
    color: #143551;
    letter-spacing: -0.03em;
}

.chat-shell-inline-summary {
    flex: 0 0 auto;
}

.chat-shell-inline-summary details {
    display: inline-block;
}

.chat-shell-inline-summary summary {
    list-style: none;
    cursor: pointer;
    font-size: 12px;
    font-weight: 700;
    color: #607589;
    white-space: nowrap;
}

.chat-shell-inline-summary summary::-webkit-details-marker {
    display: none;
}

.chat-shell-inline-summary summary::before {
    content: "▸";
    display: inline-block;
    margin-right: 6px;
    color: #607589;
}

.chat-shell-inline-summary details[open] summary::before {
    content: "▾";
}

.chat-shell-inline-summary-body {
    padding: 10px 14px 8px 14px;
    border-bottom: 1px solid #e8eef4;
    background: rgba(247, 251, 254, 0.78);
}

.chat-history {
    padding: 14px 14px 2px 14px;
    min-height: 0;
    background:
        linear-gradient(180deg, rgba(250,252,254,0.88) 0%, rgba(244,248,252,0.92) 100%);
}

.composer-wrap {
    padding: 4px 12px 12px 12px;
    background: transparent;
}

.topic-response-region {
    margin-top: -8px;
}

.topic-toolbar + div[data-testid="stButton"] {
    position: sticky;
    top: 10px;
    z-index: 25;
    display: flex;
    justify-content: flex-end;
    margin: 0 0 8px 0;
}

.topic-toolbar + div[data-testid="stButton"] > button {
    width: auto !important;
    min-width: 230px !important;
    border-radius: 999px !important;
    padding: 0.55rem 1rem !important;
    border: 1px solid #f2c4c4 !important;
    background: #fff6f6 !important;
    color: #a33b3b !important;
    box-shadow: 0 8px 16px rgba(163, 59, 59, 0.08) !important;
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

.chat-row.assistant.current-question .chat-bubble {
    background: linear-gradient(180deg, #fffdf7 0%, #fff7de 100%);
    border: 1px solid #f2d98a;
    box-shadow: 0 10px 22px rgba(191, 149, 0, 0.10);
}

.chat-row.assistant.current-question .chat-role::after {
    content: "  CURRENT QUESTION";
    color: #9b7a0a;
    font-weight: 800;
    letter-spacing: 0.08em;
    margin-left: 4px;
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

.report-dashboard {
    background:
        linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(247,250,253,0.98) 100%);
    border: 1px solid #d8e4ee;
    border-radius: 28px;
    padding: 24px 24px 20px 24px;
    box-shadow: var(--shadow);
}

.report-summary-banner {
    background: linear-gradient(180deg, #fff4cf 0%, #fff8e7 100%);
    border: 1px solid #ebd28a;
    border-radius: 22px;
    padding: 18px 20px;
    margin: 14px 0 18px 0;
    display: grid;
    grid-template-columns: 88px 1fr;
    gap: 16px;
    align-items: center;
}

.report-summary-avatar {
    width: 76px;
    height: 76px;
    border-radius: 18px;
    border: 1px solid #dfc26b;
    background: linear-gradient(180deg, #fffdf7 0%, #fff4d2 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 34px;
}

.report-summary-copy {
    color: #5e4a12;
    font-size: 13px;
    line-height: 1.65;
}

.report-summary-title {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b6c1b;
    margin-bottom: 4px;
}

.report-topic-card {
    border-radius: 18px;
    padding: 0;
    overflow: hidden;
    border: 1px solid #d7e1eb;
    background: #ffffff;
    min-height: 148px;
    box-shadow: 0 8px 18px rgba(23, 50, 74, 0.05);
    margin-bottom: 10px;
}

.report-topic-card.red {
    border-color: #ef9c9c;
    background: linear-gradient(180deg, #fff7f7 0%, #fff1f1 100%);
}

.report-topic-card.green {
    border-color: #9fd1a8;
    background: linear-gradient(180deg, #f6fff7 0%, #effbf1 100%);
}

.report-topic-strip {
    padding: 8px 12px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: white;
    background: #9aa8b7;
}

.report-topic-card.red .report-topic-strip {
    background: #d84c43;
}

.report-topic-card.green .report-topic-strip {
    background: #3f8f49;
}

.report-topic-body {
    padding: 10px 12px;
}

.report-topic-name {
    font-size: 13px;
    font-weight: 800;
    color: #16324b;
    margin-bottom: 8px;
}

.report-topic-compare {
    display: grid;
    grid-template-columns: 1fr;
    gap: 8px;
}

.report-topic-compare-row {
    border: 1px solid #e4ebf2;
    border-radius: 14px;
    padding: 8px 9px;
    background: rgba(255,255,255,0.86);
}

.report-topic-compare-label {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8295a9;
    margin-bottom: 2px;
}

.report-topic-compare-value {
    font-size: 12px;
    line-height: 1.45;
    color: #18344d;
}

.report-topic-open {
    margin-top: -4px;
    margin-bottom: 12px;
}

.report-detail-shell {
    margin-top: 10px;
    border: 1px solid #d7e2eb;
    border-radius: 24px;
    background: linear-gradient(180deg, #ffffff 0%, #f9fbfd 100%);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
}

.report-detail-shell.inline {
    margin-top: 10px;
    margin-bottom: 18px;
    animation: reportSlideDown 180ms ease-out;
}

.report-detail-header {
    padding: 16px 18px;
    border-bottom: 1px solid #e5edf4;
    background: rgba(255,255,255,0.88);
}

.report-detail-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    padding: 16px 18px 6px 18px;
}

.report-detail-panel {
    border: 1px solid #e2eaf1;
    border-radius: 18px;
    padding: 14px;
    background: #ffffff;
}

.report-detail-label {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7d90a3;
    margin-bottom: 6px;
}

.report-detail-text {
    font-size: 14px;
    line-height: 1.6;
    color: #17324a;
}

@media (max-width: 900px) {
    .report-summary-banner,
    .report-detail-grid {
        grid-template-columns: 1fr;
    }
}

@keyframes reportSlideDown {
    from {
        opacity: 0;
        transform: translateY(-6px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
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
    background: rgba(247, 251, 254, 0.92);
    border: 1px solid #d9e4ed;
    border-radius: 18px;
    padding: 10px 12px;
    margin: 0 0 10px 0;
    box-shadow: 0 8px 18px rgba(23, 50, 74, 0.04);
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
    border-radius: 22px;
    padding: 6px 12px 10px 12px;
    box-shadow: 0 18px 36px rgba(23, 50, 74, 0.08);
    backdrop-filter: blur(10px);
    position: relative;
}

.composer-shell.compact {
    padding: 6px 12px 10px 12px;
}

.composer-inline-voice {
    display: flex;
    align-items: stretch;
}

.composer-shell [data-testid="stHorizontalBlock"] {
    gap: 0.55rem !important;
    align-items: stretch !important;
}

.composer-shell [data-testid="column"] {
    display: flex;
    flex-direction: column;
    justify-content: stretch;
}

.composer-shell [data-testid="column"]:last-child {
    max-width: 100%;
    min-width: 0;
    flex: 1 1 auto;
}

.composer-shell [data-testid="column"]:last-child > div {
    height: 100%;
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

.common-answer-buttons {
    margin: 0 0 10px 0;
}

.common-answer-buttons [data-testid="stRadio"] label {
    font-size: 12px !important;
    font-weight: 700 !important;
    color: #607589 !important;
}

.suggested-replies-note {
    margin: 8px 0 10px 2px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #6b7d92;
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

.composer-shell [data-testid="stTextInput"] > label,
.composer-shell [data-testid="stTextInput"] label {
    margin-bottom: 2px !important;
    padding-bottom: 0 !important;
}

.composer-shell [data-testid="stTextInput"] label p {
    margin: 0 !important;
    line-height: 1.2 !important;
}

.composer-shell [data-testid="stTextInput"] input {
    min-height: 52px !important;
    height: 52px !important;
    background: #f9fcff !important;
    border: 1px solid #d6e4ef !important;
    padding-left: 16px !important;
}

.composer-shell form > div:first-child,
.composer-shell [data-testid="stForm"] > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

.composer-shell [data-testid="stSelectbox"] {
    margin-bottom: 0 !important;
}

.composer-shell [data-testid="stAudioInput"] {
    background: linear-gradient(180deg, #fdfefe 0%, #f5f9fd 100%);
    border: 1px solid #d7e4ee;
    border-radius: 999px;
    min-height: 38px;
    max-width: 38px;
    width: 38px;
    min-width: 38px;
    margin-left: auto;
    margin-right: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    margin-top: 0;
    margin-bottom: 0;
    box-shadow: 0 6px 12px rgba(23, 50, 74, 0.06) !important;
    overflow: hidden;
}

.composer-shell [data-testid="stAudioInput"] > div {
    width: 38px;
    min-width: 38px;
    height: 38px;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden;
}

.composer-shell [data-testid="stAudioInput"] button {
    border-radius: 999px !important;
    width: 38px !important;
    height: 38px !important;
    min-height: 38px !important;
    min-width: 38px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
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
    background: rgba(255,255,255,0.65) !important;
    box-shadow: none !important;
}

.composer-shell [data-testid="stAudioInput"] button svg {
    width: 14px !important;
    height: 14px !important;
    color: #0f6cbd !important;
}

.composer-shell [data-testid="stAudioInput"] audio {
    display: none !important;
}

.inline-voice-row {
    display: flex;
    justify-content: flex-end;
    align-items: center;
}

.inline-voice-row [data-testid="stAudioInput"] {
    width: 38px !important;
    min-width: 38px !important;
    max-width: 38px !important;
    height: 38px !important;
    min-height: 38px !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: hidden !important;
    font-size: 0 !important;
    line-height: 0 !important;
    color: transparent !important;
}

.inline-voice-row [data-testid="stAudioInput"] > div {
    width: 38px !important;
    min-width: 38px !important;
    max-width: 38px !important;
    height: 38px !important;
    min-height: 38px !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}

.inline-voice-row [data-testid="stAudioInput"] button {
    width: 38px !important;
    min-width: 38px !important;
    max-width: 38px !important;
    height: 38px !important;
    min-height: 38px !important;
    border-radius: 999px !important;
    border: 1px solid #d7e4ee !important;
    background: linear-gradient(180deg, #ffffff 0%, #f5f9fd 100%) !important;
    box-shadow: 0 6px 12px rgba(23, 50, 74, 0.06) !important;
    padding: 0 !important;
    margin: 0 !important;
    position: relative !important;
}

.inline-voice-row [data-testid="stAudioInput"] button svg {
    display: none !important;
}

.inline-voice-row [data-testid="stAudioInput"] button::after {
    content: "🎙️";
    font-size: 14px;
    line-height: 38px;
    position: absolute;
    inset: 0;
    text-align: center;
}

.inline-voice-row [data-testid="stAudioInput"] audio,
.demo-reasoning-card {
    background: rgba(255,255,255,0.92);
    border: 1px solid #d9e4ed;
    border-radius: 22px;
    padding: 14px 14px 10px 14px;
    box-shadow: 0 18px 36px rgba(23, 50, 74, 0.08);
    backdrop-filter: blur(10px);
    position: sticky;
    top: 72px;
}

.demo-reasoning-card .demo-title {
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7d92;
    margin-bottom: 6px;
}

.demo-reasoning-card .demo-subtitle {
    font-family: 'Manrope', sans-serif;
    font-size: 18px;
    line-height: 1.25;
    font-weight: 800;
    color: #153652;
    letter-spacing: -0.03em;
    margin-bottom: 10px;
}

.demo-flow {
    margin-top: 10px;
    border: 1px solid #d9e4ed;
    border-radius: 22px;
    background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(247,251,254,0.96) 100%);
    padding: 12px;
    box-shadow: 0 18px 36px rgba(23, 50, 74, 0.08);
}

.demo-flow-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 10px;
    padding: 4px 2px 10px 2px;
}

.demo-flow-kicker {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7b90a3;
    margin-bottom: 4px;
}

.demo-flow-title {
    font-family: 'Manrope', sans-serif;
    font-size: 13px;
    line-height: 1.45;
    font-weight: 800;
    color: #17324a;
}

.demo-status {
    flex: 0 0 auto;
    border-radius: 999px;
    padding: 5px 9px;
    font-size: 10px;
    font-weight: 800;
    white-space: nowrap;
    border: 1px solid transparent;
}

.demo-status.flow {
    color: #0f6b3d;
    background: #ecfdf5;
    border-color: #bbf7d0;
}

.demo-status.ai {
    color: #0f5d93;
    background: #edf7ff;
    border-color: #bfdbfe;
}

.demo-node {
    display: grid;
    grid-template-columns: 28px 1fr;
    gap: 10px;
    align-items: start;
    border: 1px solid #e1eaf2;
    border-radius: 18px;
    background: #ffffff;
    padding: 10px;
}

.demo-node.current {
    border-color: #b7d5eb;
    background: linear-gradient(180deg, #f7fbff 0%, #eef7ff 100%);
}

.demo-node-index {
    width: 26px;
    height: 26px;
    border-radius: 999px;
    background: #17324a;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 800;
}

.demo-node-label {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7d92a7;
    margin-bottom: 4px;
}

.demo-node-text {
    font-size: 13px;
    line-height: 1.5;
    color: #17324a;
    font-weight: 750;
    word-break: break-word;
}

.demo-node-subtext {
    margin-top: 4px;
    font-size: 12px;
    line-height: 1.45;
    color: #64788c;
}

.demo-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 7px;
}

.demo-chip {
    border-radius: 999px;
    padding: 3px 8px;
    background: #f1f6fb;
    border: 1px solid #dce8f2;
    color: #39566f;
    font-size: 10px;
    font-weight: 800;
}

.demo-connector {
    width: 2px;
    height: 16px;
    margin: 2px 0 2px 23px;
    background: linear-gradient(180deg, #bfd3e4 0%, #d9e7f2 100%);
}

.demo-mini-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 6px;
    margin-top: 9px;
}

.demo-mini {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    border-radius: 12px;
    background: #f7fbfe;
    border: 1px solid #e1ebf3;
    padding: 6px 8px;
}

.demo-mini span {
    font-size: 10px;
    color: #70869a;
    font-weight: 800;
}

.demo-mini strong {
    font-size: 10px;
    color: #17324a;
    text-align: right;
}

@media (max-width: 1100px) {
    .demo-reasoning-card {
        position: static;
        top: auto;
        margin-top: 10px;
    }
}
.inline-voice-row [data-testid="stAudioInput"] small,
.inline-voice-row [data-testid="stAudioInput"] span,
.inline-voice-row [data-testid="stAudioInput"] p {
    display: none !important;
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


def _dynamic_step_text(topic_key: Optional[str], step: dict, state: Optional[dict] = None) -> str:
    question_text = step["text"]
    if not ENABLE_DYNAMIC_PROMPT_REWRITE:
        return question_text
    if not state or not topic_key or not openai_client:
        return question_text

    prompt_cache = state.setdefault("generated_prompts", {})
    cached = prompt_cache.get(step["id"])
    if cached:
        return cached

    last_visit_same_question_answer = st.session_state.get("last_checkin", {}).get(topic_key, {}).get(step.get("id"))
    result = run_question_writer_agent(
        step,
        topic_history=_recent_topic_history(state),
        recent_questions=_recent_topic_questions(state),
        last_visit_same_question_answer=last_visit_same_question_answer,
    )
    rewritten = str(result.get("question_text") or question_text).strip() or question_text
    prompt_cache[step["id"]] = rewritten
    return rewritten


def _step_prompt_text(step: dict, topic_key: Optional[str] = None, state: Optional[dict] = None) -> str:
    question_text = _dynamic_step_text(topic_key, step, state)
    if step.get("type") == "options" and step.get("id") != "med_adherence_issue":
        question_text += " (Choose an option below, or answer in your own words if that fits better.)"
    return question_text


def _clear_current_prompt_flags(state: dict):
    for msg in state.get("chat", []):
        if msg.get("role") == "assistant":
            msg["is_current_prompt"] = False


def _append_assistant_message(
    state: dict,
    text: str,
    *,
    prompt_step: Optional[dict] = None,
    prompt_text: str = "",
):
    text = (text or "").strip()
    if not text:
        return
    if state["chat"] and state["chat"][-1]["role"] == "assistant" and state["chat"][-1]["content"].strip() == text:
        if prompt_step or prompt_text:
            _clear_current_prompt_flags(state)
            state["chat"][-1]["is_current_prompt"] = True
            if prompt_step:
                state["chat"][-1]["prompt_step_id"] = prompt_step.get("id")
            if prompt_text:
                state["chat"][-1]["prompt_text"] = prompt_text.strip()
        return
    message = {"role": "assistant", "content": text}
    if prompt_step or prompt_text:
        _clear_current_prompt_flags(state)
        message["is_current_prompt"] = True
        if prompt_step:
            message["prompt_step_id"] = prompt_step.get("id")
        if prompt_text:
            message["prompt_text"] = prompt_text.strip()
    state["chat"].append(message)


def _remember_prompted_step(state: dict, step: Optional[dict], prompt_text: str = ""):
    state["last_prompted_step_id"] = step.get("id") if step else None
    state["last_prompted_text"] = (prompt_text or "").strip()


def _ensure_step_prompted(topic_key: str, state: dict, step: Optional[dict]):
    if not step:
        return
    prompt_text = _step_prompt_text(step, topic_key=topic_key, state=state)
    last_id = state.get("last_prompted_step_id")
    last_text = state.get("last_prompted_text", "")
    if last_id == step.get("id") and (
        last_text == prompt_text or _is_semantically_redundant_question(last_text, prompt_text)
    ):
        _append_assistant_message(state, last_text or prompt_text, prompt_step=step, prompt_text=prompt_text)
        return
    _append_assistant_message(state, prompt_text, prompt_step=step, prompt_text=prompt_text)
    _remember_prompted_step(state, step, prompt_text)


def _recent_topic_history(state: dict, limit: int = 10) -> list[dict[str, str]]:
    history = []
    for msg in state.get("chat", [])[-limit:]:
        role = msg.get("role", "")
        content = str(msg.get("content", "")).strip()
        if role and content:
            history.append({"role": role, "content": content})
    return history


def _recent_topic_questions(state: dict, limit: int = 8) -> list[str]:
    questions = []
    for msg in state.get("chat", [])[-limit * 2:]:
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        parts = [part.strip() for part in content.split("\n\n") if part.strip()]
        for part in parts:
            if "?" in part:
                questions.append(part)
    return questions[-limit:]


def _question_already_asked(state: dict, question_text: str) -> bool:
    candidate = (question_text or "").strip()
    if not candidate:
        return False
    for asked in _recent_topic_questions(state, limit=10):
        if _is_semantically_redundant_question(asked, candidate):
            return True
    return False


def render_chat_bubble(role: str, content: str, highlight: bool = False):
    safe = _html.escape(content or "").replace("\n", "<br>")
    role_cls = "user" if role == "user" else "assistant"
    if highlight and role != "user":
        role_cls = f"{role_cls} current-question"
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

# Performance defaults: keep the common path fast.
ENABLE_DYNAMIC_PROMPT_REWRITE = False
ENABLE_LLM_SEMANTIC_REDUNDANCY = False
ENABLE_FULL_PIPELINE_FOR_EXACT_STRUCTURED_OPTIONS = False

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

# ══════════════════════════════════════════════════════════════════
# FLOW ENGINE
# ══════════════════════════════════════════════════════════════════

def _step_is_relevant(topic_key: str, step: dict, data: dict, raw_answers: Optional[dict] = None) -> bool:
    raw_answers = raw_answers or {}
    if topic_key == "oral" and step.get("id") == "oral_rinse_open":
        denied_oral_symptoms = (
            data.get("mouth_sores") == "No"
            and data.get("dry_mouth") == "No"
            and data.get("mucus_issues") == "No"
            and data.get("teeth_gum_issues") == "No"
        )
        if denied_oral_symptoms:
            return False

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


def get_upcoming_steps(topic_key: str, data: dict, raw_answers: Optional[dict] = None, limit: int = 5) -> list[dict]:
    upcoming = []
    for step in FLOWS.get(topic_key, []):
        when = step.get("when")
        if when and not when(data):
            continue
        if not _step_is_relevant(topic_key, step, data, raw_answers):
            continue
        if step["id"] in data:
            continue
        upcoming.append(step)
        if len(upcoming) >= limit:
            break
    return upcoming


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


def _topic_status_label(topic_key: str) -> str:
    state = st.session_state.topic_states.get(topic_key, {})
    status = state.get("status", "not_started")
    if status == "completed":
        return "Completed"
    if status == "in_progress":
        return "In progress"
    if state.get("data"):
        return "Incomplete"
    return "Not started"


def _guided_current_topic_key() -> Optional[str]:
    """Return the next clinician-ordered topic the patient should answer."""
    selected = st.session_state.get("selected_topic")
    if selected and selected != "freeform":
        selected_state = st.session_state.topic_states.get(selected, {})
        if selected_state.get("status") == "in_progress" and not topic_is_complete(
            selected,
            selected_state.get("data", {}),
            selected_state.get("raw_answers"),
        ):
            return selected

    for _, key in TOPICS:
        state = st.session_state.topic_states[key]
        if state.get("status") == "in_progress" and not topic_is_complete(
            key, state.get("data", {}), state.get("raw_answers")
        ):
            return key

    for _, key in TOPICS:
        if st.session_state.topic_states[key].get("status") != "completed":
            return key
    return None


def _sync_guided_topic_selection() -> Optional[str]:
    current = _guided_current_topic_key()
    st.session_state.selected_topic = current
    return current


def _overall_progress() -> tuple[int, int, int]:
    answered_total = 0
    applicable_total = 0
    completed_topics = 0
    for _, key in TOPICS:
        state = st.session_state.topic_states[key]
        answered, applicable = get_topic_progress(key, state.get("data", {}), state.get("raw_answers"))
        answered_total += answered
        applicable_total += applicable
        if state.get("status") == "completed":
            completed_topics += 1
    return answered_total, applicable_total, completed_topics


def _record_response_metadata(
    topic_key: str,
    step: dict,
    answer: Any,
    source: str,
    raw_answer: Any,
    display: str,
):
    """Store structured response metadata for clinician summary and demos."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "topic": topic_key,
        "question_id": step.get("id"),
        "question_text": step.get("text"),
        "answer_type": step.get("type", "options"),
        "answer": answer,
        "raw_answer": raw_answer,
        "display_answer": display,
        "source": source,
        "is_followup_answer": source == "followup" or str(step.get("id", "")).endswith("_llm_followup"),
    }
    st.session_state.setdefault("structured_responses", []).append(entry)


def _record_agent_trace(topic_key: str, step: dict, pipeline: dict):
    """Keep a compact demo trace of the latest agent/orchestrator decisions."""
    trace = {
        "topic": topic_key,
        "question_id": step.get("id"),
        "question": step.get("text"),
        "mode": "single_pass",
        "ai_used": True,
        "source": pipeline.get("source", "typed/free-text/voice"),
        "patient_answer": pipeline.get("patient_answer"),
        "agent_1_answer_interpreter": {
            "matched_option": pipeline.get("matched_option"),
        },
        "agent_2_urgency": {
            "urgency_tier": pipeline.get("urgency_tier", 0),
            "urgency_message_shown": bool(pipeline.get("urgency_message")),
        },
        "agent_3_engagement": {
            "reduce_follow_up": pipeline.get("reduce_follow_up", False),
            "wants_to_stop": pipeline.get("wants_to_stop", False),
        },
        "agent_4_doctor_relevance": {
            "doctor_note": pipeline.get("doctor_note"),
            "clinical_priority": pipeline.get("clinical_priority"),
            "follow_up_goal": pipeline.get("follow_up_goal"),
            "next_step_action": pipeline.get("next_step_action"),
        },
        "agent_5_next_move": {
            "follow_up_triggered": pipeline.get("follow_up", False),
            "follow_up_question": pipeline.get("follow_up_question"),
        },
        "orchestrator": {
            "assistant_message": pipeline.get("assistant_message"),
            "special_signals": pipeline.get("special_signals", {}),
            "final_decision": pipeline.get("demo_decision"),
            "next_question": None,
        },
    }
    st.session_state.setdefault("agent_traces", []).append(trace)
    st.session_state["last_agent_trace"] = trace


def _record_fastpath_trace(
    topic_key: str,
    step: dict,
    answer: Any,
    source: str,
    note: str,
):
    trace = {
        "topic": topic_key,
        "question_id": step.get("id"),
        "question": step.get("text"),
        "mode": "fast_path",
        "ai_used": False,
        "source": source,
        "patient_answer": answer,
        "agent_1_answer_interpreter": {
            "matched_option": answer,
        },
        "agent_2_urgency": {
            "urgency_tier": 0,
        },
        "agent_3_engagement": {
            "wants_to_stop": False,
            "reduce_follow_up": False,
        },
        "agent_4_doctor_relevance": {
            "clinical_priority": "low",
            "follow_up_goal": None,
        },
        "agent_5_next_move": {
            "follow_up_triggered": False,
            "follow_up_question": None,
        },
        "orchestrator": {
            "assistant_message": None,
            "final_decision": note,
            "next_question": None,
        },
    }
    st.session_state.setdefault("agent_traces", []).append(trace)
    st.session_state["last_agent_trace"] = trace


def _finalize_demo_trace(decision: str, next_question: Optional[str] = None):
    trace = st.session_state.get("last_agent_trace")
    if not trace:
        return
    orch = trace.setdefault("orchestrator", {})
    orch["final_decision"] = decision
    orch["next_question"] = next_question
    st.session_state["last_agent_trace"] = trace
    traces = st.session_state.get("agent_traces", [])
    if traces:
        traces[-1] = trace
        st.session_state["agent_traces"] = traces


def _current_prompt_text_for_topic(topic_key: Optional[str]) -> str:
    if not topic_key:
        return ""
    state = st.session_state.get("topic_states", {}).get(topic_key, {})
    for msg in reversed(state.get("chat", [])):
        if msg.get("role") == "assistant" and msg.get("is_current_prompt"):
            content = str(msg.get("prompt_text") or msg.get("content") or "").strip()
            if content:
                parts = [part.strip() for part in content.split("\n\n") if part.strip()]
                return parts[-1] if parts else content
    return str(state.get("last_prompted_text") or "").strip()


def _mark_patient_fatigue(topic_key: Optional[str] = None):
    st.session_state["patient_fatigue"] = True
    st.session_state["fatigue_requested_at"] = datetime.now().isoformat(timespec="seconds")
    if topic_key and topic_key in st.session_state.topic_states:
        state = st.session_state.topic_states[topic_key]
        state["_patient_fatigue"] = True
        _append_assistant_message(
            state,
            "I understand. I’ll keep this as brief as I can and focus on the most important questions.",
        )


def _render_demo_agent_panel(topic_key: Optional[str] = None):
    if not st.session_state.get("demo_mode"):
        return
    trace = st.session_state.get("last_agent_trace")
    current_prompt = _current_prompt_text_for_topic(
        topic_key or st.session_state.get("selected_topic")
    )
    st.markdown(
        '<div class="demo-reasoning-card">'
        '<div class="demo-title">Demo Mode</div>'
        '<div class="demo-subtitle">Decision trace</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if not trace:
        st.caption("Demo mode shows what happened after the latest answer.")
        if current_prompt:
            st.markdown(f"**Current question now:** {current_prompt}")
        st.info("No answer has been processed yet.")
        return

    mode = trace.get("mode", "fast_path")
    source = trace.get("source", "")
    ai_used = bool(trace.get("ai_used", mode != "fast_path"))
    if mode == "fast_path" or source == "structured":
        ai_used = False

    interp = trace.get("agent_1_answer_interpreter") or {}
    urgency = trace.get("agent_2_urgency") or {}
    engagement = trace.get("agent_3_engagement") or {}
    doctor = trace.get("agent_4_doctor_relevance") or {}
    next_move = trace.get("agent_5_next_move") or {}
    orchestrator = trace.get("orchestrator") or {}

    if not ai_used:
        path_text = "Flowchart only. No GPT call was made."
    else:
        path_text = "Single-pass AI orchestrator. One GPT call handled interpretation, urgency, and next-step logic."

    decision = orchestrator.get("final_decision") or "Move to the next applicable flowchart question."
    next_question = orchestrator.get("next_question")
    follow_up_question = next_move.get("follow_up_question")
    if next_move.get("follow_up_triggered") and follow_up_question:
        decision = f"Ask one follow-up: {follow_up_question}"
    elif next_question:
        decision = f"Move to next question: {next_question}"

    status_class = "ai" if ai_used else "flow"
    status_label = "AI used" if ai_used else "No AI"
    path_label = "Single-pass AI" if ai_used else "Flowchart only"
    answer_text = str(trace.get("patient_answer") or "")
    question_text = str(trace.get("question") or "Not available")
    current_text = current_prompt or "Topic complete or waiting for the next action."
    matched = interp.get("matched_option")
    urgency_tier = urgency.get("urgency_tier", 0)
    reduce_follow_up = engagement.get("reduce_follow_up", False)
    wants_to_stop = engagement.get("wants_to_stop", False)
    priority = doctor.get("clinical_priority") or "not specified"

    details_html = ""
    if ai_used:
        detail_items = []
        if matched:
            detail_items.append(("Interpreted as", str(matched)))
        detail_items.extend([
            ("Urgency tier", str(urgency_tier)),
            ("Follow-up depth", "reduced" if reduce_follow_up else "standard"),
            ("Patient stop signal", "yes" if wants_to_stop else "no"),
            ("Clinical priority", str(priority)),
        ])
        details_html = "".join(
            f'<div class="demo-mini"><span>{_html.escape(label)}</span><strong>{_html.escape(value)}</strong></div>'
            for label, value in detail_items
        )
    else:
        details_html = (
            '<div class="demo-mini"><span>Interpretation</span><strong>Pre-defined option accepted</strong></div>'
            '<div class="demo-mini"><span>GPT calls</span><strong>0</strong></div>'
        )

    st.caption("Demo mode shows the latest processed answer and the next system action.")
    st.markdown(
        f'''
        <div class="demo-flow">
            <div class="demo-flow-top">
                <div>
                    <div class="demo-flow-kicker">Latest answer</div>
                    <div class="demo-flow-title">{_html.escape(question_text)}</div>
                </div>
                <div class="demo-status {status_class}">{status_label}</div>
            </div>
            <div class="demo-node">
                <div class="demo-node-index">1</div>
                <div class="demo-node-body">
                    <div class="demo-node-label">Patient input</div>
                    <div class="demo-node-text">{_html.escape(answer_text)}</div>
                    <div class="demo-chip-row">
                        <span class="demo-chip">{_html.escape(source or "unknown")}</span>
                    </div>
                </div>
            </div>
            <div class="demo-connector"></div>
            <div class="demo-node">
                <div class="demo-node-index">2</div>
                <div class="demo-node-body">
                    <div class="demo-node-label">Processing path</div>
                    <div class="demo-node-text">{_html.escape(path_label)}</div>
                    <div class="demo-node-subtext">{_html.escape(path_text)}</div>
                    <div class="demo-mini-grid">{details_html}</div>
                </div>
            </div>
            <div class="demo-connector"></div>
            <div class="demo-node">
                <div class="demo-node-index">3</div>
                <div class="demo-node-body">
                    <div class="demo-node-label">Decision</div>
                    <div class="demo-node-text">{_html.escape(decision)}</div>
                </div>
            </div>
            <div class="demo-connector"></div>
            <div class="demo-node current">
                <div class="demo-node-index">4</div>
                <div class="demo-node-body">
                    <div class="demo-node-label">Current question now</div>
                    <div class="demo-node-text">{_html.escape(current_text)}</div>
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )



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


_ROUTING_SUPPORT_SYS = """
You are a compact routing support agent for a clinical chatbot.

You will receive a mode plus the relevant inputs.

Modes:
  - question_relation
  - pain_location_focus

For mode=question_relation:
  Decide whether two patient-facing questions are asking essentially the same thing.
  Rules:
    - Treat paraphrases as the same intent.
    - If one question is only a softer or more natural wording of the other, mark true.
    - If one asks for a different detail, mark false.
    - Be especially sensitive to duplicate symptom questions in clinical chat.

For mode=pain_location_focus:
  Decide whether a patient-described pain location should trigger the focused
  head-and-neck follow-up branch.
  Rules:
    - Mark true only when the described location is specifically in or very near the
      ear, jaw, mouth, lips, gums, teeth, cheek, palate, tongue, or throat.
    - Mark false for broad or nonspecific areas like head, face, neck, arm, back, chest,
      or anything outside that focused region.

Return ONLY valid JSON with the fields needed for the mode:
{
  "same_intent": true/false,
  "head_neck_focused": true/false
}
"""


def _run_routing_support_agent(mode: str, payload: dict, max_tokens: int = 80) -> dict:
    if not openai_client:
        return {}
    result = _call_agent(_ROUTING_SUPPORT_SYS, {"mode": mode, **payload}, max_tokens=max_tokens)
    return result or {}


def run_question_relation_agent(question_a: str, question_b: str) -> dict:
    default = {"same_intent": False}
    result = _run_routing_support_agent("question_relation", {
        "question_a": question_a,
        "question_b": question_b,
    })
    return {**default, **result}


def run_pain_location_focus_agent(location_text: str) -> dict:
    default = {"head_neck_focused": False}
    if not str(location_text or "").strip():
        return default
    result = _run_routing_support_agent("pain_location_focus", {
        "location_text": location_text,
    })
    return {**default, **result}


_CONVERSATION_COPY_SYS = """
You are a compact conversation copy agent for a clinical chatbot.

You will receive a mode plus the relevant inputs.

Modes:
  - question_rewrite
  - clarification
  - quick_replies

For mode=question_rewrite:
  Rewrite the next patient-facing question naturally.
  Rules:
    - The form question is a question bank hint, not fixed wording.
    - Write one natural nurse-like question that asks for the same clinical detail.
    - Use topic history so the question feels like a direct continuation of the conversation.
    - Avoid repeating a recent question from this topic.
    - Do not rewrite a narrower follow-up detail question back into the broader parent question.
    - If the patient already named a specific location, medication, symptom, or person, do not turn the next question back into a generic chooser they already answered.
    - Mention last-visit history only when it genuinely helps orient the patient.
    - Never prepend awkward phrases like "Last visit you reported yes."
    - Do not imply symptoms the patient has not endorsed.
    - Do not turn one patient answer into two questions. Ask one clear question only.
    - If the next formal step is already specific and usable as written, stay close to it instead of inventing a bridge question.
    - This rule applies across all topics: do not narrow the same concept through a chain of micro-questions unless the form step itself clearly requires that narrower detail.
    - If recent topic history shows the assistant is circling around the same concept, rewrite the next question in the most direct single-step form possible.
    - If the patient has effectively said the symptom/problem is absent, okay, manageable, or not concerning, prefer concise wrap-up wording rather than probing for more details.
    - If last_visit_same_question_answer is provided and is meaningfully comparable, use it to orient the wording naturally when helpful.
    - Good uses of history: ask whether something is the same, better, worse, lower, higher, more frequent, or less frequent than last time.
    - Do not force history into the wording when it adds no value or would sound awkward.
    - Keep the question concise and conversational.

For mode=clarification:
  Write a short clarification question when the patient's reply did not clearly answer the current question.
  Rules:
    - Ask for the same missing information more clearly, not a different detail.
    - Be warm and brief.
    - If options exist, you may gently restate the kind of answer needed.
    - Use the recent topic history to see whether you are clarifying the same detail repeatedly.
    - Ask for only one missing detail.
    - Never ask the patient to repeat details they already gave.
    - If the reply is partially usable, ask only for the missing part rather than restarting the whole question.
    - If the patient likely gave a typo, synonym, or close real-world term for one option, ask a single confirmation question rather than restarting the whole option list.
    - This must generalize to every topic: location, timing, severity, medications, support, bowel habits, hydration, sleep, mood, skin, breathing, hearing, and any other domain.
    - Prefer confirming the closest likely interpretation over re-listing all options when the patient appears to be answering in plain language.
    - Do not create serial narrowing questions about the same detail. One clarification is enough unless the missing detail is high priority.
    - If the patient's meaning is essentially "it's okay", "no problem", "manageable", or another reassuring negative screen, avoid clarification unless the original question truly cannot be completed safely without it.
    - If the patient sounds frustrated, terse, or resistant, keep the clarification extremely short and avoid drilling deeper.
    - Do not sound robotic or blame the patient.
    - Do not repeat the original question verbatim.

For mode=quick_replies:
  Suggest clickable quick-reply buttons.
  Rules:
    - Return 2 to 4 short patient-style reply options when helpful.
    - These are suggestion buttons, not the only valid answers. Free text will still be available.
    - Suggestions must fit the current question naturally and must not contradict the recent topic history.
    - Do not repeat a detail the patient already gave as if it still needs to be answered.
    - If predefined_options already exist, return an empty list.
    - For number questions, suggestions must be numeric strings within range.
    - For free-text questions, keep suggestions short and natural, like something a patient would actually tap.
    - Prefer suggestions that help the patient answer quickly, not a full exhaustive list.
    - If no helpful suggestions are obvious, return an empty list.

Return ONLY valid JSON with the fields needed for the mode:
{
  "question_text": "...",
  "clarification_question": "...",
  "suggestions": ["...", "..."]
}
"""


def _run_conversation_copy_agent(mode: str, payload: dict, max_tokens: int) -> dict:
    if not openai_client:
        return {}
    result = _call_agent(_CONVERSATION_COPY_SYS, {"mode": mode, **payload}, max_tokens=max_tokens)
    return result or {}


def run_question_writer_agent(
    step: dict,
    topic_history: list[dict[str, str]],
    recent_questions: list[str],
    last_visit_same_question_answer: Any = None,
) -> dict:
    default = {"question_text": step.get("text", "")}
    result = _run_conversation_copy_agent("question_rewrite", {
        "base_question_text": step.get("text", ""),
        "question_type": step.get("type", "options"),
        "options": step.get("opts", []),
        "recent_topic_history": topic_history,
        "recent_question_texts": recent_questions,
        "last_visit_same_question_answer": last_visit_same_question_answer,
    }, max_tokens=140)
    return {**default, **result}


def run_clarification_writer_agent(
    step: dict,
    patient_reply: str,
    topic_history: list[dict[str, str]],
    recent_questions: list[str],
) -> dict:
    default = {"clarification_question": "Could you tell me a little more about that?"}
    result = _run_conversation_copy_agent("clarification", {
        "original_question": step.get("text", ""),
        "question_type": step.get("type", "options"),
        "options": step.get("opts", []),
        "patient_reply": patient_reply,
        "recent_topic_history": topic_history,
        "recent_question_texts": recent_questions,
    }, max_tokens=120)
    return {**default, **result}


def run_quick_reply_suggester_agent(step: dict, topic_history: list[dict[str, str]], recent_questions: list[str]) -> dict:
    default = {"suggestions": []}
    result = _run_conversation_copy_agent("quick_replies", {
        "question_text": step.get("text", ""),
        "question_type": step.get("type", "free_text"),
        "predefined_options": step.get("opts", []),
        "placeholder": step.get("placeholder"),
        "min_value": step.get("min_v"),
        "max_value": step.get("max_v"),
        "recent_topic_history": topic_history,
        "recent_question_texts": recent_questions,
    }, max_tokens=140)
    if not result:
        return default

    suggestions = result.get("suggestions", [])
    if not isinstance(suggestions, list):
        return default

    cleaned = []
    seen = set()
    for item in suggestions:
        text = str(item or "").strip()
        if not text:
            continue
        key = _norm_text(text)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= 4:
            break
    return {"suggestions": cleaned}


_REPORT_SUPPORT_SYS = """
You are a compact report support agent for a clinical symptom dashboard.

You will receive a mode plus the relevant inputs.

Modes:
  - topic_summary
  - topic_insight
  - overview

For mode=topic_summary:
  Write a one-sentence human summary of the patient's last check-in for a single topic.
  Rules:
    - Write one natural sentence, not bullet points.
    - Focus on the most clinically relevant details.
    - Prefer plain language.
    - If the data is sparse, write a modest summary instead of inventing detail.
    - Do not mention internal field names.
    - Keep it under 20 words.

For mode=topic_insight:
  Compare one topic from a patient's current check-in against their last check-in.
  Rules:
    - worsened: same issue but clearly more severe, broader, or more disruptive now
    - new_issue: a clinically meaningful issue is present now but was absent or not reported last visit
    - improved: clearly better now than last visit
    - stable: same overall, unchanged, or no meaningful difference
    - unanswered: current topic has essentially no usable answer
    - Keep summaries patient/clinician-readable, plain language, and concise.
    - detail_lines should be short factual lines that help a clinician compare last vs now.
    - attention_lines should include only the most clinically relevant items.

For mode=overview:
  Write a short patient-summary banner for a clinical topic dashboard.
  Rules:
    - Use the provided topic insight summaries.
    - Keep each item short.
    - Mention only the most meaningful changes.
    - If a category has nothing meaningful, return an empty list or null.

Return ONLY valid JSON with the fields needed for the mode:
{
  "summary": "...",
  "status": "worsened|new_issue|improved|stable|unanswered",
  "status_label": "...",
  "last_summary": "..." or null,
  "current_summary": "..." or null,
  "detail_lines": ["...", "..."],
  "attention_lines": ["...", "..."],
  "main_issue": "..." or null,
  "new_issues": ["...", "..."],
  "improvements": ["...", "..."],
  "needs_attention": ["...", "..."]
}
"""


def _run_report_support_agent(mode: str, payload: dict, max_tokens: int) -> dict:
    if not openai_client:
        return {}
    result = _call_agent(_REPORT_SUPPORT_SYS, {"mode": mode, **payload}, max_tokens=max_tokens)
    return result or {}


def run_topic_summary_agent(topic_label: str, topic_data: dict) -> dict:
    default = {"summary": "Information was recorded for this topic."}
    result = _run_report_support_agent("topic_summary", {
        "topic_label": topic_label,
        "topic_data": topic_data,
    }, max_tokens=100)
    return {**default, **result}


def run_report_topic_insight_agent(topic_label: str, last_topic_data: dict, current_topic_data: dict) -> dict:
    default = {
        "status": "unanswered" if not current_topic_data else "stable",
        "status_label": "Not answered" if not current_topic_data else "Stable",
        "last_summary": None,
        "current_summary": None,
        "detail_lines": [],
        "attention_lines": [],
    }
    result = _run_report_support_agent("topic_insight", {
        "topic_label": topic_label,
        "last_topic_data": last_topic_data,
        "current_topic_data": current_topic_data,
    }, max_tokens=220)
    return {**default, **result}


def run_report_overview_agent(topic_insights: list[dict]) -> dict:
    default = {
        "main_issue": None,
        "new_issues": [],
        "improvements": [],
        "needs_attention": [],
    }
    result = _run_report_support_agent("overview", {
        "topic_insights": topic_insights,
    }, max_tokens=180)
    return {**default, **result}


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
   - For location questions with a catch-all option such as "Somewhere else" or "Other",
     any concrete anatomical location that is not one of the named specific options MUST
     map to the catch-all option.
   - If the patient wording directly names one of the listed specific options in natural language,
     prefer that specific option over the catch-all.
   - For yes/no options, treat brief natural replies as valid:
     "fine", "not really", "none", "no issues", "doing okay" should map to the appropriate negative option
     when they clearly answer the current symptom question.
   - Across all topics, prefer the patient's real-world meaning over the literal wording of the option list.
   - Accept common typos, misspellings, speech-to-text errors, shorthand, and close synonyms when the meaning is reasonably clear.
   - If the patient gives a concrete everyday answer and one option is clearly the best fit, map to that option instead of forcing a no_match.
   - This includes locations such as nose, face, cheek, scalp, lip, gums, ear, shoulder,
     chest, back, stomach, leg, foot, or any other specific body area.

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
       "nose"     → "Somewhere else"
       "my face"  → "Somewhere else"
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

TOPIC-HISTORY RULES:
  - You will receive recent conversation history for this topic only.
  - Use it to resolve conversational replies like "yes", "no", "only soup", "my sister helps some",
    or "every day almost" in context of the current question.
  - Do not treat the patient's answer as unrelated just because it is brief; use the immediate topic history.
  - Prefer the most specific valid option, not the most generic one.
  - If the assistant has already been clarifying the same concept, use that context to resolve short confirmation replies like "yes", "right", "that one", or "the left side".
  - Do not force a mapping if the answer is meaningful but clearly does not fit any option; prefer the catch-all option when available.
  - If the patient already gave a concrete location, medication, food type, support source, or other real-world example,
    preserve that meaning by mapping to the correct catch-all option instead of asking them to classify it themselves.

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


def run_answer_interpreter(step: dict, patient_answer: str, topic_history: Optional[list[dict[str, str]]] = None) -> dict:
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
        "recent_topic_history": topic_history or [],
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

# ══════════════════════════════════════════════════════════════════
# AGENT 2 — URGENCY & CRITICALITY
# ══════════════════════════════════════════════════════════════════

_URGENCY_SYS = f"""
You are the Urgency & Criticality Agent for a clinical chatbot serving head and
neck cancer patients. {_HNC_CONTEXT}

You monitor patient safety. Read ALL raw answers across the session — urgency
signals often appear in free-text not captured by structured options.
Be conservative about calling something urgent. Use Tier 2 only for clearly serious,
time-sensitive problems that likely require same-day outreach. When in doubt between
Tier 1 and Tier 2, prefer Tier 1 unless there is a strong acute-risk signal.

RED FLAGS TO DETECT:
{_RED_FLAGS}

TIER DEFINITIONS:
  0 — NO URGENCY: Continue normally.
  1 — WATCH: Notable signal. Log for report, continue session normally.
  2 — URGENT: Care team must contact patient today. Continue session.
       Show one care team message to the patient.
  3 — EMERGENCY: Immediate threat. Terminate session. Patient to emergency services.

STRICT TIERING PRINCIPLE:
  - Tier 2 should be used only for clearly acute, high-risk, or rapidly worsening problems.
  - Moderate symptoms, expected treatment side effects, partial information, or manageable problems
    should usually stay at Tier 0 or Tier 1.
  - Do not use Tier 2 just because something sounds clinically important; use it only when it sounds truly urgent.

SIGNAL RULES:

  IMPORTANT CLINICAL CONTEXT FOR HNC PATIENTS:
  Pain scores of 7-8/10 are clinically expected during active chemoradiation for head
  and neck cancer — mucositis, dysphagia, and treatment toxicity routinely produce this
  level. A high pain score ALONE does not warrant Tier 2 escalation in this population.
  You MUST look for additional signals beyond the number itself.

  Medical signals:
    M1 — SEVERE UNCONTROLLED PAIN:
      Tier 1 (WATCH): Pain 7-9/10 alone, without other signals.
      Tier 2 (URGENT): Pain 7-9/10 AND strong acute-risk evidence such as:
        - Patient states nothing helps OR medication is not working
        - Patient cannot eat or drink at all because of pain
        - Pain is suddenly much worse than baseline
        - Fever/chills present alongside severe pain
      Tier 1 (WATCH): Pain disrupting sleep or daily activities but still sounding manageable / expected during treatment
      Tier 2 (URGENT): Pain reported as 10/10 or "worst of my life" or "unbearable"
    M2: Sudden new severe pain in head/neck — Tier 2
    M3: Fever with chills in an HNC patient (immunocompromised) — Tier 2 always
    M4: Complete inability to swallow ANY liquids — Tier 2
    M5: Breathing difficulty at rest or wheezing — Tier 2
    M6: Prescription medication suddenly stopped — Tier 2 only if it is clearly causing uncontrolled symptoms or creates clear immediate risk; otherwise Tier 1
    M7: Reported falls — Tier 1
    M8: Significant functional disruption (3+ nights no sleep, cannot eat for days, or similarly severe prolonged disruption) — Tier 2
  
  Psychological signals:
    P1: Explicit suicidal ideation or self-harm intent → TIER 3 ALWAYS
    P2: Passive death wish ("I don't care if I make it") → Tier 2
    P3: Crisis desperation ("I can't take this anymore" about life, not just pain) → Tier 2
    P4: Complete isolation ("no one to help me") with high pain → Tier 2 only if the overall picture sounds unstable or unsafe; otherwise Tier 1

  Accumulation: multiple Tier 1 signals alone do NOT automatically require Tier 2 unless together they create a clearly urgent picture.

IMPORTANT NON-ESCALATION GUARDRAILS:
  - Do NOT escalate just because an answer is brief, partial, or missing one detail.
  - Do NOT escalate just because the patient does not remember a dose, timing, or exact amount.
  - Do NOT escalate just because a patient reports PRN or non-daily use without saying it is prescribed daily.
  - Do NOT treat "every 2 days", "sometimes", or similar medication-use frequency by itself as urgent.
  - Do NOT escalate expected but moderate treatment side effects to Tier 2 unless they sound severe, acute, or clearly unsafe.
  - Do NOT escalate manageable pain, manageable nausea, manageable fatigue, or mild functional impact to Tier 2.
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
  - Repeated clarifications can themselves cause frustration; use recent topic history to detect that.

CONTEXT RULES:
  - You will receive recent topic history and recent question texts from this topic.
  - If the assistant has asked very similar clarification questions more than once and the patient replies tersely,
    you may treat that as resistance or declining engagement.
  - Apply this consistently across every topic, not just emotional topics. Repetition-driven frustration can happen anywhere in the interview.
  - Be strict and safety-biased about conversational burden: if there is meaningful evidence that the patient wants less questioning, treat it as real.

DIMENSION SCORES:
  emotional_state: positive|neutral|fatigued|distressed|frustrated|anxious|overwhelmed|resigned
  engagement_level: high|moderate|low|resistant|confused
  engagement_trajectory: stable|improving|declining|insufficient_data

SIGNALS TO DETECT (set to true if present):
  E3_resistance: patient explicitly pushes back ("I already told you", "can we be done")
    or gives terse/frustrated replies after repeated similar clarifications in the same topic
  E7_wants_to_stop: patient clearly wants the conversation paused, stopped, or shortened
    Examples include: "stop", "pause", "skip this", "move on", "enough", "I'm done",
    "I need to stop", "I can't do this right now", or similarly clear control statements
    about how the chat should continue.
    Treat this strictly: if the patient uses short control language that reasonably sounds like ending, pausing,
    or cutting down the questioning, prefer E7_wants_to_stop=true over continuing the interview.
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
  - When reduce_follow_up_depth=true, this means the system should prefer the next formal question or no extra follow-up, not another custom clarification.
  - If the patient's emotional tone is calm/reassuring and their symptom report suggests they are doing okay, you may support less follow-up depth.
  - If E7_wants_to_stop is true, the chat should pause rather than continue probing that topic.
  - If E3_resistance is true and the current question is non-urgent, strongly favor reducing or ending further probing in that topic.

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
    topic_history: list[dict[str, str]],
    recent_questions: list[str],
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
        "recent_topic_history": topic_history,
        "recent_question_texts": recent_questions,
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

You evaluate patient answers from the physician's perspective in one pass. Your outputs:
  1. Clinical sufficiency verdict on the current answer
  2. Follow-up recommendation with a precise information GOAL (not the question itself)
  3. Comparison to the same question from the prior visit when prior data exists
  4. A compact doctor-facing note (≤35 words, third person)
  5. A short patient-facing comparison or acknowledgment note only when useful

FOLLOW-UP RULES:
  - The question list is a question bank, not a rigid script. Judge the current answer
    like a clinician deciding whether anything important is still missing.
  - Follow-up should be rare. If the next formal step already gathers the missing detail, do not request a custom follow-up.
  - These follow-up rules must generalize across all topics and all question types. Do not rely on topic-specific assumptions.
  - Be strict about reducing conversational burden: when there is evidence of frustration, resistance, or a desire to stop, bias strongly toward no follow-up.
  - A meaningful free-text answer in the patient's own words is clinically usable even if it does not match the option wording.
  - If a free-text question contains yes/no wording and the patient gives a simple "yes" or "no", treat that as minimally usable data unless the question clearly asked for a descriptive detail like where, when, how often, or what kind.
  - If the patient gave a broad but meaningful answer, break down what is missing conceptually; do NOT treat it as meaningless.
  - If the patient gives a meaningful negative screen ("no", "not really", "I am okay", "fine") to a broad symptom or emotional check-in question, treat that as a usable answer rather than forcing an unnecessary impact follow-up.
  - If the patient gives a negative screen to a broad opener, set screen_negative_signal=true when the next likely question would otherwise just ask about downstream impact of the same denied problem.
  - If the patient clearly indicates they do not have a problem in that domain, prefer skipping nonessential downstream questions rather than completing the whole branch mechanically.
  - If the patient indicates the symptom is okay, controlled, mild, manageable, resolved, improving, or not affecting function, strongly prefer ending that branch quickly unless a safety-critical detail is still missing.
  - In general, follow-up questions are most useful when there is an active problem, worsening symptom, functional impact, uncontrolled symptom, treatment issue, or meaningful uncertainty.
  - Ask only questions that are still clinically necessary after the patient's actual answer.
  - If the patient already explained the reason in their own words, do NOT recommend a generic "what is making this difficult" follow-up.
  - If the patient supplies one detail and explicitly does not know another, accept the known detail and only ask for the missing one if it is truly necessary.
  - If the missing detail is something the patient reasonably may not know right now, prefer no follow-up over repetitive questioning.
  - Never imply the presence of a symptom the patient just denied.
  - You will receive recent conversation history for this topic only. Use it to avoid repeated questions.
  - Do NOT create a custom follow-up whose only purpose is to ask the same thing as the candidate next step in different words.
  - If the next formal step already covers the natural next question, prefer no custom follow-up and let that next step be asked once.
  - A patient should never have to answer a natural-language version of a question and then immediately answer the form version of the same question.
  - If recent topic history shows the patient has already been asked about the same detail once or twice, strongly prefer no custom follow-up unless the missing detail is clinically high priority.
  - If the assistant has already asked for clarification on the same concept and the patient replies briefly, assume repetition risk is high and avoid another custom follow-up unless the detail is truly essential.
  - If the patient already gave a real-world answer that the system can store directly, prefer next_step_action with carry_forward_answer over follow-up.
  - This is especially important after structured option answers like Yes/No or category selections: if the next formal step can ask the next needed detail directly, prefer no custom bridge follow-up.
  - If the current answer already addresses the candidate next step, set next_step_action to skip that step.
  - If the current answer is reassuring and suggests no active problem, use next_step_action aggressively to skip downstream burden/management questions that only matter when the symptom is present.
  - If several upcoming questions become unnecessary for the same reason, include them in next_step_action.plan.
  - If the patient's raw wording already fully answers the candidate next step, skip that step and carry the raw detail forward instead of asking it again.
  - If the candidate next step, or any proposed follow-up, would substantially repeat a recent question already asked in this topic, suppress it.
  - If a natural assistant acknowledgment has already effectively asked the next question, do not ask it again.
  - If the patient explicitly says they do not know a detail, treat that as usable uncertainty rather than pushing repeatedly.
  - If the patient appears frustrated, resistant, or terse after repeated questioning, downgrade nonessential follow-up across any topic.
  - If the patient appears frustrated, resistant, or controlling the pace of the chat, recommend follow-up only for urgent or clearly high-value missing information.
  - If the patient provided some but not all of the detail, mark it as partial and describe the single missing detail in follow_up_goal.
  - ONLY recommend follow-up if information_completeness is "partial" or "none"
    AND follow_up_count is 0 AND the missing info is clinically meaningful
  - NEVER recommend follow-up if follow_up_count ≥ 1 (absolute limit: 1 per question)
  - NEVER recommend follow-up if patient showed resistance in their answer

PRIOR-COMPARISON RULES:
  - You will receive last_checkin_answer for this same question when available.
  - You should actively use comparable prior data when it exists; comparison is not optional background.
  - Compare current answer to the same question from the prior visit.
  - change_direction:
      improved       — current answer suggests less pain / better status
      worsened       — current answer suggests more pain / worse status
      neutral_change — changed but direction unclear
      no_change      — answers are the same or equivalent
      new_data       — no prior data available
  - change_magnitude:
      numeric severity: large=3+ points difference, moderate=2, small=1, none=0
      non-numeric: large if clinically major, moderate for meaningful change, small for minor wording difference
  - clinical_note:
      one short plain-English comparison sentence; if no prior data, say no prior data available
  - patient_facing_note:
      one short natural sentence only when the comparison adds value
      good uses: weight up/down, pain improved/worsened, symptom burden better/worse
      do NOT use for trivial yes/no comparisons like "Last time you said yes"
  - If there is no comparable prior answer for this exact question but related prior topic data suggests a meaningful comparison, you may use that context conservatively in your reasoning.
  - patient_acknowledgment:
      optional short acknowledgment when uncertainty itself should be accepted naturally
      example: "Thanks for sharing that. It's okay if you're not sure of the exact dose."

SPECIAL CLINICAL SIGNALS (set if present):
  trajectory_mismatch: patient says improving but comparison shows worsening (or vice versa)
  medication_stop_signal: patient stopped taking prescription medication without explanation
  aggravating_medication_signal: patient reports their medication makes symptoms worse
  severity_underreporting: patient rates low severity but describes severe functional impact
  screen_negative_signal: patient's answer functions as a meaningful negative screen for the symptom/concern being assessed

next_step_action:
  - Use this to suppress an immediate next question when it no longer makes clinical sense
  - This is the main mechanism for skipping downstream impact/management/change questions generically across topics
  - Only use it when the candidate next step would be unnecessary, redundant, or context-mismatched given the current answer and session answers
  - Prefer using next_step_action instead of follow-up whenever the issue is redundant branching rather than truly missing information
  - When the patient's answer is reassuring or indicates the symptom is okay, controlled, absent, or not bothersome, prefer next_step_action over additional questioning whenever clinically safe
  - Prefer suggested_answer to be an exact option from the candidate next step when obvious, often "No"
  - plan is optional and may list additional upcoming steps that should also be auto-resolved to avoid unnecessary questioning
  - carry_forward_answer is optional and should be used when the patient's current raw answer already provides the value for a downstream step, especially a free-text detail step that would otherwise repeat the same question
  - Good examples:
    - Patient denies emotional distress and next step asks whether anxiety is affecting sleep/eating → skip with suggested_answer "No"
    - Patient denies depression or feeling down and the next questions only elaborate on mood burden or support needs → skip them unless there is another clear concern
    - Patient says the symptom is mild, okay, or not bothering them and the next questions are about burden, management failure, or escalation → skip those downstream questions
    - Patient answers a location chooser with a specific body part like "nose" and the next step asks which body part hurts → skip that next step and carry forward "nose"
    - Patient says a sore is not painful and next step asks whether treatment for painful sores is helping → skip with suggested_answer "No"
    - Patient says IV fluids are helping and gives no sign they want changes, and next step asks about adjusting frequency → skip with suggested_answer "No"
    - Patient says medication is not causing drowsiness and next step asks whether drowsiness is affecting schedule → skip with suggested_answer "No"
  - Do not use this to skip structurally essential questions like a severity rating or a new symptom location unless the current answer already fully covers them

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
  "patient_acknowledgment": "..." or null,
  "answered_with_uncertainty": false,
  "has_prior_data": true/false,
  "last_answer": "..." or null,
  "change_detected": true/false,
  "change_direction": "improved|worsened|neutral_change|no_change|new_data",
  "change_magnitude": "large|moderate|small|none",
  "clinical_note": "..." or null,
  "patient_facing_note": "..." or null,
  "change_significance": "critical|notable|stable|no_baseline",
  "clinical_priority": "high|medium|low",
  "doctor_note": "..." or null,
  "next_step_action": {{
    "skip_immediate_next_step": false,
    "suggested_answer": "..." or null,
    "reason": "..." or null,
    "carry_forward_answer": "..." or null,
    "plan": [
      {{
        "step_id": "...",
        "suggested_answer": "..." or null,
        "carry_forward_answer": "..." or null,
        "reason": "..." or null
      }}
    ]
  }},
  "special_signals": {{
    "trajectory_mismatch": false,
    "medication_stop_signal": false,
    "aggravating_medication_signal": false,
    "severity_underreporting": false,
    "screen_negative_signal": false
  }}
}}
"""


def run_doctor_relevance(
    step: dict,
    current_answer_raw: str,
    current_answer_matched: Optional[str],
    last_topic_data: dict,
    session_answers: dict,
    followup_count: int,
    topic_history: list[dict[str, str]],
    recent_questions: list[str],
    candidate_next_step: Optional[dict] = None,
    upcoming_steps: Optional[list[dict]] = None,
) -> dict:
    """
    Clinical assessment agent: assess sufficiency, prior-visit comparison,
    and follow-up relevance in one pass.
    """
    last_answer = None
    if last_topic_data:
        raw_last = last_topic_data.get(step["id"])
        if raw_last is not None:
            last_answer = str(raw_last)
    default = {
        "information_completeness": "complete", "clinical_value_score": 0.7,
        "follow_up_recommended": False, "follow_up_goal": None,
        "follow_up_urgency": "none", "change_significance": "no_baseline",
        "patient_acknowledgment": None,
        "answered_with_uncertainty": False,
        "has_prior_data": bool(last_answer),
        "last_answer": last_answer,
        "change_detected": False,
        "change_direction": "new_data" if last_answer is None else "no_change",
        "change_magnitude": "none",
        "clinical_note": "No prior data available." if last_answer is None else "",
        "patient_facing_note": None,
        "clinical_priority": "medium", "doctor_note": None,
        "next_step_action": {
            "skip_immediate_next_step": False,
            "suggested_answer": None,
            "reason": None,
            "carry_forward_answer": None,
            "plan": [],
        },
        "special_signals": {
            "trajectory_mismatch": False, "medication_stop_signal": False,
            "aggravating_medication_signal": False, "severity_underreporting": False,
            "screen_negative_signal": False,
        },
    }
    result = _call_agent(_DOCTOR_RELEVANCE_SYS, {
        "question_text": step.get("text", ""),
        "question_type": step.get("type", "options"),
        "options": step.get("opts", []),
        "current_answer_raw": current_answer_raw,
        "current_answer_matched": current_answer_matched,
        "last_checkin_answer": last_answer,
        "session_answers_so_far": session_answers,
        "recent_topic_history": topic_history,
        "recent_question_texts": recent_questions,
        "follow_up_count_this_question": followup_count,
        "candidate_next_step": {
            "id": candidate_next_step.get("id"),
            "text": candidate_next_step.get("text"),
            "type": candidate_next_step.get("type"),
            "options": candidate_next_step.get("opts", []),
        } if candidate_next_step else None,
        "upcoming_steps": [
            {
                "id": s.get("id"),
                "text": s.get("text"),
                "type": s.get("type"),
                "options": s.get("opts", []),
            }
            for s in (upcoming_steps or [])
        ],
    }, max_tokens=400)

    if not result:
        return default

    # Hard-enforce the follow-up limit
    if followup_count >= 1:
        result["follow_up_recommended"] = False
        result["follow_up_goal"] = None

    merged = {**default, **result}
    if "next_step_action" in result and isinstance(result["next_step_action"], dict):
        merged["next_step_action"] = {**default["next_step_action"], **result["next_step_action"]}
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
  - Never contradict an explicit "no" or "yes" the patient just gave
  - Never write a follow-up like "Besides anxiety..." or otherwise imply a symptom exists after the patient denied it
  - You will receive recent topic history and recent question texts from this topic only
  - Do not write a question that substantially repeats any recent question in that history
  - If the candidate next step already asks the same thing, return null instead of paraphrasing it
  - If the follow-up goal materially overlaps the candidate next step, return null
  - If recent topic history suggests the patient is getting frustrated by repetition, return null rather than asking another version of the same detail
  - These rules apply across every topic and every question type. When in doubt, avoid repeating the same concept in a new wording.
  - If the patient appears okay with respect to the symptom being discussed and there is no active problem to explore, return null rather than creating another follow-up.
  - Be strict: if there is credible evidence the patient wants less questioning, return null unless the missing detail is urgent.
  - If comparable last-visit information is provided and it helps make the question clearer, you may use it briefly to frame the question naturally.
  - Good uses of prior history: "Is that still about the same as last time?" or "Is this lower than your usual weight?" when such wording is directly supported by the provided history.
  - Do not mention prior history if it would sound awkward, speculative, or repetitive.
  - After a structured option answer, do not pre-ask the next formal step in different words just to sound conversational.
  - Never ask the patient to translate their own concrete answer into the form's categories. For example, after a patient says "nose", do not ask "throat, tongue, or somewhere else?" because that classification should happen internally.
  - More generally: do not ask the patient to convert a real-world answer into the app's taxonomy when the system can infer it.
  - If the patient's answer already gives a concrete real-world example, assume the system can preserve it and ask only the next clinically meaningful question.
  - If prior-comparison context is clinically useful, you may briefly reflect it in a natural way, but only as conversational context, never as a rigid template
  - Write in second person, conversational language
  - Never use medical jargon without immediate plain explanation
  - NEVER ask a multi-part question
  - NEVER repeat the original question verbatim
  - Keep the question to ≤25 words
  - An acknowledgment may be shown BEFORE your question — do not repeat it
  - If simplify=true: use the shortest phrasing possible

Return ONLY valid JSON:
{{
  "follow_up_question": "..." or null,
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
    topic_history: list[dict[str, str]],
    recent_questions: list[str],
    candidate_next_step: Optional[dict],
    last_checkin_answer: Optional[str] = None,
    candidate_next_step_last_answer: Optional[str] = None,
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
        "recent_topic_history": topic_history,
        "recent_question_texts": recent_questions,
        "last_checkin_answer": last_checkin_answer,
        "candidate_next_step": {
            "id": candidate_next_step.get("id"),
            "text": candidate_next_step.get("text"),
            "type": candidate_next_step.get("type"),
            "options": candidate_next_step.get("opts", []),
        } if candidate_next_step else None,
        "candidate_next_step_last_answer": candidate_next_step_last_answer,
    }, max_tokens=120)

    if result and "follow_up_question" in result:
        return result
    # Fallback: derive a question from the goal
    return {
        "follow_up_question": "Could you tell me a bit more about that?",
        "preamble": None,
    }


_ONE_PASS_PIPELINE_SYS = f"""
You are the Single-Pass Clinical Orchestrator for ChatReport, a clinical chatbot
serving head and neck cancer patients. {_HNC_CONTEXT}

Your job is to make all live conversation decisions in ONE response:
1. Interpret the patient's current answer against the current question/options.
2. Screen urgency and safety.
3. Detect whether the patient wants to stop or needs less follow-up.
4. Decide whether the answer is clinically sufficient.
5. Decide whether to ask one follow-up, skip downstream questions, or move on.
6. Write any short patient-facing acknowledgment/message.

IMPORTANT PERFORMANCE CONTRACT:
- This is the only model call for the current answer.
- Return the final follow-up question yourself if one is truly needed.
- Do not rely on later agents.

MATCHING RULES:
- If the answer clearly maps to one option, return that option verbatim.
- If a catch-all option exists such as "Somewhere else", "Other", "None of these",
  or "Something else", use it for meaningful answers that do not match a specific option.
- If the question asks for a body location and the answer names a real body part,
  that is meaningful. Map to the best specific option or catch-all.
- Do not accept wrong-type answers. For example, if asked where pain is and the
  patient gives timing, matched_option should be null.
- For yes/no options, natural answers like "yeah", "not really", "fine", "none",
  or "no issues" may map to the appropriate option when clear.

URGENCY RULES:
{_RED_FLAGS}

Urgency tiers:
0 no urgency, continue normally.
1 watch/log for report, continue normally.
2 urgent same-day care-team attention, show a care team message and pause topic.
3 emergency, stop session and show emergency guidance.

Use Tier 2 only for clearly acute or unsafe situations. Pain 7-9/10 alone in active
HNC treatment is usually Tier 1 unless uncontrolled, suddenly worse, preventing all
intake, or accompanied by fever/chills. Suicidal ideation or self-harm intent is
Tier 3.

FOLLOW-UP RULES:
- Follow-up should be rare.
- Ask at most one follow-up for the current question.
- If the patient gives a meaningful, clinically usable answer in their own words,
  accept it even if it does not use the exact option wording.
- Do not ask for clarification just because the answer is not phrased like the
  predefined answers. Map it to the closest option or catch-all when reasonable.
- Only ask "I did not catch that" style clarification when the answer is empty,
  nonsensical, unrelated, or the wrong information type for the question.
- Never ask a follow-up that repeats the current question or the candidate next step.
- If the candidate next step naturally gathers the missing detail, do not ask a
  custom follow-up.
- If the patient appears frustrated, resistant, tired, or wants to stop, avoid
  nonessential follow-up.
- If the patient answers with a reassuring negative screen, skip unnecessary
  downstream impact/management questions when safe.
- If the patient already gave detail that answers the immediate next step, use
  next_step_action to skip/carry it forward instead of asking again.
- Numeric severity questions may need one clarification if no usable number was given.

Return ONLY valid JSON with this exact shape:
{{
  "matched_option": "..." or null,
  "follow_up": true/false,
  "follow_up_question": "..." or "",
  "acknowledgment": "..." or "",
  "assistant_message": "..." or "",
  "urgency_tier": 0,
  "urgency_message": null,
  "reduce_follow_up": false,
  "wants_to_stop": false,
  "doctor_note": null,
  "clinical_priority": "low|medium|high",
  "follow_up_goal": null,
  "change_significance": "critical|notable|stable|no_baseline",
  "change_clinical_note": "",
  "next_step_action": {{
    "skip_immediate_next_step": false,
    "suggested_answer": null,
    "reason": null,
    "carry_forward_answer": null,
    "plan": []
  }},
  "special_signals": {{
    "trajectory_mismatch": false,
    "medication_stop_signal": false,
    "aggravating_medication_signal": false,
    "severity_underreporting": false,
    "screen_negative_signal": false
  }},
  "sentiment_note": null,
  "new_urgency_signals": [],
  "new_sentiment_signals": []
}}
"""


def run_single_pass_pipeline_agent(
    topic_key: str,
    step: dict,
    current_raw_answer: str,
    session_answers: dict,
    prior_baseline: dict,
    last_topic_data: dict,
    followup_count: int,
    topic_history: list[dict[str, str]],
    recent_questions: list[str],
    candidate_next_step: Optional[dict],
    upcoming_steps: list[dict],
    active_urgency_signals: list,
    active_sentiment_signals: list,
) -> dict:
    default = {
        **_pipeline_default(),
        "patient_answer": current_raw_answer,
        "new_urgency_signals": [],
        "new_sentiment_signals": [],
    }
    result = _call_agent(_ONE_PASS_PIPELINE_SYS, {
        "topic_key": topic_key,
        "current_question": {
            "id": step.get("id"),
            "text": step.get("text", ""),
            "type": step.get("type", "options"),
            "options": step.get("opts", []),
        },
        "patient_answer": current_raw_answer,
        "session_answers_so_far": session_answers,
        "prior_baseline_for_topic": prior_baseline,
        "last_topic_data": last_topic_data,
        "last_answer_for_same_question": last_topic_data.get(step.get("id")) if last_topic_data else None,
        "follow_up_count_this_question": followup_count,
        "recent_topic_history": topic_history,
        "recent_question_texts": recent_questions,
        "candidate_next_step": {
            "id": candidate_next_step.get("id"),
            "text": candidate_next_step.get("text"),
            "type": candidate_next_step.get("type"),
            "options": candidate_next_step.get("opts", []),
        } if candidate_next_step else None,
        "upcoming_steps": [
            {
                "id": s.get("id"),
                "text": s.get("text"),
                "type": s.get("type"),
                "options": s.get("opts", []),
            }
            for s in (upcoming_steps or [])
        ],
        "active_urgency_signals": active_urgency_signals,
        "active_sentiment_signals": active_sentiment_signals,
        "patient_fatigue": bool(st.session_state.get("patient_fatigue", False)),
    }, max_tokens=650)

    if not result:
        return default

    matched = result.get("matched_option")
    if matched and matched not in step.get("opts", []):
        matched = None
    merged = {**default, **result, "matched_option": matched}

    if followup_count >= 1:
        merged["follow_up"] = False
        merged["follow_up_question"] = ""
        merged["follow_up_goal"] = None

    if st.session_state.get("patient_fatigue") and merged.get("clinical_priority") != "high":
        merged["reduce_follow_up"] = True
        merged["follow_up"] = False
        merged["follow_up_question"] = ""

    if merged.get("wants_to_stop"):
        merged["follow_up"] = False
        merged["follow_up_question"] = ""

    if merged.get("urgency_tier", 0) == 3:
        merged["wants_to_stop"] = True

    if merged.get("follow_up") and not str(merged.get("follow_up_question") or "").strip():
        merged["follow_up"] = False

    if not str(merged.get("assistant_message") or "").strip() and not merged.get("follow_up"):
        merged["assistant_message"] = _default_chatty_reply(topic_key, current_raw_answer, step, last_topic_data)

    if merged.get("follow_up") and merged.get("follow_up_question"):
        merged["demo_decision"] = f"Ask one follow-up: {merged.get('follow_up_question')}"
    elif merged.get("wants_to_stop"):
        merged["demo_decision"] = "Pause or stop because the patient signaled they want to stop."
    elif merged.get("urgency_tier", 0) >= 2:
        merged["demo_decision"] = "Pause and show care-team or emergency guidance."
    elif merged.get("next_step_action"):
        merged["demo_decision"] = "Apply skip/carry-forward logic, then move to the next flowchart question."
    else:
        merged["demo_decision"] = "Move to the next applicable flowchart question."

    return merged


def _deterministic_head_neck_focus(location_text: str) -> bool:
    normalized = _norm_text(location_text)
    focused_terms = {
        "ear", "ears", "jaw", "mouth", "lip", "lips", "gum", "gums", "tooth",
        "teeth", "cheek", "palate", "tongue", "throat",
    }
    words = set(normalized.split())
    return bool(words & focused_terms)


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


def _build_topic_history(topic_key: str) -> list[dict[str, str]]:
    state = st.session_state.topic_states.get(topic_key, {})
    return _recent_topic_history(state)


def _build_recent_question_texts(topic_key: str) -> list[str]:
    state = st.session_state.topic_states.get(topic_key, {})
    return _recent_topic_questions(state)


def _build_prior_baseline(topic_key: str) -> dict:
    """Return a compact prior check-in summary for agent context."""
    last = st.session_state.last_checkin.get(topic_key, {})
    if not last:
        return {}
    # Return key fields only to keep the payload small
    keys = list(last.keys())[:10]
    return {k: str(last[k]) for k in keys}


def _last_checkin_answer(topic_key: str, step_id: Optional[str]) -> Optional[str]:
    if not topic_key or not step_id:
        return None
    value = st.session_state.get("last_checkin", {}).get(topic_key, {}).get(step_id)
    if value is None:
        return None
    return str(value)


def _build_all_topic_data() -> dict:
    payload = {}
    topic_progress = {}
    for _, key in TOPICS:
        topic_state = st.session_state.topic_states[key]
        topic_data = dict(topic_state.get("data", {}))
        raw_answers = topic_state.get("raw_answers", {})
        if raw_answers:
            topic_data["_verbatim_answers"] = dict(raw_answers)
        payload[key] = topic_data
        answered, applicable = get_topic_progress(key, topic_state.get("data", {}), raw_answers)
        topic_progress[key] = {
            "status": topic_state.get("status", "not_started"),
            "answered": answered,
            "applicable": applicable,
        }
    payload["_topic_progress"] = topic_progress
    payload["_structured_responses"] = st.session_state.get("structured_responses", [])
    payload["_patient_context"] = {
        "patient_fatigue": bool(st.session_state.get("patient_fatigue", False)),
        "fatigue_requested_at": st.session_state.get("fatigue_requested_at"),
    }
    return payload


def run_agent_pipeline(
    topic_key: str,
    step: dict,
    answer: str,
    raw_answer: Optional[str],
    state: dict,
    last_topic_data: dict,
) -> dict:
    """
    Single live-model orchestration call for typed/free-text/voice answers.
    Predefined structured clicks bypass this function before it is called.
    """
    if not openai_client:
        return _pipeline_default()

    current_raw_answer = str(raw_answer if raw_answer is not None else answer)
    session_answers = _build_session_answers(topic_key)
    prior_baseline  = _build_prior_baseline(topic_key)
    followup_count  = state.get("followup_counts", {}).get(step["id"], 0)
    topic_history = _build_topic_history(topic_key)
    recent_questions = _build_recent_question_texts(topic_key)
    candidate_next_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
    upcoming_steps = get_upcoming_steps(topic_key, state["data"], state.get("raw_answers"), limit=5)
    active_urgency_signals = st.session_state.get("urgency_state", {}).get("all_signals", [])
    active_sentiment_signals = st.session_state.get("sentiment_state", {}).get("all_signals", [])

    pipeline = run_single_pass_pipeline_agent(
        topic_key=topic_key,
        step=step,
        current_raw_answer=current_raw_answer,
        session_answers=session_answers,
        prior_baseline=prior_baseline,
        last_topic_data=last_topic_data,
        followup_count=followup_count,
        topic_history=topic_history,
        recent_questions=recent_questions,
        candidate_next_step=candidate_next_step,
        upcoming_steps=upcoming_steps,
        active_urgency_signals=active_urgency_signals,
        active_sentiment_signals=active_sentiment_signals,
    )

    tier = _safe_int_value(pipeline.get("urgency_tier"), 0)
    _merge_urgency_state(tier, {"new_signals": pipeline.get("new_urgency_signals", [])})
    _merge_sentiment_state({
        "signals": {signal: True for signal in pipeline.get("new_sentiment_signals", [])},
        "engagement_trajectory": "declining" if pipeline.get("reduce_follow_up") else "stable",
        "emotional_state": "fatigued" if pipeline.get("reduce_follow_up") else "neutral",
    })

    if tier == 2:
        default_msg = (
            "Thank you for sharing this with us. We can see you're having a really difficult time. "
            "A member of your care team will be reaching out to you today. Please keep your phone nearby. "
            "Your responses have been saved."
        )
        pipeline["urgency_message"] = pipeline.get("urgency_message") or default_msg
    elif tier == 3:
        default_msg = (
            "We're concerned about what you've shared. Please call 911 or go to your nearest "
            "emergency room immediately. Your care team has been notified."
        )
        pipeline["urgency_message"] = pipeline.get("urgency_message") or default_msg
        pipeline["wants_to_stop"] = True

    urg_state = st.session_state.get("urgency_state", {})
    if tier == 2 and pipeline.get("urgency_message") and not urg_state.get("escalation_shown", False):
        urg_state["escalation_shown"] = True
        st.session_state["urgency_state"] = urg_state
    elif tier == 2:
        pipeline["urgency_message"] = None

    return pipeline


def _pipeline_default() -> dict:
    """Safe default when agents are unavailable."""
    return {
        "matched_option": None, "follow_up": False, "follow_up_question": "",
        "acknowledgment": "", "assistant_message": "",
        "urgency_tier": 0, "urgency_message": None,
        "reduce_follow_up": False, "wants_to_stop": False,
        "doctor_note": None, "clinical_priority": "medium",
        "change_significance": "no_baseline", "change_clinical_note": "",
        "next_step_action": None,
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

def interpret_user_input_with_options(step, user_input, topic_history: Optional[list[dict[str, str]]] = None):
    """
    Classify simple typed text against question options without a model call.
    Falls back to catch-all options for clear deterministic cases.
    Returns matched option string if found, else original input.
    """
    if not step.get("opts"):
        return user_input

    normalized = _norm_text(user_input)
    for opt in step.get("opts", []):
        if _norm_text(opt) == normalized:
            return opt

    local_match = _local_option_match(step, user_input)
    if local_match:
        return local_match

    opts = step.get("opts", [])
    if (
        "Somewhere else" in opts
        and _looks_like_body_location_phrase(user_input)
        and ("where" in _norm_text(step.get("text", "")) or "location" in _norm_text(step.get("text", "")))
    ):
        return "Somewhere else"

    return user_input


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
[One bold subsection per topic with any current-session data, even if the topic was not fully completed. Include:
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
- Treat partially completed topics as valid current-session data when answers are present
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
    return "I've noted that for your care team."



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
                "last_prompted_step_id": None,
                "last_prompted_text": "",
                "generated_prompts": {},
                "generated_quick_replies": {},
            }
            for _, key in TOPICS
        },
        "report":              "",
        "report_saved":        False,
        "last_checkin":        {},
        "has_prev_checkin":    False,
        "freeform_chat":       [],
        "structured_responses": [],
        "agent_traces":        [],
        "last_agent_trace":    None,
        "demo_mode":           False,
        "patient_fatigue":     False,
        "fatigue_requested_at": None,
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


def _fresh_topic_states() -> dict:
    return {
        key: {
            "status": "not_started",
            "data": {},
            "chat": [],
            "followup_counts": {},
            "raw_answers": {},
            "last_prompted_step_id": None,
            "last_prompted_text": "",
            "generated_prompts": {},
            "generated_quick_replies": {},
        }
        for _, key in TOPICS
    }


def _clear_transient_widget_state():
    transient_prefixes = (
        "radio_",
        "text_",
        "dropdown_",
        "voice_",
        "multi_",
        "num_",
        "num_text_",
        "suggested_",
        "unit_",
        "ft_",
        "pending_followup_",
        "_vt_",
        "_vh_",
        "_vrec_",
    )
    transient_exact = {
        "freeform_chat_input",
    }
    for key in list(st.session_state.keys()):
        if key in transient_exact or key.startswith(transient_prefixes):
            st.session_state.pop(key, None)


def _reset_checkin_session_state(preserve_demo_mode: bool = True):
    demo_mode = bool(st.session_state.get("demo_mode", False)) if preserve_demo_mode else False
    _clear_transient_widget_state()
    st.session_state.topic_states = _fresh_topic_states()
    st.session_state.selected_topic = TOPIC_KEYS[0] if TOPIC_KEYS else None
    st.session_state.report = ""
    st.session_state.report_saved = False
    st.session_state.freeform_chat = []
    st.session_state.structured_responses = []
    st.session_state.agent_traces = []
    st.session_state.last_agent_trace = None
    st.session_state.patient_fatigue = False
    st.session_state.fatigue_requested_at = None
    st.session_state.urgency_state = {
        "current_tier": 0,
        "all_signals": [],
        "escalation_shown": False,
        "emergency_shown": False,
        "patient_message": None,
    }
    st.session_state.sentiment_state = {
        "all_signals": [],
        "engagement_trajectory": "insufficient_data",
        "emotional_state": "neutral",
    }
    st.session_state.demo_mode = demo_mode


def _invalidate_report_cache():
    st.session_state.report = ""
    st.session_state.report_saved = False


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
        ("pain_severity",       "Severity"),
        ("pain_timing",         "Timing"),
        ("pain_medications",    "Medications"),
        ("med_side_effects",    "Medication side effects"),
    ],
    "nutrition": [
        ("eating_ability",        "Eating"),
        ("swallowing_difficulty", "Swallowing"),
        ("feeding_tube",          "Feeding tube"),
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
    ],
    "activity": [
        ("activity_level",           "Activity level"),
        ("activity_limiting_factor", "Limiting factor"),
    ],
    "mood": [
        ("emotional_state",   "Emotional state"),
    ],
    "other": [
        ("additional_symptoms", "Additional symptoms"),
        ("breathing_issues",  "Breathing"),
        ("hearing_changes",   "Hearing"),
        ("dizziness",         "Dizziness"),
        ("numbness",          "Numbness/tingling"),
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
    """Return a short rule-based sentence summarising a topic's previous answers."""
    if not data:
        return ""
    fields = _SUMMARY_FIELDS.get(topic_key, [])
    parts = []
    for field_id, label in fields:
        value = data.get(field_id)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            value_text = ", ".join(str(v) for v in value if str(v).strip())
        else:
            value_text = str(value)
        value_text = value_text.strip()
        if not value_text:
            continue
        if len(value_text) > 42:
            value_text = value_text[:39] + "..."
        parts.append(f"{label}: {value_text}")
        if len(parts) >= 2:
            break
    if parts:
        return "; ".join(parts)
    answered = len([k for k, v in data.items() if v not in (None, "", [], {}) and not str(k).endswith("_other_detail")])
    return f"{answered} details recorded last visit." if answered else ""


def _report_topic_fallback(topic_key: str, topic_label: str, last_topic_data: dict, current_topic_data: dict) -> dict:
    last_summary = _natural_summary(topic_key, last_topic_data) if last_topic_data else "No prior details recorded."
    current_summary = _natural_summary(topic_key, current_topic_data) if current_topic_data else "Not answered this visit."
    if not current_topic_data:
        status = "unanswered"
    elif not last_topic_data:
        status = "new_issue"
    else:
        status = "stable"
    status_label = {
        "worsened": "Worsened",
        "new_issue": "New issue",
        "improved": "Improved",
        "stable": "Stable",
        "unanswered": "Not answered",
    }.get(status, "Stable")
    return {
        "topic_key": topic_key,
        "topic_label": topic_label,
        "status": status,
        "status_label": status_label,
        "last_summary": last_summary,
        "current_summary": current_summary,
        "detail_lines": [],
        "attention_lines": [],
    }


def _report_topic_insights(all_data: dict) -> list[dict]:
    insights = []
    last_ck = st.session_state.get("last_checkin", {})
    for label, key in TOPICS:
        current_topic_data = all_data.get(key, {}) or {}
        last_topic_data = last_ck.get(key, {}) or {}
        insights.append(_report_topic_fallback(key, label, last_topic_data, current_topic_data))
    return insights


def _report_status_class(status: str) -> str:
    if status in {"worsened", "new_issue"}:
        return "red"
    if status == "improved":
        return "green"
    return ""


def _report_status_text(status: str) -> str:
    mapping = {
        "improved": "Improved",
        "worsened": "Worsened",
        "new_issue": "New issue",
        "stable": "No meaningful change",
        "unanswered": "No current information",
    }
    return mapping.get(status, "No meaningful change")


def _render_report_summary_banner(topic_insights: list[dict]):
    attention_topics = [
        str(insight.get("topic_label", "")).split(" ", 1)[1]
        for insight in topic_insights
        if insight.get("status") in {"worsened", "new_issue"} and insight.get("current_summary")
    ]
    main_issue = (
        f"New or updated information recorded for {', '.join(attention_topics[:3])}."
        if attention_topics else
        "Clinical check-in summary ready for review."
    )
    new_issues = attention_topics[:3]
    improvements = [
        str(insight.get("topic_label", "")).split(" ", 1)[1]
        for insight in topic_insights
        if insight.get("status") == "improved"
    ][:3]
    needs_attention = [
        str(insight.get("topic_label", "")).split(" ", 1)[1]
        for insight in topic_insights
        if insight.get("attention_lines")
    ][:4]

    parts = [
        '<div class="report-summary-banner">',
        '<div class="report-summary-avatar">🧑</div>',
        '<div class="report-summary-copy">',
        '<div class="report-summary-title">Patient Summary</div>',
        f'<div><strong>Main issue:</strong> {_html.escape(str(main_issue))}</div>',
    ]
    if new_issues:
        parts.append(f'<div><strong>New issues:</strong> {_html.escape("; ".join(str(x) for x in new_issues[:3]))}</div>')
    if improvements:
        parts.append(f'<div><strong>Improvement:</strong> {_html.escape("; ".join(str(x) for x in improvements[:3]))}</div>')
    if needs_attention:
        parts.append(f'<div><strong>Needs attention:</strong> {_html.escape("; ".join(str(x) for x in needs_attention[:4]))}</div>')
    parts.extend(['</div>', '</div>'])
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_report_topic_card(insight: dict):
    status_class = _report_status_class(insight.get("status", "stable"))
    status_label = _report_status_text(insight.get("status", "stable"))
    topic_name = insight.get("topic_label", "").split(" ", 1)[1] if " " in insight.get("topic_label", "") else insight.get("topic_label", "")
    st.markdown(
        f'<div class="report-topic-card {status_class}">'
        f'  <div class="report-topic-strip">{_html.escape(status_label)}</div>'
        f'  <div class="report-topic-body">'
        f'    <div class="report-topic-name">{_html.escape(topic_name)}</div>'
        f'    <div class="report-topic-compare">'
        f'      <div class="report-topic-compare-row"><div class="report-topic-compare-label">Last</div><div class="report-topic-compare-value">{_html.escape(str(insight.get("last_summary") or "No prior details recorded."))}</div></div>'
        f'      <div class="report-topic-compare-row"><div class="report-topic-compare-label">Now</div><div class="report-topic-compare-value">{_html.escape(str(insight.get("current_summary") or "Not answered this visit."))}</div></div>'
        f'    </div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_report_topic_detail(insight: dict, all_data: dict):
    topic_key = insight["topic_key"]
    last_topic_data = st.session_state.get("last_checkin", {}).get(topic_key, {}) or {}
    current_topic_data = all_data.get(topic_key, {}) or {}

    with st.expander("More details", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Last check-in**")
            st.markdown(str(insight.get("last_summary") or "No prior details recorded."))
        with col2:
            st.markdown("**Current check-in**")
            st.markdown(str(insight.get("current_summary") or "Not answered this visit."))

        if insight.get("attention_lines"):
            st.markdown("**Key points**")
            for line in insight["attention_lines"]:
                st.markdown(f"- {line}")
        if insight.get("detail_lines"):
            st.markdown("**Comparison details**")
            for line in insight["detail_lines"]:
                st.markdown(f"- {line}")

        last_html = _checkin_summary_html(topic_key, last_topic_data)
        now_html = _checkin_summary_html(topic_key, current_topic_data)
        if last_html or now_html:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Last visit details**")
                st.markdown(last_html or '<div style="color:#7a8ea4;">No prior details recorded.</div>', unsafe_allow_html=True)
            with col2:
                st.markdown("**Current visit details**")
                st.markdown(now_html or '<div style="color:#7a8ea4;">No current details recorded.</div>', unsafe_allow_html=True)

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
    prompt_consumed = False
    if message and next_text and _is_semantically_redundant_question(message, next_text):
        prompt_consumed = True
    if next_text and not prompt_consumed:
        combined = "\n\n".join(part for part in [message, next_text] if part)
        _append_assistant_message(state, combined, prompt_step=next_step, prompt_text=next_text)
        _remember_prompted_step(state, next_step, next_text)
    elif message:
        _append_assistant_message(state, message)
    elif not next_step:
        _clear_current_prompt_flags(state)
        _remember_prompted_step(state, None, "")


def _maybe_skip_next_impact_question(topic_key: str, state: dict):
    return


def _apply_agent_next_step_action(topic_key: str, state: dict, action: Optional[dict]):
    if not action:
        return

    plan = []
    if action.get("skip_immediate_next_step"):
        next_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
        if next_step:
            plan.append({
                "step_id": next_step.get("id"),
                "suggested_answer": action.get("suggested_answer"),
                "carry_forward_answer": action.get("carry_forward_answer"),
                "reason": action.get("reason"),
            })
    for item in action.get("plan", []) or []:
        if isinstance(item, dict) and item.get("step_id"):
            plan.append(item)

    for item in plan:
        step_id = item.get("step_id")
        step = STEP_BY_ID.get(step_id)
        if not step or step_id in state["data"]:
            continue
        if step.get("type") == "free_text":
            carry = item.get("carry_forward_answer")
            if isinstance(carry, str) and carry.strip():
                value = carry.strip()
                state["data"][step_id] = value
                state["raw_answers"][step_id] = value
            continue
        if step.get("type") == "options":
            opts = step.get("opts", [])
            suggested = item.get("suggested_answer")
            chosen = None
            if suggested in opts:
                chosen = suggested
            else:
                normalized_opts = {_norm_text(opt): opt for opt in opts}
                if "no" in normalized_opts:
                    chosen = normalized_opts["no"]
                else:
                    for opt in opts:
                        if _norm_text(opt).startswith("no"):
                            chosen = opt
                            break
            if chosen:
                state["data"][step_id] = chosen
                state["raw_answers"][step_id] = chosen


def _apply_generic_fallback_next_step_action(topic_key: str, state: dict):
    return


def _maybe_apply_prompt_driven_skip(topic_key: str, state: dict, pipeline: dict):
    return


def _capture_rich_answer_into_next_step(
    topic_key: str,
    state: dict,
    current_step: dict,
    resolved_answer: Any,
    raw_answer: Any,
):
    """
    If a patient answers an option question with richer free text that already
    answers the immediate next step, capture that detail now so the app does not
    ask for it again in a different form.
    """
    if current_step.get("type") != "options":
        return

    raw_text = str(raw_answer or "").strip()
    if not raw_text:
        return
    if isinstance(resolved_answer, str) and _norm_text(raw_text) == _norm_text(resolved_answer):
        return

    next_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
    if not next_step or next_step.get("id") in state["data"]:
        return
    if next_step.get("type") != "options":
        return
    next_opts_norm = {_norm_text(opt) for opt in next_step.get("opts", [])}
    if next_opts_norm == {"yes", "no"}:
        return

    interpreted = interpret_user_input_with_options(
        next_step,
        raw_text,
        topic_history=_recent_topic_history(state),
    )
    if interpreted not in next_step.get("opts", []):
        return

    coerced = _coerce_structured_answer(
        topic_key,
        next_step,
        interpreted,
        state["data"],
        raw_answer=raw_text,
    )
    state["data"][next_step["id"]] = coerced
    state["raw_answers"][next_step["id"]] = raw_text


def _backfill_next_step_from_topic_history(topic_key: str, state: dict, next_step: Optional[dict]):
    """
    Safety net: if the next unresolved step is a location chooser with a catch-all
    and the patient already named a concrete body part earlier in this topic,
    resolve it internally instead of surfacing the chooser again.
    """
    if not next_step or next_step.get("id") in state["data"]:
        return
    if next_step.get("type") != "options":
        return
    if "Somewhere else" not in (next_step.get("opts") or []):
        return
    step_text = _norm_text(next_step.get("text", ""))
    if "where" not in step_text and "location" not in step_text:
        return

    raw_answers = state.get("raw_answers", {})
    for step_id, raw_text in reversed(list(raw_answers.items())):
        if step_id == next_step.get("id"):
            continue
        text = str(raw_text or "").strip()
        if not _looks_like_body_location_phrase(text):
            continue
        interpreted = interpret_user_input_with_options(
            next_step,
            text,
            topic_history=_recent_topic_history(state),
        )
        if interpreted not in next_step.get("opts", []):
            continue
        coerced = _coerce_structured_answer(
            topic_key,
            next_step,
            interpreted,
            state["data"],
            raw_answer=text,
        )
        state["data"][next_step["id"]] = coerced
        state["raw_answers"][next_step["id"]] = text
        break


def _resolve_next_step(topic_key: str, state: dict) -> Optional[dict]:
    next_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
    _backfill_next_step_from_topic_history(topic_key, state, next_step)
    return get_next_step(topic_key, state["data"], state.get("raw_answers"))


def _suggestions_for_prompt_text(prompt_text: str, step: Optional[dict] = None) -> list[str]:
    if step and step.get("opts"):
        opts = []
        for opt in step.get("opts", []):
            text = str(opt or "").strip()
            if not text:
                continue
            if text.lower() in {"other", "something else"}:
                continue
            opts.append(text)
        if opts:
            return opts[:5]

    text = _norm_text(prompt_text or "")
    if not text:
        return []
    if any(word in text for word in ("when", "start", "started", "how long", "since")):
        return ["Today", "Yesterday", "A few days ago", "More than a week ago", "Not sure"]
    if "dose" in text or "how often" in text:
        return ["As prescribed", "Once a day", "Twice a day", "Three times a day", "Not sure"]
    if "where exactly" in text or "where is the pain" in text or "which body part" in text:
        return ["Throat", "Tongue", "Jaw", "Neck", "Mouth/cheek"]
    if "what are you able to eat" in text or "what can you eat" in text:
        return ["Soft foods", "Liquids", "Small meals", "Not sure"]
    if "who is supporting you" in text or "who supports you" in text:
        return ["Family", "Friends", "Caregiver", "No one nearby"]
    if "what kind of support" in text:
        return ["Emotional support", "Transportation help", "Help at home", "More information"]
    return []


def _quick_reply_suggestions(topic_key: str, state: dict, step: dict) -> list[str]:
    if step.get("opts"):
        return []
    if step.get("type") not in {"free_text", "number"}:
        return []

    explicit = step.get("suggestions")
    cleaned = []
    if isinstance(explicit, (list, tuple)):
        for item in explicit:
            text = str(item or "").strip()
            if text:
                cleaned.append(text)
    if cleaned:
        return cleaned
    return _suggestions_for_prompt_text(step.get("text", ""), step=step)


def _render_suggested_reply_buttons(
    suggestions: list[str],
    key_prefix: str,
    target_input_key: Optional[str] = None,
) -> Optional[str]:
    suggestions = [str(s or "").strip() for s in suggestions if str(s or "").strip()]
    if not suggestions:
        return None
    clicked = None
    st.markdown('<div class="common-answer-buttons">', unsafe_allow_html=True)
    cols = st.columns(len(suggestions))
    for idx, suggestion in enumerate(suggestions):
        with cols[idx]:
            if st.button(suggestion, key=f"{key_prefix}_{idx}", use_container_width=True):
                if target_input_key:
                    st.session_state[target_input_key] = suggestion
                clicked = suggestion
    st.markdown('</div>', unsafe_allow_html=True)
    return clicked


def _render_numeric_choice_buttons(
    values: list[int],
    key_prefix: str,
) -> Optional[int]:
    clean_values = [int(v) for v in values]
    if not clean_values:
        return None
    clicked = None
    st.markdown('<div class="common-answer-buttons">', unsafe_allow_html=True)
    cols = st.columns(len(clean_values))
    for idx, value in enumerate(clean_values):
        with cols[idx]:
            if st.button(str(value), key=f"{key_prefix}_{idx}", use_container_width=True):
                clicked = value
    st.markdown('</div>', unsafe_allow_html=True)
    return clicked


def _mark_submission_once(submitted_key: str, candidate: str) -> bool:
    if not candidate or st.session_state.get(submitted_key) == candidate:
        return False
    st.session_state[submitted_key] = candidate
    return True


def _looks_like_engagement_strain_signal(text: str) -> bool:
    normalized = _norm_text(text)
    if not normalized:
        return False
    explicit_control = (
        "stop", "pause", "skip", "move on", "enough", "done",
        "cant do this", "can't do this", "no more", "leave it",
        "leave me alone", "go away", "forget it", "nevermind", "never mind",
    )
    if any(signal in normalized for signal in explicit_control):
        return True

    uncertainty = (
        "not sure", "dont know", "don't know", "i dont know", "i don't know",
        "idk", "maybe", "whatever", "i guess", "not really", "hard to say",
    )
    if any(signal in normalized for signal in uncertainty):
        return True

    # In the retry/unmatched-option path, very short non-specific replies often
    # reflect frustration, confusion, or a wish to stop narrowing the same point.
    token_count = len(normalized.split())
    if token_count <= 2 and normalized not in {"yes", "no", "left", "right", "both"}:
        return True

    return False


def _respect_patient_control_signal(topic_key: str, step: dict, candidate: str) -> bool:
    """
    Let the existing sentiment agent intercept replies that may reflect
    frustration, uncertainty, or a wish to stop/shorten the interaction
    before the app falls into a retry loop.
    """
    text = str(candidate or "").strip()
    if not text or not openai_client or not _looks_like_engagement_strain_signal(text):
        return False

    state = st.session_state.topic_states[topic_key]
    session_answers = _build_session_answers(topic_key)
    topic_history = _recent_topic_history(state)
    recent_questions = _recent_topic_questions(state)
    active_sentiment_signals = st.session_state.get("sentiment_state", {}).get("all_signals", [])
    question_count = len(state.get("data", {}))

    sentiment_out = run_sentiment_agent(
        step,
        text,
        session_answers,
        active_sentiment_signals,
        question_count,
        topic_history=topic_history,
        recent_questions=recent_questions,
    )
    _merge_sentiment_state(sentiment_out)

    signals = sentiment_out.get("signals", {})
    if signals.get("E7_wants_to_stop"):
        state["chat"].append({"role": "user", "content": text})
        acknowledgment = sentiment_out.get("adaptation", {}).get("acknowledgment_text")
        closing = "Of course — we'll pause here. The answers you've shared have been saved for your care team."
        if acknowledgment:
            closing = f"{acknowledgment}\n\n{closing}"
        state["chat"].append({"role": "assistant", "content": closing})
        state["status"] = "completed"
        st.rerun()
        return True

    return False


def _render_choice_button_grid(options: list[str], key_prefix: str) -> Optional[str]:
    if not options:
        return None
    cols_per_row = 2 if len(options) > 1 else 1
    for idx in range(0, len(options), cols_per_row):
        row = st.columns(cols_per_row)
        for offset, option in enumerate(options[idx:idx + cols_per_row]):
            with row[offset]:
                if st.button(option, key=f"{key_prefix}_{idx + offset}", use_container_width=True):
                    return option
    return None


def _process_option_submission(
    topic_key: str,
    step: dict,
    candidate: str,
    source: str,
    submitted_key: str,
    topic_history: list[dict[str, str]],
) -> bool:
    if not _mark_submission_once(submitted_key, candidate):
        return False
    interpreted = interpret_user_input_with_options(step, candidate, topic_history=topic_history)
    if interpreted in step.get("opts", []):
        handle_answer(
            topic_key,
            step,
            interpreted,
            source="structured",
            display_override=candidate,
            raw_answer=candidate,
        )
    else:
        if openai_client and source in {"typed", "voice"}:
            handle_answer(
                topic_key,
                step,
                candidate,
                source=source,
                display_override=candidate,
                raw_answer=candidate,
            )
            return True
        if _respect_patient_control_signal(topic_key, step, candidate):
            return True
        _request_retry_for_step(topic_key, step, candidate, source=source)
    return True


def _process_multiselect_submission(
    topic_key: str,
    step: dict,
    candidate: str,
    source: str,
    submitted_key: str,
) -> bool:
    if not _mark_submission_once(submitted_key, candidate):
        return False
    parsed = parse_multi_select_typed_input(step, candidate)
    if parsed:
        handle_answer(
            topic_key,
            step,
            parsed,
            source="structured",
            display_override=candidate,
            raw_answer=candidate,
        )
    else:
        if _respect_patient_control_signal(topic_key, step, candidate):
            return True
        _request_retry_for_step(topic_key, step, candidate, source=source)
    return True


def _process_number_submission(topic_key: str, step: dict, candidate: str, submitted_key: str) -> bool:
    if not _mark_submission_once(submitted_key, candidate):
        return False
    try:
        value = int(float(candidate))
    except ValueError:
        if _respect_patient_control_signal(topic_key, step, candidate):
            return True
        st.warning("Please enter a number.")
        return True
    if value < step["min_v"] or value > step["max_v"]:
        st.warning(f"Please enter a value between {int(step['min_v'])} and {int(step['max_v'])}.")
        return True
    handle_answer(topic_key, step, value, source="typed")
    return True


def _is_exact_structured_option_reply(answer: Any, raw_answer: Any, step: dict, source: str) -> bool:
    if source != "structured":
        return False
    if not isinstance(answer, str):
        return False
    if step.get("type") != "options":
        return False
    raw_text = str(raw_answer if raw_answer is not None else answer).strip()
    if not raw_text:
        return False
    return _norm_text(raw_text) == _norm_text(answer)


def _store_followup_prompt(
    topic_key: str,
    state: dict,
    step: dict,
    question: str,
    assistant_message: str = "",
    retry_current_step: bool = False,
    allow_other_detail: bool = False,
    target_step: Optional[dict] = None,
):
    state["waiting_for_followup"] = True
    prompt_step = target_step or step
    state["pending_followup"] = {
        "source_step_id": step["id"],
        "question": question,
        "answer_key": f"{step['id']}_llm_followup",
        "assistant_message": assistant_message.strip(),
        "retry_current_step": retry_current_step,
        "allow_other_detail": allow_other_detail,
        "target_step_id": target_step.get("id") if target_step else None,
    }
    combined_prompt = "\n\n".join([part for part in [assistant_message.strip(), question.strip()] if part])
    _append_assistant_message(state, combined_prompt, prompt_step=prompt_step, prompt_text=combined_prompt)
    _remember_prompted_step(state, prompt_step, combined_prompt)


def _request_retry_for_step(topic_key: str, step: dict, raw_input: str, source: str = "typed"):
    state = st.session_state.topic_states[topic_key]
    text = (raw_input or "").strip()
    if text:
        state["chat"].append({"role": "user", "content": text})
    retry_question = _build_retry_prompt(
        step,
        text,
        topic_history=_recent_topic_history(state),
        recent_questions=_recent_topic_questions(state),
    )
    _store_followup_prompt(
        topic_key,
        state,
        step,
        retry_question,
        retry_current_step=True,
        allow_other_detail=("Other" in step.get("opts", [])),
    )
    if state.get("pending_followup") is not None:
        state["pending_followup"]["raw_input"] = text
    st.rerun()


def _clear_step_inputs(topic_key: str, step: dict):
    sid = step["id"]
    stype = step["type"]

    keys_to_clear = []
    if stype == "options":
        keys_to_clear.extend([
            f"text_{topic_key}_{sid}",
            f"text_{topic_key}_{sid}_submitted",
            f"radio_{topic_key}_{sid}",
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
            f"multi_{topic_key}_{sid}",
            f"dropdown_{topic_key}_{sid}",
            f"dropdown_{topic_key}_{sid}_submitted",
            f"voice_{topic_key}_{sid}_submitted",
            f"_vt_{topic_key}_{sid}_multi",
            f"_vh_{topic_key}_{sid}_multi",
        ])
        for idx, _ in enumerate(step.get("opts", [])):
            keys_to_clear.append(f"multi_{topic_key}_{sid}_{idx}")
    elif stype == "number":
        keys_to_clear.extend([
            f"text_{topic_key}_{sid}",
            f"text_{topic_key}_{sid}_submitted",
            f"num_{topic_key}_{sid}",
            f"num_text_{topic_key}_{sid}",
            f"suggested_{topic_key}_{sid}",
            f"suggested_{topic_key}_{sid}_submitted",
            f"_vt_{topic_key}_{sid}_num",
            f"_vh_{topic_key}_{sid}_num",
        ])
    elif stype == "free_text":
        keys_to_clear.extend([
            f"ft_{topic_key}_{sid}",
            f"ft_{topic_key}_{sid}_submitted",
            f"suggested_{topic_key}_{sid}",
            f"suggested_{topic_key}_{sid}_submitted",
            f"ft_{topic_key}_{sid}_voice_sync",
            f"_vt_{topic_key}_{sid}",
            f"_vh_{topic_key}_{sid}",
        ])

    for key in keys_to_clear:
        st.session_state.pop(key, None)


def _clear_pending_followup_inputs(topic_key: str, pending_suffix: str):
    keys_to_clear = [
        f"pending_followup_{topic_key}_{pending_suffix}",
        f"pending_followup_{topic_key}_{pending_suffix}_submitted",
        f"pending_followup_{topic_key}_{pending_suffix}_voice_sync",
        f"_vt_pending_{topic_key}_{pending_suffix}",
        f"_vh_pending_{topic_key}_{pending_suffix}",
        f"_vrec_pending_{topic_key}_{pending_suffix}",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def handle_pending_followup(topic_key: str, answer: str, source: str = "typed"):
    state = st.session_state.topic_states[topic_key]
    _invalidate_report_cache()
    pending = state.get("pending_followup") or {}
    answer_key = pending.get("answer_key")
    pending_suffix = pending.get("answer_key", "pending")
    if not answer_key:
        state["waiting_for_followup"] = False
        state.pop("pending_followup", None)
        _clear_current_prompt_flags(state)
        _clear_pending_followup_inputs(topic_key, pending_suffix)
        st.rerun()
        return

    if pending.get("retry_current_step"):
        source_step_id = pending.get("source_step_id")
        source_step = STEP_BY_ID.get(source_step_id)
        state["waiting_for_followup"] = False
        state.pop("pending_followup", None)
        _clear_current_prompt_flags(state)
        if not source_step:
            st.rerun()
            return

        retry_text = (answer or "").strip()
        if source_step["type"] == "options":
            previous_raw = _norm_text(pending.get("raw_input", ""))
            retry_norm = _norm_text(retry_text)
            interpreted = None
            if (
                source_step.get("id") == "sore_location"
                and retry_norm in {"yes", "yeah", "yep", "correct"}
                and previous_raw in {"chick", "cheek"}
                and "Inside the mouth/cheek" in source_step.get("opts", [])
            ):
                interpreted = "Inside the mouth/cheek"
            if interpreted is None:
                interpreted = interpret_user_input_with_options(source_step, retry_text, topic_history=_recent_topic_history(state))
            if interpreted in source_step.get("opts", []):
                handle_answer(
                    topic_key,
                    source_step,
                    interpreted,
                    source="structured",
                    raw_answer=retry_text,
                    display_override=retry_text,
                )
                return
            if _respect_patient_control_signal(topic_key, source_step, retry_text):
                return
            _request_retry_for_step(topic_key, source_step, retry_text, source=source)
            return

        if source_step["type"] == "multi_select":
            parsed = parse_multi_select_typed_input(source_step, retry_text)
            if parsed:
                handle_answer(
                    topic_key,
                    source_step,
                    parsed,
                    source="structured",
                    raw_answer=retry_text,
                    display_override=retry_text,
                )
                return
            if pending.get("allow_other_detail") and retry_text:
                state["data"][f"{source_step['id']}_other_detail"] = retry_text
                handle_answer(
                    topic_key,
                    source_step,
                    ["Other"],
                    source="structured",
                    display_override=retry_text,
                    raw_answer=retry_text,
                )
                return
            _request_retry_for_step(topic_key, source_step, retry_text, source=source)
            return

    target_step_id = pending.get("target_step_id")
    target_step = STEP_BY_ID.get(target_step_id) if target_step_id else None
    if target_step:
        state["waiting_for_followup"] = False
        state.pop("pending_followup", None)

        followup_text = (answer or "").strip()
        if target_step["type"] == "options":
            interpreted = interpret_user_input_with_options(
                target_step,
                followup_text,
                topic_history=_recent_topic_history(state),
            )
            if interpreted in target_step.get("opts", []):
                handle_answer(
                    topic_key,
                    target_step,
                    interpreted,
                    source="structured",
                    raw_answer=followup_text,
                    display_override=followup_text,
                )
                return
            if _respect_patient_control_signal(topic_key, target_step, followup_text):
                return
            _request_retry_for_step(topic_key, target_step, followup_text, source=source)
            return

        if target_step["type"] == "multi_select":
            parsed = parse_multi_select_typed_input(target_step, followup_text)
            if parsed:
                handle_answer(
                    topic_key,
                    target_step,
                    parsed,
                    source="structured",
                    raw_answer=followup_text,
                    display_override=followup_text,
                )
                return
            if "Other" in target_step.get("opts", []) and followup_text:
                state["data"][f"{target_step['id']}_other_detail"] = followup_text
                handle_answer(
                    topic_key,
                    target_step,
                    ["Other"],
                    source="structured",
                    display_override=followup_text,
                    raw_answer=followup_text,
                )
                return
            _request_retry_for_step(topic_key, target_step, followup_text, source=source)
            return

        if target_step["type"] == "number":
            try:
                numeric_value = int(float(followup_text))
            except (TypeError, ValueError):
                if _respect_patient_control_signal(topic_key, target_step, followup_text):
                    return
                _request_retry_for_step(topic_key, target_step, followup_text, source=source)
                return
            handle_answer(
                topic_key,
                target_step,
                numeric_value,
                source="structured",
                display_override=followup_text,
                raw_answer=followup_text,
            )
            return

        handle_answer(
            topic_key,
            target_step,
            followup_text,
            source="free_text",
            raw_answer=followup_text,
            display_override=followup_text,
        )
        return

    state["chat"].append({"role": "user", "content": answer})
    state["data"][answer_key] = answer
    _record_response_metadata(
        topic_key,
        {
            "id": answer_key,
            "text": pending.get("question", ""),
            "type": "free_text",
        },
        answer,
        source,
        answer,
        str(answer),
    )
    pending_key = f"pending_followup_{topic_key}_{pending.get('answer_key', 'pending')}"
    _clear_pending_followup_inputs(topic_key, pending_suffix)
    state["waiting_for_followup"] = False
    state.pop("pending_followup", None)
    _clear_current_prompt_flags(state)

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
    _invalidate_report_cache()

    # ── Ensure followup_counts dict exists (backward compat) ──────
    if "followup_counts" not in state:
        state["followup_counts"] = {}
    if "raw_answers" not in state:
        state["raw_answers"] = {}
    if "last_prompted_step_id" not in state:
        state["last_prompted_step_id"] = None
    if "last_prompted_text" not in state:
        state["last_prompted_text"] = ""
    if "generated_prompts" not in state:
        state["generated_prompts"] = {}
    if "generated_quick_replies" not in state:
        state["generated_quick_replies"] = {}
    if state.get("last_prompted_step_id") == step.get("id"):
        _clear_current_prompt_flags(state)
        _remember_prompted_step(state, None, "")
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
    _record_response_metadata(topic_key, step, answer, source, verbatim, display)
    _capture_rich_answer_into_next_step(topic_key, state, step, answer, verbatim)
    if topic_key == "pain" and step.get("id") == "pain_medications":
        meds = answer if isinstance(answer, list) else [answer]
        if "No pain medication" in meds:
            for stale_id in ("med_dose_freq", "taking_as_prescribed", "med_adherence_issue", "med_side_effects"):
                state["data"].pop(stale_id, None)
                state["raw_answers"].pop(stale_id, None)
    if step.get("id") == "other_pain_desc":
        state["data"]["other_pain_head_neck_focused"] = _deterministic_head_neck_focus(
            verbatim if isinstance(verbatim, str) else str(answer)
        )
    if isinstance(verbatim, str) and not openai_client:
        _auto_capture_following_answers(topic_key, state, verbatim)
    next_step = _resolve_next_step(topic_key, state)
    state["status"] = "in_progress"

    last_topic_data = st.session_state.last_checkin.get(topic_key, {})

    if (
        _is_exact_structured_option_reply(answer, verbatim, step, source)
        and not ENABLE_FULL_PIPELINE_FOR_EXACT_STRUCTURED_OPTIONS
    ):
        _record_fastpath_trace(
            topic_key,
            step,
            answer,
            source,
            "Exact structured option matched. Agents skipped and the app moved to the next regular step.",
        )
        if topic_is_complete(topic_key, state["data"], state.get("raw_answers")):
            _finalize_demo_trace("Topic complete.")
            state["status"] = "completed"
            state["chat"].append({
                "role": "assistant",
                "content": "✅ Thank you — I have everything I need for this topic.",
            })
            st.rerun()
            return
        _finalize_demo_trace(
            "Move to the next regular question.",
            _step_prompt_text(next_step, topic_key=topic_key, state=state) if next_step else None,
        )
        _append_next_question(topic_key, state, next_step)
        st.rerun()
        return

    # ══════════════════════════════════════════════════════════════
    # BRANCH A — Structured non-string answers (fast path)
    # Lists/numbers do not benefit much from the language pipeline.
    # String answers, including button/option replies like "Yes" or "No",
    # still go through the agents so the app can skip irrelevant follow-ups.
    # ══════════════════════════════════════════════════════════════
    if source == "structured" and not isinstance(answer, str):
        _record_fastpath_trace(
            topic_key,
            step,
            answer,
            source,
            "Structured numeric or multi-select answer accepted. Agents skipped and the app moved forward.",
        )
        if topic_is_complete(topic_key, state["data"], state.get("raw_answers")):
            _finalize_demo_trace("Topic complete.")
            state["status"] = "completed"
            state["chat"].append({
                "role": "assistant",
                "content": "✅ Thank you — I have everything I need for this topic.",
            })
            st.rerun()
            return
        _finalize_demo_trace(
            "Move to the next regular question.",
            _step_prompt_text(next_step, topic_key=topic_key, state=state) if next_step else None,
        )
        _append_next_question(topic_key, state, next_step)
        st.rerun()
        return

    # ══════════════════════════════════════════════════════════════
    # BRANCH B — String answers — run full agent pipeline
    # ══════════════════════════════════════════════════════════════
    if isinstance(answer, str):
        if source in {"typed", "voice", "free_text"} and not openai_client and not answer.strip():
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
                    raw_answer=verbatim if isinstance(verbatim, str) else str(verbatim),
                    state=state,
                    last_topic_data=last_topic_data,
                )
                pipeline["source"] = source
                _record_agent_trace(topic_key, step, pipeline)

                matched_option = pipeline.get("matched_option")
                if step.get("type") == "options" and matched_option in step.get("opts", []):
                    answer = matched_option
                    state["data"][step["id"]] = matched_option
                    if st.session_state.get("structured_responses"):
                        st.session_state["structured_responses"][-1]["answer"] = matched_option
                    next_step = _resolve_next_step(topic_key, state)

            # ── Emergency: terminate session ──────────────────────
            if pipeline.get("urgency_tier", 0) == 3:
                _finalize_demo_trace("Stop the session and show emergency guidance.")
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
                _finalize_demo_trace("Pause here because the patient appears to want to stop.")
                closing = "Of course — we'll pause here. The answers you've shared have been saved for your care team."
                if ack:
                    closing = f"{ack}\n\n{closing}"
                state["chat"].append({"role": "assistant", "content": closing})
                state["status"] = "completed"
                st.rerun()
                return

            # ── Tier 2: avoid detached follow-ups in the same turn ─
            if tier2_msg:
                _finalize_demo_trace("Pause this topic so the care team can follow up directly.")
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
                    next_step_action = pipeline.get("next_step_action")
                    if next_step_action:
                        _apply_agent_next_step_action(topic_key, state, next_step_action)
                        next_step = _resolve_next_step(topic_key, state)
                    # Increment follow-up counter
                    fc = state["followup_counts"]
                    fc[step["id"]] = fc.get(step["id"], 0) + 1
                    _store_followup_prompt(
                        topic_key, state, step, fq, ack,
                        target_step=next_step,
                    )
                    _finalize_demo_trace("Ask one follow-up question.", fq)
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

            _apply_agent_next_step_action(topic_key, state, pipeline.get("next_step_action"))
            next_step = _resolve_next_step(topic_key, state)

        else:
            # No OpenAI — use fallback reply
            _record_fastpath_trace(
                topic_key,
                step,
                answer,
                source,
                "OpenAI unavailable. Used fallback handling without live agent calls.",
            )
            assistant_message = _default_chatty_reply(
                topic_key, answer, step, last_topic_data
            )
            _apply_generic_fallback_next_step_action(topic_key, state)
            next_step = _resolve_next_step(topic_key, state)

    else:
        # Non-string answer (numeric, list from multi_select on structured path)
        assistant_message = ""

    # ── Topic complete check ──────────────────────────────────────
    if topic_is_complete(topic_key, state["data"], state.get("raw_answers")):
        _finalize_demo_trace("Topic complete.")
        state["status"] = "completed"
        final_message = "✅ Thank you — I have everything I need for this topic."
        if assistant_message:
            final_message = f"{assistant_message}\n\n{final_message}"
        state["chat"].append({"role": "assistant", "content": final_message})
        st.rerun()
        return

    _finalize_demo_trace(
        "Move to the next regular question.",
        _step_prompt_text(next_step, topic_key=topic_key, state=state) if next_step else None,
    )
    _append_next_question(topic_key, state, next_step, assistant_message)
    st.rerun()
    return


# ══════════════════════════════════════════════════════════════════
# INPUT RENDERING
# ══════════════════════════════════════════════════════════════════


def render_input(topic_key: str, step: dict):
    """Render the appropriate input widget for the current question."""
    stype = step["type"]
    sid   = step["id"]

    state = st.session_state.topic_states[topic_key]
    topic_history = _recent_topic_history(state)
    # ── Options ─────────────────────────────────────────────────
    if stype == "options":
        st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
        opts = step.get("opts", [])
        form_key = f"form_{topic_key}_{sid}_options"
        with st.form(form_key, clear_on_submit=False):
            selected = st.radio("Choose one", opts, key=f"radio_{topic_key}_{sid}", horizontal=(len(opts) <= 3))
            text_col, mic_col = st.columns([20, 1], vertical_alignment="bottom")
            with text_col:
                typed = st.text_input(
                    "Add details or type a different answer",
                    key=f"text_{topic_key}_{sid}",
                    placeholder="Optional details...",
                )
            with mic_col:
                st.markdown('<div class="inline-voice-row">', unsafe_allow_html=True)
                voice_text = voice_widget(f"{topic_key}_{sid}_opt", label="🎙️")
                st.markdown('</div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Continue", type="primary", use_container_width=True)
        if submitted:
            typed_clean = (typed or "").strip()
            if typed_clean:
                st.session_state.pop(f"text_{topic_key}_{sid}_submitted", None)
                if _process_option_submission(topic_key, step, typed_clean, "typed", f"text_{topic_key}_{sid}_submitted", topic_history):
                    return
            else:
                handle_answer(topic_key, step, selected, source="structured")
                return
        if _process_option_submission(topic_key, step, voice_text, "voice", f"voice_{topic_key}_{sid}_submitted", topic_history):
            return
        st.markdown('</div>', unsafe_allow_html=True)
                

    # ── Multi-select ─────────────────────────────────────────────
    elif stype == "multi_select":
        st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
        opts = step.get("opts", [])
        form_key = f"form_{topic_key}_{sid}_multi"
        with st.form(form_key, clear_on_submit=False):
            st.markdown("**Choose all that apply**")
            selected = []
            checkbox_cols = st.columns(2)
            for idx, opt in enumerate(opts):
                with checkbox_cols[idx % 2]:
                    if st.checkbox(opt, key=f"multi_{topic_key}_{sid}_{idx}"):
                        selected.append(opt)
            text_col, mic_col = st.columns([20, 1], vertical_alignment="bottom")
            with text_col:
                typed = st.text_input(
                    "Other or details",
                    key=f"text_{topic_key}_{sid}",
                    placeholder="Optional: type another medication or detail...",
                )
            with mic_col:
                st.markdown('<div class="inline-voice-row">', unsafe_allow_html=True)
                voice_text = voice_widget(f"{topic_key}_{sid}_multi", label="🎙️")
                st.markdown('</div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Continue", type="primary", use_container_width=True)
        if submitted:
            typed_clean = (typed or "").strip()
            if typed_clean:
                payload = list(selected)
                if "No pain medication" in payload and typed_clean:
                    payload = [item for item in payload if item != "No pain medication"]
                if "None of these" in payload and typed_clean:
                    payload = [item for item in payload if item != "None of these"]
                if "Other" not in payload:
                    payload.append("Other")
                display_parts = [item for item in payload if item != "Other"] + [typed_clean]
                display_value = ", ".join(display_parts)
                handle_answer(
                    topic_key,
                    step,
                    payload,
                    source="structured",
                    display_override=display_value,
                    raw_answer=typed_clean,
                )
                return
            if selected:
                handle_answer(topic_key, step, selected, source="structured")
                return
            if typed_clean:
                st.session_state.pop(f"text_{topic_key}_{sid}_submitted", None)
            if typed_clean and _process_multiselect_submission(topic_key, step, typed_clean, "typed", f"text_{topic_key}_{sid}_submitted"):
                return
            st.warning("Please choose at least one option, or type an answer.")
        if _process_multiselect_submission(topic_key, step, voice_text, "voice", f"voice_{topic_key}_{sid}_submitted"):
            return
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Number ───────────────────────────────────────────────────
    elif stype == "number":
        st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
        min_v = int(step.get("min_v", 0))
        max_v = int(step.get("max_v", 10))
        default_v = int(step.get("default_v", min_v))
        is_weight_step = topic_key == "nutrition" and sid == "weight"
        direct_button_values = list(range(min_v, max_v + 1)) if (max_v - min_v) <= 10 and not is_weight_step else []
        unit = "lbs"

        if direct_button_values:
            clicked_value = _render_numeric_choice_buttons(
                direct_button_values,
                key_prefix=f"num_btn_{topic_key}_{sid}",
            )
            if clicked_value is not None:
                handle_answer(topic_key, step, clicked_value, source="structured")
                return

        if is_weight_step:
            unit_key = f"unit_{topic_key}_{sid}"
            if unit_key not in st.session_state:
                st.session_state[unit_key] = "lbs"
            unit = st.radio(
                "Weight unit",
                ["lbs", "kg"],
                key=unit_key,
                horizontal=True,
            )
            quick_values = [100, 120, 140, 160, 180, 200, 220] if unit == "lbs" else [45, 55, 65, 75, 85, 95, 105]
            clicked_weight = _render_numeric_choice_buttons(
                quick_values,
                key_prefix=f"weight_btn_{topic_key}_{sid}_{unit}",
            )
            if clicked_weight is not None:
                final_value = clicked_weight if unit == "lbs" else round(clicked_weight * 2.20462, 1)
                handle_answer(
                    topic_key,
                    step,
                    final_value,
                    source="structured",
                    display_override=f"{clicked_weight} {unit}",
                )
                return

        with st.form(f"form_{topic_key}_{sid}_number", clear_on_submit=False):
            input_col, mic_col = st.columns([20, 1], vertical_alignment="center")
            with input_col:
                input_min = min_v
                input_max = max_v
                input_default = max(min(default_v, input_max), input_min)
                input_label = f"Enter a number from {min_v} to {max_v}"
                if is_weight_step and unit == "kg":
                    input_min = max(20, round(min_v / 2.20462))
                    input_max = round(max_v / 2.20462)
                    input_default = max(min(round(default_v / 2.20462), input_max), input_min)
                    input_label = f"Enter your weight in kilograms from {input_min} to {input_max}"
                elif is_weight_step:
                    input_label = f"Enter your weight in pounds from {input_min} to {input_max}"
                value_text = st.text_input(
                    input_label,
                    key=f"num_text_{topic_key}_{sid}",
                    value=str(input_default),
                )
            with mic_col:
                st.markdown('<div class="inline-voice-row">', unsafe_allow_html=True)
                voice_text = voice_widget(f"{topic_key}_{sid}_num", label="🎙️")
                st.markdown('</div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Continue", type="primary", use_container_width=True)
        if submitted:
            typed_value = (value_text or "").strip()
            st.session_state.pop(f"text_{topic_key}_{sid}_submitted", None)
            try:
                numeric_value = float(typed_value)
            except (TypeError, ValueError):
                st.warning("Please enter a valid number.")
            else:
                final_value = numeric_value
                display_value = typed_value
                if is_weight_step and unit == "kg":
                    final_value = round(numeric_value * 2.20462, 1)
                    display_value = f"{typed_value} kg"
                elif is_weight_step:
                    display_value = f"{typed_value} lbs"
                if float(final_value) < min_v or float(final_value) > max_v:
                    st.warning(f"Please enter a value between {min_v} and {max_v}.")
                else:
                    int_or_float = int(final_value) if float(final_value).is_integer() else round(float(final_value), 1)
                    handle_answer(topic_key, step, int_or_float, source="structured", display_override=display_value)
                    return
        if _process_number_submission(topic_key, step, voice_text or "", f"voice_{topic_key}_{sid}_submitted"):
            return
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
        suggestions = _quick_reply_suggestions(topic_key, state, step)
        clicked_suggestion = None
        if suggestions:
            clicked_suggestion = _render_suggested_reply_buttons(
                suggestions,
                key_prefix=f"suggested_btn_{topic_key}_{sid}",
                target_input_key=widget_key,
            )
        if clicked_suggestion:
            st.session_state[submit_key] = clicked_suggestion
            handle_answer(topic_key, step, clicked_suggestion, source="free_text")
            return
        with st.form(f"form_{topic_key}_{sid}_free", clear_on_submit=False):
            text_col, mic_col = st.columns([20, 1], vertical_alignment="bottom")
            with text_col:
                free_text = st.text_input(
                    "Your answer",
                    placeholder=step.get("placeholder", "Please describe..."),
                    key=widget_key,
                )
            with mic_col:
                st.markdown('<div class="inline-voice-row">', unsafe_allow_html=True)
                voice_text = voice_widget(f"{topic_key}_{sid}", label="🎙️")
                st.markdown('</div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Continue", type="primary", use_container_width=True)
        if voice_text and voice_text != st.session_state.get(f"{widget_key}_voice_sync"):
            st.session_state[f"{widget_key}_voice_sync"] = voice_text
            st.session_state[submit_key] = voice_text
            handle_answer(topic_key, step, voice_text, source="voice")
            return

        if submitted:
            candidate = (free_text or "").strip()
            if not candidate:
                st.warning("Please type an answer or choose a common answer.")
            else:
                st.session_state[submit_key] = candidate
                handle_answer(topic_key, step, candidate, source="free_text")

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
        '      <div class="chat-shell-name">Anything else you’d like to share?</div>'
        '    </div>'
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
            _invalidate_report_cache()
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
    if "last_prompted_step_id" not in state:
        state["last_prompted_step_id"] = None
    if "last_prompted_text" not in state:
        state["last_prompted_text"] = ""

    # ── Urgency banner (Tier 1–3 from multi-agent system) ───────────
    render_urgency_banner()

    # ── Previous check-in summary content ────────────────────────
    chips_html = _checkin_summary_html(topic_key, last_data) if (has_prev and last_data) else ""
    inline_summary_html = ""
    if has_prev:
        summary_inner = (
            '<div class="chat-shell-inline-summary-body">'
            '<div style="font-size:12px;color:#5f7386;line-height:1.5;margin-bottom:6px;">'
            'These answers are from your last visit. You can change any of them for this visit.'
            '</div>'
            + (chips_html if chips_html else '<div style="font-size:12px;color:#6d7f90;">No prior summary recorded for this topic.</div>')
            + '</div>'
        )
        inline_summary_html = (
            '<div class="chat-shell-inline-summary">'
            '<details>'
            '<summary>Last visit summary</summary>'
            f'{summary_inner}'
            '</details>'
            '</div>'
        )

    # ── Initialize topic on first visit ─────────────────────────
    if state["status"] == "not_started":
        state["status"] = "in_progress"
        intro = TOPIC_INTROS.get(topic_key, "Let's go through this section together.")
        state["chat"] = [{"role": "assistant", "content": intro}]
        first_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
        if first_step:
            _ensure_step_prompted(topic_key, state, first_step)

    st.markdown('<div class="topic-toolbar"></div>', unsafe_allow_html=True)
    if st.button("I’m getting tired / need to stop soon", key=f"fatigue_topic_{topic_key}", use_container_width=False):
        _mark_patient_fatigue(topic_key)
        st.rerun()

    # ── Header with progress bar ─────────────────────────────────
    answered, applicable = get_topic_progress(topic_key, state.get("data", {}), state.get("raw_answers"))
    _, _, completed_topics = _overall_progress()
    st.caption(
        f"Current topic: {answered}/{applicable or 1} questions answered · "
        f"Overall: {completed_topics}/{len(TOPICS)} topics complete"
    )

    current_step = None
    if state["status"] != "completed" and not state.get("waiting_for_followup"):
        current_step = get_next_step(topic_key, state["data"], state.get("raw_answers"))
        if current_step:
            _ensure_step_prompted(topic_key, state, current_step)

    header_html = (
        '<div class="chat-shell">'
        '  <div class="chat-shell-header">'
        '    <div class="chat-shell-title">'
        f'      <div class="chat-shell-name">{_html.escape(topic_label)}</div>'
        '    </div>'
        f'    {inline_summary_html}'
        '  </div>'
        '  <div class="chat-history">'
    )
    st.markdown(
        header_html,
        unsafe_allow_html=True,
    )

    # ── Chat history ─────────────────────────────────────────────
    if state["chat"]:
        with st.container(border=False):
            for msg in state["chat"]:
                highlight = bool(msg.get("role") == "assistant" and msg.get("is_current_prompt"))
                render_chat_bubble(msg["role"], msg["content"], highlight=highlight)
    st.markdown('</div></div>', unsafe_allow_html=True)

    main_col, demo_col = st.columns([3.3, 1.15], gap="large")
    if st.session_state.get("demo_mode"):
        with demo_col:
            _render_demo_agent_panel(topic_key)

    # ── Completed ────────────────────────────────────────────────
    with main_col:
        st.markdown('<div class="topic-response-region">', unsafe_allow_html=True)
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
            st.markdown('</div>', unsafe_allow_html=True)
            return

        # ── Current question ─────────────────────────────────────────
        if state.get("waiting_for_followup"):
            pending = state.get("pending_followup") or {}
            pending_suffix = pending.get("answer_key", "pending")
            pending_key = f"pending_followup_{topic_key}_{pending_suffix}"
            pending_submit_key = f"{pending_key}_submitted"
            pending_target_step = None
            if pending.get("target_step_id"):
                pending_target_step = STEP_BY_ID.get(pending.get("target_step_id"))
            elif pending.get("source_step_id"):
                pending_target_step = STEP_BY_ID.get(pending.get("source_step_id"))
            pending_suggestions = _suggestions_for_prompt_text(
                pending.get("question", ""),
                step=pending_target_step,
            )
            if pending_key not in st.session_state:
                st.session_state[pending_key] = ""
            st.markdown('<div class="composer-shell compact">', unsafe_allow_html=True)
            clicked_suggestion = None
            if pending_suggestions:
                clicked_suggestion = _render_suggested_reply_buttons(
                    pending_suggestions,
                    key_prefix=f"pending_suggested_btn_{topic_key}_{pending_suffix}",
                    target_input_key=pending_key,
                )
            if clicked_suggestion:
                st.session_state[pending_submit_key] = clicked_suggestion
                handle_pending_followup(topic_key, clicked_suggestion, source="followup")
                return
            with st.form(f"form_pending_{topic_key}_{pending_suffix}", clear_on_submit=False):
                text_col, mic_col = st.columns([20, 1], vertical_alignment="bottom")
                with text_col:
                    pending_text = st.text_input(
                        "Reply",
                        key=pending_key,
                        placeholder="Type or speak your answer here...",
                    )
                with mic_col:
                    st.markdown('<div class="inline-voice-row">', unsafe_allow_html=True)
                    pending_voice = voice_widget(f"pending_{topic_key}_{pending_suffix}", label="🎙️")
                    st.markdown('</div>', unsafe_allow_html=True)
                pending_submitted = st.form_submit_button("Continue", type="primary", use_container_width=True)
            if pending_voice and pending_voice != st.session_state.get(f"{pending_key}_voice_sync"):
                st.session_state[f"{pending_key}_voice_sync"] = pending_voice
                st.session_state[pending_submit_key] = pending_voice
                handle_pending_followup(topic_key, pending_voice, source="voice")
                return

            if pending_submitted:
                candidate = (pending_text or "").strip()
                if candidate:
                    st.session_state[pending_submit_key] = candidate
                    handle_pending_followup(topic_key, candidate, source="followup")

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            return
        next_step = current_step or get_next_step(topic_key, state["data"], state.get("raw_answers"))
        if next_step:
            render_input(topic_key, next_step)
        st.markdown('</div>', unsafe_allow_html=True)


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
        st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

        has_prev = st.session_state.get("has_prev_checkin", False)
        last_ck  = st.session_state.get("last_checkin", {})
        current_topic = _guided_current_topic_key()

        for label, key in TOPICS:
            status = st.session_state.topic_states[key]["status"]
            icon   = {"completed": "✅", "in_progress": "🔵"}.get(status, "⚪")
            dname  = label.split(" ", 1)[1] if " " in label else label
            marker = "▶ " if current_topic == key else "   "
            answered, applicable = get_topic_progress(
                key,
                st.session_state.topic_states[key].get("data", {}),
                st.session_state.topic_states[key].get("raw_answers"),
            )
            status_text = _topic_status_label(key)
            line = f"{marker}{icon} {dname}\n   {status_text}"
            if applicable:
                line += f" · {answered}/{applicable}"
            if has_prev:
                prev_data = last_ck.get(key, {})
                if prev_data:
                    snip = _natural_summary(key, prev_data)
                    line += f"\n   Last: {snip}" if snip else "\n   Last: data recorded"
                else:
                    line += "\n   Last: no prior data"
            st.markdown(
                f'<div style="white-space:pre-wrap;border:1px solid #dde6f5;'
                f'border-radius:12px;padding:8px 10px;margin-bottom:6px;'
                f'background:{"#eef6ff" if current_topic == key else "#ffffff"};'
                f'font-size:13px;color:#26364a;">{_html.escape(line)}</div>',
                unsafe_allow_html=True,
            )

        # ── Anything else? ────────────────────────────────────────
        ff_msgs  = [m for m in st.session_state.freeform_chat if m["role"] == "user"]
        st.markdown(
            f'<div style="font-size:12px;color:#6b7280;margin-top:8px;">'
            f'Anything else notes saved: <strong>{len(ff_msgs)}</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.session_state["demo_mode"] = st.checkbox(
            "Show decision trace (Demo mode)",
            value=bool(st.session_state.get("demo_mode", False)),
            key="demo_mode_checkbox",
        )

        # ── Submit ────────────────────────────────────────────────
        st.markdown(
            '<hr style="margin:8px 0 8px 0;border:none;border-top:1px solid #dde6f5;">',
            unsafe_allow_html=True,
        )
        all_topics_done = completed == total
        partial_allowed = bool(st.session_state.get("patient_fatigue", False))
        any_started = completed >= 1 or in_progress >= 1
        if any_started:
            submit_label = "📤 Submit Check-In" if all_topics_done else "💾 Save Partial Check-In"
            if not all_topics_done and not partial_allowed:
                st.caption("Submit unlocks after the guided topics are complete. Use the tired/stop-soon button if you need to save a partial check-in.")
            if st.button(
                submit_label,
                use_container_width=True,
                type="primary",
                key="sidebar_submit",
                disabled=not (all_topics_done or partial_allowed),
            ):
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
                    saved_ok = save_to_sheet(st.session_state.patient_name, all_data, report)
                st.session_state.report_saved = bool(saved_ok)
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
            You can answer by selecting choices, typing, or using voice. We will guide you through the check-in in the order your care team needs.
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        name = st.text_input("Please enter your name:", placeholder="First and last name…")
        if st.button("Begin Check-In →", type="primary", use_container_width=True):
            if name.strip():
                _reset_checkin_session_state()
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

    _, col, _ = st.columns([1, 2, 1])
    with col:
        if st.button("Continue to Topics →", type="primary", use_container_width=True):
            st.session_state.app_stage = "main"
            st.rerun()


def screen_main():
    render_sidebar()

    selected = _sync_guided_topic_selection()

    if not selected:
        st.markdown('<div class="card"><div style="font-size:12px;font-weight:800;color:#6b7b92;text-transform:uppercase;letter-spacing:0.08em;">Check-in complete</div><div style="font-size:28px;font-weight:800;letter-spacing:-0.03em;margin-top:6px;">You have finished the guided topics</div><div style="font-size:14px;color:#5f6f84;line-height:1.7;margin-top:8px;">You can submit the check-in from the sidebar, or add anything else below for your care team.</div></div>', unsafe_allow_html=True)
        _render_demo_agent_panel()
        render_freeform_chat()
        return

    topic_label = TOPIC_LABELS.get(selected, selected)
    render_topic_detail(topic_label, selected)


def screen_report():
    render_sidebar()

    all_data = _build_all_topic_data()
    ff_msgs  = [m for m in st.session_state.freeform_chat if m["role"] == "user"]
    if ff_msgs:
        all_data["freeform_notes"] = [m["content"] for m in ff_msgs]

    if not st.session_state.report:
        with st.spinner("Generating clinical report…"):
            st.session_state.report = generate_report(
                st.session_state.patient_name, all_data
            )
    topic_insights = _report_topic_insights(all_data)

    st.markdown('<div class="report-dashboard">', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:28px;font-weight:800;letter-spacing:-0.04em;color:#10233d;">📄 Clinical Check-In Report</div>'
        f'<div style="font-size:13px;color:#627287;line-height:1.7;margin-top:6px;"><strong>Patient:</strong> {_html.escape(st.session_state.patient_name)} &nbsp;|&nbsp; <strong>Date:</strong> {datetime.now().strftime("%B %d, %Y")}</div>',
        unsafe_allow_html=True,
    )
    _render_report_summary_banner(topic_insights)

    for row_start in range(0, len(topic_insights), 4):
        row_items = topic_insights[row_start:row_start + 4]
        cols = st.columns(len(row_items))
        for col, insight in zip(cols, row_items):
            with col:
                _render_report_topic_card(insight)
                _render_report_topic_detail(insight, all_data)

    with st.expander("Full clinical narrative report", expanded=False):
        st.markdown('<div class="report-box">', unsafe_allow_html=True)
        st.markdown(st.session_state.report)
        st.markdown('</div>', unsafe_allow_html=True)
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
                saved_ok = save_to_sheet(
                    st.session_state.patient_name,
                    all_data,
                    st.session_state.report,
                )
            st.session_state.report_saved = bool(saved_ok)
            if saved_ok:
                st.success("Saved successfully!")
            st.rerun()

    with col3:
        if st.button("📋 Copy to Clipboard (manual)"):
            st.info("Select the report text above and copy (Ctrl+C / Cmd+C).")


# ══════════════════════════════════════════════════════════════════
# MAIN DISPATCH
# ══════════════════════════════════════════════════════════════════

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
