# ── SHARED SYSTEM CONTEXT ────────────────────────────────────────
_SYSTEM_CONTEXT = (
    "You are a compassionate, clinically trained nurse at a head and neck cancer (HNC) center. "
    "You are conducting a structured symptom check-in with a patient currently receiving "
    "chemoradiation for head and neck cancer. "
    "This patient population frequently experiences: severe mucositis, dysphagia, pain, "
    "significant weight loss, fatigue, depression, and impaired communication. "
    "Many patients have low health literacy or face barriers to care. "
    "Your tone is always warm, clear, and non-alarming — even when probing for clinically "
    "urgent information. Never use medical jargon without explaining it simply. "
    "Never minimize a patient's reported symptom."
)

_TOPIC_LABELS = {
    "pain":      "Pain & Pain Medications",
    "nutrition": "Nutrition, Fluids & Weight",
    "oral":      "Oral Symptoms (mouth sores, dryness, mucus)",
    "gi":        "GI Symptoms (nausea, vomiting, constipation)",
    "fatigue":   "Fatigue & Sleep",
    "activity":  "Daily Activity & Independence",
    "mood":      "Emotional Health & Support",
    "other":     "Other Symptoms (breathing, skin, hearing, etc.)",
}

# ── RED FLAG CRITERIA (used in both clarification and report) ────
_RED_FLAGS = (
    "- Pain severity ≥ 7/10, uncontrolled or worsening despite medication\n"
    "- Blood when coughing (hemoptysis) — any amount\n"
    "- Fever ≥ 100.4°F / 38°C or chills with possible infection signs\n"
    "- Significant unintentional weight loss (> 5 lbs since last visit)\n"
    "- Complete inability to swallow liquids or take any oral intake\n"
    "- Feeding tube complications: leakage, blockage, site infection\n"
    "- Breathing difficulty at rest or worsening shortness of breath / wheezing\n"
    "- Falls or near-falls, especially with dizziness\n"
    "- Suicidal ideation or expression of wanting to harm oneself\n"
    "- Severe depression or distress that is interfering with daily functioning\n"
    "- New neurological symptoms: sudden weakness, numbness, confusion\n"
    "- No bowel movement for > 3 days with discomfort\n"
    "- Medication non-adherence affecting symptom control"
)


def get_llm_clarification(topic_key: str, step: dict,
                           answer: str, chat_history: list) -> Optional[str]:
    """
    After a patient gives a free-text answer, decide:
      (a) Is the answer clinically sufficient? → Return a warm 1-sentence acknowledgment.
      (b) Does the answer need one targeted follow-up probe? → Return that probe as a question.
      (c) Does the answer contain a RED FLAG? → Acknowledge and note urgency gently.
    
    Returns a single string (either acknowledgment or follow-up question), or None to skip.
    """
    if len(answer.strip().split()) < 3:
        return None  # Too short to warrant LLM involvement

    topic_label = _TOPIC_LABELS.get(topic_key, topic_key)

    # Build a compact conversation history string (last 6 turns max)
    recent_chat = chat_history[-6:] if len(chat_history) > 6 else chat_history
    history_str = "\n".join(
        f"{'Nurse' if m['role'] == 'assistant' else 'Patient'}: {m['content']}"
        for m in recent_chat
        if m["content"] != answer  # exclude the current answer (added separately)
    )

    prompt = (
        f"{_SYSTEM_CONTEXT}\n\n"
        f"=== CURRENT TOPIC: {topic_label} ===\n\n"
        f"=== RECENT CONVERSATION ===\n"
        f"{history_str}\n\n"
        f"Nurse asked: \"{step['text']}\"\n"
        f"Patient said: \"{answer}\"\n\n"
        f"=== RED FLAG CRITERIA TO WATCH FOR ===\n"
        f"{_RED_FLAGS}\n\n"
        f"=== YOUR TASK ===\n"
        f"Decide which ONE of these applies:\n\n"
        f"A) The patient's answer is COMPLETE and clinically sufficient for this question. "
        f"   → Write one warm, specific sentence acknowledging what they shared. "
        f"   Do NOT ask another question. Do NOT use hollow openers like 'I see' or 'Thank you'. "
        f"   Reference something specific they said.\n\n"
        f"B) The answer is INCOMPLETE or vague — a single targeted follow-up would "
        f"   capture important clinical detail not yet provided. "
        f"   → Ask exactly ONE short, plain-language follow-up question. "
        f"   It must be directly relevant to what they said and to this topic. "
        f"   Do not introduce an entirely new subject.\n\n"
        f"C) The answer contains or strongly implies a RED FLAG listed above. "
        f"   → Write a calm, caring sentence that gently acknowledges the concern "
        f"   and lets them know their care team will be notified. "
        f"   Example: 'Thank you for telling me — this is something your team will want "
        f"   to know about right away, and I've made sure it's flagged in your report.'\n\n"
        f"Return ONLY your chosen response — one sentence or one question. "
        f"No labels, no preamble, no explanation of which option you chose."
    )

    return _call_openai(prompt, max_tokens=120, temp=0.3) or None


def generate_report(name: str, all_data: dict) -> str:
    """
    Generate a structured clinical pre-visit report from all collected topic data.
    Falls back to a plain-text report if OpenAI is unavailable.
    """
    # Build per-topic data, labeling keys with readable names
    topic_blocks = {}
    for label, key in TOPICS:
        d = all_data.get(key, {})
        if d:
            topic_blocks[label] = d

    if not openai_client:
        # Plain-text fallback
        lines = [
            "CHATREPORT — PRE-VISIT CLINICAL SUMMARY",
            f"Patient: {name}",
            f"Date: {datetime.now().strftime('%B %d, %Y')}",
            "=" * 56, "",
        ]
        for label, data in topic_blocks.items():
            lines.append(f"[ {label.upper()} ]")
            for k, v in data.items():
                val = ", ".join(v) if isinstance(v, list) else str(v)
                lines.append(f"  • {k.replace('_', ' ').title()}: {val}")
            lines.append("")
        return "\n".join(lines)

    # Serialize collected data
    data_json = json.dumps(topic_blocks, indent=2)

    prompt = (
        f"{_SYSTEM_CONTEXT}\n\n"
        f"You are now generating a structured pre-visit clinical summary for a provider "
        f"(oncologist, radiation oncologist, or NP). This report will be reviewed "
        f"BEFORE the patient's appointment and must be concise, clinically precise, "
        f"and provider-ready.\n\n"
        f"=== PATIENT: {name} ===\n"
        f"=== DATE: {datetime.now().strftime('%B %d, %Y')} ===\n\n"
        f"=== PATIENT-REPORTED DATA ===\n"
        f"{data_json}\n\n"
        f"=== RED FLAGS TO SCREEN FOR ===\n"
        f"{_RED_FLAGS}\n\n"
        f"=== REPORT FORMAT INSTRUCTIONS ===\n"
        f"Use the EXACT structure below. Use bullet points within each section. "
        f"Be concise — providers will skim this in under 2 minutes. "
        f"Convert patient-language answers into accurate clinical language where appropriate. "
        f"If a topic was not completed or has no data, omit it entirely (do not write 'N/A').\n\n"
        f"---\n"
        f"CHATREPORT — PRE-VISIT CLINICAL SUMMARY\n"
        f"Patient: {name}  |  Date: {datetime.now().strftime('%B %d, %Y')}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"CLINICAL OVERVIEW (2–3 sentences)\n"
        f"[High-level summary of the patient's current status, most prominent issues, "
        f"and any notable changes since last visit]\n\n"
        f"⚠️ FLAGS FOR PROVIDER ATTENTION\n"
        f"[List ONLY items that match the red flag criteria above. "
        f"If none, write: No urgent flags identified.]\n\n"
        f"SYMPTOM DETAILS BY DOMAIN\n"
        f"[For each completed topic, use the section header followed by bullets. "
        f"Include: symptom presence/absence, severity where reported, patient's own words "
        f"in quotes where clinically meaningful, medications mentioned, "
        f"and any patient-reported management strategies.]\n\n"
        f"SUGGESTED DISCUSSION POINTS\n"
        f"[2–4 bullet points: items the provider may want to address or follow up on "
        f"based on the data — e.g., medication adjustment, referral, patient education.]\n"
        f"---\n\n"
        f"Write only the completed report. Do not include these instructions in your output. "
        f"Do not add disclaimers or notes about the AI."
    )

    return _call_openai(prompt, max_tokens=2000, temp=0.2) or \
        "Report generation failed — please check your OpenAI API configuration."
