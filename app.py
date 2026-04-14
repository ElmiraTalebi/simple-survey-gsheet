# llm_helper.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LLMGuidance:
    supportive_reply: str
    risk_level: str
    suggested_followup: Optional[str]
    answer_status: str
    scope: str
    redirect_message: str
    capture_note: str


SYSTEM_PROMPT = """
You are an oncology symptom triage assistant helping with patient check-in.
Goals:
1) Write a brief supportive acknowledgement in plain language only when useful.
2) Assess risk_level as one of: low, moderate, high, emergency.
3) Suggest one concise follow-up question if helpful.
4) Decide whether the patient answered the current question.
5) Detect whether the message is in scope for the current question.
6) If out of scope, provide a gentle redirect.
7) Capture one short note if the patient mentioned something important that is not covered.
Return strict JSON with keys:
- supportive_reply (string; can be empty if not needed)
- risk_level (low|moderate|high|emergency)
- suggested_followup (string|null)
- answer_status (answered|partial|not_answered)
- scope (on_topic|adjacent|out_of_scope)
- redirect_message (string; can be empty unless out_of_scope)
- capture_note (string; can be empty)
If emergency signs appear (e.g., severe breathing difficulty, coughing blood, inability to swallow liquids, confusion, chest pain), mark emergency.
Keep language simple and short.
""".strip()


def get_llm_guidance(
    patient_text: str,
    candidate_followups: List[str],
    current_question: str,
    question_options: Optional[List[str]] = None,
    previous_answer: str = "",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[LLMGuidance]:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        payload = {
            "current_question": current_question,
            "question_options": question_options or [],
            "previous_answer": previous_answer,
            "patient_text": patient_text,
            "candidate_followups": candidate_followups[:10],
            "rules": {
                "supportive_reply_max_sentences": 1,
                "redirect_max_sentences": 1,
                "reading_level": "simple",
            },
        }
        resp = client.responses.create(
            model=model or os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            text={"format": {"type": "json_object"}},
        )
        raw = (resp.output_text or "").strip()
        data = json.loads(raw)
        return LLMGuidance(
            supportive_reply=data.get("supportive_reply", "").strip(),
            risk_level=data.get("risk_level", "low"),
            suggested_followup=data.get("suggested_followup"),
            answer_status=data.get("answer_status", "answered"),
            scope=data.get("scope", "on_topic"),
            redirect_message=data.get("redirect_message", "").strip(),
            capture_note=data.get("capture_note", "").strip(),
        )
    except Exception:
        return None
