"""Streamlit front end for the nurse check-in chatbot.

All chatbot logic stays in app_eli.py: this file only shows app_eli.build_view(state)
with Streamlit widgets and sends each click to the same app_eli action functions the
web page (static/index.html) uses. So the check-in behaves exactly the same in both.

Run:  streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
"""

import hmac
import json
import os
from copy import deepcopy
from typing import Any, Dict

import streamlit as st

st.set_page_config(page_title="Nurse Assistant Check-In", page_icon="🩺", layout="centered")


def _streamlit_secrets() -> Dict[str, Any]:
    """Secrets from Streamlit (the Cloud app's Secrets settings, or a local
    .streamlit/secrets.toml), as plain values. Empty when none are configured."""
    try:
        return {key: (dict(value) if hasattr(value, "keys") else value) for key, value in st.secrets.items()}
    except Exception:
        return {}


_SECRETS = _streamlit_secrets()
# app_eli creates its OpenAI client when imported and reads the key from
# .streamlit/secrets.toml or the environment; on Streamlit Cloud there is no such file,
# so pass the key through the environment before importing it.
if _SECRETS.get("OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = str(_SECRETS["OPENAI_API_KEY"])

import app_eli  # noqa: E402

# The Google Sheets settings (gcp_service_account, gsheet_id) are read through
# app_eli._secret; let it find them in Streamlit's secrets too.
_secret_from_file = app_eli._secret
app_eli._secret = lambda name, default=None: (
    _SECRETS[name] if name in _SECRETS else _secret_from_file(name, default)
)

REVIEWING = "Nurse assistant is reviewing your response..."
SPINNER_TEXT = {
    "start": "Starting check-in...",
    "generate_patient_summary": "Preparing your summary...",
    "generate_summary": "Preparing doctor summary...",
    "patient_summary_continue": "Saving...",
    "add_symptom": "Adding...",
    "closing_finish": "Finishing...",
}

st.markdown(
    """
    <style>
      .red-flag-notice {border:2px solid #b91c1c;background:#fef2f2;color:#7f1d1d;
        border-radius:10px;padding:0.6rem 0.8rem;margin:0.2rem 0 0.8rem 0;font-weight:600;}
      .banner {background:#eef2ff;border-radius:8px;padding:0.45rem 0.7rem;font-size:0.85rem;
        color:#1e3a8a;margin-bottom:0.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- state & actions
def _app_state():
    if "app_state" not in st.session_state:
        st.session_state.app_state = app_eli.new_session_state()
    return st.session_state.app_state


def queue(action: str, payload: Dict[str, Any] = None, echo: str = "", reviewing: bool = False) -> None:
    """Button callback: remember the action; it runs at the end of this script run, so
    the page (with the patient's message echoed) is visible while the model works.
    echo = the patient's words to show meanwhile; reviewing = a model reply is coming."""
    st.session_state.pending = {
        "action": action, "payload": payload or {}, "echo": echo, "reviewing": reviewing or bool(echo),
    }
    st.session_state.error = ""


def run_pending() -> None:
    """Run the queued action exactly as the web app does (app_eli.api_action): on any
    failure the session is put back as it was, so nothing is half-applied."""
    pending = st.session_state.pop("pending", None)
    if not pending:
        return
    state = _app_state()
    action = app_eli.ACTIONS.get(pending["action"])
    snapshot = deepcopy(state)
    spinner = REVIEWING if pending["reviewing"] else SPINNER_TEXT.get(pending["action"], "Working...")
    try:
        with st.spinner(spinner):
            error = action(state, pending["payload"]) if action else ""
            app_eli._process_pending_addon(state)
    except Exception:
        app_eli.logger.exception("Action %s failed", pending["action"])
        st.session_state.app_state = snapshot
        error = "Sorry - the assistant couldn't respond just now. Nothing was lost; please try again."
        st.session_state.retry = pending
    st.session_state.error = error or ""
    if not error:
        st.session_state.pop("retry", None)
    if error and pending["action"] == "send":
        # Not answered: give the patient their words back.
        st.session_state.restore_composer = pending["payload"].get("text", "")
    if not error and pending["action"] == "start":
        # Hand-typed history belongs to this check-in only; cleared on the next run
        # (a widget's value cannot change after it was drawn in this run).
        st.session_state.clear_prior = True
    st.rerun()


def retry() -> None:
    pending = st.session_state.pop("retry", None)
    if pending:
        st.session_state.pending = pending
        st.session_state.error = ""


# ---------------------------------------------------------------- sidebar
def render_sidebar(view: Dict[str, Any], state) -> None:
    busy = "pending" in st.session_state
    with st.sidebar:
        panel = view.get("patient_panel")
        if panel:
            if panel.get("topic_boxes_html"):
                st.markdown("**Your check-in**")
                st.markdown(panel["topic_boxes_html"], unsafe_allow_html=True)
            if panel.get("addon_labels") is not None:
                st.caption("Remembered another symptom? Add it and we'll make sure to cover it.")
                with st.expander("➕ Add a symptom"):
                    if panel["addon_labels"]:
                        for label in panel["addon_labels"]:
                            st.button(f"➕ {label}", key=f"addon_{label}", width="stretch", disabled=busy,
                                      on_click=queue, args=("add_symptom", {"label": label}))
                    else:
                        st.caption("Everything is already on your list.")
            st.markdown("**Pace**")
            col1, col2 = st.columns(2)
            col1.button("🐢 Slow down", width="stretch", disabled=busy,
                        help="Give me more room to explain - the assistant won't rush you.",
                        on_click=queue, args=("pace", {"mode": "slower"}))
            col2.button("🐇 Speed up", width="stretch", disabled=busy,
                        help="I'm getting tired - keep it brief and wrap up sooner.",
                        on_click=queue, args=("pace", {"mode": "faster"}))
            pace_label = {
                "normal": "Normal pace",
                "faster": "Going faster · tap again for normal",
                "slower": "Taking it slower · tap again for normal",
            }[panel["pace_mode"]]
            st.caption(f"Pace: {pace_label}")
            st.button("Wrap up check-in", type="primary", width="stretch", disabled=busy,
                      help="Review your check-in list and confirm before finishing - your responses are always saved.",
                      on_click=queue, args=("wrap_up", {}))
            st.divider()

        if st.session_state.pop("clear_prior", False):
            st.session_state.setup_prior = ""
        st.caption(f"Status: {view['status']}")
        st.text_input("Patient name *", key="setup_patient", placeholder="Required")
        st.text_input("Doctor name", key="setup_doctor", placeholder="Optional")
        st.text_input("Week of therapy", key="setup_week", placeholder="Example: Week 3")
        if view["check_in_started"]:
            st.text_area("Prior patient history", value=view["saved_prior_history"], height=200,
                         disabled=True, placeholder="No previous check-in found for this patient.",
                         key=f"prior_used_{state.session_id}")
            if view["prior_history_status"]:
                st.caption(view["prior_history_status"]
                           + (" · used by the chatbot for this check-in" if view["saved_prior_history"] else ""))
        else:
            st.text_area("Prior patient history", key="setup_prior", height=110,
                         placeholder="Leave empty to load this patient's last check-in automatically, "
                                     "or type the history here.")
        with st.expander("⚙️ Chatbot prompt"):
            if "setup_prompt" not in st.session_state:
                st.session_state.setup_prompt = view["default_system_prompt"]
            st.text_area("Editable chatbot instructions", key="setup_prompt", height=260)

        def start():
            if not st.session_state.get("setup_patient", "").strip():
                st.session_state.side_error = "Please enter the patient's name before starting the check-in."
                return
            st.session_state.side_error = ""
            queue("start", {
                "patient_name": st.session_state.get("setup_patient", ""),
                "doctor_name": st.session_state.get("setup_doctor", ""),
                "therapy_week": st.session_state.get("setup_week", ""),
                "prior_history": st.session_state.get("setup_prior", ""),
                "system_prompt": st.session_state.get("setup_prompt", view["default_system_prompt"]),
            })

        st.button("Start new check-in", width="stretch", disabled=busy, on_click=start)
        if st.session_state.get("side_error"):
            st.error(st.session_state.side_error)


# ---------------------------------------------------------------- main screens
def render_checklist(view, state) -> None:
    c = view["checklist"]
    sid = state.session_id
    st.subheader(c["question"])
    st.caption("Check all that apply. We'll only ask follow-up questions about the things you select - "
               "everything else is noted for your care team automatically.")
    cols = st.columns(2)
    for i, label in enumerate(c["labels"]):
        cols[i % 2].checkbox(label, key=f"chk_{sid}_{label}")
    chosen = [label for label in c["labels"] if st.session_state.get(f"chk_{sid}_{label}")]
    other = ""
    if "Something else" in chosen:
        other = st.text_input("Tell us briefly what else is bothering you", max_chars=200, key=f"other_{sid}")
    none = st.checkbox(c["none_label"], key=f"none_{sid}", disabled=bool(chosen))

    def continue_checklist():
        # Read the boxes at click time, so nothing typed just before the click is lost.
        labels = [label for label in c["labels"] if st.session_state.get(f"chk_{sid}_{label}")]
        queue("checklist_continue", {
            "labels": labels,
            "other_description": st.session_state.get(f"other_{sid}", "") if "Something else" in labels else "",
            "none": bool(st.session_state.get(f"none_{sid}")) and not labels,
        }, "", True)

    st.button("Continue", type="primary", width="stretch", disabled=not (chosen or none),
              on_click=continue_checklist)


def render_messages(view) -> None:
    for message in view["messages"]:
        user = message["role"] == "user"
        with st.chat_message("user" if user else "assistant", avatar="🙂" if user else "🩺"):
            st.markdown(message["html"], unsafe_allow_html=True)
        if message["red_flag"]:
            notice = view["self_harm_notice"] if message["self_harm"] else view["red_flag_notice"]
            st.markdown(f'<div class="red-flag-notice" role="alert">⚠️ {app_eli.html.escape(notice)}</div>',
                        unsafe_allow_html=True)
    pending = st.session_state.get("pending")
    if pending and pending["echo"]:
        with st.chat_message("user", avatar="🙂"):
            st.markdown(app_eli.html.escape(pending["echo"]))


def _append_suggestion(key: str, suggestion: str) -> None:
    # Tapping a suggestion ADDS it to the response box - it never submits on its own.
    current = st.session_state.get(key, "").rstrip()
    if not current:
        st.session_state[key] = suggestion
    else:
        st.session_state[key] = current + (" " if current[-1] in ".!?,;:" else ", ") + suggestion


def render_chat(view, state) -> None:
    sid = state.session_id
    render_messages(view)
    busy = "pending" in st.session_state
    mode = view.get("mode")

    if mode == "worst_pick":
        st.markdown("#### Which one is bothering you the most?")
        st.caption("Tap the symptom that is troubling you most - we'll spend the most time on that one, "
                   "and still cover the others.")
        for label in view["worst_labels"]:
            st.button(label, key=f"worst_{label}", width="stretch", disabled=busy, on_click=queue,
                      args=("worst_pick", {"label": label}, f"The {label.lower()} is bothering me the most."))
        return

    if mode == "closing_review":
        # One set of boxes per visit to this screen; it only changes when the patient adds
        # something here and the chat resumes.
        visits = sum(1 for m in state.messages if m.get("response_mode") == "closing_addon")
        round_key = f"{sid}_{visits}"
        st.markdown("#### Before we finish — is there anything else?")
        st.caption("Here's your check-in list, with the symptoms you told me about already ticked. Tick anything "
                   "else you'd like to talk about, or write it in the box below - including anything I haven't "
                   "asked about, or a question for your doctor. If not, press \"No, nothing else — finish check-in\".")
        cols = st.columns(2)
        picked = []
        for i, item in enumerate(view["review_items"]):
            if item["already"]:
                cols[i % 2].checkbox(item["label"], value=True, disabled=True, key=f"rev_{round_key}_{item['label']}")
            elif cols[i % 2].checkbox(item["label"], key=f"rev_{round_key}_{item['label']}"):
                picked.append(item["label"])
        other = st.text_area("Anything else?", key=f"closing_other_{round_key}", height=80,
                             placeholder="Optional - anything I haven't asked about, or a question for your doctor")
        def closing_inputs():
            # Read at click time, so text typed just before the click is not lost.
            labels = [item["label"] for item in view["review_items"]
                      if not item["already"] and st.session_state.get(f"rev_{round_key}_{item['label']}")]
            return labels, st.session_state.get(f"closing_other_{round_key}", "")

        def add_and_continue():
            labels, text = closing_inputs()
            queue("closing_add", {"labels": labels, "other_text": text}, "", True)

        def finish():
            _labels, text = closing_inputs()
            queue("closing_finish", {"other_text": text})

        col1, col2 = st.columns(2)
        if picked or other.strip():
            col1.button("Add these & keep going", width="stretch", disabled=busy, on_click=add_and_continue)
        col2.button("No, nothing else — finish check-in", type="primary", width="stretch", disabled=busy,
                    on_click=finish)
        return

    # Compose mode
    if view.get("addon_toast"):
        st.info(f"I've added {view['addon_toast'].lower()} to your list - I'll ask you about it after you answer "
                "this question.")
    if view.get("pending_offer"):
        wrap = view["pending_offer"] == "wrap"
        st.caption("Tap your choice — or just type your answer below.")
        col1, col2 = st.columns(2)
        col1.button("Keep going", width="stretch", disabled=busy, on_click=queue,
                    args=("offer", {"choice": "continue"}, "Let's keep going."))
        col2.button("Wrap up now" if wrap else "Finish now", width="stretch", disabled=busy, on_click=queue,
                    args=("offer", {"choice": "wrap"}))

    composer_key = f"composer_{sid}"
    if "restore_composer" in st.session_state:
        st.session_state[composer_key] = st.session_state.pop("restore_composer")
    suggestions = view.get("suggestions") or []
    if suggestions:
        st.button("Hide suggestions" if view.get("show_suggestions") else "Show suggestions",
                  disabled=busy, on_click=queue, args=("toggle_suggestions", {}))
        if view.get("show_suggestions"):
            st.caption("Optional — tap any that apply to add them to your response. You can pick several, "
                       "edit the text, or type your own, then press Send.")
            for i, suggestion in enumerate(suggestions):
                st.button(suggestion, key=f"sugg_{len(state.messages)}_{i}", width="stretch", disabled=busy,
                          on_click=_append_suggestion, args=(composer_key, suggestion))

    st.text_area("Your response", key=composer_key, height=100, label_visibility="collapsed",
                 placeholder="Type your response here, or tap a suggestion above to start from it…")

    def send():
        text = st.session_state.get(composer_key, "").strip()
        if not text:
            return
        st.session_state[composer_key] = ""
        queue("send", {"text": text, "id": app_eli.secrets.token_hex(8)}, text)

    st.button("Send", type="primary", width="stretch", disabled=busy, on_click=send)


def render_patient_summary(view) -> None:
    p = view["patient_summary"]
    st.markdown("### Your check-in summary")
    st.subheader("Thank you - your check-in is complete")
    st.write("This is what I'll share with your care team:")
    st.info(p["text"])
    if p["notice"]:
        st.markdown(f'<div class="red-flag-notice" role="alert">⚠️ {app_eli.html.escape(p["notice"])}</div>',
                    unsafe_allow_html=True)
    correction = st.text_area("**Is anything missing or not quite right?** (optional)", key="summary_correction",
                              height=80, placeholder="Anything you'd like your care team to know or correct")
    st.button("Continue to the doctor's report", type="primary", width="stretch",
              disabled="pending" in st.session_state,
              # Read the box at click time, so a correction typed just before the click is kept.
              on_click=lambda: queue("patient_summary_continue",
                                     {"correction": st.session_state.get("summary_correction", "")}))


def render_dashboard(view, state) -> None:
    d = view["dashboard"]
    if d.get("warning"):
        st.warning(d["warning"])
        st.button("Regenerate summary", on_click=queue, args=("regenerate_summary", {}))
    if not d.get("html"):
        return
    st.markdown(d["html"], unsafe_allow_html=True)
    with st.expander("Prior history"):
        st.text(d["prior_history"])
    with st.expander("Full conversation"):
        for m in d["conversation"]:
            st.markdown(f"**{m['label']}**" + (f" ({m['mode']})" if m.get("mode") else ""))
            st.text(m["content"])
            if m.get("suggested_answers"):
                st.caption("Suggestions generated: " + " | ".join(m["suggested_answers"]))
    if state.is_complete and state.summary_generated and state.doctor_summary_structured:
        st.download_button(
            "Download reproducibility record (JSON)",
            data=json.dumps(app_eli.build_export_payload(state), ensure_ascii=False, indent=2),
            file_name=f"check_in_{state.session_id}.json",
            mime="application/json",
            width="stretch",
        )
    col1, col2 = st.columns(2)
    col1.button("Start new check-in", width="stretch", on_click=queue, args=("new_checkin", {}))
    col2.button("Regenerate summary", width="stretch", key="regen_bottom", on_click=queue,
                args=("regenerate_summary", {}))


def password_ok() -> bool:
    """Optional gate for a public link: when APP_PASSWORD is set in the secrets, nothing
    is shown until it is entered (the sidebar can load a patient's last check-in by name)."""
    required = _SECRETS.get("APP_PASSWORD")
    if not required or st.session_state.get("authenticated"):
        return True
    st.markdown("### 🩺 Nurse Assistant Check-In")
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), str(required).encode()):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    return False


def main() -> None:
    if not password_ok():
        return
    state = _app_state()
    view = app_eli.build_view(state)
    render_sidebar(view, state)

    screen = view["screen"]
    if screen == "dashboard":
        render_dashboard(view, state)
    elif screen == "preparing":
        if "pending" not in st.session_state:
            queue("generate_summary")
    elif screen == "preparing_patient":
        if "pending" not in st.session_state:
            queue("generate_patient_summary")
    elif screen == "patient_summary":
        render_patient_summary(view)
    else:
        st.markdown("### 🩺 Your pre-visit check-in")
        if screen == "setup":
            st.info("Enter the patient name in the sidebar, add prior patient history if available, then click "
                    "**Start new check-in** to begin.")
        elif screen == "welcome":
            w = view["welcome"]
            st.subheader(w["title"])
            st.write(w["body"])
            st.markdown(f'<div class="red-flag-notice" style="border-color:#b45309;background:#fffbeb;'
                        f'color:#78350f;font-weight:400">{w["disclaimer_html"]}</div>', unsafe_allow_html=True)
            st.button(w["button"], type="primary", width="stretch", on_click=queue, args=("acknowledge", {}))
        else:
            st.markdown(f'<div class="banner">{app_eli.html.escape(view["disclaimer_banner"])}</div>',
                        unsafe_allow_html=True)
            if screen == "checklist":
                render_checklist(view, state)
            elif screen == "chat":
                render_chat(view, state)

    if st.session_state.get("error"):
        st.error(st.session_state.error)
        if st.session_state.get("retry"):
            st.button("Try again", on_click=retry)

    run_pending()


main()
