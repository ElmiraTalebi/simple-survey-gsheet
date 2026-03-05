import json
from datetime import datetime
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Cancer Symptom Check-In", page_icon="🩺", layout="centered")

# ============================================================
# Google Sheets
# ============================================================

sheet = None


def init_sheets():
    global sheet

    if sheet:
        return

    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )

        gc = gspread.authorize(creds)
        book = gc.open_by_key(st.secrets["gsheet_id"])

        try:
            sheet = book.worksheet("Form")
        except:
            sheet = book.add_worksheet(title="Form", rows=2000, cols=20)
            sheet.append_row(["timestamp", "name", "json"])

    except Exception as e:
        st.warning(f"Sheets unavailable: {e}")


def load_last_checkin(name):

    init_sheets()

    if sheet is None:
        return None

    rows = sheet.get_all_values()

    for row in reversed(rows[1:]):
        if row[1].lower() == name.lower():

            try:
                return json.loads(row[2])
            except:
                return None

    return None


def save_to_sheet(data):

    init_sheets()

    if sheet is None:
        return

    sheet.append_row(
        [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data["name"],
            json.dumps(data),
        ]
    )


# ============================================================
# Body map SVG
# ============================================================

def body_svg():
    return """
<svg width="250" height="380" viewBox="0 0 320 520" xmlns="http://www.w3.org/2000/svg">
<circle cx="160" cy="70" r="38" fill="#e0e0e0"/>
<rect x="110" y="120" width="100" height="70" rx="24" fill="#e0e0e0"/>
<rect x="115" y="195" width="90" height="70" rx="22" fill="#e0e0e0"/>
<rect x="120" y="260" width="30" height="120" fill="#e0e0e0"/>
<rect x="170" y="260" width="30" height="120" fill="#e0e0e0"/>
</svg>
"""


# ============================================================
# Session state defaults
# ============================================================

defaults = dict(
    stage=-1,
    name="",
    feeling=None,
    pain=None,
    pain_locations=set(),
    pain_severity={},
    last_severity={},
    symptoms=set(),
    show_other=False,
)

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# Header
# ============================================================

st.title("🩺 Cancer Symptom Check-In")


# ============================================================
# Stage -1 : Name
# ============================================================

if st.session_state.stage == -1:

    name = st.text_input("Enter your name")

    if st.button("Start Check-In"):

        if name:

            st.session_state.name = name

            last = load_last_checkin(name)

            if last and isinstance(last.get("pain_severity"), dict):
                st.session_state.last_severity = last["pain_severity"]
            else:
                st.session_state.last_severity = {}

            st.session_state.stage = 0
            st.rerun()

    st.stop()


stage = st.session_state.stage


# ============================================================
# Stage 0 Fast exit
# ============================================================

if stage == 0:

    st.subheader("How have you been since your last visit?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Same as yesterday"):
            st.success("Check-in complete.")
            st.session_state.stage = 5
            st.rerun()

    with col2:
        if st.button("Something changed"):
            st.session_state.stage = 1
            st.rerun()


# ============================================================
# Stage 1 Feeling
# ============================================================

elif stage == 1:

    feeling = st.radio(
        "How are you feeling today?",
        ["Excellent", "Very good", "Good", "Fair", "Poor"],
    )

    if st.button("Next"):
        st.session_state.feeling = feeling
        st.session_state.stage = 2
        st.rerun()


# ============================================================
# Stage 2 Pain Yes/No
# ============================================================

elif stage == 2:

    pain = st.radio("Do you have pain today?", ["No", "Yes"])

    if st.button("Next"):

        st.session_state.pain = pain

        if pain == "Yes":
            st.session_state.stage = 3
        else:
            st.session_state.stage = 4

        st.rerun()


# ============================================================
# Stage 3 Body Map
# ============================================================

elif stage == 3:

    st.subheader("Mark where you feel pain")

    st.markdown(body_svg(), unsafe_allow_html=True)

    parts = [
        "Head",
        "Neck",
        "Chest",
        "Abdomen",
        "Left Arm",
        "Right Arm",
        "Left Leg",
        "Right Leg",
    ]

    cols = st.columns(2)

    for i, part in enumerate(parts):

        with cols[i % 2]:

            selected = part in st.session_state.pain_locations
            color = "🔴" if selected else "🟢"

            if st.button(f"{color} {part}", key=part):

                if selected:
                    st.session_state.pain_locations.remove(part)
                else:
                    st.session_state.pain_locations.add(part)

                st.rerun()

            if part in st.session_state.pain_locations:

                st.info("How severe is it (0-10)?")

                last_val = 0
                if isinstance(st.session_state.last_severity, dict):
                    last_val = st.session_state.last_severity.get(part, 0)

                severity = st.number_input(
                    f"{part} severity",
                    0,
                    10,
                    value=last_val,
                    key=f"sev_{part}",
                )

                st.session_state.pain_severity[part] = severity

                if severity > 6 or severity > last_val:

                    st.warning("What seems to be causing the worsening?")

                    st.text_input(
                        f"Reason for {part}",
                        key=f"reason_{part}",
                    )

    st.markdown("---")

    if st.button("Other location"):
        st.session_state.show_other = True

    if st.session_state.show_other:

        other = st.text_input("Describe other pain location")

        if other:

            st.session_state.pain_locations.add(other)

            sev = st.number_input(
                "Severity (0-10)",
                0,
                10,
                key="other_sev",
            )

            st.session_state.pain_severity[other] = sev

    if st.button("Next"):
        st.session_state.stage = 4
        st.rerun()


# ============================================================
# Stage 4 Symptoms (Clickable)
# ============================================================

elif stage == 4:

    st.subheader("Symptoms today")

    symptoms_list = [
        "Fatigue",
        "Nausea",
        "Dry mouth",
        "Difficulty swallowing",
        "Hoarseness",
        "Mouth sores",
        "Skin irritation",
        "Loss of taste",
    ]

    cols = st.columns(2)

    for i, sym in enumerate(symptoms_list):

        with cols[i % 2]:

            selected = sym in st.session_state.symptoms
            color = "🔴" if selected else "🟢"

            if st.button(f"{color} {sym}", key=f"sym_{sym}"):

                if selected:
                    st.session_state.symptoms.remove(sym)
                else:
                    st.session_state.symptoms.add(sym)

                st.rerun()

    st.markdown("---")

    if st.button("Other symptom"):
        st.session_state.show_other = True

    if st.session_state.show_other:

        other = st.text_input("Describe other symptom")

        if other:
            st.session_state.symptoms.add(other)

    st.markdown("---")

    if st.button("Submit Check-In"):

        data = dict(
            name=st.session_state.name,
            feeling=st.session_state.feeling,
            pain=st.session_state.pain,
            pain_locations=list(st.session_state.pain_locations),
            pain_severity=st.session_state.pain_severity,
            symptoms=list(st.session_state.symptoms),
        )

        save_to_sheet(data)

        st.session_state.stage = 5
        st.rerun()


# ============================================================
# Stage 5 Done
# ============================================================

elif stage == 5:

    st.success("Check-in complete.")

    st.write("Your care team will review your responses.")

    if st.button("Start another check-in"):

        for k in defaults:
            st.session_state[k] = defaults[k]

        st.rerun()
