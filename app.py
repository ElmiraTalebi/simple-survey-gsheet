import streamlit as st

st.set_page_config(
    page_title="HNC Symptom Questionnaire",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------------------------------------
# CSS STYLING
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .question-row {
        padding-top:6px;
        padding-bottom:6px;
        border-bottom:1px solid #f0f0f0;
    }

    .question-text {
        font-size:15px;
        font-weight:500;
    }

    .score-button button {
        width:100%;
        height:38px;
        font-size:15px;
        border-radius:8px;
    }

    .selected button {
        background-color:#4CAF50 !important;
        color:white !important;
        font-weight:600;
        border:1px solid #4CAF50 !important;
    }

    .unselected button {
        background-color:#f5f5f5;
        color:#333;
        border:1px solid #ddd;
    }

    .submit-area{
        padding-top:25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# QUESTION LIST (28 QUESTIONS)
# ------------------------------------------------------------

questions = [
"Pain in throat",
"Dry mouth",
"Difficulty swallowing",
"Difficulty chewing",
"Pain when swallowing",
"Change in voice",
"Hoarseness",
"Lump in neck",
"Ear pain",
"Weight loss",
"Loss of appetite",
"Difficulty breathing",
"Coughing",
"Fatigue",
"Mouth sores",
"Taste changes",
"Bleeding in mouth",
"Jaw stiffness",
"Difficulty opening mouth",
"Speech difficulty",
"Burning sensation in mouth",
"Thick saliva",
"Trouble sleeping",
"Anxiety",
"Depression",
"Difficulty concentrating",
"Skin irritation in treatment area",
"Swelling in face or neck"
]

# ------------------------------------------------------------
# SIMULATED PREVIOUS ANSWERS
# ------------------------------------------------------------

default_previous = {
q: 0 for q in questions
}

default_previous.update({
"Pain in throat":2,
"Dry mouth":3,
"Difficulty swallowing":1
})

# ------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------

if "previous_answers" not in st.session_state:
    st.session_state.previous_answers = default_previous.copy()

if "current_answers" not in st.session_state:
    st.session_state.current_answers = st.session_state.previous_answers.copy()

if "show_followup" not in st.session_state:
    st.session_state.show_followup = False

if "followup_response" not in st.session_state:
    st.session_state.followup_response = ""

# ------------------------------------------------------------
# HELPER FUNCTION
# ------------------------------------------------------------

def set_answer(question, value):
    st.session_state.current_answers[question] = value


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.title("Head & Neck Cancer Symptom Questionnaire")

st.markdown(
"""
Please rate the severity of each symptom.

0 = None  
5 = Severe
"""
)

# ------------------------------------------------------------
# QUESTION RENDER FUNCTION
# ------------------------------------------------------------

def render_question(question):

    cols = st.columns([6,1,1,1,1,1,1])

    cols[0].markdown(f"<div class='question-text'>{question}</div>", unsafe_allow_html=True)

    current_value = st.session_state.current_answers.get(question,0)

    for i in range(6):

        with cols[i+1]:

            selected = current_value == i

            class_name = "selected" if selected else "unselected"

            st.markdown(f"<div class='score-button {class_name}'>", unsafe_allow_html=True)

            if st.button(
                str(i),
                key=f"{question}_{i}"
            ):
                set_answer(question, i)
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# RENDER QUESTIONNAIRE
# ------------------------------------------------------------

for q in questions:
    st.markdown("<div class='question-row'>", unsafe_allow_html=True)
    render_question(q)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# SUBMIT BUTTON
# ------------------------------------------------------------

st.markdown("<div class='submit-area'>", unsafe_allow_html=True)

submit = st.button("Submit Questionnaire", use_container_width=True)

# ------------------------------------------------------------
# FOLLOW-UP LOGIC
# ------------------------------------------------------------

if submit:

    previous = st.session_state.previous_answers
    current = st.session_state.current_answers

    followup_trigger = False

    for q in questions:

        prev = previous.get(q,0)
        curr = current.get(q,0)

        if curr >= 5:
            followup_trigger = True

        if curr - prev > 2:
            followup_trigger = True

    st.session_state.show_followup = followup_trigger

    st.success("Questionnaire submitted.")

# ------------------------------------------------------------
# FOLLOW-UP QUESTIONS
# ------------------------------------------------------------

if st.session_state.show_followup:

    st.warning("We noticed a significant increase in symptom severity.")

    st.session_state.followup_response = st.text_area(
        "Can you briefly describe what changed?",
        value=st.session_state.followup_response
    )

# ------------------------------------------------------------
# DEMO OUTPUT
# ------------------------------------------------------------

if submit:

    st.subheader("Collected Answers")

    st.json(st.session_state.current_answers)

    if st.session_state.show_followup:

        st.subheader("Follow-up Response")

        st.write(st.session_state.followup_response)

    # Save current as previous for next visit simulation
    st.session_state.previous_answers = st.session_state.current_answers.copy()
