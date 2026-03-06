import streamlit as st

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Patient Symptom Questionnaire",
    layout="wide"
)

# ------------------------------------------------------------
# CSS styling
# ------------------------------------------------------------
st.markdown("""
<style>

.question-row {
    padding:4px 0px;
}

.question-text {
    font-size:15px;
    font-weight:500;
}

.score-btn button {
    width:55px;
    height:40px;
    font-size:16px;
    border-radius:8px;
}

.selected-btn button {
    background-color:#2b7cff;
    color:white;
    border:2px solid #1f5bd8;
}

.submit-btn button {
    width:250px;
    height:45px;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Questionnaire questions (28)
# ------------------------------------------------------------
QUESTIONS = [
    "Pain in throat",
    "Difficulty swallowing",
    "Dry mouth",
    "Taste changes",
    "Mouth sores",
    "Hoarseness",
    "Difficulty speaking",
    "Neck swelling",
    "Jaw pain",
    "Ear pain",
    "Fatigue",
    "Nausea",
    "Loss of appetite",
    "Weight loss",
    "Shortness of breath",
    "Cough",
    "Chest discomfort",
    "Sleep disturbance",
    "Anxiety",
    "Depression",
    "Skin irritation",
    "Difficulty concentrating",
    "Memory problems",
    "Headache",
    "Dizziness",
    "Muscle weakness",
    "Numbness",
    "General discomfort"
]


# ------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------
if "current_answers" not in st.session_state:
    st.session_state.current_answers = {q: 0 for q in QUESTIONS}

if "previous_answers" not in st.session_state:
    st.session_state.previous_answers = {q: 0 for q in QUESTIONS}

if "followups_needed" not in st.session_state:
    st.session_state.followups_needed = []

if "followup_responses" not in st.session_state:
    st.session_state.followup_responses = {}


# ------------------------------------------------------------
# Function to render one question row
# ------------------------------------------------------------
def render_question_row(question):

    cols = st.columns([4,1,1,1,1,1,1])

    with cols[0]:
        st.markdown(f'<div class="question-text">{question}</div>', unsafe_allow_html=True)

    current_value = st.session_state.current_answers[question]

    for i in range(6):

        button_class = "score-btn"

        if current_value == i:
            button_class = "selected-btn"

        with cols[i+1]:

            if st.button(
                str(i),
                key=f"{question}_{i}"
            ):
                st.session_state.current_answers[question] = i


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("Patient Symptom Questionnaire")

st.write("Please rate each symptom from **0 (none)** to **5 (severe)**.")

st.markdown("---")


# ------------------------------------------------------------
# Render questionnaire grid
# ------------------------------------------------------------
for q in QUESTIONS:

    st.markdown('<div class="question-row">', unsafe_allow_html=True)

    render_question_row(q)

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Submit Button
# ------------------------------------------------------------
st.markdown("---")

submit = st.button("Submit Questionnaire")


# ------------------------------------------------------------
# Follow-up logic
# ------------------------------------------------------------
if submit:

    followups = []

    for q in QUESTIONS:

        current = st.session_state.current_answers[q]
        previous = st.session_state.previous_answers[q]

        if current >= 5 or (current - previous) > 2:
            followups.append(q)

    st.session_state.followups_needed = followups

    if len(followups) == 0:

        st.success("Thank you. No follow-up questions required.")

    else:

        st.warning("We noticed your symptom score increased significantly.")

        for q in followups:

            response = st.text_input(
                f"{q}: Can you briefly describe what changed?",
                key=f"followup_{q}"
            )

            st.session_state.followup_responses[q] = response

    # Print answers for demo
    st.markdown("### Collected Answers")

    st.json({
        "current_answers": st.session_state.current_answers,
        "followup_responses": st.session_state.followup_responses
    })

    # Save current as previous for next visit
    st.session_state.previous_answers = st.session_state.current_answers.copy()
