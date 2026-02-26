import streamlit as st
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="ChatReport - Symptom Reporting",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.symptoms = {}
    st.session_state.patient_name = ""
    st.session_state.conversation = []

def add_to_conversation(speaker, message):
    """Add message to conversation history"""
    st.session_state.conversation.append({
        'speaker': speaker,
        'message': message,
        'time': datetime.now().strftime("%H:%M:%S")
    })

def display_conversation():
    """Display conversation history"""
    for msg in st.session_state.conversation:
        if msg['speaker'] == 'bot':
            st.markdown(f"🤖 **ChatReport:** {msg['message']}")
        else:
            st.markdown(f"👤 **You:** {msg['message']}")
        st.markdown("---")

def generate_report():
    """Generate clinical report"""
    report = []
    report.append("## 📋 CHATREPORT SYMPTOM SUMMARY")
    report.append("---")
    report.append(f"**Patient:** {st.session_state.patient_name}")
    report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("---")
    
    # Pain
    if st.session_state.symptoms.get('pain', {}).get('present'):
        report.append("### 🔴 PAIN - PRESENT")
        details = st.session_state.symptoms['pain']['details']
        report.append(f"- **Location:** {details.get('location', 'N/A')}")
        report.append(f"- **Severity:** {details.get('severity', 'N/A')}/10")
        report.append(f"- **Frequency:** {details.get('frequency', 'N/A')}")
        report.append(f"- **Management:** {details.get('management', 'N/A')}")
    else:
        report.append("### ✅ Pain - Not reported")
    
    # Mouth
    if st.session_state.symptoms.get('mouth', {}).get('present'):
        report.append("### 🔴 MOUTH SYMPTOMS - PRESENT")
        details = st.session_state.symptoms['mouth']['details']
        report.append(f"- **Description:** {details.get('description', 'N/A')}")
        report.append(f"- **Severity:** {details.get('severity', 'N/A')}")
    else:
        report.append("### ✅ Mouth - Not reported")
    
    # Swallowing
    if st.session_state.symptoms.get('swallowing', {}).get('present'):
        report.append("### 🔴 SWALLOWING DIFFICULTY - PRESENT")
        details = st.session_state.symptoms['swallowing']['details']
        report.append(f"- **Description:** {details.get('description', 'N/A')}")
        report.append(f"- **Food types:** {details.get('food_types', 'N/A')}")
    else:
        report.append("### ✅ Swallowing - No difficulty")
    
    # Nutrition
    if 'nutrition' in st.session_state.symptoms:
        report.append("### 📊 NUTRITION")
        details = st.session_state.symptoms['nutrition']['details']
        report.append(f"- **Appetite:** {details.get('appetite', 'N/A')}")
        report.append(f"- **Weight:** {details.get('weight', 'N/A')}")
    
    # Mood
    if 'mood' in st.session_state.symptoms:
        report.append("### 💭 MOOD")
        details = st.session_state.symptoms['mood']['details']
        report.append(f"- **Mood:** {details.get('mood', 'N/A')}")
        if 'sleep' in details:
            report.append(f"- **Sleep:** {details['sleep']}")
    
    # Additional notes
    if 'additional_notes' in st.session_state.symptoms:
        report.append("### 📝 ADDITIONAL NOTES")
        report.append(f"{st.session_state.symptoms['additional_notes']}")
    
    return "\n\n".join(report)

# Header
st.title("🏥 ChatReport")
st.subheader("Symptom Reporting System - Fox Chase Cancer Center")
st.markdown("---")

# Main conversation flow
if st.session_state.step == 0:
    st.markdown("### Welcome!")
    st.write("Hello! I'm ChatReport. I'll help you share how you've been feeling before your appointment.")
    st.write("This takes about 10 minutes. Let's get started!")
    
    name = st.text_input("What's your first name?", key="name_input")
    
    if st.button("Start"):
        if name:
            st.session_state.patient_name = name
            add_to_conversation('bot', "What's your first name?")
            add_to_conversation('user', name)
            add_to_conversation('bot', f"Thanks, {name}! Let's get started.")
            st.session_state.step = 1
            st.rerun()
        else:
            st.warning("Please enter your name to continue.")

elif st.session_state.step == 1:
    st.markdown("### Pain Assessment")
    display_conversation()
    
    st.write("🤖 **ChatReport:** Have you had any pain since your last appointment?")
    pain_response = st.radio("Select one:", ["Yes", "No"], key="pain_yn")
    
    if st.button("Next", key="pain_next"):
        add_to_conversation('bot', "Have you had any pain since your last appointment?")
        add_to_conversation('user', pain_response)
        
        if pain_response == "No":
            st.session_state.symptoms['pain'] = {'present': False}
            add_to_conversation('bot', "Good to hear.")
            st.session_state.step = 5
        else:
            st.session_state.symptoms['pain'] = {'present': True, 'details': {}}
            st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.markdown("### Pain Details")
    display_conversation()
    
    st.write("🤖 **ChatReport:** Where is the pain located?")
    location = st.text_input("Pain location:", key="pain_location")
    
    if st.button("Next", key="pain_loc_next"):
        if location:
            add_to_conversation('bot', "Where is the pain located?")
            add_to_conversation('user', location)
            st.session_state.symptoms['pain']['details']['location'] = location
            st.session_state.step = 3
            st.rerun()
        else:
            st.warning("Please provide a response.")

elif st.session_state.step == 3:
    st.markdown("### Pain Severity")
    display_conversation()
    
    st.write("🤖 **ChatReport:** On a scale of 0-10, how bad is it at its worst?")
    severity = st.slider("Pain severity:", 0, 10, 5, key="pain_severity")
    
    if st.button("Next", key="pain_sev_next"):
        add_to_conversation('bot', "On a scale of 0-10, how bad is it at its worst?")
        add_to_conversation('user', str(severity))
        st.session_state.symptoms['pain']['details']['severity'] = str(severity)
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    st.markdown("### Pain Frequency & Management")
    display_conversation()
    
    st.write("🤖 **ChatReport:** How often do you have this pain?")
    frequency = st.text_input("Frequency:", key="pain_freq")
    
    st.write("🤖 **ChatReport:** What are you doing to manage it?")
    management = st.text_area("Management strategies:", key="pain_mgmt")
    
    if st.button("Next", key="pain_mgmt_next"):
        if frequency and management:
            add_to_conversation('bot', "How often do you have this pain?")
            add_to_conversation('user', frequency)
            add_to_conversation('bot', "What are you doing to manage it?")
            add_to_conversation('user', management)
            st.session_state.symptoms['pain']['details']['frequency'] = frequency
            st.session_state.symptoms['pain']['details']['management'] = management
            st.session_state.step = 5
            st.rerun()
        else:
            st.warning("Please answer both questions.")

elif st.session_state.step == 5:
    st.markdown("### Mouth Symptoms")
    display_conversation()
    
    st.write("🤖 **ChatReport:** Have you noticed any dry mouth, mouth sores, or taste changes?")
    mouth_response = st.radio("Select one:", ["Yes", "No"], key="mouth_yn")
    
    if st.button("Next", key="mouth_next"):
        add_to_conversation('bot', "Have you noticed any dry mouth, mouth sores, or taste changes?")
        add_to_conversation('user', mouth_response)
        
        if mouth_response == "No":
            st.session_state.symptoms['mouth'] = {'present': False}
            st.session_state.step = 7
        else:
            st.session_state.symptoms['mouth'] = {'present': True, 'details': {}}
            st.session_state.step = 6
        st.rerun()

elif st.session_state.step == 6:
    st.markdown("### Mouth Symptom Details")
    display_conversation()
    
    st.write("🤖 **ChatReport:** Can you describe what you're experiencing?")
    description = st.text_area("Description:", key="mouth_desc")
    
    st.write("🤖 **ChatReport:** How severe is it?")
    severity = st.selectbox("Severity:", ["Mild", "Moderate", "Severe"], key="mouth_sev")
    
    if st.button("Next", key="mouth_det_next"):
        if description:
            add_to_conversation('bot', "Can you describe what you're experiencing?")
            add_to_conversation('user', description)
            add_to_conversation('bot', "How severe is it?")
            add_to_conversation('user', severity)
            st.session_state.symptoms['mouth']['details']['description'] = description
            st.session_state.symptoms['mouth']['details']['severity'] = severity
            st.session_state.step = 7
            st.rerun()
        else:
            st.warning("Please provide a description.")

elif st.session_state.step == 7:
    st.markdown("### Swallowing")
    display_conversation()
    
    st.write("🤖 **ChatReport:** Any difficulty swallowing?")
    swallow_response = st.radio("Select one:", ["Yes", "No"], key="swallow_yn")
    
    if st.button("Next", key="swallow_next"):
        add_to_conversation('bot', "Any difficulty swallowing?")
        add_to_conversation('user', swallow_response)
        
        if swallow_response == "No":
            st.session_state.symptoms['swallowing'] = {'present': False}
            st.session_state.step = 9
        else:
            st.session_state.symptoms['swallowing'] = {'present': True, 'details': {}}
            st.session_state.step = 8
        st.rerun()

elif st.session_state.step == 8:
    st.markdown("### Swallowing Details")
    display_conversation()
    
    st.write("🤖 **ChatReport:** Tell me more about the difficulty.")
    description = st.text_area("Description:", key="swallow_desc")
    
    st.write("🤖 **ChatReport:** What types of foods can you eat?")
    food_types = st.text_input("Food types (regular, soft, liquids, etc.):", key="food_types")
    
    if st.button("Next", key="swallow_det_next"):
        if description and food_types:
            add_to_conversation('bot', "Tell me more about the difficulty.")
            add_to_conversation('user', description)
            add_to_conversation('bot', "What types of foods can you eat?")
            add_to_conversation('user', food_types)
            st.session_state.symptoms['swallowing']['details']['description'] = description
            st.session_state.symptoms['swallowing']['details']['food_types'] = food_types
            st.session_state.step = 9
            st.rerun()
        else:
            st.warning("Please answer both questions.")

elif st.session_state.step == 9:
    st.markdown("### Nutrition")
    display_conversation()
    
    st.write("🤖 **ChatReport:** How has your appetite been?")
    appetite = st.text_input("Appetite:", key="appetite")
    
    st.write("🤖 **ChatReport:** Have you noticed any weight changes?")
    weight = st.text_input("Weight changes:", key="weight")
    
    if st.button("Next", key="nutrition_next"):
        if appetite and weight:
            add_to_conversation('bot', "How has your appetite been?")
            add_to_conversation('user', appetite)
            add_to_conversation('bot', "Have you noticed any weight changes?")
            add_to_conversation('user', weight)
            st.session_state.symptoms['nutrition'] = {
                'details': {
                    'appetite': appetite,
                    'weight': weight
                }
            }
            st.session_state.step = 10
            st.rerun()
        else:
            st.warning("Please answer both questions.")

elif st.session_state.step == 10:
    st.markdown("### Mood")
    display_conversation()
    
    st.write("🤖 **ChatReport:** How would you describe your mood lately?")
    mood = st.text_area("Mood description:", key="mood")
    
    concern_words = ['worried', 'anxious', 'depressed', 'sad', 'down', 'hopeless']
    show_sleep = any(word in mood.lower() for word in concern_words)
    
    sleep = None
    if show_sleep:
        st.write("🤖 **ChatReport:** Are you having trouble sleeping?")
        sleep = st.text_input("Sleep:", key="sleep")
    
    if st.button("Next", key="mood_next"):
        if mood:
            add_to_conversation('bot', "How would you describe your mood lately?")
            add_to_conversation('user', mood)
            
            mood_data = {'mood': mood}
            if show_sleep and sleep:
                add_to_conversation('bot', "Are you having trouble sleeping?")
                add_to_conversation('user', sleep)
                mood_data['sleep'] = sleep
            
            st.session_state.symptoms['mood'] = {'details': mood_data}
            st.session_state.step = 11
            st.rerun()
        else:
            st.warning("Please describe your mood.")

elif st.session_state.step == 11:
    st.markdown("### Final Notes")
    display_conversation()
    
    st.write("🤖 **ChatReport:** Is there anything else you'd like your doctor to know?")
    additional = st.text_area("Additional notes (optional):", key="additional")
    
    if st.button("Finish", key="finish"):
        if additional:
            add_to_conversation('bot', "Is there anything else you'd like your doctor to know?")
            add_to_conversation('user', additional)
            st.session_state.symptoms['additional_notes'] = additional
        
        add_to_conversation('bot', f"Thank you, {st.session_state.patient_name}! Your report is ready for your doctor.")
        st.session_state.step = 12
        st.rerun()

elif st.session_state.step == 12:
    st.markdown("### ✅ Report Complete!")
    st.success(f"Thank you, {st.session_state.patient_name}! Your report has been generated.")
    
    st.markdown("---")
    st.markdown(generate_report())
    st.markdown("---")
    
    if st.button("Start New Report"):
        # Reset everything
        st.session_state.step = 0
        st.session_state.symptoms = {}
        st.session_state.patient_name = ""
        st.session_state.conversation = []
        st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Progress")
    progress = st.session_state.step / 12
    st.progress(progress)
    st.write(f"Step {st.session_state.step} of 12")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.write("ChatReport helps you share symptom information with your healthcare provider before your appointment.")
    
    if st.session_state.step > 0:
        st.markdown("---")
        if st.button("🔄 Restart"):
            st.session_state.step = 0
            st.session_state.symptoms = {}
            st.session_state.patient_name = ""
            st.session_state.conversation = []
            st.rerun()
