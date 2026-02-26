import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Mini Survey", page_icon="📝", layout="centered")

st.title("📝 Mini Survey")
st.write("Please fill out this short survey. Your responses will be saved to a Google Sheet.")

# --- Connect to Google Sheets using Streamlit secrets ---
# Expecting:
# st.secrets["gcp_service_account"] -> full service account JSON
# st.secrets["gsheet_id"] -> the Google Sheet ID string
SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPE)
client = gspread.authorize(creds)
sheet = client.open_by_key(st.secrets["gsheet_id"]).worksheet("Form")

# --- Survey form ---
with st.form("survey_form", clear_on_submit=True):
    name = st.text_input("Your name (optional)", "")
    role = st.selectbox("Your role", ["Clinician", "Researcher", "Student", "Other"])
    rating = st.slider("How useful is this tool for you?", 1, 5, 3)
    feedback = st.text_area("Any comments or suggestions?")
    submitted = st.form_submit_button("Submit")

if submitted:
    # Attempt basic user agent / IP hints (best-effort; Streamlit Cloud restricts some headers)
    user_agent = st.experimental_user.get("browser", "unknown") if hasattr(st, "experimental_user") else "unknown"
    ip_guess = st.experimental_user.get("ip", "unknown") if hasattr(st, "experimental_user") else "unknown"

    row = [
        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        name.strip(),
        role,
        int(rating),
        feedback.strip(),
        user_agent,
        ip_guess,
    ]
    try:
        sheet.append_row(row, value_input_option="USER_ENTERED")
        st.success("Thanks! Your response has been recorded.")
    except Exception as e:
        st.error(f"Failed to save your response: {e}")

st.divider()

# --- Admin view (optional) ---
with st.expander("🔐 Admin: view responses"):
    try:
        data = sheet.get_all_records()
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No responses yet.")
    except Exception as e:
        st.error(f"Could not read the sheet: {e}")
