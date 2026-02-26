# Simple Survey to Google Sheets (Streamlit)

- Hosted free on Streamlit Community Cloud
- Saves rows to a Google Sheet via a service account
- Code: `app.py`

## Deploy
1) Create a Google Sheet and note its ID (the big string in the URL between `/d/` and `/edit`).
2) Create a Google Cloud service account (no billing needed), enable Google Sheets API, and download its JSON key.
3) In Streamlit Cloud, add `Secrets`:
