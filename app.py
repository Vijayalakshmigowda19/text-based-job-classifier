import streamlit as st
import fitz  # PyMuPDF
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Required qualifications
required_qualifications = {
    "data analysis": ["data analysis", "data analytics", "analyzing data"],
    "sql": ["sql", "mysql", "postgresql"],
    "data visualization": ["data visualization", "visualize data", "power bi", "matplotlib", "seaborn", "tableau"]
}

# Function to check qualifications
def check_qualifications(text):
    text = text.lower()
    missing = []
    for key, keywords in required_qualifications.items():
        if not any(keyword in text for keyword in keywords):
            missing.append(key)
    return len(missing) == 0, missing

# Function to extract PDF text
def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

# Streamlit UI
st.set_page_config(page_title="Job Role Classifier", layout="centered")
st.title("🔍 Job Role Classifier")

st.markdown("### Step 1: Choose how you want to input your data")
input_option = st.radio("Select input method:", ("Upload Resume (PDF)", "Enter Job Description (Text)"))

text_input = ""

# Handle input option
if input_option == "Upload Resume (PDF)":
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])
    if uploaded_file:
        text_input = extract_text_from_pdf(uploaded_file)

elif input_option == "Enter Job Description (Text)":
    text_input = st.text_area("Paste the job description here", height=200)

# Predict button
if st.button("Predict"):
    if not text_input.strip():
        st.warning("⚠️ Please enter job description or upload resume.")
    else:
        qualified, missing_skills = check_qualifications(text_input)
        if not qualified:
            st.error("❌ Rejected: Missing required qualifications.")
            st.markdown(f"**Missing:** `{', '.join(missing_skills)}`")
            st.markdown("**Required:** `data analysis`, `sql`, `data visualization`")
        else:
            predicted_role = model.predict([text_input])[0]
            st.success(f"✅ Accepted. Predicted Role: **{predicted_role}**")
