import streamlit as st
import pandas as pd
import pickle
import os
import google.generativeai as genai
from dotenv import load_dotenv
from fpdf import FPDF
import fitz  # PyMuPDF
from PIL import Image

# Load ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Gemini API
load_dotenv("ini.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

# Define the required columns based on diabetes.ipynb
columns = [
    'HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump',
    'AnyHealthcare', 'Sex', 'Age'
]

# Suggestion function
def generate_suggestion(data, prediction):
    input_text = ", ".join([f"{col}: {val}" for col, val in zip(columns, data)])
    if prediction == 0:
        return "You are not diabetic, but maintain a healthy lifestyle."
    elif prediction == 1:
        return "You are Pre-Diabetic. Maintain Healthy Life Style."
    else:
        input_string = f"Provide a suggestion for a diabetic patient with data: {input_text}"
        response = gemini_model.generate_content(input_string)
        return response.text

# PDF generator
def generate_full_report(patient_data, prediction, suggestion, ehr_summary=None, ehr_qna=None, ehr_filename=None):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Patient Prediction & EHR Summary Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Patient Information", ln=True)
    pdf.set_font("Arial", '', 12)
    for key, value in patient_data.items():
        pdf.cell(0, 10, f"{key}: {value}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Prediction Result", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Prediction: {prediction}", ln=True)
    pdf.multi_cell(0, 10, f"Gemini Suggestion:\n{suggestion}")
    pdf.ln(5)

    if ehr_summary:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "EHR Summary", ln=True)
        pdf.set_font("Arial", '', 12)
        if ehr_filename:
            pdf.cell(0, 10, f"File: {ehr_filename}", ln=True)
        pdf.multi_cell(0, 10, ehr_summary)
        pdf.ln(5)

    if ehr_qna:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Questions & Answers", ln=True)
        pdf.set_font("Arial", '', 12)
        for i, qna in enumerate(ehr_qna, 1):
            pdf.multi_cell(0, 10, f"Q{i}: {qna['question']}")
            pdf.multi_cell(0, 10, f"A{i}: {qna['answer']}")
            pdf.ln(3)

    output_path = "patient_summary_report.pdf"
    pdf.output(output_path)
    return output_path

# Process uploaded image
def process_image(uploaded_img):
    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        bytes_data = uploaded_img.getvalue()
        image_parts = [{"mime_type": uploaded_img.type, "data": bytes_data}]
        return img, image_parts
    return None, None

# Get calorie estimation
def get_calorie_estimate(image_parts, user_prompt="Estimate calories in this image"):
    if not image_parts:
        return "No image provided."
    prompt = "You are a nutritionist. Estimate the approximate calories in this image. Provide the food names and total estimated calories."
    try:
        response = gemini_model.generate_content([prompt, image_parts[0], user_prompt])
        return response.text
    except Exception as e:
        return f"Error in Gemini API: {e}"

# Reset session function
def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.page = "form"

# Session-based page routing
if "page" not in st.session_state:
    st.session_state.page = "form"

# Step 1 - Form
if st.session_state.page == "form":
    st.title("🩺 Step 1: Enter Patient Details")
    highbp = st.selectbox("High Blood Pressure?", ["No", "Yes"])
    highchol = st.selectbox("High Cholesterol?", ["No", "Yes"])
    bmi = st.number_input("BMI", 10.0, 100.0, 25.0)
    smoker = st.selectbox("Smoker?", ["No", "Yes"])
    stroke = st.selectbox("Stroke?", ["No", "Yes"])
    heart_attack = st.selectbox("Heart Disease or Attack?", ["No", "Yes"])
    phys_activity = st.selectbox("Physically Active?", ["No", "Yes"])
    heavy_alcohol = st.selectbox("Heavy Alcohol Consumption?", ["No", "Yes"])
    healthcare = st.selectbox("Any Healthcare Access?", ["No", "Yes"])
    sex = st.selectbox("Sex", ["Female", "Male"])
    age = st.slider("Age", 0, 120, 10)

    if st.button("Continue ➡️"):
        st.session_state.input_data = [
            1 if highbp == "Yes" else 0,
            1 if highchol == "Yes" else 0,
            bmi,
            1 if smoker == "Yes" else 0,
            1 if stroke == "Yes" else 0,
            1 if heart_attack == "Yes" else 0,
            1 if phys_activity == "Yes" else 0,
            1 if heavy_alcohol == "Yes" else 0,
            1 if healthcare == "Yes" else 0,
            1 if sex == "Male" else 0,
            age
        ]
        st.session_state.page = "action"
        st.rerun()

# Step 2 - Action Page
elif st.session_state.page == "action":
    st.title("⚙️ Step 2: Predict or Explore")

    if st.button("🧠 Predict & Suggest"):
        st.session_state.page = "result"
        st.rerun()

    if st.button("📄 EHR Summary"):
        st.session_state.page = "ehr"
        st.rerun()

    if st.button("🍽️ Calorie Counter"):
        st.session_state.page = "calories"
        st.rerun()

    if st.button("⬅️ Back"):
        st.session_state.page = "form"
        st.rerun()

# Step 3 - Prediction Result
elif st.session_state.page == "result":
    st.title("📊 Step 3: Prediction & Suggestions")

    input_data = st.session_state.input_data
    input_df = pd.DataFrame([input_data], columns=columns)
    model.monotonic_cst = None
    prediction = model.predict(input_df)[0]

    if prediction == 0:
        prediction_text = "Non-Diabetic"
    elif prediction == 1:
        prediction_text = "Pre-Diabetic"
    else:
        prediction_text = "Diabetic"

    st.success(f"Prediction: {prediction_text}")

    suggestion = generate_suggestion(input_data, prediction)
    st.info(suggestion)

    if st.button("📥 Download Full Report"):
        patient_data = dict(zip(columns, input_data))
        report_path = generate_full_report(
            patient_data=patient_data,
            prediction=prediction_text,
            suggestion=suggestion
        )
        with open(report_path, "rb") as f:
            st.download_button("⬇️ Click to Download PDF", data=f, file_name="Patient_Report.pdf")

    col1, col2 = st.columns(2)
    with col1:
        st.button("🔁 Start Over", on_click=reset_session)
    with col2:
        st.button("⬅️ Back to Actions", on_click=lambda: st.session_state.update({"page": "action"}))

# Step 4 - EHR Summary
elif st.session_state.page == "ehr":
    st.title("📄 Step 3: EHR Summary & Chat")

    uploaded_file = st.file_uploader("📑 Upload Medical Report (PDF)", type=["pdf"])
    if uploaded_file:
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())

        doc = fitz.open("uploaded.pdf")
        text = "\n".join([page.get_text() for page in doc])

        with st.spinner("Summarizing with Gemini..."):
            summary_prompt = f"Summarize this medical report for a doctor:\n\n{text}"
            summary = gemini_model.generate_content(summary_prompt).text
            st.session_state.ehr_summary = summary
            st.session_state.ehr_text = text

        st.subheader("📋 Summary")
        st.write(summary)

    if "ehr_text" in st.session_state:
        question = st.text_input("💬 Ask about your report:")
        if question:
            full_prompt = f"{st.session_state.ehr_text}\n\nQuestion: {question}"
            answer = gemini_model.generate_content(full_prompt).text

            if "ehr_qna" not in st.session_state:
                st.session_state.ehr_qna = []
            st.session_state.ehr_qna.append({"question": question, "answer": answer})

            st.write("🧠", answer)

    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅️ Back to Actions", on_click=lambda: st.session_state.update({"page": "action"}))
    with col2:
        st.button("🔁 Start Over", on_click=reset_session)

# Step 5 - Calorie Counter
elif st.session_state.page == "calories":
    st.title("🍽️ Step 3: Calorie Estimation from Food Image")

    uploaded_img = st.file_uploader("Upload an image of your meal 🍱", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        img, image_parts = process_image(uploaded_img)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Estimating calories using Gemini..."):
            calorie_response = get_calorie_estimate(image_parts)
            st.success("Calorie Estimate:")
            st.write(calorie_response)

    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅️ Back to Actions", on_click=lambda: st.session_state.update({"page": "action"}))
    with col2:
        st.button("🔁 Start Over", on_click=reset_session)