import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
import streamlit as st
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="MediPredict",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SAFE UI STYLING
# Compatible with older Streamlit versions
# ============================================================
st.markdown("""
<style>
.stApp { background-color: #1A2238 ; }
.main .block-container { padding-top: 2rem; padding-left: 2.5rem; padding-right: 2.5rem; max-width: 1400px; }
section[data-testid="stSidebar"] { background-color: #0f172a; }
section[data-testid="stSidebar"] > div { background-color: #0f172a; }
h1 { color: Black !important; }
h2, h3 { color: #f5f7fb !important; }
.stButton > button { border-radius: 9px; font-weight: 600; min-height: 44px; }
div[data-testid="stMetric"] { background: grey; border: 1px solid #e2e8f0; padding: 15px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# diabetes_model = joblib.load("models/diabetes_model.sav")
diabetes_model = joblib.load("models/diabetes_model_5features.joblib")
heart_model = joblib.load("models/heart_disease_model.sav")
# parkinson_model = joblib.load("models/parkinsons_model.sav")
# Load the lung cancer prediction model
lung_cancer_model = joblib.load('models/lung_cancer_model.sav')

# Load the pre-trained model
breast_cancer_model = joblib.load('models/breast_cancer.sav')

# Load the pre-trained model
chronic_disease_model = joblib.load('models/chronic_model.sav')

# Load the hepatitis prediction model
hepatitis_model = joblib.load('models/hepititisc_model.sav')


liver_model = joblib.load('models/liver_model.sav')# Load the lung cancer prediction model



# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:18px 5px 22px 5px;">
        <div style="font-size:40px;">🏥</div>
        <h2 style="color:white !important; margin:5px 0;">MediPredict</h2>
        <p style="color:#94a3b8; font-size:13px;">AI-Powered Health Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=[
            'Disease Prediction',
            'Diabetes Prediction',
            'Heart disease Prediction',
            'Liver prediction',
            'Hepatitis prediction',
            'Lung Cancer Prediction',
            'Chronic Kidney prediction',
            
        ],
        icons=[
            'activity', 'droplet', 'heart-pulse', 'heart',
            'virus', 'lungs', 'person', 'clipboard2-pulse'
        ],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#0f172a"},
            "icon": {"color": "#94a3b8", "font-size": "17px"},
            "nav-link": {
                "font-size": "14px", "text-align": "left",
                "margin": "3px 0", "padding": "10px 12px",
                "border-radius": "8px", "color": "#cbd5e1",
                "--hover-color": "#1e293b"
            },
            "nav-link-selected": {"background-color": "#2563eb", "color": "white"}
        }
    )

# multiple disease prediction
if selected == 'Disease Prediction': 
    # Create disease class and load ML model
    disease_model = DiseaseModel()
    disease_model.load_xgboost('model/xgboost_model.json')

    # Title
    st.write('# Disease Prediction using Machine Learning')

    symptoms = st.multiselect('What are your symptoms?', options=disease_model.all_symptoms)

    X = prepare_symptoms_array(symptoms)

    # Trigger XGBoost model
    if st.button('Predict'): 
        # Run the model with the python script
        
        prediction, prob = disease_model.predict(X)
        st.write(f'## Disease: {prediction} with {prob*100:.2f}% probability')


        tab1, tab2= st.tabs(["Description", "Precautions"])

        with tab1:
            st.write(disease_model.describe_predicted_disease())

        with tab2:
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
                st.write(f'{i+1}. {precautions[i]}')




# ============================================================
# DIABETES PREDICTION
# ============================================================
if selected == 'Diabetes Prediction':

    st.markdown("""
    <div style="background:linear-gradient(90deg,#2563eb,#1d4ed8);padding:28px 32px;border-radius:14px;margin-bottom:28px;">
        <h1 style="color:white !important;margin:0;font-size:32px;">🩸 Diabetes Risk Assessment</h1>
        <p style="color:#dbeafe;margin:8px 0 0 0;font-size:15px;">
            Enter basic health information to estimate diabetes risk using machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👤 Personal Information")
    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Full Name", placeholder="Enter your name")

    with col2:
        Age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1)

    st.markdown("### 🩺 Health Measurements")
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input(
            "Number of Pregnancies", min_value=0, max_value=20, value=0, step=1
        )

    with col2:
        Glucose = st.number_input(
            "Glucose Level (mg/dL)", min_value=0, max_value=300, value=100, step=1
        )

    with col3:
        BloodPressure = st.number_input(
            "Blood Pressure (mm Hg)", min_value=0, max_value=200, value=80, step=1
        )

    col1, col2 = st.columns(2)

    with col1:
        Height = st.number_input(
            "Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.1
        )

    with col2:
        Weight = st.number_input(
            "Weight (kg)", min_value=20.0, max_value=250.0, value=65.0, step=0.1
        )

    BMI = Weight / ((Height / 100) ** 2)

    st.markdown("### 📊 Calculated BMI")
    bmi_col1, bmi_col2, bmi_col3 = st.columns([1, 2, 1])
    with bmi_col2:
        st.metric("Body Mass Index", f"{BMI:.2f}")

    st.markdown("### 🔍 Risk Assessment")

    if st.button("Run Diabetes Risk Assessment", type="primary"):
        diabetes_input = [[Pregnancies, Glucose, BloodPressure, BMI, Age]]
        diabetes_prediction = diabetes_model.predict(diabetes_input)

        if diabetes_prediction[0] == 1:
            diabetes_result = "The model estimates an elevated risk of diabetes."
            st.warning(f"⚠️ {name if name else 'User'} — {diabetes_result}")
            try:
                image = Image.open('positive.jpg')
                st.image(image, width=300)
            except Exception:
                pass
        else:
            diabetes_result = "The model estimates a lower risk of diabetes."
            st.success(f"✓ {name if name else 'User'} — {diabetes_result}")
            try:
                image = Image.open('negative.jpg')
                st.image(image, width=300)
            except Exception:
                pass

    st.markdown("""
    <div style="margin-top:22px;padding:12px 16px;background:#eef2f7;border-radius:8px;color:#475569;font-size:13px;">
        <b>Medical Disclaimer:</b> This application provides a machine-learning based risk estimate and is not a medical diagnosis. Please consult a qualified healthcare professional for medical advice.
    </div>
    """, unsafe_allow_html=True)


# Heart prediction page
if selected == 'Heart disease Prediction':

    st.markdown("""
    <div style="background:linear-gradient(90deg,#2563eb,#1d4ed8);padding:28px 32px;border-radius:14px;margin-bottom:28px;">
        <h1 style="color:white !important;margin:0;font-size:32px;">🫀 Heart Disease Risk Assessment</h1>
        <p style="color:#dbeafe;margin:8px 0 0 0;font-size:15px;">
            Enter cardiovascular health information to estimate heart disease risk using machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👤 Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", placeholder="Enter your name")
    with col2:
        age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1)

    st.markdown("### 🩺 Health Measurements")
    col1, col2, col3 = st.columns(3)

    with col1:
        sex = 0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male": sex = 1
        elif value == "female": sex = 0

    with col2:
        cp = 0
        display = ("typical angina", "atypical angina", "non — anginal pain", "asymptotic")
        options = list(range(len(display)))
        value = st.selectbox("Chest Pain Type", options, format_func=lambda x: display[x])
        if value == "typical angina": cp = 0
        elif value == "atypical angina": cp = 1
        elif value == "non — anginal pain": cp = 2
        elif value == "asymptotic": cp = 3

    with col3:
        trestbps = st.number_input("Resting Blood Pressure", min_value=0, value=120)

    with col1:
        chol = st.number_input("Serum Cholesterol", min_value=0, value=200)
    with col2:
        restecg = 0
        display = ("normal", "ST-T wave abnormality", "left ventricular hypertrophy")
        options = list(range(len(display)))
        value = st.selectbox("Resting ECG", options, format_func=lambda x: display[x])
        if value == "normal": restecg = 0
        elif value == "ST-T wave abnormality": restecg = 1
        elif value == "left ventricular hypertrophy": restecg = 2
    with col3:
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, value=150)

    with col1:
        oldpeak = st.number_input("ST Depression", min_value=0.0, value=1.0, step=0.1)
    with col2:
        slope = 0
        display = ("upsloping", "flat", "downsloping")
        options = list(range(len(display)))
        value = st.selectbox("Peak Exercise ST Segment", options, format_func=lambda x: display[x])
        if value == "upsloping": slope = 0
        elif value == "flat": slope = 1
        elif value == "downsloping": slope = 2
    with col3:
        ca = st.number_input("Major Vessels (0–3)", min_value=0, max_value=3, value=0, step=1)

    with col1:
        thal = 0
        display = ("normal", "fixed defect", "reversible defect")
        options = list(range(len(display)))
        value = st.selectbox("Thalassemia", options, format_func=lambda x: display[x])
        if value == "normal": thal = 0
        elif value == "fixed defect": thal = 1
        elif value == "reversible defect": thal = 2
    with col2:
        exang = 1 if st.checkbox("Exercise-induced Angina") else 0
    with col3:
        fbs = 1 if st.checkbox("Fasting Blood Sugar > 120 mg/dL") else 0

    st.markdown("### 🔍 Risk Assessment")
    if st.button("Run Heart Disease Risk Assessment", type="primary"):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        if heart_prediction[0] == 1:
            heart_dig = "The model estimates an elevated risk of heart disease."
            try: st.image(Image.open('positive.jpg'), width=300)
            except Exception: pass
            st.warning(f"⚠️ {name if name else 'User'} — {heart_dig}")
        else:
            heart_dig = "The model estimates a lower risk of heart disease."
            try: st.image(Image.open('negative.jpg'), width=300)
            except Exception: pass
            st.success(f"✓ {name if name else 'User'} — {heart_dig}")

    st.markdown("""
    <div style="margin-top:22px;padding:12px 16px;background:#eef2f7;border-radius:8px;color:#475569;font-size:13px;">
        <b>Medical Disclaimer:</b> This application provides a machine-learning based risk estimate and is not a medical diagnosis. Please consult a qualified healthcare professional for medical advice.
    </div>
    """, unsafe_allow_html=True)



# Load the dataset
lung_cancer_data = pd.read_csv('data/lung_cancer.csv')

# Convert 'M' to 0 and 'F' to 1 in the 'GENDER' column
lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].map({'M': 'Male', 'F': 'Female'})

# Lung Cancer prediction page
if selected == 'Lung Cancer Prediction':

    st.markdown("""
    <div style="background:linear-gradient(90deg,#2563eb,#1d4ed8);padding:28px 32px;border-radius:14px;margin-bottom:28px;">
        <h1 style="color:white !important;margin:0;font-size:32px;">🫁 Lung Cancer Risk Assessment</h1>
        <p style="color:#dbeafe;margin:8px 0 0 0;font-size:15px;">
            Enter health and lifestyle information for a machine-learning based risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👤 Personal Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Full Name", placeholder="Enter your name")
    with col2:
        gender = st.selectbox("Gender", lung_cancer_data['GENDER'].unique())
    with col3:
        age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1)

    st.markdown("### 🩺 Health & Lifestyle Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        smoking = st.selectbox("Smoking", ['NO', 'YES'])
    with col2:
        yellow_fingers = st.selectbox("Yellow Fingers", ['NO', 'YES'])
    with col3:
        anxiety = st.selectbox("Anxiety", ['NO', 'YES'])
    with col1:
        peer_pressure = st.selectbox("Peer Pressure", ['NO', 'YES'])
    with col2:
        chronic_disease = st.selectbox("Chronic Disease", ['NO', 'YES'])
    with col3:
        fatigue = st.selectbox("Fatigue", ['NO', 'YES'])
    with col1:
        allergy = st.selectbox("Allergy", ['NO', 'YES'])
    with col2:
        wheezing = st.selectbox("Wheezing", ['NO', 'YES'])
    with col3:
        alcohol_consuming = st.selectbox("Alcohol Consuming", ['NO', 'YES'])
    with col1:
        coughing = st.selectbox("Coughing", ['NO', 'YES'])
    with col2:
        shortness_of_breath = st.selectbox("Shortness of Breath", ['NO', 'YES'])
    with col3:
        swallowing_difficulty = st.selectbox("Swallowing Difficulty", ['NO', 'YES'])
    with col1:
        chest_pain = st.selectbox("Chest Pain", ['NO', 'YES'])

    st.markdown("### 🔍 Risk Assessment")
    if st.button("Run Lung Cancer Risk Assessment", type="primary"):
        user_data = pd.DataFrame({
            'GENDER': [gender], 'AGE': [age], 'SMOKING': [smoking], 'YELLOW_FINGERS': [yellow_fingers],
            'ANXIETY': [anxiety], 'PEER_PRESSURE': [peer_pressure], 'CHRONICDISEASE': [chronic_disease],
            'FATIGUE': [fatigue], 'ALLERGY': [allergy], 'WHEEZING': [wheezing], 'ALCOHOLCONSUMING': [alcohol_consuming],
            'COUGHING': [coughing], 'SHORTNESSOFBREATH': [shortness_of_breath], 'SWALLOWINGDIFFICULTY': [swallowing_difficulty],
            'CHESTPAIN': [chest_pain]
        })
        user_data.replace({'NO': 1, 'YES': 2}, inplace=True)
        user_data.columns = user_data.columns.str.strip()
        numeric_columns = ['AGE', 'FATIGUE', 'ALLERGY', 'ALCOHOLCONSUMING', 'COUGHING', 'SHORTNESSOFBREATH']
        user_data[numeric_columns] = user_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
        cancer_prediction = lung_cancer_model.predict(user_data)
        if cancer_prediction[0] == 'YES':
            cancer_result = "The model estimates an elevated risk of lung cancer."
            try: st.image(Image.open('positive.jpg'), width=300)
            except Exception: pass
            st.warning(f"⚠️ {name if name else 'User'} — {cancer_result}")
        else:
            cancer_result = "The model estimates a lower risk of lung cancer."
            try: st.image(Image.open('negative.jpg'), width=300)
            except Exception: pass
            st.success(f"✓ {name if name else 'User'} — {cancer_result}")

    st.markdown("""<div style="margin-top:22px;padding:12px 16px;background:#eef2f7;border-radius:8px;color:#475569;font-size:13px;"><b>Medical Disclaimer:</b> This application provides a machine-learning based risk estimate and is not a medical diagnosis. Please consult a qualified healthcare professional for medical advice.</div>""", unsafe_allow_html=True)



# Liver prediction page
if selected == 'Liver prediction':

    st.markdown("""<div style="background:linear-gradient(90deg,#2563eb,#1d4ed8);padding:28px 32px;border-radius:14px;margin-bottom:28px;"><h1 style="color:white !important;margin:0;font-size:32px;">⚕️ Liver Disease Risk Assessment</h1><p style="color:#dbeafe;margin:8px 0 0 0;font-size:15px;">Enter personal and laboratory information for a machine-learning based risk assessment.</p></div>""", unsafe_allow_html=True)
    st.markdown("### 👤 Personal Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Full Name", placeholder="Enter your name")
    with col2:
        Sex = 0
        value = st.selectbox("Gender", ["male", "female"])
        Sex = 0 if value == "male" else 1
    with col3:
        age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1)

    st.markdown("### 🧪 Liver Health Measurements")
    col1, col2, col3 = st.columns(3)
    with col1: Total_Bilirubin = st.number_input("Total Bilirubin")
    with col2: Direct_Bilirubin = st.number_input("Direct Bilirubin")
    with col3: Alkaline_Phosphotase = st.number_input("Alkaline Phosphotase")
    with col1: Alamine_Aminotransferase = st.number_input("Alamine Aminotransferase")
    with col2: Aspartate_Aminotransferase = st.number_input("Aspartate Aminotransferase")
    with col3: Total_Protiens = st.number_input("Total Proteins")
    with col1: Albumin = st.number_input("Albumin")
    with col2: Albumin_and_Globulin_Ratio = st.number_input("Albumin and Globulin Ratio")

    st.markdown("### 🔍 Risk Assessment")
    if st.button("Run Liver Disease Risk Assessment", type="primary"):
        liver_prediction = liver_model.predict([[Sex,age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])
        if liver_prediction[0] == 1:
            liver_dig = "The model estimates an elevated risk of liver disease."
            try: st.image(Image.open('positive.jpg'), width=300)
            except Exception: pass
            st.warning(f"⚠️ {name if name else 'User'} — {liver_dig}")
        else:
            liver_dig = "The model estimates a lower risk of liver disease."
            try: st.image(Image.open('negative.jpg'), width=300)
            except Exception: pass
            st.success(f"✓ {name if name else 'User'} — {liver_dig}")
    st.markdown("""<div style="margin-top:22px;padding:12px 16px;background:#eef2f7;border-radius:8px;color:#475569;font-size:13px;"><b>Medical Disclaimer:</b> This application provides a machine-learning based risk estimate and is not a medical diagnosis. Please consult a qualified healthcare professional for medical advice.</div>""", unsafe_allow_html=True)



# Hepatitis prediction page
if selected == 'Hepatitis prediction':

    st.markdown("""<div style="background:linear-gradient(90deg,#2563eb,#1d4ed8);padding:28px 32px;border-radius:14px;margin-bottom:28px;"><h1 style="color:white !important;margin:0;font-size:32px;">🦠 Hepatitis Risk Assessment</h1><p style="color:#dbeafe;margin:8px 0 0 0;font-size:15px;">Enter laboratory and personal information for a machine-learning based risk assessment.</p></div>""", unsafe_allow_html=True)
    st.markdown("### 👤 Personal Information")
    col1, col2 = st.columns(2)
    with col1: name = st.text_input("Full Name", placeholder="Enter your name")
    with col2:
        age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1)
    st.markdown("### 🧪 Laboratory Measurements")
    col1, col2, col3 = st.columns(3)
    with col1: sex = st.selectbox("Gender", ["Male", "Female"]); sex = 1 if sex == "Male" else 2
    with col2: total_bilirubin = st.number_input("Total Bilirubin")
    with col3: direct_bilirubin = st.number_input("Direct Bilirubin")
    with col1: alkaline_phosphatase = st.number_input("Alkaline Phosphatase")
    with col2: alamine_aminotransferase = st.number_input("Alamine Aminotransferase")
    with col3: aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase")
    with col1: total_proteins = st.number_input("Total Proteins")
    with col2: albumin = st.number_input("Albumin")
    with col3: albumin_and_globulin_ratio = st.number_input("Albumin and Globulin Ratio")
    with col1: your_ggt_value = st.number_input("GGT Value")
    with col2: your_prot_value = st.number_input("PROT Value")

    st.markdown("### 🔍 Risk Assessment")
    if st.button("Run Hepatitis Risk Assessment", type="primary"):
        user_data = pd.DataFrame({'Age':[age],'Sex':[sex],'ALB':[total_bilirubin],'ALP':[direct_bilirubin],'ALT':[alkaline_phosphatase],'AST':[alamine_aminotransferase],'BIL':[aspartate_aminotransferase],'CHE':[total_proteins],'CHOL':[albumin],'CREA':[albumin_and_globulin_ratio],'GGT':[your_ggt_value],'PROT':[your_prot_value]})
        hepatitis_prediction = hepatitis_model.predict(user_data)
        if hepatitis_prediction[0] == 1:
            hepatitis_result = "The model estimates an elevated risk of hepatitis."
            try: st.image(Image.open('positive.jpg'), width=300)
            except Exception: pass
            st.warning(f"⚠️ {name if name else 'User'} — {hepatitis_result}")
        else:
            hepatitis_result = "The model estimates a lower risk of hepatitis."
            try: st.image(Image.open('negative.jpg'), width=300)
            except Exception: pass
            st.success(f"✓ {name if name else 'User'} — {hepatitis_result}")
    st.markdown("""<div style="margin-top:22px;padding:12px 16px;background:#eef2f7;border-radius:8px;color:#475569;font-size:13px;"><b>Medical Disclaimer:</b> This application provides a machine-learning based risk estimate and is not a medical diagnosis. Please consult a qualified healthcare professional for medical advice.</div>""", unsafe_allow_html=True)


# Chronic Kidney Disease Prediction Page
if selected == 'Chronic Kidney prediction':

    st.markdown("""<div style="background:linear-gradient(90deg,#2563eb,#1d4ed8);padding:28px 32px;border-radius:14px;margin-bottom:28px;"><h1 style="color:white !important;margin:0;font-size:32px;">🧪 Chronic Kidney Disease Risk Assessment</h1><p style="color:#dbeafe;margin:8px 0 0 0;font-size:15px;">Enter health and laboratory information for a machine-learning based risk assessment.</p></div>""", unsafe_allow_html=True)
    st.markdown("### 👤 Personal Information")
    col1, col2, col3 = st.columns(3)
    with col1: name = st.text_input("Full Name", placeholder="Enter your name")
    with col2: age = st.slider("Age", 1, 100, 25)
    with col3: bp = st.slider("Blood Pressure", 50, 200, 120)

    st.markdown("### 🧪 Kidney Health Measurements")
    col1, col2, col3 = st.columns(3)
    with col1: sg = st.slider("Specific Gravity", 1.0, 1.05, 1.02)
    with col2: al = st.slider("Albumin", 0, 5, 0)
    with col3: su = st.slider("Sugar", 0, 5, 0)
    with col1: rbc_label = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"]); rbc = 1 if rbc_label == "Normal" else 0
    with col2: pc_label = st.selectbox("Pus Cells", ["Normal", "Abnormal"]); pc = 1 if pc_label == "Normal" else 0
    with col3: pcc_label = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"]); pcc = 1 if pcc_label == "Present" else 0
    with col1: ba_label = st.selectbox("Bacteria", ["Present", "Not Present"]); ba = 1 if ba_label == "Present" else 0
    with col2: bgr = st.slider("Random Blood Glucose", 50, 200, 120)
    with col3: bu = st.slider("Blood Urea", 10, 200, 60)
    with col1: sc = st.slider("Serum Creatinine", 0, 10, 3)
    with col2: sod = st.slider("Sodium", 100, 200, 140)
    with col3: pot = st.slider("Potassium", 2, 7, 4)
    with col1: hemo = st.slider("Hemoglobin", 3, 17, 12)
    with col2: pcv = st.slider("Packed Cell Volume", 20, 60, 40)
    with col3: wc = st.slider("White Blood Cell Count", 2000, 20000, 10000)
    with col1: rc = st.slider("Red Blood Cell Count", 2, 8, 4)
    with col2: htn_label = st.selectbox("Hypertension", ["Yes", "No"]); htn = 1 if htn_label == "Yes" else 0
    with col3: dm_label = st.selectbox("Diabetes Mellitus", ["Yes", "No"]); dm = 1 if dm_label == "Yes" else 0
    with col1: cad_label = st.selectbox("Coronary Artery Disease", ["Yes", "No"]); cad = 1 if cad_label == "Yes" else 0
    with col2: appet_label = st.selectbox("Appetite", ["Good", "Poor"]); appet = 1 if appet_label == "Good" else 0
    with col3: pe_label = st.selectbox("Pedal Edema", ["Yes", "No"]); pe = 1 if pe_label == "Yes" else 0
    with col1: ane_label = st.selectbox("Anemia", ["Yes", "No"]); ane = 1 if ane_label == "Yes" else 0

    st.markdown("### 🔍 Risk Assessment")
    if st.button("Run Kidney Disease Risk Assessment", type="primary"):
        user_input = pd.DataFrame({'age':[age],'bp':[bp],'sg':[sg],'al':[al],'su':[su],'rbc':[rbc],'pc':[pc],'pcc':[pcc],'ba':[ba],'bgr':[bgr],'bu':[bu],'sc':[sc],'sod':[sod],'pot':[pot],'hemo':[hemo],'pcv':[pcv],'wc':[wc],'rc':[rc],'htn':[htn],'dm':[dm],'cad':[cad],'appet':[appet],'pe':[pe],'ane':[ane]})
        kidney_prediction = chronic_disease_model.predict(user_input)
        if kidney_prediction[0] == 1:
            kidney_result = "The model estimates an elevated risk of chronic kidney disease."
            try: st.image(Image.open('positive.jpg'), width=300)
            except Exception: pass
            st.warning(f"⚠️ {name if name else 'User'} — {kidney_result}")
        else:
            kidney_result = "The model estimates a lower risk of chronic kidney disease."
            try: st.image(Image.open('negative.jpg'), width=300)
            except Exception: pass
            st.success(f"✓ {name if name else 'User'} — {kidney_result}")
    st.markdown("""<div style="margin-top:22px;padding:12px 16px;background:#eef2f7;border-radius:8px;color:#475569;font-size:13px;"><b>Medical Disclaimer:</b> This application provides a machine-learning based risk estimate and is not a medical diagnosis. Please consult a qualified healthcare professional for medical advice.</div>""", unsafe_allow_html=True)


 