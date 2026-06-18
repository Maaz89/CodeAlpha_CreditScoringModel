import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configurations with a wide layout for a dashboard experience
st.set_page_config(
    page_title="Credit Scoring Dashboard", 
    page_icon="🏦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force clean, high-contrast text styling and custom background cards
st.markdown("""
    <style>
        /* Force main body background to light gray */
        .stApp {
            background-color: #f1f5f9 !important;
        }
        
        /* Force sidebar formatting */
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {
            color: #ffffff !important;
        }
        
        /* Custom Header Styling (Forcing Dark Typography on White Canvas) */
        .title-text {
            font-size: 2.8rem;
            font-weight: 800;
            color: #0f172a !important;
            margin-bottom: 0rem;
            padding-top: 1rem;
        }
        .subtitle-text {
            font-size: 1.15rem;
            color: #475569 !important;
            margin-bottom: 2rem;
        }
        
        /* Section Heading Styling */
        .section-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1e293b !important;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        /* Styling regular cards/containers white with a crisp shadow */
        div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            margin-bottom: 1rem;
        }
        
        /* Submit Button Styling */
        .stButton>button {
            background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 0.75rem 2.5rem !important;
            border-radius: 8px !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3);
            transition: all 0.3s ease;
            width: 100%;
            font-size: 1.1rem !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(5, 150, 105, 0.4);
            background: linear-gradient(135deg, #047857 0%, #065f46 100%) !important;
        }
        
        /* Input Labels styling fallback color override */
        label, div[data-testid="stWidgetLabel"] p {
            color: #334155 !important;
            font-weight: 600 !important;
        }
        
        /* Metric Box Enhancement */
        div[data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            color: #0f172a !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('credit_scoring_ensemble_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    ensemble_model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Please run 'python train.py' in your terminal first to generate your model files!")
    st.stop()

# --- HEADER SECTION ---
st.markdown('<p class="title-text">🏦 CapitalGuard AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Institutional Credit Scoring & Intelligent Risk Underwriting Dashboard</p>', unsafe_allow_html=True)

# --- SIDEBAR INFO ---
with st.sidebar:
    st.markdown("### 📋 Risk Parameters")
    st.info("This system utilizes an advanced Machine Learning Ensemble model to predict default probabilities based on historical client demographics and banking data.")
    st.markdown("---")
    st.caption("🔒 Secured End-to-End Encryption")
    st.caption("📅 Core Engine Version: 2026.1")

# --- UI FORMS & INPUTS ---
st.markdown('<p class="section-header">👤 Applicant Demographics & Financial Parameters</p>', unsafe_allow_html=True)

# Grid Layout 1
row1_col1, row1_col2, row1_col3 = st.columns(3)
with row1_col1:
    age = st.number_input("Age (Years)", min_value=18, max_value=100, value=35)
    sex = st.selectbox("Biological Sex", ["male", "female"])
with row1_col2:
    duration = st.number_input("Loan Term (Months)", min_value=1, max_value=100, value=24)
    housing = st.selectbox("Housing Asset Status", ["own", "rent", "free"])
with row1_col3:
    credit_amount = st.number_input("Requested Loan Principal ($)", min_value=100, max_value=50000, value=3000)
    job = st.selectbox("Employment Skill Tier (0-3)", [0, 1, 2, 3])

st.markdown('<p class="section-header">🏦 Liquid Asset Liquidity Profiles</p>', unsafe_allow_html=True)

# Grid Layout 2
row2_col1, row2_col2, row2_col3 = st.columns(3)
with row2_col1:
    saving_account = st.selectbox("Savings Account Liquidity Tier", ["none", "little", "moderate", "quite rich", "rich"])
with row2_col2:
    checking_account = st.selectbox("Checking Account Liquidity Tier", ["none", "little", "moderate", "rich"])
with row2_col3:
    purpose = st.selectbox("Capital Allocation Purpose", ['radio/TV', 'education', 'furniture/equipment', 'car', 'business', 'domestic appliances', 'repairs', 'vacation/others'])

st.markdown("<br>", unsafe_allow_html=True)

# --- EVALUATION ENGINE ---
_, center_btn_col, _ = st.columns([1, 2, 1])

with center_btn_col:
    evaluate_clicked = st.button("📊 RUN CREDIT ASSESSMENT", type="primary")

if evaluate_clicked:
    credit_per_month = credit_amount / duration
    
    # Matching dummy feature map array structure 
    feature_columns = [
        'Age', 'Job', 'Credit amount', 'Duration', 'Credit_per_month',
        'Sex_female', 'Sex_male', 'Housing_free', 'Housing_own', 'Housing_rent',
        'Saving accounts_little', 'Saving accounts_moderate', 'Saving accounts_none', 'Saving accounts_quite rich', 'Saving accounts_rich',
        'Checking account_little', 'Checking account_moderate', 'Checking account_none', 'Checking account_rich',
        'Purpose_business', 'Purpose_car', 'Purpose_domestic appliances', 'Purpose_education', 'Purpose_furniture/equipment', 'Purpose_radio/TV', 'Purpose_repairs', 'Purpose_vacation/others'
    ]
    
    processed_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Inserting Values 
    processed_df['Age'] = age
    processed_df['Job'] = job
    processed_df['Credit amount'] = credit_amount
    processed_df['Duration'] = duration
    processed_df['Credit_per_month'] = credit_per_month
    
    # Map Categorical Flag Metrics
    if f"Sex_{sex}" in processed_df.columns: processed_df[f"Sex_{sex}"] = 1
    if f"Housing_{housing}" in processed_df.columns: processed_df[f"Housing_{housing}"] = 1
    if f"Saving accounts_{saving_account}" in processed_df.columns: processed_df[f"Saving accounts_{saving_account}"] = 1
    if f"Checking account_{checking_account}" in processed_df.columns: processed_df[f"Checking account_{checking_account}"] = 1
    if f"Purpose_{purpose}" in processed_df.columns: processed_df[f"Purpose_{purpose}"] = 1
    
    # Standard Scaler Normalization mapping 
    num_cols = ['Age', 'Credit amount', 'Duration', 'Credit_per_month']
    processed_df[num_cols] = scaler.transform(processed_df[num_cols])
    
    # Compute Predictions
    prediction = ensemble_model.predict(processed_df)[0]
    risk_probability = ensemble_model.predict_proba(processed_df)[0][1] * 100

    st.markdown('<p class="section-header">📈 Automated Underwriting Verdict</p>', unsafe_allow_html=True)
    
    # Custom Result Output Presentation
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        if prediction == 0:
            st.markdown(f"""
                <div style="background-color: #ecfdf5; border-left: 6px solid #10b981; padding: 20px; border-radius: 8px;">
                    <h3 style="color: #065f46; margin: 0 0 10px 0;">🟢 CREDIT APPLICATION APPROVED</h3>
                    <p style="color: #047857; margin: 0; font-size: 1.1rem;">The applicant exhibits an optimal financial risk index profile. Expected performance falls cleanly within low-risk default guidelines.</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
                <div style="background-color: #fef2f2; border-left: 6px solid #ef4444; padding: 20px; border-radius: 8px;">
                    <h3 style="color: #991b1b; margin: 0 0 10px 0;">🔴 CREDIT APPLICATION REJECTED</h3>
                    <p style="color: #b91c1c; margin: 0; font-size: 1.1rem;">High credit exposure profile detected. Default warning flag tripped due to elevated risk parameters.</p>
                </div>
            """, unsafe_allow_html=True)
            
    with res_col2:
        delta_color_mode = "inverse" if prediction == 0 else "normal"
        st.metric(
            label="Calculated Default Risk Probability", 
            value=f"{risk_probability:.2f}%",
            delta=f"{'Low Risk Profile' if prediction == 0 else 'High Default Risk Profile'}",
            delta_color=delta_color_mode
        )