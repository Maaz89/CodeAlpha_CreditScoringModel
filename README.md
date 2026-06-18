# 🏦 CapitalGuard AI: Institutional Credit Scoring & Underwriting System

CapitalGuard AI is a high-performance, telemetry-driven financial risk evaluation dashboard built with **Streamlit**, **Scikit-Learn**, and **Pandas**. The system processes granular applicant demographics, financial histories, and asset liquidity profiles to compute instant default risk probabilities using an advanced machine learning ensemble backend.

Designed with a clean, premium fintech aesthetic, the dashboard provides interactive risk categorization, instant data normalization, and visual automated underwriting verdicts for credit risk officers.

---

## 🚀 Features

* **Ensemble ML Engine:** Integrates a robust classifier architecture to generate fine-grained default likelihood percentages.
* **Automated Feature Preprocessing:** Dynamically standardizes numerical features and handles dummy variable categorical mapping in real-time.
* **Premium Fintech UI:** Optimized high-contrast interface featuring card-based data inputs, streamlined forms, and visual metric callouts.
* **Robust Diagnostic Safety:** Gracefully manages environment states with user-friendly warnings if pipeline artifacts are missing.

---

## 📁 Repository Structure

```text
├── .streamlit
├── app.py                  # Core Streamlit UI & implementation file
├── train.py                # Model training, serialization, and export pipeline
├── scaler.pkl              # Serialized Scikit-Learn standard scaler object
├── credit_scoring_ensemble_model.pkl # Serialized ensemble model artifact
└── README.md               # Project documentation

🛠️ Installation & SetupEnsure you have Python 3.9+ installed on your local environment.
1. Clone the Repository:
Bashgit clone [https://github.com/YOUR_USERNAME/capitalguard-ai.git](https://github.com/YOUR_USERNAME/capitalguard-ai.git)
cd capitalguard-ai

2. Install Required Dependencies:
Install the required data science stack:Bashpip install streamlit pandas numpy scikit-learn joblib

3. Generate Model ArtifactsBefore:
launching the user interface, initialize and run the automated training pipeline to save the required scaler and ensemble binaries:
Bashpython train.py

4. Run the DashboardExecute the Streamlit server to deploy the local application instance:Bashstreamlit run app.py

📊 Core Data ArchitectureThe underwriting model expects and processes a structured 27-feature matrix mapping the following variables:

Feature Category                    Technical Inputs Map
Numerical FeaturesAge,        Job, Credit Amount, Duration, Credit Per Month
Demographics                  Biological Sex (Sex_male, Sex_female)
Asset Metadata                Housing Status (own, rent, free)
Liquidity ProfilesSavings     Account & Checking Account Tiers (none, little, moderate, rich)
Capital Allocation            Loan Purposes (business, car, education, furniture, radio/TV, etc.)

🔒 Security & ComplianceData Isolation:
All data inputs remain ephemeral inside memory contexts during run execution and are not persisted to unauthorized backends.
Version Control: System operates under Engine Track Core Version 2026.1.
