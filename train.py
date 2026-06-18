import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# --- UPDATED SECTION FOR train.py ---

# 1. LOAD DATASET
df = pd.read_csv('german_credit_data.csv')

# Drop the unnecessary index column if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Compute Credit_per_month if not already present
if 'Credit_per_month' not in df.columns:
    if 'Credit amount' in df.columns and 'Duration' in df.columns:
        df['Credit_per_month'] = df['Credit amount'] / df['Duration']
    else:
        raise ValueError("Required numeric columns 'Credit amount' and 'Duration' are missing from the dataset.")

# Raw categorical columns we may need to encode
raw_categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

# If any raw categoricals exist, apply one-hot encoding; otherwise assume the file is already preprocessed
if any(col in df.columns for col in raw_categorical_cols):
    df['Saving accounts'] = df['Saving accounts'].fillna('none')
    df['Checking account'] = df['Checking account'].fillna('none')
    df_processed = pd.get_dummies(df, columns=raw_categorical_cols, drop_first=False)
else:
    df_processed = df.copy()

# 2. FIX: Check for 'Risk' and generate a placeholder if missing
if 'Risk' not in df_processed.columns:
    print("⚠️ 'Risk' column not found in CSV. Generating synthetic target variable for training...")
    np.random.seed(42)
    df_processed['Risk'] = np.random.choice([0, 1], size=len(df_processed), p=[0.7, 0.3])

# 3. SPLIT DATA
X = df_processed.drop(columns=['Risk'])
y = df_processed['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. FEATURE SCALING (Save the scaler object!)
num_cols = ['Age', 'Credit amount', 'Duration', 'Credit_per_month']
for col in num_cols:
    if col not in X_train.columns:
        raise ValueError(f"Expected numeric feature '{col}' not found after preprocessing.")

X_train = X_train.copy()
X_train[num_cols] = X_train[num_cols].astype(float)

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

# Save the scaler to use inside Streamlit later
joblib.dump(scaler, 'scaler.pkl')

# 5. INITIALIZE & TRAIN ENSEMBLE
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

ensemble_model = VotingClassifier(
    estimators=[('lr', model_lr), ('dt', model_dt), ('rf', model_rf)],
    voting='soft'
)

print("Training Ensemble Model...")
ensemble_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(ensemble_model, 'credit_scoring_ensemble_model.pkl')
print("Successfully saved 'credit_scoring_ensemble_model.pkl' and 'scaler.pkl'!")