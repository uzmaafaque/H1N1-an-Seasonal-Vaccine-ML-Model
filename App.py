# ------------------------
# Vaccine Prediction App 💉 (Full)
# ------------------------

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# ------------------------
# App Title
# ------------------------
st.title("Vaccine Uptake Prediction App 💉")
st.write("Predict likelihood of getting H1N1 or Seasonal vaccine using Logistic Regression or Random Forest.")

# ------------------------
# Load Data
# ------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Vaccines.csv")

df = load_data()

# ------------------------
# Vaccine Selection
# ------------------------
vaccine_choice = st.sidebar.selectbox("Select Vaccine to Predict", ['h1n1_vaccine', 'seasonal_vaccine'])
st.write(f"You selected: **{vaccine_choice}**")

# ------------------------
# Define Columns
# ------------------------
all_categorical_cols = df.select_dtypes(include='object').columns.tolist()
all_categorical_cols = [c for c in all_categorical_cols if c != vaccine_choice]

numeric_cols = [c for c in df.columns if c not in all_categorical_cols + [vaccine_choice]]

# Key categorical features for user input
key_categorical = [
    'age_group','education','race','sex','income_poverty',
    'marital_status','rent_or_own','employment_status'
]

# ------------------------
# Sidebar: User chooses which columns to answer
# ------------------------
st.sidebar.title("Choose Input Type")
input_type = st.sidebar.radio("Which columns do you want to answer?", ["Categorical Columns", "Other Columns"])

user_data = {}

if input_type == "Categorical Columns":
    st.sidebar.subheader("Provide values for categorical features")
    for col in key_categorical:
        options = sorted(df[col].dropna().unique())
        user_data[col] = [st.sidebar.selectbox(f"{col}", options)]
else:
    st.sidebar.subheader("Provide values for other columns")
    for col in numeric_cols:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        user_data[col] = [st.sidebar.slider(f"{col}", min_value=min_val, max_value=max_val, value=(min_val+max_val)/2)]

input_df = pd.DataFrame(user_data)

# ------------------------
# One-Hot Encoding for all categorical columns
# ------------------------
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = df[all_categorical_cols]
X_encoded = encoder.fit_transform(X_cat)

# Combine with numeric columns if any
if numeric_cols:
    X_numeric = df[numeric_cols].to_numpy()
    X_full = np.hstack([X_encoded, X_numeric])
else:
    X_full = X_encoded

y = df[vaccine_choice]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

# ------------------------
# Align input_df with training columns
# ------------------------
# Fill missing categorical columns
for col in all_categorical_cols:
    if col not in input_df.columns:
        input_df[col] = ""  # default category

# Fill missing numeric columns
for col in numeric_cols:
    if col not in input_df.columns:
        input_df[col] = df[col].median()

# Reorder categorical columns and transform
input_encoded_cat = encoder.transform(input_df[all_categorical_cols])

# Combine with numeric
if numeric_cols:
    input_numeric = input_df[numeric_cols].to_numpy()
    input_encoded = np.hstack([input_encoded_cat, input_numeric])
else:
    input_encoded = input_encoded_cat

# ------------------------
# Train Models
# ------------------------
logreg = LogisticRegression(max_iter=500, class_weight='balanced')
logreg.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# ------------------------
# Predictions
# ------------------------
def get_likelihood_color(prob):
    if prob >= 0.5:
        return "Highly Likely", "green"
    elif prob >= 0.3:
        return "Moderate Likely", "orange"
    else:
        return "Unlikely", "red"

# Logistic Regression
pred_log = logreg.predict(input_encoded)[0]
proba_log = logreg.predict_proba(input_encoded)[0][1]
likelihood_log, color_log = get_likelihood_color(proba_log)

# Random Forest
pred_rf = rf.predict(input_encoded)[0]
proba_rf = rf.predict_proba(input_encoded)[0][1]
likelihood_rf, color_rf = get_likelihood_color(proba_rf)

# ------------------------
# Tabs
# ------------------------
tabs = st.tabs(["Logistic Regression", "Random Forest", "Comparison"])

with tabs[0]:
    st.header("📌 Logistic Regression")
    st.write(f"**Model Accuracy:** {accuracy_score(y_test, logreg.predict(X_test)):.2f}")
    st.markdown(f"**Uptake Prediction:** {'Yes' if pred_log==1 else 'No'}")
    st.markdown(f"**Likelihood:** <span style='color:{color_log}; font-weight:bold'>{likelihood_log}</span>", unsafe_allow_html=True)
    st.progress(int(proba_log*100))
    st.markdown(f"<span style='color:{color_log}; font-weight:bold'>{proba_log*100:.1f}%</span>", unsafe_allow_html=True)

with tabs[1]:
    st.header("🌳 Random Forest")
    st.write(f"**Model Accuracy:** {accuracy_score(y_test, rf.predict(X_test)):.2f}")
    st.markdown(f"**Uptake Prediction:** {'Yes' if pred_rf==1 else 'No'}")
    st.markdown(f"**Likelihood:** <span style='color:{color_rf}; font-weight:bold'>{likelihood_rf}</span>", unsafe_allow_html=True)
    st.progress(int(proba_rf*100))
    st.markdown(f"<span style='color:{color_rf}; font-weight:bold'>{proba_rf*100:.1f}%</span>", unsafe_allow_html=True)

with tabs[2]:
    st.header("🔍 Model Comparison")
    def get_color(prob):
        if prob >= 0.5: return 'green'
        elif prob >= 0.3: return 'orange'
        else: return 'red'

    st.write("**Predicted Probability Comparison:**")
    st.markdown("**Logistic Regression:**")
    st.progress(int(proba_log*100))
    st.markdown(f"<span style='color:{get_color(proba_log)}; font-weight:bold'>{proba_log*100:.1f}%</span>", unsafe_allow_html=True)

    st.markdown("**Random Forest:**")
    st.progress(int(proba_rf*100))
    st.markdown(f"<span style='color:{get_color(proba_rf)}; font-weight:bold'>{proba_rf*100:.1f}%</span>", unsafe_allow_html=True)

    comparison_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [accuracy_score(y_test, logreg.predict(X_test)), accuracy_score(y_test, rf.predict(X_test))],
        "Uptake Prediction": ['Yes' if pred_log==1 else 'No', 'Yes' if pred_rf==1 else 'No'],
        "Probability (%)": [f"{proba_log*100:.1f}%", f"{proba_rf*100:.1f}%"],
        "Likelihood": [likelihood_log, likelihood_rf]
    })
    st.subheader("Tabular Comparison")
    st.dataframe(comparison_df.style.applymap(lambda x: 'color: green' if x=="Highly Likely" else ('color: orange' if x=="Moderate Likely" else 'color: red'), subset=['Likelihood']))
