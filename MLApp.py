# ------------------------
# Vaccine Prediction App (CSV + Feature Engineering + Tuned Random Forest)
# ------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ------------------------
# Segmentation Function
# ------------------------
def get_segment(prob):
    if prob >= 0.25:
        return "High Likelihood", "green"
    elif prob >= 0.1:
        return "Medium Likelihood", "orange"
    else:
        return "Low Likelihood", "red"

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Vaccine Prediction", layout="wide")
st.title("💉 Vaccine Uptake Prediction App")

# ------------------------
# Load CSV
# ------------------------
csv_file = st.text_input("Enter CSV file name", "Vaccines.csv")

try:
    df = pd.read_csv(csv_file)
    st.session_state.df = df
    st.success(f"✅ Loaded data from {csv_file}")
except FileNotFoundError:
    st.error(f"❌ Could not find {csv_file}. Please check the file path.")
    st.stop()

df = st.session_state.get("df", None)

if df is not None:
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(), height=200)

    # ------------------------
    # Target Selection
    # ------------------------
    target_options = ["h1n1_vaccine", "seasonal_vaccine"]
    target = st.selectbox("Select Target Variable", target_options)

    if "respondent_id" in df.columns:
        df = df.drop("respondent_id", axis=1)

    # ------------------------
    # Feature Engineering
    # ------------------------
    cat_features = ["age_group", "education", "race", "sex",
                    "income_poverty", "marital_status", "rent_or_own", "employment_status"]

    # Interaction features
    df['age_income'] = df['age_group'].astype(str) + "_" + df['income_poverty'].astype(str)
    df['emp_edu'] = df['employment_status'].astype(str) + "_" + df['education'].astype(str)

    all_features = cat_features + ['age_income', 'emp_edu']

    # Encode categorical features
    le_dict = {}
    for col in all_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    X = df[all_features]
    y = df[target]

    # ------------------------
    # Train/Test Split
    # ------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ------------------------
    # Logistic Regression
    # ------------------------
    logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
    logreg.fit(X_train, y_train)

    # ------------------------
    # Random Forest with Hyperparameter Tuning
    # ------------------------
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'n_estimators': [200, 400, 600],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', None]
    }
    rf_search = RandomizedSearchCV(
        rf, param_distributions=param_grid, n_iter=20, cv=5,
        scoring='roc_auc', n_jobs=-1, random_state=42
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

    # ------------------------
    # Tabs
    # ------------------------
    tabs = st.tabs(["Logistic Regression", "Random Forest Classification"])

    # ------------------------
    # Tab 1: Logistic Regression
    # ------------------------
    with tabs[0]:
        st.header("📌 Logistic Regression Classification")
        y_pred = logreg.predict(X_test)
        y_proba = logreg.predict_proba(X_test)[:,1]

        st.subheader("Model Performance")
        st.dataframe(pd.DataFrame({
            "Accuracy": [accuracy_score(y_test, y_pred)],
            "ROC-AUC": [roc_auc_score(y_test, y_proba)]
        }), height=80)

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Top Feature Coefficients")
        coef_df = pd.DataFrame({
            "feature": X.columns,
            "coefficient": np.abs(logreg.coef_[0])
        }).sort_values(by="coefficient", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(4,2))
        sns.barplot(x="coefficient", y="feature", data=coef_df, palette="viridis", ax=ax)
        ax.set_title("Top 10 Logistic Regression Features")
        st.pyplot(fig)

        st.subheader("User Input Prediction")
        with st.form("logreg_form"):
            user_input = {}
            for col in all_features:
                if col in le_dict:
                    options = list(le_dict[col].classes_)
                    choice = st.selectbox(f"{col}", options)
                    user_input[col] = le_dict[col].transform([choice])[0]
            submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([user_input])
            input_df = input_df[all_features]
            pred = logreg.predict(input_df)[0]
            proba = logreg.predict_proba(input_df)[0][1]
            segment_label, segment_color = get_segment(proba)
            st.success(f"Predicted {target}: {'Yes' if pred==1 else 'No'}")
            st.markdown(f"**Prediction Confidence:** {proba:.2f}")
            st.markdown(f"**User Segment:** <span style='color:{segment_color}; font-weight:bold'>{segment_label}</span>", unsafe_allow_html=True)

    # ------------------------
    # Tab 2: Random Forest Classification
    # ------------------------
    with tabs[1]:
        st.header("🌳 Random Forest Classification")
        y_pred = best_rf.predict(X_test)
        y_proba = best_rf.predict_proba(X_test)[:,1]

        st.subheader("Model Performance")
        st.dataframe(pd.DataFrame({
            "Accuracy": [accuracy_score(y_test, y_pred)],
            "ROC-AUC": [roc_auc_score(y_test, y_proba)]
        }), height=80)

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Top Feature Importances")
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": best_rf.feature_importances_
        }).sort_values(by="importance", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(4,2))
        sns.barplot(x="importance", y="feature", data=importance_df, palette="viridis", ax=ax)
        ax.set_title("Top 10 Random Forest Features")
        st.pyplot(fig)

        st.subheader("User Input Prediction")
        with st.form("rf_form"):
            user_input = {}
            for col in all_features:
                if col in le_dict:
                    options = list(le_dict[col].classes_)
                    choice = st.selectbox(f"{col}", options)
                    user_input[col] = le_dict[col].transform([choice])[0]
            submitted = st.form_submit_button("Predict RF")

        if submitted:
            input_df = pd.DataFrame([user_input])
            input_df = all_features
            input_df = pd.DataFrame([user_input])
            pred = best_rf.predict(input_df)[0]
            proba = best_rf.predict_proba(input_df)[0][1]
            segment_label, segment_color = get_segment(proba)
            st.success(f"Predicted {target}: {'Yes' if pred==1 else 'No'}")
            st.markdown(f"**Prediction Confidence:** {proba:.2f}")
            st.markdown(f"**User Segment:** <span style='color:{segment_color}; font-weight:bold'>{segment_label}</span>", unsafe_allow_html=True)
