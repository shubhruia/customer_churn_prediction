import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load model and files
model = joblib.load('model/churn_model.pkl')
feature_columns = joblib.load('model/feature_columns.pkl')
best_thresh = float(open('model/best_threshold.txt').read())

# App setup
st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔄", layout="wide")
st.title("Customer Churn Prediction")

# Sidebar
with st.sidebar:
    menu = st.sidebar.selectbox("Choose Action", [
    "Single Prediction", "Batch Prediction", "Visual Analysis"
])

    st.markdown(f"""
    ### About the Model

    This churn prediction model was trained on a dataset of [telecom customer profiles](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) to identify users likely to stop using the service.

    **Model Type:** Gradient Boosting Classifier   
    **Target Variable:** `Churn` (binary classification)  
    **Optimized Threshold:** `{best_thresh:.2f}`

    ### Evaluation Metrics (test set)
    - ROC-AUC: 89.6%
    - Accuracy: 82.7%
    - Precision (Churn): 80%
    - Recall (Churn): 88%
    """)

# Preprocessing
def preprocess_input(df):
    df = df.copy()
    df.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')
    df_encoded = pd.get_dummies(df)
    return df_encoded.reindex(columns=feature_columns, fill_value=0)

# Prediction
def predict(df, threshold):
    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds, probs

# SHAP explanation
def show_shap(df_encoded):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_encoded)

    expected_value = explainer.expected_value
    shap_val = shap_values

    if isinstance(expected_value, list):
        expected_value = expected_value[1]
        shap_val = shap_values[1]

    st.subheader("Feature Importance (SHAP)")
    st_shap(shap.force_plot(expected_value, shap_val, df_encoded), height=300)

# Single Prediction Page
if menu == "Single Prediction":
    with st.form("input_form"):
        st.subheader("Enter Customer Details")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["Yes", "No"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure_group = st.selectbox("Tenure Group", ["0-12", "13-24", "25-48", "49-60", "61-72"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges", 0.0, 500.0)
        total_charges = st.number_input("Total Charges", 0.0, 10000.0)
        threshold_slider = st.slider("Prediction Threshold", 0.0, 1.0, best_thresh, 0.01)
        avg_customer_value = st.number_input("Estimated Lifetime Value ($)", 100, 10000, step=50)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'TenureGroup': tenure_group,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }])

        df_encoded = preprocess_input(input_data)
        pred, prob = predict(df_encoded, threshold_slider)
        st.markdown(f"### Churn Probability: `{prob[0]:.2f}`")

        if pred[0]:
            st.error("⚠️ There's a high chance this customer won't stick around.")
            st.warning(f"Potential loss: **${avg_customer_value:.2f}**")
        else:
            st.success("✅ This customer looks happy and likely to stay.")

        show_shap(df_encoded)

# Batch Prediction Page
elif menu == "Batch Prediction":
    st.subheader("Upload Customer CSV")
    file = st.file_uploader("Upload a CSV file", type="csv")
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, best_thresh, 0.01)

    if file:
        df = pd.read_csv(file)
        df_encoded = preprocess_input(df)
        preds, probs = predict(df_encoded, threshold)

        result_df = df.copy()
        result_df['Churn Probability'] = probs
        result_df['Predicted Churn'] = preds

        st.dataframe(result_df.head())
        st.download_button("Download Predictions", result_df.to_csv(index=False), "predictions.csv")

        st.subheader("Batch Summary")
        churn_rate = result_df['Predicted Churn'].mean()
        st.metric("Churn Rate", f"{churn_rate:.2%}")

        total_customers = len(result_df)
        num_churned = result_df['Predicted Churn'].sum()
        avg_prob = result_df['Churn Probability'].mean()

        st.markdown("#### Additional Insights")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", total_customers)
        col2.metric("Predicted to Churn", num_churned)
        col3.metric("Avg Churn Probability", f"{avg_prob:.2%}")

# Visualization  Page
elif menu == "Visual Analysis":
    st.subheader("Visual Insights")
    file = st.file_uploader("Upload CSV with Predictions", type="csv", key="viz")

    if file:
        df = pd.read_csv(file)

        if "Predicted Churn" not in df.columns:
            st.warning("Upload a CSV with `Predicted Churn` column.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Churn by Contract Type")
                st.bar_chart(df.groupby("Contract")["Predicted Churn"].mean())

            with col2:
                st.markdown("### Churn by Internet Service")
                st.bar_chart(df.groupby("InternetService")["Predicted Churn"].mean())

            st.markdown("### Monthly Charges Distribution")
            fig, ax = plt.subplots()
            sns.histplot(data=df, x="MonthlyCharges", hue="Predicted Churn", multiple="stack", ax=ax)
            st.pyplot(fig)
