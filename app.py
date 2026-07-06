
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="NPS Prediction - Retention Tool",
    layout="wide"
)

## Path to models - works both locally and on Streamlit Cloud
BASE = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# LOAD ARTIFACTS
# =============================================================================

@st.cache_resource
def load_artifacts():
    model         = joblib.load(f"{BASE}/models/lgbm_final.pkl")
    feature_names = joblib.load(f"{BASE}/models/feature_names.pkl")
    label_names   = joblib.load(f"{BASE}/models/label_names.pkl")
    label_map     = joblib.load(f"{BASE}/models/label_map.pkl")
    return model, feature_names, label_names, label_map

@st.cache_data
def load_dataset():
    return pd.read_csv(f"{BASE}/data/telco_features.csv")

model, feature_names, label_names, label_map = load_artifacts()
df_full = load_dataset()

COLORS = {
    "Detractor": "#c0392b",
    "Passive"  : "#e67e22",
    "Promoter" : "#27ae60"
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def predict_customer(input_df):
    X          = input_df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0)
    proba      = model.predict_proba(X)[0]
    pred_class = label_names[np.argmax(proba)]
    return pred_class, proba

def get_shap_values(input_df):
    X         = input_df[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    shap_det  = shap_vals[0, :, 0]
    return pd.DataFrame({
        "feature"   : feature_names,
        "shap_value": shap_det
    }).sort_values("shap_value", ascending=False)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("NPS Retention Tool")
st.sidebar.markdown("Predict NPS category for a customer and identify key drivers.")

mode = st.sidebar.radio(
    "Input mode",
    ["Select existing customer", "Enter attributes manually"]
)

# =============================================================================
# MAIN PANEL
# =============================================================================

st.title("Customer NPS Prediction")
st.markdown("This tool helps the retention team identify and prioritise Detractors.")

if mode == "Select existing customer":
    st.subheader("Select a customer")
    selected_id  = st.selectbox("Customer index", df_full.index.tolist())
    customer_row = df_full.loc[[selected_id]]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Key attributes**")
        display_cols = [
            "Tenure in Months", "Monthly Charge", "Contract_Month-to-Month",
            "Online Security_No", "nb_services", "is_autopay"
        ]
        display_cols = [c for c in display_cols if c in customer_row.columns]
        st.dataframe(customer_row[display_cols].T.rename(columns={selected_id: "Value"}))

    pred_class, proba = predict_customer(customer_row)

    with col2:
        st.markdown("**Prediction**")
        color = COLORS[pred_class]
        st.markdown(f"<h2 style='color:{color}'>{pred_class}</h2>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Class"      : label_names,
            "Probability": [f"{p:.1%}" for p in proba]
        }), hide_index=True)

    st.subheader("Top drivers for this prediction")
    shap_df = get_shap_values(customer_row)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Pushing towards Detractor**")
        st.dataframe(shap_df[shap_df["shap_value"] > 0].head(5)[["feature", "shap_value"]].round(3), hide_index=True)

    with col4:
        st.markdown("**Pushing away from Detractor**")
        st.dataframe(shap_df[shap_df["shap_value"] < 0].tail(5)[["feature", "shap_value"]].round(3), hide_index=True)

else:
    st.subheader("Enter customer attributes")
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure         = st.slider("Tenure (months)", 1, 72, 24)
        monthly_charge = st.slider("Monthly Charge ($)", 18, 120, 65)
        nb_services    = st.slider("Number of services", 1, 11, 5)

    with col2:
        contract_m2m = st.checkbox("Month-to-Month contract", value=True)
        no_security  = st.checkbox("No Online Security", value=True)
        is_autopay   = st.checkbox("Auto-pay enabled", value=False)

    with col3:
        age          = st.slider("Age", 19, 80, 45)
        cltv         = st.slider("CLTV", 2000, 6500, 4400)
        nb_referrals = st.slider("Number of referrals", 0, 11, 1)

    if st.button("Predict NPS"):
        input_data = {f: [0] for f in feature_names}
        input_df   = pd.DataFrame(input_data)

        for feat, val in [
            ("Tenure in Months", tenure),
            ("Monthly Charge", monthly_charge),
            ("nb_services", nb_services),
            ("Age", age),
            ("CLTV", cltv),
            ("Number of Referrals", nb_referrals),
            ("is_autopay", int(is_autopay)),
            ("Contract_Month-to-Month", int(contract_m2m)),
            ("Online Security_No", int(no_security)),
        ]:
            if feat in input_df.columns:
                input_df[feat] = val

        pred_class, proba = predict_customer(input_df)
        color = COLORS[pred_class]
        st.markdown(f"<h2 style='color:{color}'>Predicted NPS : {pred_class}</h2>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Class"      : label_names,
            "Probability": [f"{p:.1%}" for p in proba]
        }), hide_index=True)

        shap_df = get_shap_values(input_df)
        st.subheader("Top drivers")
        st.dataframe(
            shap_df.head(10)[["feature", "shap_value"]].round(3),
            hide_index=True
        )

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    "Model : LightGBM trained on IBM Telco 11.1.3+ | "
    "Fairness flags : Dependents gap (0.237) and Senior Citizen gap (0.156) "
    "must be reviewed before production deployment."
)
