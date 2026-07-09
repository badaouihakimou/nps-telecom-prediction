
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os


# CONFIGURATION

st.set_page_config(
    page_title="NPS Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

## Hide Streamlit menu and footer
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))

COLORS = {
    "Detractor": "#c0392b",
    "Passive"  : "#e67e22",
    "Promoter" : "#27ae60"
}

# LOAD ARTIFACTS


@st.cache_resource
def load_artifacts():
    model         = joblib.load(f"{BASE}/models/lgbm_final.pkl")
    feature_names = joblib.load(f"{BASE}/models/feature_names.pkl")
    label_names   = joblib.load(f"{BASE}/models/label_names.pkl")
    label_map     = joblib.load(f"{BASE}/models/label_map.pkl")
    return model, feature_names, label_names, label_map

@st.cache_data
def load_data():
    df       = pd.read_csv(f"{BASE}/data/telco_features.csv")
    silent   = pd.read_csv(f"{BASE}/data/silent_base_predictions.csv")
    fairness = pd.read_csv(f"{BASE}/data/fairness_report.csv")
    shap_imp = pd.read_csv(f"{BASE}/data/shap_importance.csv")
    model_c  = pd.read_csv(f"{BASE}/data/model_comparison.csv")
    return df, silent, fairness, shap_imp, model_c

model, feature_names, label_names, label_map = load_artifacts()
df_full, silent_df, fairness_df, shap_imp, model_comp = load_data()

# HELPER FUNCTIONS

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

def show_image(path, caption=""):
    if os.path.exists(path):
        st.image(path, caption=caption, use_column_width=True)
    else:
        st.warning(f"Figure not found : {path}")

# SIDEBAR NAVIGATION

st.sidebar.title("NPS Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Overview",
     "Individual Prediction",
     "Analytics Dashboard",
     "Silent Base Explorer",
     "Fairness Report"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model** : LightGBM  \n"
    "**Dataset** : IBM Telco 11.1.3+  \n"
    "**Customers** : 7,043  \n"
    "**Bal. Accuracy** : 0.426"
)

# PAGE 1 : OVERVIEW

if page == "Overview":
    st.title("Customer NPS Prediction Overview")
    st.markdown(
        "A telecom operator sends NPS surveys but only **15% of customers respond**. "
        "This system predicts the NPS category (Detractor / Passive / Promoter) "
        "for the silent **85%** using account and behavioural data."
    )

    st.markdown("---")

    ## KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", "7,043")
    col2.metric("Respondents (15%)", "1,056")
    col3.metric("Silent Base (85%)", "5,987")
    col4.metric("Predicted Detractors", "3,262", "54.5%")

    st.markdown("---")

    ## NPS Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("NPS Class Distribution")
        st.markdown(
            "The target variable has 3 ordered classes : "
            "**Detractor** (0-6) < **Passive** (7-8) < **Promoter** (9-10). "
            "58.3% of customers are Detractors the majority class."
        )
        dist_data = {"Detractor": 4105, "Passive": 1789, "Promoter": 1149}
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(dist_data.keys(), dist_data.values(),
                      color=[COLORS[k] for k in dist_data.keys()],
                      edgecolor="white", width=0.5)
        for bar, v in zip(bars, dist_data.values()):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 30,
                    f"{v}\n({v/7043*100:.1f}%)",
                    ha="center", fontsize=9)
        ax.set_ylabel("Number of customers")
        ax.set_ylim(0, 5200)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Model Comparison")
        st.markdown(
            "Four models were compared. LightGBM was selected as the only "
            "model predicting all 3 classes reasonably with native SHAP support."
        )
        st.dataframe(model_comp, hide_index=True, use_container_width=True)

    st.markdown("---")

    ## Top drivers
    st.subheader("Top 5 Detraction Drivers")
    st.markdown(
        "SHAP values measure each feature contribution to the Detractor prediction. "
        "The top 2 drivers (**Online Security** and **Contract type**) are actionable — "
        "the business can offer security bundles and contract upgrades."
    )
    top5 = shap_imp.head(5)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.barh(top5["feature"][::-1], top5["importance"][::-1],
            color="#c0392b", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.warning(
        " **Fairness flags before production** : "
        "Dependents gap = 0.237 | Senior Citizen gap = 0.156. "
        "Legal review required. See Fairness Report page."
    )

# PAGE 2 : INDIVIDUAL PREDICTION

elif page == "Individual Prediction":
    st.title("Individual Customer Prediction")
    st.markdown(
        "Select an existing customer or enter attributes manually "
        "to get a NPS prediction with individual SHAP explanations."
    )

    mode = st.radio(
        "Input mode",
        ["Select existing customer", "Enter attributes manually"],
        horizontal=True
    )

    st.markdown("---")

    if mode == "Select existing customer":
        selected_id  = st.selectbox("Customer index", df_full.index.tolist())
        customer_row = df_full.loc[[selected_id]]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Key attributes**")
            display_cols = [
                "Tenure in Months", "Monthly Charge",
                "Contract_Month-to-Month", "Online Security_No",
                "nb_services", "is_autopay"
            ]
            display_cols = [c for c in display_cols if c in customer_row.columns]
            st.dataframe(
                customer_row[display_cols].T.rename(columns={selected_id: "Value"}),
                use_container_width=True
            )

        pred_class, proba = predict_customer(customer_row)

        with col2:
            st.markdown("**Prediction**")
            color = COLORS[pred_class]
            st.markdown(
                f"<h2 style='color:{color}'>{pred_class}</h2>",
                unsafe_allow_html=True
            )
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.barh(label_names, proba,
                    color=[COLORS[l] for l in label_names],
                    edgecolor="white")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            for i, v in enumerate(proba):
                ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.subheader("SHAP Drivers for this customer")
        st.markdown(
            "Positive SHAP values push towards **Detractor**. "
            "Negative values push away from Detractor."
        )
        shap_df = get_shap_values(customer_row)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Pushing towards Detractor**")
            st.dataframe(
                shap_df[shap_df["shap_value"] > 0].head(5)[["feature", "shap_value"]].round(3),
                hide_index=True
            )
        with col4:
            st.markdown("**Pushing away from Detractor**")
            st.dataframe(
                shap_df[shap_df["shap_value"] < 0].tail(5)[["feature", "shap_value"]].round(3),
                hide_index=True
            )

    else:
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

        if st.button("Predict NPS", type="primary"):
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

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"<h2 style='color:{color}'>Predicted NPS : {pred_class}</h2>",
                    unsafe_allow_html=True
                )
                prob_df = pd.DataFrame({
                    "Class"      : label_names,
                    "Probability": [f"{p:.1%}" for p in proba]
                })
                st.dataframe(prob_df, hide_index=True)

            with col2:
                fig, ax = plt.subplots(figsize=(5, 2))
                ax.barh(label_names, proba,
                        color=[COLORS[l] for l in label_names],
                        edgecolor="white")
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                for i, v in enumerate(proba):
                    ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.markdown("---")
            st.subheader("Top SHAP drivers")
            shap_df = get_shap_values(input_df)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Pushing towards Detractor**")
                st.dataframe(
                    shap_df[shap_df["shap_value"] > 0].head(5)[["feature", "shap_value"]].round(3),
                    hide_index=True
                )
            with col2:
                st.markdown("**Pushing away from Detractor**")
                st.dataframe(
                    shap_df[shap_df["shap_value"] < 0].tail(5)[["feature", "shap_value"]].round(3),
                    hide_index=True
                )

# PAGE 3 : ANALYTICS DASHBOARD

elif page == "Analytics Dashboard":
    st.title("Analytics Dashboard")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Overview",
        "Model Performance",
        "SHAP Analysis",
        "Segment Analysis",
        "Verbatim Sentiment"
    ])

    with tab1:
        st.subheader("Missing Values")
        st.markdown(
            "4 columns have missing values all expected and meaningful. "
            "Churn Category/Reason are only filled for churned customers (73.5%). "
            "Offer (55%) and Internet Type (22%) nulls were filled with explicit labels."
        )
        show_image(f"{BASE}/figures/missing_values.png")

        st.subheader("Key Numeric Features Distribution")
        st.markdown(
            "Tenure shows many new (0-5m) and loyal (72m) customers. "
            "Monthly Charge ranges from $18 to $118. "
            "Satisfaction Score peaks at 3 consistent with 58% Detractors."
        )
        show_image(f"{BASE}/figures/numeric_distributions.png")

        st.subheader("Key Signal Exploration")
        st.markdown(
            "Critical finding : no customer with Satisfaction >= 4 ever churned. "
            "Satisfaction and churn are perfectly aligned confirms the baseline NPS mapping."
        )
        show_image(f"{BASE}/figures/key_signals.png")

        st.subheader("NPS Target Distribution")
        show_image(f"{BASE}/figures/target_distribution.png")

        st.subheader("Mapping Sensitivity Analysis")
        st.markdown(
            "Three mappings were compared. Mapping A (baseline) produces the most "
            "balanced distribution. Mapping C removes the Passive class entirely unusable."
        )
        show_image(f"{BASE}/figures/mapping_sensitivity.png")

        st.subheader("Engineered Features Distribution")
        st.markdown(
            "10 new features were built from raw data. Most charge-related features "
            "are sparse (concentrated at 0) but carry strong signal in non-zero cases."
        )
        show_image(f"{BASE}/figures/engineered_features.png")

    with tab2:
        st.subheader("Model Comparison")
        st.markdown(
            "LightGBM was selected as the only model predicting all 3 classes. "
            "Ordinal Regression never predicts Promoter (recall=0.00). "
            "TabPFN collapses to majority class on CPU without class weighting."
        )
        show_image(f"{BASE}/figures/model_comparison.png")

        st.subheader("Confusion Matrices (5-fold CV)")
        st.markdown(
            "LightGBM is the most balanced 577 extreme violations vs 1483 for LR. "
            "Ordinal Regression has only 2 extreme violations but never predicts Promoter."
        )
        show_image(f"{BASE}/figures/confusion_matrices.png")

        st.subheader("Calibration Plots")
        st.markdown(
            "Detractor probabilities are well calibrated the retention team can "
            "trust the Detractor probability ranking. Passive and Promoter are less reliable."
        )
        show_image(f"{BASE}/figures/calibration_plots.png")

        st.subheader("Lift Curve — Detractor Targeting Efficiency")
        st.markdown(
            "Contacting the **top 30%** by predicted Detractor probability captures "
            "**39.8% of all Detractors** a 33% efficiency gain over random targeting."
        )
        show_image(f"{BASE}/figures/lift_curve.png")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Feature Importance (Bar)")
            st.markdown(
                "Mean absolute SHAP value per feature. "
                "Online Security_No and Contract_Month-to-Month are the top 2 drivers."
            )
            show_image(f"{BASE}/figures/shap_bar_detractor.png")
        with col2:
            st.subheader("SHAP Beeswarm")
            st.markdown(
                "Each dot = one customer. Red = high feature value. "
                "Dots to the right push towards Detractor. "
                "High Monthly Charge (red) strongly pushes towards Detractor."
            )
            show_image(f"{BASE}/figures/shap_beeswarm_detractor.png")

    with tab4:
        st.subheader("Detractor Rates by Customer Segment")
        st.markdown(
            "Month-to-Month customers : 68.3% Detractor rate (vs 46.5% for Two Year). "
            "New customers (0-12m) : 67.5% (vs 51.9% for long-tenure). "
            "Fiber Optic : 67.4% highest despite premium service (unmet expectations)."
        )
        show_image(f"{BASE}/figures/segment_detractor_rates.png")

        st.subheader("Mapping Sensitivity")
        show_image(f"{BASE}/figures/mapping_sensitivity.png")

    with tab5:
        st.subheader("Verbatim Sentiment Analysis")
        st.markdown(
            "200 synthetic verbatims generated with Mistral API. "
            "Detractor mean polarity = -0.091 (negative). "
            "Promoter mean polarity = +0.238 (positive). "
            "15% noise inverts the tone visible in the right chart."
        )
        show_image(f"{BASE}/figures/verbatim_sentiment.png")

# PAGE 4 : SILENT BASE EXPLORER

elif page == "Silent Base Explorer":
    st.title("Silent Base Priority List")
    st.markdown(
        "**5,987 customers** who never answered the NPS survey, "
        "sorted by predicted Detractor probability. "
        "The retention team uses this list to prioritise outreach."
    )

    st.markdown("---")

    ## Summary metrics
    if "predicted_nps" in silent_df.columns:
        dist = silent_df["predicted_nps"].value_counts()
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Detractors",
                    dist.get("Detractor", 0),
                    f"{dist.get('Detractor', 0)/len(silent_df)*100:.1f}%")
        col2.metric("Predicted Passives",
                    dist.get("Passive", 0),
                    f"{dist.get('Passive', 0)/len(silent_df)*100:.1f}%")
        col3.metric("Predicted Promoters",
                    dist.get("Promoter", 0),
                    f"{dist.get('Promoter', 0)/len(silent_df)*100:.1f}%")

    st.markdown("---")

    ## Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        nps_filter = st.multiselect(
            "Filter by predicted NPS",
            ["Detractor", "Passive", "Promoter"],
            default=["Detractor"]
        )
    with col2:
        min_prob = st.slider("Min. Detractor probability", 0.0, 1.0, 0.5)
    with col3:
        top_n = st.selectbox("Show top N customers", [50, 100, 200, 500], index=1)

    ## Filter data
    if "predicted_nps" in silent_df.columns:
        filtered = silent_df[
            (silent_df["predicted_nps"].isin(nps_filter)) &
            (silent_df["prob_detractor"] >= min_prob)
        ].head(top_n)
    else:
        filtered = silent_df.head(top_n)

    st.metric("Customers matching filters", len(filtered))

    ## Show data
    display_cols = ["predicted_nps", "prob_detractor", "prob_passive", "prob_promoter"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].round(3),
        use_container_width=True
    )

    ## Distribution chart
    if "predicted_nps" in silent_df.columns:
        st.subheader("Predicted NPS Distribution Silent Base")
        fig, ax = plt.subplots(figsize=(8, 3))
        dist_vals = [dist.get(l, 0) for l in ["Detractor", "Passive", "Promoter"]]
        bars = ax.bar(["Detractor", "Passive", "Promoter"], dist_vals,
                      color=[COLORS[l] for l in ["Detractor", "Passive", "Promoter"]],
                      edgecolor="white", width=0.5)
        for bar, v in zip(bars, dist_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 20,
                    f"{v}\n({v/len(silent_df)*100:.1f}%)",
                    ha="center", fontsize=10)
        ax.set_ylabel("Number of customers")
        ax.set_ylim(0, max(dist_vals) * 1.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# PAGE 5 : FAIRNESS REPORT

elif page == "Fairness Report":
    st.title("Fairness Audit Report")
    st.markdown(
        "The model allocates retention budget it must treat all demographic "
        "groups equally. We measure **Detractor recall** per group : "
        "of all true Detractors in this group, how many did the model find ?"
    )

    st.markdown("---")

    ## Summary table
    st.subheader("Fairness Summary")
    st.dataframe(fairness_df, use_container_width=True, hide_index=True)

    ## Fairness chart
    show_image(f"{BASE}/figures/fairness_audit.png")

    st.markdown("---")

    ## Flags with detailed explanations
    st.subheader("Required Actions Before Production")

    st.error(
        " **Dependents gap = 0.237 FLAG Legal review required**\n\n"
        "The model misses **53% of Detractors** among customers with dependents "
        "(recall = 0.474) vs 29% for customers without (recall = 0.711). "
        "The retention team would systematically under-serve families. "
        "\n\n**Mitigation options :**\n"
        "- Lower decision threshold for customers with dependents\n"
        "- Train a separate model for this segment\n"
        "- Add features capturing family-specific service issues"
    )

    st.warning(
        " **Senior Citizen gap = 0.156 FLAG Monitor post-deployment**\n\n"
        "The model detects MORE senior Detractors (79.2%) than non-seniors (63.6%). "
        "Counter-intuitive may reflect overfitting on senior-correlated patterns. "
        "Monitor closely after deployment."
    )

    st.success("**Gender gap = 0.003 OK** : No significant disparity between genders.")
    st.success("**Married gap = 0.053 OK** : Acceptable difference. No action required.")

    st.markdown("---")
    st.subheader("Geographic Features Decision")
    st.info(
        "**Latitude, Longitude and Population were removed** from the model. \n\n"
        "These features can proxy socio-economic status (rich vs poor neighbourhoods). "
        "A model using these would allocate retention budget unevenly by area  "
        "potentially discriminatory. \n\n"
        "**Accuracy cost : negligible** balanced accuracy improved slightly "
        "after removal (0.430 → 0.434)."
    )

# FOOTER


st.markdown("---")
st.markdown(
    "<small>NPS Prediction Dashboard | "
    "Model : LightGBM | Dataset : IBM Telco 11.1.3+ | "
    "Challenge : Artefact Take-Home | "
    "Live : https://nps-telecom-prediction.streamlit.app</small>",
    unsafe_allow_html=True
)
