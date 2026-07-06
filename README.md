# Customer NPS Prediction — Telecom Operator

**Challenge** : Artefact Take-Home — NPS Prediction  
**Dataset** : IBM Telco Customer Churn 11.1.3+  
**Live Demo** : https://nps-telecom-prediction.streamlit.app/

## Project Overview

This project builds a machine learning system that predicts the NPS category
(Detractor / Passive / Promoter) of telecom customers from their account and
behavioural data. The predictions feed a retention workflow that prioritises
Detractors for proactive outreach.

Only 15% of customers answer NPS surveys. This system scores the silent 85%.

## Live Interface

The Streamlit app is deployed and permanently accessible at :

**https://nps-telecom-prediction.streamlit.app/**

Features :
- Select an existing customer by index and see their predicted NPS category
- Enter customer attributes manually and get a prediction
- See the top SHAP drivers for each individual prediction

## Repository Structure

nps-telecom-prediction/
├── app.py                          # Streamlit interface
├── requirements.txt                # Python dependencies
├── notebooks/
│   └── nps_telecom_full.ipynb     # Full pipeline notebook
├── models/
│   ├── lgbm_final.pkl             # Trained LightGBM model
│   ├── feature_names.pkl
│   ├── label_map.pkl
│   └── label_names.pkl
├── data/
│   └── telco_features.csv
├── figures/                        # All visualisations
├── verbatims/
│   └── synthetic_verbatims.csv    # 200 verbatims generated with Mistral
├── README.md
├── .env.example
└── .gitignore

## Setup and Run

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm \
            shap imbalanced-learn mord streamlit joblib
```

### Run the full pipeline

Open `notebooks/nps_telecom_full.ipynb` in Google Colab and run all cells.

### Launch Streamlit locally

```bash
streamlit run app.py
```

## Key Results

**Selected model : LightGBM**

| Model | Balanced Accuracy | QWK | Detractor Recall |
|---|---|---|---|
| Logistic Regression (baseline) | 0.499 | 0.281 | 0.49 |
| Ordinal Regression (mord) | 0.407 | 0.284 | 0.77 |
| LightGBM | 0.426 | 0.224 | 0.67 |
| TabPFN (n=2000) | 0.342 | 0.036 | 0.96 |

**Top detraction drivers (SHAP) :**
1. Online Security_No
2. Contract_Month-to-Month
3. Monthly Charge
4. Avg Monthly GB Download
5. Number of Referrals

**Lift** : contacting the top 30% captures 39.8% of all Detractors (33% gain over random)

## Fairness Findings

| Group | Gap | Status |
|---|---|---|
| Dependents | 0.237 | FLAG - Legal review required |
| Senior Citizen | 0.156 | FLAG - Monitor post-deployment |
| Gender | 0.003 | OK |
| Married | 0.053 | OK |

## Sections Covered

- 4.1 NPS target construction with leakage analysis and sensitivity testing
- 4.2 Dataset preparation and validation strategy (15%/85% split)
- 4.3 Feature engineering (10 features built and justified)
- 4.4 Synthetic verbatims generated with Mistral API
- 4.5 Modelling and evaluation (4 model families compared)
- 4.6 Drivers of detraction by segment
- 4.7 Fairness and bias audit
- 4.8 Model persistence and Streamlit interface
- 4.9 Monitoring and retraining proposal

## Limitations

1. NPS label derived from Satisfaction Score, not a real NPS survey
2. Perfect satisfaction/churn alignment is unrealistic in production
3. Passive class recall is weak across all models (F1=0.31)
4. Dependents fairness gap must be resolved before production
5. Verbatims are synthetic
6. TabPFN could not be fully evaluated due to Colab constraints
