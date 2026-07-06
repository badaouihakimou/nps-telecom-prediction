# Customer NPS Prediction - Telecom Operator

Take-Home Challenge - Artefact
Dataset : IBM Telco Customer Churn 11.1.3+

## Project overview

This project builds a machine learning system that predicts the NPS category
(Detractor / Passive / Promoter) of telecom customers from their account and
behavioural data. The predictions feed a retention workflow that prioritises
Detractors for proactive outreach.

Only 15% of customers answer NPS surveys. This system scores the silent 85%.

## Repository structure

nps_challenge/
- data/        : datasets, predictions, reports
- models/      : trained model and preprocessing artifacts
- figures/     : all visualisations
- verbatims/   : 200 synthetic verbatims generated with Mistral
- app.py       : Streamlit interface
- nps_telecom_full.ipynb : full pipeline notebook

## Setup

pip install pandas numpy matplotlib seaborn scikit-learn lightgbm shap imbalanced-learn mord streamlit joblib

## Run pipeline

Open nps_telecom_full.ipynb in Google Colab and run all cells in order.

## Launch Streamlit

streamlit run app.py

## Key findings

Selected model : LightGBM
Balanced Accuracy : 0.426 / QWK : 0.224

Top detraction drivers :
1. Online Security_No
2. Contract_Month-to-Month
3. Monthly Charge

Fairness flags before production :
- Dependents gap : 0.237 -> Legal review required
- Senior Citizen gap : 0.156 -> Monitor post-deployment

## AI tools disclosure

AI coding assistants were used to scaffold and review parts of this code.
All decisions and results are the author own.

## Limitations

1. NPS label derived from Satisfaction Score, not a real NPS survey
2. Perfect satisfaction/churn alignment is unrealistic in production
3. Passive class recall is weak across all models
4. Dependents fairness gap must be resolved before production
5. Verbatims are synthetic
6. TabPFN could not be fully evaluated due to Colab constraints
