# Customer NPS Prediction for a Telecom Operator
Technical Write-up

Challenge : Artefact Take-Home
Dataset : IBM Telco Customer Churn 11.1.3+
Author : Moustapha Mahamat Abdelhakim

## 1. Problem framing

A pan-African telecom operator runs regular NPS surveys but only 15% of
customers respond. This project builds a ML system that predicts the NPS
category (Detractor / Passive / Promoter) for the silent 85%.

NPS is an ordinal target : Detractor < Passive < Promoter.
Three formulations were considered : multiclass classification, ordinal
classification, and regression with thresholding. The third was ruled out
(no granular continuous proxy available). Multiclass (LightGBM) and ordinal
(mord) were both implemented and compared.

## 2. Target construction

Mapping : Satisfaction 5 = Promoter / 4 = Passive / <=3 = Detractor.
Refined mapping with CLTV tested - negligible impact on model metrics.
5% realistic noise added to simulate survey uncertainty.
7 columns excluded from features due to leakage risk.

Key finding : no customer with Satisfaction >= 4 churned in this dataset.
Perfect alignment between satisfaction and churn makes a refined mapping
identical to the baseline.

## 3. Dataset preparation and feature engineering

50 raw columns -> 35 after dropping identifiers, leakage, and geo features.
Missing values : Offer filled with No Offer / Internet Type with No Internet.
10 engineered features : tenure_bucket, charge_per_month_ratio, refund_rate,
extra_charges_rate, long_distance_share, charge_per_service, nb_services,
is_autopay, household_size, has_referred.

Class imbalance : 58.3% Detractor / 25.4% Passive / 16.3% Promoter.
Mitigation : class_weight=balanced in all models.
Validation : 15% train / 85% test split + 5-fold stratified CV.

## 4. Modelling and evaluation

Metrics : Balanced Accuracy, QWK, per-class recall, calibration, lift curve.

Results :
- Logistic Regression : BA=0.499 / QWK=0.281 / Det recall=0.49
- Ordinal Regression  : BA=0.407 / QWK=0.284 / Det recall=0.77
- LightGBM            : BA=0.426 / QWK=0.224 / Det recall=0.67
- TabPFN (n=2000)     : BA=0.342 / QWK=0.036 / Det recall=0.96

Selected model : LightGBM.
Only model predicting all three classes reasonably.
Native SHAP support. Fast inference for production.
Ordinal regression never predicts Promoter (recall=0.00).
TabPFN collapses to majority class without GPU and class weighting.

Lift : contacting top 30% captures 39.8% of all Detractors (33% gain
over random targeting).

## 5. Drivers of detraction

Top 5 SHAP drivers :
1. Online Security_No (0.215)
2. Contract_Month-to-Month (0.200)
3. Monthly Charge (0.184)
4. Avg Monthly GB Download (0.168)
5. Number of Referrals (0.152)

Segment findings :
- M2M contracts : 68.3% Detractor rate vs 46.5% for Two Year
- New customers (0-12m) : 67.5% vs 51.9% for long-tenure
- Fiber Optic : 67.4% - paradoxically highest despite premium service

Single most likely lever : offer discounted annual contract to M2M Detractors.
Caution : correlation only, not proven causation. A/B test required.

## 6. Fairness and bias

Audit results :
- Senior Citizen gap : 0.156 -> FLAG
- Gender gap : 0.003 -> OK
- Married gap : 0.053 -> OK
- Dependents gap : 0.237 -> FLAG - Legal review required

Dependents gap : model misses 53% of Detractors with dependents.
Must be escalated before production deployment.

Geographic features removed : proxy for socio-economic status.
Accuracy cost negligible.

## 7. Synthetic verbatims

200 verbatims generated with Mistral API (mistral-small-latest).
Conditioned on tenure, contract, monthly charge, nb_services, security.
15% intentionally noisy. Seed fixed at 42.

Sentiment analysis : Detractor=-0.098 / Passive=-0.012 / Promoter=+0.192.
Large class overlap -> text adds limited value over tabular baseline.
Real call-centre verbatims would likely show stronger signal.

## 8. Productization

Model persisted as lgbm_final.pkl with all preprocessing artifacts.
Streamlit interface allows retention managers to :
- Select existing customer by index
- Enter attributes manually
- See predicted NPS with probabilities and top SHAP drivers

## 9. Monitoring

Input drift : 5 features tracked weekly, alert at 2-sigma.
Prediction drift : alert if Detractor share shifts >5 points.
Retraining trigger : 500+ new labeled responses or QWK < 0.15.
Feedback loop : log all predictions, use holdout groups.

## 10. Limitations and next steps

Limitations :
1. NPS derived from Satisfaction Score, not a real survey
2. Perfect satisfaction/churn alignment unrealistic in production
3. Passive class recall weak across all models (F1=0.31)
4. Dependents fairness gap unresolved
5. Verbatims are synthetic
6. TabPFN not fully evaluated

Next steps :
1. Collect real NPS survey responses
2. Add real call-centre verbatims
3. Resolve Dependents fairness gap
4. A/B test contract upgrade recommendation
5. Implement monitoring with Evidently AI
6. Retrain on larger labeled dataset
