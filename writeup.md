# Customer NPS Prediction for a Telecom Operator
Technical Write-up

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
Refined mapping with churn tested identical results (the dataset aligns
satisfaction and churn perfectly : no customer with Satisfaction >= 4 ever
churned, an artefact documented as a limitation).
Sensitivity analysis at two levels : distribution (Detractor share moves
from 58% to 84% across mappings A/B/C the mapping is a modelling
decision) and model (training on Mapping B gives BA 0.463 / QWK 0.214,
same performance range the pipeline is not fragile to the mapping).
5% label noise simulated ; model robustness verified (metrics move by
less than +/-0.02 when evaluated against noisy labels).
7 columns excluded from features due to leakage risk.

## 3. Dataset preparation and feature engineering

50 raw columns -> 75 model features after cleaning, feature engineering
and one-hot encoding. Geographic features (Latitude, Longitude,
Population, Zip Code) excluded as socio-economic proxies.
Missing values : Offer filled with No Offer / Internet Type with No Internet.
10 engineered features : tenure_bucket, charge_per_month_ratio, refund_rate,
extra_charges_rate, long_distance_share, charge_per_service, nb_services,
is_autopay, household_size, has_referred. Three of them rank in the
SHAP top 15.

Class imbalance : 58.3% Detractor / 25.4% Passive / 16.3% Promoter.
Mitigation : class_weight=balanced in all models.
Validation : 5-fold stratified CV for model comparison, plus a
15% train / 85% test split simulating the production scenario
(only respondents have labels). Scaling is fitted inside each CV fold
via sklearn Pipelines no test-fold statistics leak into training.

## 4. Modelling and evaluation

Metrics : Balanced Accuracy, QWK, per-class recall, extreme ordinal
violations, calibration, lift curve.

Cross-validation results (model comparison) :
- Logistic Regression : BA=0.499 / QWK=0.281 / Det recall=0.49 / viol=1485
- Ordinal Regression  : BA=0.407 / QWK=0.281 / Det recall=0.76 / viol=407
- LightGBM            : BA=0.427 / QWK=0.224 / Det recall=0.67 / viol=1030
- TabPFN (n=2000)     : BA=0.343 / QWK=0.034 / Det recall=0.96

Selected model : LightGBM. No model dominates on every metric ;
LightGBM offers the best multi-criteria balance : reasonable performance
on all three classes, manageable extreme violations, native SHAP support,
fast inference. Ordinal regression maximises Detractor recall (0.76) but
under-serves the minority classes. TabPFN drifts towards the majority
class on CPU (Detractor recall 0.96 is degenerate, QWK 0.03).

Production model : trained on the 15% respondents only (1,056 customers) -
a true deployment simulation, since silent customers' labels would not
exist. Evaluated on the 85% silent hold-out (5,987 customers) :
BA=0.394 / QWK=0.165 / Det recall=0.716. The ~0.03 BA drop versus CV is
the real cost of the production scenario ; both levels are reported :
CV compares models, hold-out measures deployment.

Calibration : Detractor probabilities are monotonically calibrated -
reliable for ranking, which is what the retention workflow uses.
Passive and Promoter probabilities are overestimated (expected effect
of class_weight=balanced) ; a post-hoc recalibration would fix this
if absolute probabilities were needed.

Lift : contacting the top 30% captures 39.2% of all Detractors
(1.31x over random targeting). With 58% Detractor prevalence the
theoretical ceiling is 1.72x : the model achieves about half of the
maximum possible gain, and nearly saturates the ceiling on the
highest-confidence percentiles.

## 5. Drivers of detraction

Top 5 SHAP drivers (production model) :
1. Contract_Month-to-Month (0.70) - dominant, nearly twice the second
2. Tenure in Months (0.38) - low tenure increases risk
3. Age (0.35) - higher age increases risk ; monitored by the fairness audit
4. Avg Monthly Long Distance Charges (0.33)
5. charge_per_service (0.29) - engineered feature, non-linear effect

Segment findings :
- M2M contracts : 68.3% Detractor rate vs 46.5% for Two Year -
  and M2M is 51% of the base : biggest AND most actionable segment
- New customers (0-12m) : 67.5% vs 51.9% for long-tenure
- Seniors : 66.6% vs 56.7% for non-seniors
- Fiber Optic : 67.4% - paradoxically highest despite premium service
- The apparent Security paradox (Security Yes customers more
  detractor-prone in raw rates) is a composition effect : the
  Security-No group includes 1,526 no-internet customers, who are
  structurally less detractor-prone (38.6%)

Single most likely lever : offer discounted annual contract to M2M
Detractors. Caution : correlation only, not proven causation. A/B test
required.

## 6. Fairness and bias

Audit run on the production model, evaluated on the 85% hold-out :
- Dependents gap : 0.178 -> FLAG - Legal review required
- Married gap : 0.108 -> BORDERLINE monitor post-deployment
- Senior Citizen gap : 0.088 -> OK (resolved - was 0.156 in an
  earlier model version)
- Gender gap : 0.022 -> OK

Dependents gap : the model misses 43% of Detractors with dependents
(recall 0.573 vs 0.751 without). Families would be systematically
under-served. Must be escalated before production. Candidate
mitigations (group-specific threshold, re-weighting the family
segment, family-specific features) are documented but not yet applied.
Married is correlated with Dependents and likely shares the same
mechanism.

Senior gap resolved : the production model relies more on Age
(3rd SHAP driver) and serves seniors better (recall 0.788 vs 0.700).
Using a sensitive variable and discriminating are distinct - the
performance gap is what the audit measures.

Geographic features removed : proxy for socio-economic status.
Accuracy cost : none (BA 0.427 without geo vs 0.423 with) - the
fairness decision is free.

## 7. Synthetic verbatims

200 verbatims generated once with the Mistral API (mistral-small-latest),
then reloaded from disk on later runs (the API is not deterministic -
regenerating would break reproducibility). Conditioned on tenure,
contract, monthly charge, nb_services, security. 15% intentionally
noisy (opposite tone). Seed fixed at 42 for the customer sample and
noise flips. Prompts stored alongside verbatims for auditability.

Sentiment analysis (TextBlob) : Detractor=-0.121 / Passive=+0.026 /
Promoter=+0.206 (separation 0.45 on clean verbatims). Noisy verbatims
show reversed polarity in every class - the noise mechanism is
validated end-to-end. Caveat : verbatims were generated from the
labels, so this validates the generation chain, not the incremental
value of real text ; sentiment is kept descriptive (using it as a
feature would re-inject the label circular leakage). Real
call-centre verbatims would be a legitimate candidate feature.

## 8. Productization

Production model persisted as lgbm_final.pkl with all preprocessing
artifacts (feature names, scaler, label maps), loaded from disk by
the Streamlit app never from memory. The dashboard (5 pages,
deployed on Streamlit Cloud) allows retention managers to :
- Browse the ranked silent-base priority list (customer ID, risk
  scores, contract and tenure context)
- Select an existing customer or enter attributes manually
- See predicted NPS with probabilities and individual SHAP drivers
- Review model performance, segment analysis and the fairness report

## 9. Monitoring

Input drift : 5 features tracked weekly, alert at 2-sigma from the
training baseline (computed on the 1,056-respondent training set).
Prediction drift : alert if the Detractor share shifts >5 points from
the deployment baseline (65.2%).
Retraining triggers : 500+ new labeled responses, or QWK < 0.12 on new
responses (~75% of the production hold-out QWK of 0.165).
Feedback loop : retention outreach creates survivorship bias in future
labels log all predictions and interventions, use holdout groups,
never retrain solely on contacted customers.
Baselines and thresholds are persisted as monitoring artifacts.

## 10. Limitations and next steps

Limitations :
1. NPS derived from Satisfaction Score, not a real survey
2. Perfect satisfaction/churn alignment unrealistic in production
3. Passive class recall weak across all models (F1 0.28-0.31)
4. Dependents fairness gap (0.178) unresolved
5. Verbatims are synthetic
6. TabPFN not fully evaluated (CPU constraints)

Next steps :
1. Collect real NPS survey responses to validate the label construction
2. Add real call-centre verbatims
3. Resolve the Dependents fairness gap
4. A/B test the contract upgrade recommendation
5. Implement monitoring with Evidently AI
6. Retrain as new labeled responses accumulate
