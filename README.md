# Customer NPS Prediction for a Telecom Operator

**Dataset** : IBM Telco Customer Churn 11.1.3+  
**Live Demo** : https://nps-telecom-prediction.streamlit.app/  

## What is this project about ?

### The business problem

A telecom company (a business that sells phone and internet subscriptions)
sends satisfaction surveys to its customers periodically.
The survey asks one simple question :
*"On a scale of 0 to 10, would you recommend our company to a friend?"*

**The problem** : only 15% of customers answer the survey.
The other 85% are silent we do not know if they are happy or not.

**The consequence** : the retention team (the team responsible for keeping
customers) can only target 15% of the customer base. They miss 85% of
unhappy customers who are about to leave.

**This project solves this problem** : we build a machine learning system
that predicts the satisfaction level of the 85% who never answered,
using information already available in the CRM (customer database).
The production model is trained on the 15% respondents only a true
simulation of deployment, where silent customers' labels do not exist.


## Key Definitions

### NPS (Net Promoter Score)
A customer satisfaction measurement system used worldwide.
One question : *"Would you recommend us to a friend? (0-10)"*

Customers are grouped into 3 categories based on their answer :
- **Detractor** (score 0 to 6) : unhappy customer, likely to leave and
  speak negatively about the company
- **Passive** (score 7 to 8) : neutral customer, neither happy nor unhappy,
  could leave if a competitor makes a better offer
- **Promoter** (score 9 to 10) : very happy customer, loyal and recommends
  the company to friends and family

**NPS formula** : NPS = % Promoters - % Detractors  
Result between -100 (everyone is unhappy) and +100 (everyone is happy)

### Churn
When a customer cancels their subscription and moves to a competitor.
Example : a customer who leaves Orange to join MTN = churn.
Why it matters : acquiring a new customer costs 5 to 7 times more
than keeping an existing one.

### Machine Learning (ML)
A computer program that learns patterns from historical data
and uses them to make predictions on new data.
In this project : the program learns that customers with a monthly contract
and a short tenure are often Detractors, then applies this
logic to predict the NPS of all customers whose satisfaction is unknown.

### Dataset
A table of data with rows and columns.
Here : 7043 rows (customers) and 50 columns (information about each customer).

### SHAP (SHapley Additive exPlanations)
A method that explains WHY the model made a specific prediction.
It measures the contribution of each feature (column) to the prediction.
Example : "This customer is predicted Detractor mainly because they have
a monthly contract (+0.70) and a very short tenure."

### Feature Engineering
Creating new variables (columns) from existing ones to help the model.
Example : dividing Monthly Charge by number of services gives
charge_per_service a measure of value perception.

### Cross-Validation
A technique to evaluate the model on data it has never seen.
The data is split into 5 groups. The model trains on 4 groups and
is evaluated on the 5th. This is repeated 5 times. The average
performance gives a reliable estimate for comparing models.

### Balanced Accuracy
A metric that weights each class equally regardless of size.
A model predicting only Detractor (58% of data) gets 33% balanced
accuracy not 58%. Essential for imbalanced datasets.

### QWK (Quadratic Weighted Kappa)
A metric that penalises extreme errors more than small ones.
Predicting Promoter for a true Detractor is punished much more
than predicting Passive. Perfect for an ordered target like NPS.

### Leakage
Using information in the model that would not be available
at prediction time. It inflates performance artificially.
Example : using Churn Score (which contains satisfaction information)
to predict NPS would be cheating.

### Calibration
Measuring whether predicted probabilities are trustworthy.
A perfectly calibrated model predicts 70% probability for events
that actually occur 70% of the time.

### Lift
Measures how much more efficient the model is compared to random targeting.
Lift of 1.31 = the model is 31% more efficient than random.

## The Dataset

**Source** : IBM Telco Customer Churn 11.1.3+  
**Link** : https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3

**Why this specific version ?**
The standard Kaggle version only has 21 columns.
This version has 50 columns including the Satisfaction Score (1 to 5)
which we need to build our NPS target variable.

**What it contains :**
- 7043 customers (rows)
- 50 columns including demographics, services, contract, charges and satisfaction

## How We Built the NPS Target

The dataset does not have a real NPS score (0-10).
It has a Satisfaction Score (1-5). We convert it :

| Satisfaction Score | NPS Category |
|---|---|
| 5 | Promoter |
| 4 | Passive |
| 1, 2 or 3 | Detractor |

**Result** : 58.3% Detractors / 25.4% Passives / 16.3% Promoters

**Key finding** : no customer with Satisfaction >= 4 ever churned.
Satisfaction and churn are perfectly aligned - an artefact of this
IBM dataset, documented as a limitation.

**Columns excluded to avoid leakage** :
- Satisfaction Score : directly builds the target
- Churn Score : too correlated with satisfaction
- Churn Label : reveals if the customer left
- Churn Category / Churn Reason : post-event information
- Customer Status : reveals the outcome
- Total Revenue : too correlated with CLTV

A sensitivity analysis compared three alternative mappings, at the
distribution level and at the model level : the mapping is a modelling
decision (Detractor share moves from 58% to 84% across variants) but
the pipeline is not fragile to it.


## Feature Engineering

10 new variables were created from the raw data :

| Feature | What it measures | Business logic |
|---|---|---|
| tenure_bucket | Customer age group (0-12m, 13-24m, etc.) | Loyal customers are more satisfied |
| charge_per_month_ratio | Total charges / tenure | High ratio = poor value perception |
| refund_rate | Refunds / total charges | High refunds = billing frustration |
| extra_charges_rate | Extra data charges / total charges | Unexpected costs damage satisfaction |
| long_distance_share | Long distance / total charges | High share = specific pain point |
| charge_per_service | Monthly charge / number of services | Value perception per service |
| nb_services | Count of active services | More services = more risk points |
| is_autopay | 1 if auto-pay, 0 if manual | Auto-pay = more engaged customer |
| household_size | Number of dependents + 1 | Families have more complex needs |
| has_referred | 1 if referred someone, 0 otherwise | Direct NPS proxy behaviour |

Three engineered features rank in the SHAP top 15 (charge_per_service,
long_distance_share, charge_per_month_ratio).


## Models Compared

Four model families were compared with 5-fold cross-validation :

| Model | Balanced Accuracy | QWK | Detractor Recall | Violations |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.499 | 0.281 | 0.49 | 1485 |
| Ordinal Regression (mord) | 0.407 | 0.281 | 0.76 | 407 |
| **LightGBM (selected)** | **0.427** | **0.224** | **0.67** | **1030** |
| TabPFN (bonus, n=2000) | 0.343 | 0.034 | 0.96 | N/A |

No model dominates on every metric. LightGBM offers the best
multi-criteria balance : reasonable performance on all three classes,
manageable extreme violations, native SHAP support and fast inference.
Ordinal Regression maximises Detractor recall but under-serves the
minority classes ; TabPFN drifts towards the majority class on CPU.

### Production model

The deployed model is trained on the **15% respondents only**
(1,056 customers) - silent customers' labels would not exist in
production. Evaluated on the 85% silent hold-out (5,987 customers) :

| Metric | CV (model comparison) | Hold-out (deployment) |
|---|---|---|
| Balanced Accuracy | 0.427 | 0.394 |
| QWK | 0.224 | 0.165 |
| Detractor Recall | 0.667 | 0.716 |

The ~0.03 BA drop is the real cost of the production scenario
both levels are reported honestly : CV compares models, hold-out
measures deployment.

### What is LightGBM ?
Light Gradient Boosting Machine, developed by Microsoft.
It builds many small decision trees sequentially, each correcting
the mistakes of the previous one. Very fast and accurate on tabular data.

## Key Findings

### Top 5 Detraction Drivers (SHAP, production model)

1. **Contract_Month-to-Month** (0.70) : by far the dominant driver
   nearly twice the second. Lowest switching cost, most fragile loyalty
2. **Tenure in Months** (0.38) : recent customers are at higher risk
3. **Age** (0.35) : higher age increases risk - a business signal
   (senior-adapted support), monitored by the fairness audit
4. **Avg Monthly Long Distance Charges** (0.33) : heavy long-distance
   usage flags a specific pain point
5. **charge_per_service** (0.29) : engineered feature with a
   non-linear effect (interaction with bundle size)

### Segment Analysis

| Segment | Detractor Rate |
|---|---|
| Contract Month-to-Month | 68.3% |
| Contract Two Year | 46.5% |
| Tenure 0-12 months | 67.5% |
| Tenure 49-72 months | 51.9% |
| Senior citizens | 66.6% |
| Non-seniors | 56.7% |
| Fiber-optic internet | 67.4% |
| No Internet | 38.6% |

### Business Recommendation
For any predicted Detractor on a month-to-month contract :
offer a discounted annual contract as the first retention action.
This directly addresses the primary detraction driver.
*Caution : A/B testing is required to confirm causal impact.*

### Lift
Contacting the top 30% by predicted Detractor probability captures
**39.2% of all Detractors** a **1.31x lift** over random targeting.
With 58% Detractor prevalence the theoretical ceiling is 1.72x :
the model achieves about half of the maximum possible gain.

## Fairness Audit

The model allocates retention budget. We verify it treats all
demographic groups equally, measuring Detractor recall per group
on the production model (85% hold-out).

| Group | Recall Gap | Status |
|---|---|---|
| Dependents | 0.178 | **FLAG - Legal review required** |
| Married | 0.108 | BORDERLINE - monitor |
| Senior Citizen | 0.088 | OK (resolved - was 0.156 in an earlier model version) |
| Gender | 0.022 | OK |

**Dependents gap (0.178)** : the model misses 43% of Detractors among
customers with dependents (recall 0.573) vs 25% without (0.751).
Families would be systematically under-served. This must be resolved
or mitigated before production ; candidate mitigations (group-specific
threshold, re-weighting the family segment, family-specific features)
are documented but not yet applied.

**Senior gap resolved** : the production model relies more on Age
(3rd SHAP driver) and serves seniors BETTER (recall 0.788 vs 0.700).
Using a sensitive variable and discriminating are distinct things -
the performance gap is what matters.

**Geographic features removed** : Latitude, Longitude and Population
were removed to avoid socio-economic discrimination. Accuracy cost :
none (BA 0.427 without geo vs 0.423 with) the decision is free.

## Synthetic Verbatims

200 synthetic customer interaction notes were generated once with
the Mistral AI API (mistral-small-latest), conditioned on each
customer's profile (tenure, contract, charges, services), then
reloaded from disk on later runs (the API is not deterministic).

15% of verbatims are intentionally noisy (tone opposite to NPS label)
to simulate real-world imperfection.

Sentiment analysis (TextBlob) validates the generation chain :
- Detractor mean polarity : -0.121 (negative)
- Passive mean polarity : +0.026 (neutral)
- Promoter mean polarity : +0.206 (positive)
- Noisy verbatims show reversed polarity in every class

Sentiment is kept as a descriptive analysis using it as a model
feature would re-inject the label (circular leakage), since the
verbatims were generated from the labels.


## Silent Base Predictions

5,987 customers who never answered the NPS survey were scored by
the production model (which has never seen their labels) :
- Predicted Detractors : 3,902 (65.2%)
- Predicted Passives : 1,347 (22.5%)
- Predicted Promoters : 738 (12.3%)

The predicted Detractor share is ~7 points above the respondent rate
(58.3%) : this reflects the model's behaviour (it slightly over-predicts
the Detractor class), not a calibrated prevalence estimate. The RANKING
is what the retention workflow uses, and it is reliable (monotonic
calibration).

## Monitoring

5 key features are tracked weekly :
Tenure, Monthly Charge, nb_services, charge_per_month_ratio, CLTV.

Alert threshold : 2 standard deviations from the training baseline
(computed on the 1,056-respondent training set).

Prediction drift : alert if the Detractor share shifts by more than
5 points from the deployment baseline (65.2%).

Retraining triggers :
- 500+ new labeled survey responses collected
- QWK drops below 0.12 on new responses (~75% of the production
  hold-out QWK of 0.165)
- Any feature drifts beyond 2-sigma for 3 consecutive weeks

Feedback loop : retention outreach affects future survey answers
(survivorship bias). Mitigation : log all predictions and
interventions, use holdout groups, never retrain solely on
contacted customers.

## Limitations

1. NPS label derived from Satisfaction Score, not a real NPS survey
2. Perfect alignment between satisfaction and churn is unrealistic in production
3. Passive class recall is weak across all models (F1 0.28-0.31)
4. Dependents fairness gap (0.178) must be resolved before production
5. Verbatims are synthetic - real call-centre text would add more value
6. TabPFN could not be fully evaluated due to Colab memory constraints
## Repository Structure

```
nps-telecom-prediction/
├── app.py                          # Streamlit dashboard (5 pages)
├── requirements.txt                # Python dependencies
├── notebooks/
│   └── nps_telecom_full.ipynb      # Full pipeline notebook
├── models/
│   ├── lgbm_final.pkl              # Production LightGBM (trained on respondents)
│   ├── feature_names.pkl           # 75 feature names in correct order
│   ├── scaler.pkl                  # StandardScaler fitted on the training set
│   ├── label_map.pkl               # Label encoding (0/1/2)
│   ├── label_names.pkl             # Class names
│   ├── monitoring_baselines.pkl    # Training distribution baselines (drift)
│   ├── monitoring_baselines.csv    # Same, human-readable
│   └── threshold_config.pkl        # Monitoring and retraining thresholds
├── data/
│   ├── telco_features.csv          # Encoded feature matrix (7043 customers)
│   ├── silent_base_predictions.csv # Ranked predictions for the silent base
│   ├── fairness_report.csv         # Fairness audit results
│   ├── shap_importance.csv         # SHAP feature importance
│   └── model_comparison.csv        # Model comparison table
├── figures/                        # All visualisations (18 charts)
├── verbatims/
│   └── synthetic_verbatims.csv     # 200 AI-generated customer notes
├── writeup.md                      # Technical write-up
├── README.md
├── .env.example
└── .gitignore
```

## Setup and Run

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn \
            lightgbm shap mord streamlit joblib textblob requests
```

### Run the full pipeline

Open `notebooks/nps_telecom_full.ipynb` in Google Colab
and run all cells in order. API keys (Mistral, TabPFN) are loaded
from Colab Secrets - never hardcoded.

### Launch Streamlit locally

```bash
streamlit run app.py
```

## Sections Covered

- NPS target construction with leakage analysis and sensitivity testing
- Dataset preparation and 15%/85% validation strategy
- Feature engineering (10 features built and justified)
- Synthetic verbatims generated with Mistral AI API
- Modelling and evaluation (4 model families, CV + production hold-out)
- Drivers of detraction by segment (actionable vs non-actionable)
- Fairness and bias audit (1 flag documented with mitigation options)
- Model persistence and Streamlit dashboard
- Monitoring and retraining proposal with saved baselines


## Author

Abdelhakim Moustapha Mahamat  
Master in Data Science from AIMS Rwanda  
Actuarial Science from Sorbonne
