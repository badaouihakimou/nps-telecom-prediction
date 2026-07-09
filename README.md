# Customer NPS Prediction for a Telecom Operator

**Dataset** : IBM Telco Customer Churn 11.1.3+  
**Live Demo** : https://nps-telecom-prediction.streamlit.app/  

## What is this project about ?

### The business problem

A telecom company (a business that sells phone and internet subscriptions)
sends satisfaction surveys to its customers every few months.
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
and no online security are often Detractors, then applies this
logic to predict the NPS of all customers whose satisfaction is unknown.

### Dataset
A table of data with rows and columns.
Here : 7043 rows (customers) and 50 columns (information about each customer).

### SHAP (SHapley Additive exPlanations)
A method that explains WHY the model made a specific prediction.
It measures the contribution of each feature (column) to the prediction.
Example : "This customer is predicted Detractor mainly because they have
a monthly contract (+0.42) and no online security (+0.21)."

### Feature Engineering
Creating new variables (columns) from existing ones to help the model.
Example : dividing Monthly Charge by number of services gives
charge_per_service a measure of value perception.

### Cross-Validation
A technique to evaluate the model on data it has never seen.
The data is split into 5 groups. The model trains on 4 groups and
is evaluated on the 5th. This is repeated 5 times. The average
performance gives a reliable estimate of real-world performance.

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
Lift of 1.33 = the model is 33% more efficient than random.

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
Satisfaction and churn are perfectly aligned in this dataset.

**Columns excluded to avoid leakage** :
- Satisfaction Score : directly builds the target
- Churn Score : too correlated with satisfaction
- Churn Label : reveals if the customer left
- Churn Category / Churn Reason : post-event information
- Customer Status : reveals the outcome
- Total Revenue : too correlated with CLTV


## Feature Engineering

10 new variables were created from the raw data :

| Feature | What it measures | Business logic |
|---|---|---|
| tenure_bucket | Customer age group (0-12m, 13-24m, etc.) | Loyal customers are more satisfied |
| charge_per_month_ratio | Total charges / tenure | High ratio = poor value perception |
| refund_rate | Refunds / total charges | High refunds = billing frustration |
| extra_charges_rate | Extra data charges / monthly charge | Unexpected costs damage satisfaction |
| long_distance_share | Long distance / total charges | High share = specific pain point |
| charge_per_service | Monthly charge / number of services | Value perception per service |
| nb_services | Count of active services | More services = more risk points |
| is_autopay | 1 if auto-pay, 0 if manual | Auto-pay = more engaged customer |
| household_size | Number of dependents + 1 | Families have more complex needs |
| has_referred | 1 if referred someone, 0 otherwise | Direct NPS proxy behaviour |


## Models Compared

Four machine learning models were trained and compared :

| Model | Balanced Accuracy | QWK | Detractor Recall | Violations |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.499 | 0.281 | 0.49 | 1483 |
| Ordinal Regression (mord) | 0.407 | 0.284 | 0.77 | 401 |
| **LightGBM (selected)** | **0.426** | **0.224** | **0.67** | **1027** |
| TabPFN (bonus, n=2000) | 0.342 | 0.036 | 0.96 | N/A |

### Why LightGBM was selected

1. Only model predicting all 3 classes reasonably
2. Logistic Regression misses 51% of Detractors unacceptable
3. Ordinal Regression never predicts Promoter (recall = 0.00)
4. TabPFN collapses to majority class without GPU
5. Native SHAP support for interpretability
6. Fast inference for production use

### What is LightGBM ?
Light Gradient Boosting Machine, developed by Microsoft.
It builds many small decision trees sequentially, each correcting
the mistakes of the previous one. Very fast and accurate on tabular data.

## Key Findings

### Top 5 Detraction Drivers (SHAP)

1. **Online Security_No** (0.215) : customers without online security
   feel exposed and underserved strongest detraction signal
2. **Contract_Month-to-Month** (0.200) : monthly contracts = low commitment,
   easy to switch to a competitor
3. **Monthly Charge** (0.184) : high charges damage value perception
4. **Avg Monthly GB Download** (0.167) : high data usage + dissatisfaction
   suggests service quality issues
5. **Number of Referrals** (0.152) : low referral count = unhappy customer

### Segment Analysis

| Segment | Detractor Rate |
|---|---|
| Contract Month-to-Month | 68.3% |
| Contract Two Year | 46.5% |
| Tenure 0-12 months | 67.5% |
| Tenure 49-72 months | 51.9% |
| Internet Fiber Optic | 67.4% |
| No Internet | 38.6% |

### Business Recommendation
For any predicted Detractor on a month-to-month contract :
offer a discounted annual contract as the first retention action.
This directly addresses the primary detraction driver.
*Caution : A/B testing is required to confirm causal impact.*

### Lift
Contacting the top 30% by predicted Detractor probability
captures **39.8% of all Detractors** — a 33% efficiency gain
over random targeting.

## Fairness Audit

The model allocates retention budget. We verify it treats
all demographic groups equally.

| Group | Recall Gap | Status |
|---|---|---|
| Dependents | 0.237 | FLAG Legal review required |
| Senior Citizen | 0.156 | FLAG Monitor post-deployment |
| Gender | 0.003 | OK |
| Married | 0.053 | OK |

**Dependents gap (0.237)** : the model misses 53% of Detractors
among customers with dependents. Families would be systematically
under-served. Must be resolved before production.

**Geographic features removed** : Latitude, Longitude and Population
were removed to avoid socio-economic discrimination. Accuracy cost : negligible.

## Synthetic Verbatims

200 synthetic customer interaction notes were generated using
the Mistral AI API (mistral-small-latest), conditioned on each
customer's profile (tenure, contract, charges, services).

15% of verbatims are intentionally noisy (tone opposite to NPS label)
to simulate real-world survey uncertainty.

Sentiment analysis (TextBlob) confirms the expected trend :
- Detractor mean polarity : -0.091 (negative)
- Passive mean polarity : +0.017 (neutral)
- Promoter mean polarity : +0.238 (positive)


## Silent Base Predictions

5,987 customers who never answered the NPS survey were scored :
- Predicted Detractors : 3,262 (54.5%)
- Predicted Passives : 1,612 (26.9%)
- Predicted Promoters : 1,113 (18.6%)

The predicted distribution matches the respondent distribution closely,
confirming the model is consistent and respondents are representative.

## Monitoring

5 key features are tracked weekly :
Tenure, Monthly Charge, nb_services, charge_per_month_ratio, CLTV.

Alert threshold : 2 standard deviations from training baseline.

Retraining triggers :
- 500+ new labeled survey responses collected
- QWK drops below 0.15 on new responses
- Any feature drifts beyond 2-sigma for 3 consecutive weeks

## Limitations

1. NPS label derived from Satisfaction Score, not a real NPS survey
2. Perfect alignment between satisfaction and churn is unrealistic in production
3. Passive class recall is weak across all models (F1 = 0.31)
4. Dependents fairness gap must be resolved before production
5. Verbatims are synthetic — real call-centre text would add more value
6. TabPFN could not be fully evaluated due to Colab memory constraints

## Repository Structure

```
nps-telecom-prediction/
├── app.py                          # Streamlit dashboard (5 pages)
├── requirements.txt                # Python dependencies
├── notebooks/
│   └── nps_telecom_full.ipynb     # Full pipeline notebook
├── models/
│   ├── lgbm_final.pkl             # Trained LightGBM model
│   ├── feature_names.pkl          # Feature names in correct order
│   ├── label_map.pkl              # Label encoding (0/1/2)
│   └── label_names.pkl            # Class names
├── data/
│   ├── telco_features.csv         # Encoded feature matrix (7043 customers)
│   ├── silent_base_predictions.csv # Predictions for silent base
│   ├── fairness_report.csv        # Fairness audit results
│   ├── shap_importance.csv        # SHAP feature importance
│   └── model_comparison.csv       # Model comparison table
├── figures/                        # All visualisations (16 charts)
├── verbatims/
│   └── synthetic_verbatims.csv    # 200 AI-generated customer notes
├── writeup.md                      # 5-page technical write-up
├── README.md
├── .env.example
└── .gitignore
```

## Setup and Run

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn \
            lightgbm shap imbalanced-learn mord streamlit joblib
```

### Run the full pipeline

Open `notebooks/nps_telecom_full.ipynb` in Google Colab
and run all cells in order.

### Launch Streamlit locally

```bash
streamlit run app.py
```

### Access the live demo

https://nps-telecom-prediction.streamlit.app/

## Sections Covered

- NPS target construction with leakage analysis and sensitivity testing
- Dataset preparation and 15%/85% validation strategy
- Feature engineering (10 features built and justified)
- Synthetic verbatims generated with Mistral AI API
- Modelling and evaluation (4 model families compared)
- Drivers of detraction by segment (actionable vs non-actionable)
- Fairness and bias audit (4 demographic groups)
- Model persistence and Streamlit dashboard
- Monitoring and retraining proposal


## Author

Abdelhakim Moustapha Mahamat  
Master in Data Science AIMS Rwanda  
Actuarial Science Sorbonne
