# Customer Profitability & Action Optimization Engine

## 1. Project Overview

Build a production-style **Customer Profitability & Action Optimization Engine for a Neobank**.

The system must estimate the historical and expected economic profitability of individual customers, calculate risk-adjusted Customer Lifetime Value (CLV), identify economically meaningful customer segments, simulate potential customer-level actions, and optimize the allocation of limited resources toward actions that maximize incremental economic profit.

The project should combine:

- Financial economics
- Customer profitability accounting
- Funds Transfer Pricing (FTP)
- Credit risk
- Econometrics
- Machine Learning
- Customer Lifetime Value
- Customer segmentation
- Scenario simulation
- Mathematical optimization
- Decision intelligence
- Explainable AI

The project must use **synthetic but realistic data** generated locally and must not depend on proprietary banking data.

The final result should be a complete analytical system rather than a single ML model.

---

# 2. Core Business Question

The primary business question is:

> **How much economic value does each customer generate for the neobank, how much value are they expected to generate in the future, and which actions should the bank take to maximize incremental customer profitability under financial, operational, and risk constraints?**

The system should answer four progressively more advanced questions:

### Question 1 — Historical profitability

> How profitable has each customer been?

### Question 2 — Future profitability

> How profitable is each customer expected to be?

### Question 3 — Customer Lifetime Value

> What is the expected discounted economic value of each customer over their remaining relationship with the bank?

### Question 4 — Decision optimization

> Given a limited budget and operational capacity, which actions should the bank take for which customers?

---

# 3. Important Conceptual Distinction

Do NOT build this as a simple revenue-ranking project.

The system must distinguish between:

```text
Revenue
    ↓
Gross Contribution
    ↓
Risk-Adjusted Contribution
    ↓
Economic Profit
    ↓
Expected Future Profit
    ↓
Customer Lifetime Value
    ↓
Incremental Value of Actions
    ↓
Optimal Resource Allocation
```

A customer generating high revenue is not necessarily a highly profitable customer.

The model must account for:

- Revenue
- Funding costs
- Expected credit losses
- Transaction costs
- Operational costs
- Acquisition costs
- Retention costs
- Product servicing costs
- Fraud-related costs where applicable

---

# 4. Target Portfolio

The synthetic neobank should contain several product families.

## 4.1 Current Account

Variables:

- average balance
- inflows
- outflows
- transaction count
- account tenure

## 4.2 Savings / Deposit Account

Variables:

- average balance
- deposit rate
- interest expense
- balance volatility
- deposit inflows
- deposit outflows

## 4.3 Debit / Credit Card

Variables:

- transaction volume
- transaction count
- interchange revenue
- international transaction volume
- ATM withdrawal volume
- payment processing costs

## 4.4 Consumer Credit

Variables:

- outstanding balance
- credit limit
- utilization
- interest rate
- interest income
- probability of default
- LGD
- EAD
- expected loss

## 4.5 Customer Service

Variables:

- number of support contacts
- average handling time
- service channel
- estimated servicing cost

---

# 5. High-Level Architecture

Implement the following architecture:

```text
                    SYNTHETIC RAW DATA
                           │
                           ▼
                  DATA QUALITY ENGINE
                           │
                           ▼
                     CUSTOMER 360
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
    HISTORICAL PROFITABILITY       BEHAVIORAL FEATURES
             │                           │
             ▼                           ▼
      PROFITABILITY ENGINE          ML MODELS
             │                 ┌─────────┼─────────┐
             │                 │         │         │
             │              Churn     Revenue    Balance
             │                 │         │         │
             └─────────────────┴─────────┴─────────┘
                               │
                               ▼
                    FUTURE PROFITABILITY
                               │
                               ▼
                     CUSTOMER LIFETIME
                         VALUE (CLV)
                               │
                               ▼
                   ECONOMIC SEGMENTATION
                               │
                               ▼
                     ACTION SIMULATION
                               │
                               ▼
                       OPTIMIZATION ENGINE
                               │
                               ▼
                     RECOMMENDED ACTIONS
                               │
                               ▼
                         STREAMLIT APP
```

---

# 6. Technology Stack

Use Python as the primary language.

## Data

- Python
- Pandas
- NumPy
- DuckDB
- PyArrow

## Econometrics

- Statsmodels
- SciPy

## Machine Learning

- Scikit-learn
- XGBoost

## Explainability

- SHAP

## Optimization

- OR-Tools
- scipy.optimize where appropriate

## Visualization

- Plotly
- Matplotlib

## Application

- Streamlit

## Testing

- Pytest

## Configuration

- YAML

## Development

- Git
- GitHub
- Makefile

Use type hints throughout the codebase.

Prefer modular, testable Python functions over large notebooks.

---

# 7. Repository Structure

Create the following repository structure:

```text
customer-profitability-engine/
│
├── README.md
├── PROJECT_SPEC.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── config/
│   └── settings.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_customer_profitability.ipynb
│   ├── 04_econometrics.ipynb
│   ├── 05_ml_models.ipynb
│   ├── 06_clv.ipynb
│   ├── 07_segmentation.ipynb
│   ├── 08_action_simulation.ipynb
│   ├── 09_optimization.ipynb
│   └── 10_final_analysis.ipynb
│
├── src/
│   └── customer_profitability/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── generator.py
│       │   ├── schemas.py
│       │   └── quality.py
│       │
│       ├── features/
│       │   ├── customer_features.py
│       │   ├── transaction_features.py
│       │   ├── credit_features.py
│       │   └── behavioral_features.py
│       │
│       ├── profitability/
│       │   ├── revenue.py
│       │   ├── funding_cost.py
│       │   ├── risk_cost.py
│       │   ├── operating_cost.py
│       │   ├── ftp.py
│       │   └── engine.py
│       │
│       ├── econometrics/
│       │   ├── revenue_models.py
│       │   ├── churn_models.py
│       │   └── diagnostics.py
│       │
│       ├── models/
│       │   ├── churn.py
│       │   ├── revenue.py
│       │   ├── balance.py
│       │   ├── product_adoption.py
│       │   └── evaluation.py
│       │
│       ├── clv/
│       │   ├── calculator.py
│       │   ├── forecasting.py
│       │   └── discounting.py
│       │
│       ├── segmentation/
│       │   ├── clustering.py
│       │   └── profiling.py
│       │
│       ├── actions/
│       │   ├── definitions.py
│       │   ├── simulator.py
│       │   └── incremental_value.py
│       │
│       ├── optimization/
│       │   ├── model.py
│       │   ├── constraints.py
│       │   └── solver.py
│       │
│       ├── simulation/
│       │   ├── monte_carlo.py
│       │   └── scenarios.py
│       │
│       └── utils/
│           ├── config.py
│           ├── logging.py
│           └── metrics.py
│
├── app/
│   ├── app.py
│   ├── pages/
│   │   ├── overview.py
│   │   ├── customer.py
│   │   ├── profitability.py
│   │   ├── clv.py
│   │   ├── segmentation.py
│   │   ├── actions.py
│   │   └── optimization.py
│   └── components/
│       ├── charts.py
│       ├── tables.py
│       └── filters.py
│
├── tests/
│   ├── test_data_generation.py
│   ├── test_data_quality.py
│   ├── test_profitability.py
│   ├── test_ftp.py
│   ├── test_risk_cost.py
│   ├── test_models.py
│   ├── test_clv.py
│   ├── test_segmentation.py
│   ├── test_actions.py
│   └── test_optimization.py
│
└── reports/
    ├── figures/
    └── final_report.md
```

---

# 8. Development Principles

Follow these principles throughout the project:

1. Build incrementally by phase.
2. Do not fabricate final business results.
3. Generate realistic synthetic data.
4. Make all randomness reproducible.
5. Use a global seed of `42`.
6. Keep business logic outside notebooks.
7. Notebooks should call functions from `src/`.
8. Every important calculation must have unit tests.
9. Avoid data leakage.
10. Separate training and test periods for predictive models.
11. Document assumptions.
12. Make financial calculations traceable.
13. Make model outputs explainable.
14. Do not recommend actions based solely on ML predictions.
15. Every optimization decision must be linked to an economic objective.

---

# PHASE 0 — Project Setup

## Objective

Create the project skeleton and development environment.

## Tasks

- Create repository structure.
- Configure `pyproject.toml`.
- Configure linting/formatting if desired.
- Configure pytest.
- Create `settings.yaml`.
- Create Makefile.
- Create package initialization files.
- Implement configuration loader.
- Implement logging utilities.
- Establish global random seed = 42.

## Acceptance Criteria

The following commands should work:

```bash
make install
make test
```

and:

```bash
python -m pytest
```

---

# PHASE 1 — Synthetic Data Generation

## Objective

Create a realistic longitudinal neobank dataset.

The dataset should represent at least:

- 20,000 customers
- 24–36 months of historical observations
- Multiple customer segments
- Multiple financial products

Prefer monthly customer-level snapshots for profitability calculations.

Generate transaction-level data separately where useful.

---

## 1.1 Customer Generation

Generate:

- customer_id
- age
- income
- country
- acquisition_channel
- acquisition_date
- customer_segment
- employment_status
- tenure

Avoid completely independent random variables.

Relationships should be realistic.

Examples:

- income should correlate with age
- credit limits should correlate with income
- balances should correlate with income
- premium customers should have higher balances
- rate-sensitive customers should react more strongly to deposit rates

---

## 1.2 Product Adoption

Generate probabilities for:

- current account
- savings account
- debit card
- credit card
- consumer loan
- investment account

Product adoption should depend on customer characteristics.

---

## 1.3 Deposits

Generate monthly:

- average_balance
- opening_balance
- closing_balance
- inflows
- outflows
- deposit_rate

Deposit balances should depend on:

- customer income
- customer segment
- offered rate
- market rate
- tenure
- historical balance
- seasonality

---

## 1.4 Card Transactions

Generate:

- transaction_count
- transaction_volume
- interchange_revenue
- ATM withdrawals
- international transactions

Transaction behavior should depend on:

- income
- age
- customer segment
- account tenure

---

## 1.5 Loans

Generate:

- loan_id
- outstanding_balance
- credit_limit
- utilization
- interest_rate
- interest_income
- PD
- LGD
- EAD

Credit risk should be correlated with relevant customer characteristics.

Do not generate PD completely independently.

---

## 1.6 Customer Service

Generate:

- support_contacts
- average_handling_time
- support_channel
- estimated_service_cost

Higher support usage should produce higher operating costs.

---

## 1.7 Churn

Generate customer churn over time.

Churn probability should depend on:

- customer satisfaction proxy
- transaction activity
- deposit rate competitiveness
- tenure
- product usage
- customer profitability
- recent balance changes

The churn-generating mechanism must be documented because the synthetic data is used for model validation.

---

## PHASE 1 ACCEPTANCE CRITERIA

The generated data must:

- contain no impossible financial values
- have realistic distributions
- contain temporal structure
- contain correlations between variables
- contain customer heterogeneity
- include some missing values and duplicate records for data-quality testing
- support all later modeling phases
- be reproducible with seed 42

Create:

```text
data/raw/customers.parquet
data/raw/accounts.parquet
data/raw/deposits.parquet
data/raw/cards.parquet
data/raw/loans.parquet
data/raw/customer_service.parquet
data/raw/customer_monthly.parquet
```

---

# PHASE 2 — Data Quality Engine

## Objective

Create a reusable data quality layer before any financial or ML analysis.

## Checks

### Structural

- expected columns
- data types
- unique identifiers
- date validity

### Missing Data

Detect:

- missing customer_id
- missing transaction values
- missing balances
- missing rates
- missing product information

Distinguish:

- expected missing
- suspicious missing
- critical missing

### Duplicates

Detect:

- duplicate customer records
- duplicate transactions
- duplicate monthly observations

### Financial Validity

Check:

```text
balance >= 0
transaction_amount >= 0 where applicable
interest_rate >= 0
PD ∈ [0,1]
LGD ∈ [0,1]
utilization ∈ [0,1]
```

### Temporal Consistency

Check:

- no observations after churn
- no transactions before customer acquisition
- no loan activity before loan origination
- no negative tenure

### Referential Integrity

Verify:

```text
customer_id in transactions
customer_id in accounts
customer_id in loans
```

## Output

Generate a Data Quality Report containing:

- total records
- duplicate count
- missingness
- invalid values
- temporal violations
- quality score

Do not silently drop bad records.

Support:

```text
RAW
VALIDATED
QUARANTINED
```

---

# PHASE 3 — Customer 360 Feature Engineering

## Objective

Create a unified analytical customer-level dataset.

Features should include:

### Demographic

- age
- income
- tenure
- segment

### Deposit

- average deposit balance
- balance volatility
- monthly inflows
- monthly outflows
- deposit rate
- rate differential vs market

### Transaction

- monthly transaction count
- transaction volume
- average transaction value
- card utilization
- ATM usage

### Credit

- outstanding balance
- utilization
- credit limit
- estimated PD
- estimated LGD
- expected loss

### Engagement

- login frequency
- product count
- transaction frequency
- support interactions

### Behavioral

- balance growth
- spending growth
- deposit growth
- recent activity
- activity volatility

Use rolling features where appropriate:

```text
3-month average
6-month average
12-month average
```

Ensure all predictive features use information available before the prediction date.

---

# PHASE 4 — Historical Customer Profitability Engine

## Objective

Calculate actual/historical economic profitability for each customer and month.

---

## 4.1 Revenue

Calculate:

### Interest Revenue

\[
InterestRevenue =
AverageLoanBalance \times LoanRate
\]

### Fee Revenue

Include:

- account fees where applicable
- loan fees
- transfer fees
- premium subscription fees

### Interchange Revenue

\[
InterchangeRevenue =
CardTransactionVolume \times InterchangeRate
\]

### Other Revenue

Include only clearly defined synthetic revenue streams.

---

# 4.2 Deposit Funding Cost

Implement a Funds Transfer Pricing mechanism.

Define:

```text
Customer Deposit Rate
Internal FTP Rate
Market Reference Rate
```

The bank's economic value from deposits should reflect the difference between the internal funding value and the rate paid to customers.

For example:

\[
DepositContribution =
AverageDepositBalance
\times
(FTPRate - CustomerDepositRate)
\]

Document the sign convention carefully.

---

# 4.3 Credit Risk Cost

Calculate:

\[
ExpectedLoss =
PD \times LGD \times EAD
\]

Where:

- PD = Probability of Default
- LGD = Loss Given Default
- EAD = Exposure at Default

For monthly accounting, use an appropriate monthly allocation.

---

# 4.4 Operating Costs

Calculate customer-level operating costs from:

- customer service
- payment processing
- card servicing
- account servicing
- operational activity

Example:

\[
SupportCost =
Contacts \times CostPerContact
\]

---

# 4.5 Acquisition Cost

Allocate Customer Acquisition Cost (CAC) to customers.

Possible channels:

- organic
- paid search
- referral
- partnership
- advertising

The CAC should vary by acquisition channel.

---

# 4.6 Economic Profit

Define:

\[
EconomicProfit =
TotalRevenue
+
DepositContribution
-
ExpectedLoss
-
OperatingCost
-
AcquisitionCost
\]

Create:

- monthly profit
- quarterly profit
- annual profit

---

# 4.7 Profitability Metrics

For each customer calculate:

- total revenue
- total costs
- economic profit
- profit margin
- revenue per month
- cost per month
- risk-adjusted profit
- contribution margin

---

# PHASE 5 — Econometric Analysis

## Objective

Use econometrics to understand economic relationships rather than simply maximize predictive accuracy.

---

## 5.1 Revenue Drivers

Estimate models explaining customer revenue.

Example:

\[
Revenue_{i,t}
=
\beta_0
+
\beta_1 Balance_{i,t}
+
\beta_2 Transactions_{i,t}
+
\beta_3 Products_{i,t}
+
\beta_4 Income_i
+
\epsilon_{i,t}
\]

Interpret coefficients.

---

## 5.2 Churn Drivers

Estimate a logistic model:

\[
P(Churn_i=1)
=
\sigma(
\beta_0
+
\beta_1 RateGap_i
+
\beta_2 Activity_i
+
\beta_3 Tenure_i
+
\beta_4 Profitability_i
)
\]

Interpret the signs and economic meaning of coefficients.

---

## 5.3 Econometric Diagnostics

Include:

- multicollinearity checks
- residual analysis
- heteroskedasticity checks
- goodness of fit
- coefficient significance
- confidence intervals

Do not treat statistical significance as economic significance.

---

# PHASE 6 — Machine Learning Models

## Objective

Predict future customer behavior and future economic outcomes.

ML should complement, not replace, the economic models.

---

## 6.1 Churn Model

Predict:

\[
P(Churn_{i,t+1}=1)
\]

Models:

- Logistic Regression baseline
- Random Forest
- XGBoost

Evaluate:

- ROC-AUC
- PR-AUC
- precision
- recall
- calibration
- Brier score

Calibration is particularly important because probabilities will feed CLV calculations.

---

# 6.2 Future Revenue Model

Predict:

\[
ExpectedRevenue_{i,t+1}
\]

Compare:

- linear regression baseline
- Random Forest
- Gradient Boosting / XGBoost

Metrics:

- MAE
- RMSE
- MAPE where appropriate
- R²

---

# 6.3 Future Balance Model

Predict future:

- deposit balance
- loan balance
- transaction volume

Use time-based train/test splits.

---

# 6.4 Product Adoption Model

Predict probability that a customer adopts an additional product.

Examples:

\[
P(SavingsAccount)
\]

\[
P(Loan)
\]

\[
P(InvestmentProduct)
\]

These predictions will later support action simulation.

---

# 6.5 Explainability

Use SHAP for tree-based models.

For every major model provide:

- global feature importance
- SHAP summary
- individual customer explanation

Avoid presenting ML predictions without explaining major drivers.

---

# PHASE 7 — Customer Lifetime Value

## Objective

Estimate the expected discounted economic value of each customer.

---

## 7.1 Future Profit Forecast

For each customer:

\[
ExpectedProfit_{i,t}
=
ExpectedRevenue_{i,t}
-
ExpectedCosts_{i,t}
\]

Incorporate:

- churn probability
- expected balances
- expected transactions
- expected credit risk
- expected operating costs

---

## 7.2 Churn-Adjusted Profit

If:

\[
P(Survival_{i,t})
\]

represents the probability the customer remains active, then:

\[
ExpectedProfit_{i,t}^{adj}
=
P(Survival_{i,t})
\times
ExpectedProfit_{i,t}
\]

---

## 7.3 Discounting

Use:

\[
PV_t =
\frac{ExpectedProfit_t}{(1+r)^t}
\]

where `r` is the configurable discount rate.

---

## 7.4 CLV

Calculate:

\[
CLV_i =
HistoricalEconomicProfit_i
+
\sum_{t=1}^{T}
\frac{
P(Survival_{i,t})
\times
ExpectedProfit_{i,t}
}{
(1+r)^t
}
\]

Clearly document whether historical profit is included in the displayed CLV or whether CLV refers exclusively to future value.

Prefer maintaining both:

- Historical Economic Profit
- Future CLV
- Total Customer Economic Value

---

## 7.5 CLV Uncertainty

Do not present CLV as an exact number.

Create:

- base estimate
- lower confidence/scenario estimate
- upper confidence/scenario estimate

Use Monte Carlo simulation for uncertainty where appropriate.

---

# PHASE 8 — Economic Customer Segmentation

## Objective

Identify economically meaningful customer groups.

Do not cluster solely on demographics.

Use variables such as:

- historical profit
- CLV
- revenue
- funding contribution
- expected loss
- engagement
- transaction volume
- product count
- churn probability
- balance
- risk

Standardize features before clustering.

Evaluate:

- K-Means
- hierarchical clustering where useful

Use silhouette score and business interpretability.

---

## Expected Example Segments

Possible names:

- High Value
- Deposit Funders
- Credit-Driven
- Transactional
- Low Profitability
- High Cost
- At-Risk High Value

Do not force these exact labels if the data produces different clusters.

The final segment labels must be based on measurable cluster characteristics.

---

# PHASE 9 — Customer Action Framework

## Objective

Move from customer analysis to decision-making.

Define a set of possible actions.

Examples:

### Action A — Retention Incentive

Offer a temporary incentive to reduce churn.

### Action B — Savings Cross-Sell

Offer a savings product.

### Action C — Credit Product

Offer a credit product subject to risk constraints.

### Action D — Investment Product

Offer an investment product.

### Action E — Premium Subscription

Offer a premium banking plan.

### Action F — No Action

Always include:

```text
NO_ACTION
```

This is critical.

The system must compare every intervention against doing nothing.

---

# PHASE 10 — Action Impact Simulation

## Objective

Estimate the incremental economic value of each possible action.

For each customer/action pair estimate:

\[
IncrementalProfit_{i,a}
=
ExpectedProfit_{i,a}
-
ExpectedProfit_{i,NoAction}
-
ActionCost_{i,a}
\]

Examples:

```text
Customer A
Retention offer
Expected incremental revenue = €100
Expected cost = €20
Incremental profit = €80
```

---

## Action Simulation Components

For each action model:

- probability of acceptance
- probability of churn reduction
- additional revenue
- additional balance
- additional risk
- operational cost
- incentive cost

Do not assume every accepted action creates identical value.

---

# PHASE 11 — Optimization Engine

## Objective

Select the portfolio of customer actions that maximizes incremental economic profit.

Define:

\[
\max
\sum_{i}
\sum_{a}
x_{i,a}
\cdot
IncrementalProfit_{i,a}
\]

where:

\[
x_{i,a}\in\{0,1\}
\]

---

## Constraints

Include:

### Budget

\[
\sum_{i,a}
x_{i,a}Cost_{i,a}
\le Budget
\]

### Operational Capacity

\[
\sum_{i,a}x_{i,a}
\le Capacity
\]

### Risk Constraints

Do not select actions that violate defined risk thresholds.

### One Action Per Customer

\[
\sum_a x_{i,a}\le1
\]

### Eligibility

Some actions should only be available to eligible customers.

---

## Optimization Outputs

Return:

- selected customers
- selected actions
- action cost
- incremental revenue
- incremental profit
- total budget consumed
- remaining budget
- expected CLV impact
- expected retention impact
- risk impact

Also calculate the counterfactual:

```text
No optimization
vs
Optimized allocation
```

Do not claim the optimized strategy is causal unless the underlying action-effect estimates support such a conclusion.

---

# PHASE 12 — Monte Carlo Simulation

## Objective

Quantify uncertainty in:

- churn
- future revenue
- future balances
- expected losses
- CLV
- action outcomes
- total portfolio profit

Run at least:

```text
10,000 simulations
```

where computationally feasible.

---

## Scenarios

Create:

### Base Case

Normal economic environment.

### Downside

- higher churn
- lower transaction activity
- lower balances
- higher credit losses

### Upside

- lower churn
- stronger product adoption
- higher balances

### Stress

- high churn
- deposit outflows
- increased credit losses
- reduced customer activity

---

## Outputs

Calculate:

- expected profit
- P5
- P50
- P95
- probability of negative incremental profit
- probability of budget overrun
- CLV distribution

---

# PHASE 13 — Streamlit Decision Intelligence Dashboard

## Objective

Build an executive-quality dashboard.

The dashboard should not simply display charts.

It should support decision-making.

---

# Page 1 — Executive Overview

Display:

- Total Customers
- Total Economic Profit
- Average Customer Profit
- Total CLV
- Average CLV
- % Profitable Customers
- % Loss-Making Customers

Charts:

- Profitability distribution
- CLV distribution
- Profit by segment
- CLV by segment
- Revenue vs cost

---

# Page 2 — Customer 360

Allow selection of a customer.

Display:

- customer profile
- products
- balances
- revenue
- costs
- economic profit
- risk
- churn probability
- CLV
- segment

Include a timeline of historical profitability.

---

# Page 3 — Profitability

Display:

- revenue decomposition
- funding contribution
- risk cost
- operating cost
- acquisition cost
- economic profit

Allow filtering by:

- segment
- product
- acquisition channel
- profitability band

---

# Page 4 — CLV

Display:

- CLV distribution
- historical vs future value
- survival probability
- expected future profit
- uncertainty intervals

---

# Page 5 — Segmentation

Display:

- cluster sizes
- segment profitability
- CLV
- churn
- risk
- product usage

Use a 2D visualization such as:

```text
Profitability
      ↑
      │       High Value
      │
      │
      │
      └────────────────────→ CLV
```

---

# Page 6 — Actions

For a selected customer show:

```text
Recommended Actions

Action              Cost    ΔProfit    ΔCLV
------------------------------------------------
Retention           €20     +€80       +€120
Savings             €5      +€35       +€50
Credit Product      €15     +€90       +€130
No Action           €0       €0          €0
```

The displayed recommendation must come from the action simulation and optimization logic.

---

# Page 7 — Optimization

Allow user inputs:

- budget
- operational capacity
- maximum risk
- minimum expected ROI

Run optimization.

Display:

- selected actions
- budget utilization
- expected incremental profit
- customers targeted
- action distribution
- segment distribution

---

# Page 8 — Scenario Analysis

Allow users to select:

- Base
- Downside
- Upside
- Stress

Display:

- CLV
- churn
- profit
- expected loss
- incremental action value

---

# PHASE 14 — Explainability & Governance

## Objective

Make the system interpretable.

Document:

### Profitability

How every component of economic profit is calculated.

### CLV

How future value is forecast.

### ML

Why a model produces a particular prediction.

### Optimization

Why a particular customer/action combination was selected.

---

## Customer-Level Decision Explanation

For every recommended action provide:

```text
Customer X

Recommended Action:
Retention Incentive

Why:
- High expected CLV
- Elevated churn probability
- Strong historical profitability
- Positive incremental value

Expected impact:
- Cost: €X
- Incremental profit: €Y
- CLV impact: €Z
```

---

# PHASE 15 — Model Validation

Every model must have:

## Baseline

Compare sophisticated models against simple baselines.

Examples:

```text
Churn:
Logistic Regression vs XGBoost

Revenue:
Linear Regression vs XGBoost
```

## Time-Based Validation

Avoid random splits for temporal prediction.

Use:

```text
Train: Months 1–18
Validation: Months 19–21
Test: Months 22–24
```

Adjust based on the generated dataset.

---

# PHASE 16 — Testing

Create extensive tests.

## Data

Test:

- schema
- missing values
- duplicates
- temporal consistency
- referential integrity

## Financial Calculations

Test:

- interest revenue
- interchange revenue
- FTP
- expected loss
- operating costs
- economic profit

Use hand-calculated examples.

---

## CLV

Test:

- discounting
- survival adjustment
- future profit aggregation
- edge cases

---

## Optimization

Test:

- budget constraint
- capacity constraint
- one action per customer
- ineligible customers
- zero-budget case
- zero-capacity case

The optimizer must never violate constraints.

---

# PHASE 17 — Final Report

Create:

```text
reports/final_report.md
```

The report should contain:

## 1. Executive Summary

Explain:

- business problem
- methodology
- main findings

## 2. Customer Economics

Explain:

- revenue
- cost
- profitability

## 3. Risk-Adjusted Profitability

Explain:

- expected loss
- credit risk
- risk-adjusted contribution

## 4. CLV

Explain:

- methodology
- assumptions
- uncertainty

## 5. Segmentation

Explain economic customer segments.

## 6. Action Optimization

Explain:

- action framework
- incremental value
- optimization formulation

## 7. Scenario Analysis

Explain:

- base
- upside
- downside
- stress

## 8. Limitations

Clearly state:

- synthetic data
- modeling assumptions
- causal limitations
- potential model risk
- simplified FTP assumptions
- simplified customer behavior

---

# PHASE 18 — README

Create a polished README suitable for a professional Data Science / Economics portfolio.

The README should emphasize:

## Business Problem

What economic decision are we solving?

## Solution

Explain:

```text
Customer 360
→ Profitability
→ Risk Adjustment
→ ML
→ CLV
→ Segmentation
→ Action Simulation
→ Optimization
```

## Key Technical Components

Highlight:

- Customer profitability
- Funds Transfer Pricing
- Expected Credit Loss
- Econometrics
- Machine Learning
- CLV
- Monte Carlo
- Optimization

## Mathematical Formulation

Include the key equations.

## Architecture

Include a Mermaid architecture diagram.

## Dashboard

Include screenshots if available.

## Reproducibility

Explain how to generate the data and run the pipeline.

---

# 19. Key Mathematical Framework

The project should ultimately connect the following equations.

## Revenue

\[
Revenue =
InterestRevenue
+
FeeRevenue
+
InterchangeRevenue
\]

## Deposit Contribution

\[
DepositContribution =
DepositBalance
\times
(FTPRate - DepositRate)
\]

## Expected Loss

\[
EL =
PD\times LGD\times EAD
\]

## Economic Profit

\[
EconomicProfit =
Revenue
+
DepositContribution
-
ExpectedLoss
-
OperatingCost
-
AcquisitionCost
\]

## Future Profit

\[
ExpectedProfit_t =
ExpectedRevenue_t
-
ExpectedCost_t
\]

## CLV

\[
CLV =
HistoricalProfit
+
\sum_{t=1}^{T}
\frac{
P(Survival_t)
\times
ExpectedProfit_t
}{
(1+r)^t
}
\]

## Incremental Action Value

\[
IncrementalProfit_{i,a}
=
ExpectedProfit_{i,a}
-
ExpectedProfit_{i,NoAction}
-
ActionCost_{i,a}
\]

## Optimization

\[
\max
\sum_i\sum_a
x_{i,a}
IncrementalProfit_{i,a}
\]

subject to:

\[
\sum_i\sum_a
x_{i,a}Cost_{i,a}
\le Budget
\]

\[
\sum_a x_{i,a}\le1
\]

plus eligibility, risk and capacity constraints.

---

# 20. Configuration

Create:

```yaml
seed: 42

data:
  n_customers: 20000
  n_months: 36

models:
  test_months: 6
  validation_months: 3

clv:
  horizon_months: 36
  annual_discount_rate: 0.08

simulation:
  n_simulations: 10000

optimization:
  default_budget: 100000
  default_capacity: 5000
```

All important parameters should be configurable.

Do not hard-code business assumptions throughout the codebase.

---

# 21. Reproducibility

The entire project should be reproducible from scratch.

A new user should be able to run:

```bash
make install
make generate-data
make quality
make features
make models
make clv
make optimize
make test
make app
```

Define these commands in the Makefile.

---

# 22. Expected Final Workflow

The final workflow should be:

```text
1. Generate synthetic data
        ↓
2. Validate data
        ↓
3. Build Customer 360
        ↓
4. Calculate historical profitability
        ↓
5. Estimate economic relationships
        ↓
6. Train ML models
        ↓
7. Forecast future customer economics
        ↓
8. Calculate CLV
        ↓
9. Segment customers economically
        ↓
10. Simulate possible actions
        ↓
11. Estimate incremental value
        ↓
12. Optimize action allocation
        ↓
13. Run Monte Carlo scenarios
        ↓
14. Visualize decisions
        ↓
15. Produce final report
```

---

# 23. Implementation Order

Claude Code should implement the project in the following exact order.

### Phase 0
Project infrastructure.

### Phase 1
Synthetic data generation.

### Phase 2
Data quality.

### Phase 3
Customer 360.

### Phase 4
Historical profitability engine.

### Phase 5
Econometric analysis.

### Phase 6
Machine Learning.

### Phase 7
CLV.

### Phase 8
Economic segmentation.

### Phase 9
Action definitions.

### Phase 10
Action simulation.

### Phase 11
Optimization.

### Phase 12
Monte Carlo.

### Phase 13
Streamlit dashboard.

### Phase 14
Explainability and governance.

### Phase 15
Testing and validation.

### Phase 16
Final report.

### Phase 17
README and portfolio polish.

Do not jump directly to the dashboard or optimization before the underlying financial calculations and models are implemented and tested.

---

# 24. Definition of Done

The project is complete when:

- Synthetic data can be generated from scratch.
- Data quality checks run successfully.
- Customer-level profitability is calculated.
- FTP is implemented and documented.
- Expected credit loss is calculated.
- Historical economic profit is available.
- Econometric relationships are estimated and interpreted.
- Churn prediction is implemented.
- Future revenue/balance predictions are implemented.
- CLV is calculated.
- CLV uncertainty is quantified.
- Economic customer segments are identified.
- Customer actions are simulated.
- Incremental action value is calculated.
- An optimization model allocates actions under constraints.
- Monte Carlo scenarios are available.
- Streamlit dashboard is functional.
- Explainability is available.
- Tests cover core business logic.
- Results are reproducible.
- No business metrics are fabricated.
- README explains the project clearly.
- Final report documents assumptions and limitations.

---

# 25. Important Modeling Philosophy

This project is not intended to demonstrate that one ML algorithm is better than another.

The primary objective is to demonstrate the ability to translate:

```text
Economic Problem
        ↓
Financial Measurement
        ↓
Statistical Modeling
        ↓
Machine Learning
        ↓
Forecasting
        ↓
Optimization
        ↓
Business Decision
```

The final system should therefore prioritize:

1. Economic coherence
2. Financial correctness
3. Interpretability
4. Reproducibility
5. Predictive performance
6. Optimization quality

rather than optimizing predictive metrics in isolation.

---

# 26. Responsible Decision-Making

The system must not use sensitive personal characteristics to determine customer value or treatment.

Clearly document that:

- the dataset is synthetic
- profitability estimates are model-based
- ML predictions contain uncertainty
- optimization outputs are decision-support tools
- causal claims require appropriate experimental or quasi-experimental evidence
- real-world deployment would require additional regulatory, legal, fairness, and model-risk controls

The project should demonstrate **decision intelligence**, not automated unquestionable decision-making.

---

# 27. Final Portfolio Positioning

The project should ultimately be presented as:

> **A customer-level economic decision engine for a neobank that combines profitability accounting, Funds Transfer Pricing, credit risk, econometrics, machine learning, Customer Lifetime Value, simulation, and mathematical optimization to determine where customer value comes from and how a bank can allocate limited resources to maximize incremental economic profit.**

The key differentiator is:

> **The system does not only predict customer behavior. It translates predictions into economic value and then into constrained business decisions.**
