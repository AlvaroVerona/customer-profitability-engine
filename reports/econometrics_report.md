# Econometric Analysis Report

Explanatory (same-month) models -- see module docstrings in `src/customer_profitability/econometrics/` for why these are not point-in-time forecasters. Statistical significance (p < 0.05) and economic significance (effect size vs. an analyst-set threshold) are reported as separate columns; do not conflate them.

## 5.1 Revenue drivers (OLS)

n = 384183, R² = 0.2069, adjusted R² = 0.2069

| index | coef | std_err | statistic | p_value | ci_lower | ci_upper | significant_p05 | economic_threshold | economically_significant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| const | -54.177 | 0.2935 | -184.5824 | 0.0 | -54.7523 | -53.6017 | True | inf | False |
| average_balance | -0.0001 | 0.0 | -5.9613 | 0.0 | -0.0002 | -0.0001 | True | 0.01 | False |
| transaction_count | 0.1449 | 0.0229 | 6.3192 | 0.0 | 0.1 | 0.1898 | True | 0.5 | False |
| product_count | 15.4408 | 0.0829 | 186.1915 | 0.0 | 15.2782 | 15.6033 | True | 1.0 | True |
| income | 0.0005 | 0.0 | 52.5886 | 0.0 | 0.0004 | 0.0005 | True | 0.0005 | False |

**VIF (multicollinearity):**

| index | vif |
| --- | --- |
| average_balance | 2.5031 |
| transaction_count | 2.8801 |
| product_count | 1.2953 |
| income | 4.5728 |

**Breusch-Pagan heteroskedasticity test:** LM p-value = 0 (heteroskedastic at 5%) -- the model is fit with HC3 heteroskedasticity-robust standard errors for exactly this reason (see `revenue_models.fit_revenue_model` docstring); the coefficients/p-values above already reflect that correction.

**Residual normality (Jarque-Bera):** p-value = 0

**Interpretation:**

- A one-unit increase in `average_balance` is associated with a -0.0001 change in monthly revenue (decreases revenue), holding the other drivers fixed; statistically significant (p=2.503e-09).
- A one-unit increase in `transaction_count` is associated with a +0.1449 change in monthly revenue (increases revenue), holding the other drivers fixed; statistically significant (p=2.629e-10).
- A one-unit increase in `product_count` is associated with a +15.4408 change in monthly revenue (increases revenue), holding the other drivers fixed; statistically significant (p=0).
- A one-unit increase in `income` is associated with a +0.0005 change in monthly revenue (increases revenue), holding the other drivers fixed; statistically significant (p=0).

## 5.2 Churn drivers (Logistic regression)

n = 384306, churn rate = 0.0190, McFadden pseudo-R² = 0.0595

| index | coef | std_err | statistic | p_value | ci_lower | ci_upper | significant_p05 | economic_threshold | economically_significant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| const | -2.651 | 0.0914 | -28.9898 | 0.0 | -2.8302 | -2.4717 | True | inf | False |
| rate_gap | 0.378 | 0.0359 | 10.5264 | 0.0 | 0.3076 | 0.4483 | True | 0.05 | True |
| activity | -0.0738 | 0.0032 | -22.7301 | 0.0 | -0.0801 | -0.0674 | True | 0.02 | True |
| tenure | -0.0185 | 0.0006 | -32.6531 | 0.0 | -0.0196 | -0.0174 | True | 0.01 | True |
| profitability | -0.0067 | 0.0006 | -10.8955 | 0.0 | -0.0079 | -0.0055 | True | 0.001 | True |

**VIF (multicollinearity):**

| index | vif |
| --- | --- |
| rate_gap | 1.9519 |
| activity | 1.8856 |
| tenure | 1.0978 |
| profitability | 1.339 |

**Interpretation:**

- A one-unit increase in `rate_gap` (1 percentage point of (market_rate - deposit_rate)) increases the odds of churn by a factor of 1.4593 (log-odds coef=+0.3780), holding the other drivers fixed; statistically significant (p=6.53e-26).
- A one-unit increase in `activity` (1 login/month) decreases the odds of churn by a factor of 0.9289 (log-odds coef=-0.0738), holding the other drivers fixed; statistically significant (p=2.258e-114).
- A one-unit increase in `tenure` (1 month of tenure) decreases the odds of churn by a factor of 0.9817 (log-odds coef=-0.0185), holding the other drivers fixed; statistically significant (p=7.252e-234).
- A one-unit increase in `profitability` (1 EUR of monthly economic profit) decreases the odds of churn by a factor of 0.9934 (log-odds coef=-0.0067), holding the other drivers fixed; statistically significant (p=1.211e-27).

