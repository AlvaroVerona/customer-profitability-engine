# Final Report -- Customer Profitability & Action Optimization Engine

## 1. Executive Summary

**Business problem.** A neobank does not know, at the customer level, where its profit actually comes from, which customers are worth investing in, or how to allocate a limited retention/growth budget across millions of possible customer-action pairs. This project builds a synthetic (but internally consistent, seed-42-reproducible) neobank dataset and a full analytical pipeline that answers exactly that: **which customers are profitable, why, what they are worth going forward, and which actions the bank should fund given a hard budget/capacity/risk constraint.**

**Methodology.** The pipeline runs Customer 360 -> historical profitability (revenue, Funds Transfer Pricing, Expected Credit Loss, operating cost) -> econometric driver analysis -> machine-learned forecasts (churn, revenue, balances, product adoption) -> Customer Lifetime Value with Monte Carlo uncertainty -> economic segmentation -> action simulation -> a CP-SAT constrained-optimization allocation engine -> portfolio-level Monte Carlo scenario stress-testing -> a Streamlit decision-support dashboard with customer-level decision explanations.

**Main findings, from the current pipeline run (not illustrative numbers):**

- Data quality: 99.6/100 overall score across all raw tables (every defective row is quarantined and retained, never silently dropped).
- 20,000 customers, 94.3% individually profitable, EUR 11,291,852 total historical economic profit.
- 12,651 of 20,000 customers still active, with EUR 21,778,352 total customer economic value (historical profit already booked + discounted, survival-weighted future profit).
- The optimization engine, under the current budget/capacity/risk constraints, selects 5,000 of 44,406 candidate customer/action pairs for EUR 13,486 spent, generating EUR 1,380,050 of incremental profit (status: OPTIMAL).

## 2. Customer Economics

Every customer-month's economics follow one waterfall (`profitability/engine.py`): `gross_contribution = total_revenue + deposit_contribution - operating_cost`, `risk_adjusted_contribution = gross_contribution - expected_loss`, `economic_profit = risk_adjusted_contribution - acquisition_cost` (acquisition cost recognized once, in the acquisition month, and only if that month falls inside the observed data window).

**Revenue composition** (sum across all customer-months observed):

- Interest revenue: EUR 9,195,105 (76.0% of revenue)
- Interchange revenue: EUR 2,341,907 (19.4% of revenue)
- Fee revenue (low-balance + Premium subscription): EUR 554,728 (4.6% of revenue)
- Deposit (FTP) contribution: EUR 5,459,838 (the bank's internal funding value of customer deposits, net of what is paid to the customer)

**Cost composition:**

- Operating cost (support + card processing + account servicing): EUR 2,061,076
- Acquisition cost (recognized once per customer, in-window only): EUR 30,670

**Bottom line:** EUR 15,490,502 total gross contribution, 94.3% of the 20,000 customers individually economic-profit-positive over their observed history -- profitability is concentrated, not evenly spread (see Segmentation, section 5).

## 3. Risk-Adjusted Profitability

`expected_loss = PD x LGD x EAD`, summed over every loan a customer holds each month (`profitability/risk_cost.py`). Across 103,773 loan-month observations: mean PD 0.71%/month, mean LGD 44.3%, mean expected-loss rate 0.31%/month of outstanding balance, on EUR 1,237,553,566 total outstanding loan balance in the current book.

Expected credit loss (EUR 4,167,980 total) is the single largest cost line ahead of operating cost (EUR 2,061,076) -- credit risk, not servicing cost, is what most separates a profitable customer from an unprofitable one in this dataset. Risk-adjusted contribution (EUR 11,322,522) is gross contribution minus this expected loss, before acquisition cost -- the single number every downstream stage (CLV, action value, optimization) actually optimizes against, not raw revenue.

## 4. Customer Lifetime Value

`CLV_i = HistoricalEconomicProfit_i + sum_t [P(Survival_t) x ExpectedProfit_t] / (1+r)^t` (`clv/calculator.py`), reported as three separate columns -- `historical_economic_profit`, `future_clv`, `total_customer_economic_value` -- so a reader is never left guessing which one a plain "CLV" figure means.

**Methodology and assumptions:**

- Survival probability comes from Phase 6's *calibrated* churn model (Platt scaling on top of a class-rebalanced classifier, since rebalancing needed for usable ranking on a ~2%/month event distorts raw `predict_proba`), evaluated once on each customer's latest snapshot and held as a **constant monthly hazard** over the horizon -- a documented simplification, not a month-by-month re-forecast.
- `ExpectedProfit` is each customer's own trailing 3-month run-rate across the Phase 4 waterfall, held **flat** over the horizon, consistent with the flat-hazard assumption.
- A Monte Carlo band (`clv_p5`/`clv_p50`/`clv_p95`) exists specifically so CLV is never presented as one exact number: it draws stochastic churn timing (geometric survival draw) and profit-level uncertainty (log-normal multiplier sized by each customer's own historical profit volatility).

**Portfolio totals (current run):**

- 12,651 of 20,000 customers active.
- Total historical economic profit already booked: EUR 11,291,852.
- Total future CLV (discounted, survival-weighted): EUR 10,486,500.
- Total customer economic value: EUR 21,778,352.

## 5. Economic Segmentation

K-Means on standardized economic features (never demographics), k=7 selected under a documented business-interpretability floor of 5 segments over the raw silhouette-argmax (silhouette 0.223) -- the raw metric alone picked a coarser k that collapsed clearly distinct segments together.

| Segment | Customers | Avg. historical profit | Avg. total CLV | Avg. churn risk/mo. |
|---|---:|---:|---:|---:|
| High Value I | 247 | EUR 5,092 | EUR 9,211 | 0.5% |
| High Value II | 1,144 | EUR 2,718 | EUR 5,219 | 0.7% |
| Credit-Driven | 1,185 | EUR 757 | EUR 1,867 | 1.9% |
| Deposit Funders | 2,524 | EUR 1,022 | EUR 1,844 | 0.6% |
| Transactional | 1,656 | EUR 331 | EUR 1,512 | 1.2% |
| Balanced Segment 0 | 3,808 | EUR 321 | EUR 588 | 1.9% |
| Balanced Segment 2 | 2,083 | EUR 46 | EUR 164 | 5.5% |

The two "High Value" segments hold a disproportionate share of total customer economic value despite being the smallest by headcount -- the same concentration pattern visible in the Customer Economics section, now attributable to specific, behaviorally-defined groups rather than an undifferentiated "profitable customers" bucket.

## 6. Action Optimization

**Action framework** (`actions/definitions.py`): 5 real actions (retention incentive, savings cross-sell, credit product, investment product, Premium subscription) plus NO_ACTION as an explicit, always-eligible baseline. Eligibility is behavioral/financial only -- product ownership, income, employment status for credit risk -- never a protected demographic attribute.

**Simulated incremental value per action** (mean over eligible active customers):

| Action | Eligible customers | Acceptance prob. | Mean incremental profit | Total incremental profit |
|---|---:|---:|---:|---:|
| Credit Product | 8,170 | 30% | EUR 470.55 | EUR 3,844,363 |
| Investment Product | 8,161 | 20% | EUR 24.71 | EUR 201,622 |
| Premium Subscription | 10,921 | 25% | EUR 69.26 | EUR 756,392 |
| Retention Incentive | 12,651 | 35% | EUR 8.16 | EUR 103,184 |
| Savings Cross-Sell | 5,791 | 40% | EUR 21.85 | EUR 126,531 |

**Optimization formulation** (`optimization/model.py`, exact CP-SAT integer programming, not a heuristic): maximize total incremental profit subject to one-action-per-customer, a budget cap, an operational capacity cap, and a maximum incremental monthly risk cap.

**Current run:** status OPTIMAL, 5,000 of 44,406 candidates selected, EUR 13,486 spent (EUR 86,514 budget remaining), EUR 1,380,050 total incremental profit, ~0.55 expected monthly churns averted.

| Selected action | Count |
|---|---:|
| PREMIUM_SUBSCRIPTION | 3,577 |
| CREDIT_PRODUCT | 1,150 |
| SAVINGS_CROSS_SELL | 182 |
| RETENTION_INCENTIVE | 91 |

## 7. Scenario Analysis

Portfolio-level Monte Carlo simulation (`simulation/monte_carlo.py`) re-draws action acceptance (Bernoulli) and applies each scenario's churn/profit multipliers across many simulated portfolios, rather than reporting a single point estimate.

| Scenario | Mean total portfolio value | Mean incremental action profit | P(negative incremental profit) | P(budget overrun) |
|---|---:|---:|---:|---:|
| Base | EUR 23,168,302 | EUR 1,390,200 | 0.0% | 0.0% |
| Upside | EUR 25,557,078 | EUR 1,627,760 | 0.0% | 0.0% |
| Downside | EUR 20,981,241 | EUR 1,160,776 | 0.0% | 0.0% |
| Stress | EUR 18,239,089 | EUR 859,958 | 0.0% | 0.0% |

Even under Stress, incremental action profit and total portfolio value stay positive in every simulated draw -- the optimizer's selection is not fragile to the specific point estimates it was solved against, though this is a property of the *simulated* scenario multipliers, not a guarantee about real-world stress events (see Limitations).

## 8. Limitations

- **Synthetic data.** Every table in this project is synthetically generated (seed 42, `data/generator.py`) for a fictional neobank. No real customer, transaction, or credit data is used anywhere. Correlational and behavioral patterns are deliberately engineered to be realistic, not fit to any real institution's book.
- **Modeling assumptions.** CLV assumes a constant monthly churn hazard and a flat trailing-3-month profit run-rate over the forecast horizon (documented in section 4) -- a real customer's hazard and profit both evolve, and a production system would need genuine time-varying survival modeling. FTP uses a single portfolio-wide base rate rather than a full yield-curve-based transfer-pricing framework (see below). Action acceptance probabilities are calibrated assumptions from the simulation config, not measured from a real A/B test.
- **Causal limitations.** Action incremental value and the optimizer's ranking are *simulated* treatment effects under `actions/simulator.py`'s assumed acceptance/impact model, not a causally identified effect from a randomized experiment or quasi-experimental design. Real deployment would require an actual uplift model estimated from experimental or quasi-experimental data before these numbers could be treated as causal.
- **Potential model risk.** Every Phase 6 forecasting target is benchmarked against a simple linear/logistic-regression baseline (`models/validation.py`, `reports/model_validation_report.md`); the baseline is in fact the champion on 3 of 8 targets, meaning added model complexity does not uniformly buy predictive power on this dataset -- a reminder that model selection should be evaluated per-target, not assumed in favor of the most sophisticated available model.
- **Simplified FTP assumptions.** Funds Transfer Pricing uses one base annual FTP rate and one market rate from `config/settings.yaml` (`profitability/ftp.py`), not a full term-structure/duration-matched transfer-pricing curve as used by real treasury functions -- deposit contribution direction and rough magnitude are realistic, but the exact rate would differ under a production FTP framework.
- **Simplified customer behavior.** The synthetic generator's churn, spending, and product-adoption mechanisms are deliberately interpretable multi-factor models (documented in `data/README.md`), not a full agent-based or microsimulation model of customer decision-making -- real customer behavior has richer feedback loops (e.g. peer effects, macroeconomic shocks) not represented here.

See `reports/explainability_governance.md` for the companion Responsible Decision-Making documentation (PROJECT_SPEC.md section 26).

