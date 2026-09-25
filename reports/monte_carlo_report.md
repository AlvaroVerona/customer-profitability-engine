# Monte Carlo Scenario Simulation Report

10,000 simulations per scenario. Scenario multipliers (`simulation/scenarios.py`) are documented assumptions -- there is no historical stress-test or economic-cycle data in this project's synthetic dataset to calibrate them against.

## Total portfolio value (CLV distribution) by scenario

| scenario | mean | P5 | P50 | P95 |
|---|---:|---:|---:|---:|
| Base | 23,168,302 | 23,010,753 | 23,166,356 | 23,332,826 |
| Downside | 20,981,241 | 20,849,012 | 20,979,191 | 21,121,507 |
| Upside | 25,557,078 | 25,374,012 | 25,554,663 | 25,745,307 |
| Stress | 18,239,089 | 18,138,133 | 18,237,397 | 18,344,381 |

## Incremental action profit (Phase 11 selected actions) by scenario

| scenario | mean | P5 | P50 | P95 | P(negative) | P(budget overrun) |
|---|---:|---:|---:|---:|---:|---:|
| Base | 1,390,200 | 1,309,541 | 1,389,963 | 1,471,027 | 0.00% | 0.00% |
| Downside | 1,160,776 | 1,093,749 | 1,160,266 | 1,229,529 | 0.00% | 0.00% |
| Upside | 1,627,760 | 1,532,576 | 1,627,543 | 1,723,669 | 0.00% | 0.00% |
| Stress | 859,958 | 810,057 | 859,690 | 911,079 | 0.00% | 0.00% |

Budget checked against: 100,000.00 (Phase 11's `optimization.default_budget`). "P(budget overrun)" is the probability that *realized* cost -- an all-or-nothing incentive payout per customer who actually accepts, redrawn each simulation -- exceeds the budget that was set against the *expected* (acceptance-weighted) cost.

Both probabilities are 0.00% in every scenario, including Stress. This is a genuine result of the pipeline, not a rounding artifact or a scenario that was tuned too mild to bind: Phase 11 selected only the best 5,000 of 44,406 eligible candidates (the top ~11% by incremental profit, after an ROI pre-filter), so even the *worst* selected candidate's value comfortably survives a 35% profit haircut and an 80% higher churn hazard -- and the maximum possible realized cost if every single selected customer accepted (48,596) is well under the 100,000 budget regardless of scenario. A less curated candidate pool (e.g. before Phase 10/11's filtering) would show materially higher probabilities under Stress; this result says the *optimizer's selection* is robust, not that the underlying action economics can never go negative.

