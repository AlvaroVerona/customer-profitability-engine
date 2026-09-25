"""Phase 12 -- Monte Carlo scenario definitions.

Four scenarios per PROJECT_SPEC.md 12: Base, Downside, Upside, Stress. Two
composite multipliers per scenario rather than four separate ones, because
`expected_profit` (Phase 7's trailing run-rate) is already the *net*
monthly profit (revenue + deposit contribution - expected loss - operating
cost) -- there is no separately-tracked balance/activity/credit-loss
component left at this stage of the pipeline to shock independently
without re-deriving each from the underlying drivers, which this
project's aggregation does not support. `profit_multiplier` is a single,
documented stand-in for the combined narrative effect of PROJECT_SPEC.md
12's "lower transaction activity, lower balances, higher credit losses"
(downside/stress) or the mirror-image upside effects; `churn_multiplier`
is applied directly since churn probability is tracked explicitly and
individually per customer. These multipliers are documented assumptions --
there is no historical stress-test or economic-cycle data anywhere in
this project's synthetic dataset to calibrate them against.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    churn_multiplier: float
    profit_multiplier: float


SCENARIOS: dict[str, Scenario] = {
    "base": Scenario("Base", churn_multiplier=1.0, profit_multiplier=1.0),
    "downside": Scenario("Downside", churn_multiplier=1.3, profit_multiplier=0.85),
    "upside": Scenario("Upside", churn_multiplier=0.7, profit_multiplier=1.15),
    "stress": Scenario("Stress", churn_multiplier=1.8, profit_multiplier=0.65),
}
