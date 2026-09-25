"""Phase 14 -- Customer-Level Decision Explanation.

Generates the structured "Why" explanation PROJECT_SPEC.md Phase 14 shows
as a mockup, for a single customer's top-ranked recommended action:

    Customer X
    Recommended Action: Retention Incentive
    Why:
    - High expected CLV
    - Elevated churn probability
    ...
    Expected impact:
    - Cost: EUR X
    - Incremental profit: EUR Y
    - CLV impact: EUR Z

Every "Why" bullet is a *computed* condition against portfolio benchmarks
(CLV/profit/churn percentile among active customers, product ownership,
incremental value) -- not a canned line reproduced for every customer.
A customer who isn't extreme on any dimension still gets an honest,
if shorter, explanation; a customer for whom no action beats NO_ACTION
gets an explicit "no action recommended" explanation, not a fabricated
justification for one.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from customer_profitability.actions.definitions import ActionType, build_action_catalog
from customer_profitability.actions.incremental_value import recommend_action
from customer_profitability.utils.config import Settings

HIGH_PERCENTILE = 0.75  # "top quartile" threshold for CLV/profit/churn-risk bullets

_CROSS_SELL_PRODUCT_LABEL = {
    ActionType.SAVINGS_CROSS_SELL.value: "a savings account",
    ActionType.CREDIT_PRODUCT.value: "a consumer credit line",
    ActionType.INVESTMENT_PRODUCT.value: "an investment account",
    ActionType.PREMIUM_SUBSCRIPTION.value: "the Premium tier",
}


@dataclass(frozen=True)
class CustomerExplanation:
    customer_id: str
    recommended_action: str
    why: list[str]
    cost: float
    incremental_profit: float
    clv_impact: float


def _percentile_rank(series: pd.Series, value: float) -> float:
    return float((series <= value).mean())


def explain_customer_recommendation(
    customer_id: str, incremental_value: pd.DataFrame, clv: pd.DataFrame, settings: Settings
) -> CustomerExplanation:
    """`incremental_value`: Phase 10's full table (used both to rank this
    customer's options and, from every active customer's best row, as the
    percentile-benchmark population). `clv`: Phase 7's table, for the
    historical-profit/CLV/churn percentile benchmarks."""
    ranked = recommend_action(incremental_value, customer_id, settings)
    best = ranked.iloc[0]

    active_clv = clv[clv["is_active"]]
    customer_clv_row = active_clv[active_clv["customer_id"] == customer_id]

    if best["action_type"] == ActionType.NO_ACTION.value or customer_clv_row.empty:
        why = ["No eligible action provides positive incremental value over doing nothing for this customer."]
        return CustomerExplanation(
            customer_id=customer_id,
            recommended_action="No Action",
            why=why,
            cost=0.0,
            incremental_profit=0.0,
            clv_impact=0.0,
        )

    customer_clv = customer_clv_row.iloc[0]
    clv_pct = _percentile_rank(active_clv["total_customer_economic_value"], customer_clv["total_customer_economic_value"])
    profit_pct = _percentile_rank(active_clv["historical_economic_profit"], customer_clv["historical_economic_profit"])
    churn_pct = _percentile_rank(active_clv["p_churn_monthly"], customer_clv["p_churn_monthly"])

    why: list[str] = []
    if clv_pct >= HIGH_PERCENTILE:
        why.append(
            f"High expected CLV (top {100 * (1 - clv_pct):.0f}% of active customers, "
            f"EUR {customer_clv['total_customer_economic_value']:,.0f})"
        )
    if profit_pct >= HIGH_PERCENTILE:
        why.append(
            f"Strong historical profitability (top {100 * (1 - profit_pct):.0f}% of active customers, "
            f"EUR {customer_clv['historical_economic_profit']:,.0f})"
        )
    if churn_pct >= HIGH_PERCENTILE:
        why.append(
            f"Elevated churn probability (top {100 * (1 - churn_pct):.0f}% churn risk, "
            f"{customer_clv['p_churn_monthly']:.1%}/month)"
        )

    action_type = best["action_type"]
    if action_type in _CROSS_SELL_PRODUCT_LABEL:
        why.append(f"Customer does not currently hold {_CROSS_SELL_PRODUCT_LABEL[action_type]}")

    why.append(f"Positive incremental value (+EUR {best['incremental_profit']:,.2f} over doing nothing)")

    catalog = {a.action_type.value: a for a in build_action_catalog(settings.actions)}
    nominal_cost = catalog[action_type].incentive_cost

    return CustomerExplanation(
        customer_id=customer_id,
        recommended_action=best["action_name"],
        why=why,
        cost=nominal_cost,
        incremental_profit=float(best["incremental_profit"]),
        clv_impact=float(best["delta_clv"]),
    )


def format_explanation_text(explanation: CustomerExplanation) -> str:
    lines = [
        f"Customer {explanation.customer_id}",
        "",
        "Recommended Action:",
        explanation.recommended_action,
        "",
        "Why:",
        *[f"- {reason}" for reason in explanation.why],
        "",
        "Expected impact:",
        f"- Cost: EUR {explanation.cost:,.2f}",
        f"- Incremental profit: EUR {explanation.incremental_profit:,.2f}",
        f"- CLV impact: EUR {explanation.clv_impact:,.2f}",
    ]
    return "\n".join(lines)
