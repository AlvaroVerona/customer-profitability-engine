"""Phase 9 -- Customer Action Framework.

Defines the finite set of actions the bank can take on a customer, and the
eligibility rule for each. `NO_ACTION` is not optional: PROJECT_SPEC.md
Phase 9 is explicit that "the system must compare every intervention
against doing nothing," and Phase 10/11 depend on `NO_ACTION` always being
present in the action set for every customer so the incremental-value and
optimization steps have a baseline to net against.

Costs are read from `config.actions` rather than hardcoded (Development
Principle #12). `retention_incentive_cost`/`savings_cross_sell_cost`/
`credit_product_cost` (20/5/15) match the example figures in
PROJECT_SPEC.md's Page 6 action-recommendation mockup; investment and
premium-subscription costs have no figure given in the spec, so
10.0/8.0 are this project's own documented assumption (roughly in line
with the given examples, not derived from any external source).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from customer_profitability.clv.forecasting import latest_snapshot
from customer_profitability.models.product_adoption import ownership_flags
from customer_profitability.utils.config import ActionsConfig, Settings, load_settings
from customer_profitability.utils.logging import get_logger

logger = get_logger(__name__)

# A customer who is unemployed, or whose income is below this floor, is not
# offered a new credit line -- PROJECT_SPEC.md 9's "Credit Product ... subject
# to risk constraints". There is no existing PD estimate to gate on for a
# customer who has never held a loan (the exact customers this action targets
# by construction), so eligibility uses the income/employment signals every
# customer has, mirroring a real pre-screening/underwriting check.
CREDIT_PRODUCT_EXCLUDE_UNEMPLOYED = True

PRODUCT_FOR_ACTION = {
    "SAVINGS_CROSS_SELL": "savings_account",
    "CREDIT_PRODUCT": "consumer_loan",
    "INVESTMENT_PRODUCT": "investment_account",
}


class ActionType(str, Enum):
    NO_ACTION = "NO_ACTION"
    RETENTION_INCENTIVE = "RETENTION_INCENTIVE"
    SAVINGS_CROSS_SELL = "SAVINGS_CROSS_SELL"
    CREDIT_PRODUCT = "CREDIT_PRODUCT"
    INVESTMENT_PRODUCT = "INVESTMENT_PRODUCT"
    PREMIUM_SUBSCRIPTION = "PREMIUM_SUBSCRIPTION"


@dataclass(frozen=True)
class ActionDefinition:
    action_type: ActionType
    name: str
    description: str
    incentive_cost: float
    requires_absent_product: str | None = None
    requires_not_premium: bool = False
    requires_risk_check: bool = False


def build_action_catalog(actions_config: ActionsConfig) -> list[ActionDefinition]:
    return [
        ActionDefinition(
            ActionType.NO_ACTION,
            "No Action",
            "Baseline: do nothing. Every other action is measured against this.",
            incentive_cost=0.0,
        ),
        ActionDefinition(
            ActionType.RETENTION_INCENTIVE,
            "Retention Incentive",
            "Temporary pricing or cashback incentive aimed at reducing churn risk.",
            incentive_cost=actions_config.retention_incentive_cost,
        ),
        ActionDefinition(
            ActionType.SAVINGS_CROSS_SELL,
            "Savings Cross-Sell",
            "Offer a savings account to a customer who doesn't already have one.",
            incentive_cost=actions_config.savings_cross_sell_cost,
            requires_absent_product=PRODUCT_FOR_ACTION["SAVINGS_CROSS_SELL"],
        ),
        ActionDefinition(
            ActionType.CREDIT_PRODUCT,
            "Credit Product",
            "Offer a consumer credit line, subject to a credit-risk eligibility gate.",
            incentive_cost=actions_config.credit_product_cost,
            requires_absent_product=PRODUCT_FOR_ACTION["CREDIT_PRODUCT"],
            requires_risk_check=True,
        ),
        ActionDefinition(
            ActionType.INVESTMENT_PRODUCT,
            "Investment Product",
            "Offer an investment account to a customer who doesn't already have one.",
            incentive_cost=actions_config.investment_product_cost,
            requires_absent_product=PRODUCT_FOR_ACTION["INVESTMENT_PRODUCT"],
        ),
        ActionDefinition(
            ActionType.PREMIUM_SUBSCRIPTION,
            "Premium Subscription",
            "Upgrade a non-Premium customer to the paid Premium tier.",
            incentive_cost=actions_config.premium_subscription_cost,
            requires_not_premium=True,
        ),
    ]


def _ownership_matrix(accounts: pd.DataFrame, snapshot: pd.DataFrame, products: list[str]) -> pd.DataFrame:
    """`snapshot`: one row per (still-active) customer with its most
    recently observed month. Returns customer_id + one boolean column per
    product in `products`."""
    out = snapshot[["customer_id"]].copy()
    for product in products:
        out[product] = ownership_flags(accounts, product, snapshot).to_numpy()
    return out


def evaluate_eligibility(
    customers: pd.DataFrame, accounts: pd.DataFrame, snapshot: pd.DataFrame, actions_config: ActionsConfig
) -> pd.DataFrame:
    """One row per customer in `snapshot`, one boolean column per
    `ActionType` (column names are the enum's string values)."""
    products = sorted(set(PRODUCT_FOR_ACTION.values()))
    owned = _ownership_matrix(accounts, snapshot, products)

    df = customers[["customer_id", "customer_segment", "income", "employment_status"]].merge(
        owned, on="customer_id", how="inner"
    )

    credit_risk_ok = df["income"] >= actions_config.credit_product_min_income
    if CREDIT_PRODUCT_EXCLUDE_UNEMPLOYED:
        credit_risk_ok &= df["employment_status"] != "unemployed"

    return pd.DataFrame(
        {
            "customer_id": df["customer_id"],
            ActionType.NO_ACTION.value: True,
            ActionType.RETENTION_INCENTIVE.value: True,
            ActionType.SAVINGS_CROSS_SELL.value: ~df[PRODUCT_FOR_ACTION["SAVINGS_CROSS_SELL"]],
            ActionType.CREDIT_PRODUCT.value: ~df[PRODUCT_FOR_ACTION["CREDIT_PRODUCT"]] & credit_risk_ok,
            ActionType.INVESTMENT_PRODUCT.value: ~df[PRODUCT_FOR_ACTION["INVESTMENT_PRODUCT"]],
            ActionType.PREMIUM_SUBSCRIPTION.value: df["customer_segment"] != "Premium",
        }
    )


def eligible_actions_for_customer(eligibility: pd.DataFrame, customer_id: str) -> list[ActionType]:
    row = eligibility.loc[eligibility["customer_id"] == customer_id].iloc[0]
    return [action for action in ActionType if bool(row[action.value])]


def _read_inputs(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customers = pd.read_parquet(settings.processed_dir / "customers_validated.parquet").drop_duplicates("customer_id")
    accounts = pd.read_parquet(settings.processed_dir / "accounts_validated.parquet")
    customer_360 = pd.read_parquet(settings.features_dir / "customer_360.parquet")
    return customers, accounts, customer_360


def main() -> None:
    settings = load_settings()
    customers, accounts, customer_360 = _read_inputs(settings)

    still_active_ids = set(customers.loc[~customers["churn_flag"], "customer_id"])
    snapshot = latest_snapshot(customer_360)
    snapshot = snapshot[snapshot["customer_id"].isin(still_active_ids)][["customer_id", "month"]]

    catalog = build_action_catalog(settings.actions)
    eligibility = evaluate_eligibility(customers, accounts, snapshot, settings.actions)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.processed_dir / "action_eligibility.parquet"
    eligibility.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d active customers)", out_path, len(eligibility))

    counts = {a.action_type.value: int(eligibility[a.action_type.value].sum()) for a in catalog}
    for name, count in counts.items():
        logger.info("Eligible for %s: %d / %d active customers", name, count, len(eligibility))

    reports_dir = settings.raw_dir.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Customer Action Framework",
        "",
        "| Action | Cost (EUR) | Eligible customers | Eligibility rule |",
        "|---|---:|---:|---|",
    ]
    rule_text = {
        ActionType.NO_ACTION: "always eligible (baseline)",
        ActionType.RETENTION_INCENTIVE: "always eligible (any active customer)",
        ActionType.SAVINGS_CROSS_SELL: "doesn't already hold a savings account",
        ActionType.CREDIT_PRODUCT: "no existing consumer loan; income >= "
        f"{settings.actions.credit_product_min_income:,.0f} and not unemployed",
        ActionType.INVESTMENT_PRODUCT: "doesn't already hold an investment account",
        ActionType.PREMIUM_SUBSCRIPTION: "not already in the Premium segment",
    }
    for a in catalog:
        lines.append(
            f"| {a.name} | {a.incentive_cost:.2f} | {counts[a.action_type.value]} | {rule_text[a.action_type]} |"
        )
    (reports_dir / "actions_report.md").write_text("\n".join(lines) + "\n")
    (reports_dir / "actions_report.json").write_text(json.dumps({"eligible_counts": counts, "n_active_customers": len(eligibility)}, indent=2))
    logger.info("Wrote reports/actions_report.{md,json}")


if __name__ == "__main__":
    main()
