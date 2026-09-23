"""Phase 10 -- Action Impact Simulation.

For every (customer, eligible action) pair, estimates the effect of
offering that action *conditional on acceptance*: acceptance probability,
monthly churn-hazard reduction, additional monthly revenue, additional
balance, additional monthly expected credit loss, and recurring
operational cost. `incremental_value.py` blends these by acceptance
probability into the *expected* value of offering the action -- not the
value of a guaranteed acceptance.

There is no historical offer/acceptance data anywhere in this project's
synthetic dataset (Phase 1 never generated any such events) to fit an
acceptance or response model from. Acceptance probability and every effect
size here is therefore a documented, transparent heuristic formula per
action (`config.action_simulation`), not a fitted model dressed up as one
-- consistent with Development Principle #2 ("do not fabricate results")
and PROJECT_SPEC.md 14's #14 ("do not recommend actions based solely on ML
predictions"): these are assumptions an analyst can see, challenge, and
recalibrate against real data if this were ever deployed, not a black box.

Per PROJECT_SPEC.md 10 ("do not assume every accepted action creates
identical value"), each action affects a different subset of these
columns:

    action                 balance   revenue   risk   churn reduction
    RETENTION_INCENTIVE       -         -        -          yes (largest)
    SAVINGS_CROSS_SELL       yes       yes        -          small
    CREDIT_PRODUCT           yes       yes       yes         small
    INVESTMENT_PRODUCT        -        yes        -          small
    PREMIUM_SUBSCRIPTION      -        yes        -          medium
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from customer_profitability.actions.definitions import ActionType
from customer_profitability.profitability.ftp import annual_to_monthly
from customer_profitability.utils.config import Settings

EFFECT_COLUMNS = [
    "acceptance_probability",
    "churn_reduction_pct",
    "additional_revenue_monthly",
    "additional_balance",
    "additional_risk_monthly",
    "operational_cost_monthly",
]


def _base_frame(df: pd.DataFrame, action_type: ActionType) -> pd.DataFrame:
    n = len(df)
    return pd.DataFrame(
        {
            "customer_id": df["customer_id"].to_numpy(),
            "action_type": action_type.value,
            "acceptance_probability": np.zeros(n),
            "churn_reduction_pct": np.zeros(n),
            "additional_revenue_monthly": np.zeros(n),
            "additional_balance": np.zeros(n),
            "additional_risk_monthly": np.zeros(n),
            "operational_cost_monthly": np.zeros(n),
        }
    )


def _no_action_effects(df: pd.DataFrame) -> pd.DataFrame:
    out = _base_frame(df, ActionType.NO_ACTION)
    out["acceptance_probability"] = 1.0  # deterministic baseline; every other column stays 0
    return out


def _retention_effects(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = _base_frame(df, ActionType.RETENTION_INCENTIVE)
    p_churn = df["p_churn_monthly"].to_numpy()
    out["acceptance_probability"] = np.clip(
        params["acceptance_base"] + params["acceptance_churn_sensitivity"] * p_churn, 0.0, 1.0
    )
    out["churn_reduction_pct"] = params["churn_reduction_pct"]
    return out


def _savings_cross_sell_effects(df: pd.DataFrame, params: dict, ftp_rate_annual: float, servicing_cost: float) -> pd.DataFrame:
    out = _base_frame(df, ActionType.SAVINGS_CROSS_SELL)
    balance = df["average_balance_avg_3m"].fillna(0.0).to_numpy()
    additional_balance = balance * params["balance_uplift_pct"]
    monthly_spread = annual_to_monthly(ftp_rate_annual - params["assumed_savings_rate_annual"])
    out["acceptance_probability"] = params["acceptance_base"]
    out["churn_reduction_pct"] = params["churn_reduction_pct"]
    out["additional_balance"] = additional_balance
    out["additional_revenue_monthly"] = additional_balance * monthly_spread
    out["operational_cost_monthly"] = servicing_cost
    return out


def _credit_product_effects(df: pd.DataFrame, params: dict, servicing_cost: float) -> pd.DataFrame:
    out = _base_frame(df, ActionType.CREDIT_PRODUCT)
    income = df["income"].fillna(0.0).to_numpy()
    credit_limit = income * params["credit_limit_income_multiple"]
    new_balance = credit_limit * params["expected_utilization"]
    out["acceptance_probability"] = params["acceptance_base"]
    out["churn_reduction_pct"] = params["churn_reduction_pct"]
    out["additional_balance"] = new_balance
    out["additional_revenue_monthly"] = new_balance * (params["interest_rate_annual"] / 12)
    out["additional_risk_monthly"] = params["expected_pd_monthly"] * params["expected_lgd"] * new_balance
    out["operational_cost_monthly"] = servicing_cost
    return out


def _investment_product_effects(df: pd.DataFrame, params: dict, servicing_cost: float) -> pd.DataFrame:
    out = _base_frame(df, ActionType.INVESTMENT_PRODUCT)
    out["acceptance_probability"] = params["acceptance_base"]
    out["churn_reduction_pct"] = params["churn_reduction_pct"]
    out["additional_revenue_monthly"] = params["monthly_fee_revenue"]
    out["operational_cost_monthly"] = servicing_cost
    return out


def _premium_subscription_effects(df: pd.DataFrame, params: dict, premium_monthly_fee: float, servicing_cost: float) -> pd.DataFrame:
    out = _base_frame(df, ActionType.PREMIUM_SUBSCRIPTION)
    out["acceptance_probability"] = params["acceptance_base"]
    out["churn_reduction_pct"] = params["churn_reduction_pct"]
    out["additional_revenue_monthly"] = premium_monthly_fee
    out["operational_cost_monthly"] = servicing_cost
    return out


def simulate_action_effects(customer_df: pd.DataFrame, eligibility: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """`customer_df`: customer_id, p_churn_monthly, income, average_balance_avg_3m
    (one row per still-active customer). `eligibility`: Phase 9's per-customer,
    per-action boolean table. Returns long format, one row per eligible
    (customer_id, action_type) pair, with `EFFECT_COLUMNS`.
    """
    params = settings.action_simulation.params
    servicing_cost = settings.operating.account_servicing_cost_monthly

    all_effects = {
        ActionType.NO_ACTION.value: _no_action_effects(customer_df),
        ActionType.RETENTION_INCENTIVE.value: _retention_effects(customer_df, params["retention_incentive"]),
        ActionType.SAVINGS_CROSS_SELL.value: _savings_cross_sell_effects(
            customer_df, params["savings_cross_sell"], settings.ftp.base_ftp_rate_annual, servicing_cost
        ),
        ActionType.CREDIT_PRODUCT.value: _credit_product_effects(customer_df, params["credit_product"], servicing_cost),
        ActionType.INVESTMENT_PRODUCT.value: _investment_product_effects(
            customer_df, params["investment_product"], servicing_cost
        ),
        ActionType.PREMIUM_SUBSCRIPTION.value: _premium_subscription_effects(
            customer_df, params["premium_subscription"], settings.revenue.premium_monthly_fee, servicing_cost
        ),
    }

    frames = []
    for action_type, effects in all_effects.items():
        eligible_ids = set(eligibility.loc[eligibility[action_type], "customer_id"])
        frames.append(effects[effects["customer_id"].isin(eligible_ids)])
    return pd.concat(frames, ignore_index=True)
