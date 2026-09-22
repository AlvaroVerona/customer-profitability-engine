"""Synthetic neobank data generator (Phase 1).

Design notes (also see data/README.md for the human-readable version):

- All customer behaviour (balances, transactions, support, satisfaction,
  churn) is produced by a single month-by-month state simulation, vectorized
  across all customers with numpy. This is what creates realistic temporal
  structure and cross-variable correlation (e.g. a customer who is unhappy
  this month is also more likely to churn next month) instead of drawing
  every column independently.
- Product adoption dates are drawn independently of the churn simulation.
  When the monthly panels are assembled, a product-month row is only kept
  if the month falls between the product's adoption date and the customer's
  churn date (or window end) -- so adoption/churn consistency is enforced
  by filtering rather than by a circular simulation.
- The churn-generating mechanism (Phase 1.7) is a logistic hazard evaluated
  every month for every still-active customer. It increases with: a wide
  gap between the market deposit rate and the customer's own deposit rate
  (scaled by the customer's latent rate sensitivity), low satisfaction, low
  engagement, a negative recent balance trend, and low monthly economic
  profit proxy; it decreases with tenure (loyalty effect). Coefficients are
  defined in `_CHURN_COEF` below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from customer_profitability.data import schemas
from customer_profitability.utils.config import Settings, load_settings
from customer_profitability.utils.logging import get_logger

logger = get_logger(__name__)

SEED = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _rank01(x: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1], used to make effects scale-free."""
    return pd.Series(x).rank(pct=True).to_numpy()


# ---------------------------------------------------------------------------
# 1.1 Customers
# ---------------------------------------------------------------------------


def generate_customers(n: int, seed: int, window_start: pd.Timestamp, n_months: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = np.array([f"CUST_{i:06d}" for i in range(n)])
    age = np.clip(rng.normal(42, 13, n), 18, 85).round().astype(int)

    employment_status = np.empty(n, dtype=object)
    buckets = {
        "student": (age <= 24, {"student": 0.55, "employed": 0.30, "unemployed": 0.10, "self_employed": 0.05, "retired": 0.0}),
        "young": ((age > 24) & (age <= 39), {"employed": 0.68, "self_employed": 0.15, "unemployed": 0.12, "student": 0.03, "retired": 0.02}),
        "mid": ((age > 39) & (age <= 64), {"employed": 0.62, "self_employed": 0.20, "unemployed": 0.08, "student": 0.0, "retired": 0.10}),
        "senior": (age > 64, {"retired": 0.75, "employed": 0.10, "self_employed": 0.10, "unemployed": 0.03, "student": 0.02}),
    }
    for mask, probs in buckets.values():
        cats = list(probs.keys())
        p = np.array(list(probs.values()))
        p = p / p.sum()
        employment_status[mask] = rng.choice(cats, size=mask.sum(), p=p)

    # Income: age-shaped earnings curve x employment multiplier x lognormal noise.
    age_curve = 18000 + 48000 * np.exp(-((age - 47) ** 2) / (2 * 20**2))
    emp_multiplier = np.select(
        [
            employment_status == "employed",
            employment_status == "self_employed",
            employment_status == "unemployed",
            employment_status == "student",
            employment_status == "retired",
        ],
        [1.0, 1.05, 0.35, 0.25, 0.55],
    )
    income = age_curve * emp_multiplier * rng.lognormal(mean=0.0, sigma=0.35, size=n)
    income = np.clip(income, 6000, 400000).round(2)
    income_rank = _rank01(income)

    country = rng.choice(schemas.COUNTRIES, size=n, p=[0.45, 0.12, 0.15, 0.12, 0.10, 0.06])
    acquisition_channel = rng.choice(
        schemas.ACQUISITION_CHANNELS, size=n, p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )

    customer_segment = np.where(
        employment_status == "student",
        "Student",
        np.where(income_rank >= 0.90, "Premium", np.where(income_rank >= 0.65, "Affluent", "Mass")),
    )

    rate_sensitivity = np.clip(
        rng.beta(2, 2, n) + 0.15 * (income_rank - 0.5), 0, 1
    )
    risk_score = np.clip(
        rng.beta(2, 5, n)
        + 0.25 * (employment_status == "unemployed")
        - 0.15 * (income_rank - 0.5) * 2,
        0.01,
        0.99,
    )
    satisfaction_baseline = np.clip(
        rng.beta(5, 2, n) + 0.05 * (customer_segment == "Premium"), 0, 1
    )

    # Acquisition timing: 45% already on-book before the observation window
    # (tenure 1-60 months), 55% acquired during the window (growth cohort).
    pre_window = rng.random(n) < 0.45
    offset = np.empty(n, dtype=int)
    offset[pre_window] = -rng.integers(1, 61, size=pre_window.sum())
    n_in_window = (~pre_window).sum()
    growth_weights = np.linspace(0.6, 1.4, n_months)
    growth_weights = growth_weights / growth_weights.sum()
    offset[~pre_window] = rng.choice(n_months, size=n_in_window, p=growth_weights)
    acquisition_date = window_start + pd.to_timedelta(offset * 30, unit="D")
    acquisition_date = pd.to_datetime(acquisition_date).to_period("M").to_timestamp()

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "income": income,
            "country": country,
            "acquisition_channel": acquisition_channel,
            "acquisition_date": acquisition_date,
            "customer_segment": customer_segment,
            "employment_status": employment_status,
            "rate_sensitivity": rate_sensitivity,
            "satisfaction_baseline": satisfaction_baseline,
            "churn_flag": False,
            "churn_date": pd.NaT,
        }
    )
    # Internal-only helper columns consumed by later generation steps; not part
    # of the published schema (dropped by generate_all before writing to disk).
    df["_acquisition_offset"] = offset
    df["_risk_score"] = risk_score
    return df


# ---------------------------------------------------------------------------
# 1.2 Product adoption
# ---------------------------------------------------------------------------


def generate_product_adoption(customers: pd.DataFrame, seed: int, n_months: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    n = len(customers)
    income_rank = _rank01(customers["income"].to_numpy())
    age = customers["age"].to_numpy()
    segment = customers["customer_segment"].to_numpy()
    employment = customers["employment_status"].to_numpy()
    rate_sensitivity = customers["rate_sensitivity"].to_numpy()

    def adoption_month(prob_immediate: float) -> np.ndarray:
        """Months after acquisition; skewed early via a capped geometric draw."""
        immediate = rng.random(n) < prob_immediate
        later = np.clip(rng.geometric(p=0.08, size=n), 1, 48)
        return np.where(immediate, 0, later)

    rows = []

    # Current account: universal, opens on acquisition date.
    rows.append(
        _adoption_rows(customers, "current_account", np.zeros(n, dtype=int), np.ones(n, dtype=bool))
    )

    # Debit card: near-universal, opens within 0-2 months.
    debit_adopt = rng.random(n) < np.where(segment == "Premium", 0.99, 0.95)
    debit_month = rng.integers(0, 3, size=n)
    rows.append(_adoption_rows(customers, "debit_card", debit_month, debit_adopt))

    # Savings account.
    savings_logit = -0.6 + 1.6 * income_rank + 0.5 * (segment == "Premium") + 0.3 * (segment == "Affluent") + 0.4 * rate_sensitivity + 0.01 * (age - 42)
    savings_adopt = rng.random(n) < _sigmoid(savings_logit)
    rows.append(_adoption_rows(customers, "savings_account", adoption_month(0.25), savings_adopt))

    # Credit card.
    credit_logit = -1.0 + 1.4 * income_rank - 0.8 * (employment == "unemployed") - 0.6 * (employment == "student") + 0.3 * ((age >= 22) & (age <= 70))
    credit_adopt = rng.random(n) < _sigmoid(credit_logit)
    rows.append(_adoption_rows(customers, "credit_card", adoption_month(0.15), credit_adopt))

    # Consumer loan (single loan per adopter).
    loan_logit = -1.3 + 0.9 * income_rank - 1.0 * customers["_risk_score"].to_numpy() + 0.4 * ((age >= 25) & (age <= 60))
    loan_adopt = rng.random(n) < _sigmoid(loan_logit)
    rows.append(_adoption_rows(customers, "consumer_loan", adoption_month(0.05), loan_adopt))

    # Investment account.
    invest_logit = -1.8 + 2.0 * income_rank + 0.6 * (segment == "Premium") + 0.3 * (segment == "Affluent") + 0.015 * (age - 30)
    invest_adopt = rng.random(n) < _sigmoid(invest_logit)
    rows.append(_adoption_rows(customers, "investment_account", adoption_month(0.05), invest_adopt))

    accounts = pd.concat(rows, ignore_index=True)
    return accounts[accounts["is_active"]].drop(columns=["is_active"]).reset_index(drop=True)


def _adoption_rows(
    customers: pd.DataFrame, product: str, month_offset: np.ndarray, adopted: np.ndarray
) -> pd.DataFrame:
    open_date = customers["acquisition_date"] + pd.to_timedelta(
        np.clip(month_offset, 0, None) * 30, unit="D"
    )
    open_date = pd.to_datetime(open_date).dt.to_period("M").dt.to_timestamp()
    return pd.DataFrame(
        {
            "customer_id": customers["customer_id"],
            "product": product,
            "account_open_date": open_date,
            "account_close_date": pd.NaT,
            "is_active": adopted,
        }
    )


# ---------------------------------------------------------------------------
# 1.3 - 1.7 Monthly behavioural simulation (deposits, cards, service, churn)
# ---------------------------------------------------------------------------

_CHURN_COEF = {
    "base": -3.0,
    "satisfaction": 3.0,
    "engagement": 2.0,
    "rate_gap": 8.0,
    "log_tenure": 0.55,
    "profit": 1.2,
    "balance_drop": 5.0,
}


def simulate_monthly_behaviour(
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    settings: Settings,
    seed: int,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed + 2)
    n = len(customers)
    n_months = settings.data.n_months
    window_start = pd.Timestamp(settings.data.window_start_date)
    months = pd.date_range(window_start, periods=n_months, freq="MS")

    offset = customers["_acquisition_offset"].to_numpy()
    income = customers["income"].to_numpy()
    income_rank = _rank01(income)
    age = customers["age"].to_numpy()
    segment = customers["customer_segment"].to_numpy()
    rate_sensitivity = customers["rate_sensitivity"].to_numpy()
    satisfaction_baseline = customers["satisfaction_baseline"].to_numpy()

    has_card = customers["customer_id"].isin(
        accounts.loc[accounts["product"].isin(["debit_card", "credit_card"]), "customer_id"]
    ).to_numpy()
    has_loan = customers["customer_id"].isin(
        accounts.loc[accounts["product"] == "consumer_loan", "customer_id"]
    ).to_numpy()

    segment_deposit_share = np.select(
        [segment == "Premium", segment == "Affluent", segment == "Student"],
        [0.90, 0.75, 0.40],
        default=0.55,
    )
    market_rate_annual = settings.ftp.market_rate_annual
    macro = market_rate_annual + np.cumsum(rng.normal(0, 0.0008, n_months))
    macro = np.clip(macro, 0.015, 0.06)
    deposit_rate_annual = macro.mean() * segment_deposit_share * (1 + rng.normal(0, 0.03, n))
    deposit_rate_annual = np.clip(deposit_rate_annual, 0.001, 0.05)

    seasonal = 1 + 0.04 * np.sin(2 * np.pi * np.arange(n_months) / 12)

    initial_balance = np.clip(
        (income / 12) * rng.uniform(0.8, 2.5, n) * segment_deposit_share * 2, 50, None
    )

    balance = np.zeros(n)
    prev_closing = np.zeros(n)
    engagement = np.clip(rng.beta(3, 2, n) * 0.6 + 0.2, 0, 1)
    satisfaction = satisfaction_baseline.copy()
    churned = np.zeros(n, dtype=bool)
    churn_month_idx = np.full(n, -1, dtype=int)
    started = np.zeros(n, dtype=bool)

    deposit_rows, card_rows, service_rows, monthly_rows = [], [], [], []

    for t in range(n_months):
        newly_started = (t == offset) | ((offset < 0) & (t == 0) & (~started))
        started = started | newly_started
        balance = np.where(newly_started, initial_balance, balance)
        prev_closing = np.where(newly_started, initial_balance, prev_closing)

        active_mask = started & (~churned)
        if not active_mask.any():
            continue

        tenure = np.clip(t - offset, 0, None)
        rate_gap = np.clip(macro[t] - deposit_rate_annual, 0, None)

        drift = (
            0.004 * income
            + 0.002 * balance * (segment_deposit_share - 0.5)
            - 0.03 * balance * rate_gap * rate_sensitivity
        )
        noise = rng.normal(0, np.abs(balance) * 0.06 + income * 0.01, n)
        new_balance = np.clip(balance * 0.965 * seasonal[t] + drift + noise, 0, None)
        opening_balance = balance
        closing_balance = new_balance
        average_balance = (opening_balance + closing_balance) / 2
        net_change = closing_balance - opening_balance
        inflows = np.clip(np.abs(net_change) / 2 + income / 12 * rng.uniform(0.3, 0.9, n), 0, None)
        outflows = np.clip(inflows - net_change, 0, None)

        engagement_target = np.clip(0.3 + 0.5 * income_rank - 0.003 * np.abs(age - 35) + 0.1 * tenure / 60, 0, 1)
        engagement = np.clip(engagement * 0.85 + 0.15 * engagement_target + rng.normal(0, 0.05, n), 0, 1)

        txn_lambda = np.clip((3 + 18 * income_rank) * (0.4 + engagement) * has_card, 0.01, None)
        transaction_count = rng.poisson(txn_lambda)
        avg_ticket = 25 + 60 * income_rank
        transaction_volume = transaction_count * avg_ticket * rng.lognormal(0, 0.25, n)
        interchange_revenue = transaction_volume * 0.007
        intl_share = np.where(np.isin(segment, ["Premium", "Affluent"]), 0.18, 0.06)
        international_transaction_volume = transaction_volume * intl_share
        atm_withdrawal_volume = transaction_volume * 0.10 * (1 - income_rank * 0.5)
        payment_processing_cost = transaction_volume * 0.0015 + transaction_count * 0.02

        support_lambda = np.clip(0.12 + 0.45 * (1 - satisfaction) + 0.15 * has_loan + 0.05 * has_card, 0.01, None)
        support_contacts = rng.poisson(support_lambda)
        average_handling_time = np.clip(rng.lognormal(np.log(8), 0.3, n), 1, None)
        estimated_service_cost = support_contacts * settings.risk.cost_per_support_contact

        rate_gap_effect = 0.06 * rate_sensitivity * (deposit_rate_annual - macro[t]) * 20
        service_effect = -0.05 * (support_contacts >= 3)
        satisfaction = np.clip(
            satisfaction * 0.92 + 0.08 * satisfaction_baseline + rate_gap_effect + service_effect + rng.normal(0, 0.03, n),
            0,
            1,
        )

        monthly_spread = (settings.ftp.base_ftp_rate_annual - deposit_rate_annual) / 12
        profit_proxy = average_balance * monthly_spread + interchange_revenue - estimated_service_cost
        profit_rank = _rank01(np.where(active_mask, profit_proxy, np.nan))
        profit_rank = np.nan_to_num(profit_rank, nan=0.5)

        with np.errstate(divide="ignore", invalid="ignore"):
            recent_balance_change = np.where(
                prev_closing > 0, (closing_balance - prev_closing) / np.where(prev_closing > 0, prev_closing, 1), 0
            )

        # Coefficients apply to *deviations* from a typical customer (satisfaction
        # ~0.65, engagement ~0.55) so that an average customer sits near the base
        # hazard, and unhappy/disengaged customers are pushed up from there.
        logit = (
            _CHURN_COEF["base"]
            + _CHURN_COEF["satisfaction"] * (0.65 - satisfaction)
            + _CHURN_COEF["engagement"] * (0.55 - engagement)
            + _CHURN_COEF["rate_gap"] * rate_sensitivity * rate_gap
            - _CHURN_COEF["log_tenure"] * np.log1p(tenure)
            - _CHURN_COEF["profit"] * (profit_rank - 0.5)
            + _CHURN_COEF["balance_drop"] * np.clip(-recent_balance_change, 0, 1)
        )
        p_churn = np.clip(_sigmoid(logit), 0, 0.25)
        churn_draw = (rng.random(n) < p_churn) & active_mask & (tenure >= 1)

        month_ts = months[t]
        idx = np.where(active_mask)[0]
        deposit_rows.append(
            pd.DataFrame(
                {
                    "customer_id": customers["customer_id"].to_numpy()[idx],
                    "month": month_ts,
                    "product": "current_account",
                    "opening_balance": opening_balance[idx],
                    "closing_balance": closing_balance[idx],
                    "average_balance": average_balance[idx],
                    "inflows": inflows[idx],
                    "outflows": outflows[idx],
                    "deposit_rate": deposit_rate_annual[idx],
                    "market_rate": macro[t],
                }
            )
        )
        card_idx = idx[has_card[idx]]
        card_rows.append(
            pd.DataFrame(
                {
                    "customer_id": customers["customer_id"].to_numpy()[card_idx],
                    "month": month_ts,
                    "transaction_count": transaction_count[card_idx],
                    "transaction_volume": transaction_volume[card_idx],
                    "interchange_revenue": interchange_revenue[card_idx],
                    "atm_withdrawal_volume": atm_withdrawal_volume[card_idx],
                    "international_transaction_volume": international_transaction_volume[card_idx],
                    "payment_processing_cost": payment_processing_cost[card_idx],
                }
            )
        )
        service_idx = idx[support_contacts[idx] > 0]
        service_rows.append(
            pd.DataFrame(
                {
                    "customer_id": customers["customer_id"].to_numpy()[service_idx],
                    "month": month_ts,
                    "support_contacts": support_contacts[service_idx],
                    "average_handling_time": average_handling_time[service_idx],
                    "support_channel": rng.choice(schemas.SUPPORT_CHANNELS, size=len(service_idx)),
                    "estimated_service_cost": estimated_service_cost[service_idx],
                }
            )
        )
        monthly_rows.append(
            pd.DataFrame(
                {
                    "customer_id": customers["customer_id"].to_numpy()[idx],
                    "month": month_ts,
                    "tenure_months": tenure[idx],
                    "transaction_count": transaction_count[idx],
                    "transaction_volume": transaction_volume[idx],
                    "total_deposit_balance": average_balance[idx],
                    "support_contacts": support_contacts[idx],
                    "login_frequency": np.clip(engagement[idx] * 30, 0, 30),
                    "satisfaction_proxy": satisfaction[idx],
                    "churned_this_month": churn_draw[idx],
                }
            )
        )

        churned = churned | churn_draw
        churn_month_idx = np.where(churn_draw, t, churn_month_idx)
        balance = np.where(churned, 0, new_balance)
        prev_closing = np.where(active_mask, closing_balance, prev_closing)

    months_arr = months.to_numpy()
    churn_date = np.where(
        churn_month_idx >= 0, months_arr[np.clip(churn_month_idx, 0, n_months - 1)], np.datetime64("NaT", "ns")
    )
    customers = customers.copy()
    customers["churn_flag"] = churn_month_idx >= 0
    customers["churn_date"] = churn_date

    return {
        "customers": customers,
        "deposits": pd.concat(deposit_rows, ignore_index=True),
        "cards": pd.concat(card_rows, ignore_index=True),
        "customer_service": pd.concat(service_rows, ignore_index=True),
        "customer_monthly_raw": pd.concat(monthly_rows, ignore_index=True),
    }


# ---------------------------------------------------------------------------
# 1.5 Loans
# ---------------------------------------------------------------------------


def generate_loans(
    customers: pd.DataFrame, accounts: pd.DataFrame, settings: Settings, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 3)
    loan_accounts = accounts[accounts["product"] == "consumer_loan"].merge(
        customers[["customer_id", "income", "_risk_score", "churn_date", "customer_segment"]],
        on="customer_id",
        how="left",
    )
    n_loans = len(loan_accounts)
    if n_loans == 0:
        return pd.DataFrame(columns=list(schemas.LOANS_SCHEMA.keys()))

    n_months = settings.data.n_months
    window_start = pd.Timestamp(settings.data.window_start_date)
    window_end = window_start + pd.DateOffset(months=n_months - 1)
    months = pd.date_range(window_start, periods=n_months, freq="MS")

    credit_limit = np.clip(loan_accounts["income"].to_numpy() * rng.uniform(0.15, 0.6, n_loans), 500, 60000)
    lgd = np.clip(rng.beta(2, 3, n_loans) * 0.6 + 0.2, 0.1, 0.9)
    base_rate = 0.06 + 0.14 * loan_accounts["_risk_score"].to_numpy()
    interest_rate = np.clip(base_rate + rng.normal(0, 0.01, n_loans), 0.03, 0.29)
    loan_id = [f"LOAN_{i:06d}" for i in range(n_loans)]

    origination = pd.to_datetime(loan_accounts["account_open_date"]).clip(lower=window_start, upper=window_end)
    churn_date = pd.to_datetime(loan_accounts["churn_date"])

    utilization = np.clip(rng.beta(2, 2, n_loans), 0.05, 0.98)
    rows = []
    for t, month_ts in enumerate(months):
        active = (origination <= month_ts) & (churn_date.isna() | (churn_date >= month_ts))
        if not active.any():
            continue
        utilization = np.clip(utilization + rng.normal(0, 0.05, n_loans), 0.02, 0.99)
        outstanding_balance = credit_limit * utilization
        ead = outstanding_balance.copy()
        pd_monthly_risk = np.clip(
            0.002 + 0.02 * loan_accounts["_risk_score"].to_numpy() + 0.01 * (utilization - 0.5), 0.0005, 0.15
        )
        interest_income = outstanding_balance * interest_rate / 12
        idx = np.where(active.to_numpy())[0]
        rows.append(
            pd.DataFrame(
                {
                    "loan_id": np.array(loan_id)[idx],
                    "customer_id": loan_accounts["customer_id"].to_numpy()[idx],
                    "month": month_ts,
                    "origination_date": origination.to_numpy()[idx],
                    "outstanding_balance": outstanding_balance[idx],
                    "credit_limit": credit_limit[idx],
                    "utilization": utilization[idx],
                    "interest_rate": interest_rate[idx],
                    "interest_income": interest_income[idx],
                    "pd": pd_monthly_risk[idx],
                    "lgd": lgd[idx],
                    "ead": ead[idx],
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=list(schemas.LOANS_SCHEMA.keys()))


# ---------------------------------------------------------------------------
# Consolidation, QA-issue injection, orchestration
# ---------------------------------------------------------------------------


def build_customer_monthly(
    customers: pd.DataFrame, monthly_raw: pd.DataFrame, loans: pd.DataFrame, accounts: pd.DataFrame
) -> pd.DataFrame:
    loan_balance = (
        loans.groupby(["customer_id", "month"])["outstanding_balance"].sum().rename("total_loan_balance")
    )
    product_count = accounts.groupby("customer_id")["product"].nunique().rename("product_count")

    df = monthly_raw.merge(loan_balance, on=["customer_id", "month"], how="left")
    df["total_loan_balance"] = df["total_loan_balance"].fillna(0.0)
    df = df.merge(product_count, on="customer_id", how="left")
    df["is_active"] = True
    df = df.rename(columns={"total_deposit_balance": "total_deposit_balance"})
    return df[
        [
            "customer_id",
            "month",
            "tenure_months",
            "is_active",
            "product_count",
            "total_deposit_balance",
            "total_loan_balance",
            "transaction_count",
            "transaction_volume",
            "support_contacts",
            "login_frequency",
            "satisfaction_proxy",
            "churned_this_month",
        ]
    ]


def _inject_quality_issues(
    dfs: dict[str, pd.DataFrame], seed: int, missing_rate: float, duplicate_rate: float
) -> dict[str, pd.DataFrame]:
    """Deliberately seed missing values and duplicate rows for Phase 2 testing."""
    rng = np.random.default_rng(seed + 99)
    out = {}
    nullable_cols = {
        "deposits": ["average_balance", "deposit_rate"],
        "cards": ["transaction_volume"],
        "customer_service": ["average_handling_time"],
        "customer_monthly": ["satisfaction_proxy"],
        "loans": ["interest_rate"],
    }
    for name, df in dfs.items():
        df = df.copy()
        cols = nullable_cols.get(name, [])
        for col in cols:
            if col in df.columns and len(df) > 0:
                mask = rng.random(len(df)) < missing_rate
                df.loc[mask, col] = np.nan
        if len(df) > 0:
            n_dupe = int(len(df) * duplicate_rate)
            if n_dupe > 0:
                dupe_rows = df.sample(n=n_dupe, random_state=seed, replace=False)
                df = pd.concat([df, dupe_rows], ignore_index=True)
        out[name] = df
    return out


def generate_all(settings: Settings | None = None) -> dict[str, pd.DataFrame]:
    settings = settings or load_settings()
    seed = settings.seed
    window_start = pd.Timestamp(settings.data.window_start_date)

    logger.info("Generating %d customers", settings.data.n_customers)
    customers = generate_customers(settings.data.n_customers, seed, window_start, settings.data.n_months)

    logger.info("Generating product adoption")
    accounts = generate_product_adoption(customers, seed, settings.data.n_months)

    logger.info("Simulating monthly behaviour and churn over %d months", settings.data.n_months)
    sim = simulate_monthly_behaviour(customers, accounts, settings, seed)
    customers = sim["customers"]

    logger.info("Generating loans")
    loans = generate_loans(customers, accounts, settings, seed)

    logger.info("Assembling customer_monthly")
    customer_monthly = build_customer_monthly(customers, sim["customer_monthly_raw"], loans, accounts)

    # Close accounts of churned customers.
    accounts = accounts.merge(customers[["customer_id", "churn_date"]], on="customer_id", how="left")
    accounts["account_close_date"] = accounts["churn_date"]
    accounts["is_active"] = accounts["churn_date"].isna()
    accounts = accounts.drop(columns=["churn_date"])

    customers_out = customers.drop(columns=["_acquisition_offset", "_risk_score"])

    tables = {
        "customers": customers_out,
        "accounts": accounts,
        "deposits": sim["deposits"],
        "cards": sim["cards"],
        "loans": loans,
        "customer_service": sim["customer_service"],
        "customer_monthly": customer_monthly,
    }

    logger.info("Injecting missing values / duplicates for Phase 2 QA testing")
    tables = _inject_quality_issues(
        tables, seed, settings.data.missing_rate, settings.data.duplicate_rate
    )
    return tables


def write_tables(tables: dict[str, pd.DataFrame], raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        out_path = raw_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(df))


def main() -> None:
    settings = load_settings()
    tables = generate_all(settings)
    write_tables(tables, settings.raw_dir)


if __name__ == "__main__":
    main()
