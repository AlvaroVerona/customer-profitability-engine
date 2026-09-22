"""Phase 4 -- Historical Customer Profitability Engine.

Assembles revenue, deposit funding contribution, expected credit loss,
operating cost, and acquisition cost into the economic-profit waterfall from
PROJECT_SPEC.md section 3:

    Revenue -> Gross Contribution -> Risk-Adjusted Contribution -> Economic Profit

    gross_contribution        = total_revenue + deposit_contribution - operating_cost
    risk_adjusted_contribution = gross_contribution - expected_loss
    economic_profit            = risk_adjusted_contribution - acquisition_cost

which expands to exactly the spec's flat formula:

    EconomicProfit = TotalRevenue + DepositContribution - ExpectedLoss - OperatingCost - AcquisitionCost

Reads from `data/processed/*_validated.parquet` (Phase 2 output), matching
Customer 360 (Phase 3).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from customer_profitability.profitability.funding_cost import calculate_deposit_contribution
from customer_profitability.profitability.operating_cost import calculate_operating_cost
from customer_profitability.profitability.revenue import (
    calculate_fee_revenue,
    calculate_interchange_revenue,
    calculate_interest_revenue,
    calculate_total_revenue,
)
from customer_profitability.profitability.risk_cost import calculate_expected_loss
from customer_profitability.utils.config import Settings, load_settings
from customer_profitability.utils.logging import get_logger
from customer_profitability.utils.metrics import safe_divide

logger = get_logger(__name__)


def calculate_acquisition_cost(
    customers: pd.DataFrame, panel_months: pd.DatetimeIndex, cac_by_channel: dict[str, float]
) -> pd.DataFrame:
    """CAC is a one-time cost recognized in the customer's acquisition month.

    Customers acquired before the observed window (common for the "back
    book" cohort generated in Phase 1) already incurred their CAC in the
    past -- it is not re-charged inside the window, so their
    acquisition_cost here is 0 for every observed month.
    """
    df = customers[["customer_id", "acquisition_channel", "acquisition_date"]].drop_duplicates("customer_id").copy()
    df["month"] = pd.to_datetime(df["acquisition_date"]).dt.to_period("M").dt.to_timestamp()
    df["acquisition_cost"] = df["acquisition_channel"].map(cac_by_channel).fillna(0.0)
    df = df[df["month"].isin(panel_months)]
    return df[["customer_id", "month", "acquisition_cost"]]


def build_monthly_profitability(tables: dict[str, pd.DataFrame], settings: Settings) -> pd.DataFrame:
    customer_monthly = tables["customer_monthly"]
    customers = tables["customers"].drop_duplicates("customer_id")
    panel = customer_monthly[["customer_id", "month"]].drop_duplicates()

    interest = calculate_interest_revenue(tables["loans"])
    interchange = calculate_interchange_revenue(tables["cards"], settings.revenue)
    fees = calculate_fee_revenue(tables["deposits"], customers, settings.revenue)
    revenue = calculate_total_revenue(interest, interchange, fees)

    deposit_contribution = calculate_deposit_contribution(tables["deposits"], settings.ftp)
    expected_loss = calculate_expected_loss(tables["loans"])
    operating_cost = calculate_operating_cost(
        tables["customer_service"], tables["cards"], panel, settings.operating
    )
    acquisition_cost = calculate_acquisition_cost(
        customers, panel["month"].unique(), settings.risk.cac_by_channel
    )

    df = (
        panel.merge(revenue, on=["customer_id", "month"], how="left")
        .merge(deposit_contribution, on=["customer_id", "month"], how="left")
        .merge(expected_loss, on=["customer_id", "month"], how="left")
        .merge(operating_cost, on=["customer_id", "month"], how="left")
        .merge(acquisition_cost, on=["customer_id", "month"], how="left")
    )

    fill_zero = [
        "interest_revenue",
        "interchange_revenue",
        "account_fee",
        "premium_subscription_fee",
        "fee_revenue",
        "total_revenue",
        "deposit_contribution",
        "expected_loss",
        "support_cost",
        "payment_processing_cost",
        "account_servicing_cost",
        "operating_cost",
        "acquisition_cost",
    ]
    df[fill_zero] = df[fill_zero].fillna(0.0)

    df["gross_contribution"] = df["total_revenue"] + df["deposit_contribution"] - df["operating_cost"]
    df["risk_adjusted_contribution"] = df["gross_contribution"] - df["expected_loss"]
    df["economic_profit"] = df["risk_adjusted_contribution"] - df["acquisition_cost"]
    return df


def aggregate_profit(monthly: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Roll the monthly panel up to quarterly (`freq="Q"`) or annual
    (`freq="A"`) economic profit per customer."""
    df = monthly.copy()
    df["period"] = df["month"].dt.to_period(freq)
    amount_cols = [
        "total_revenue",
        "deposit_contribution",
        "expected_loss",
        "operating_cost",
        "acquisition_cost",
        "gross_contribution",
        "risk_adjusted_contribution",
        "economic_profit",
    ]
    return df.groupby(["customer_id", "period"], as_index=False)[amount_cols].sum()


def build_customer_profitability_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    """Phase 4.7 profitability metrics, one row per customer."""
    agg = monthly.groupby("customer_id", as_index=False).agg(
        total_revenue=("total_revenue", "sum"),
        total_deposit_contribution=("deposit_contribution", "sum"),
        total_expected_loss=("expected_loss", "sum"),
        total_operating_cost=("operating_cost", "sum"),
        total_acquisition_cost=("acquisition_cost", "sum"),
        total_gross_contribution=("gross_contribution", "sum"),
        total_risk_adjusted_profit=("risk_adjusted_contribution", "sum"),
        economic_profit=("economic_profit", "sum"),
        months_observed=("month", "nunique"),
    )
    agg["total_costs"] = agg["total_expected_loss"] + agg["total_operating_cost"] + agg["total_acquisition_cost"]
    agg["profit_margin"] = safe_divide(agg["economic_profit"], agg["total_revenue"], fill=float("nan"))
    agg["revenue_per_month"] = safe_divide(agg["total_revenue"], agg["months_observed"])
    agg["cost_per_month"] = safe_divide(agg["total_costs"], agg["months_observed"])
    agg["contribution_margin"] = safe_divide(agg["total_gross_contribution"], agg["total_revenue"], fill=float("nan"))
    return agg


def _read_validated(processed_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ["customers", "deposits", "cards", "loans", "customer_service", "customer_monthly"]:
        tables[name] = pd.read_parquet(processed_dir / f"{name}_validated.parquet")
    return tables


def main() -> None:
    settings: Settings = load_settings()
    tables = _read_validated(settings.processed_dir)

    monthly = build_monthly_profitability(tables, settings)
    summary = build_customer_profitability_summary(monthly)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = settings.processed_dir / "profitability_monthly.parquet"
    summary_path = settings.processed_dir / "profitability_customer_summary.parquet"
    monthly.to_parquet(monthly_path, index=False)
    summary.to_parquet(summary_path, index=False)
    logger.info("Wrote %s (%d rows)", monthly_path, len(monthly))
    logger.info("Wrote %s (%d rows)", summary_path, len(summary))
    logger.info(
        "Portfolio: %d customers, total economic profit %.0f, %.1f%% profitable",
        len(summary),
        summary["economic_profit"].sum(),
        100 * (summary["economic_profit"] > 0).mean(),
    )


if __name__ == "__main__":
    main()
