"""Shared test fixtures/helpers."""

from __future__ import annotations

from customer_profitability.utils.config import (
    ActionsConfig,
    ActionSimulationConfig,
    ClvConfig,
    DataConfig,
    FtpConfig,
    ModelsConfig,
    OperatingConfig,
    OptimizationConfig,
    RevenueConfig,
    RiskConfig,
    Settings,
    SimulationConfig,
)


def small_settings(seed: int = 42, n_customers: int = 500, n_months: int = 12) -> Settings:
    """A full Settings object at a scale small enough to run in well under a
    second, so tests exercise the exact same code paths as production
    config without needing the full 20k x 36 dataset."""
    return Settings(
        seed=seed,
        data=DataConfig(
            n_customers=n_customers,
            n_months=n_months,
            window_start_date="2023-01-01",
            raw_dir="data/raw",
            processed_dir="data/processed",
            features_dir="data/features",
            missing_rate=0.01,
            duplicate_rate=0.003,
        ),
        models=ModelsConfig(test_months=3, validation_months=2),
        clv=ClvConfig(horizon_months=12, annual_discount_rate=0.08),
        simulation=SimulationConfig(n_simulations=100),
        optimization=OptimizationConfig(
            default_budget=10000,
            default_capacity=500,
            default_max_incremental_risk_monthly=2000,
            default_min_expected_roi=0.0,
        ),
        ftp=FtpConfig(base_ftp_rate_annual=0.045, market_rate_annual=0.035),
        risk=RiskConfig(cost_per_support_contact=8.0, cac_by_channel={"organic": 15.0, "paid_search": 65.0}),
        revenue=RevenueConfig(
            interchange_rate=0.007, premium_monthly_fee=12.0, low_balance_fee=4.0, low_balance_threshold=300.0
        ),
        operating=OperatingConfig(account_servicing_cost_monthly=0.8),
        actions=ActionsConfig(
            retention_incentive_cost=20.0,
            savings_cross_sell_cost=5.0,
            credit_product_cost=15.0,
            investment_product_cost=10.0,
            premium_subscription_cost=8.0,
            credit_product_min_income=12000.0,
        ),
        action_simulation=ActionSimulationConfig(
            params={
                "retention_incentive": {
                    "acceptance_base": 0.25,
                    "acceptance_churn_sensitivity": 5.0,
                    "churn_reduction_pct": 0.35,
                },
                "savings_cross_sell": {
                    "acceptance_base": 0.40,
                    "balance_uplift_pct": 0.15,
                    "assumed_savings_rate_annual": 0.02,
                    "churn_reduction_pct": 0.10,
                },
                "credit_product": {
                    "acceptance_base": 0.30,
                    "credit_limit_income_multiple": 0.25,
                    "expected_utilization": 0.50,
                    "interest_rate_annual": 0.15,
                    "expected_pd_monthly": 0.01,
                    "expected_lgd": 0.45,
                    "churn_reduction_pct": 0.10,
                },
                "investment_product": {
                    "acceptance_base": 0.20,
                    "monthly_fee_revenue": 6.0,
                    "churn_reduction_pct": 0.10,
                },
                "premium_subscription": {
                    "acceptance_base": 0.25,
                    "churn_reduction_pct": 0.15,
                },
            }
        ),
    )
