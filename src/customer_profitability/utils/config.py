"""Configuration loading for the customer profitability engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class DataConfig:
    n_customers: int
    n_months: int
    window_start_date: str
    raw_dir: str
    processed_dir: str
    features_dir: str
    missing_rate: float
    duplicate_rate: float


@dataclass(frozen=True)
class ModelsConfig:
    test_months: int
    validation_months: int


@dataclass(frozen=True)
class ClvConfig:
    horizon_months: int
    annual_discount_rate: float


@dataclass(frozen=True)
class SimulationConfig:
    n_simulations: int


@dataclass(frozen=True)
class OptimizationConfig:
    default_budget: float
    default_capacity: int
    default_max_incremental_risk_monthly: float
    default_min_expected_roi: float


@dataclass(frozen=True)
class FtpConfig:
    base_ftp_rate_annual: float
    market_rate_annual: float


@dataclass(frozen=True)
class RiskConfig:
    cost_per_support_contact: float
    cac_by_channel: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RevenueConfig:
    interchange_rate: float
    premium_monthly_fee: float
    low_balance_fee: float
    low_balance_threshold: float


@dataclass(frozen=True)
class OperatingConfig:
    account_servicing_cost_monthly: float


@dataclass(frozen=True)
class ActionsConfig:
    retention_incentive_cost: float
    savings_cross_sell_cost: float
    credit_product_cost: float
    investment_product_cost: float
    premium_subscription_cost: float
    credit_product_min_income: float


@dataclass(frozen=True)
class ActionSimulationConfig:
    """Per-action Phase 10 simulation parameters, e.g.
    `params["retention_incentive"]["churn_reduction_pct"]`. Kept as a plain
    nested dict (like `RiskConfig.cac_by_channel`) rather than one dataclass
    per action, since each action's parameter set is genuinely different
    (a savings cross-sell has a balance uplift, a credit product has a PD/LGD
    pair, etc.) and a shared dataclass would need every field optional."""

    params: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    seed: int
    data: DataConfig
    models: ModelsConfig
    clv: ClvConfig
    simulation: SimulationConfig
    optimization: OptimizationConfig
    ftp: FtpConfig
    risk: RiskConfig
    revenue: RevenueConfig
    operating: OperatingConfig
    actions: ActionsConfig
    action_simulation: ActionSimulationConfig

    @property
    def raw_dir(self) -> Path:
        return PROJECT_ROOT / self.data.raw_dir

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / self.data.processed_dir

    @property
    def features_dir(self) -> Path:
        return PROJECT_ROOT / self.data.features_dir


def _load_raw(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_settings(path: Path | str | None = None) -> Settings:
    """Load and validate settings.yaml into a typed Settings object."""
    raw = _load_raw(Path(path) if path else DEFAULT_CONFIG_PATH)
    return Settings(
        seed=raw["seed"],
        data=DataConfig(**raw["data"]),
        models=ModelsConfig(**raw["models"]),
        clv=ClvConfig(**raw["clv"]),
        simulation=SimulationConfig(**raw["simulation"]),
        optimization=OptimizationConfig(**raw["optimization"]),
        ftp=FtpConfig(**raw["ftp"]),
        risk=RiskConfig(**raw["risk"]),
        revenue=RevenueConfig(**raw["revenue"]),
        operating=OperatingConfig(**raw["operating"]),
        actions=ActionsConfig(**raw["actions"]),
        action_simulation=ActionSimulationConfig(params=raw["action_simulation"]),
    )
