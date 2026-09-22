"""Phase 2 -- Data Quality Engine.

Runs structural, missing-data, duplicate, financial-validity, temporal, and
referential-integrity checks on every raw table before anything downstream
(Customer 360, profitability, econometrics, ML) is allowed to consume it.

Policy (documented here because it drives every downstream phase):

    A row is QUARANTINED -- kept, never dropped -- if it fails a *critical*
    check: a missing primary-key component, a critical-severity missing
    value, an impossible financial value, a temporal impossibility (activity
    before acquisition/origination or after churn), a referential integrity
    break, or if it is a duplicate of an earlier row (by primary key).
    Everything else is VALIDATED, including rows with *suspicious* or
    *expected* missingness -- those are counted and reported, not removed.

Quarantined and validated rows are still written to disk (RAW is what the
generator wrote; VALIDATED / QUARANTINED are this module's output), so
nothing is silently discarded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from customer_profitability.data import schemas
from customer_profitability.utils.config import Settings, load_settings
from customer_profitability.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnMissingness:
    column: str
    missing_count: int
    missing_rate: float
    severity: str


@dataclass
class TableQualityReport:
    table: str
    total_records: int
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    missing_primary_key_count: int = 0
    duplicate_count: int = 0
    missingness: list[ColumnMissingness] = field(default_factory=list)
    invalid_value_counts: dict[str, int] = field(default_factory=dict)
    temporal_violations: dict[str, int] = field(default_factory=dict)
    referential_violations: dict[str, int] = field(default_factory=dict)
    quarantined_count: int = 0
    validated_count: int = 0
    quality_score: float = 100.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataQualityReport:
    tables: dict[str, TableQualityReport]
    overall_quality_score: float

    def as_dict(self) -> dict:
        return {
            "overall_quality_score": self.overall_quality_score,
            "tables": {name: r.as_dict() for name, r in self.tables.items()},
        }

    def to_markdown(self) -> str:
        lines = [
            "# Data Quality Report",
            "",
            f"**Overall quality score: {self.overall_quality_score:.1f} / 100**",
            "",
            "| Table | Records | Duplicates | Quarantined | Validated | Score |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, r in self.tables.items():
            lines.append(
                f"| {name} | {r.total_records} | {r.duplicate_count} | "
                f"{r.quarantined_count} | {r.validated_count} | {r.quality_score:.1f} |"
            )
        for name, r in self.tables.items():
            lines += [
                "",
                f"## {name}",
            ]
            if r.missing_columns or r.extra_columns:
                lines.append(
                    f"- Structural: missing columns {r.missing_columns or 'none'}, "
                    f"extra columns {r.extra_columns or 'none'}"
                )
            if r.missing_primary_key_count:
                lines.append(f"- Missing primary-key values: {r.missing_primary_key_count}")
            if r.missingness:
                lines.append("- Missingness:")
                for m in r.missingness:
                    lines.append(
                        f"  - `{m.column}`: {m.missing_count} "
                        f"({m.missing_rate:.2%}) -- {m.severity}"
                    )
            if r.invalid_value_counts:
                lines.append("- Invalid values:")
                for k, v in r.invalid_value_counts.items():
                    if v:
                        lines.append(f"  - {k}: {v}")
            if r.temporal_violations:
                lines.append("- Temporal violations:")
                for k, v in r.temporal_violations.items():
                    if v:
                        lines.append(f"  - {k}: {v}")
            if r.referential_violations:
                lines.append("- Referential violations:")
                for k, v in r.referential_violations.items():
                    if v:
                        lines.append(f"  - {k}: {v}")
        return "\n".join(lines) + "\n"


def _check_structural(df: pd.DataFrame, table: str) -> tuple[list[str], list[str]]:
    expected = set(schemas.TABLE_SCHEMAS[table].keys())
    actual = set(df.columns)
    return sorted(expected - actual), sorted(actual - expected)


def _check_missing_primary_key(df: pd.DataFrame, table: str) -> pd.Series:
    pk = schemas.PRIMARY_KEYS[table]
    missing = pd.Series(False, index=df.index)
    for col in pk:
        if col in df.columns:
            missing = missing | df[col].isna()
    return missing


def _check_duplicates(df: pd.DataFrame, table: str) -> pd.Series:
    pk = [c for c in schemas.PRIMARY_KEYS[table] if c in df.columns]
    return df.duplicated(subset=pk, keep="first")


def _check_missingness(df: pd.DataFrame, table: str) -> tuple[list[ColumnMissingness], pd.Series]:
    severity_map = schemas.MISSING_SEVERITY.get(table, {})
    reports = []
    critical_mask = pd.Series(False, index=df.index)
    n = len(df)
    for col in df.columns:
        missing = df[col].isna()
        count = int(missing.sum())
        if count == 0:
            continue
        severity = severity_map.get(col, "suspicious")
        reports.append(
            ColumnMissingness(
                column=col,
                missing_count=count,
                missing_rate=count / n if n else 0.0,
                severity=severity,
            )
        )
        if severity == "critical":
            critical_mask = critical_mask | missing
    return reports, critical_mask


def _check_financial_validity(df: pd.DataFrame, table: str) -> tuple[dict[str, int], pd.Series]:
    invalid_mask = pd.Series(False, index=df.index)
    counts: dict[str, int] = {}

    def flag(name: str, condition: pd.Series) -> None:
        nonlocal invalid_mask
        condition = condition.fillna(False)
        counts[name] = int(condition.sum())
        invalid_mask.loc[condition] = True

    if table == "customers":
        flag("age_out_of_range", ~df["age"].between(18, 100))
        flag("income_negative", df["income"] < 0)
        flag("churn_before_acquisition", df["churn_date"].notna() & (df["churn_date"] < df["acquisition_date"]))
    elif table == "deposits":
        for col in ["opening_balance", "closing_balance", "average_balance", "inflows", "outflows"]:
            flag(f"{col}_negative", df[col] < 0)
        flag("deposit_rate_negative", df["deposit_rate"] < 0)
    elif table == "cards":
        for col in ["transaction_count", "transaction_volume", "atm_withdrawal_volume", "payment_processing_cost"]:
            flag(f"{col}_negative", df[col] < 0)
    elif table == "loans":
        for col in ["outstanding_balance", "credit_limit", "interest_income", "ead"]:
            flag(f"{col}_negative", df[col] < 0)
        flag("interest_rate_negative", df["interest_rate"] < 0)
        flag("pd_out_of_range", ~df["pd"].between(0, 1))
        flag("lgd_out_of_range", ~df["lgd"].between(0, 1))
        flag("utilization_out_of_range", ~df["utilization"].between(0, 1))
    elif table == "customer_service":
        flag("support_contacts_negative", df["support_contacts"] < 0)
        flag("handling_time_negative", df["average_handling_time"] < 0)
        flag("service_cost_negative", df["estimated_service_cost"] < 0)
    elif table == "customer_monthly":
        flag("tenure_negative", df["tenure_months"] < 0)
        flag("product_count_negative", df["product_count"] < 0)
        flag("satisfaction_out_of_range", ~df["satisfaction_proxy"].between(0, 1, inclusive="both") & df["satisfaction_proxy"].notna())

    return counts, invalid_mask


def _month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def _check_temporal(df: pd.DataFrame, table: str, customers: pd.DataFrame) -> tuple[dict[str, int], pd.Series]:
    violations: dict[str, int] = {}
    mask = pd.Series(False, index=df.index)
    if table not in {"deposits", "cards", "loans", "customer_service", "customer_monthly"}:
        return violations, mask

    cust = customers[["customer_id", "acquisition_date", "churn_date"]].drop_duplicates("customer_id")
    merged = df.merge(cust, on="customer_id", how="left")
    merged.index = df.index

    before_acq = merged["month"] < _month_start(merged["acquisition_date"])
    after_churn = merged["churn_date"].notna() & (merged["month"] > _month_start(merged["churn_date"]))
    violations["before_acquisition"] = int(before_acq.fillna(False).sum())
    violations["after_churn"] = int(after_churn.fillna(False).sum())
    mask = mask | before_acq.fillna(False) | after_churn.fillna(False)

    if table == "loans":
        before_origination = df["month"] < _month_start(df["origination_date"])
        violations["before_origination"] = int(before_origination.fillna(False).sum())
        mask = mask | before_origination.fillna(False)

    if table == "customer_monthly":
        negative_tenure = df["tenure_months"] < 0
        violations["negative_tenure"] = int(negative_tenure.sum())
        mask = mask | negative_tenure

    return violations, mask


def _check_referential(df: pd.DataFrame, table: str, valid_customer_ids: set[str]) -> tuple[dict[str, int], pd.Series]:
    if table == "customers" or "customer_id" not in df.columns:
        return {}, pd.Series(False, index=df.index)
    orphan = ~df["customer_id"].isin(valid_customer_ids)
    return {"customer_id_not_in_customers": int(orphan.sum())}, orphan


def run_quality_checks(tables: dict[str, pd.DataFrame]) -> tuple[DataQualityReport, dict[str, pd.Series]]:
    """Validate every table. Returns the report plus a per-table boolean mask
    (True = quarantine) so callers can split RAW into VALIDATED/QUARANTINED
    without recomputing the checks."""
    customers = tables["customers"]
    valid_customer_ids = set(customers.loc[~_check_missing_primary_key(customers, "customers"), "customer_id"])

    reports: dict[str, TableQualityReport] = {}
    quarantine_masks: dict[str, pd.Series] = {}

    for table, df in tables.items():
        n = len(df)
        missing_cols, extra_cols = _check_structural(df, table)
        missing_pk_mask = _check_missing_primary_key(df, table)
        duplicate_mask = _check_duplicates(df, table)
        missingness, critical_missing_mask = _check_missingness(df, table)
        invalid_counts, invalid_mask = _check_financial_validity(df, table)
        temporal_counts, temporal_mask = _check_temporal(df, table, customers)
        referential_counts, referential_mask = _check_referential(df, table, valid_customer_ids)

        quarantine_mask = (
            missing_pk_mask
            | duplicate_mask
            | critical_missing_mask
            | invalid_mask
            | temporal_mask
            | referential_mask
        )
        quarantine_masks[table] = quarantine_mask

        suspicious_rate = np.mean(
            [m.missing_rate for m in missingness if m.severity == "suspicious"]
        ) if any(m.severity == "suspicious" for m in missingness) else 0.0
        quality_score = 100.0
        if n:
            quality_score -= 100.0 * (quarantine_mask.sum() / n)
            quality_score -= 10.0 * suspicious_rate
        quality_score = float(np.clip(quality_score, 0, 100))

        reports[table] = TableQualityReport(
            table=table,
            total_records=n,
            missing_columns=missing_cols,
            extra_columns=extra_cols,
            missing_primary_key_count=int(missing_pk_mask.sum()),
            duplicate_count=int(duplicate_mask.sum()),
            missingness=missingness,
            invalid_value_counts=invalid_counts,
            temporal_violations=temporal_counts,
            referential_violations=referential_counts,
            quarantined_count=int(quarantine_mask.sum()),
            validated_count=int(n - quarantine_mask.sum()),
            quality_score=quality_score,
        )
        logger.info("%s: %d records, quality score %.1f", table, n, quality_score)

    total_records = sum(r.total_records for r in reports.values())
    overall = (
        sum(r.quality_score * r.total_records for r in reports.values()) / total_records
        if total_records
        else 100.0
    )
    return DataQualityReport(tables=reports, overall_quality_score=overall), quarantine_masks


def split_validated_quarantined(
    tables: dict[str, pd.DataFrame], quarantine_masks: dict[str, pd.Series]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    validated = {name: df.loc[~quarantine_masks[name]].reset_index(drop=True) for name, df in tables.items()}
    quarantined = {name: df.loc[quarantine_masks[name]].reset_index(drop=True) for name, df in tables.items()}
    return validated, quarantined


def _read_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(raw_dir / f"{name}.parquet") for name in schemas.TABLE_SCHEMAS}


def main() -> None:
    settings: Settings = load_settings()
    tables = _read_raw(settings.raw_dir)

    report, quarantine_masks = run_quality_checks(tables)
    validated, quarantined = split_validated_quarantined(tables, quarantine_masks)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    for name, df in validated.items():
        df.to_parquet(settings.processed_dir / f"{name}_validated.parquet", index=False)
    for name, df in quarantined.items():
        df.to_parquet(settings.processed_dir / f"{name}_quarantined.parquet", index=False)

    reports_dir = settings.raw_dir.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "data_quality_report.json").write_text(json.dumps(report.as_dict(), indent=2, default=str))
    (reports_dir / "data_quality_report.md").write_text(report.to_markdown())
    logger.info("Overall data quality score: %.1f / 100", report.overall_quality_score)
    logger.info("Wrote reports/data_quality_report.{json,md} and data/processed/*_{validated,quarantined}.parquet")


if __name__ == "__main__":
    main()
