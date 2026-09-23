"""Phase 8 -- segment profiling and business labeling.

Labels are assigned by a deterministic rule cascade over each cluster's
*standardized* centroid (the actual K-Means `cluster_centers_`, in the same
z-score units the model clustered on) -- not hand-picked. A cluster only
gets one of PROJECT_SPEC.md 8's example names (High Value, Deposit
Funders, ...) if its centroid is actually extreme on the matching
dimension(s); otherwise it falls through to a generic "Balanced Segment N"
label, per the spec's explicit "do not force these exact labels if the
data produces different clusters."

The cascade's priority order encodes which trait is considered most
economically distinctive when a cluster is extreme on more than one axis
(e.g. a cluster that is both high-risk and high-value is labeled by its
risk first, since "at risk" is the more actionable framing for that
combination than "high value" alone).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from customer_profitability.segmentation.clustering import FEATURE_COLUMNS

HIGH = 0.75  # standardized-centroid threshold for "notably above portfolio average"
LOW = -0.75


def label_segments(cluster_centers: np.ndarray, feature_columns: list[str] = FEATURE_COLUMNS) -> dict[int, str]:
    idx = {name: i for i, name in enumerate(feature_columns)}
    labels: dict[int, str] = {}
    profit_by_cluster: dict[int, float] = {}
    for cluster_id, center in enumerate(cluster_centers):
        profit = center[idx["historical_economic_profit"]]
        clv = center[idx["total_customer_economic_value"]]
        deposit = center[idx["total_deposit_contribution"]]
        loss = center[idx["total_expected_loss"]]
        opcost = center[idx["total_operating_cost"]]
        txn = center[idx["transaction_volume_avg_3m"]]
        churn = center[idx["p_churn_monthly"]]
        util = center[idx["utilization"]]
        profit_by_cluster[cluster_id] = profit

        if churn > HIGH and clv > 0.25:
            label = "At-Risk High Value"
        elif profit > HIGH and clv > HIGH:
            label = "High Value"
        elif deposit > HIGH and deposit >= max(util, txn):
            label = "Deposit Funders"
        elif max(loss, util) > HIGH:
            label = "Credit-Driven"
        elif opcost > HIGH and profit < LOW:
            label = "High Cost"
        elif txn > HIGH and profit < 0.25:
            label = "Transactional"
        elif profit < LOW and clv < LOW:
            label = "Low Profitability"
        else:
            label = f"Balanced Segment {cluster_id}"
        labels[cluster_id] = label

    return _disambiguate_duplicate_labels(labels, profit_by_cluster)


def _disambiguate_duplicate_labels(labels: dict[int, str], profit_by_cluster: dict[int, float]) -> dict[int, str]:
    """When two or more clusters independently trigger the same named
    archetype (e.g. a "regular" and an "ultra" High Value cluster), suffix
    them by rank on economic profit (I = highest) rather than leaving
    ambiguous duplicate names in the report."""
    by_label: dict[str, list[int]] = {}
    for cluster_id, label in labels.items():
        by_label.setdefault(label, []).append(cluster_id)

    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
    out = dict(labels)
    for label, cluster_ids in by_label.items():
        if len(cluster_ids) < 2:
            continue
        ranked = sorted(cluster_ids, key=lambda c: profit_by_cluster[c], reverse=True)
        for rank, cluster_id in enumerate(ranked):
            out[cluster_id] = f"{label} {roman[rank]}"
    return out


def profile_segments(
    df: pd.DataFrame, cluster_labels: np.ndarray, segment_names: dict[int, str], feature_columns: list[str] = FEATURE_COLUMNS
) -> pd.DataFrame:
    """Per-segment size and raw (unstandardized) feature means -- what an
    analyst would actually read to sanity-check or challenge the cluster
    labels, since standardized centroids alone aren't business-interpretable."""
    work = df[feature_columns].copy()
    work["segment_id"] = cluster_labels
    profile = work.groupby("segment_id")[feature_columns].mean()
    profile.insert(0, "n_customers", work.groupby("segment_id").size())
    profile.insert(1, "segment_name", profile.index.map(segment_names))
    return profile.reset_index()


def assign_segments(df: pd.DataFrame, cluster_labels: np.ndarray, segment_names: dict[int, str]) -> pd.DataFrame:
    out = df[["customer_id"]].copy()
    out["segment_id"] = cluster_labels
    out["segment_name"] = out["segment_id"].map(segment_names)
    return out
