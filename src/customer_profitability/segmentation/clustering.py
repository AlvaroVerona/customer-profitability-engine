"""Phase 8 -- Economic customer segmentation.

Clusters on *economic* variables (profit, CLV, funding contribution, risk,
engagement, transactional behaviour, churn) rather than demographics, per
PROJECT_SPEC.md 8: "Do not cluster solely on demographics."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from customer_profitability.models.evaluation import RANDOM_STATE

# One column per bullet in PROJECT_SPEC.md 8's feature list: historical
# profit, CLV, revenue, funding contribution, expected loss, engagement,
# transaction volume, product count, churn probability, balance, and risk
# (operating cost added on top -- needed to let a genuine "High Cost"
# archetype emerge in profiling.py, which the example segment list names
# explicitly but none of the other columns capture).
FEATURE_COLUMNS = [
    "historical_economic_profit",
    "total_customer_economic_value",
    "total_revenue",
    "total_deposit_contribution",
    "total_expected_loss",
    "total_operating_cost",
    "login_frequency",
    "transaction_volume_avg_3m",
    "product_count",
    "p_churn_monthly",
    "average_balance_avg_3m",
    "utilization",
]

K_RANGE = range(3, 10)

# PROJECT_SPEC.md 8 asks for k chosen by "silhouette score AND business
# interpretability" -- on this data the raw silhouette-argmax is k=3, which
# scores marginally higher but produces one clear "High Value" cluster and
# two large, barely-differentiated ones (neither crosses the labeling
# cascade's extremity threshold, in profiling.py). k=3 is excluded from
# selection for being too coarse to be an actionable segmentation, in favor
# of the best silhouette among a business-standard 5-8 segment range.
MIN_BUSINESS_K = 5


def build_segmentation_features(
    profitability_summary: pd.DataFrame, clv: pd.DataFrame, customer_360: pd.DataFrame
) -> pd.DataFrame:
    """Segments only *still-active* customers.

    Already-churned customers have `p_churn_monthly` forced to 1.0 by
    construction in the CLV table (see `clv.report`), a placeholder
    meaning "certain, already happened" rather than a genuine calibrated
    risk estimate -- mixing them in with active customers on that
    dimension would confound "at risk of leaving" with "already left".
    Segmentation exists to drive Phase 9-11's action framework, which by
    definition can only act on customers who are still there.
    """
    clv = clv[clv["is_active"]]
    latest = (
        customer_360.sort_values(["customer_id", "month"])
        .groupby("customer_id", as_index=False)
        .tail(1)[["customer_id", "login_frequency", "transaction_volume_avg_3m", "product_count", "average_balance_avg_3m", "utilization"]]
    )
    df = (
        profitability_summary[["customer_id", "total_revenue", "total_deposit_contribution", "total_expected_loss", "total_operating_cost"]]
        .merge(clv[["customer_id", "historical_economic_profit", "total_customer_economic_value", "p_churn_monthly", "is_active"]], on="customer_id", how="left")
        .merge(latest, on="customer_id", how="left")
    )
    # utilization is NaN for customers who never held a loan -- "no credit
    # risk exposure" is a real, meaningful 0 here, not a missing measurement.
    df["utilization"] = df["utilization"].fillna(0.0)
    return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def standardize_features(df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_columns])
    return X, scaler


def evaluate_k_range(X: np.ndarray, k_range: range = K_RANGE, sample_size: int = 5000) -> pd.DataFrame:
    """Silhouette score is O(n^2) -- evaluated on a fixed random subsample
    for tractability at 20k rows, per scikit-learn's own recommendation for
    large datasets."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        score = silhouette_score(X, km.labels_, sample_size=min(sample_size, len(X)), random_state=RANDOM_STATE)
        rows.append({"k": k, "silhouette_score": score, "inertia": km.inertia_})
    return pd.DataFrame(rows)


def fit_kmeans(X: np.ndarray, k: int) -> tuple[np.ndarray, KMeans]:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
    return model.labels_, model


def fit_hierarchical(X: np.ndarray, k: int, sample_size: int = 5000) -> tuple[np.ndarray, float]:
    """Agglomerative (Ward) clustering as a structural cross-check against
    K-Means, per PROJECT_SPEC.md 8 ("hierarchical clustering where
    useful"). Scikit-learn's AgglomerativeClustering is O(n^2) memory, so
    this runs on a fixed random subsample rather than the full 20k rows;
    it exists to corroborate that K-Means' k is a reasonable structural
    choice, not to produce the final labels for every customer."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X[idx]
    model = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X_sample)
    score = silhouette_score(X_sample, model.labels_)
    return model.labels_, score
