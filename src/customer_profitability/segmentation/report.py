"""Phase 8 orchestrator: builds segmentation features, selects k by
silhouette score, fits K-Means (cross-checked against hierarchical
clustering on a subsample), labels and profiles the resulting segments,
and writes a report.
"""

from __future__ import annotations

import json

import pandas as pd

from customer_profitability.segmentation.clustering import (
    FEATURE_COLUMNS,
    K_RANGE,
    MIN_BUSINESS_K,
    build_segmentation_features,
    evaluate_k_range,
    fit_hierarchical,
    fit_kmeans,
    standardize_features,
)
from customer_profitability.segmentation.profiling import (
    assign_segments,
    label_segments,
    profile_segments,
)
from customer_profitability.utils.config import Settings, load_settings
from customer_profitability.utils.logging import get_logger

logger = get_logger(__name__)


def run_segmentation(settings: Settings) -> dict:
    profitability_summary = pd.read_parquet(settings.processed_dir / "profitability_customer_summary.parquet")
    clv = pd.read_parquet(settings.processed_dir / "clv.parquet")
    customer_360 = pd.read_parquet(settings.features_dir / "customer_360.parquet")

    df = build_segmentation_features(profitability_summary, clv, customer_360)
    logger.info("Segmentation feature table: %d customers, %d features", len(df), len(FEATURE_COLUMNS))
    X, _scaler = standardize_features(df)

    k_eval = evaluate_k_range(X, K_RANGE)
    business_range = k_eval[k_eval["k"] >= MIN_BUSINESS_K]
    best_k = int(business_range.loc[business_range["silhouette_score"].idxmax(), "k"])
    logger.info("K selection by silhouette score (k >= %d for business interpretability):\n%s", MIN_BUSINESS_K, k_eval.to_string(index=False))
    logger.info("Selected k=%d", best_k)

    labels, kmeans_model = fit_kmeans(X, best_k)
    _hier_labels, hier_silhouette = fit_hierarchical(X, best_k)
    kmeans_silhouette = float(k_eval.loc[k_eval["k"] == best_k, "silhouette_score"].iloc[0])
    logger.info("K-Means silhouette=%.4f, hierarchical (subsample) silhouette=%.4f", kmeans_silhouette, hier_silhouette)

    segment_names = label_segments(kmeans_model.cluster_centers_)
    profile = profile_segments(df, labels, segment_names)
    assignments = assign_segments(df, labels, segment_names)

    return {
        "df": df,
        "k_eval": k_eval,
        "best_k": best_k,
        "kmeans_silhouette": kmeans_silhouette,
        "hierarchical_silhouette": hier_silhouette,
        "segment_names": segment_names,
        "profile": profile,
        "assignments": assignments,
    }


def to_markdown(result: dict) -> str:
    profile = result["profile"].copy()
    profile_display = profile.drop(columns=["segment_id"]).set_index("segment_name")
    lines = [
        "# Economic Customer Segmentation Report",
        "",
        (
            "Clustered on economic variables only (historical profit, CLV, "
            "revenue, funding contribution, expected loss, operating cost, "
            "engagement, transaction volume, product count, churn "
            "probability, balance, credit utilization) -- not demographics. "
            "Segment names come from a deterministic rule over each "
            "cluster's standardized centroid (see `segmentation.profiling`); "
            "a cluster only gets one of the spec's example names if its "
            "centroid is actually extreme on the matching dimension(s)."
        ),
        "",
        (
            f"- k selected by silhouette score: **{result['best_k']}** "
            f"(K-Means silhouette={result['kmeans_silhouette']:.4f}, "
            f"hierarchical cross-check on a 5,000-customer subsample={result['hierarchical_silhouette']:.4f})"
        ),
        "",
        "## k selection",
        "",
        (
            f"The full table below includes k=3-4, which score marginally "
            f"higher on raw silhouette but produce clusters too coarse to be "
            f"business-actionable (a lone 'High Value' cluster plus one or two "
            f"large, barely-differentiated ones -- see `clustering.MIN_BUSINESS_K`). "
            f"The selected k is the silhouette-argmax restricted to k >= {MIN_BUSINESS_K}."
        ),
        "",
        result["k_eval"].round(4).to_string(index=False),
        "",
        "## Segment profiles (raw, unstandardized feature means)",
        "",
        profile_display.round(2).to_string(),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = load_settings()
    result = run_segmentation(settings)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    result["assignments"].to_parquet(settings.processed_dir / "customer_segments.parquet", index=False)
    result["profile"].to_parquet(settings.processed_dir / "segment_profiles.parquet", index=False)
    logger.info("Wrote data/processed/customer_segments.parquet and segment_profiles.parquet")

    reports_dir = settings.raw_dir.parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "segmentation_report.md").write_text(to_markdown(result))
    summary = {
        "best_k": result["best_k"],
        "kmeans_silhouette": result["kmeans_silhouette"],
        "hierarchical_silhouette": result["hierarchical_silhouette"],
        "segment_names": result["segment_names"],
        "segment_sizes": result["profile"].set_index("segment_name")["n_customers"].to_dict(),
    }
    (reports_dir / "segmentation_report.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote reports/segmentation_report.{md,json}")


if __name__ == "__main__":
    main()
