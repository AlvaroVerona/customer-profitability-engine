"""Phase 13 -- dashboard data-assembly tests.

Most of `app/pages/*.py` is Streamlit widget calls (`st.selectbox`,
`st.dataframe`, ...) that aren't meaningfully unit-testable without a
disproportionate mocking harness for this phase's scope; Phase 13 was
instead validated by running the actual app in a browser (see the
session's manual walkthrough of all 8 pages). This file covers the one
piece of `app/components/` that has real logic: `data_loader.py`'s table
assembly -- in particular the customer_segment column-collision bug caught
while building Page 1 (merging `customer_360`, which carries its own copy
of `customer_segment`, together with `customers_validated`'s copy would
have silently produced `customer_segment_x`/`_y` instead of one column).

These are integration-style tests against the real pipeline output on
disk (not hand-built fixtures, unlike every other phase's tests) --
skipped if `make generate-data` through `make montecarlo` haven't been run
yet, since `data_loader.py`'s job is specifically to read those fixed
paths, not a parameterized function.
"""

from __future__ import annotations

import pytest

from customer_profitability.utils.config import load_settings

pytest.importorskip("streamlit")

_settings = load_settings()
_PIPELINE_OUTPUTS_EXIST = (
    (_settings.processed_dir / "clv.parquet").exists()
    and (_settings.processed_dir / "customer_segments.parquet").exists()
    and (_settings.features_dir / "customer_360.parquet").exists()
)

pytestmark = pytest.mark.skipif(
    not _PIPELINE_OUTPUTS_EXIST, reason="requires the full pipeline (make generate-data ... montecarlo) to have run"
)


def test_merged_customer_overview_has_single_unambiguous_segment_column() -> None:
    from components.data_loader import merged_customer_overview

    df = merged_customer_overview()
    assert "customer_segment" in df.columns
    assert "customer_segment_x" not in df.columns
    assert "customer_segment_y" not in df.columns
    # A real category, not silently empty/NaN for everyone.
    assert df["customer_segment"].notna().any()


def test_merged_customer_overview_one_row_per_customer() -> None:
    from components.data_loader import load_profitability_summary, merged_customer_overview

    df = merged_customer_overview()
    summary = load_profitability_summary()
    assert len(df) == len(summary)
    assert df["customer_id"].is_unique


def test_merged_customer_overview_segment_name_never_set_for_churned_customers() -> None:
    """Phase 8 segments active customers only, so no churned customer
    should have a segment_name -- but a handful of active customers can
    legitimately lack one too (dropped for incomplete feature history in
    `segmentation.clustering.build_segmentation_features`'s dropna), so the
    inverse ("every active customer has a segment") is not asserted."""
    from components.data_loader import merged_customer_overview

    df = merged_customer_overview()
    assert df.loc[~df["is_active"], "segment_name"].isna().all()
    assert df.loc[df["is_active"], "segment_name"].notna().mean() > 0.99
