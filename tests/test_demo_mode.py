"""Every dashboard page renders from the committed demo sample.

Runs each Streamlit page through `AppTest` (a real script run). On a fresh checkout there is no
pipeline output, so the pages read `demo_data/`; on a machine that has run the pipeline they read
the full tables instead, and the test passes either way.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGES = sorted((APP_DIR / "pages").glob("[a-z]*.py"))


@pytest.mark.parametrize("page", PAGES, ids=[p.name for p in PAGES])
def test_page_renders_without_exception(page: Path) -> None:
    at = AppTest.from_file(str(page), default_timeout=120)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]


def test_home_page_renders_without_exception() -> None:
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
