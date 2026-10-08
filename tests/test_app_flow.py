"""Drive the Streamlit app headlessly through every page (no browser needed)."""
import logging
import warnings

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from conftest import ROOT  # noqa: E402

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def at():
    return AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=300).run()


def visit(at, page, *buttons):
    at.sidebar.selectbox[0].set_value(page).run()
    for label in buttons:
        next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_full_flow(at):
    visit(at, "📁 Data Upload & Processing", "Use Sample Dataset", "Process Transaction Data")
    visit(at, "🔍 RFM Analysis", "Run RFM Analysis")
    visit(at, "⚠️ Churn Prediction", "Train Churn Models")
    visit(at, "⚠️ Churn Prediction", "Predict Churn Risk")   # model must be reloaded from disk
    visit(at, "💰 Customer Lifetime Value", "Calculate Customer Lifetime Value")
    for page in ("📊 Dashboard Overview", "👥 Customer Segmentation", "📈 Business Insights", "⚙️ Settings"):
        visit(at, page)
