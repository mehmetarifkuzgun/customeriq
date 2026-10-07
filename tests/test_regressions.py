"""Regression tests for bugs found by actually running the app end to end."""
import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from src.churn_predictor import ChurnPredictor
from src.clv_calculator import CLVCalculator
from src.data_processor import DataProcessor
from src.rfm_analyzer import RFMAnalyzer

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def processed():
    dp = DataProcessor()
    tx = dp.clean_transaction_data(dp.create_sample_dataset(n_customers=600))
    return dp, tx, dp.create_customer_features(tx)


def test_sample_data_has_varied_recency(processed):
    _, _, customers = processed
    days = customers["days_since_last_purchase"]
    assert days.nunique() > 50          # used to be ~0 for almost everyone
    assert days.min() >= 0 and days.max() > 300


def test_rfm_runs_on_default_sample_data(processed):
    _, _, customers = processed
    rfm = RFMAnalyzer().calculate_rfm(customers)   # used to raise "Bin edges must be unique"
    assert set(rfm["R_Score"]) == {1, 2, 3, 4, 5}


def test_churn_label_not_leaked(processed):
    dp, tx, _ = processed
    cp = ChurnPredictor(churn_threshold_days=90)
    labeled, past = cp.build_temporal_training_set(tx, dp.create_customer_features)

    assert 0.05 < labeled["is_churned"].mean() < 0.95
    # no order after the cutoff leaks into the features
    assert pd.to_datetime(labeled["last_purchase_date"]).max() <= pd.to_datetime(tx["order_date"]).max() - pd.Timedelta(days=90)
    assert past["order_date"].max() <= pd.to_datetime(tx["order_date"]).max() - pd.Timedelta(days=90)

    X_train, X_test, y_train, y_test, names = cp.prepare_training_data(cp.engineer_features(labeled, past))
    perf = cp.train_models(X_train, y_train, X_test, y_test)
    # a model that merely re-derives the label scores exactly 1.0; honest ones don't
    assert all(m["auc_roc"] < 0.995 for m in perf.values())
    assert all(m["auc_roc"] > 0.6 for m in perf.values())


def test_bgnbd_converges_and_clv_predicts(processed):
    _, tx, customers = processed
    calc = CLVCalculator()
    clv_data = calc.prepare_clv_data(tx.copy(), customers)
    calc.train_bgf_model(clv_data)
    calc.train_ggf_model(clv_data)
    assert calc.bgf_model.params_.notna().all()
    assert (calc.bgf_model.params_ > 0).all()
