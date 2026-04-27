import numpy as np
import pytest
from src.evaluation.metrics import calculate_metrics, calculate_metrics_from_log


def test_metrics_perfect_predictions():
    y_true = np.array([10_000_000.0, 15_000_000.0, 20_000_000.0])
    y_pred = np.array([10_000_000.0, 15_000_000.0, 20_000_000.0])

    m = calculate_metrics(y_true, y_pred)
    assert np.isclose(m["mdape"], 0.0)
    assert np.isclose(m["mape"], 0.0)
    assert np.isclose(m["wape"], 0.0)
    assert np.isclose(m["pep10"], 100.0)
    assert np.isclose(m["pep20"], 100.0)
    assert np.isclose(m["rmsle"], 0.0)
    assert np.isclose(m["r2_log"], 1.0)


def test_metrics_pep_thresholds():
    y_true = np.array([100.0, 100.0, 100.0, 100.0])
    # 5% error, 15% error, 25% error, 30% error
    y_pred = np.array([105.0, 115.0, 125.0, 130.0])

    m = calculate_metrics(y_true, y_pred)
    # PEP10: only 1 out of 4 (25%)
    assert np.isclose(m["pep10"], 25.0)
    # PEP20: 2 out of 4 (50%)
    assert np.isclose(m["pep20"], 50.0)
    # MdAPE: median of [5, 15, 25, 30] = 20%
    assert np.isclose(m["mdape"], 20.0)


def test_metrics_from_log_transform():
    y_true_log = np.log1p(np.array([1_000_000.0, 2_000_000.0]))
    y_pred_log = y_true_log.copy()

    m = calculate_metrics_from_log(y_true_log, y_pred_log)
    assert np.isclose(m["mdape"], 0.0)
    assert m["pep20"] == 100.0
