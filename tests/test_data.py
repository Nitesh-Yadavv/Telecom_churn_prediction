"""
tests/test_data.py

Unit tests for app/data.py cached data loaders.
Validates file loading, return types, schema consistency, missing-file handling,
empty-file handling, corrupt-file handling, SHAP validation, and plot fallbacks.
"""
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from app import data


def test_load_dashboard_data():
    df = data.load_dashboard_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7043
    assert "customer_id" in df.columns
    assert "probability" in df.columns
    assert "band" in df.columns
    assert "split" in df.columns


def test_load_test_predictions():
    df = data.load_test_predictions()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1409
    assert "p_cal" in df.columns
    assert "actual_churn" in df.columns


def test_load_metrics():
    metrics = data.load_metrics()
    assert isinstance(metrics, dict)
    assert pytest.approx(metrics["cutoff"], rel=1e-3) == 1 / 3
    assert metrics["n_test"] == 1409
    assert metrics["flagged"] == 496


def test_load_overview_numbers():
    overview = data.load_overview_numbers()
    assert isinstance(overview, dict)
    assert overview["n_customers"] == 7043
    assert overview["revenue_lost_share"] == 0.305


def test_load_segment_revenue():
    df = data.load_segment_revenue()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "dimension" in df.columns
    assert "revenue_lost" in df.columns


def test_load_segment_metrics():
    df = data.load_segment_metrics()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "group_by" in df.columns
    assert "recall" in df.columns


def test_load_band_table():
    df = data.load_band_table()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 8
    assert set(df["sample"].unique()) == {"test", "oof"}


def test_load_cv_results():
    df = data.load_cv_results()
    assert isinstance(df, pd.DataFrame)
    assert "pipeline" in df.columns
    assert "auc_pr" in df.columns


def test_load_curves():
    pr = data.load_pr_curve()
    roc = data.load_roc_curve()
    gains = data.load_gains_curve()
    cal = data.load_calibration_curve()
    cutoff = data.load_cutoff_sweep()

    assert {"precision", "recall"}.issubset(pr.columns)
    assert {"fpr", "tpr"}.issubset(roc.columns)
    assert {"frac_customers", "frac_churners"}.issubset(gains.columns)
    assert {"model", "mean_pred", "frac_pos"}.issubset(cal.columns)
    assert "actual_saving" in cutoff.columns


def test_load_shap_importance():
    df = data.load_shap_importance()
    assert isinstance(df, pd.DataFrame)
    assert "feature" in df.columns
    assert "importance" in df.columns
    assert pd.api.types.is_numeric_dtype(df["importance"])
    assert len(df) > 0


def test_shap_importance_invalid_schema(tmp_path):
    one_col = tmp_path / "shap_one_col.csv"
    one_col.write_text("only_one_column\nval1\nval2\n", encoding="utf-8")
    with (
        patch("streamlit.error") as mock_err,
        pytest.raises(ValueError, match="requires at least 2 columns"),
    ):
        data.load_shap_importance(path=one_col)
    mock_err.assert_called()


def test_load_shap_plot_path():
    path = data.load_shap_plot_path()
    assert isinstance(path, Path)
    assert path.is_file()
    assert path.suffix == ".png"


def test_shap_plot_fallback(tmp_path):
    plots_dir = tmp_path / "outputs" / "plots"
    plots_dir.mkdir(parents=True)
    fallback_file = plots_dir / "shap_grouped_importance.png"
    fallback_file.write_text("fake_png", encoding="utf-8")

    resolved = data.load_shap_plot_path(plots_dir=plots_dir)
    assert resolved == fallback_file


def test_shap_plot_missing_all(tmp_path):
    plots_dir = tmp_path / "outputs" / "plots"
    plots_dir.mkdir(parents=True)

    with (
        patch("streamlit.error") as mock_err,
        pytest.raises(FileNotFoundError, match="No SHAP plot file found"),
    ):
        data.load_shap_plot_path(plots_dir=plots_dir)
    mock_err.assert_called_once()


def test_load_risk_bands():
    bands = data.load_risk_bands()
    assert isinstance(bands, dict)
    assert "labels" in bands
    assert "edges" in bands


def test_load_model_config():
    cfg = data.load_model_config()
    assert isinstance(cfg, dict)
    assert "feature_order" in cfg
    assert "created" in cfg


def test_missing_file_raises_and_shows_error(tmp_path):
    missing = tmp_path / "non_existent.csv"
    with patch("streamlit.error") as mock_st_error:
        with pytest.raises(FileNotFoundError, match="Missing required data file"):
            data._load_csv(missing)
        mock_st_error.assert_called_once()
        assert "non_existent.csv" in mock_st_error.call_args[0][0]


def test_empty_csv_raises_and_shows_error(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with patch("streamlit.error") as mock_st_error:
        with pytest.raises(ValueError, match="Data file is empty"):
            data._load_csv(empty_csv)
        mock_st_error.assert_called_once()


def test_corrupt_json_raises_and_shows_error(tmp_path):
    corrupt = tmp_path / "bad.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    with patch("streamlit.error") as mock_st_error:
        with pytest.raises(ValueError, match="Corrupt or unreadable JSON file"):
            data._load_json(corrupt)
        mock_st_error.assert_called_once()


def test_corrupt_csv_raises_and_shows_error(tmp_path):
    corrupt = tmp_path / "bad.csv"
    corrupt.write_bytes(b"\x80abc,123\n")
    with patch("streamlit.error") as mock_st_error:
        with pytest.raises(ValueError, match="Corrupt or unreadable CSV file"):
            data._load_csv(corrupt)
        mock_st_error.assert_called_once()


def test_caching_returns_same_dataframe():
    df1 = data.load_dashboard_data()
    df2 = data.load_dashboard_data()
    pd.testing.assert_frame_equal(df1, df2)
