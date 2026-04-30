import pandas as pd

from pathlib import Path

import pytest

from data_validation import add_common_calendar_metrics, build_validation_report, validate_dataframe


def test_validate_dataframe_flags_duplicate_timestamps_and_bad_ohlc():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "open": [10.0, 10.2, 10.5],
            "high": [10.5, 10.1, 10.7],
            "low": [9.8, 10.0, 10.4],
            "close": [10.2, 10.3, 10.6],
            "volume": [1000, 1200, -1],
            "amount": [10000, 12200, 11130],
        }
    )

    summary = validate_dataframe(df, "sh.000001_daily.pkl")

    assert summary["duplicate_timestamps"] == 1
    assert summary["negative_value_rows"] == 1
    assert summary["ohlc_violation_rows"] == 1
    assert summary["is_chronologically_ordered"] is True


def test_validate_dataframe_flags_unsorted_dates_and_large_return_jumps():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"]),
            "open": [10.0, 20.0, 10.0],
            "high": [10.2, 20.5, 10.5],
            "low": [9.9, 19.8, 9.8],
            "close": [10.0, 20.0, 10.0],
            "volume": [1000, 1200, 1300],
            "amount": [10000, 24000, 13000],
        }
    )

    summary = validate_dataframe(df, "sh.000001_daily.pkl", jump_threshold=0.4)

    assert summary["is_chronologically_ordered"] is False
    assert summary["large_return_jump_rows"] == 2


def test_validate_dataframe_allows_negative_return_fields():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.0],
            "volume": [1000, 1100],
            "amount": [10100, 11000],
            "pctChg": [-1.0, -0.5],
        }
    )

    summary = validate_dataframe(df, "sh.000001_daily.pkl")

    assert summary["negative_value_rows"] == 0


def test_common_calendar_gaps_are_reported_without_changing_hard_validation_status():
    first = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "open": [10.0],
            "high": [10.2],
            "low": [9.9],
            "close": [10.1],
            "volume": [1000],
        }
    )
    second = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [1000, 1200],
        }
    )
    summaries = [
        validate_dataframe(first, "sh.000001_daily.pkl"),
        validate_dataframe(second, "sh.000002_daily.pkl"),
    ]

    add_common_calendar_metrics(
        summaries,
        [(Path("sh.000001_daily.pkl"), first), (Path("sh.000002_daily.pkl"), second)],
    )

    assert summaries[0]["missing_against_common_calendar"] == 1
    assert summaries[0]["validation_status"] == "pass"


def test_large_return_jump_is_warning_not_fail():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [10.0, 20.0],
            "high": [10.2, 20.5],
            "low": [9.8, 19.5],
            "close": [10.0, 20.0],
            "volume": [1000, 1200],
        }
    )

    summary = validate_dataframe(df, "sh.000001_daily.pkl", jump_threshold=0.25)

    assert summary["large_return_jump_rows"] == 1
    assert summary["validation_status"] == "warning"


def test_preclose_mismatch_is_reported():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "open": [10.0, 10.2, 10.4],
            "high": [10.5, 10.6, 10.8],
            "low": [9.8, 10.0, 10.2],
            "close": [10.1, 10.3, 10.5],
            "preclose": [10.0, 10.1, 99.0],
            "volume": [1000, 1200, 1300],
        }
    )

    summary = validate_dataframe(df, "sh.000001_daily.pkl")

    assert summary["preclose_mismatch_rows"] == 1
    assert summary["validation_status"] == "warning"


def test_build_validation_report_raises_for_empty_data_dir(tmp_path):
    with pytest.raises(ValueError, match="No .pkl files found"):
        build_validation_report(tmp_path)


def test_report_includes_machine_readable_feature_set():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "open": [10.0],
            "high": [10.2],
            "low": [9.9],
            "close": [10.1],
            "volume": [1000],
        }
    )

    summary = validate_dataframe(df, "sh.000001_daily.pkl")

    assert summary["final_feature_set_json"] == '["date", "open", "high", "low", "close", "volume"]'


def test_indicator_invalid_values_are_reported_after_warmup():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [1000, 1200],
            "RSI": [50.0, float("inf")],
            "MACD_DIF": [0.1, 0.2],
        }
    )

    summary = validate_dataframe(df, "sh.000001_daily.pkl")

    assert summary["engineered_feature_infinite_rows"] == 1
    assert summary["validation_status"] == "fail"


def test_minute_common_calendar_uses_trading_days_not_exact_timestamps():
    first = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02 09:35:00", "2024-01-02 09:40:00"]),
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [1000, 1200],
        }
    )
    second = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02 09:35:00"]),
            "open": [20.0],
            "high": [20.2],
            "low": [19.9],
            "close": [20.1],
            "volume": [1000],
        }
    )
    summaries = [
        validate_dataframe(first, "sh.000001_minute.pkl"),
        validate_dataframe(second, "sh.000002_minute.pkl"),
    ]

    add_common_calendar_metrics(
        summaries,
        [(Path("sh.000001_minute.pkl"), first), (Path("sh.000002_minute.pkl"), second)],
    )

    assert summaries[1]["missing_against_common_calendar"] == 0
    assert summaries[0]["intraday_expected_observations_per_day"] == 2
    assert summaries[1]["intraday_low_coverage_days"] == 0
