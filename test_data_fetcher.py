import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

sys.modules.setdefault(
    "baostock",
    types.SimpleNamespace(query_history_k_data_plus=None, login=None, logout=None),
)

import data_fetcher


class FakeBaoStockResult:
    def __init__(self, fields, rows, error_code="0", error_msg=""):
        self.fields = fields
        self._rows = rows
        self.error_code = error_code
        self.error_msg = error_msg
        self._idx = -1

    def next(self):
        self._idx += 1
        return self._idx < len(self._rows)

    def get_row_data(self):
        return self._rows[self._idx]


MINUTE_FIELDS = ["date", "time", "code", "open", "high", "low", "close", "volume", "amount", "adjustflag"]


def test_fetch_minute_converts_numeric_columns_and_sorts(monkeypatch):
    rows = [
        ["2020-01-02", "20200102100000000", "sh.000001", "11", "12", "10", "11.5", "200", "2300", "3"],
        ["2020-01-02", "20200102093500000", "sh.000001", "10", "11", "9", "10.5", "100", "1050", "3"],
    ]

    monkeypatch.setattr(
        data_fetcher.bs,
        "query_history_k_data_plus",
        lambda *args, **kwargs: FakeBaoStockResult(MINUTE_FIELDS, rows),
    )

    df = data_fetcher._fetch_minute("sh.000001", "2020-01-01", "2020-01-03")

    assert df["datetime"].is_monotonic_increasing
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        assert pd.api.types.is_numeric_dtype(df[column])


def test_fetch_daily_raises_on_api_error(monkeypatch):
    monkeypatch.setattr(data_fetcher.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        data_fetcher.bs,
        "query_history_k_data_plus",
        lambda *args, **kwargs: FakeBaoStockResult([], [], error_code="100", error_msg="mock failure"),
    )

    with pytest.raises(RuntimeError, match="mock failure"):
        data_fetcher._fetch_daily("sh.000001", "2020-01-01", "2020-01-03")


def test_fetch_weekly_raises_on_empty_result(monkeypatch):
    monkeypatch.setattr(
        data_fetcher.bs,
        "query_history_k_data_plus",
        lambda *args, **kwargs: FakeBaoStockResult(MINUTE_FIELDS, []),
    )

    with pytest.raises(ValueError, match="No data returned"):
        data_fetcher._fetch_weekly("sh.000001", "2020-01-01", "2020-01-03")


def test_add_technical_indicators_does_not_mutate_input_and_sorts():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=40, freq="D")[::-1],
            "open": range(40, 80),
            "high": range(41, 81),
            "low": range(39, 79),
            "close": range(40, 80),
            "volume": range(100, 140),
            "amount": range(1000, 1040),
        }
    )
    original_columns = list(df.columns)

    result = data_fetcher.add_technical_indicators(df)

    assert list(df.columns) == original_columns
    assert result["date"].is_monotonic_increasing
    assert "DX" in result.columns


def test_parse_stock_specs_accepts_custom_stock_names():
    stocks = data_fetcher.parse_stock_specs(["sh.600000=Shanghai Pudong Development Bank", "sz.000001"])

    assert stocks == {
        "sh.600000": "Shanghai Pudong Development Bank",
        "sz.000001": "sz.000001",
    }


def test_parse_args_accepts_custom_run_configuration():
    args = data_fetcher.parse_args(
        [
            "--start-date",
            "2021-01-01",
            "--end-date",
            "2021-12-31",
            "--data-dir",
            "custom_data",
            "--adjustflag",
            "2",
            "--stock",
            "sh.600000=SPD Bank",
        ]
    )

    assert args.start_date == "2021-01-01"
    assert args.end_date == "2021-12-31"
    assert args.data_dir == "custom_data"
    assert args.adjustflag == "2"
    assert args.stock == ["sh.600000=SPD Bank"]


def test_save_pickle_with_metadata_records_source_api_version(tmp_path, monkeypatch):
    monkeypatch.setattr(data_fetcher, "_source_package_version", lambda: "1.2.3")
    path = tmp_path / "sample.pkl"
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "close": [10.0]})

    data_fetcher._save_pickle_with_metadata(
        df,
        str(path),
        "sh.000001",
        "Sample",
        "daily",
        "3",
        "2024-01-01",
        "2024-01-31",
    )

    metadata = pd.read_json(f"{path}.metadata.json", typ="series")
    assert metadata["source_api"] == "query_history_k_data_plus"
    assert metadata["source_api_version"] == "1.2.3"
    assert metadata["start_date"] == "2024-01-01"
    assert metadata["end_date"] == "2024-01-31"


def test_cli_help_does_not_require_baostock_package():
    script_path = Path(__file__).with_name("data_fetcher.py")

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--start-date" in result.stdout
