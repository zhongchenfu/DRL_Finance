"""
Data-loading helpers for the R-PPO environment setup.

This module bridges the validated raw market data produced by data_fetcher.py
with the multi-frequency trading environment. It does not train models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"

PRICE_VOLUME_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


@dataclass
class MultiFrequencyData:
    low: pd.DataFrame
    mid: pd.DataFrame
    high: pd.DataFrame
    trade_start_date: pd.Timestamp | None = None


def _safe_code(code: str) -> str:
    return code.replace(".", "_")


def _load_pickle(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required market data file not found: {path}")
    return pd.read_pickle(path)


def _prepare_frame(df: pd.DataFrame, date_col: str, required_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in [date_col, *required_columns] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    prepared = df.copy()
    prepared[date_col] = pd.to_datetime(prepared[date_col], errors="coerce")
    invalid_dates = int(prepared[date_col].isna().sum())
    if invalid_dates:
        raise ValueError(f"{date_col} contains {invalid_dates} invalid values.")

    for column in required_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    invalid_numeric = int(prepared[required_columns].isna().sum().sum())
    if invalid_numeric:
        raise ValueError(f"Numeric market columns contain {invalid_numeric} invalid values.")

    return prepared.sort_values(date_col).drop_duplicates(subset=[date_col]).reset_index(drop=True)


def _split_by_time(
    df: pd.DataFrame,
    date_col: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    train = df[(df[date_col] >= train_start_ts) & (df[date_col] <= train_end_ts)].copy()
    test = df[(df[date_col] >= test_start_ts) & (df[date_col] <= test_end_ts)].copy()
    if train.empty:
        raise ValueError(f"Training split is empty for {date_col}.")
    if test.empty:
        raise ValueError(f"Test split is empty for {date_col}.")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _slice_with_context(
    df: pd.DataFrame,
    date_col: str,
    period_start: str,
    period_end: str,
    context_start: str | None = None,
) -> pd.DataFrame:
    period_start_ts = pd.Timestamp(period_start)
    period_end_ts = pd.Timestamp(period_end)
    context_start_ts = pd.Timestamp(context_start or period_start)
    active = df[(df[date_col] >= period_start_ts) & (df[date_col] <= period_end_ts)]
    if active.empty:
        raise ValueError(f"Split is empty for {date_col}: {period_start} to {period_end}.")
    return df[(df[date_col] >= context_start_ts) & (df[date_col] <= period_end_ts)].copy().reset_index(drop=True)


def _load_prepared_frames(code: str, data_dir: str | Path) -> MultiFrequencyData:
    safe = _safe_code(code)
    data_path = Path(data_dir)
    return MultiFrequencyData(
        mid=_prepare_frame(
            _load_pickle(data_path / f"{safe}_daily.pkl"),
            "date",
            PRICE_VOLUME_COLUMNS,
        ),
        low=_prepare_frame(
            _load_pickle(data_path / f"{safe}_weekly.pkl"),
            "date",
            PRICE_VOLUME_COLUMNS,
        ),
        high=_prepare_frame(
            _load_pickle(data_path / f"{safe}_minute.pkl"),
            "datetime",
            PRICE_VOLUME_COLUMNS,
        ),
    )


def load_experiment_data(
    code: str,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    train_start: str = "2020-03-01",
    train_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
    test_end: str = "2024-12-31",
) -> tuple[MultiFrequencyData, MultiFrequencyData]:
    prepared = _load_prepared_frames(code, data_dir)
    daily, weekly, minute = prepared.mid, prepared.low, prepared.high

    train_mid, test_mid = _split_by_time(daily, "date", train_start, train_end, test_start, test_end)
    train_low, test_low = _split_by_time(weekly, "date", train_start, train_end, test_start, test_end)
    train_high, test_high = _split_by_time(minute, "datetime", train_start, train_end, test_start, test_end)

    return (
        MultiFrequencyData(low=train_low, mid=train_mid, high=train_high),
        MultiFrequencyData(low=test_low, mid=test_mid, high=test_high),
    )


def load_experiment_data_with_validation(
    code: str,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    train_start: str = "2020-03-01",
    train_end: str = "2023-06-30",
    validation_start: str = "2023-07-01",
    validation_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
    test_end: str = "2024-12-31",
) -> tuple[MultiFrequencyData, MultiFrequencyData, MultiFrequencyData]:
    """
    Load train, validation, and test periods with leakage-safe warm-up context.

    Validation and test frames retain earlier rows for rolling observations.
    Environments use trade_start_date to avoid trading during that context.
    """
    prepared = _load_prepared_frames(code, data_dir)

    def build_period(period_start: str, period_end: str, context_start: str | None = None) -> MultiFrequencyData:
        return MultiFrequencyData(
            low=_slice_with_context(prepared.low, "date", period_start, period_end, context_start),
            mid=_slice_with_context(prepared.mid, "date", period_start, period_end, context_start),
            high=_slice_with_context(prepared.high, "datetime", period_start, period_end, context_start),
            trade_start_date=pd.Timestamp(period_start),
        )

    train = build_period(train_start, train_end)
    validation = build_period(validation_start, validation_end, context_start=train_start)
    test = build_period(test_start, test_end, context_start=train_start)
    return train, validation, test
