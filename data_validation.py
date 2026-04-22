"""
Validate downloaded market data and produce a data quality report.

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


PRICE_COLUMNS = ["open", "high", "low", "close", "preclose"]
NON_NEGATIVE_COLUMNS = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn"]


def infer_date_column(df: pd.DataFrame) -> str:
    if "datetime" in df.columns:
        return "datetime"
    if "date" in df.columns:
        return "date"
    raise ValueError("No date or datetime column found.")


def infer_frequency(source_name: str) -> str:
    lower_name = source_name.lower()
    if "minute" in lower_name:
        return "minute"
    if "weekly" in lower_name:
        return "weekly"
    if "daily" in lower_name:
        return "daily"
    return "unknown"


def infer_symbol(source_name: str) -> str:
    stem = Path(source_name).stem
    for suffix in ("_daily", "_weekly", "_minute"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)].replace("_", ".")
    return stem.replace("_", ".")


def _numeric_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    present = [col for col in columns if col in df.columns]
    if not present:
        return pd.DataFrame(index=df.index)
    return df[present].apply(pd.to_numeric, errors="coerce")


def validate_dataframe(
    df: pd.DataFrame,
    source_name: str,
    jump_threshold: float = 0.25,
) -> Dict[str, object]:
    date_col = infer_date_column(df)
    timestamps = pd.to_datetime(df[date_col], errors="coerce")
    missing_cells = int(df.isna().sum().sum())
    rows_with_missing = int(df.isna().any(axis=1).sum())

    numeric_values = _numeric_frame(df, NON_NEGATIVE_COLUMNS)
    price_values = _numeric_frame(df, PRICE_COLUMNS)

    if numeric_values.empty:
        negative_value_rows = 0
    else:
        negative_value_rows = int((numeric_values < 0).any(axis=1).sum())

    if price_values.empty:
        non_positive_price_rows = 0
    else:
        non_positive_price_rows = int((price_values <= 0).any(axis=1).sum())

    volume_values = _numeric_frame(df, ["volume"])
    zero_volume_rows = int((volume_values["volume"] == 0).sum()) if "volume" in volume_values else 0

    if all(col in df.columns for col in ["open", "high", "low", "close"]):
        ohlc = _numeric_frame(df, ["open", "high", "low", "close"])
        max_inside_bar = ohlc[["open", "low", "close"]].max(axis=1)
        min_inside_bar = ohlc[["open", "high", "close"]].min(axis=1)
        ohlc_violations = (ohlc["high"] < max_inside_bar) | (ohlc["low"] > min_inside_bar)
        ohlc_violation_rows = int(ohlc_violations.sum())
    else:
        ohlc_violation_rows = 0

    if "close" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce")
        close_returns = close.pct_change().replace([np.inf, -np.inf], np.nan).abs()
        large_return_jump_rows = int((close_returns > jump_threshold).sum())
        max_abs_close_return = close_returns.max(skipna=True)
    else:
        large_return_jump_rows = 0
        max_abs_close_return = np.nan

    summary = {
        "file": source_name,
        "symbol": infer_symbol(source_name),
        "frequency": infer_frequency(source_name),
        "date_column": date_col,
        "start_timestamp": timestamps.min(),
        "end_timestamp": timestamps.max(),
        "observations": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "is_chronologically_ordered": bool(timestamps.is_monotonic_increasing),
        "missing_cells": missing_cells,
        "rows_with_missing_values": rows_with_missing,
        "missing_cell_ratio": float(missing_cells / max(df.shape[0] * df.shape[1], 1)),
        "negative_value_rows": negative_value_rows,
        "non_positive_price_rows": non_positive_price_rows,
        "zero_volume_rows": zero_volume_rows,
        "ohlc_violation_rows": ohlc_violation_rows,
        "large_return_jump_rows": large_return_jump_rows,
        "max_abs_close_return": float(max_abs_close_return) if pd.notna(max_abs_close_return) else np.nan,
        "final_feature_set": ";".join(df.columns.astype(str)),
        "preprocessing_decisions": (
            "Downloaded from BaoStock; forward-adjusted prices requested with adjustflag=3; "
            "daily technical indicators are computed with rolling/ewm transformations; "
            "no imputation or scaling is applied in this validation step."
        ),
    }
    summary["validation_status"] = _status_from_summary(summary)
    return summary


def _status_from_summary(summary: Dict[str, object]) -> str:
    hard_fail_fields = [
        "duplicate_timestamps",
        "negative_value_rows",
        "non_positive_price_rows",
        "ohlc_violation_rows",
    ]
    if not summary["is_chronologically_ordered"]:
        return "review_required"
    if any(int(summary[field]) > 0 for field in hard_fail_fields):
        return "review_required"
    if int(summary["large_return_jump_rows"]) > 0:
        return "review_required"
    return "pass"


def load_pickles(data_dir: Path) -> List[Tuple[Path, pd.DataFrame]]:
    paths = sorted(data_dir.glob("*.pkl"))
    datasets = []
    for path in paths:
        datasets.append((path, pd.read_pickle(path)))
    return datasets


def add_common_calendar_metrics(
    summaries: List[Dict[str, object]],
    datasets: List[Tuple[Path, pd.DataFrame]],
) -> None:
    calendars: Dict[str, set] = {}
    file_dates: Dict[str, set] = {}

    for path, df in datasets:
        frequency = infer_frequency(path.name)
        date_col = infer_date_column(df)
        timestamps = pd.to_datetime(df[date_col], errors="coerce")
        if frequency == "minute":
            comparable_dates = set(timestamps.dropna())
        else:
            comparable_dates = set(timestamps.dropna().dt.normalize())
        calendars.setdefault(frequency, set()).update(comparable_dates)
        file_dates[path.name] = comparable_dates

    for summary in summaries:
        frequency = str(summary["frequency"])
        calendar = calendars.get(frequency, set())
        dates = file_dates.get(str(summary["file"]), set())
        missing_count = len(calendar - dates)
        summary["common_calendar_observations"] = len(calendar)
        summary["missing_against_common_calendar"] = missing_count
        summary["missing_against_common_calendar_ratio"] = float(missing_count / len(calendar)) if calendar else 0.0


def build_validation_report(data_dir: Path) -> pd.DataFrame:
    datasets = load_pickles(data_dir)
    summaries = [validate_dataframe(df, path.name) for path, df in datasets]
    add_common_calendar_metrics(summaries, datasets)
    return pd.DataFrame(summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate downloaded market data and write a CSV quality report.")
    parser.add_argument("--data-dir", default="data", help="Directory containing downloaded .pkl market data files.")
    parser.add_argument(
        "--output",
        default=os.path.join("results", "data_validation_summary.csv"),
        help="Output CSV path for the validation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_validation_report(data_dir)
    report.to_csv(output_path, index=False)
    print(f"Wrote data validation report to {output_path}")
    print(report[["file", "observations", "validation_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
