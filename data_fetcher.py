"""
Data fetching module for multi-frequency A-share data from BaoStock.
data range: 2020-01-01 to 2025-01-01
- Stocks: 601628 China Life, 601688 Huatai Securities, 600519 Kweichow Moutai
- Frequencies: daily data (medium frequency), weekly data (low frequency),
  and 5-minute data (high frequency)
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

import pandas as pd
import numpy as np

try:
    import baostock as bs
except ImportError:
    bs = None

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

START_DATE = "2020-01-01"
END_DATE   = "2025-01-01"
ADJUSTFLAG = "3"
FETCH_RETRIES = 3
RETRY_SLEEP_SECONDS = 1.0

logger = logging.getLogger(__name__)

# Default target stocks
# stocks: 601628 China Life, 601688 Huatai Securities, 600519 Kweichow Moutai
DEFAULT_TARGET_STOCKS = {
    "sh.601628": "China Life",
    "sh.601688": "Huatai Securities",
    "sh.600519": "Kweichow Moutai",
}
TARGET_STOCKS = DEFAULT_TARGET_STOCKS

# BaoStock field definitions
DAILY_FIELDS = ("date,code,open,high,low,close,preclose,volume,amount,"
                "adjustflag,turn,tradestatus,pctChg,isST")
WEEKLY_FIELDS = ("date,code,open,high,low,close,volume,amount,adjustflag,"
                 "turn,pctChg")
MINUTE_FIELDS = ("date,time,code,open,high,low,close,volume,amount,adjustflag")

DAILY_NUMERIC_COLUMNS = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]
WEEKLY_NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]
MINUTE_NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
def parse_stock_specs(stock_specs: list[str]) -> dict[str, str]:
    """Parse CLI stock specs in CODE or CODE=NAME format."""
    stocks = {}
    for spec in stock_specs:
        if "=" in spec:
            code, name = spec.split("=", 1)
            code = code.strip()
            name = name.strip()
        else:
            code = spec.strip()
            name = code
        if not code:
            raise ValueError("Stock code cannot be empty.")
        stocks[code] = name or code
    return stocks


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download multi-frequency A-share data from BaoStock.")
    parser.add_argument("--start-date", default=START_DATE, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", default=END_DATE, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory where downloaded files are stored.")
    parser.add_argument("--adjustflag", default=ADJUSTFLAG, help="BaoStock adjustment flag.")
    parser.add_argument(
        "--stock",
        action="append",
        help="Target stock in CODE or CODE=NAME format. Can be repeated. Defaults to the built-in thesis sample.",
    )
    return parser.parse_args(argv)


def _source_package_version() -> str:
    try:
        return version("baostock")
    except PackageNotFoundError:
        if bs is None:
            return "not-installed"
        return getattr(bs, "__version__", "unknown")


def _require_baostock():
    if bs is None:
        raise RuntimeError(
            "BaoStock is required to download data. Install it with "
            "`pip install -r requirements_data.txt` from the code directory."
        )
    return bs


def _yearly_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date.")

    ranges = []
    for year in range(start.year, end.year + 1):
        year_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        year_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        ranges.append((year_start.strftime("%Y-%m-%d"), year_end.strftime("%Y-%m-%d")))
    return ranges


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to daily data: RSI, MACD, BOLL, CCI, SMA, DX."""
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise ValueError("Daily data contains invalid date values before indicator calculation.")
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

    required_columns = ["close", "high", "low"]
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns for technical indicators: {missing_required}")

    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    indicator_columns = []

    # RSI (14)
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs    = avg_g / (avg_l + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    indicator_columns.append("RSI")

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD_DIF"] = ema12 - ema26
    df["MACD_DEA"] = df["MACD_DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_BAR"] = (df["MACD_DIF"] - df["MACD_DEA"]) * 2
    indicator_columns.extend(["MACD_DIF", "MACD_DEA", "MACD_BAR"])

    # BOLL (20)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BOLL_MID"]   = sma20
    df["BOLL_UPPER"] = sma20 + 2 * std20
    df["BOLL_LOWER"] = sma20 - 2 * std20
    indicator_columns.extend(["BOLL_MID", "BOLL_UPPER", "BOLL_LOWER"])

    # CCI (20)
    tp = (high + low + close) / 3
    df["CCI"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-9)
    indicator_columns.append("CCI")

    # SMA (5, 10, 20)
    df["SMA5"]  = close.rolling(5).mean()
    df["SMA10"] = close.rolling(10).mean()
    df["SMA20"] = sma20
    indicator_columns.extend(["SMA5", "SMA10", "SMA20"])

    # DX (14)
    tr    = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr14 = tr.ewm(com=13, adjust=False).mean()
    pos14 = plus.ewm(com=13, adjust=False).mean()
    neg14 = minus.ewm(com=13, adjust=False).mean()
    plus_di  = 100 * pos14 / (atr14 + 1e-9)
    minus_di = 100 * neg14 / (atr14 + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df["DX"] = dx
    indicator_columns.append("DX")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna(subset=indicator_columns).reset_index(drop=True)


def _query_history(code: str, fields: str, start: str, end: str, frequency: str, adjustflag: str):
    bao = _require_baostock()
    last_error = ""
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            rs = bao.query_history_k_data_plus(
                code,
                fields,
                start_date=start,
                end_date=end,
                frequency=frequency,
                adjustflag=adjustflag,
            )
        except Exception as exc:
            last_error = str(exc)
        else:
            if rs.error_code == "0":
                return rs
            last_error = getattr(rs, "error_msg", "")

        if attempt < FETCH_RETRIES:
            wait_seconds = RETRY_SLEEP_SECONDS * attempt
            logger.warning(
                "BaoStock query failed for %s, frequency=%s, attempt %s/%s. Retrying in %.1f seconds.",
                code,
                frequency,
                attempt,
                FETCH_RETRIES,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"BaoStock query failed for {code}, frequency={frequency}, "
        f"date range={start} to {end}: {last_error}"
    )


def _result_to_dataframe(rs, code: str, frequency: str) -> pd.DataFrame:
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        raise ValueError(f"No data returned for {code}, frequency={frequency}.")
    return pd.DataFrame(rows, columns=rs.fields)


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _parse_timestamp_column(df: pd.DataFrame, column: str, source: str, date_format: Optional[str] = None) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"{source} data is missing required timestamp column: {column}")
    df = df.copy()
    df[column] = pd.to_datetime(df[column], format=date_format, errors="coerce")
    invalid_count = int(df[column].isna().sum())
    if invalid_count:
        raise ValueError(f"{source} data contains {invalid_count} invalid {column} values.")
    return df


def _sort_and_deduplicate(df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    return (
        df.sort_values(timestamp_column)
        .drop_duplicates(subset=[timestamp_column], keep="last")
        .reset_index(drop=True)
    )


def _metadata_for_dataframe(
    df: pd.DataFrame,
    code: str,
    name: str,
    frequency: str,
    adjustflag: str,
    start_date: str,
    end_date: str,
) -> dict:
    return {
        "source": "BaoStock",
        "source_api": "query_history_k_data_plus",
        "source_api_version": _source_package_version(),
        "symbol": code,
        "name": name,
        "frequency": frequency,
        "start_date": start_date,
        "end_date": end_date,
        "adjustflag": adjustflag,
        "query_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_created_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_status": "created_during_download",
        "rows": int(len(df)),
        "columns": list(df.columns),
    }


def _write_metadata(path: str, metadata: dict) -> None:
    metadata_path = f"{path}.metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _save_pickle_with_metadata(
    df: pd.DataFrame,
    path: str,
    code: str,
    name: str,
    frequency: str,
    adjustflag: str,
    start_date: str,
    end_date: str,
) -> None:
    df.to_pickle(path)
    metadata = _metadata_for_dataframe(
        df,
        code,
        name,
        frequency,
        adjustflag,
        start_date,
        end_date,
    )
    _write_metadata(path, metadata)


def _fetch_daily(code: str, start: str, end: str, adjustflag: str = "3") -> pd.DataFrame:
    """Fetch daily data with forward-adjusted prices."""
    rs = _query_history(code, DAILY_FIELDS, start, end, "d", adjustflag)
    df = _result_to_dataframe(rs, code, "daily")
    df = _coerce_numeric_columns(df, DAILY_NUMERIC_COLUMNS)
    df = _parse_timestamp_column(df, "date", "daily")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"])
    return _sort_and_deduplicate(df, "date")


def _fetch_weekly(code: str, start: str, end: str, adjustflag: str = "3") -> pd.DataFrame:
    """Fetch weekly data with forward-adjusted prices."""
    rs = _query_history(code, WEEKLY_FIELDS, start, end, "w", adjustflag)
    df = _result_to_dataframe(rs, code, "weekly")
    df = _coerce_numeric_columns(df, WEEKLY_NUMERIC_COLUMNS)
    df = _parse_timestamp_column(df, "date", "weekly")
    df = df.dropna(subset=["close"])
    return _sort_and_deduplicate(df, "date")


def _fetch_minute(code: str, start: str, end: str, adjustflag: str = "3") -> pd.DataFrame:
    """Fetch 5-minute data with forward-adjusted prices."""
    rs = _query_history(code, MINUTE_FIELDS, start, end, "5", adjustflag)
    df = _result_to_dataframe(rs, code, "minute")
    df = _coerce_numeric_columns(df, MINUTE_NUMERIC_COLUMNS)
    # time field looks like '20200102093500000', slice first 14 chars to get %Y%m%d%H%M%S
    if "time" not in df.columns:
        raise ValueError("Minute data is missing required time column.")
    df["datetime"] = df["time"].astype("string").str.slice(0, 14)
    df = _parse_timestamp_column(df, "datetime", "minute", "%Y%m%d%H%M%S")
    df = df.dropna(subset=["close"])
    return _sort_and_deduplicate(df, "datetime")


# -------------------------------------------------------------
# Main download workflow
# -------------------------------------------------------------
def download_market_data(
    target_stocks: Optional[dict[str, str]] = None,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    data_dir: str = DATA_DIR,
    adjustflag: str = ADJUSTFLAG,
):
    """Download multi-frequency market data for the target stocks."""
    os.makedirs(data_dir, exist_ok=True)
    stocks = target_stocks or DEFAULT_TARGET_STOCKS
    logger.info("Downloading market data.")
    for code, name in stocks.items():
        safe = code.replace(".", "_")
        logger.info("Processing %s (%s).", name, code)

        # Daily data
        path = os.path.join(data_dir, f"{safe}_daily.pkl")
        if not os.path.exists(path):
            logger.info("Fetching daily data for %s.", code)
            df = _fetch_daily(code, start_date, end_date, adjustflag)
            df = add_technical_indicators(df)
            _save_pickle_with_metadata(df, path, code, name, "daily", adjustflag, start_date, end_date)
            logger.info("Daily data saved to %s (%s rows).", path, len(df))
        else:
            logger.info("Daily data already exists: %s.", path)

        # Weekly data
        path = os.path.join(data_dir, f"{safe}_weekly.pkl")
        if not os.path.exists(path):
            logger.info("Fetching weekly data for %s.", code)
            df = _fetch_weekly(code, start_date, end_date, adjustflag)
            _save_pickle_with_metadata(df, path, code, name, "weekly", adjustflag, start_date, end_date)
            logger.info("Weekly data saved to %s (%s rows).", path, len(df))
        else:
            logger.info("Weekly data already exists: %s.", path)

        # 5-minute data, fetched year by year because of its larger volume
        path = os.path.join(data_dir, f"{safe}_minute.pkl")
        if not os.path.exists(path):
            logger.info("Fetching 5-minute data year by year for %s.", code)
            dfs = []
            for s, e in _yearly_ranges(start_date, end_date):
                logger.info("Fetching %s 5-minute data from %s to %s.", code, s, e)
                df_y = _fetch_minute(code, s, e, adjustflag)
                dfs.append(df_y)
                time.sleep(0.5)  # Avoid overly frequent requests.
            df = _sort_and_deduplicate(pd.concat(dfs, ignore_index=True), "datetime")
            _save_pickle_with_metadata(df, path, code, name, "5-minute", adjustflag, start_date, end_date)
            logger.info("Minute data saved to %s (%s rows).", path, len(df))
        else:
            logger.info("Minute data already exists: %s.", path)

        time.sleep(0.5)


def main():
    args = parse_args()
    target_stocks = parse_stock_specs(args.stock) if args.stock else DEFAULT_TARGET_STOCKS
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bao = _require_baostock()
    logger.info("Logging in to BaoStock...")
    lg = bao.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")
    logger.info("Login successful: %s", lg.error_msg)

    try:
        download_market_data(
            target_stocks=target_stocks,
            start_date=args.start_date,
            end_date=args.end_date,
            data_dir=args.data_dir,
            adjustflag=args.adjustflag,
        )
    finally:
        bao.logout()
        logger.info("Logged out from BaoStock.")

    logger.info("Data download completed.")
    logger.info("Data directory: %s", os.path.abspath(args.data_dir))


if __name__ == "__main__":
    main()
