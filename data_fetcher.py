"""
Data fetching module for multi-frequency A-share data from BaoStock.
data range: 2020-01-01 to 2025-01-01
- Stocks: 601628 China Life, 601688 Huatai Securities, 600519 Kweichow Moutai
- Frequencies: daily data (medium frequency), weekly data (low frequency),
  and 5-minute data (high frequency)
"""  

import os
import time
import baostock as bs
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

START_DATE = "2020-01-01"
END_DATE   = "2025-01-01"

# Target stocks
# stocks: 601628 China Life, 601688 Huatai Securities, 600519 Kweichow Moutai
TARGET_STOCKS = {
    "sh.601628": "China Life",
    "sh.601688": "Huatai Securities",
    "sh.600519": "Kweichow Moutai",
}

# BaoStock field definitions
DAILY_FIELDS = ("date,code,open,high,low,close,preclose,volume,amount,"
                "adjustflag,turn,tradestatus,pctChg,isST")
WEEKLY_FIELDS = ("date,code,open,high,low,close,volume,amount,adjustflag,"
                 "turn,pctChg")
MINUTE_FIELDS = ("date,time,code,open,high,low,close,volume,amount,adjustflag")


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to daily data: RSI, MACD, BOLL, CCI, SMA, DX."""  
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)

    # RSI (14)
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs    = avg_g / (avg_l + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD_DIF"] = ema12 - ema26
    df["MACD_DEA"] = df["MACD_DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_BAR"] = (df["MACD_DIF"] - df["MACD_DEA"]) * 2

    # BOLL (20)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BOLL_MID"]   = sma20
    df["BOLL_UPPER"] = sma20 + 2 * std20
    df["BOLL_LOWER"] = sma20 - 2 * std20

    # CCI (20)
    tp = (high + low + close) / 3
    df["CCI"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-9)

    # SMA (5, 10, 20)
    df["SMA5"]  = close.rolling(5).mean()
    df["SMA10"] = close.rolling(10).mean()
    df["SMA20"] = sma20

    # DX / ADX (14)
    tr    = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    plus  = (high - high.shift()).clip(lower=0)
    minus = (low.shift() - low).clip(lower=0)
    atr14 = tr.ewm(com=13, adjust=False).mean()
    pos14 = plus.ewm(com=13, adjust=False).mean()
    neg14 = minus.ewm(com=13, adjust=False).mean()
    plus_di  = 100 * pos14 / (atr14 + 1e-9)
    minus_di = 100 * neg14 / (atr14 + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df["DX"] = dx

    df.dropna(inplace=True)
    return df


def _fetch_daily(code: str, start: str, end: str, adjustflag: str = "3") -> pd.DataFrame:
    """Fetch daily data with forward-adjusted prices."""  
    rs = bs.query_history_k_data_plus(
        code, DAILY_FIELDS,
        start_date=start, end_date=end,
        frequency="d", adjustflag=adjustflag
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    numeric_cols = ["open", "high", "low", "close", "preclose",
                    "volume", "amount", "turn", "pctChg"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df.dropna(subset=["close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _fetch_weekly(code: str, start: str, end: str, adjustflag: str = "3") -> pd.DataFrame:
    """Fetch weekly data with forward-adjusted prices."""  
    rs = bs.query_history_k_data_plus(
        code, WEEKLY_FIELDS,
        start_date=start, end_date=end,
        frequency="w", adjustflag=adjustflag
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    for c in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df.dropna(subset=["close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _fetch_minute(code: str, start: str, end: str, adjustflag: str = "3") -> pd.DataFrame:
    """Fetch 5-minute data with forward-adjusted prices."""  
    rs = bs.query_history_k_data_plus(
        code, MINUTE_FIELDS,
        start_date=start, end_date=end,
        frequency="5", adjustflag=adjustflag
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    # time field looks like '20200102093500000', slice first 14 chars to get %Y%m%d%H%M%S
    df["datetime"] = pd.to_datetime(df["time"].str.slice(0, 14), format="%Y%m%d%H%M%S")
    df.dropna(subset=["close"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# Main download workflow
# ─────────────────────────────────────────────
def download_market_data():
    """Download multi-frequency market data for the target stocks."""
    print("\n===== Downloading market data =====")
    for code, name in TARGET_STOCKS.items():
        safe = code.replace(".", "_")
        print(f"\n[{name} ({code})]")

        # Daily data
        path = os.path.join(DATA_DIR, f"{safe}_daily.pkl")
        if not os.path.exists(path):
            print("  Fetching daily data...")
            df = _fetch_daily(code, START_DATE, END_DATE)
            df = add_technical_indicators(df)
            df.to_pickle(path)
            print(f"  Daily data saved: {path}  ({len(df)} rows)")
        else:
            print(f"  Daily data already exists: {path}")

        # Weekly data
        path = os.path.join(DATA_DIR, f"{safe}_weekly.pkl")
        if not os.path.exists(path):
            print("  Fetching weekly data...")
            df = _fetch_weekly(code, START_DATE, END_DATE)
            df.to_pickle(path)
            print(f"  Weekly data saved: {path}  ({len(df)} rows)")
        else:
            print(f"  Weekly data already exists: {path}")

        # 5-minute data, fetched year by year because of its larger volume
        path = os.path.join(DATA_DIR, f"{safe}_minute.pkl")
        if not os.path.exists(path):
            print("  Fetching 5-minute data year by year...")
            dfs = []
            years = range(2020, 2025)
            for y in years:
                s = f"{y}-01-01"
                e = f"{y}-12-31"
                print(f"    Year {y}...")
                df_y = _fetch_minute(code, s, e)
                dfs.append(df_y)
                time.sleep(0.5)  # Avoid overly frequent requests.
            df = pd.concat(dfs, ignore_index=True)
            df.to_pickle(path)
            print(f"  Minute data saved: {path}  ({len(df)} rows)")
        else:
            print(f"  Minute data already exists: {path}")

        time.sleep(0.5)


def main():
    print("Logging in to BaoStock...")
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")
    print(f"Login successful: {lg.error_msg}")

    try:
        download_market_data()
    finally:
        bs.logout()
        print("\nLogged out from BaoStock.")

    print("\n===== Data download completed =====")
    print("Data directory:", os.path.abspath(DATA_DIR))


if __name__ == "__main__":
    main()
