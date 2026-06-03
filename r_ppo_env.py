"""
R-PPO multi-frequency trading environment.

Timing convention:
- At execution day t, the observation contains information available up to
  day t - decision_lag.
- The action is executed at the open price of day t.
- The reward is realized from the open of day t to the open of day t + 1.

With the default decision_lag=1, the agent cannot observe same-day daily or
5-minute data before deciding the position for day t. The default high-frequency
window length is 48, which corresponds to one trading day of 5-minute bars.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


REWARD_CLIP = 0.05
NORMALIZED_FEATURE_CLIP = 10.0
PORTFOLIO_FEATURES = 3
DEFAULT_REBALANCE_THRESHOLD = 0.05
DEFAULT_TURNOVER_PENALTY_COEF = 0.0010
BASE_FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
OHLC_COLUMNS = {"open", "high", "low", "close"}
LOG_SCALE_COLUMNS = {"volume", "amount"}
PRICE_LEVEL_INDICATOR_COLUMNS = {
    "BOLL_MID",
    "BOLL_UPPER",
    "BOLL_LOWER",
    "SMA5",
    "SMA10",
    "SMA20",
}
PRICE_DELTA_INDICATOR_COLUMNS = {"MACD_DIF", "MACD_DEA", "MACD_BAR"}


def normalize_feature_frame(
    df: pd.DataFrame,
    columns: list[str],
    normalization_stats: Optional[dict] = None,
) -> tuple[np.ndarray, dict]:
    """
    Convert raw market columns into stationary features and standardize them.

    Statistics are fitted when normalization_stats is omitted. Test
    environments must reuse statistics fitted on the corresponding training
    environment.
    """
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    close = numeric["close"]
    previous_close = close.shift(1)
    transformed = pd.DataFrame(index=df.index)

    for column in columns:
        values = numeric[column]
        if column in OHLC_COLUMNS:
            transformed[column] = values / previous_close - 1.0
        elif column in LOG_SCALE_COLUMNS:
            transformed[column] = np.log1p(values.clip(lower=0.0))
        elif column in PRICE_LEVEL_INDICATOR_COLUMNS:
            transformed[column] = values / close - 1.0
        elif column in PRICE_DELTA_INDICATOR_COLUMNS:
            transformed[column] = values / close
        else:
            transformed[column] = values

    values = (
        transformed.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    if normalization_stats is None:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        normalization_stats = {
            "columns": tuple(columns),
            "mean": mean,
            "std": std,
        }
    elif tuple(columns) != tuple(normalization_stats["columns"]):
        raise ValueError("Normalization columns do not match the environment feature columns.")

    normalized = (values - normalization_stats["mean"]) / normalization_stats["std"]
    normalized = np.clip(normalized, -NORMALIZED_FEATURE_CLIP, NORMALIZED_FEATURE_CLIP)
    return normalized, normalization_stats


class MultiFreqStockEnv(gym.Env):
    """
    Multi-frequency individual-stock trading environment for R-PPO.

    Observation:
        Flattened concatenation of low-frequency, mid-frequency, and
        high-frequency rolling windows, followed by position weight, cash
        ratio, and the previous action.

    Action:
        A continuous scalar in [-1, 1], mapped to target long-only stock
        weight in [0, 1].
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df_low: pd.DataFrame,
        df_mid: pd.DataFrame,
        df_high: pd.DataFrame,
        initial_amount: float = 1_000_000,
        transaction_fee_percent: float = 0.003,
        low_len: int = 8,
        mid_len: int = 15,
        high_len: int = 48,
        tech_indicator_list: Optional[list[str]] = None,
        decision_lag: int = 1,
        normalization_stats: Optional[dict[str, dict]] = None,
        trade_start_date: Optional[str | pd.Timestamp] = None,
        rebalance_threshold: float = DEFAULT_REBALANCE_THRESHOLD,
        turnover_penalty_coef: float = DEFAULT_TURNOVER_PENALTY_COEF,
    ):
        super().__init__()
        if decision_lag < 0:
            raise ValueError("decision_lag must be non-negative.")
        if not 0.0 <= rebalance_threshold <= 1.0:
            raise ValueError("rebalance_threshold must be between 0 and 1.")
        if turnover_penalty_coef < 0.0:
            raise ValueError("turnover_penalty_coef must be non-negative.")

        self.df_low = self._prepare_frame(df_low, "date")
        self.df_mid = self._prepare_frame(df_mid, "date")
        self.df_high = self._prepare_frame(df_high, "datetime")

        self.initial_amount = float(initial_amount)
        self.transaction_fee_percent = float(transaction_fee_percent)
        self.low_len = int(low_len)
        self.mid_len = int(mid_len)
        self.high_len = int(high_len)
        self.decision_lag = int(decision_lag)
        self.tech_indicator_list = list(tech_indicator_list or [])
        self.trade_start_date = pd.Timestamp(trade_start_date) if trade_start_date is not None else None
        self.rebalance_threshold = float(rebalance_threshold)
        self.turnover_penalty_coef = float(turnover_penalty_coef)

        self.low_cols = self._available_columns(self.df_low, BASE_FEATURE_COLUMNS)
        self.mid_cols = self._available_columns(self.df_mid, BASE_FEATURE_COLUMNS + self.tech_indicator_list)
        self.high_cols = self._available_columns(self.df_high, BASE_FEATURE_COLUMNS)

        self.low_features = len(self.low_cols)
        self.mid_features = len(self.mid_cols)
        self.high_features = len(self.high_cols)
        if "open" not in self.mid_cols:
            raise ValueError("The environment requires an open column for trade execution.")

        self.state_dim = (
            self.low_len * self.low_features
            + self.mid_len * self.mid_features
            + self.high_len * self.high_features
            + PORTFOLIO_FEATURES
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        supplied_stats = normalization_stats or {}
        self.np_low, low_stats = normalize_feature_frame(self.df_low, self.low_cols, supplied_stats.get("low"))
        self.np_mid, mid_stats = normalize_feature_frame(self.df_mid, self.mid_cols, supplied_stats.get("mid"))
        self.np_high, high_stats = normalize_feature_frame(self.df_high, self.high_cols, supplied_stats.get("high"))
        self.normalization_stats = {
            "low": low_stats,
            "mid": mid_stats,
            "high": high_stats,
        }
        self.execution_prices = pd.to_numeric(self.df_mid["open"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(self.execution_prices).all() or (self.execution_prices <= 0).any():
            raise ValueError("The environment requires finite positive open prices.")

        self.mid_dates = pd.to_datetime(self.df_mid["date"])
        self.low_dates = pd.to_datetime(self.df_low["date"])
        self.high_datetime = pd.to_datetime(self.df_high["datetime"])
        self.high_dates = self.high_datetime.dt.normalize()

        self._validate_lengths()
        self._align_indices()
        self.start_day = self._resolve_start_day()

        self.day = 0
        self.terminal = False
        self.cash = self.initial_amount
        self.shares = 0
        self.asset_value = self.initial_amount
        self.reward = 0.0
        self.last_action = -1.0

    @staticmethod
    def _prepare_frame(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        if date_col not in df.columns:
            raise ValueError(f"Missing required date column: {date_col}")
        prepared = df.copy()
        prepared[date_col] = pd.to_datetime(prepared[date_col], errors="coerce")
        if prepared[date_col].isna().any():
            raise ValueError(f"{date_col} contains invalid datetime values.")
        return prepared.sort_values(date_col).reset_index(drop=True)

    @staticmethod
    def _available_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
        columns = [column for column in candidates if column in df.columns]
        if "close" not in columns:
            raise ValueError("The environment requires a close column.")
        return columns

    def _validate_lengths(self) -> None:
        minimum_mid_rows = self.mid_len + self.decision_lag + 1
        if len(self.df_mid) < minimum_mid_rows:
            raise ValueError(
                "df_mid is too short for the requested mid_len and decision_lag. "
                f"Need at least {minimum_mid_rows} rows."
            )
        if len(self.df_low) == 0 or len(self.df_high) == 0:
            raise ValueError("df_low and df_high must not be empty.")

    @property
    def execution_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.mid_dates.iloc[self.day])

    @property
    def observation_mid_index(self) -> int:
        return max(self.day - self.decision_lag, 0)

    @property
    def observation_end_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.mid_dates.iloc[self.observation_mid_index])

    def _align_indices(self) -> None:
        self.aligned_low_idx = np.zeros(len(self.mid_dates), dtype=int)
        self.aligned_high_idx = np.zeros(len(self.mid_dates), dtype=int)

        for i in range(len(self.mid_dates)):
            observation_date = self.mid_dates.iloc[max(i - self.decision_lag, 0)]

            valid_low = np.where(self.low_dates.to_numpy() <= observation_date.to_datetime64())[0]
            self.aligned_low_idx[i] = valid_low[-1] if len(valid_low) > 0 else 0

            valid_high = np.where(self.high_dates.to_numpy() <= observation_date.normalize().to_datetime64())[0]
            self.aligned_high_idx[i] = valid_high[-1] if len(valid_high) > 0 else 0

    def _resolve_start_day(self) -> int:
        minimum_day = self.mid_len - 1 + self.decision_lag
        if self.trade_start_date is None:
            start_day = minimum_day
        else:
            start_day = int(np.searchsorted(self.mid_dates.to_numpy(), self.trade_start_date.to_datetime64()))
            start_day = max(start_day, minimum_day)
        if start_day >= len(self.df_mid) - 1:
            raise ValueError("The requested trade_start_date leaves no executable transition.")
        return start_day

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = self.start_day
        self.terminal = False
        self.cash = self.initial_amount
        self.shares = 0
        self.asset_value = self.initial_amount
        self.reward = 0.0
        self.last_action = -1.0
        return self._get_state(), {}

    @staticmethod
    def _window_with_padding(data: np.ndarray, end_idx: int, window_len: int) -> np.ndarray:
        end_idx = int(max(end_idx, 0))
        start_idx = end_idx - window_len + 1
        if start_idx >= 0:
            return data[start_idx : end_idx + 1]

        pad_count = -start_idx
        pad = np.tile(data[0], (pad_count, 1))
        return np.vstack([pad, data[0 : end_idx + 1]])

    def _get_state(self) -> np.ndarray:
        mid_idx = self.observation_mid_index
        low_idx = self.aligned_low_idx[self.day]
        high_idx = self.aligned_high_idx[self.day]

        low_state = self._window_with_padding(self.np_low, low_idx, self.low_len)
        mid_state = self._window_with_padding(self.np_mid, mid_idx, self.mid_len)
        high_state = self._window_with_padding(self.np_high, high_idx, self.high_len)
        portfolio_state = self._portfolio_state()

        return np.concatenate(
            [
                np.nan_to_num(low_state).flatten(),
                np.nan_to_num(mid_state).flatten(),
                np.nan_to_num(high_state).flatten(),
                portfolio_state,
            ]
        ).astype(np.float32)

    def _portfolio_state(self, price: Optional[float] = None) -> np.ndarray:
        if price is None:
            price = float(self.execution_prices[self.day])
        total_asset = self.cash + self.shares * price
        if total_asset <= 0:
            return np.array([0.0, 0.0, self.last_action], dtype=np.float32)
        return np.array(
            [
                self.shares * price / total_asset,
                self.cash / total_asset,
                self.last_action,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        if self.terminal:
            raise RuntimeError("step() called after the episode terminated. Call reset() first.")

        current_price = float(self.execution_prices[self.day])

        clipped_action = float(np.clip(float(action[0]), -1.0, 1.0))
        target_weight = (clipped_action + 1.0) / 2.0
        total_asset = self.cash + self.shares * current_price
        current_weight = self.shares * current_price / total_asset if total_asset > 0 else 0.0
        rebalance_skipped = abs(target_weight - current_weight) < self.rebalance_threshold
        if rebalance_skipped:
            target_shares = self.shares
        else:
            target_shares = int((target_weight * total_asset) / current_price)
            target_shares = (target_shares // 100) * 100
        delta_shares = target_shares - self.shares
        transaction_cost = 0.0

        if delta_shares > 0:
            cost = delta_shares * current_price * (1 + self.transaction_fee_percent)
            if self.cash >= cost:
                self.cash -= cost
                self.shares += delta_shares
            else:
                max_buy = int(self.cash / (current_price * (1 + self.transaction_fee_percent)))
                max_buy = (max_buy // 100) * 100
                self.cash -= max_buy * current_price * (1 + self.transaction_fee_percent)
                self.shares += max_buy
                delta_shares = max_buy
        elif delta_shares < 0:
            sell_shares = min(-delta_shares, self.shares)
            sell_shares = (sell_shares // 100) * 100
            self.cash += sell_shares * current_price * (1 - self.transaction_fee_percent)
            self.shares -= sell_shares
            delta_shares = -sell_shares

        transaction_cost = abs(delta_shares) * current_price * self.transaction_fee_percent
        turnover = abs(delta_shares) * current_price / total_asset if total_asset > 0 else 0.0
        self.last_action = clipped_action

        next_price = float(self.execution_prices[self.day + 1])
        new_asset = self.cash + self.shares * next_price
        raw_reward = float(np.log(new_asset / self.asset_value)) if self.asset_value > 0 and new_asset > 0 else 0.0
        turnover_penalty = self.turnover_penalty_coef * turnover
        self.reward = float(np.clip(raw_reward - turnover_penalty, -REWARD_CLIP, REWARD_CLIP))
        self.asset_value = new_asset

        info = self._info(
            execution_price=current_price,
            next_price=next_price,
            transaction_cost=transaction_cost,
            turnover=turnover,
            raw_reward=raw_reward,
            turnover_penalty=turnover_penalty,
            rebalance_skipped=rebalance_skipped,
        )
        self.day += 1
        self.terminal = self.day >= len(self.df_mid) - 1
        return self._get_state(), self.reward, self.terminal, False, info

    def _info(
        self,
        execution_price: Optional[float] = None,
        next_price: Optional[float] = None,
        transaction_cost: float = 0.0,
        turnover: float = 0.0,
        raw_reward: float = 0.0,
        turnover_penalty: float = 0.0,
        rebalance_skipped: bool = False,
    ) -> dict:
        next_idx = min(self.day + 1, len(self.df_mid) - 1)
        portfolio_state = self._portfolio_state(execution_price)
        return {
            "asset_value": self.asset_value,
            "cash": self.cash,
            "shares": self.shares,
            "execution_date": pd.Timestamp(self.mid_dates.iloc[self.day]),
            "observation_end_date": self.observation_end_date,
            "next_date": pd.Timestamp(self.mid_dates.iloc[next_idx]),
            "execution_price": execution_price,
            "next_price": next_price,
            "position_weight": float(portfolio_state[0]),
            "cash_ratio": float(portfolio_state[1]),
            "last_action": self.last_action,
            "transaction_cost": transaction_cost,
            "turnover": turnover,
            "raw_reward": raw_reward,
            "turnover_penalty": turnover_penalty,
            "rebalance_skipped": rebalance_skipped,
        }
