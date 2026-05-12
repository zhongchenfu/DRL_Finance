"""
R-PPO multi-frequency trading environment.

Timing convention:
- At execution day t, the observation contains information available up to
  day t - decision_lag.
- The action is executed at the close price of day t.
- The reward is realized from the close of day t to the close of day t + 1.

With the default decision_lag=1, the agent cannot observe same-day daily or
5-minute data before deciding the position for day t.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


REWARD_CLIP = 0.05
BASE_FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


class MultiFreqStockEnv(gym.Env):
    """
    Multi-frequency individual-stock trading environment for R-PPO.

    Observation:
        Flattened concatenation of low-frequency, mid-frequency, and
        high-frequency rolling windows.

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
        high_len: int = 240,
        tech_indicator_list: Optional[list[str]] = None,
        decision_lag: int = 1,
    ):
        super().__init__()
        if decision_lag < 0:
            raise ValueError("decision_lag must be non-negative.")

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

        self.low_cols = self._available_columns(self.df_low, BASE_FEATURE_COLUMNS)
        self.mid_cols = self._available_columns(self.df_mid, BASE_FEATURE_COLUMNS + self.tech_indicator_list)
        self.high_cols = self._available_columns(self.df_high, BASE_FEATURE_COLUMNS)

        self.low_features = len(self.low_cols)
        self.mid_features = len(self.mid_cols)
        self.high_features = len(self.high_cols)
        self.close_col_mid = self.mid_cols.index("close")

        self.state_dim = (
            self.low_len * self.low_features
            + self.mid_len * self.mid_features
            + self.high_len * self.high_features
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        self.np_low = self.df_low[self.low_cols].to_numpy(dtype=np.float64)
        self.np_mid = self.df_mid[self.mid_cols].to_numpy(dtype=np.float64)
        self.np_high = self.df_high[self.high_cols].to_numpy(dtype=np.float64)

        self.mid_dates = pd.to_datetime(self.df_mid["date"])
        self.low_dates = pd.to_datetime(self.df_low["date"])
        self.high_datetime = pd.to_datetime(self.df_high["datetime"])
        self.high_dates = self.high_datetime.dt.normalize()

        self._validate_lengths()
        self._align_indices()

        self.day = 0
        self.terminal = False
        self.cash = self.initial_amount
        self.shares = 0
        self.asset_value = self.initial_amount
        self.reward = 0.0

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = self.mid_len - 1 + self.decision_lag
        self.terminal = False
        self.cash = self.initial_amount
        self.shares = 0
        self.asset_value = self.initial_amount
        self.reward = 0.0
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

        return np.concatenate(
            [
                np.nan_to_num(low_state).flatten(),
                np.nan_to_num(mid_state).flatten(),
                np.nan_to_num(high_state).flatten(),
            ]
        ).astype(np.float32)

    def step(self, action):
        self.terminal = self.day >= len(self.df_mid) - 2
        if self.terminal:
            return self._get_state(), self.reward, self.terminal, False, self._info()

        current_price = float(self.np_mid[self.day, self.close_col_mid])
        if current_price <= 0:
            self.day += 1
            return self._get_state(), 0.0, self.terminal, False, self._info()

        target_weight = np.clip((float(action[0]) + 1.0) / 2.0, 0.0, 1.0)
        total_asset = self.cash + self.shares * current_price
        target_shares = int((target_weight * total_asset) / current_price)
        target_shares = (target_shares // 100) * 100
        delta_shares = target_shares - self.shares

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
        elif delta_shares < 0:
            sell_shares = min(-delta_shares, self.shares)
            sell_shares = (sell_shares // 100) * 100
            self.cash += sell_shares * current_price * (1 - self.transaction_fee_percent)
            self.shares -= sell_shares

        next_price = float(self.np_mid[self.day + 1, self.close_col_mid])
        new_asset = self.cash + self.shares * next_price
        raw_reward = float(np.log(new_asset / self.asset_value)) if self.asset_value > 0 and new_asset > 0 else 0.0
        self.reward = float(np.clip(raw_reward, -REWARD_CLIP, REWARD_CLIP))
        self.asset_value = new_asset

        info = self._info(execution_price=current_price, next_price=next_price)
        self.day += 1
        return self._get_state(), self.reward, self.terminal, False, info

    def _info(self, execution_price: Optional[float] = None, next_price: Optional[float] = None) -> dict:
        next_idx = min(self.day + 1, len(self.df_mid) - 1)
        return {
            "asset_value": self.asset_value,
            "cash": self.cash,
            "shares": self.shares,
            "execution_date": pd.Timestamp(self.mid_dates.iloc[self.day]),
            "observation_end_date": self.observation_end_date,
            "next_date": pd.Timestamp(self.mid_dates.iloc[next_idx]),
            "execution_price": execution_price,
            "next_price": next_price,
        }
