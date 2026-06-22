"""
Final thesis experiment for the multi-frequency PPO trading model.

This standalone experiment file combines the original training/backtesting
utilities with the final Transaction-Cost-Aware R-PPO experiment used in the thesis. The
model is reported as R-PPO for continuity, but its technical implementation is PPO with a
multi-frequency LSTM-based feature extractor. PPO uses standard rollout storage;
each stored observation is already a truncated multi-frequency sequence state
produced by MultiFreqStockEnv.
"""

import copy
import os
import random
from pathlib import Path

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from r_ppo_data import load_experiment_data_with_validation
from r_ppo_env import (
    DEFAULT_REBALANCE_THRESHOLD,
    DEFAULT_TURNOVER_PENALTY_COEF,
    PORTFOLIO_FEATURES,
    MultiFreqStockEnv,
    normalize_feature_frame,
)
from r_ppo_network import MultiFreqFeaturesExtractor
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus-sign rendering.

warnings.filterwarnings("ignore")

# ================================
# SB3 / PyTorch Zip Load Bug Fix
# ================================
import zipfile
import io
import torch

_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    if isinstance(f, zipfile.ZipExtFile):
        f = io.BytesIO(f.read())
    kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load
# ================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Train, validation, and test split.
TRAIN_START = "2020-03-01"
TRAIN_END = "2023-06-30"
VALIDATION_START = "2023-07-01"
VALIDATION_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"

STOCKS = {
    "sh.600519": "Kweichow Moutai",
    "sh.601628": "China Life",
    "sh.601688": "Huatai Securities",
}
# ================================
# Configuration
# ================================
CONFIG = {
    "TRAIN_MODELS": False,  # The selected R-PPO checkpoints are loaded, not retrained.
    "TRAIN_BASELINES_IF_MISSING": True,
    "MODELS_TO_RUN": ["R-PPO", "PPO (Daily)", "A2C (Daily)", "GARCH", "Buy&Hold"],
    "R_PPO_SEEDS": [0, 1, 2],
    "BASELINE_SEEDS": [0, 1, 2],
    # The main experiment uses 1,000,000 by default. For capital-sensitivity experiments, use:
    # [1_000_000, 5_000_000, 10_000_000]
    "INITIAL_AMOUNTS": [1_000_000],
    "TOTAL_TIMESTEPS": 50_000,
    "RANDOM_SEED": 42,
}
REBALANCE_THRESHOLD = DEFAULT_REBALANCE_THRESHOLD
TURNOVER_PENALTY_COEF = DEFAULT_TURNOVER_PENALTY_COEF
R_PPO_VARIANTS = {
    "R-PPO-Full": True,
    "R-PPO-NoHigh": False,
}


class ValidationEarlyStoppingCallback(BaseCallback):
    """Keep the best validation policy and stop after repeated non-improvement."""

    def __init__(self, eval_env, eval_freq=2048, patience=5, min_evals=3, min_delta=1e-6, verbose=1):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.patience = int(patience)
        self.min_evals = int(min_evals)
        self.min_delta = float(min_delta)
        self.evaluations = 0
        self.no_improvement_evals = 0
        self.best_validation_sharpe = -np.inf
        self.best_policy_state = None

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True

        validation_sharpe = evaluate_validation_sharpe(self.model, self.eval_env)
        self.evaluations += 1
        self.logger.record("validation/sharpe_ratio", validation_sharpe)

        if validation_sharpe > self.best_validation_sharpe + self.min_delta:
            self.best_validation_sharpe = validation_sharpe
            self.best_policy_state = copy.deepcopy(self.model.policy.state_dict())
            self.no_improvement_evals = 0
        else:
            self.no_improvement_evals += 1

        should_stop = self.evaluations >= self.min_evals and self.no_improvement_evals >= self.patience
        if should_stop and self.verbose:
            print(
                "  [Validation] Early stopping after "
                f"{self.evaluations} evaluations; best Sharpe={self.best_validation_sharpe:.6f}"
            )
        return not should_stop

    def restore_best_policy(self):
        if self.best_policy_state is not None:
            self.model.policy.load_state_dict(self.best_policy_state)


def evaluate_validation_sharpe(model, env_validation):
    """Evaluate a deterministic policy with account-value returns."""
    obs, _ = env_validation.reset()
    done = False
    asset_history = [float(env_validation.initial_amount)]
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env_validation.step(action)
        asset_history.append(float(info["asset_value"]))

    returns = pd.Series(asset_history, dtype=np.float64).pct_change().dropna()
    volatility = returns.std(ddof=1)
    if not np.isfinite(volatility) or volatility <= 0:
        return 0.0
    return float(np.sqrt(252) * returns.mean() / volatility)


def learn_with_validation(model, env_validation, total_timesteps=None):
    if total_timesteps is None:
        total_timesteps = CONFIG["TOTAL_TIMESTEPS"]
    callback = None
    if env_validation is not None:
        callback = ValidationEarlyStoppingCallback(env_validation)
    model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=callback)
    if callback is not None:
        callback.restore_best_policy()
    return model


def load_data(code):
    return load_experiment_data_with_validation(
        code,
        data_dir=DATA_DIR,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        validation_start=VALIDATION_START,
        validation_end=VALIDATION_END,
        test_start=TEST_START,
        test_end=TEST_END,
    )

# ================================
# Baseline Environment (PPO/A2C)
# ================================
class SimpleBaselineEnv(gym.Env):
    """
    Daily-frequency baseline environment.

    It uses the same timing convention as MultiFreqStockEnv: observations end at
    day t - decision_lag, actions execute at the open of day t, and rewards are
    realized from the open of day t to the open of day t + 1.
    """
    def __init__(
        self,
        df_mid,
        initial_amount=1000000,
        transaction_fee_percent=0.003,
        window_size=15,
        tech_indicator_list=None,
        decision_lag=1,
        normalization_stats=None,
        trade_start_date=None,
        rebalance_threshold=DEFAULT_REBALANCE_THRESHOLD,
        turnover_penalty_coef=DEFAULT_TURNOVER_PENALTY_COEF,
    ):
        super().__init__()
        if decision_lag < 0:
            raise ValueError("decision_lag must be non-negative.")
        if not 0.0 <= rebalance_threshold <= 1.0:
            raise ValueError("rebalance_threshold must be between 0 and 1.")
        if turnover_penalty_coef < 0.0:
            raise ValueError("turnover_penalty_coef must be non-negative.")

        self.df = df_mid.copy()
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        if self.df["date"].isna().any():
            raise ValueError("date contains invalid datetime values.")
        self.df = self.df.sort_values("date").reset_index(drop=True)

        self.initial_amount = initial_amount
        self.fee = transaction_fee_percent
        self.window_size = int(window_size)
        self.decision_lag = int(decision_lag)
        self.trade_start_date = pd.Timestamp(trade_start_date) if trade_start_date is not None else None
        self.rebalance_threshold = float(rebalance_threshold)
        self.turnover_penalty_coef = float(turnover_penalty_coef)
        
        self.cols = ["open", "high", "low", "close", "volume", "amount"] + list(tech_indicator_list or [])
        self.cols = [c for c in self.cols if c in self.df.columns]
        if "close" not in self.cols:
            raise ValueError("The baseline environment requires a close column.")
        if "open" not in self.cols:
            raise ValueError("The baseline environment requires an open column for trade execution.")
        self.np_data, self.normalization_stats = normalize_feature_frame(
            self.df,
            self.cols,
            normalization_stats,
        )
        self.execution_prices = pd.to_numeric(self.df["open"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(self.execution_prices).all() or (self.execution_prices <= 0).any():
            raise ValueError("The baseline environment requires finite positive open prices.")
        
        self.state_dim = self.window_size * len(self.cols) + PORTFOLIO_FEATURES
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        self.day = 0
        self.terminal = False
        self.reward = 0.0
        self.start_day = self._resolve_start_day()
        self.last_action = -1.0

        minimum_rows = self.window_size + self.decision_lag + 1
        if len(self.df) < minimum_rows:
            raise ValueError(
                "df_mid is too short for the requested window_size and decision_lag. "
                f"Need at least {minimum_rows} rows."
            )

    @property
    def observation_index(self):
        return max(self.day - self.decision_lag, 0)

    @property
    def execution_date(self):
        return pd.Timestamp(self.df["date"].iloc[self.day])

    @property
    def observation_end_date(self):
        return pd.Timestamp(self.df["date"].iloc[self.observation_index])
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.day = self.start_day
        self.terminal = False
        self.cash = self.initial_amount
        self.shares = 0
        self.asset_value = self.initial_amount
        self.reward = 0.0
        self.last_action = -1.0
        return self._get_state(), {}

    def _resolve_start_day(self):
        minimum_day = self.window_size - 1 + self.decision_lag
        if self.trade_start_date is None:
            start_day = minimum_day
        else:
            start_day = int(np.searchsorted(self.df["date"].to_numpy(), self.trade_start_date.to_datetime64()))
            start_day = max(start_day, minimum_day)
        if start_day >= len(self.df) - 1:
            raise ValueError("The requested trade_start_date leaves no executable transition.")
        return start_day
        
    def _get_state(self):
        end = self.observation_index
        start = end - self.window_size + 1
        state = self.np_data[start : end + 1]
        state = np.concatenate([np.nan_to_num(state).flatten(), self._portfolio_state()])
        return state.astype(np.float32)

    def _portfolio_state(self, price=None):
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

        current_price = self.execution_prices[self.day]

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
        
        if delta_shares > 0:
            cost = delta_shares * current_price * (1 + self.fee)
            if self.cash >= cost:
                self.cash -= cost
                self.shares += delta_shares
            else:
                max_buy = int(self.cash / (current_price * (1 + self.fee)))
                max_buy = (max_buy // 100) * 100
                self.cash -= max_buy * current_price * (1 + self.fee)
                self.shares += max_buy
                delta_shares = max_buy
        elif delta_shares < 0:
            sell_shares = min(-delta_shares, self.shares)
            sell_shares = (sell_shares // 100) * 100
            self.cash += sell_shares * current_price * (1 - self.fee)
            self.shares -= sell_shares
            delta_shares = -sell_shares

        transaction_cost = abs(delta_shares) * current_price * self.fee
        turnover = abs(delta_shares) * current_price / total_asset if total_asset > 0 else 0.0
        self.last_action = clipped_action
            
        next_price = self.execution_prices[self.day + 1]
        new_asset = self.cash + self.shares * next_price
        
        raw_reward = float(np.log(new_asset / self.asset_value)) if self.asset_value > 0 and new_asset > 0 else 0.0
        turnover_penalty = self.turnover_penalty_coef * turnover
        self.reward = float(np.clip(raw_reward - turnover_penalty, -0.05, 0.05))
            
        self.asset_value = new_asset
        info = self._info(
            execution_price=current_price,
            next_price=next_price,
            transaction_cost=transaction_cost,
            turnover=turnover,
            raw_reward=raw_reward,
            turnover_penalty=turnover_penalty,
            rebalance_skipped=rebalance_skipped,
            target_weight=target_weight,
        )
        self.day += 1
        self.terminal = self.day >= len(self.df) - 1
        return self._get_state(), self.reward, self.terminal, False, info

    def _info(
        self,
        execution_price=None,
        next_price=None,
        transaction_cost=0.0,
        turnover=0.0,
        raw_reward=0.0,
        turnover_penalty=0.0,
        rebalance_skipped=False,
        target_weight=None,
    ):
        next_idx = min(self.day + 1, len(self.df) - 1)
        portfolio_state = self._portfolio_state(execution_price)
        return {
            "asset_value": self.asset_value,
            "execution_date": self.execution_date,
            "observation_end_date": self.observation_end_date,
            "next_date": pd.Timestamp(self.df["date"].iloc[next_idx]),
            "execution_price": execution_price,
            "next_price": next_price,
            "position_weight": float(portfolio_state[0]),
            "target_weight": float(target_weight) if target_weight is not None else None,
            "cash_ratio": float(portfolio_state[1]),
            "last_action": self.last_action,
            "transaction_cost": transaction_cost,
            "turnover": turnover,
            "raw_reward": raw_reward,
            "turnover_penalty": turnover_penalty,
            "rebalance_skipped": rebalance_skipped,
        }

def train_r_ppo(env_train, env_validation=None, model_label="R-PPO", use_high_frequency=True):
    print(f"  [{model_label}] Training...")
    
    # The environment observation already contains truncated low/mid/high
    # frequency windows. Standard PPO keeps random minibatch sampling while the
    # custom feature extractor learns recurrent representations from each window.
    policy_kwargs = dict(
        features_extractor_class=MultiFreqFeaturesExtractor,
        features_extractor_kwargs=dict(
            low_len=env_train.low_len, low_features=env_train.low_features, 
            mid_len=env_train.mid_len, mid_features=env_train.mid_features,
            high_len=env_train.high_len, high_features=env_train.high_features,
            hidden_size=32, portfolio_features=PORTFOLIO_FEATURES,
            use_high_frequency=use_high_frequency,
        ),
        net_arch=dict(pi=[32, 32], vf=[32, 32])
    )
    
    model = PPO("MlpPolicy", env_train, policy_kwargs=policy_kwargs, 
                learning_rate=0.0003, 
                n_steps=512,
                batch_size=64,
                gamma=0.99, seed=CONFIG["RANDOM_SEED"], verbose=0)
    
    return learn_with_validation(model, env_validation)

def train_baseline_ppo(env_train, env_validation=None):
    print("  [PPO Base] Training...")
    model = PPO("MlpPolicy", env_train, learning_rate=0.0003, n_steps=512,
                batch_size=64, gamma=0.99, seed=CONFIG["RANDOM_SEED"], verbose=0)
    return learn_with_validation(model, env_validation)

def train_baseline_a2c(env_train, env_validation=None):
    print("  [A2C Base] Training...")
    # model = A2C("MlpPolicy", env_train, n_steps=1000, learning_rate=0.0003, verbose=0)
    # Tune the A2C parameters used by the training function.
    model = A2C("MlpPolicy", env_train, 
                n_steps=100,            # Increase the rollout length; the default of 5 is too short.
                learning_rate=0.0003, 
                seed=CONFIG["RANDOM_SEED"],
                verbose=0)
    return learn_with_validation(model, env_validation)

def evaluate_account_value(account_value):
    account_value = account_value.copy()
    if account_value.empty:
        return {}, account_value

    account_value['daily_return'] = account_value['account_value'].pct_change()

    returns = account_value.set_index('date')['daily_return'].dropna()
    returns.index = pd.to_datetime(returns.index)

    if returns.empty:
        return {
            "Annual return": np.nan,
            "Cumulative return": np.nan,
            "Annual volatility": np.nan,
            "Sharpe ratio": np.nan,
            "Calmar ratio": np.nan,
            "Max drawdown": np.nan,
            "Sortino ratio": np.nan,
        }, account_value

    cum_ret = account_value['account_value'].iloc[-1] / account_value['account_value'].iloc[0] - 1
    periods = max(len(returns), 1)
    ann_return = (1 + cum_ret) ** (252 / periods) - 1 if cum_ret > -1 else -1.0
    ann_vol = returns.std(ddof=1) * np.sqrt(252)
    sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252) if returns.std(ddof=1) > 0 else np.nan

    equity_curve = (1 + returns).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_dd = float(drawdown.min()) if not drawdown.empty else np.nan
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    downside_returns = returns[returns < 0]
    down_std = downside_returns.std(ddof=1) * np.sqrt(252)
    sortino = ann_return / down_std if down_std > 0 else np.nan

    return {
        "Annual return": ann_return,
        "Cumulative return": cum_ret,
        "Annual volatility": ann_vol,
        "Sharpe ratio": sharpe,
        "Calmar ratio": calmar,
        "Max drawdown": max_dd,
        "Sortino ratio": sortino
    }, account_value


def backtest(model, env_test, df_test_mid, return_actions=False):
    obs, _ = env_test.reset()
    done = False
    initial_date = getattr(env_test, "execution_date", None)
    if initial_date is None:
        asset_history = []
        date_history = []
    else:
        asset_history = [float(getattr(env_test, "initial_amount", 1000000))]
        date_history = [pd.Timestamp(initial_date)]
    action_records = []
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env_test.step(action)
        asset_history.append(info.get('asset_value', 1000000))
        if "next_date" in info:
            date_history.append(pd.Timestamp(info["next_date"]))
        elif "execution_date" in info:
            date_history.append(pd.Timestamp(info["execution_date"]))
        action_value = float(np.asarray(action).reshape(-1)[0])
        action_records.append(
            {
                "date": pd.Timestamp(info.get("execution_date", info.get("next_date"))),
                "next_date": pd.Timestamp(info["next_date"]) if "next_date" in info else pd.NaT,
                "action": action_value,
                "target_weight": info.get("target_weight", (action_value + 1.0) / 2.0),
                "position_weight": info.get("position_weight", np.nan),
                "cash_ratio": info.get("cash_ratio", np.nan),
                "turnover": info.get("turnover", 0.0),
                "transaction_cost": info.get("transaction_cost", 0.0),
                "asset_value": info.get("asset_value", np.nan),
                "raw_reward": info.get("raw_reward", np.nan),
                "turnover_penalty": info.get("turnover_penalty", np.nan),
                "drawdown_penalty": info.get("drawdown_penalty", np.nan),
                "current_drawdown": info.get("current_drawdown", np.nan),
                "rebalance_skipped": info.get("rebalance_skipped", False),
                "reward": reward,
            }
        )
        
    if len(date_history) == len(asset_history):
        eval_dates = date_history
    else:
        start_idx = (
            getattr(env_test, 'mid_len', getattr(env_test, 'window_size', 15))
            - 1
            + getattr(env_test, 'decision_lag', 0)
        )
        eval_dates = df_test_mid['date'].values[start_idx : start_idx + len(asset_history)]
    account_value = pd.DataFrame({'date': eval_dates, 'account_value': asset_history})
    metrics, account_value = evaluate_account_value(account_value)
    if return_actions:
        return metrics, account_value, pd.DataFrame(action_records)
    return metrics, account_value


def summarize_backtest(model_name, metrics, account_value, action_history):
    if account_value.empty:
        final_nav = np.nan
        total_return = np.nan
    else:
        final_nav = account_value["account_value"].iloc[-1] / account_value["account_value"].iloc[0]
        total_return = final_nav - 1.0

    if action_history.empty:
        total_turnover = np.nan
        average_turnover = np.nan
        trade_count = 0
    else:
        turnover = pd.to_numeric(action_history["turnover"], errors="coerce").fillna(0.0)
        total_turnover = float(turnover.sum())
        average_turnover = float(turnover.mean())
        trade_count = int((turnover > 1e-8).sum())

    return {
        "model": model_name,
        "final_nav": final_nav,
        "total_return": total_return,
        "sharpe": metrics.get("Sharpe ratio", np.nan),
        "max_drawdown": metrics.get("Max drawdown", np.nan),
        "total_turnover": total_turnover,
        "average_turnover": average_turnover,
        "trade_count": trade_count,
    }


def plot_action_history(action_histories, code, capital_tag):
    if not action_histories:
        return None

    fig, axes = plt.subplots(len(action_histories), 1, figsize=(12, 3.6 * len(action_histories)), sharex=True)
    if len(action_histories) == 1:
        axes = [axes]

    for ax, (model_name, history) in zip(axes, action_histories.items()):
        if history.empty:
            continue
        dates = pd.to_datetime(history["date"])
        ax.plot(dates, history["target_weight"], label="Target weight", color="#1F77B4", linewidth=1.4)
        ax.plot(dates, history["position_weight"], label="Position weight", color="#D62728", linewidth=1.4, alpha=0.85)
        ax.bar(dates, history["turnover"], label="Turnover", color="#7F7F7F", alpha=0.25)
        ax.set_title(model_name, fontsize=13, fontweight="bold")
        ax.set_ylabel("Weight / Turnover")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="upper left", fontsize=9)

    axes[-1].set_xlabel("Date")
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    img_path = os.path.join(RESULTS_DIR, f"Action_History_{code}_{capital_tag}.png")
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return img_path


def backtest_buy_and_hold(
    df_mid,
    initial_amount=1000000,
    fee=0.003,
    window_size=15,
    decision_lag=1,
    trade_start_date=None,
):
    df_mid = df_mid.reset_index(drop=True)
    minimum_start_idx = window_size - 1 + decision_lag
    if trade_start_date is None:
        start_idx = minimum_start_idx
    else:
        dates = pd.to_datetime(df_mid["date"])
        start_idx = int(np.searchsorted(dates.to_numpy(), pd.Timestamp(trade_start_date).to_datetime64()))
        start_idx = max(start_idx, minimum_start_idx)
    if len(df_mid) <= start_idx:
        return {}, pd.DataFrame()

    prices = pd.to_numeric(df_mid["open"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Buy&Hold requires finite positive open prices.")

    initial_price = prices[start_idx]
    shares = int(initial_amount / (initial_price * (1 + fee)))
    shares = (shares // 100) * 100
    cash = initial_amount - shares * initial_price * (1 + fee)

    dates = [pd.Timestamp(df_mid["date"].iloc[start_idx])]
    assets = [float(initial_amount)]
    for idx in range(start_idx + 1, len(df_mid)):
        dates.append(pd.Timestamp(df_mid["date"].iloc[idx]))
        assets.append(float(cash + shares * prices[idx]))

    account_value = pd.DataFrame({"date": dates, "account_value": assets})
    return evaluate_account_value(account_value)


def backtest_garch(train_data, test_data, initial_amount=1000000, fee=0.003, window_size=15, decision_lag=1):
    try:
        from arch import arch_model
    except ImportError:
        print("Please install arch package: pip install arch")
        return {}, pd.DataFrame()
        
    print("  [GARCH Base] Evaluating...")
    df_train = (train_data.mid if hasattr(train_data, "mid") else train_data[1]).reset_index(drop=True)
    df_test = (test_data.mid if hasattr(test_data, "mid") else test_data[1]).reset_index(drop=True)
    trade_start_date = getattr(test_data, "trade_start_date", None)
    if trade_start_date is None:
        start_idx = window_size - 1 + decision_lag
    else:
        start_idx = int(np.searchsorted(df_test["date"].to_numpy(), pd.Timestamp(trade_start_date).to_datetime64()))
        start_idx = max(start_idx, window_size - 1 + decision_lag)
    if len(df_test) <= start_idx:
        return {}, pd.DataFrame()

    asset_history = [float(initial_amount)]
    date_history = [pd.Timestamp(df_test['date'].iloc[start_idx])]
    cash = initial_amount
    shares = 0
    asset_value = initial_amount

    for i in range(start_idx, len(df_test) - 1):
        current_price = float(df_test['open'].iloc[i])

        observation_end = i - decision_lag
        if trade_start_date is not None:
            history_close = df_test['close'].iloc[: observation_end + 1]
        elif observation_end >= 0:
            observed_test_close = df_test['close'].iloc[: observation_end + 1]
            history_close = pd.concat([df_train['close'], observed_test_close], ignore_index=True)
        else:
            history_close = df_train['close'].reset_index(drop=True)

        returns_history = history_close.pct_change().dropna() * 100
        pred_next_ret = 0.0
        if len(returns_history) >= max(2, window_size):
            am = arch_model(returns_history, mean='AR', lags=1, vol='Garch', p=1, q=1)
            res = am.fit(disp='off')
            forecasts = res.forecast(horizon=1, reindex=False)
            pred_next_ret = float(forecasts.mean['h.1'].iloc[-1])

        target_weight = 1.0 if pred_next_ret > 0 else 0.0

        target_shares = int((target_weight * asset_value) / (current_price * (1 + fee)))
        target_shares = (target_shares // 100) * 100

        if target_shares != shares:
            diff = target_shares - shares
            if diff > 0:
                cost = diff * current_price * (1 + fee)
                cash -= cost
            else:
                cash += abs(diff) * current_price * (1 - fee)
            shares = target_shares

        next_price = float(df_test['open'].iloc[i + 1])
        asset_value = cash + shares * next_price
        asset_history.append(asset_value)
        date_history.append(pd.Timestamp(df_test['date'].iloc[i + 1]))

    account_value = pd.DataFrame({
        'date': date_history,
        'account_value': asset_history
    })

    return evaluate_account_value(account_value)

def _model_slug(model_name):
    return (
        model_name.lower()
        .replace("&", "and")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _safe_code(code):
    return code.replace(".", "_")


def _set_training_seed(seed):
    CONFIG["RANDOM_SEED"] = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _account_with_nav(account_value):
    account_value = account_value.copy()
    if account_value.empty:
        account_value["nav"] = pd.Series(dtype=float)
        return account_value
    account_value["date"] = pd.to_datetime(account_value["date"])
    account_value["nav"] = account_value["account_value"] / account_value["account_value"].iloc[0]
    return account_value


def _mean_seed_account(seed_accounts):
    merged = None
    for seed, account_value in seed_accounts.items():
        nav = _account_with_nav(account_value)[["date", "nav"]].rename(columns={"nav": f"seed_{seed}"})
        merged = nav if merged is None else merged.merge(nav, on="date", how="inner")
    if merged is None or merged.empty:
        return pd.DataFrame(columns=["date", "account_value", "daily_return", "nav"])

    seed_cols = [column for column in merged.columns if column.startswith("seed_")]
    merged["nav"] = merged[seed_cols].mean(axis=1)
    merged["nav_std"] = merged[seed_cols].std(axis=1)
    merged["nav_lower"] = merged["nav"] - merged["nav_std"]
    merged["nav_upper"] = merged["nav"] + merged["nav_std"]
    merged["account_value"] = merged["nav"] * 1_000_000
    merged["daily_return"] = merged["account_value"].pct_change()
    return merged[["date", "account_value", "daily_return", "nav", "nav_std", "nav_lower", "nav_upper"]]


def _baseline_account(reference_account, initial_amount):
    baseline = reference_account[["date"]].copy()
    baseline["account_value"] = float(initial_amount)
    baseline["daily_return"] = 0.0
    return baseline


def _summarize_seed_rows(rows):
    metric_cols = [
        "final_nav",
        "total_return",
        "sharpe",
        "max_drawdown",
        "total_turnover",
        "average_turnover",
        "trade_count",
    ]
    summary = pd.DataFrame(rows).groupby(["code", "stock", "model"], dropna=False)[metric_cols].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column).strip("_") for column in summary.columns]
    return summary.reset_index()


def _make_rppo_env(data, available_tech, initial_amount, normalization_stats=None):
    return MultiFreqStockEnv(
        data.low,
        data.mid,
        data.high,
        initial_amount=initial_amount,
        tech_indicator_list=available_tech,
        normalization_stats=normalization_stats,
        trade_start_date=data.trade_start_date,
        rebalance_threshold=REBALANCE_THRESHOLD,
        turnover_penalty_coef=TURNOVER_PENALTY_COEF,
        use_high_frequency=True,
    )


def _make_baseline_env(data, available_tech, initial_amount, normalization_stats=None):
    return SimpleBaselineEnv(
        data.mid,
        initial_amount=initial_amount,
        tech_indicator_list=available_tech,
        normalization_stats=normalization_stats,
        trade_start_date=data.trade_start_date,
        rebalance_threshold=REBALANCE_THRESHOLD,
        turnover_penalty_coef=TURNOVER_PENALTY_COEF,
    )


def _load_or_train_baseline(model_label, model_path, env_train, env_validation, seed):
    if model_path.exists():
        print(f"  [{model_label} seed={seed}] Loading model from {model_path}")
        return PPO.load(model_path) if model_label == "PPO (Daily)" else A2C.load(model_path)

    if not CONFIG["TRAIN_BASELINES_IF_MISSING"]:
        raise FileNotFoundError(f"Missing baseline model: {model_path}")

    print(f"  [{model_label} seed={seed}] Missing model, training a new baseline.")
    _set_training_seed(seed)
    model = train_baseline_ppo(env_train, env_validation) if model_label == "PPO (Daily)" else train_baseline_a2c(env_train, env_validation)
    model.save(model_path)
    print(f"  [{model_label} seed={seed}] Model saved to {model_path}")
    return model


def _plot_strategy_comparison(code, stock, capital_tag, account_values, mean_accounts):
    plt.figure(figsize=(12, 7))
    model_colors = {
        "R-PPO": "#D62728",
        "PPO (Daily)": "#1F77B4",
        "A2C (Daily)": "#2CA02C",
        "GARCH": "#9467BD",
        "Buy&Hold": "#E69F00",
    }
    model_styles = {
        "R-PPO": "-",
        "PPO (Daily)": "-",
        "A2C (Daily)": "-",
        "GARCH": "-",
        "Buy&Hold": "-.",
    }

    plt.axhline(y=1.0, color="#6E6E6E", linestyle="--", linewidth=1.5, alpha=0.9, label="Baseline (1.0)")

    band_alpha = {"R-PPO": 0.12, "PPO (Daily)": 0.08, "A2C (Daily)": 0.08}
    for model_name in ["R-PPO", "PPO (Daily)", "A2C (Daily)"]:
        mean_account = mean_accounts.get(model_name)
        if mean_account is None or mean_account.empty or not {"nav_lower", "nav_upper"}.issubset(mean_account.columns):
            continue
        dates = pd.to_datetime(mean_account["date"])
        plt.fill_between(
            dates,
            mean_account["nav_lower"].astype(float),
            mean_account["nav_upper"].astype(float),
            color=model_colors[model_name],
            alpha=band_alpha[model_name],
            linewidth=0,
        )

    for model_name in ["R-PPO", "PPO (Daily)", "A2C (Daily)", "GARCH", "Buy&Hold"]:
        account_value = account_values.get(model_name)
        if account_value is None or account_value.empty:
            continue
        normalized = _account_with_nav(account_value)
        plt.plot(
            normalized["date"],
            normalized["nav"],
            label=model_name,
            color=model_colors.get(model_name),
            linewidth=2.0,
            linestyle=model_styles.get(model_name, "-"),
            alpha=0.92,
        )

    plt.title(f"Strategy Cumulative Return Comparison: {stock} ({code})", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Return (Normalized)", fontsize=14)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=True, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.55)
    sns.despine(left=True, bottom=True)
    plt.gcf().autofmt_xdate()
    from matplotlib.dates import DateFormatter
    plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    plt.tight_layout()

    img_path = os.path.join(RESULTS_DIR, f"Return_Ratio_{code}_{capital_tag}.png")
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close()
    return img_path


def legacy_main():
    print("Starting final strategy comparison with selected StandardBuffer R-PPO models...")
    np.random.seed(CONFIG["RANDOM_SEED"])
    torch.manual_seed(CONFIG["RANDOM_SEED"])
    tech_indicators = ["RSI", "MACD_DIF", "MACD_DEA", "MACD_BAR", "BOLL_MID", "BOLL_UPPER", "BOLL_LOWER", "CCI", "SMA5", "SMA10", "SMA20", "DX"]

    all_diagnostic_rows = []
    for initial_amount, (code, stock) in (
        (amount, stock_item)
        for amount in CONFIG["INITIAL_AMOUNTS"]
        for stock_item in STOCKS.items()
    ):
        capital_tag = f"{initial_amount // 10_000}w"
        print(f"\nEvaluating {stock} ({code}), initial amount={initial_amount:,.0f}")
        train_data, validation_data, test_data = load_data(code)
        available_tech = [c for c in tech_indicators if c in train_data.mid.columns]

        account_values = {}
        mean_accounts = {}
        action_histories = {}
        diagnostic_rows = []

        env_rppo_train = _make_rppo_env(train_data, available_tech, initial_amount)
        rppo_seed_accounts = {}
        rppo_model_dir = Path(RESULTS_DIR) / "buffer_sampling_ablation"

        if "R-PPO" in CONFIG["MODELS_TO_RUN"]:
            for seed in CONFIG["R_PPO_SEEDS"]:
                model_path = rppo_model_dir / f"model_StandardBuffer_{_safe_code(code)}_seed{seed}_{capital_tag}.zip"
                if not model_path.exists():
                    raise FileNotFoundError(f"Missing selected R-PPO model: {model_path}")
                print(f"  [R-PPO seed={seed}] Loading model from {model_path}")
                model = PPO.load(model_path)
                env_rppo_test = _make_rppo_env(test_data, available_tech, initial_amount, env_rppo_train.normalization_stats)
                metrics, account_value, action_history = backtest(
                    model,
                    env_rppo_test,
                    test_data.mid,
                    return_actions=True,
                )
                rppo_seed_accounts[seed] = account_value
                action_histories[f"R-PPO seed={seed}"] = action_history

                account_path = os.path.join(RESULTS_DIR, f"Account_Value_r_ppo_seed{seed}_{code}_{capital_tag}.csv")
                action_path = os.path.join(RESULTS_DIR, f"Action_History_r_ppo_seed{seed}_{code}_{capital_tag}.csv")
                _account_with_nav(account_value).to_csv(account_path, index=False)
                action_history.to_csv(action_path, index=False)
                row = summarize_backtest("R-PPO", metrics, account_value, action_history)
                row.update({"code": code, "stock": stock, "seed": seed})
                diagnostic_rows.append(row)

            rppo_mean_account = _mean_seed_account(rppo_seed_accounts)
            account_values["R-PPO"] = rppo_mean_account
            mean_accounts["R-PPO"] = rppo_mean_account
            _account_with_nav(rppo_mean_account).to_csv(
                os.path.join(RESULTS_DIR, f"Account_Value_r_ppo_{code}_{capital_tag}.csv"),
                index=False,
            )
        else:
            rppo_mean_account = pd.DataFrame()

        env_base_for_timing = _make_baseline_env(train_data, available_tech, initial_amount)
        env_base_test_for_timing = _make_baseline_env(
            test_data,
            available_tech,
            initial_amount,
            env_base_for_timing.normalization_stats,
        )

        if "PPO (Daily)" in CONFIG["MODELS_TO_RUN"]:
            model_name = "PPO (Daily)"
            ppo_seed_accounts = {}
            for seed in CONFIG["BASELINE_SEEDS"]:
                env_base_train = _make_baseline_env(train_data, available_tech, initial_amount)
                env_base_validation = _make_baseline_env(
                    validation_data,
                    available_tech,
                    initial_amount,
                    env_base_train.normalization_stats,
                )
                env_base_test = _make_baseline_env(
                    test_data,
                    available_tech,
                    initial_amount,
                    env_base_train.normalization_stats,
                )
                model_path = Path(RESULTS_DIR) / f"model_ppo_base_{code}_seed{seed}_{capital_tag}.zip"
                model = _load_or_train_baseline(model_name, model_path, env_base_train, env_base_validation, seed)
                metrics, account_value, action_history = backtest(model, env_base_test, test_data.mid, return_actions=True)
                ppo_seed_accounts[seed] = account_value
                action_histories[f"{model_name} seed={seed}"] = action_history
                _account_with_nav(account_value).to_csv(os.path.join(RESULTS_DIR, f"Account_Value_ppo_daily_seed{seed}_{code}_{capital_tag}.csv"), index=False)
                action_history.to_csv(os.path.join(RESULTS_DIR, f"Action_History_ppo_daily_seed{seed}_{code}_{capital_tag}.csv"), index=False)
                row = summarize_backtest(model_name, metrics, account_value, action_history)
                row.update({"code": code, "stock": stock, "seed": seed})
                diagnostic_rows.append(row)

            ppo_mean_account = _mean_seed_account(ppo_seed_accounts)
            account_values[model_name] = ppo_mean_account
            mean_accounts[model_name] = ppo_mean_account
            _account_with_nav(ppo_mean_account).to_csv(os.path.join(RESULTS_DIR, f"Account_Value_ppo_daily_{code}_{capital_tag}.csv"), index=False)

        if "A2C (Daily)" in CONFIG["MODELS_TO_RUN"]:
            model_name = "A2C (Daily)"
            a2c_seed_accounts = {}
            for seed in CONFIG["BASELINE_SEEDS"]:
                env_base_train = _make_baseline_env(train_data, available_tech, initial_amount)
                env_base_validation = _make_baseline_env(
                    validation_data,
                    available_tech,
                    initial_amount,
                    env_base_train.normalization_stats,
                )
                env_base_test = _make_baseline_env(
                    test_data,
                    available_tech,
                    initial_amount,
                    env_base_train.normalization_stats,
                )
                model_path = Path(RESULTS_DIR) / f"model_a2c_base_{code}_seed{seed}_{capital_tag}.zip"
                model = _load_or_train_baseline(model_name, model_path, env_base_train, env_base_validation, seed)
                metrics, account_value, action_history = backtest(model, env_base_test, test_data.mid, return_actions=True)
                a2c_seed_accounts[seed] = account_value
                action_histories[f"{model_name} seed={seed}"] = action_history
                _account_with_nav(account_value).to_csv(os.path.join(RESULTS_DIR, f"Account_Value_a2c_daily_seed{seed}_{code}_{capital_tag}.csv"), index=False)
                action_history.to_csv(os.path.join(RESULTS_DIR, f"Action_History_a2c_daily_seed{seed}_{code}_{capital_tag}.csv"), index=False)
                row = summarize_backtest(model_name, metrics, account_value, action_history)
                row.update({"code": code, "stock": stock, "seed": seed})
                diagnostic_rows.append(row)

            a2c_mean_account = _mean_seed_account(a2c_seed_accounts)
            account_values[model_name] = a2c_mean_account
            mean_accounts[model_name] = a2c_mean_account
            _account_with_nav(a2c_mean_account).to_csv(os.path.join(RESULTS_DIR, f"Account_Value_a2c_daily_{code}_{capital_tag}.csv"), index=False)

        if "GARCH" in CONFIG["MODELS_TO_RUN"]:
            model_name = "GARCH"
            metrics, account_value = backtest_garch(train_data, test_data, initial_amount=initial_amount)
            if not account_value.empty:
                account_values[model_name] = account_value
                _account_with_nav(account_value).to_csv(os.path.join(RESULTS_DIR, f"Account_Value_garch_{code}_{capital_tag}.csv"), index=False)
            row = summarize_backtest(model_name, metrics, account_value, pd.DataFrame())
            row.update({"code": code, "stock": stock, "seed": np.nan})
            diagnostic_rows.append(row)

        if "Buy&Hold" in CONFIG["MODELS_TO_RUN"]:
            model_name = "Buy&Hold"
            metrics, account_value = backtest_buy_and_hold(
                test_data.mid,
                initial_amount=initial_amount,
                window_size=env_base_test_for_timing.window_size,
                decision_lag=env_base_test_for_timing.decision_lag,
                trade_start_date=test_data.trade_start_date,
            )
            account_values[model_name] = account_value
            _account_with_nav(account_value).to_csv(os.path.join(RESULTS_DIR, f"Account_Value_buyandhold_{code}_{capital_tag}.csv"), index=False)
            row = summarize_backtest(model_name, metrics, account_value, pd.DataFrame())
            row.update({"code": code, "stock": stock, "seed": np.nan})
            diagnostic_rows.append(row)

            baseline = _baseline_account(account_value, initial_amount)
            _account_with_nav(baseline).to_csv(os.path.join(RESULTS_DIR, f"Account_Value_baseline_1_0_{code}_{capital_tag}.csv"), index=False)

        diagnostics = pd.DataFrame(diagnostic_rows)
        diagnostics_path = os.path.join(RESULTS_DIR, f"Strategy_Comparison_{code}_{capital_tag}_diagnostics.csv")
        diagnostics.to_csv(diagnostics_path, index=False)
        all_diagnostic_rows.extend(diagnostic_rows)
        print(diagnostics[["model", "seed", "final_nav", "sharpe", "max_drawdown", "total_turnover", "trade_count"]])
        print(f"  [Output] Saved diagnostics to {diagnostics_path}")

        img_path = _plot_strategy_comparison(code, stock, capital_tag, account_values, mean_accounts)
        print(f"  [Output] Saved return-ratio plot to {img_path}")

        action_plot_path = plot_action_history(action_histories, code, capital_tag)
        if action_plot_path is not None:
            print(f"  [Output] Saved action-history plot to {action_plot_path}")

    all_diagnostics = pd.DataFrame(all_diagnostic_rows)
    all_diagnostics_path = os.path.join(RESULTS_DIR, "Strategy_Comparison_All_Diagnostics.csv")
    all_diagnostics.to_csv(all_diagnostics_path, index=False)
    summary = _summarize_seed_rows(all_diagnostic_rows)
    summary_path = os.path.join(RESULTS_DIR, "Strategy_Comparison_Summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n[Output] Saved all diagnostics to {all_diagnostics_path}")
    print(f"[Output] Saved summary to {summary_path}")


# =============================================================================
# Final thesis experiment entry point
# =============================================================================

FINAL_MODEL_NAME = "Transaction-Cost-Aware R-PPO"
FINAL_OUTPUT_DIR = Path(RESULTS_DIR) / "final_transaction_cost_aware_rppo"
FINAL_DRAW_DOWN_MULTI_DIR = Path(RESULTS_DIR) / "rppo_drawdown_multiseed"
FINAL_DRAW_DOWN_SINGLE_DIR = Path(RESULTS_DIR) / "rppo_drawdown_penalty"
FINAL_STABILITY_DIR = Path(RESULTS_DIR) / "rppo_stability_tuning"
FINAL_FIRST_ROUND_DIR = Path(RESULTS_DIR) / "final_stock_first_round"

FINAL_SEEDS = [0, 1, 2]
FINAL_INITIAL_AMOUNT = 1_000_000
FINAL_CAPITAL_TAG = "100w"

FINAL_R_PPO_REBALANCE_THRESHOLD = 0.05
FINAL_R_PPO_TURNOVER_PENALTY = 0.005
FINAL_R_PPO_DRAWDOWN_PENALTY = 0.0

FINAL_BASELINE_REBALANCE_THRESHOLD = REBALANCE_THRESHOLD
FINAL_BASELINE_TURNOVER_PENALTY = TURNOVER_PENALTY_COEF

FINAL_STOCKS = {
    "sh.601628": "China Life",
    "sh.601688": "Huatai Securities",
    "sz.002594": "BYD",
    "sh.600519": "Kweichow Moutai",
    "sz.002415": "Hikvision",
    "sz.000963": "Huadong Medicine",
    "sh.600887": "Yili Group",
    "sh.600585": "Conch Cement",
}

FINAL_TECH_INDICATORS = [
    "RSI",
    "MACD_DIF",
    "MACD_DEA",
    "MACD_BAR",
    "BOLL_MID",
    "BOLL_UPPER",
    "BOLL_LOWER",
    "CCI",
    "SMA5",
    "SMA10",
    "SMA20",
    "DX",
]


def final_set_seed(seed: int) -> None:
    CONFIG["RANDOM_SEED"] = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def final_make_rppo_env(data, available_tech, normalization_stats=None):
    return MultiFreqStockEnv(
        data.low,
        data.mid,
        data.high,
        initial_amount=FINAL_INITIAL_AMOUNT,
        tech_indicator_list=available_tech,
        normalization_stats=normalization_stats,
        trade_start_date=data.trade_start_date,
        rebalance_threshold=FINAL_R_PPO_REBALANCE_THRESHOLD,
        turnover_penalty_coef=FINAL_R_PPO_TURNOVER_PENALTY,
        drawdown_penalty_coef=FINAL_R_PPO_DRAWDOWN_PENALTY,
        use_high_frequency=True,
    )


def final_make_baseline_env(data, available_tech, normalization_stats=None):
    return SimpleBaselineEnv(
        data.mid,
        initial_amount=FINAL_INITIAL_AMOUNT,
        tech_indicator_list=available_tech,
        normalization_stats=normalization_stats,
        trade_start_date=data.trade_start_date,
        rebalance_threshold=FINAL_BASELINE_REBALANCE_THRESHOLD,
        turnover_penalty_coef=FINAL_BASELINE_TURNOVER_PENALTY,
    )


def final_output_model_path(model_key: str, code: str, seed: int) -> Path:
    return FINAL_OUTPUT_DIR / f"model_{model_key}_{_safe_code(code)}_seed{seed}_{FINAL_CAPITAL_TAG}.zip"


def final_reusable_transaction_cost_aware_rppo_paths(code: str, seed: int) -> list[Path]:
    return [
        final_output_model_path("transaction_cost_aware_rppo", code, seed),
    ]


def final_reusable_baseline_paths(model_key: str, code: str, seed: int) -> list[Path]:
    if model_key == "ppo_daily":
        root_name = f"model_ppo_base_{code}_seed{seed}_{FINAL_CAPITAL_TAG}.zip"
    elif model_key == "a2c_daily":
        root_name = f"model_a2c_base_{code}_seed{seed}_{FINAL_CAPITAL_TAG}.zip"
    else:
        raise ValueError(f"Unknown baseline model key: {model_key}")

    return [
        final_output_model_path(model_key, code, seed),
        FINAL_FIRST_ROUND_DIR / f"model_{model_key}_{_safe_code(code)}_seed{seed}_{FINAL_CAPITAL_TAG}.zip",
        Path(RESULTS_DIR) / root_name,
    ]


def final_load_or_train_transaction_cost_aware_rppo(code: str, seed: int, train_data, validation_data, available_tech):
    final_set_seed(seed)
    env_train = final_make_rppo_env(train_data, available_tech)
    env_validation = final_make_rppo_env(validation_data, available_tech, env_train.normalization_stats)
    target_path = final_output_model_path("transaction_cost_aware_rppo", code, seed)

    for path in final_reusable_transaction_cost_aware_rppo_paths(code, seed):
        if path.exists():
            print(f"  [{FINAL_MODEL_NAME} seed={seed}] Loading {path}", flush=True)
            model = PPO.load(path)
            if path != target_path:
                model.save(target_path)
            return model, env_train.normalization_stats

    print(
        f"  [{FINAL_MODEL_NAME} seed={seed}] Training "
        f"turnover={FINAL_R_PPO_TURNOVER_PENALTY}, threshold={FINAL_R_PPO_REBALANCE_THRESHOLD}",
        flush=True,
    )
    model = train_r_ppo(
        env_train,
        env_validation,
        model_label=f"{FINAL_MODEL_NAME} seed={seed}",
        use_high_frequency=True,
    )
    model.save(target_path)
    print(f"  [{FINAL_MODEL_NAME} seed={seed}] Saved {target_path}", flush=True)
    return model, env_train.normalization_stats


def final_load_or_train_baseline(model_key: str, label: str, code: str, seed: int, env_train, env_validation):
    target_path = final_output_model_path(model_key, code, seed)
    for path in final_reusable_baseline_paths(model_key, code, seed):
        if path.exists():
            print(f"  [{label} seed={seed}] Loading {path}", flush=True)
            model = PPO.load(path) if model_key == "ppo_daily" else A2C.load(path)
            if path != target_path:
                model.save(target_path)
            return model

    print(f"  [{label} seed={seed}] Training", flush=True)
    final_set_seed(seed)
    model = train_baseline_ppo(env_train, env_validation) if model_key == "ppo_daily" else train_baseline_a2c(env_train, env_validation)
    model.save(target_path)
    print(f"  [{label} seed={seed}] Saved {target_path}", flush=True)
    return model


def final_save_seed_outputs(
    model_key: str,
    code: str,
    seed: int,
    account_value: pd.DataFrame,
    action_history: pd.DataFrame | None = None,
) -> None:
    _account_with_nav(account_value).to_csv(
        FINAL_OUTPUT_DIR / f"Account_Value_{model_key}_{_safe_code(code)}_seed{seed}_{FINAL_CAPITAL_TAG}.csv",
        index=False,
    )
    if action_history is not None:
        action_history.to_csv(
            FINAL_OUTPUT_DIR / f"Action_History_{model_key}_{_safe_code(code)}_seed{seed}_{FINAL_CAPITAL_TAG}.csv",
            index=False,
        )


def final_save_reference_output(model_key: str, code: str, account_value: pd.DataFrame) -> None:
    _account_with_nav(account_value).to_csv(
        FINAL_OUTPUT_DIR / f"Account_Value_{model_key}_{_safe_code(code)}_{FINAL_CAPITAL_TAG}.csv",
        index=False,
    )


def final_mean_seed_account(accounts_by_seed: dict[int, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for seed, account_value in sorted(accounts_by_seed.items()):
        nav = _account_with_nav(account_value)[["date", "nav"]].rename(columns={"nav": f"seed_{seed}"})
        merged = nav if merged is None else merged.merge(nav, on="date", how="inner")
    if merged is None or merged.empty:
        return pd.DataFrame(columns=["date", "account_value", "daily_return", "nav", "nav_std", "nav_lower", "nav_upper"])

    seed_cols = [column for column in merged.columns if column.startswith("seed_")]
    merged["nav"] = merged[seed_cols].mean(axis=1)
    merged["nav_std"] = merged[seed_cols].std(axis=1).fillna(0.0)
    merged["nav_lower"] = merged["nav"] - merged["nav_std"]
    merged["nav_upper"] = merged["nav"] + merged["nav_std"]
    merged["account_value"] = merged["nav"] * FINAL_INITIAL_AMOUNT
    merged["daily_return"] = merged["account_value"].pct_change()
    return merged[["date", "account_value", "daily_return", "nav", "nav_std", "nav_lower", "nav_upper"]]


def final_summarize_multiseed(seed_rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "final_nav",
        "total_return",
        "sharpe",
        "max_drawdown",
        "total_turnover",
        "average_turnover",
        "trade_count",
    ]
    summary = seed_rows.groupby(["code", "stock", "model"], dropna=False)[metric_cols].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column).strip("_") for column in summary.columns]
    return summary.reset_index()


def final_plot_comparison(code: str, stock: str, account_values: dict[str, pd.DataFrame], mean_accounts: dict[str, pd.DataFrame]) -> Path:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    colors = {
        FINAL_MODEL_NAME: "#D62728",
        "PPO (Daily)": "#1F77B4",
        "A2C (Daily)": "#2CA02C",
        "GARCH": "#9467BD",
        "Buy&Hold": "#E69F00",
    }
    styles = {
        FINAL_MODEL_NAME: "-",
        "PPO (Daily)": "-",
        "A2C (Daily)": "-",
        "GARCH": "-",
        "Buy&Hold": "-.",
    }
    line_labels = {
        FINAL_MODEL_NAME: "R-PPO mean",
        "PPO (Daily)": "PPO mean",
        "A2C (Daily)": "A2C mean",
    }

    plt.figure(figsize=(12, 7))
    plt.axhline(y=1.0, color="#6E6E6E", linestyle="--", linewidth=1.5, alpha=0.9, label="Baseline (1.0)")

    account = mean_accounts.get(FINAL_MODEL_NAME)
    if account is not None and not account.empty:
        dates = pd.to_datetime(account["date"])
        plt.fill_between(
            dates,
            account["nav_lower"].astype(float),
            account["nav_upper"].astype(float),
            color=colors[FINAL_MODEL_NAME],
            alpha=0.12,
            linewidth=0,
        )

    for model_name in [FINAL_MODEL_NAME, "PPO (Daily)", "A2C (Daily)", "GARCH", "Buy&Hold"]:
        account = account_values.get(model_name)
        if account is None or account.empty:
            continue
        nav = _account_with_nav(account)
        plt.plot(
            nav["date"],
            nav["nav"],
            label=line_labels.get(model_name, model_name),
            color=colors[model_name],
            linestyle=styles[model_name],
            linewidth=2.0,
            alpha=0.92,
        )

    plt.title(f"Final Transaction-Cost-Aware R-PPO Strategy Comparison: {stock} ({code})", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Return (Normalized)", fontsize=14)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=True, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.55)
    sns.despine(left=True, bottom=True)
    plt.gcf().autofmt_xdate()
    from matplotlib.dates import DateFormatter

    plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    plt.tight_layout()

    path = FINAL_OUTPUT_DIR / f"Final_Return_Ratio_{_safe_code(code)}_{FINAL_CAPITAL_TAG}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def final_run_stock(code: str, stock: str) -> list[dict]:
    print(f"\nStock: {stock} ({code})", flush=True)
    train_data, validation_data, test_data = load_data(code)
    available_tech = [column for column in FINAL_TECH_INDICATORS if column in train_data.mid.columns]

    rows = []
    account_values: dict[str, pd.DataFrame] = {}
    mean_accounts: dict[str, pd.DataFrame] = {}

    rppo_accounts = {}
    for seed in FINAL_SEEDS:
        model, normalization_stats = final_load_or_train_transaction_cost_aware_rppo(code, seed, train_data, validation_data, available_tech)
        test_env = final_make_rppo_env(test_data, available_tech, normalization_stats)
        metrics, account_value, action_history = backtest(model, test_env, test_data.mid, return_actions=True)
        rppo_accounts[seed] = account_value
        final_save_seed_outputs("transaction_cost_aware_rppo", code, seed, account_value, action_history)
        row = summarize_backtest(FINAL_MODEL_NAME, metrics, account_value, action_history)
        row.update(
            {
                "code": code,
                "stock": stock,
                "seed": seed,
                "turnover_penalty": FINAL_R_PPO_TURNOVER_PENALTY,
                "rebalance_threshold": FINAL_R_PPO_REBALANCE_THRESHOLD,
                "drawdown_penalty": FINAL_R_PPO_DRAWDOWN_PENALTY,
            }
        )
        rows.append(row)

    rppo_mean = final_mean_seed_account(rppo_accounts)
    account_values[FINAL_MODEL_NAME] = rppo_mean
    mean_accounts[FINAL_MODEL_NAME] = rppo_mean
    final_save_reference_output("transaction_cost_aware_rppo_mean", code, rppo_mean)

    for model_key, label in [("ppo_daily", "PPO (Daily)"), ("a2c_daily", "A2C (Daily)")]:
        baseline_accounts = {}
        for seed in FINAL_SEEDS:
            final_set_seed(seed)
            env_train = final_make_baseline_env(train_data, available_tech)
            env_validation = final_make_baseline_env(validation_data, available_tech, env_train.normalization_stats)
            env_test = final_make_baseline_env(test_data, available_tech, env_train.normalization_stats)
            model = final_load_or_train_baseline(model_key, label, code, seed, env_train, env_validation)
            metrics, account_value, action_history = backtest(model, env_test, test_data.mid, return_actions=True)
            baseline_accounts[seed] = account_value
            final_save_seed_outputs(model_key, code, seed, account_value, action_history)
            row = summarize_backtest(label, metrics, account_value, action_history)
            row.update(
                {
                    "code": code,
                    "stock": stock,
                    "seed": seed,
                    "turnover_penalty": FINAL_BASELINE_TURNOVER_PENALTY,
                    "rebalance_threshold": FINAL_BASELINE_REBALANCE_THRESHOLD,
                    "drawdown_penalty": np.nan,
                }
            )
            rows.append(row)

        mean_account = final_mean_seed_account(baseline_accounts)
        account_values[label] = mean_account
        mean_accounts[label] = mean_account
        final_save_reference_output(f"{model_key}_mean", code, mean_account)

    metrics, account_value = backtest_garch(train_data, test_data, initial_amount=FINAL_INITIAL_AMOUNT)
    account_values["GARCH"] = account_value
    final_save_reference_output("garch", code, account_value)
    row = summarize_backtest("GARCH", metrics, account_value, pd.DataFrame())
    row.update({"code": code, "stock": stock, "seed": np.nan})
    rows.append(row)

    timing_env_train = final_make_baseline_env(train_data, available_tech)
    timing_env_test = final_make_baseline_env(test_data, available_tech, timing_env_train.normalization_stats)
    metrics, account_value = backtest_buy_and_hold(
        test_data.mid,
        initial_amount=FINAL_INITIAL_AMOUNT,
        window_size=timing_env_test.window_size,
        decision_lag=timing_env_test.decision_lag,
        trade_start_date=test_data.trade_start_date,
    )
    account_values["Buy&Hold"] = account_value
    final_save_reference_output("buyhold", code, account_value)
    row = summarize_backtest("Buy&Hold", metrics, account_value, pd.DataFrame())
    row.update({"code": code, "stock": stock, "seed": np.nan})
    rows.append(row)

    baseline = account_value[["date"]].copy()
    baseline["account_value"] = FINAL_INITIAL_AMOUNT
    baseline["daily_return"] = 0.0
    final_save_reference_output("baseline_1_0", code, baseline)

    plot_path = final_plot_comparison(code, stock, account_values, mean_accounts)
    print(f"  [Output] Saved plot to {plot_path}", flush=True)
    return rows


def final_main() -> None:
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting final Transaction-Cost-Aware R-PPO thesis experiment...", flush=True)

    all_rows = []
    for code, stock in FINAL_STOCKS.items():
        stock_rows = final_run_stock(code, stock)
        all_rows.extend(stock_rows)
        diagnostics = pd.DataFrame(stock_rows)
        diagnostics_path = FINAL_OUTPUT_DIR / f"Final_TransactionCostAware_RPPO_{_safe_code(code)}_{FINAL_CAPITAL_TAG}_Diagnostics.csv"
        diagnostics.to_csv(diagnostics_path, index=False)
        print(diagnostics[["model", "seed", "final_nav", "sharpe", "max_drawdown", "total_turnover", "trade_count"]], flush=True)
        print(f"  [Output] Saved diagnostics to {diagnostics_path}", flush=True)

    seed_rows = pd.DataFrame(all_rows)
    seed_rows_path = FINAL_OUTPUT_DIR / "Final_TransactionCostAware_RPPO_SeedRows.csv"
    seed_rows.to_csv(seed_rows_path, index=False)

    summary = final_summarize_multiseed(seed_rows)
    summary_path = FINAL_OUTPUT_DIR / "Final_TransactionCostAware_RPPO_Summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nFinal Transaction-Cost-Aware R-PPO seed rows:", flush=True)
    print(seed_rows[["code", "stock", "model", "seed", "final_nav", "sharpe", "max_drawdown", "total_turnover", "trade_count"]].to_string(index=False), flush=True)
    print(f"\nSaved seed rows to {seed_rows_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    final_main()
