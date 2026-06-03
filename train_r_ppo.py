"""
Train and evaluate the multi-frequency R-PPO experiment and baseline models.

R-PPO uses standard PPO rollout storage. Each stored observation is already a
truncated multi-frequency sequence state produced by MultiFreqStockEnv.
"""

import copy
import os
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
import pyfolio  # noqa: F401
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 使用系统已安装的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

# 训练集、验证集和测试集划分
TRAIN_START = "2020-03-01"
TRAIN_END = "2023-06-30"
VALIDATION_START = "2023-07-01"
VALIDATION_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"

STOCKS = { "sh.601628": "中国人寿", "sh.601688": "华泰证券", "sh.600519": "贵州茅台"}
# STOCKS = {"sh.600519": "贵州茅台"}
# ================================
# Configuration
# ================================
CONFIG = {
    "TRAIN_MODELS": True,  # True: 训练新模型并保存; False: 加载已有权重进行评估
    "MODELS_TO_RUN": ["R-PPO", "PPO", "A2C", "GARCH"], # 自由配置要跑的模
    # "MODELS_TO_RUN": [ "A2C"], # 自由配置要跑的模型
    # 主实验默认使用 1000 万元。资金敏感性实验可改为:
    # [1_000_000, 5_000_000, 10_000_000]
    "INITIAL_AMOUNTS": [1_000_000],
}
REBALANCE_THRESHOLD = DEFAULT_REBALANCE_THRESHOLD
TURNOVER_PENALTY_COEF = DEFAULT_TURNOVER_PENALTY_COEF


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


def learn_with_validation(model, env_validation, total_timesteps=50000):
    callback = None
    if env_validation is not None:
        callback = ValidationEarlyStoppingCallback(env_validation)
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
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
            "cash_ratio": float(portfolio_state[1]),
            "last_action": self.last_action,
            "transaction_cost": transaction_cost,
            "turnover": turnover,
            "raw_reward": raw_reward,
            "turnover_penalty": turnover_penalty,
            "rebalance_skipped": rebalance_skipped,
        }

def train_r_ppo(env_train, env_validation=None):
    print("  [R-PPO] Training...")
    
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
        ),
        net_arch=dict(pi=[32, 32], vf=[32, 32])
    )
    
    model = PPO("MlpPolicy", env_train, policy_kwargs=policy_kwargs, 
                learning_rate=0.0003, 
                n_steps=512,
                batch_size=64,
                gamma=0.99, verbose=0)
    
    return learn_with_validation(model, env_validation)

def train_baseline_ppo(env_train, env_validation=None):
    print("  [PPO Base] Training...")
    model = PPO("MlpPolicy", env_train, learning_rate=0.0003, n_steps=512,
                batch_size=64, gamma=0.99, verbose=0)
    return learn_with_validation(model, env_validation)

def train_baseline_a2c(env_train, env_validation=None):
    print("  [A2C Base] Training...")
    # model = A2C("MlpPolicy", env_train, n_steps=1000, learning_rate=0.0003, verbose=0)
    # 修改训练函数中的 A2C 参数
    model = A2C("MlpPolicy", env_train, 
                n_steps=100,            # 增加每步观察的长度，默认5太短了
                learning_rate=0.0003, 
                verbose=0)
    return learn_with_validation(model, env_validation)

def evaluate_account_value(account_value):
    account_value = account_value.copy()
    if account_value.empty:
        return {}, account_value

    account_value['daily_return'] = account_value['account_value'].pct_change()

    import pyfolio as pf
    returns = account_value.set_index('date')['daily_return'].dropna()
    returns.index = pd.to_datetime(returns.index)
    returns.index = returns.index.tz_localize("UTC")

    try:
        ann_return = pf.timeseries.annual_return(returns)
        ann_vol = pf.timeseries.annual_volatility(returns)
        sharpe = pf.timeseries.sharpe_ratio(returns)
        max_dd = pf.timeseries.max_drawdown(returns)
        calmar = pf.timeseries.calmar_ratio(returns)

        downside_returns = returns[returns < 0]
        down_std = downside_returns.std() * np.sqrt(252)
        sortino = ann_return / down_std if down_std > 0 else np.nan
        cum_ret = account_value['account_value'].iloc[-1] / account_value['account_value'].iloc[0] - 1

        return {
            "Annual return": ann_return,
            "Cumulative return": cum_ret,
            "Annual volatility": ann_vol,
            "Sharpe ratio": sharpe,
            "Calmar ratio": calmar,
            "Max drawdown": max_dd,
            "Sortino ratio": sortino
        }, account_value
    except Exception as e:
        print("Error evaluating", e)
        return {}, account_value


def backtest(model, env_test, df_test_mid):
    obs, _ = env_test.reset()
    done = False
    initial_date = getattr(env_test, "execution_date", None)
    if initial_date is None:
        asset_history = []
        date_history = []
    else:
        asset_history = [float(getattr(env_test, "initial_amount", 1000000))]
        date_history = [pd.Timestamp(initial_date)]
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env_test.step(action)
        asset_history.append(info.get('asset_value', 1000000))
        if "next_date" in info:
            date_history.append(pd.Timestamp(info["next_date"]))
        elif "execution_date" in info:
            date_history.append(pd.Timestamp(info["execution_date"]))
        
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
    return evaluate_account_value(account_value)


def backtest_buy_and_hold(df_mid, initial_amount=1000000, fee=0.003, window_size=15, decision_lag=1):
    df_mid = df_mid.reset_index(drop=True)
    start_idx = window_size - 1 + decision_lag
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

def main():
    print("Starting R-PPO experiments on 3 stocks...")
    tech_indicators = ["RSI", "MACD_DIF", "MACD_DEA", "MACD_BAR", "BOLL_MID", "BOLL_UPPER", "BOLL_LOWER", "CCI", "SMA5", "SMA10", "SMA20", "DX"]
    
    for initial_amount, (code, name) in (
        (amount, stock)
        for amount in CONFIG["INITIAL_AMOUNTS"]
        for stock in STOCKS.items()
    ):
        capital_tag = f"{initial_amount // 10_000}w"
        print(f"\nEvaluating {name} ({code}), initial amount={initial_amount:,.0f}")
        train_data, validation_data, test_data = load_data(code)
        
        # Tech indicator list only the ones existing in the dataset
        avail_tech = [c for c in tech_indicators if c in train_data.mid.columns]

        res_rppo, df_rppo = {}, pd.DataFrame()
        res_ppo, df_ppo = {}, pd.DataFrame()
        res_a2c, df_a2c = {}, pd.DataFrame()
        res_garch, df_garch = {}, pd.DataFrame()
        res_bnh, bnh_df = {}, pd.DataFrame()

        # 1. R-PPO
        if "R-PPO" in CONFIG["MODELS_TO_RUN"]:
            def make_rppo_env(data, normalization_stats=None):
                return MultiFreqStockEnv(
                    data.low,
                    data.mid,
                    data.high,
                    initial_amount=initial_amount,
                    tech_indicator_list=avail_tech,
                    normalization_stats=normalization_stats,
                    trade_start_date=data.trade_start_date,
                    rebalance_threshold=REBALANCE_THRESHOLD,
                    turnover_penalty_coef=TURNOVER_PENALTY_COEF,
                )
            env_rppo_train = make_rppo_env(train_data)
            env_rppo_validation = make_rppo_env(validation_data, env_rppo_train.normalization_stats)
            env_rppo_test = make_rppo_env(test_data, env_rppo_train.normalization_stats)
            rppo_path = os.path.join(RESULTS_DIR, f"model_rppo_{code}_{capital_tag}.zip")
            
            if CONFIG["TRAIN_MODELS"]:
                rppo = train_r_ppo(env_rppo_train, env_rppo_validation)
                rppo.save(rppo_path)
                print(f"  [R-PPO] Model saved to {rppo_path}")
            else:
                print(f"  [R-PPO] Loading model from {rppo_path}")
                rppo = PPO.load(rppo_path)
            
            res_rppo, df_rppo = backtest(rppo, env_rppo_test, test_data.mid)
            
        # 2. Baselines (PPO/A2C)
        if "PPO" in CONFIG["MODELS_TO_RUN"] or "A2C" in CONFIG["MODELS_TO_RUN"] or "Buy&Hold" in CONFIG["MODELS_TO_RUN"]:
            def make_base_env(data, normalization_stats=None):
                return SimpleBaselineEnv(
                    data.mid,
                    initial_amount=initial_amount,
                    tech_indicator_list=avail_tech,
                    normalization_stats=normalization_stats,
                    trade_start_date=data.trade_start_date,
                    rebalance_threshold=REBALANCE_THRESHOLD,
                    turnover_penalty_coef=TURNOVER_PENALTY_COEF,
                )
            env_base_train = make_base_env(train_data)
            env_base_validation = make_base_env(validation_data, env_base_train.normalization_stats)
            env_base_test = make_base_env(test_data, env_base_train.normalization_stats)
            
            if "PPO" in CONFIG["MODELS_TO_RUN"]:
                ppo_path = os.path.join(RESULTS_DIR, f"model_ppo_base_{code}_{capital_tag}.zip")
                if CONFIG["TRAIN_MODELS"]:
                    ppo_base = train_baseline_ppo(env_base_train, env_base_validation)
                    ppo_base.save(ppo_path)
                    print(f"  [PPO Base] Model saved to {ppo_path}")
                else:
                    print(f"  [PPO Base] Loading model from {ppo_path}")
                    ppo_base = PPO.load(ppo_path)
                res_ppo, df_ppo = backtest(ppo_base, env_base_test, test_data.mid)
                
            if "A2C" in CONFIG["MODELS_TO_RUN"]:
                a2c_path = os.path.join(RESULTS_DIR, f"model_a2c_base_{code}_{capital_tag}.zip")
                if CONFIG["TRAIN_MODELS"]:
                    a2c_base = train_baseline_a2c(env_base_train, env_base_validation)
                    a2c_base.save(a2c_path)
                    print(f"  [A2C Base] Model saved to {a2c_path}")
                else:
                    print(f"  [A2C Base] Loading model from {a2c_path}")
                    a2c_base = A2C.load(a2c_path)
                res_a2c, df_a2c = backtest(a2c_base, env_base_test, test_data.mid)
                
            if "Buy&Hold" in CONFIG["MODELS_TO_RUN"]:
                res_bnh, bnh_df = backtest_buy_and_hold(
                    test_data.mid,
                    initial_amount=initial_amount,
                    window_size=env_base_test.window_size,
                    decision_lag=env_base_test.decision_lag,
                )

        if "GARCH" in CONFIG["MODELS_TO_RUN"]:
            res_garch, df_garch = backtest_garch(train_data, test_data, initial_amount=initial_amount)

        # Build output dataframe
        res_dict = {}
        if "R-PPO" in CONFIG["MODELS_TO_RUN"]: res_dict["R-PPO"] = res_rppo
        if "PPO" in CONFIG["MODELS_TO_RUN"]: res_dict["PPO"] = res_ppo
        if "A2C" in CONFIG["MODELS_TO_RUN"]: res_dict["A2C"] = res_a2c
        if "GARCH" in CONFIG["MODELS_TO_RUN"]: res_dict["GARCH"] = res_garch
        if "Buy&Hold" in CONFIG["MODELS_TO_RUN"]: res_dict["Buy&Hold"] = res_bnh
        
        df_res = pd.DataFrame(res_dict)
        print(df_res)
        df_res.to_csv(os.path.join(RESULTS_DIR, f"RPPO_{code}_{capital_tag}_results.csv"))
        
        # Plot Return Ratio (Beautified)
        plt.figure(figsize=(12, 7))
        
        # 定义颜色映射，确保 R-PPO 最突出
        model_colors = {
            'R-PPO': '#D62728',   # 红色 (醒目)
            'PPO': '#1F77B4',     # 蓝色
            'A2C': '#2CA02C',     # 绿色
            'GARCH': '#9467BD',   # 紫色
            'Buy & Hold': '#7F7F7F' # 灰色 (基准)
        }

        # 绘制 1.0 的基准线
        plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (1.0)')

        if not df_rppo.empty:
            plt.plot(pd.to_datetime(df_rppo['date']), df_rppo['account_value'] / df_rppo['account_value'].iloc[0], 
                     label='R-PPO', color=model_colors['R-PPO'], linewidth=2.5, zorder=10)
        if not df_ppo.empty:
            plt.plot(pd.to_datetime(df_ppo['date']), df_ppo['account_value'] / df_ppo['account_value'].iloc[0], 
                     label='PPO (Daily)', color=model_colors['PPO'], linewidth=1.5, alpha=0.8)
        if not df_a2c.empty:
            plt.plot(pd.to_datetime(df_a2c['date']), df_a2c['account_value'] / df_a2c['account_value'].iloc[0], 
                     label='A2C (Daily)', color=model_colors['A2C'], linewidth=1.5, alpha=0.8)
        if not df_garch.empty:
            plt.plot(pd.to_datetime(df_garch['date']), df_garch['account_value'] / df_garch['account_value'].iloc[0], 
                     label='GARCH', color=model_colors['GARCH'], linewidth=1.5, alpha=0.8)
        if not bnh_df.empty:
            plt.plot(pd.to_datetime(bnh_df['date']), bnh_df['account_value'] / bnh_df['account_value'].iloc[0], 
                     label='Buy & Hold', color=model_colors['Buy & Hold'], linewidth=2, linestyle=':', alpha=0.7)
            
        plt.title(f'策略累计收益率对比: {name} ({code})', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('日期', fontsize=14)
        plt.ylabel('累计收益率 (Normalized)', fontsize=14)
        
        # 优化图例
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, fontsize=12)
        
        # 优化网格和边框
        plt.grid(True, linestyle='--', alpha=0.6)
        sns.despine(left=True, bottom=True)
        
        # 优化日期轴显示
        plt.gcf().autofmt_xdate()
        from matplotlib.dates import DateFormatter
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        
        img_path = os.path.join(RESULTS_DIR, f"Return_Ratio_{code}_{capital_tag}.png")
        plt.savefig(img_path, dpi=300, bbox_inches='tight') # 提高分辨率
        plt.close()
        print(f"  [Output] Saved beautified plot to {img_path}")

if __name__ == "__main__":

    main()
