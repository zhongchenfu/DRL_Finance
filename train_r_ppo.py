"""
Train and evaluate the multi-frequency R-PPO experiment and baseline models.

R-PPO uses standard PPO rollout storage. Each stored observation is already a
truncated multi-frequency sequence state produced by MultiFreqStockEnv.
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from r_ppo_data import load_experiment_data
from r_ppo_env import MultiFreqStockEnv
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

# 训练集和测试集划分
TRAIN_START = "2020-03-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"

STOCKS = {"sh.601628": "中国人寿", "sh.601688": "华泰证券", "sh.600519": "贵州茅台"}
# ================================
# Configuration
# ================================
CONFIG = {
    "TRAIN_MODELS": False,  # True: 训练新模型并保存; False: 加载已有权重进行评估
    "MODELS_TO_RUN": ["R-PPO", "PPO", "A2C", "GARCH"], # 自由配置要跑的模
    # "MODELS_TO_RUN": [ "A2C"], # 自由配置要跑的模型
}

def load_data(code):
    train, test = load_experiment_data(
        code,
        data_dir=DATA_DIR,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
    )
    return (train.low, train.mid, train.high), (test.low, test.mid, test.high)

# ================================
# Baseline Environment (PPO/A2C)
# ================================
class SimpleBaselineEnv(gym.Env):
    """
    Daily-frequency baseline environment.

    It uses the same timing convention as MultiFreqStockEnv: observations end at
    day t - decision_lag, actions execute at the close of day t, and rewards are
    realized from day t to day t + 1.
    """
    def __init__(
        self,
        df_mid,
        initial_amount=1000000,
        transaction_fee_percent=0.003,
        window_size=15,
        tech_indicator_list=None,
        decision_lag=1,
    ):
        super().__init__()
        if decision_lag < 0:
            raise ValueError("decision_lag must be non-negative.")

        self.df = df_mid.copy()
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        if self.df["date"].isna().any():
            raise ValueError("date contains invalid datetime values.")
        self.df = self.df.sort_values("date").reset_index(drop=True)

        self.initial_amount = initial_amount
        self.fee = transaction_fee_percent
        self.window_size = int(window_size)
        self.decision_lag = int(decision_lag)
        
        self.cols = ["open", "high", "low", "close", "volume", "amount"] + list(tech_indicator_list or [])
        self.cols = [c for c in self.cols if c in self.df.columns]
        if "close" not in self.cols:
            raise ValueError("The baseline environment requires a close column.")
        self.close_idx = self.cols.index("close")
        self.np_data = self.df[self.cols].to_numpy(dtype=np.float32)
        
        self.state_dim = self.window_size * len(self.cols)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        self.day = 0
        self.terminal = False
        self.reward = 0.0

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
        self.day = self.window_size - 1 + self.decision_lag
        self.terminal = False
        self.cash = self.initial_amount
        self.shares = 0
        self.asset_value = self.initial_amount
        self.reward = 0.0
        return self._get_state(), {}
        
    def _get_state(self):
        end = self.observation_index
        start = end - self.window_size + 1
        state = self.np_data[start : end + 1]
        state = np.nan_to_num(state).flatten()
        return state.astype(np.float32)
        
    def step(self, action):
        self.terminal = self.day >= len(self.df) - 2
        if self.terminal:
            return self._get_state(), self.reward, self.terminal, False, self._info()

        current_price = self.np_data[self.day, self.close_idx]
        if current_price <= 0:
            self.day += 1
            return self._get_state(), 0.0, self.terminal, False, self._info()

        target_weight = np.clip((float(action[0]) + 1.0) / 2.0, 0.0, 1.0)
        total_asset = self.cash + self.shares * current_price
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
        elif delta_shares < 0:
            sell_shares = min(-delta_shares, self.shares)
            sell_shares = (sell_shares // 100) * 100
            self.cash += sell_shares * current_price * (1 - self.fee)
            self.shares -= sell_shares
            
        next_price = self.np_data[self.day + 1, self.close_idx]
        new_asset = self.cash + self.shares * next_price
        
        if self.asset_value > 0 and new_asset > 0:
            self.reward = float(np.clip(np.log(new_asset / self.asset_value), -0.05, 0.05))
        else:
            self.reward = 0.0
            
        self.asset_value = new_asset
        info = self._info(execution_price=current_price, next_price=next_price)
        self.day += 1
        return self._get_state(), self.reward, self.terminal, False, info

    def _info(self, execution_price=None, next_price=None):
        next_idx = min(self.day + 1, len(self.df) - 1)
        return {
            "asset_value": self.asset_value,
            "execution_date": self.execution_date,
            "observation_end_date": self.observation_end_date,
            "next_date": pd.Timestamp(self.df["date"].iloc[next_idx]),
            "execution_price": execution_price,
            "next_price": next_price,
        }

def train_r_ppo(env_train):
    print("  [R-PPO] Training...")
    
    # The environment observation already contains truncated low/mid/high
    # frequency windows. Standard PPO keeps random minibatch sampling while the
    # custom feature extractor learns recurrent representations from each window.
    policy_kwargs = dict(
        features_extractor_class=MultiFreqFeaturesExtractor,
        features_extractor_kwargs=dict(
            low_len=env_train.low_len, low_features=env_train.low_features, 
            mid_len=env_train.mid_len, mid_features=env_train.mid_features,
            high_len=env_train.high_len, high_features=env_train.high_features
        ),
        net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64])
    )
    
    model = PPO("MlpPolicy", env_train, policy_kwargs=policy_kwargs, 
                learning_rate=0.0003, 
                n_steps=4096, 
                batch_size=128,
                gamma=0.99, verbose=0)
    
    model.learn(total_timesteps=550000, progress_bar=True)
    return model

def train_baseline_ppo(env_train):
    print("  [PPO Base] Training...")
    model = PPO("MlpPolicy", env_train, learning_rate=0.0003, n_steps=4096, 
                batch_size=128, gamma=0.99, verbose=0)
    model.learn(total_timesteps=550000, progress_bar=True)
    return model

def train_baseline_a2c(env_train):
    print("  [A2C Base] Training...")
    # model = A2C("MlpPolicy", env_train, n_steps=1000, learning_rate=0.0003, verbose=0)
    # 修改训练函数中的 A2C 参数
    model = A2C("MlpPolicy", env_train, 
                n_steps=100,            # 增加每步观察的长度，默认5太短了
                learning_rate=0.0003, 
                verbose=0)
    model.learn(total_timesteps=550000, progress_bar=True)
    return model

def backtest(model, env_test, df_test_mid):
    obs, _ = env_test.reset()
    done = False
    asset_history = []
    date_history = []
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env_test.step(action)
        asset_history.append(info.get('asset_value', 1000000))
        if "execution_date" in info:
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
    account_value['daily_return'] = account_value['account_value'].pct_change()
    account_value.dropna(inplace=True)
    
    import pyfolio as pf
    returns = account_value.set_index('date')['daily_return']
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
        return {}, pd.DataFrame()

def backtest_garch(train_data, test_data, initial_amount=1000000, fee=0.003, window_size=15, decision_lag=1):
    try:
        from arch import arch_model
    except ImportError:
        print("Please install arch package: pip install arch")
        return {}, pd.DataFrame()
        
    print("  [GARCH Base] Evaluating...")
    df_train = train_data[1].reset_index(drop=True)
    df_test = test_data[1].reset_index(drop=True)

    asset_history = []
    date_history = []
    cash = initial_amount
    shares = 0
    asset_value = initial_amount

    start_idx = window_size - 1 + decision_lag

    for i in range(len(df_test) - 1):
        current_price = float(df_test['close'].iloc[i])

        observation_end = i - decision_lag
        if observation_end >= 0:
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

        target_weight = 1.0 if (pred_next_ret > 0 and i >= start_idx) else 0.0

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

        next_price = float(df_test['close'].iloc[i + 1])
        asset_value = cash + shares * next_price
        asset_history.append(asset_value)
        date_history.append(pd.Timestamp(df_test['date'].iloc[i]))

    account_value = pd.DataFrame({
        'date': date_history,
        'account_value': asset_history
    })

    account_value = account_value.iloc[start_idx:].reset_index(drop=True)
    if account_value.empty:
        return {}, account_value

    account_value['daily_return'] = account_value['account_value'].pct_change().fillna(0)
    
    import pyfolio as pf
    returns = account_value.set_index('date')['daily_return']
    returns.index = pd.to_datetime(returns.index).tz_localize("UTC")
    
    try:
        perf_stats = {
            "Annual return": pf.timeseries.annual_return(returns),
            "Cumulative return": account_value['account_value'].iloc[-1] / account_value['account_value'].iloc[0] - 1,
            "Annual volatility": pf.timeseries.annual_volatility(returns),
            "Sharpe ratio": pf.timeseries.sharpe_ratio(returns),
            "Calmar ratio": pf.timeseries.calmar_ratio(returns),
            "Max drawdown": pf.timeseries.max_drawdown(returns),
        }
        # 计算 Sortino
        downside = returns[returns < 0]
        ds_std = downside.std() * np.sqrt(252)
        perf_stats["Sortino ratio"] = perf_stats["Annual return"] / ds_std if ds_std > 0 else np.nan
        
        return perf_stats, account_value
    except Exception as e:
        print(f"  [GARCH] Metric error: {e}")
        return {}, account_value

def main():
    print("Starting R-PPO experiments on 3 stocks...")
    tech_indicators = ["RSI", "MACD_DIF", "MACD_DEA", "MACD_BAR", "BOLL_MID", "BOLL_UPPER", "BOLL_LOWER", "CCI", "SMA5", "SMA10", "SMA20", "DX"]
    
    for code, name in STOCKS.items():
        print(f"\nEvaluating {name} ({code})")
        train_data, test_data = load_data(code)
        
        # Tech indicator list only the ones existing in the dataset
        avail_tech = [c for c in tech_indicators if c in train_data[1].columns]

        res_rppo, df_rppo = {}, pd.DataFrame()
        res_ppo, df_ppo = {}, pd.DataFrame()
        res_a2c, df_a2c = {}, pd.DataFrame()
        res_garch, df_garch = {}, pd.DataFrame()
        res_bnh, bnh_df = {}, pd.DataFrame()

        # 1. R-PPO
        if "R-PPO" in CONFIG["MODELS_TO_RUN"]:
            def make_rppo_env(data):
                return MultiFreqStockEnv(data[0], data[1], data[2], tech_indicator_list=avail_tech)
            env_rppo_train = make_rppo_env(train_data)
            env_rppo_test = make_rppo_env(test_data)
            rppo_path = os.path.join(RESULTS_DIR, f"model_rppo_{code}.zip")
            
            if CONFIG["TRAIN_MODELS"]:
                rppo = train_r_ppo(env_rppo_train)
                rppo.save(rppo_path)
                print(f"  [R-PPO] Model saved to {rppo_path}")
            else:
                print(f"  [R-PPO] Loading model from {rppo_path}")
                rppo = PPO.load(rppo_path)
            
            res_rppo, df_rppo = backtest(rppo, env_rppo_test, test_data[1])
            
        # 2. Baselines (PPO/A2C)
        if "PPO" in CONFIG["MODELS_TO_RUN"] or "A2C" in CONFIG["MODELS_TO_RUN"] or "Buy&Hold" in CONFIG["MODELS_TO_RUN"]:
            def make_base_env(data):
                return SimpleBaselineEnv(data[1], tech_indicator_list=avail_tech)
            env_base_train = make_base_env(train_data)
            env_base_test = make_base_env(test_data)
            
            if "PPO" in CONFIG["MODELS_TO_RUN"]:
                ppo_path = os.path.join(RESULTS_DIR, f"model_ppo_base_{code}.zip")
                if CONFIG["TRAIN_MODELS"]:
                    ppo_base = train_baseline_ppo(env_base_train)
                    ppo_base.save(ppo_path)
                    print(f"  [PPO Base] Model saved to {ppo_path}")
                else:
                    print(f"  [PPO Base] Loading model from {ppo_path}")
                    ppo_base = PPO.load(ppo_path)
                res_ppo, df_ppo = backtest(ppo_base, env_base_test, test_data[1])
                
            if "A2C" in CONFIG["MODELS_TO_RUN"]:
                a2c_path = os.path.join(RESULTS_DIR, f"model_a2c_base_{code}.zip")
                if CONFIG["TRAIN_MODELS"]:
                    a2c_base = train_baseline_a2c(env_base_train)
                    a2c_base.save(a2c_path)
                    print(f"  [A2C Base] Model saved to {a2c_path}")
                else:
                    print(f"  [A2C Base] Loading model from {a2c_path}")
                    a2c_base = A2C.load(a2c_path)
                res_a2c, df_a2c = backtest(a2c_base, env_base_test, test_data[1])
                
            if "Buy&Hold" in CONFIG["MODELS_TO_RUN"]:
                idx_start = env_base_test.window_size - 1
                bnh_prices = test_data[1]['close'].iloc[idx_start:].values
                bnh_bdate = test_data[1]['date'].iloc[idx_start:].values
                
                if len(bnh_prices) > 0:
                    initial_price = bnh_prices[0]
                    shares = int(1000000 / initial_price)
                    cash = 1000000 - shares * initial_price
                    bnh_assets = cash + shares * bnh_prices
                    
                    bnh_df = pd.DataFrame({'date': bnh_bdate, 'account_value': bnh_assets})
                    bnh_df['daily_return'] = bnh_df['account_value'].pct_change()
                    bnh_df.dropna(inplace=True)
                    
                    import pyfolio as pf
                    bnh_returns = bnh_df.set_index('date')['daily_return']
                    bnh_returns.index = pd.to_datetime(bnh_returns.index)
                    bnh_returns.index = bnh_returns.index.tz_localize("UTC")
                    
                    try:
                        res_bnh = {
                            "Annual return": pf.timeseries.annual_return(bnh_returns),
                            "Cumulative return": bnh_assets[-1] / bnh_assets[0] - 1,
                            "Annual volatility": pf.timeseries.annual_volatility(bnh_returns),
                            "Sharpe ratio": pf.timeseries.sharpe_ratio(bnh_returns),
                            "Calmar ratio": pf.timeseries.calmar_ratio(bnh_returns),
                            "Max drawdown": pf.timeseries.max_drawdown(bnh_returns),
                        }
                        downside = bnh_returns[bnh_returns < 0]
                        d_std = downside.std() * np.sqrt(252)
                        res_bnh["Sortino ratio"] = res_bnh["Annual return"] / d_std if d_std > 0 else np.nan
                    except:
                        res_bnh = {}

        if "GARCH" in CONFIG["MODELS_TO_RUN"]:
            res_garch, df_garch = backtest_garch(train_data, test_data)

        # Build output dataframe
        res_dict = {}
        if "R-PPO" in CONFIG["MODELS_TO_RUN"]: res_dict["R-PPO"] = res_rppo
        if "PPO" in CONFIG["MODELS_TO_RUN"]: res_dict["PPO"] = res_ppo
        if "A2C" in CONFIG["MODELS_TO_RUN"]: res_dict["A2C"] = res_a2c
        if "GARCH" in CONFIG["MODELS_TO_RUN"]: res_dict["GARCH"] = res_garch
        if "Buy&Hold" in CONFIG["MODELS_TO_RUN"]: res_dict["Buy&Hold"] = res_bnh
        
        df_res = pd.DataFrame(res_dict)
        print(df_res)
        df_res.to_csv(os.path.join(RESULTS_DIR, f"RPPO_{code}_results.csv"))
        
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
        
        img_path = os.path.join(RESULTS_DIR, f"Return_Ratio_{code}.png")
        plt.savefig(img_path, dpi=300, bbox_inches='tight') # 提高分辨率
        plt.close()
        print(f"  [Output] Saved beautified plot to {img_path}")

if __name__ == "__main__":

    main()
