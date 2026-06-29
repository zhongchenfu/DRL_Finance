# DRL Finance: Multi-Frequency PPO Trading Experiment

This repository contains the final thesis experiment for a single-stock trading
strategy based on deep reinforcement learning.

The final model is reported as **Transaction-Cost-Aware R-PPO** for continuity
with the thesis experiments. Technically, it is a transaction-cost-aware PPO
trading model with a multi-frequency LSTM-based feature extractor and explicit
turnover control.

The policy receives daily, weekly, and 5-minute window features, while PPO still
uses the standard Stable-Baselines3 rollout buffer. Therefore, the implementation
should be described as PPO with recurrent-style feature extraction rather than a
modified recurrent PPO algorithm with hidden-state rollout propagation.

## Final Experiment Entry Point

Use this script for the final thesis experiment:

```bash
python final_r_ppo_experiment.py
```

`final_r_ppo_experiment.py` combines the core training/backtesting utilities and
the final Transaction-Cost-Aware R-PPO experiment configuration. It evaluates:

- Transaction-Cost-Aware R-PPO
- PPO (Daily)
- A2C (Daily)
- GARCH with volatility-scaled position sizing
- Buy&Hold

The final experiment uses:

- 8 stocks across upward, weak/downward, and range/near-flat market states
- 15 random seeds: `0` to `14`
- Initial capital: `1,000,000`
- Transaction-Cost-Aware R-PPO parameters:
  - `turnover_penalty = 0.005`
  - `rebalance_threshold = 0.05`
  - `drawdown_penalty = 0.0`

The GARCH benchmark is kept as an auxiliary econometric reference. It uses the
sign of the one-step mean forecast for market direction and scales positive
exposure by the one-step volatility forecast, so that higher predicted
volatility reduces the position size instead of always forcing an all-in trade.

## Supporting Files

The final experiment script depends on these project modules:

- `r_ppo_data.py`: chronological train/validation/test data loading
- `r_ppo_env.py`: trading environments and portfolio state logic
- `r_ppo_network.py`: multi-frequency LSTM feature extractor
- `data_fetcher.py`: BaoStock data download utilities
- `data_validation.py`: data quality checks

## Generate Thesis Tables

After running the final experiment, generate thesis-ready tables and statistical
checks with:

```bash
python generate_final_thesis_outputs.py
```

Important outputs are saved under:

```text
results/final_transaction_cost_aware_rppo/
```

Key result files:

- `Final_Return_Ratio_*_100w.png`
- `Account_Value_*.csv`
- `Action_History_*.csv`
- `Final_TransactionCostAware_RPPO_SeedRows.csv`
- `Final_TransactionCostAware_RPPO_Summary.csv`
- `Final_TransactionCostAware_RPPO_Statistical_Report.csv`
- `Final_TransactionCostAware_RPPO_Thesis_Tables.xlsx`
- `Final_TransactionCostAware_RPPO_Thesis_Summary.md`

Data-warning checks are saved under:

```text
results/data_warning_report/
```

## Final Result Summary

Across the final 8-stock test universe, Transaction-Cost-Aware R-PPO ranked
first by final NAV on 5 out of 8 stocks. It outperformed PPO (Daily), A2C
(Daily), and GARCH on all 8 stocks in the stock-level mean comparison.

Buy&Hold ranked first on the three strongly upward-trending stocks and therefore
had a slightly higher average final NAV than Transaction-Cost-Aware R-PPO. This
is expected because passive full exposure is difficult to beat after transaction
costs during strong upward trends. Transaction-Cost-Aware R-PPO outperformed PPO
(Daily), A2C (Daily), and the auxiliary GARCH benchmark on all 8 stocks in the
stock-level mean comparison. The thesis claim should therefore be framed as
improved robustness over active learning and econometric baselines, with
condition-dependent performance relative to Buy&Hold.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Notes

- Data are split chronologically into train, validation, and 2024 test periods.
- R-PPO, PPO, and A2C are evaluated with repeated seeds.
- Buy&Hold is included as the main passive trading benchmark.
- GARCH is included as an auxiliary econometric benchmark, not as the main
  policy-learning baseline.
- Account-value trajectories and action histories are saved as CSV files for
  auditability.
- The final plots show individual seed trajectories as thin lines and mean
  curves for R-PPO, PPO, and A2C.
- The R-PPO mean curve displays a 95% confidence interval rather than a
  mean-plus/minus-standard-deviation band.
- The training reward uses clipped log returns with turnover control, while the
  evaluation figures and tables report realized portfolio/account value metrics.
- The environment supports an optional drawdown penalty, but the final submitted
  experiment keeps `drawdown_penalty = 0.0`; the final model should therefore be
  described as turnover-aware rather than drawdown-aware.
