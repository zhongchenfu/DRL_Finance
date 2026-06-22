# DRL Finance: Multi-Frequency PPO Trading Experiment

This repository contains the final thesis experiment for a single-stock trading
strategy based on deep reinforcement learning.

The final model is named **Transaction-Cost-Aware R-PPO**. The name reflects the
main final-version design choice: the model keeps the multi-frequency R-PPO
framework while explicitly controlling trading turnover through transaction-cost
awareness and a rebalancing threshold.

Technically, the implementation is **PPO with a multi-frequency LSTM-based
feature extractor**. The policy receives daily, weekly, and 5-minute window
features, while PPO still uses the standard Stable-Baselines3 rollout buffer.

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
- GARCH
- Buy&Hold

The final experiment uses:

- 8 stocks across upward, weak/downward, and range/near-flat market states
- 3 random seeds: `0`, `1`, and `2`
- Initial capital: `1,000,000`
- Transaction-Cost-Aware R-PPO parameters:
  - `turnover_penalty = 0.005`
  - `rebalance_threshold = 0.05`
  - `drawdown_penalty = 0.0`

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

Across the final 8-stock test universe, Transaction-Cost-Aware R-PPO achieved
the highest average final NAV. It ranked first on 5 out of 8 stocks. It
outperformed PPO on 8/8 stocks, A2C on 7/8 stocks, GARCH on 8/8 stocks, and
Buy&Hold on 6/8 stocks.

Buy&Hold remains stronger on some strongly upward-trending stocks, so the thesis
claim should be framed as overall and condition-dependent improvement rather
than universal dominance over Buy&Hold.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Notes

- Data are split chronologically into train, validation, and 2024 test periods.
- R-PPO, PPO, and A2C are evaluated with repeated seeds.
- Buy&Hold is included as the main passive trading benchmark.
- Account-value trajectories and action histories are saved as CSV files for
  auditability.
- The final plots use mean curves across seeds for R-PPO, PPO, and A2C. Only
  the R-PPO mean curve displays the seed-level variation band.
