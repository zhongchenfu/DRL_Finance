# Final Transaction-Cost-Aware R-PPO Thesis Summary

## Experimental setup

- Main model: Transaction-Cost-Aware R-PPO.
- Parameters: `turnover_penalty=0.005`, `rebalance_threshold=0.05`, no global drawdown penalty.
- Test universe: 8 stocks across upward, weak/downward, and range/near-flat market states.
- Seeds: R-PPO, PPO (Daily), and A2C (Daily) use seed 0, 1, and 2.
- Benchmarks: PPO (Daily), A2C (Daily), GARCH, Buy&Hold, and Baseline (1.0) in plots.

## Main result

Across the 8-stock out-of-sample universe, Transaction-Cost-Aware R-PPO achieved the highest average final NAV. It ranked first on 5 of 8 stocks. Buy&Hold remained strongest on two clearly upward stocks, which is expected because a passive full-exposure strategy is naturally difficult to beat in strong trend periods.

## Overall model averages

| model | avg_final_nav | avg_sharpe | avg_max_drawdown | avg_total_turnover | avg_trade_count |
| --- | --- | --- | --- | --- | --- |
| Transaction-Cost-Aware R-PPO | 1.1585 | 0.6938 | -0.1679 | 4.6096 | 25.125 |
| PPO (Daily) | 1.0006 | 0.0635 | -0.1566 | 24.9085 | 144.1667 |
| A2C (Daily) | 1.0663 | 0.4155 | -0.1329 | 2.1351 | 23.625 |
| GARCH | 0.9366 | -0.7088 | -0.1976 |  | 0.0 |
| Buy&Hold | 1.1326 | 0.4531 | -0.2458 |  | 0.0 |

## Best-model counts by final NAV

| model | best_stock_count |
| --- | --- |
| Transaction-Cost-Aware R-PPO | 5 |
| PPO (Daily) | 0 |
| A2C (Daily) | 1 |
| GARCH | 0 |
| Buy&Hold | 2 |

## Stock-level ranking summary

| code | stock | category | best_model | rppo_rank | rppo_final_nav | rppo_minus_buyhold |
| --- | --- | --- | --- | --- | --- | --- |
| sh.601628 | China Life | Upward | Buy&Hold | 2 | 1.2961 | -0.213 |
| sh.601688 | Huatai Securities | Upward | Buy&Hold | 2 | 1.2191 | -0.087 |
| sz.002594 | BYD | Upward | Transaction-Cost-Aware R-PPO | 1 | 1.4358 | 0.0204 |
| sh.600519 | Kweichow Moutai | Weak/Downward | A2C (Daily) | 2 | 0.9687 | 0.0661 |
| sz.002415 | Hikvision | Weak/Downward | Transaction-Cost-Aware R-PPO | 1 | 1.0626 | 0.1698 |
| sz.000963 | Huadong Medicine | Weak/Downward | Transaction-Cost-Aware R-PPO | 1 | 0.9597 | 0.1027 |
| sh.600887 | Yili Group | Range/Near-flat | Transaction-Cost-Aware R-PPO | 1 | 1.183 | 0.0698 |
| sh.600585 | Conch Cement | Range/Near-flat | Transaction-Cost-Aware R-PPO | 1 | 1.1425 | 0.0783 |

## Paired statistical checks on final NAV

| comparison | n_pairs | mean_advantage | rppo_better_n | competitor_better_n | paired_t_pvalue | wilcoxon_pvalue | exact_sign_flip_pvalue | conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transaction-Cost-Aware R-PPO vs PPO (Daily) | 8 | 0.1579 | 8 | 0 | 0.002 | 0.0078 | 0.0078 | R-PPO advantage is statistically significant at the 5% level by paired t-test. |
| Transaction-Cost-Aware R-PPO vs A2C (Daily) | 8 | 0.0922 | 7 | 1 | 0.0297 | 0.0156 | 0.0156 | R-PPO advantage is statistically significant at the 5% level by paired t-test. |
| Transaction-Cost-Aware R-PPO vs GARCH | 8 | 0.2218 | 8 | 0 | 0.0011 | 0.0078 | 0.0078 | R-PPO advantage is statistically significant at the 5% level by paired t-test. |
| Transaction-Cost-Aware R-PPO vs Buy&Hold | 8 | 0.0259 | 6 | 2 | 0.5648 | 0.5469 | 0.5547 | R-PPO shows a positive average advantage and wins most paired stocks, but significance is limited by sample size. |

## Thesis Summary

The experimental results show that the proposed Transaction-Cost-Aware R-PPO model achieves competitive and generally superior performance across the eight-stock out-of-sample test universe. Under the setting of three random seeds, the model obtains the highest average final NAV among all compared methods and ranks first in five out of eight individual stocks. This indicates that incorporating transaction-cost awareness and a rebalancing threshold into the multi-frequency PPO/LSTM framework can improve the stability and practical robustness of the trading strategy.

Compared with PPO, A2C, and GARCH, the proposed model demonstrates stronger overall performance, particularly in weak, downward, and range-bound market conditions. In these market states, active position adjustment and transaction-cost control appear to help the model avoid excessive trading while maintaining adaptive exposure to changing price dynamics. The lower turnover and trade count also suggest that the transaction-cost-aware design effectively reduces unnecessary rebalancing, which is important for realistic trading applications.

However, the results also show that the proposed model does not dominate all market conditions. In strongly upward-trending stocks, the Buy&Hold benchmark remains difficult to outperform, since a passive full-exposure strategy naturally benefits from persistent positive price trends. Therefore, the empirical evidence supports a more moderate conclusion: Transaction-Cost-Aware R-PPO provides overall and condition-dependent improvements over the selected baselines, rather than universal superiority across all market regimes.

In addition, the paired statistical tests provide supporting evidence for the advantage of R-PPO over PPO, A2C, and GARCH in terms of final NAV. Nevertheless, these tests are based on only eight stock-level paired observations. Although this design improves reliability compared with single-run results, the statistical findings should still be interpreted as supportive evidence rather than definitive population-level proof. Future work could further validate the model using a larger stock universe, longer testing periods, and more market regimes.
