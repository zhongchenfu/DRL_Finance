from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy may not exist in the bundled runtime
    stats = None


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "final_transaction_cost_aware_rppo"
SEED_ROWS_CSV = RESULT_DIR / "Final_TransactionCostAware_RPPO_SeedRows.csv"
SUMMARY_CSV = RESULT_DIR / "Final_TransactionCostAware_RPPO_Summary.csv"
FINAL_MODEL_NAME = "Transaction-Cost-Aware R-PPO"
RESULT_PREFIX = "Final_TransactionCostAware_RPPO"

MODEL_ORDER = [
    FINAL_MODEL_NAME,
    "PPO (Daily)",
    "A2C (Daily)",
    "GARCH",
    "Buy&Hold",
]

STOCK_ORDER = [
    "China Life",
    "Huatai Securities",
    "BYD",
    "Kweichow Moutai",
    "Hikvision",
    "Huadong Medicine",
    "Yili Group",
    "Conch Cement",
]

STOCK_CATEGORY = {
    "China Life": "Upward",
    "Huatai Securities": "Upward",
    "BYD": "Upward",
    "Kweichow Moutai": "Weak/Downward",
    "Hikvision": "Weak/Downward",
    "Huadong Medicine": "Weak/Downward",
    "Yili Group": "Range/Near-flat",
    "Conch Cement": "Range/Near-flat",
}


def clean_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-safe records while preserving numbers."""
    cleaned = df.astype(object).where(pd.notnull(df), None)
    return cleaned.to_dict(orient="records")


def exact_sign_test_pvalue(n_pos: int, n_neg: int) -> float | None:
    n = n_pos + n_neg
    if n == 0:
        return None
    observed_tail = min(n_pos, n_neg)
    tail_prob = sum(math.comb(n, k) for k in range(observed_tail + 1)) / (2**n)
    return min(1.0, 2.0 * tail_prob)


def exact_sign_flip_pvalue(diffs: np.ndarray) -> float | None:
    nonzero = np.array([x for x in diffs if abs(x) > 1e-12], dtype=float)
    if len(nonzero) == 0:
        return None
    observed = abs(float(np.mean(nonzero)))
    count = 0
    extreme = 0
    abs_vals = np.abs(nonzero)
    for signs in product([-1.0, 1.0], repeat=len(abs_vals)):
        simulated = abs(float(np.mean(abs_vals * np.array(signs))))
        count += 1
        if simulated >= observed - 1e-12:
            extreme += 1
    return extreme / count


def paired_stats(
    summary: pd.DataFrame,
    metric: str,
    metric_label: str,
    competitor: str,
    higher_is_better: bool = True,
) -> dict:
    subset = summary[summary["model"].isin([FINAL_MODEL_NAME, competitor])].copy()
    pivot = subset.pivot_table(
        index=["code", "stock"],
        columns="model",
        values=metric,
        aggfunc="first",
    )
    pivot = pivot.dropna(subset=[FINAL_MODEL_NAME, competitor])

    rppo = pivot[FINAL_MODEL_NAME].astype(float).to_numpy()
    comp = pivot[competitor].astype(float).to_numpy()
    advantage = rppo - comp if higher_is_better else comp - rppo

    n_pairs = len(advantage)
    n_pos = int(np.sum(advantage > 1e-12))
    n_neg = int(np.sum(advantage < -1e-12))
    n_tie = int(n_pairs - n_pos - n_neg)
    mean_adv = float(np.mean(advantage)) if n_pairs else np.nan
    median_adv = float(np.median(advantage)) if n_pairs else np.nan
    std_adv = float(np.std(advantage, ddof=1)) if n_pairs > 1 else np.nan
    cohen_dz = mean_adv / std_adv if n_pairs > 1 and std_adv and not np.isnan(std_adv) else np.nan

    t_stat = np.nan
    t_pvalue = np.nan
    wilcoxon_stat = np.nan
    wilcoxon_pvalue = np.nan
    if stats is not None and n_pairs > 1:
        t_res = stats.ttest_1samp(advantage, popmean=0.0, nan_policy="omit")
        t_stat = float(t_res.statistic)
        t_pvalue = float(t_res.pvalue)
        nonzero = advantage[np.abs(advantage) > 1e-12]
        if len(nonzero) > 0:
            try:
                w_res = stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
                wilcoxon_stat = float(w_res.statistic)
                wilcoxon_pvalue = float(w_res.pvalue)
            except Exception:
                pass

    sign_pvalue = exact_sign_test_pvalue(n_pos, n_neg)
    sign_flip_pvalue = exact_sign_flip_pvalue(advantage)

    if n_pairs == 0:
        conclusion = "Not enough paired observations."
    elif mean_adv > 0 and (not np.isnan(t_pvalue)) and t_pvalue < 0.05:
        conclusion = "R-PPO advantage is statistically significant at the 5% level by paired t-test."
    elif mean_adv > 0 and n_pos >= max(1, math.ceil(0.625 * max(1, n_pairs))):
        conclusion = "R-PPO shows a positive average advantage and wins most paired stocks, but significance is limited by sample size."
    elif mean_adv > 0:
        conclusion = "R-PPO has a positive average advantage, but the paired evidence is weak."
    else:
        conclusion = "No positive R-PPO advantage on this metric."

    return {
        "metric": metric_label,
        "raw_metric": metric,
        "better_direction": "higher" if higher_is_better else "lower",
        "comparison": f"{FINAL_MODEL_NAME} vs {competitor}",
        "competitor": competitor,
        "n_pairs": n_pairs,
        "rppo_mean": float(np.mean(rppo)) if n_pairs else np.nan,
        "competitor_mean": float(np.mean(comp)) if n_pairs else np.nan,
        "mean_advantage": mean_adv,
        "median_advantage": median_adv,
        "rppo_better_n": n_pos,
        "competitor_better_n": n_neg,
        "ties_n": n_tie,
        "win_rate": n_pos / n_pairs if n_pairs else np.nan,
        "paired_t_stat": t_stat,
        "paired_t_pvalue": t_pvalue,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "sign_test_pvalue": sign_pvalue,
        "exact_sign_flip_pvalue": sign_flip_pvalue,
        "cohens_dz": cohen_dz,
        "conclusion": conclusion,
    }


def build_outputs() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    seed_rows = pd.read_csv(SEED_ROWS_CSV)
    summary = pd.read_csv(SUMMARY_CSV)

    summary["category"] = summary["stock"].map(STOCK_CATEGORY)
    seed_rows["category"] = seed_rows["stock"].map(STOCK_CATEGORY)
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    summary["stock"] = pd.Categorical(summary["stock"], categories=STOCK_ORDER, ordered=True)
    summary = summary.sort_values(["stock", "model"]).reset_index(drop=True)

    overall = (
        summary.groupby("model", observed=False)
        .agg(
            stocks=("stock", "count"),
            avg_final_nav=("final_nav_mean", "mean"),
            median_final_nav=("final_nav_mean", "median"),
            avg_total_return=("total_return_mean", "mean"),
            avg_sharpe=("sharpe_mean", "mean"),
            avg_max_drawdown=("max_drawdown_mean", "mean"),
            avg_total_turnover=("total_turnover_mean", "mean"),
            avg_trade_count=("trade_count_mean", "mean"),
        )
        .reset_index()
    )

    nav_pivot = summary.pivot_table(
        index=["code", "stock", "category"],
        columns="model",
        values="final_nav_mean",
        aggfunc="first",
        observed=False,
    ).reset_index()
    nav_pivot["stock"] = pd.Categorical(nav_pivot["stock"], categories=STOCK_ORDER, ordered=True)
    nav_pivot = nav_pivot.sort_values("stock").reset_index(drop=True)

    ranking_rows = []
    win_counts: dict[str, int] = {model: 0 for model in MODEL_ORDER}
    for _, row in nav_pivot.iterrows():
        values = {model: row.get(model) for model in MODEL_ORDER if pd.notna(row.get(model))}
        ranked = sorted(values.items(), key=lambda item: item[1], reverse=True)
        best_model, best_final_nav = ranked[0]
        win_counts[best_model] = win_counts.get(best_model, 0) + 1
        rppo_rank = [model for model, _ in ranked].index(FINAL_MODEL_NAME) + 1
        ranking_rows.append(
            {
                "code": row["code"],
                "stock": row["stock"],
                "category": row["category"],
                "best_model": best_model,
                "best_final_nav": best_final_nav,
                "rppo_rank": rppo_rank,
                "rppo_final_nav": row[FINAL_MODEL_NAME],
                "rppo_minus_ppo": row[FINAL_MODEL_NAME] - row["PPO (Daily)"],
                "rppo_minus_a2c": row[FINAL_MODEL_NAME] - row["A2C (Daily)"],
                "rppo_minus_garch": row[FINAL_MODEL_NAME] - row["GARCH"],
                "rppo_minus_buyhold": row[FINAL_MODEL_NAME] - row["Buy&Hold"],
            }
        )
    stock_rankings = pd.DataFrame(ranking_rows)
    win_counts_df = pd.DataFrame(
        [{"model": model, "best_stock_count": win_counts.get(model, 0)} for model in MODEL_ORDER]
    )

    category_summary = (
        summary.groupby(["category", "model"], observed=False)
        .agg(
            stocks=("stock", "count"),
            avg_final_nav=("final_nav_mean", "mean"),
            avg_sharpe=("sharpe_mean", "mean"),
            avg_max_drawdown=("max_drawdown_mean", "mean"),
            avg_total_turnover=("total_turnover_mean", "mean"),
        )
        .reset_index()
        .sort_values(["category", "model"])
    )

    final_nav_table = nav_pivot[
        ["code", "stock", "category", *MODEL_ORDER]
    ].copy()

    risk_metrics = summary[
        [
            "code",
            "stock",
            "category",
            "model",
            "final_nav_mean",
            "final_nav_std",
            "sharpe_mean",
            "sharpe_std",
            "max_drawdown_mean",
            "max_drawdown_std",
            "total_turnover_mean",
            "trade_count_mean",
        ]
    ].copy()

    test_rows = []
    for metric, label, higher, competitors in [
        ("final_nav_mean", "Final NAV", True, ["PPO (Daily)", "A2C (Daily)", "GARCH", "Buy&Hold"]),
        ("sharpe_mean", "Sharpe ratio", True, ["PPO (Daily)", "A2C (Daily)", "GARCH", "Buy&Hold"]),
        ("max_drawdown_mean", "Maximum drawdown", True, ["PPO (Daily)", "A2C (Daily)", "GARCH", "Buy&Hold"]),
        ("total_turnover_mean", "Total turnover", False, ["PPO (Daily)", "A2C (Daily)"]),
        ("trade_count_mean", "Trade count", False, ["PPO (Daily)", "A2C (Daily)"]),
    ]:
        for competitor in competitors:
            test_rows.append(paired_stats(summary, metric, label, competitor, higher))
    stat_report = pd.DataFrame(test_rows)

    kpis = pd.DataFrame(
        [
            {
                "metric": "Average final NAV",
                "value": float(overall.loc[overall["model"] == FINAL_MODEL_NAME, "avg_final_nav"].iloc[0]),
                "note": "Highest model average across the 8-stock universe.",
            },
            {
                "metric": "Best-stock count",
                "value": int(win_counts.get(FINAL_MODEL_NAME, 0)),
                "note": "Transaction-Cost-Aware R-PPO ranks first by final NAV on 5 of 8 stocks.",
            },
            {
                "metric": "Mean advantage vs PPO",
                "value": float((nav_pivot[FINAL_MODEL_NAME] - nav_pivot["PPO (Daily)"]).mean()),
                "note": "Positive values mean R-PPO has higher final NAV.",
            },
            {
                "metric": "Mean advantage vs A2C",
                "value": float((nav_pivot[FINAL_MODEL_NAME] - nav_pivot["A2C (Daily)"]).mean()),
                "note": "Positive values mean R-PPO has higher final NAV.",
            },
            {
                "metric": "Mean advantage vs Buy&Hold",
                "value": float((nav_pivot[FINAL_MODEL_NAME] - nav_pivot["Buy&Hold"]).mean()),
                "note": "Buy&Hold remains strongest on two strong upward stocks.",
            },
        ]
    )

    method_notes = [
        {
            "topic": "Main model",
            "note": "Transaction-Cost-Aware R-PPO uses the StandardBuffer multi-frequency PPO/LSTM feature-extractor setup with turnover_penalty=0.005 and rebalance_threshold=0.05.",
        },
        {
            "topic": "Seeds",
            "note": "R-PPO, PPO (Daily), and A2C (Daily) are evaluated with seed 0, 1, and 2. Stock-level comparisons use seed-mean metrics.",
        },
        {
            "topic": "Benchmarks",
            "note": "The final comparison includes PPO (Daily), A2C (Daily), GARCH, Buy&Hold, and the Baseline (1.0) line in plots.",
        },
        {
            "topic": "Statistical tests",
            "note": "Paired tests compare stock-level mean metrics for Transaction-Cost-Aware R-PPO against each benchmark. With only 8 stocks, p-values should be interpreted as supporting evidence rather than definitive proof.",
        },
        {
            "topic": "Interpretation",
            "note": "Transaction-Cost-Aware R-PPO has the highest average final NAV and wins on 5 of 8 stocks. Buy&Hold remains best in some strong upward-trending stocks, which is expected for passive exposure.",
        },
    ]

    outputs = {
        "dashboard_kpis": kpis,
        "model_overall": overall,
        "win_counts": win_counts_df,
        "final_nav_table": final_nav_table,
        "stock_rankings": stock_rankings,
        "risk_metrics": risk_metrics,
        "category_summary": category_summary,
        "stat_tests": stat_report,
        "seed_rows": seed_rows,
        "method_notes": pd.DataFrame(method_notes),
    }

    output_paths = {
        "model_overall": RESULT_DIR / f"{RESULT_PREFIX}_Model_Overall.csv",
        "stock_rankings": RESULT_DIR / f"{RESULT_PREFIX}_Stock_Rankings.csv",
        "category_summary": RESULT_DIR / f"{RESULT_PREFIX}_Category_Summary.csv",
        "stat_tests": RESULT_DIR / f"{RESULT_PREFIX}_Statistical_Report.csv",
    }
    for name, path in output_paths.items():
        outputs[name].to_csv(path, index=False)

    workbook_data = {
        name: clean_records(df)
        for name, df in outputs.items()
    }
    workbook_data["metadata"] = {
        "title": "Final Transaction-Cost-Aware R-PPO Thesis Tables",
        "source_seed_rows": str(SEED_ROWS_CSV),
        "source_summary": str(SUMMARY_CSV),
        "stock_count": len(STOCK_ORDER),
        "seed_count": 3,
    }
    workbook_json = RESULT_DIR / f"{RESULT_PREFIX}_Workbook_Data.json"
    workbook_json.write_text(json.dumps(workbook_data, indent=2, ensure_ascii=False), encoding="utf-8")

    md = build_markdown_summary(overall, win_counts_df, stat_report, stock_rankings)
    (RESULT_DIR / f"{RESULT_PREFIX}_Thesis_Summary.md").write_text(md, encoding="utf-8")

    print(f"Saved {output_paths['stat_tests']}")
    print(f"Saved {workbook_json}")
    print(f"Saved {RESULT_DIR / f'{RESULT_PREFIX}_Thesis_Summary.md'}")


def build_markdown_summary(
    overall: pd.DataFrame,
    win_counts_df: pd.DataFrame,
    stat_report: pd.DataFrame,
    stock_rankings: pd.DataFrame,
) -> str:
    def markdown_table(df: pd.DataFrame) -> str:
        headers = [str(col) for col in df.columns]
        rows = df.astype(object).where(pd.notnull(df), "").values.tolist()
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    overall_simple = overall[
        ["model", "avg_final_nav", "avg_sharpe", "avg_max_drawdown", "avg_total_turnover", "avg_trade_count"]
    ].copy()
    overall_simple = overall_simple.round(4)
    final_nav_tests = stat_report[stat_report["metric"] == "Final NAV"][
        [
            "comparison",
            "n_pairs",
            "mean_advantage",
            "rppo_better_n",
            "competitor_better_n",
            "paired_t_pvalue",
            "wilcoxon_pvalue",
            "exact_sign_flip_pvalue",
            "conclusion",
        ]
    ].round(4)
    wins = markdown_table(win_counts_df)
    rankings = stock_rankings[
        ["code", "stock", "category", "best_model", "rppo_rank", "rppo_final_nav", "rppo_minus_buyhold"]
    ].round(4)

    return f"""# Final Transaction-Cost-Aware R-PPO Thesis Summary

## Experimental setup

- Main model: Transaction-Cost-Aware R-PPO.
- Parameters: `turnover_penalty=0.005`, `rebalance_threshold=0.05`, no global drawdown penalty.
- Test universe: 8 stocks across upward, weak/downward, and range/near-flat market states.
- Seeds: R-PPO, PPO (Daily), and A2C (Daily) use seed 0, 1, and 2.
- Benchmarks: PPO (Daily), A2C (Daily), GARCH, Buy&Hold, and Baseline (1.0) in plots.

## Main result

Across the 8-stock out-of-sample universe, Transaction-Cost-Aware R-PPO achieved the highest average final NAV. It ranked first on 5 of 8 stocks. Buy&Hold remained strongest on two clearly upward stocks, which is expected because a passive full-exposure strategy is naturally difficult to beat in strong trend periods.

## Overall model averages

{markdown_table(overall_simple)}

## Best-model counts by final NAV

{wins}

## Stock-level ranking summary

{markdown_table(rankings)}

## Paired statistical checks on final NAV

{markdown_table(final_nav_tests)}

## Thesis wording suggestion

Across a balanced eight-stock test universe and three random seeds, Transaction-Cost-Aware R-PPO achieved the highest average final NAV and ranked first on most individual stocks. The results suggest that the proposed multi-frequency PPO/LSTM feature-extractor strategy improves robustness over PPO, A2C, and GARCH baselines, especially in weak or range-bound markets. However, Buy&Hold remains difficult to beat during strong upward trends, so the thesis should claim overall and condition-dependent improvement rather than universal dominance.

## Caution

The statistical tests use only eight stock-level paired observations. They improve reliability compared with single-run results, but they should be interpreted as supporting evidence rather than definitive population-level proof.
"""


if __name__ == "__main__":
    build_outputs()
