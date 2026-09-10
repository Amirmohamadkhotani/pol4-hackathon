"""Evaluate residual shrinkage using persisted CatBoost OOT predictions only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT / "outputs" / "catboost_expanding_oot_row_predictions.parquet"
ALPHA_SUMMARY_PATH = ROOT / "outputs" / "catboost_alpha_stabilization_summary.csv"
ALPHA_BY_CUTOFF_PATH = ROOT / "outputs" / "catboost_alpha_wape_by_cutoff.csv"
ALPHA_EXPANDING_PATH = ROOT / "outputs" / "catboost_expanding_alpha_selection.csv"
FINAL_BLIND_CUTOFF = pd.Timestamp("2025-10-22")
ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]
MIN_PRIOR_OOT_FOLDS = 3


def wape(actual: pd.Series, predicted: pd.Series) -> float:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    return float(np.abs(actual_values - predicted_values).sum() / actual_values.sum())


def add_alpha_predictions(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for alpha in ALPHAS:
        out[f"pred_alpha_{alpha:.2f}"] = np.maximum(
            out["observed_demand"],
            np.maximum(
                0.0,
                out["pred_structural_v1"] + alpha * out["predicted_residual"],
            ),
        )
    return out


def score_alphas(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_rows = []
    for cutoff, group in data.groupby("cutoff", sort=True):
        row = {"cutoff": cutoff}
        for alpha in ALPHAS:
            row[f"alpha_{alpha:.2f}_wape"] = wape(
                group["actual_final_demand"], group[f"pred_alpha_{alpha:.2f}"]
            )
        cutoff_rows.append(row)
    by_cutoff = pd.DataFrame(cutoff_rows)
    baseline_column = "alpha_0.00_wape"
    summary_rows = []
    for alpha in ALPHAS:
        score_column = f"alpha_{alpha:.2f}_wape"
        improvements = by_cutoff[baseline_column] - by_cutoff[score_column]
        summary_rows.append({
            "alpha": alpha,
            "pooled_wape": wape(
                data["actual_final_demand"], data[f"pred_alpha_{alpha:.2f}"]
            ),
            "mean_cutoff_wape": by_cutoff[score_column].mean(),
            "median_cutoff_wape": by_cutoff[score_column].median(),
            "winning_cutoffs_vs_structural": int((improvements > 0).sum()),
            "worst_degradation": float((-improvements).max()),
            "worst_degradation_cutoff": by_cutoff.loc[improvements.idxmin(), "cutoff"],
            "best_improvement": float(improvements.max()),
            "best_improvement_cutoff": by_cutoff.loc[improvements.idxmax(), "cutoff"],
        })
    return pd.DataFrame(summary_rows), by_cutoff


def expanding_alpha_selection(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    cutoffs = sorted(data["cutoff"].unique())
    rows = []
    selected_parts = []
    for test_index in range(MIN_PRIOR_OOT_FOLDS, len(cutoffs)):
        prior_cutoffs = cutoffs[:test_index]
        test_cutoff = pd.Timestamp(cutoffs[test_index])
        prior = data.loc[data["cutoff"].isin(prior_cutoffs)]
        prior_scores = {
            alpha: wape(prior["actual_final_demand"], prior[f"pred_alpha_{alpha:.2f}"])
            for alpha in ALPHAS
        }
        selected_alpha = min(ALPHAS, key=lambda alpha: (prior_scores[alpha], alpha))
        test = data.loc[data["cutoff"] == test_cutoff].copy()
        baseline_wape = wape(test["actual_final_demand"], test["pred_alpha_0.00"])
        selected_wape = wape(
            test["actual_final_demand"], test[f"pred_alpha_{selected_alpha:.2f}"]
        )
        test["pred_selected_alpha"] = test[f"pred_alpha_{selected_alpha:.2f}"]
        selected_parts.append(test)
        rows.append({
            "cutoff": test_cutoff,
            "prior_oot_folds": len(prior_cutoffs),
            "selected_alpha": selected_alpha,
            "prior_pooled_wape": prior_scores[selected_alpha],
            "Structural_V1_WAPE": baseline_wape,
            "selected_alpha_WAPE": selected_wape,
            "improvement": baseline_wape - selected_wape,
        })
    results = pd.DataFrame(rows)
    pooled = pd.concat(selected_parts, ignore_index=True)
    structural_pooled = wape(pooled["actual_final_demand"], pooled["pred_alpha_0.00"])
    selected_pooled = wape(pooled["actual_final_demand"], pooled["pred_selected_alpha"])
    improvements = results["improvement"]
    summary = {
        "structural_pooled_wape": structural_pooled,
        "selected_pooled_wape": selected_pooled,
        "absolute_improvement": structural_pooled - selected_pooled,
        "wins": int((improvements > 0).sum()),
        "losses": int((improvements < 0).sum()),
        "ties": int((improvements == 0).sum()),
    }
    return results, summary


def main() -> None:
    data = pd.read_parquet(PREDICTIONS_PATH)
    if FINAL_BLIND_CUTOFF in set(data["cutoff"]):
        raise AssertionError("Blind cutoff is present in persisted OOT predictions.")
    expected_rows = 13 * 321 * 30
    if len(data) != expected_rows or data["cutoff"].nunique() != 13:
        raise AssertionError(f"Expected 13 x 321 x 30 rows, found {len(data)}.")
    reproduced = wape(data["actual_final_demand"], data["pred_catboost_final"])
    if not np.isclose(reproduced, 0.193786, atol=5e-7):
        raise AssertionError(f"CatBoost reproduction failed: {reproduced:.10%}")
    data = add_alpha_predictions(data)
    summary, by_cutoff = score_alphas(data)
    expanding, expanding_summary = expanding_alpha_selection(data)
    summary.to_csv(ALPHA_SUMMARY_PATH, index=False)
    by_cutoff.to_csv(ALPHA_BY_CUTOFF_PATH, index=False)
    expanding.to_csv(ALPHA_EXPANDING_PATH, index=False)

    print(f"Persisted rows: {len(data):,}")
    print(f"CatBoost pooled WAPE reproduction: {reproduced:.6%}")
    print("\nALPHA SUMMARY")
    print(summary.to_string(index=False, formatters={
        "pooled_wape": "{:.4%}".format,
        "mean_cutoff_wape": "{:.4%}".format,
        "median_cutoff_wape": "{:.4%}".format,
        "worst_degradation": "{:.4%}".format,
        "best_improvement": "{:.4%}".format,
    }))
    print("\nWAPE BY CUTOFF")
    print(by_cutoff.to_string(index=False, formatters={
        column: "{:.4%}".format for column in by_cutoff.columns if column.endswith("_wape")
    }))
    print("\nEXPANDING TEMPORAL ALPHA SELECTION")
    print(expanding.to_string(index=False, formatters={
        "prior_pooled_wape": "{:.4%}".format,
        "Structural_V1_WAPE": "{:.4%}".format,
        "selected_alpha_WAPE": "{:.4%}".format,
        "improvement": "{:+.4%}".format,
    }))
    print("\nEXPANDING POOLED")
    print(f"Structural V1 WAPE: {expanding_summary['structural_pooled_wape']:.4%}")
    print(f"Selected-alpha WAPE: {expanding_summary['selected_pooled_wape']:.4%}")
    print(f"Improvement: {expanding_summary['absolute_improvement']:.4%}")
    print(
        f"Wins / losses / ties: {expanding_summary['wins']} / "
        f"{expanding_summary['losses']} / {expanding_summary['ties']}"
    )


if __name__ == "__main__":
    main()
