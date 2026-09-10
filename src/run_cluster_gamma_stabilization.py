"""Stabilize the cluster contribution using persisted alpha=0.50 OOT predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WITHOUT_PATH = ROOT / "outputs" / "catboost_expanding_oot_row_predictions.parquet"
WITH_PATH = ROOT / "outputs" / "catboost_cluster_v2_expanding_oot_row_predictions.parquet"
SUMMARY_PATH = ROOT / "outputs" / "catboost_cluster_gamma_summary.csv"
BY_CUTOFF_PATH = ROOT / "outputs" / "catboost_cluster_gamma_by_cutoff.csv"
EXPANDING_PATH = ROOT / "outputs" / "catboost_expanding_gamma_selection.csv"
LAST_THREE_PATH = ROOT / "outputs" / "catboost_cluster_gamma_last_three.csv"
BLIND_CUTOFF = pd.Timestamp("2025-10-22")
ALPHA = 0.50
GAMMAS = [0.00, 0.25, 0.50, 0.75, 1.00]
LAST_THREE = pd.to_datetime(["2025-07-22", "2025-08-22", "2025-09-22"])


def wape(actual: pd.Series, prediction: pd.Series) -> float:
    actual_values = np.asarray(actual, dtype=float)
    prediction_values = np.asarray(prediction, dtype=float)
    return float(np.abs(actual_values - prediction_values).sum() / actual_values.sum())


def load_predictions() -> pd.DataFrame:
    without = pd.read_parquet(WITHOUT_PATH)
    with_cluster = pd.read_parquet(WITH_PATH)
    keys = ["cutoff", "city_code", "checkin"]
    assert without["cutoff"].max() < BLIND_CUTOFF
    assert with_cluster["cutoff"].max() < BLIND_CUTOFF
    base_columns = keys + [
        "observed_demand", "actual_final_demand", "pred_structural_v1",
        "predicted_residual",
    ]
    data = without[base_columns].merge(
        with_cluster[keys + ["predicted_residual"]],
        on=keys, how="inner", validate="one_to_one",
        suffixes=("_without", "_with"),
    )
    assert len(data) == 13 * 321 * 30
    assert data["cutoff"].nunique() == 13
    data["pred_without_cluster"] = np.maximum(
        data["observed_demand"],
        np.maximum(
            0.0,
            data["pred_structural_v1"] + ALPHA * data["predicted_residual_without"],
        ),
    )
    data["pred_with_cluster"] = np.maximum(
        data["observed_demand"],
        np.maximum(
            0.0,
            data["pred_structural_v1"] + ALPHA * data["predicted_residual_with"],
        ),
    )
    data["pred_structural"] = np.maximum(
        data["observed_demand"], np.maximum(0.0, data["pred_structural_v1"])
    )
    data["cluster_delta"] = data["pred_with_cluster"] - data["pred_without_cluster"]
    for gamma in GAMMAS:
        raw = data["pred_without_cluster"] + gamma * data["cluster_delta"]
        data[f"pred_gamma_{gamma:.2f}"] = np.maximum(
            data["observed_demand"], np.maximum(0.0, raw)
        )
    assert np.allclose(data["pred_gamma_0.00"], data["pred_without_cluster"])
    assert np.allclose(data["pred_gamma_1.00"], data["pred_with_cluster"])
    return data


def score_gammas(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cutoff, group in data.groupby("cutoff", sort=True):
        row = {
            "cutoff": cutoff,
            "Structural_V1_WAPE": wape(group["actual_final_demand"], group["pred_structural"]),
            "no_cluster_WAPE": wape(group["actual_final_demand"], group["pred_without_cluster"]),
        }
        for gamma in GAMMAS:
            row[f"gamma_{gamma:.2f}_WAPE"] = wape(
                group["actual_final_demand"], group[f"pred_gamma_{gamma:.2f}"]
            )
        rows.append(row)
    by_cutoff = pd.DataFrame(rows)

    summary_rows = []
    baseline = by_cutoff["no_cluster_WAPE"]
    structural = by_cutoff["Structural_V1_WAPE"]
    for gamma in GAMMAS:
        column = f"gamma_{gamma:.2f}_WAPE"
        contribution = baseline - by_cutoff[column]
        summary_rows.append({
            "gamma": gamma,
            "pooled_WAPE": wape(data["actual_final_demand"], data[f"pred_gamma_{gamma:.2f}"]),
            "mean_cutoff_WAPE": by_cutoff[column].mean(),
            "median_cutoff_WAPE": by_cutoff[column].median(),
            "wins_vs_no_cluster": int((contribution > 0).sum()),
            "wins_vs_Structural_V1": int((by_cutoff[column] < structural).sum()),
            "worst_degradation_vs_no_cluster": float((-contribution).max()),
            "worst_degradation_cutoff": by_cutoff.loc[contribution.idxmin(), "cutoff"],
            "best_improvement_vs_no_cluster": float(contribution.max()),
            "best_improvement_cutoff": by_cutoff.loc[contribution.idxmax(), "cutoff"],
        })
    return pd.DataFrame(summary_rows), by_cutoff


def expanding_selection(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    cutoffs = sorted(data["cutoff"].unique())
    rows = []
    selected_parts = []
    # "Each later cutoff": the first fold supplies history, selection starts on fold two.
    for test_index in range(1, len(cutoffs)):
        prior_cutoffs = cutoffs[:test_index]
        test_cutoff = pd.Timestamp(cutoffs[test_index])
        prior = data.loc[data["cutoff"].isin(prior_cutoffs)]
        prior_scores = {
            gamma: wape(prior["actual_final_demand"], prior[f"pred_gamma_{gamma:.2f}"])
            for gamma in GAMMAS
        }
        selected_gamma = min(GAMMAS, key=lambda gamma: (prior_scores[gamma], gamma))
        test = data.loc[data["cutoff"] == test_cutoff].copy()
        test["pred_selected_gamma"] = test[f"pred_gamma_{selected_gamma:.2f}"]
        no_cluster_score = wape(test["actual_final_demand"], test["pred_without_cluster"])
        selected_score = wape(test["actual_final_demand"], test["pred_selected_gamma"])
        rows.append({
            "cutoff": test_cutoff,
            "prior_oot_cutoffs": len(prior_cutoffs),
            "selected_gamma": selected_gamma,
            "prior_pooled_WAPE": prior_scores[selected_gamma],
            "no_cluster_WAPE": no_cluster_score,
            "selected_gamma_WAPE": selected_score,
            "improvement": no_cluster_score - selected_score,
        })
        selected_parts.append(test)
    results = pd.DataFrame(rows)
    pooled = pd.concat(selected_parts, ignore_index=True)
    improvements = results["improvement"]
    summary = {
        "no_cluster_pooled_WAPE": wape(pooled["actual_final_demand"], pooled["pred_without_cluster"]),
        "selected_gamma_pooled_WAPE": wape(pooled["actual_final_demand"], pooled["pred_selected_gamma"]),
        "wins": int((improvements > 0).sum()),
        "losses": int((improvements < 0).sum()),
        "ties": int((improvements == 0).sum()),
    }
    return results, summary


def main() -> None:
    data = load_predictions()
    summary, by_cutoff = score_gammas(data)
    expanding, expanding_summary = expanding_selection(data)
    last_three = by_cutoff.loc[by_cutoff["cutoff"].isin(LAST_THREE)].copy()
    last_three_pooled = {"scope": "last_three_pooled"}
    last_three_rows = data.loc[data["cutoff"].isin(LAST_THREE)]
    last_three_pooled["Structural_V1_WAPE"] = wape(
        last_three_rows["actual_final_demand"], last_three_rows["pred_structural"]
    )
    last_three_pooled["no_cluster_WAPE"] = wape(
        last_three_rows["actual_final_demand"], last_three_rows["pred_without_cluster"]
    )
    for gamma in GAMMAS:
        last_three_pooled[f"gamma_{gamma:.2f}_WAPE"] = wape(
            last_three_rows["actual_final_demand"], last_three_rows[f"pred_gamma_{gamma:.2f}"]
        )

    summary.to_csv(SUMMARY_PATH, index=False)
    by_cutoff.to_csv(BY_CUTOFF_PATH, index=False)
    expanding.to_csv(EXPANDING_PATH, index=False)
    pd.concat([last_three, pd.DataFrame([last_three_pooled])], ignore_index=True).to_csv(
        LAST_THREE_PATH, index=False
    )

    print("GAMMA SUMMARY")
    print(summary.to_string(index=False, formatters={
        "pooled_WAPE": "{:.4%}".format,
        "mean_cutoff_WAPE": "{:.4%}".format,
        "median_cutoff_WAPE": "{:.4%}".format,
        "worst_degradation_vs_no_cluster": "{:.4%}".format,
        "best_improvement_vs_no_cluster": "{:.4%}".format,
    }))
    print("\nWAPE BY CUTOFF")
    print(by_cutoff.to_string(index=False, formatters={
        column: "{:.4%}".format for column in by_cutoff.columns if column.endswith("WAPE")
    }))
    print("\nLAST THREE")
    print(last_three.to_string(index=False, formatters={
        column: "{:.4%}".format for column in last_three.columns if column.endswith("WAPE")
    }))
    print("last-three pooled:", {
        key: f"{value:.4%}" if key != "scope" else value for key, value in last_three_pooled.items()
    })
    print("\nEXPANDING GAMMA SELECTION")
    print(expanding.to_string(index=False, formatters={
        "prior_pooled_WAPE": "{:.4%}".format,
        "no_cluster_WAPE": "{:.4%}".format,
        "selected_gamma_WAPE": "{:.4%}".format,
        "improvement": "{:+.4%}".format,
    }))
    absolute = (
        expanding_summary["no_cluster_pooled_WAPE"]
        - expanding_summary["selected_gamma_pooled_WAPE"]
    )
    print("\nEXPANDING POOLED")
    print(f"No cluster: {expanding_summary['no_cluster_pooled_WAPE']:.4%}")
    print(f"Selected gamma: {expanding_summary['selected_gamma_pooled_WAPE']:.4%}")
    print(f"Improvement: {absolute:+.4%}")
    print(
        f"Wins/losses/ties: {expanding_summary['wins']}/"
        f"{expanding_summary['losses']}/{expanding_summary['ties']}"
    )


if __name__ == "__main__":
    main()
