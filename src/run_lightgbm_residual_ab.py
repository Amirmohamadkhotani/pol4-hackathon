"""Controlled LightGBM replacement for the 92-feature residual pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_residual_baseline import feature_columns, wape


DATA_PATH = ROOT / "data" / "processed" / "ml_development_snapshots.parquet"
CATBOOST_PATH = ROOT / "outputs" / "catboost_expanding_oot_row_predictions.parquet"
ROW_OUTPUT_PATH = ROOT / "outputs" / "lightgbm_expanding_oot_row_predictions.parquet"
SUMMARY_PATH = ROOT / "outputs" / "lightgbm_alpha_summary.csv"
BY_CUTOFF_PATH = ROOT / "outputs" / "lightgbm_alpha_by_cutoff.csv"
EXPANDING_PATH = ROOT / "outputs" / "lightgbm_expanding_alpha_selection.csv"
TIMING_PATH = ROOT / "outputs" / "lightgbm_fold_training_times.csv"
BLIND_CUTOFF = pd.Timestamp("2025-10-22")
ALPHAS = [0.25, 0.50, 0.75, 1.00]


def prepare_fold_categories(
    train: pd.DataFrame, test: pd.DataFrame, categorical: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit category vocabularies on training rows only."""
    train = train.copy()
    test = test.copy()
    for column in categorical:
        categories = pd.Index(train[column].astype(str).unique())
        train[column] = pd.Categorical(train[column].astype(str), categories=categories)
        test[column] = pd.Categorical(test[column].astype(str), categories=categories)
    return train, test


def train_lightgbm() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_parquet(DATA_PATH)
    assert data["cutoff"].max() < BLIND_CUTOFF
    assert "cluster_v2" not in data.columns
    features, categorical = feature_columns(data)
    assert len(features) == 92
    cutoffs = sorted(data["cutoff"].unique())
    parts = []
    timing_rows = []
    for test_index in range(6, len(cutoffs)):
        test_cutoff = pd.Timestamp(cutoffs[test_index])
        train = data.loc[data["cutoff"].isin(cutoffs[:test_index])].copy()
        test = data.loc[data["cutoff"] == test_cutoff].copy()
        train_features, test_features = prepare_fold_categories(
            train[features], test[features], categorical
        )
        target = train["actual_final_demand"] - train["pred_structural_v1"]
        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )
        started = perf_counter()
        model.fit(train_features, target, categorical_feature=categorical)
        elapsed = perf_counter() - started
        test["predicted_residual"] = model.predict(test_features)
        parts.append(test[[
            "cutoff", "city_code", "checkin", "observed_demand",
            "actual_final_demand", "pred_structural_v1", "predicted_residual",
        ]])
        timing_rows.append({
            "cutoff": test_cutoff,
            "train_cutoffs": test_index,
            "training_rows": len(train),
            "training_seconds": elapsed,
        })
        print(
            f"LightGBM fold {test_index - 5:02d}/13: {test_cutoff.date()} "
            f"({elapsed:.2f}s)", flush=True,
        )
    predictions = pd.concat(parts, ignore_index=True)
    timings = pd.DataFrame(timing_rows)
    assert len(predictions) == 13 * 321 * 30
    assert predictions["cutoff"].max() < BLIND_CUTOFF
    predictions.to_parquet(ROW_OUTPUT_PATH, index=False)
    timings.to_csv(TIMING_PATH, index=False)
    return predictions, timings


def add_predictions(lightgbm: pd.DataFrame) -> pd.DataFrame:
    keys = ["cutoff", "city_code", "checkin"]
    catboost = pd.read_parquet(CATBOOST_PATH)
    assert catboost["cutoff"].max() < BLIND_CUTOFF
    data = lightgbm.merge(
        catboost[keys + ["predicted_residual"]],
        on=keys, how="inner", validate="one_to_one",
        suffixes=("_lightgbm", "_catboost"),
    )
    assert len(data) == 13 * 321 * 30
    data["pred_structural"] = np.maximum(
        data["observed_demand"], np.maximum(0.0, data["pred_structural_v1"])
    )
    data["pred_catboost_a050"] = np.maximum(
        data["observed_demand"],
        np.maximum(
            0.0,
            data["pred_structural_v1"] + 0.50 * data["predicted_residual_catboost"],
        ),
    )
    for alpha in ALPHAS:
        data[f"pred_lightgbm_a{alpha:.2f}"] = np.maximum(
            data["observed_demand"],
            np.maximum(
                0.0,
                data["pred_structural_v1"]
                + alpha * data["predicted_residual_lightgbm"],
            ),
        )
    return data


def score(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cutoff, group in data.groupby("cutoff", sort=True):
        row = {
            "cutoff": cutoff,
            "Structural_V1_WAPE": wape(group["actual_final_demand"], group["pred_structural"]),
            "CatBoost_a050_WAPE": wape(group["actual_final_demand"], group["pred_catboost_a050"]),
        }
        for alpha in ALPHAS:
            row[f"LightGBM_a{alpha:.2f}_WAPE"] = wape(
                group["actual_final_demand"], group[f"pred_lightgbm_a{alpha:.2f}"]
            )
        rows.append(row)
    by_cutoff = pd.DataFrame(rows)
    summaries = []
    for alpha in ALPHAS:
        column = f"LightGBM_a{alpha:.2f}_WAPE"
        contribution = by_cutoff["CatBoost_a050_WAPE"] - by_cutoff[column]
        summaries.append({
            "alpha": alpha,
            "pooled_WAPE": wape(
                data["actual_final_demand"], data[f"pred_lightgbm_a{alpha:.2f}"]
            ),
            "mean_cutoff_WAPE": by_cutoff[column].mean(),
            "median_cutoff_WAPE": by_cutoff[column].median(),
            "wins_vs_Structural_V1": int((by_cutoff[column] < by_cutoff["Structural_V1_WAPE"]).sum()),
            "wins_vs_CatBoost_a050": int((contribution > 0).sum()),
            "worst_degradation_vs_CatBoost": float((-contribution).max()),
            "worst_degradation_cutoff": by_cutoff.loc[contribution.idxmin(), "cutoff"],
            "best_improvement_vs_CatBoost": float(contribution.max()),
            "best_improvement_cutoff": by_cutoff.loc[contribution.idxmax(), "cutoff"],
        })
    return pd.DataFrame(summaries), by_cutoff


def expanding_alpha_selection(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    cutoffs = sorted(data["cutoff"].unique())
    rows = []
    selected_parts = []
    for test_index in range(1, len(cutoffs)):
        prior = data.loc[data["cutoff"].isin(cutoffs[:test_index])]
        scores = {
            alpha: wape(prior["actual_final_demand"], prior[f"pred_lightgbm_a{alpha:.2f}"])
            for alpha in ALPHAS
        }
        selected_alpha = min(ALPHAS, key=lambda alpha: (scores[alpha], alpha))
        test_cutoff = pd.Timestamp(cutoffs[test_index])
        test = data.loc[data["cutoff"] == test_cutoff].copy()
        test["pred_selected_alpha"] = test[f"pred_lightgbm_a{selected_alpha:.2f}"]
        catboost_score = wape(test["actual_final_demand"], test["pred_catboost_a050"])
        selected_score = wape(test["actual_final_demand"], test["pred_selected_alpha"])
        rows.append({
            "cutoff": test_cutoff,
            "prior_oot_cutoffs": test_index,
            "selected_alpha": selected_alpha,
            "prior_pooled_WAPE": scores[selected_alpha],
            "CatBoost_a050_WAPE": catboost_score,
            "selected_LightGBM_WAPE": selected_score,
            "improvement_vs_CatBoost": catboost_score - selected_score,
        })
        selected_parts.append(test)
    results = pd.DataFrame(rows)
    pooled = pd.concat(selected_parts, ignore_index=True)
    improvement = results["improvement_vs_CatBoost"]
    summary = {
        "CatBoost_pooled_WAPE": wape(pooled["actual_final_demand"], pooled["pred_catboost_a050"]),
        "selected_LightGBM_pooled_WAPE": wape(
            pooled["actual_final_demand"], pooled["pred_selected_alpha"]
        ),
        "wins": int((improvement > 0).sum()),
        "losses": int((improvement < 0).sum()),
        "ties": int((improvement == 0).sum()),
    }
    return results, summary


def main() -> None:
    lightgbm, timings = train_lightgbm()
    data = add_predictions(lightgbm)
    summary, by_cutoff = score(data)
    expanding, expanding_summary = expanding_alpha_selection(data)
    summary.to_csv(SUMMARY_PATH, index=False)
    by_cutoff.to_csv(BY_CUTOFF_PATH, index=False)
    expanding.to_csv(EXPANDING_PATH, index=False)

    print("\nLIGHTGBM SUMMARY")
    print(summary.to_string(index=False, formatters={
        "pooled_WAPE": "{:.4%}".format,
        "mean_cutoff_WAPE": "{:.4%}".format,
        "median_cutoff_WAPE": "{:.4%}".format,
        "worst_degradation_vs_CatBoost": "{:.4%}".format,
        "best_improvement_vs_CatBoost": "{:.4%}".format,
    }))
    print("\nWAPE BY CUTOFF")
    print(by_cutoff.to_string(index=False, formatters={
        column: "{:.4%}".format for column in by_cutoff.columns if column.endswith("WAPE")
    }))
    print("\nTRAINING TIMES")
    print(timings.to_string(index=False, formatters={"training_seconds": "{:.2f}".format}))
    print("\nEXPANDING ALPHA SELECTION")
    print(expanding.to_string(index=False, formatters={
        "prior_pooled_WAPE": "{:.4%}".format,
        "CatBoost_a050_WAPE": "{:.4%}".format,
        "selected_LightGBM_WAPE": "{:.4%}".format,
        "improvement_vs_CatBoost": "{:+.4%}".format,
    }))
    absolute = (
        expanding_summary["CatBoost_pooled_WAPE"]
        - expanding_summary["selected_LightGBM_pooled_WAPE"]
    )
    print("\nEXPANDING POOLED")
    print(f"CatBoost a=0.50: {expanding_summary['CatBoost_pooled_WAPE']:.4%}")
    print(f"Selected LightGBM: {expanding_summary['selected_LightGBM_pooled_WAPE']:.4%}")
    print(f"Improvement: {absolute:+.4%}")
    print(
        f"Wins/losses/ties: {expanding_summary['wins']}/"
        f"{expanding_summary['losses']}/{expanding_summary['ties']}"
    )


if __name__ == "__main__":
    main()
