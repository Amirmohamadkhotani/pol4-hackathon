"""Expanding OOT D15-30 residual specialists with fixed baseline settings."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_lightgbm_residual_ab import prepare_fold_categories
from src.run_residual_baseline import feature_columns


DATA_PATH = ROOT / "data" / "processed" / "ml_development_snapshots.parquet"
CAT_GENERAL_PATH = ROOT / "outputs" / "catboost_expanding_oot_row_predictions.parquet"
LGB_GENERAL_PATH = ROOT / "outputs" / "lightgbm_expanding_oot_row_predictions.parquet"
ROW_OUTPUT = ROOT / "outputs" / "long_horizon_specialist_oot_predictions.parquet"
SUMMARY_OUTPUT = ROOT / "outputs" / "long_horizon_specialist_summary.csv"
BY_CUTOFF_OUTPUT = ROOT / "outputs" / "long_horizon_specialist_by_cutoff.csv"
LONG_METRICS_OUTPUT = ROOT / "outputs" / "long_horizon_specialist_d15_30_metrics.csv"
TRAINING_OUTPUT = ROOT / "outputs" / "long_horizon_specialist_training_rows.csv"
BLIND_CUTOFF = pd.Timestamp("2025-10-22")
ALPHAS = [0.25, 0.50, 0.75, 1.00]


def wape(actual: pd.Series, prediction: pd.Series) -> float:
    return float(np.abs(np.asarray(actual) - np.asarray(prediction)).sum() / np.asarray(actual).sum())


def train_specialists() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_parquet(DATA_PATH)
    assert data["cutoff"].max() < BLIND_CUTOFF
    assert "cluster_v2" not in data.columns
    features, categorical = feature_columns(data)
    assert len(features) == 92
    cutoffs = sorted(data["cutoff"].unique())
    parts = []
    training_rows = []
    for test_index in range(6, len(cutoffs)):
        test_cutoff = pd.Timestamp(cutoffs[test_index])
        train = data.loc[
            data["cutoff"].isin(cutoffs[:test_index])
            & data["days_to_checkin"].between(15, 30)
        ].copy()
        test = data.loc[
            (data["cutoff"] == test_cutoff)
            & data["days_to_checkin"].between(15, 30)
        ].copy()
        assert len(train) == test_index * 321 * 16
        assert len(test) == 321 * 16
        target = train["actual_final_demand"] - train["pred_structural_v1"]

        cat = CatBoostRegressor(
            loss_function="MAE", iterations=500, depth=6, learning_rate=0.05,
            random_seed=42, verbose=False, allow_writing_files=False,
        )
        started = perf_counter()
        cat.fit(train[features], target, cat_features=categorical)
        cat_seconds = perf_counter() - started
        test["predicted_residual_cat_specialist"] = cat.predict(test[features])

        lgb_train, lgb_test = prepare_fold_categories(
            train[features], test[features], categorical
        )
        lgb = LGBMRegressor(
            objective="regression_l1", n_estimators=500, learning_rate=0.05,
            num_leaves=31, max_depth=-1, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=1.0,
            random_state=42, verbosity=-1, n_jobs=-1,
        )
        started = perf_counter()
        lgb.fit(lgb_train, target, categorical_feature=categorical)
        lgb_seconds = perf_counter() - started
        test["predicted_residual_lgb_specialist"] = lgb.predict(lgb_test)

        parts.append(test[[
            "cutoff", "city_code", "checkin", "days_to_checkin",
            "observed_demand", "actual_final_demand", "pred_structural_v1",
            "predicted_residual_cat_specialist", "predicted_residual_lgb_specialist",
        ]])
        training_rows.append({
            "cutoff": test_cutoff, "train_cutoffs": test_index,
            "specialist_training_rows": len(train),
            "specialist_test_rows": len(test),
            "catboost_training_seconds": cat_seconds,
            "lightgbm_training_seconds": lgb_seconds,
        })
        print(
            f"Specialist fold {test_index - 5:02d}/13 {test_cutoff.date()} | "
            f"train={len(train):,} | Cat={cat_seconds:.2f}s LGB={lgb_seconds:.2f}s",
            flush=True,
        )
    predictions = pd.concat(parts, ignore_index=True)
    training = pd.DataFrame(training_rows)
    assert len(predictions) == 13 * 321 * 16
    assert predictions["cutoff"].max() < BLIND_CUTOFF
    predictions.to_parquet(ROW_OUTPUT, index=False)
    training.to_csv(TRAINING_OUTPUT, index=False)
    return predictions, training


def build_comparison(specialists: pd.DataFrame) -> pd.DataFrame:
    cat = pd.read_parquet(CAT_GENERAL_PATH)
    lgb = pd.read_parquet(LGB_GENERAL_PATH)
    keys = ["cutoff", "city_code", "checkin"]
    data = cat[keys + [
        "observed_demand", "actual_final_demand", "pred_structural_v1",
        "predicted_residual",
    ]].merge(
        lgb[keys + ["predicted_residual"]], on=keys, how="inner",
        validate="one_to_one", suffixes=("_cat_general", "_lgb_general"),
    ).merge(
        specialists[keys + [
            "predicted_residual_cat_specialist", "predicted_residual_lgb_specialist"
        ]],
        on=keys, how="left", validate="one_to_one",
    )
    assert len(data) == 13 * 321 * 30
    data["days_to_checkin"] = (data["checkin"] - data["cutoff"]).dt.days

    def constrained(residual: pd.Series, alpha: float) -> np.ndarray:
        return np.maximum(
            data["observed_demand"],
            np.maximum(0.0, data["pred_structural_v1"] + alpha * residual),
        )

    data["Pure CatBoost a0.50"] = constrained(data["predicted_residual_cat_general"], 0.50)
    data["general_lgb_long"] = constrained(data["predicted_residual_lgb_general"], 0.50)
    data["Current ensemble"] = np.where(
        data["days_to_checkin"] <= 14,
        data["Pure CatBoost a0.50"], data["general_lgb_long"],
    )
    long_mask = data["days_to_checkin"].between(15, 30)
    for family, residual_column in [
        ("Cat specialist", "predicted_residual_cat_specialist"),
        ("LGB specialist", "predicted_residual_lgb_specialist"),
    ]:
        for alpha in ALPHAS:
            specialist_prediction = constrained(data[residual_column], alpha)
            data[f"{family} a{alpha:.2f}"] = np.where(
                long_mask, specialist_prediction, data["Pure CatBoost a0.50"]
            )
    return data


def metrics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rules = ["Pure CatBoost a0.50", "Current ensemble"] + [
        f"{family} a{alpha:.2f}"
        for family in ["Cat specialist", "LGB specialist"] for alpha in ALPHAS
    ]
    for rule in rules:
        data[f"error::{rule}"] = data[rule] - data["actual_final_demand"]
        data[f"abs_error::{rule}"] = np.abs(data[f"error::{rule}"])
    cutoffs = sorted(data["cutoff"].unique())
    last3 = data["cutoff"].isin(cutoffs[-3:])
    last6 = data["cutoff"].isin(cutoffs[-6:])
    long = data["days_to_checkin"].between(15, 30)
    far = data["days_to_checkin"].between(22, 30)
    q90 = data.loc[data["actual_final_demand"] > 0, "actual_final_demand"].quantile(0.90)
    high_long = (data["actual_final_demand"] > q90) & long

    def subset_wape(rule: str, mask: pd.Series | None = None) -> float:
        subset = data if mask is None else data.loc[mask]
        return float(subset[f"abs_error::{rule}"].sum() / subset["actual_final_demand"].sum())

    summary_rows = []
    cutoff_rows = []
    for rule in rules:
        cutoff_scores = data.groupby("cutoff").apply(
            lambda group: float(
                group[f"abs_error::{rule}"].sum() / group["actual_final_demand"].sum()
            ), include_groups=False,
        )
        summary_rows.append({
            "rule": rule,
            "overall_pooled_WAPE": subset_wape(rule),
            "D15_30_WAPE": subset_wape(rule, long),
            "D22_30_WAPE": subset_wape(rule, far),
            "Top10_demand_x_D15_30_WAPE": subset_wape(rule, high_long),
            "last3_pooled_WAPE": subset_wape(rule, last3),
            "last6_pooled_WAPE": subset_wape(rule, last6),
            "worst_cutoff_WAPE": cutoff_scores.max(),
            "worst_cutoff": pd.Timestamp(cutoff_scores.idxmax()),
            "mean_cutoff_WAPE": cutoff_scores.mean(),
            "median_cutoff_WAPE": cutoff_scores.median(),
        })
        for cutoff, value in cutoff_scores.items():
            cutoff_rows.append({"cutoff": cutoff, "rule": rule, "WAPE": value})

    long_rows = []
    actual = data.loc[long, "actual_final_demand"]
    for rule in rules:
        prediction = data.loc[long, rule]
        signed = prediction - actual
        sse = float(np.square(signed).sum())
        sst = float(np.square(actual - actual.mean()).sum())
        long_rows.append({
            "rule": rule,
            "MAE": float(np.abs(signed).mean()),
            "RMSE": float(np.sqrt(np.square(signed).mean())),
            "mean_signed_error": float(signed.mean()),
            "normalized_bias": float(signed.sum() / actual.sum()),
            "R2": float(1.0 - sse / sst),
        })
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(cutoff_rows).pivot(index="cutoff", columns="rule", values="WAPE").reset_index(),
        pd.DataFrame(long_rows),
    )


def main() -> None:
    specialists, training = train_specialists()
    data = build_comparison(specialists)
    summary, by_cutoff, long_metrics = metrics(data)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    by_cutoff.to_csv(BY_CUTOFF_OUTPUT, index=False)
    long_metrics.to_csv(LONG_METRICS_OUTPUT, index=False)
    print("\nSUMMARY")
    print(summary.to_string(index=False, formatters={
        column: "{:.4%}".format for column in summary.columns if column.endswith("WAPE")
    }))
    print("\nD15-30 ERROR METRICS")
    print(long_metrics.to_string(index=False, formatters={
        "MAE": "{:,.3f}".format, "RMSE": "{:,.3f}".format,
        "mean_signed_error": "{:+,.3f}".format,
        "normalized_bias": "{:+.4%}".format, "R2": "{:.6f}".format,
    }))
    print("\nWAPE BY CUTOFF")
    print(by_cutoff.to_string(index=False, formatters={
        column: "{:.4%}".format for column in by_cutoff.columns if column != "cutoff"
    }))
    print("\nTRAINING ROWS")
    print(training.to_string(index=False, formatters={
        "catboost_training_seconds": "{:.2f}".format,
        "lightgbm_training_seconds": "{:.2f}".format,
    }))


if __name__ == "__main__":
    main()
