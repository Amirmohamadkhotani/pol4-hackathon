"""Build the 19-cutoff development table and run expanding CatBoost OOT."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.structural_v1 import FINAL_BLIND_CUTOFF, build_structural_v1


DEVELOPMENT_CUTOFFS = pd.to_datetime([
    "2024-03-19", "2024-04-19", "2024-05-20", "2024-06-20",
    "2024-07-21", "2024-08-21", "2024-09-21", "2024-10-21",
    "2024-11-20", "2024-12-20", "2025-01-19", "2025-02-18",
    "2025-03-20", "2025-04-20", "2025-05-21", "2025-06-21",
    "2025-07-22", "2025-08-22", "2025-09-22",
])
OUTPUT_PATH = ROOT / "data" / "processed" / "ml_development_snapshots.parquet"
RESULTS_PATH = ROOT / "outputs" / "catboost_expanding_oot_results.csv"
ROW_PREDICTIONS_PATH = ROOT / "outputs" / "catboost_expanding_oot_row_predictions.parquet"

IDENTIFIER_COLUMNS = {"cutoff", "city_code", "checkin"}
TARGET_COLUMNS = {"actual_final_demand"}
AUDIT_COLUMNS = {
    "feature_max_log_date", "completion_history_max_checkin",
    "rolling_window_end", "calendar_history_max_checkin",
}
NON_FEATURE_COLUMNS = IDENTIFIER_COLUMNS | TARGET_COLUMNS | AUDIT_COLUMNS


def wape(actual: pd.Series, predicted: pd.Series) -> float:
    denominator = float(np.asarray(actual, dtype=float).sum())
    return float(np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)).sum() / denominator)


def leakage_assertions(data: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    if FINAL_BLIND_CUTOFF in set(data["cutoff"]):
        failures.append("Final blind cutoff is present.")
    expected = 321 * 30
    counts = data.groupby("cutoff").size()
    if not (counts == expected).all():
        failures.append(f"Rows per cutoff are not all {expected}: {counts[counts != expected].to_dict()}")
    if data["cutoff"].nunique() != 19:
        failures.append(f"Expected 19 cutoffs, found {data['cutoff'].nunique()}.")
    if not (data["checkin"] > data["cutoff"]).all() or not (
        data["checkin"] <= data["cutoff"] + pd.Timedelta(days=30)
    ).all():
        failures.append("Forecast check-in dates are outside D1-D30.")
    for column in AUDIT_COLUMNS:
        if not (data[column] <= data["cutoff"]).all():
            failures.append(f"{column} exceeds cutoff.")
    if not (data["rolling_window_end"] == data["cutoff"]).all():
        failures.append("At least one rolling window does not end at cutoff.")
    if (data["observed_demand"] > data["actual_final_demand"] + 1e-9).any():
        failures.append("Observed demand exceeds final demand.")
    if data["pred_structural_v1"].isna().any():
        failures.append("Structural V1 contains missing predictions.")
    forbidden_feature_names = [
        column for column in feature_columns(data.copy())[0]
        if any(token in column.lower() for token in ("actual", "remaining", "residual"))
    ]
    if forbidden_feature_names:
        failures.append(f"Future-target-like columns entered ML features: {forbidden_feature_names}")
    return failures


def build_dataset() -> pd.DataFrame:
    if FINAL_BLIND_CUTOFF in DEVELOPMENT_CUTOFFS:
        raise AssertionError("Blind cutoff must never enter development cutoffs.")
    parts = []
    for index, cutoff in enumerate(DEVELOPMENT_CUTOFFS, start=1):
        print(f"Building development snapshot {index:02d}/19: {cutoff.date()}", flush=True)
        parts.append(build_structural_v1(cutoff))
    data = pd.concat(parts, ignore_index=True)
    failures = leakage_assertions(data)
    if failures:
        raise AssertionError("Leakage/assertion failures:\n- " + "\n- ".join(failures))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT_PATH, index=False)
    return data


def feature_columns(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = NON_FEATURE_COLUMNS | {
        # Structural internals that duplicate identifiers or are not model inputs.
        "calendar_bucket_x", "calendar_bucket_y",
    }
    features = [column for column in data.columns if column not in excluded]
    categorical = [
        column for column in features
        if str(data[column].dtype) in {"object", "category", "string"}
    ]
    # CatBoost requires categorical missing values to be strings.
    for column in categorical:
        data[column] = data[column].astype("string").fillna("Missing").astype(str)
    return features, categorical


def run_expanding_validation(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    features, categorical = feature_columns(data)
    ordered_cutoffs = sorted(data["cutoff"].unique())
    results = []
    predictions = []
    for test_index in range(6, len(ordered_cutoffs)):
        test_cutoff = pd.Timestamp(ordered_cutoffs[test_index])
        train_cutoffs = ordered_cutoffs[:test_index]
        train = data.loc[data["cutoff"].isin(train_cutoffs)].copy()
        test = data.loc[data["cutoff"] == test_cutoff].copy()
        train_target = train["actual_final_demand"] - train["pred_structural_v1"]
        model = CatBoostRegressor(
            loss_function="MAE", iterations=500, depth=6,
            learning_rate=0.05, random_seed=42, verbose=False,
            allow_writing_files=False,
        )
        model.fit(train[features], train_target, cat_features=categorical)
        predicted_residual = model.predict(test[features])
        test["predicted_residual"] = predicted_residual
        test["pred_catboost_raw"] = test["pred_structural_v1"] + test["predicted_residual"]
        test["pred_catboost_final"] = np.maximum(
            test["observed_demand"],
            np.maximum(0.0, test["pred_catboost_raw"]),
        )
        structural = wape(test["actual_final_demand"], test["pred_structural_v1"])
        catboost = wape(test["actual_final_demand"], test["pred_catboost_final"])
        results.append({
            "cutoff": test_cutoff,
            "train_cutoffs": len(train_cutoffs),
            "Structural_V1_WAPE": structural,
            "CatBoost_WAPE": catboost,
            "improvement_pp": (structural - catboost) * 100,
        })
        predictions.append(test[[
            "cutoff", "city_code", "checkin", "observed_demand",
            "actual_final_demand", "pred_structural_v1", "predicted_residual",
            "pred_catboost_raw", "pred_catboost_final",
        ]])
        print(
            f"OOT {test_cutoff.date()} | train={len(train_cutoffs):2d} | "
            f"Structural={structural:.4%} | CatBoost={catboost:.4%} | "
            f"improvement={100 * (structural - catboost):+.3f} pp",
            flush=True,
        )
    result_table = pd.DataFrame(results)
    pooled = pd.concat(predictions, ignore_index=True)
    if FINAL_BLIND_CUTOFF in set(pooled["cutoff"]):
        raise AssertionError("Blind cutoff entered OOT row predictions.")
    structural_pooled = wape(pooled["actual_final_demand"], pooled["pred_structural_v1"])
    catboost_pooled = wape(pooled["actual_final_demand"], pooled["pred_catboost_final"])
    summary = {
        "structural_pooled_wape": structural_pooled,
        "catboost_pooled_wape": catboost_pooled,
        "absolute_improvement": structural_pooled - catboost_pooled,
        "relative_error_reduction": (structural_pooled - catboost_pooled) / structural_pooled,
        "winning_cutoffs": int((result_table["CatBoost_WAPE"] < result_table["Structural_V1_WAPE"]).sum()),
        "heldout_cutoffs": len(result_table),
        "feature_count": len(features),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(RESULTS_PATH, index=False)
    pooled.to_parquet(ROW_PREDICTIONS_PATH, index=False)
    return result_table, summary


def main() -> None:
    data = build_dataset()
    failures = leakage_assertions(data)
    structural_by_cutoff = data.groupby("cutoff").apply(
        lambda group: wape(group["actual_final_demand"], group["pred_structural_v1"]),
        include_groups=False,
    )
    features, _ = feature_columns(data)
    print("\n1. ML DATASET SUMMARY")
    print(f"total rows: {len(data):,}")
    print(f"number of cutoffs: {data['cutoff'].nunique()}")
    print(f"min/max cutoff: {data['cutoff'].min().date()} / {data['cutoff'].max().date()}")
    print(f"feature count: {len(features)}")
    print(f"saved: {OUTPUT_PATH}")
    print("\n2. STRUCTURAL V1 DEVELOPMENT SCORE")
    print(f"pooled Structural V1 WAPE: {wape(data['actual_final_demand'], data['pred_structural_v1']):.4%}")
    print("WAPE by cutoff:")
    for cutoff, score in structural_by_cutoff.items():
        print(f"  {pd.Timestamp(cutoff).date()}: {score:.4%}")
    print("\nLeakage/assertion failures:", "none" if not failures else failures)
    print("\n3. CATBOOST EXPANDING OOT RESULTS")
    results, summary = run_expanding_validation(data)
    print(results.to_string(index=False, formatters={
        "Structural_V1_WAPE": "{:.4%}".format,
        "CatBoost_WAPE": "{:.4%}".format,
        "improvement_pp": "{:+.3f}".format,
    }))
    print("\n4. POOLED STRUCTURAL vs CATBOOST SCORE")
    print(f"Structural V1 pooled WAPE: {summary['structural_pooled_wape']:.4%}")
    print(f"CatBoost pooled WAPE:      {summary['catboost_pooled_wape']:.4%}")
    print(f"Absolute improvement:      {summary['absolute_improvement']:.4%}")
    print(f"Relative error reduction:  {summary['relative_error_reduction']:.2%}")
    print(f"Winning cutoffs:            {summary['winning_cutoffs']}/{summary['heldout_cutoffs']}")


if __name__ == "__main__":
    main()
