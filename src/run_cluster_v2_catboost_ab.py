"""Controlled CatBoost A/B test adding only cutoff-local cluster_v2."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clustering_v2_feature import BLIND_CUTOFF, build_cluster_v2_assignments
from src.run_residual_baseline import DEVELOPMENT_CUTOFFS, feature_columns, wape


DATA_PATH = ROOT / "data" / "processed" / "ml_development_snapshots.parquet"
ASSIGNMENTS_PATH = ROOT / "data" / "processed" / "cluster_v2_development_assignments.parquet"
NO_CLUSTER_PATH = ROOT / "outputs" / "catboost_expanding_oot_row_predictions.parquet"
WITH_CLUSTER_PATH = ROOT / "outputs" / "catboost_cluster_v2_expanding_oot_row_predictions.parquet"
RESULTS_PATH = ROOT / "outputs" / "catboost_cluster_v2_ab_results.csv"
SUMMARY_PATH = ROOT / "outputs" / "catboost_cluster_v2_ab_summary.csv"


def build_and_validate_assignments(data: pd.DataFrame) -> pd.DataFrame:
    assignments = build_cluster_v2_assignments(DEVELOPMENT_CUTOFFS)
    counts = assignments.groupby("cutoff").agg(
        rows=("city_code", "size"), cities=("city_code", "nunique"),
        missing=("cluster_v2", lambda values: values.isna().sum()),
    )
    assert (counts["rows"] == 321).all() and (counts["cities"] == 321).all()
    assert (counts["missing"] == 0).all()
    assert BLIND_CUTOFF not in set(assignments["cutoff"])
    assert (assignments["cluster_snapshot_cutoff"] == assignments["cutoff"]).all()
    assert set(assignments["cutoff"]) == set(data["cutoff"])
    assignments.to_parquet(ASSIGNMENTS_PATH, index=False)
    return assignments


def run_with_cluster(data: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    enriched = data.merge(
        assignments[["cutoff", "city_code", "cluster_v2", "cluster_snapshot_cutoff"]],
        on=["cutoff", "city_code"], how="left", validate="many_to_one",
    )
    assert enriched["cluster_v2"].notna().all()
    assert (enriched["cluster_snapshot_cutoff"] == enriched["cutoff"]).all()
    enriched["cluster_v2"] = enriched["cluster_v2"].astype(int).astype(str)
    features, categorical = feature_columns(enriched)
    # Provenance is an assertion-only field, never an ML feature.
    features.remove("cluster_snapshot_cutoff")
    assert len(features) == 93
    assert "cluster_v2" in features and "cluster_v2" in categorical

    parts = []
    cutoffs = sorted(enriched["cutoff"].unique())
    for test_index in range(6, len(cutoffs)):
        test_cutoff = pd.Timestamp(cutoffs[test_index])
        assert test_cutoff < BLIND_CUTOFF
        train = enriched.loc[enriched["cutoff"].isin(cutoffs[:test_index])]
        test = enriched.loc[enriched["cutoff"] == test_cutoff].copy()
        target = train["actual_final_demand"] - train["pred_structural_v1"]
        model = CatBoostRegressor(
            loss_function="MAE", iterations=500, depth=6, learning_rate=0.05,
            random_seed=42, verbose=False, allow_writing_files=False,
        )
        model.fit(train[features], target, cat_features=categorical)
        test["predicted_residual"] = model.predict(test[features])
        test["pred_catboost_raw"] = test["pred_structural_v1"] + test["predicted_residual"]
        test["pred_catboost_final"] = np.maximum(
            test["observed_demand"], np.maximum(0.0, test["pred_catboost_raw"])
        )
        parts.append(test[[
            "cutoff", "city_code", "checkin", "observed_demand",
            "actual_final_demand", "pred_structural_v1", "cluster_v2",
            "cluster_snapshot_cutoff", "predicted_residual", "pred_catboost_raw",
            "pred_catboost_final",
        ]])
        print(f"Cluster CatBoost fold {test_index - 5:02d}/13: {test_cutoff.date()}", flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    assert len(predictions) == 13 * 321 * 30
    assert BLIND_CUTOFF not in set(predictions["cutoff"])
    predictions.to_parquet(WITH_CLUSTER_PATH, index=False)
    return predictions


def add_alpha(data: pd.DataFrame, alpha: float, name: str) -> pd.DataFrame:
    out = data.copy()
    out[name] = np.maximum(
        out["observed_demand"],
        np.maximum(0.0, out["pred_structural_v1"] + alpha * out["predicted_residual"]),
    )
    return out


def compare(no_cluster: pd.DataFrame, with_cluster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["cutoff", "city_code", "checkin"]
    required_no = keys + [
        "observed_demand", "actual_final_demand", "pred_structural_v1", "predicted_residual"
    ]
    combined = no_cluster[required_no].merge(
        with_cluster[keys + ["predicted_residual"]], on=keys, how="inner",
        suffixes=("_without", "_with"), validate="one_to_one",
    )
    assert len(combined) == 13 * 321 * 30
    for alpha, suffix in [(1.0, "a100"), (0.5, "a050")]:
        combined[f"pred_without_{suffix}"] = np.maximum(
            combined["observed_demand"],
            np.maximum(0.0, combined["pred_structural_v1"] + alpha * combined["predicted_residual_without"]),
        )
        combined[f"pred_with_{suffix}"] = np.maximum(
            combined["observed_demand"],
            np.maximum(0.0, combined["pred_structural_v1"] + alpha * combined["predicted_residual_with"]),
        )
    combined["pred_structural_floored"] = np.maximum(
        combined["observed_demand"], np.maximum(0.0, combined["pred_structural_v1"])
    )

    rows = []
    for cutoff, group in combined.groupby("cutoff", sort=True):
        row = {
            "cutoff": cutoff,
            "Structural_V1_WAPE": wape(group["actual_final_demand"], group["pred_structural_floored"]),
        }
        for suffix in ["a100", "a050"]:
            without = wape(group["actual_final_demand"], group[f"pred_without_{suffix}"])
            with_value = wape(group["actual_final_demand"], group[f"pred_with_{suffix}"])
            row[f"without_cluster_{suffix}_WAPE"] = without
            row[f"with_cluster_{suffix}_WAPE"] = with_value
            row[f"cluster_contribution_{suffix}"] = without - with_value
        rows.append(row)
    results = pd.DataFrame(rows)

    summaries = []
    for alpha, suffix in [(1.0, "a100"), (0.5, "a050")]:
        for arm in ["without", "with"]:
            prediction = f"pred_{arm}_{suffix}"
            cutoff_column = f"{arm}_cluster_{suffix}_WAPE"
            summaries.append({
                "alpha": alpha, "arm": arm,
                "pooled_WAPE": wape(combined["actual_final_demand"], combined[prediction]),
                "mean_cutoff_WAPE": results[cutoff_column].mean(),
                "wins_vs_structural": int((results[cutoff_column] < results["Structural_V1_WAPE"]).sum()),
            })
    summary = pd.DataFrame(summaries)
    results.to_csv(RESULTS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    return results, summary


def main() -> None:
    data = pd.read_parquet(DATA_PATH)
    assert data["cutoff"].max() < BLIND_CUTOFF
    assignments = build_and_validate_assignments(data)
    with_cluster = run_with_cluster(data, assignments)
    no_cluster = pd.read_parquet(NO_CLUSTER_PATH)
    assert no_cluster["cutoff"].max() < BLIND_CUTOFF
    results, summary = compare(no_cluster, with_cluster)
    print("\nSUMMARY")
    print(summary.to_string(index=False, formatters={
        "pooled_WAPE": "{:.4%}".format, "mean_cutoff_WAPE": "{:.4%}".format,
    }))
    print("\nPER CUTOFF")
    print(results.to_string(index=False, formatters={
        column: "{:+.4%}".format if "contribution" in column else "{:.4%}".format
        for column in results.columns if column != "cutoff"
    }))
    for suffix, label in [("a100", "alpha=1.00"), ("a050", "alpha=0.50")]:
        contribution = results[f"cluster_contribution_{suffix}"]
        pooled_without = summary.loc[(summary["alpha"] == float(label[-4:])) & (summary["arm"] == "without"), "pooled_WAPE"].iloc[0]
        pooled_with = summary.loc[(summary["alpha"] == float(label[-4:])) & (summary["arm"] == "with"), "pooled_WAPE"].iloc[0]
        print(f"\nCLUSTER CONTRIBUTION {label}")
        print(f"pooled improvement: {pooled_without - pooled_with:+.4%}")
        print(f"improved cutoffs: {int((contribution > 0).sum())}/13")
        print(f"worst degradation: {-contribution.min():.4%} at {results.loc[contribution.idxmin(), 'cutoff'].date()}")
        print(f"best improvement: {contribution.max():.4%} at {results.loc[contribution.idxmax(), 'cutoff'].date()}")


if __name__ == "__main__":
    main()
