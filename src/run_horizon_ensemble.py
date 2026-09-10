"""Inference-safe fixed horizon ensembles from persisted alpha=0.50 OOT rows."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_model_segment_comparison import load_data, wape


def main() -> None:
    data = load_data()
    h = data["days_to_checkin"]
    cat = data["pred_catboost"]
    lgb = data["pred_lightgbm"]

    rules: dict[str, pd.Series] = {
        "A Cat100 long": cat.copy(),
        "B Cat75 LGB25 long": np.where(h <= 14, cat, 0.75 * cat + 0.25 * lgb),
        "C Cat50 LGB50 long": np.where(h <= 14, cat, 0.50 * cat + 0.50 * lgb),
        "D Cat25 LGB75 long": np.where(h <= 14, cat, 0.25 * cat + 0.75 * lgb),
        "E LGB100 long": np.where(h <= 14, cat, lgb),
        "Split 50/50 D15-21; LGB D22-30": np.select(
            [h <= 14, h <= 21], [cat, 0.50 * cat + 0.50 * lgb], default=lgb
        ),
    }
    for name, prediction in rules.items():
        data[f"rule::{name}"] = np.maximum(
            data["observed_demand"], np.maximum(0.0, prediction)
        )
    assert np.allclose(data["rule::A Cat100 long"], cat)
    assert all((data[f"rule::{name}"] >= data["observed_demand"]).all() for name in rules)

    positive = data.loc[data["actual_final_demand"] > 0, "actual_final_demand"]
    q90 = float(positive.quantile(0.90))
    top10_long = (data["actual_final_demand"] > q90) & h.between(15, 30)
    long = h.between(15, 30)
    far = h.between(22, 30)
    cutoffs = sorted(data["cutoff"].unique())
    last3 = data["cutoff"].isin(cutoffs[-3:])
    last6 = data["cutoff"].isin(cutoffs[-6:])

    cat_cutoff = data.groupby("cutoff").apply(lambda g: wape(g, "CatBoost"), include_groups=False)
    lgb_cutoff = data.groupby("cutoff").apply(lambda g: wape(g, "LightGBM"), include_groups=False)
    rows = []
    by_cutoff_rows = []
    for name in rules:
        column = f"rule::{name}"
        error_column = f"error::{name}"
        data[error_column] = np.abs(data["actual_final_demand"] - data[column])

        def rule_wape(mask: pd.Series | None = None) -> float:
            subset = data if mask is None else data.loc[mask]
            return float(subset[error_column].sum() / subset["actual_final_demand"].sum())

        cutoff_scores = data.groupby("cutoff").apply(
            lambda g: float(g[error_column].sum() / g["actual_final_demand"].sum()),
            include_groups=False,
        )
        rows.append({
            "rule": name,
            "pooled_WAPE": rule_wape(),
            "mean_cutoff_WAPE": cutoff_scores.mean(),
            "median_cutoff_WAPE": cutoff_scores.median(),
            "last3_pooled_WAPE": rule_wape(last3),
            "last6_pooled_WAPE": rule_wape(last6),
            "worst_cutoff_WAPE": cutoff_scores.max(),
            "worst_cutoff": pd.Timestamp(cutoff_scores.idxmax()),
            "wins_vs_CatBoost": int((cutoff_scores < cat_cutoff).sum()),
            "wins_vs_LightGBM": int((cutoff_scores < lgb_cutoff).sum()),
            "D15_30_WAPE": rule_wape(long),
            "D22_30_WAPE": rule_wape(far),
            "Top10_demand_x_D15_30_WAPE": rule_wape(top10_long),
        })
        for cutoff, score in cutoff_scores.items():
            by_cutoff_rows.append({"cutoff": cutoff, "rule": name, "WAPE": score})

    summary = pd.DataFrame(rows)
    by_cutoff = pd.DataFrame(by_cutoff_rows).pivot(index="cutoff", columns="rule", values="WAPE").reset_index()
    reference = pd.DataFrame([
        {
            "model": "CatBoost alpha=0.50",
            "pooled_WAPE": wape(data, "CatBoost"),
            "last3_WAPE": wape(data.loc[last3], "CatBoost"),
            "last6_WAPE": wape(data.loc[last6], "CatBoost"),
        },
        {
            "model": "LightGBM alpha=0.50",
            "pooled_WAPE": wape(data, "LightGBM"),
            "last3_WAPE": wape(data.loc[last3], "LightGBM"),
            "last6_WAPE": wape(data.loc[last6], "LightGBM"),
        },
    ])
    print("REFERENCE")
    print(reference.to_string(index=False, formatters={
        "pooled_WAPE": "{:.4%}".format, "last3_WAPE": "{:.4%}".format,
        "last6_WAPE": "{:.4%}".format,
    }))
    print(f"\nTop-10% diagnostic threshold (positive demand): actual > {q90:g}")
    print("\nRULE SUMMARY")
    print(summary.to_string(index=False, formatters={
        column: "{:.4%}".format for column in summary.columns if column.endswith("WAPE")
    }))
    print("\nWAPE BY CUTOFF")
    print(by_cutoff.to_string(index=False, formatters={
        column: "{:.4%}".format for column in by_cutoff.columns if column != "cutoff"
    }))


if __name__ == "__main__":
    main()
