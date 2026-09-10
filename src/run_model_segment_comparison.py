"""Post-hoc segment diagnostics for persisted CatBoost and LightGBM OOT rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "outputs" / "catboost_expanding_oot_row_predictions.parquet"
LGB_PATH = ROOT / "outputs" / "lightgbm_expanding_oot_row_predictions.parquet"
BLIND_CUTOFF = pd.Timestamp("2025-10-22")
ALPHA = 0.50
MODELS = {
    "Structural": "pred_structural",
    "CatBoost": "pred_catboost",
    "LightGBM": "pred_lightgbm",
}


def load_data() -> pd.DataFrame:
    cat = pd.read_parquet(CAT_PATH)
    lgb = pd.read_parquet(LGB_PATH)
    keys = ["cutoff", "city_code", "checkin"]
    data = cat[keys + [
        "observed_demand", "actual_final_demand", "pred_structural_v1",
        "predicted_residual",
    ]].merge(
        lgb[keys + ["predicted_residual"]],
        on=keys, how="inner", validate="one_to_one",
        suffixes=("_catboost", "_lightgbm"),
    )
    assert len(data) == 13 * 321 * 30
    assert data["cutoff"].nunique() == 13
    assert data.groupby("cutoff").size().eq(321 * 30).all()
    assert data["cutoff"].max() < BLIND_CUTOFF
    data["days_to_checkin"] = (data["checkin"] - data["cutoff"]).dt.days
    data["pred_structural"] = np.maximum(
        data["observed_demand"], np.maximum(0.0, data["pred_structural_v1"])
    )
    for model in ["catboost", "lightgbm"]:
        data[f"pred_{model}"] = np.maximum(
            data["observed_demand"],
            np.maximum(
                0.0,
                data["pred_structural_v1"]
                + ALPHA * data[f"predicted_residual_{model}"],
            ),
        )
    for name, prediction in MODELS.items():
        data[f"abs_error_{name}"] = np.abs(data["actual_final_demand"] - data[prediction])
    return data


def wape(group: pd.DataFrame, model: str) -> float:
    denominator = group["actual_final_demand"].sum()
    if denominator == 0:
        return np.nan
    return float(group[f"abs_error_{model}"].sum() / denominator)


def segment_row(data: pd.DataFrame, mask: pd.Series, group: str) -> dict[str, float | int | str]:
    subset = data.loc[mask]
    actual_sum = float(subset["actual_final_demand"].sum())
    row: dict[str, float | int | str] = {
        "group": group,
        "rows": len(subset),
        "actual_sum": actual_sum,
        "demand_share": actual_sum / data["actual_final_demand"].sum(),
    }
    for model in MODELS:
        row[f"{model}_WAPE"] = wape(subset, model)
        row[f"{model}_abs_error"] = float(subset[f"abs_error_{model}"].sum())
    row["CatBoost_minus_LightGBM_WAPE"] = row["CatBoost_WAPE"] - row["LightGBM_WAPE"]
    row["CatBoost_minus_LightGBM_abs_error"] = (
        row["CatBoost_abs_error"] - row["LightGBM_abs_error"]
    )
    return row


def format_table(table: pd.DataFrame) -> str:
    formatters = {}
    for column in table.columns:
        if "WAPE" in column or column == "demand_share":
            formatters[column] = lambda value: "N/A" if pd.isna(value) else f"{value:.4%}"
        elif column.endswith("abs_error") or column == "actual_sum":
            formatters[column] = lambda value: f"{value:,.1f}"
    return table.to_string(index=False, formatters=formatters)


def main() -> None:
    data = load_data()
    positive = data.loc[data["actual_final_demand"] > 0, "actual_final_demand"]
    quantiles = positive.quantile([0.50, 0.80, 0.90, 0.95, 0.99])
    q50, q80, q90, q95, q99 = [float(quantiles.loc[q]) for q in quantiles.index]
    actual = data["actual_final_demand"]
    print("QUANTILE THRESHOLDS", {str(q): float(value) for q, value in quantiles.items()})

    exclusive_masks = [
        ("Zero demand", actual == 0),
        ("Positive bottom 50%", (actual > 0) & (actual <= q50)),
        ("P50-P80", (actual > q50) & (actual <= q80)),
        ("P80-P90", (actual > q80) & (actual <= q90)),
        ("P90-P95", (actual > q90) & (actual <= q95)),
        ("P95-P99", (actual > q95) & (actual <= q99)),
        ("Top 1%", actual > q99),
    ]
    cumulative_masks = [
        ("Top 20%", actual > q80),
        ("Top 10%", actual > q90),
        ("Top 5%", actual > q95),
        ("Top 1%", actual > q99),
    ]
    demand_exclusive = pd.DataFrame([
        segment_row(data, mask, name) for name, mask in exclusive_masks
    ])
    demand_cumulative = pd.DataFrame([
        segment_row(data, mask, name) for name, mask in cumulative_masks
    ])
    print("\nDEMAND SIZE EXCLUSIVE")
    print(format_table(demand_exclusive))
    print("\nDEMAND SIZE CUMULATIVE")
    print(format_table(demand_cumulative))

    horizon_masks = [
        ("D1-3", data["days_to_checkin"].between(1, 3)),
        ("D4-7", data["days_to_checkin"].between(4, 7)),
        ("D8-14", data["days_to_checkin"].between(8, 14)),
        ("D15-21", data["days_to_checkin"].between(15, 21)),
        ("D22-30", data["days_to_checkin"].between(22, 30)),
        ("D1-7", data["days_to_checkin"].between(1, 7)),
        ("D15-30", data["days_to_checkin"].between(15, 30)),
    ]
    horizon = pd.DataFrame([segment_row(data, mask, name) for name, mask in horizon_masks])
    print("\nHORIZON")
    print(format_table(horizon))

    high_horizon_masks = []
    for label, demand_mask in [("Top 20%", actual > q80), ("Top 10%", actual > q90)]:
        high_horizon_masks.extend([
            (f"{label} x D1-7", demand_mask & data["days_to_checkin"].between(1, 7)),
            (f"{label} x D8-14", demand_mask & data["days_to_checkin"].between(8, 14)),
            (f"{label} x D15-30", demand_mask & data["days_to_checkin"].between(15, 30)),
        ])
    high_horizon_masks.append(
        ("Top 5% x D15-30", (actual > q95) & data["days_to_checkin"].between(15, 30))
    )
    high_horizon = pd.DataFrame([
        segment_row(data, mask, name) for name, mask in high_horizon_masks
    ])
    print("\nHIGH DEMAND X HORIZON")
    print(format_table(high_horizon))

    ordered_cutoffs = sorted(data["cutoff"].unique())
    recency_rows = []
    for count, label in [(3, "Last 3"), (6, "Last 6"), (13, "All 13")]:
        subset = data.loc[data["cutoff"].isin(ordered_cutoffs[-count:])]
        recency_rows.append({
            "scope": label,
            "cutoffs": count,
            **{f"{model}_WAPE": wape(subset, model) for model in MODELS},
            "CatBoost_minus_LightGBM_WAPE": wape(subset, "CatBoost") - wape(subset, "LightGBM"),
        })
    recency = pd.DataFrame(recency_rows)
    print("\nRECENCY")
    print(format_table(recency))

    concentration_masks = [
        ("Top 1% highest demand", actual > q99),
        ("Top 5% highest demand", actual > q95),
        ("Top 10% highest demand", actual > q90),
        ("D15-30", data["days_to_checkin"].between(15, 30)),
        ("Top 10% x D15-30", (actual > q90) & data["days_to_checkin"].between(15, 30)),
    ]
    concentration_rows = []
    for label, mask in concentration_masks:
        row = {"segment": label}
        for model in MODELS:
            row[f"{model}_error_share"] = (
                data.loc[mask, f"abs_error_{model}"].sum() / data[f"abs_error_{model}"].sum()
            )
        concentration_rows.append(row)
    concentration = pd.DataFrame(concentration_rows)
    print("\nERROR CONCENTRATION")
    print(concentration.to_string(index=False, formatters={
        column: "{:.4%}".format for column in concentration.columns if column.endswith("error_share")
    }))

    top_error_columns = [
        "cutoff", "city_code", "checkin", "days_to_checkin", "observed_demand",
        "actual_final_demand", "pred_structural", "pred_catboost", "pred_lightgbm",
    ]
    for model in ["CatBoost", "LightGBM"]:
        top = data.nlargest(20, f"abs_error_{model}")[
            top_error_columns + [f"abs_error_{model}"]
        ]
        print(f"\nTOP 20 ABSOLUTE ERRORS - {model.upper()}")
        print(top.to_string(index=False, formatters={
            "observed_demand": "{:,.1f}".format,
            "actual_final_demand": "{:,.1f}".format,
            "pred_structural": "{:,.1f}".format,
            "pred_catboost": "{:,.1f}".format,
            "pred_lightgbm": "{:,.1f}".format,
            f"abs_error_{model}": "{:,.1f}".format,
        }))

    def row_for(table: pd.DataFrame, group: str) -> pd.Series:
        return table.loc[table["group"] == group].iloc[0]

    cutoff_wapes = []
    for cutoff, group in data.groupby("cutoff"):
        cutoff_wapes.append({"cutoff": cutoff, **{model: wape(group, model) for model in MODELS}})
    cutoff_wapes = pd.DataFrame(cutoff_wapes)
    last3 = data.loc[data["cutoff"].isin(ordered_cutoffs[-3:])]
    last6 = data.loc[data["cutoff"].isin(ordered_cutoffs[-6:])]
    decision_sources = {
        "Overall pooled WAPE": {model: wape(data, model) for model in MODELS},
        "Top 20% demand WAPE": {model: row_for(demand_cumulative, "Top 20%")[f"{model}_WAPE"] for model in MODELS},
        "Top 10% demand WAPE": {model: row_for(demand_cumulative, "Top 10%")[f"{model}_WAPE"] for model in MODELS},
        "Top 5% demand WAPE": {model: row_for(demand_cumulative, "Top 5%")[f"{model}_WAPE"] for model in MODELS},
        "D15-30 WAPE": {model: row_for(horizon, "D15-30")[f"{model}_WAPE"] for model in MODELS},
        "D22-30 WAPE": {model: row_for(horizon, "D22-30")[f"{model}_WAPE"] for model in MODELS},
        "Top 10% demand x D15-30 WAPE": {model: row_for(high_horizon, "Top 10% x D15-30")[f"{model}_WAPE"] for model in MODELS},
        "Last 3 cutoffs pooled WAPE": {model: wape(last3, model) for model in MODELS},
        "Last 6 cutoffs pooled WAPE": {model: wape(last6, model) for model in MODELS},
        "Worst cutoff WAPE": {model: cutoff_wapes[model].max() for model in MODELS},
    }
    decision_rows = []
    for metric, values in decision_sources.items():
        decision_rows.append({"Metric": metric, **values, "Winner": min(values, key=values.get)})
    decision = pd.DataFrame(decision_rows)
    print("\nDECISION TABLE")
    print(decision.to_string(index=False, formatters={model: "{:.4%}".format for model in MODELS}))


if __name__ == "__main__":
    main()
