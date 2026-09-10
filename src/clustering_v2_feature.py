"""Leakage-safe cutoff-local Clustering V2 assignments.

The fingerprint construction reproduces notebook cell 100 for only the fields
consumed by the corrected ``fit_clustering_v2`` implementation.  No final or
blind-cutoff assignment file is read.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
BLIND_CUTOFF = pd.Timestamp("2025-10-22")
EXPECTED_CITIES = 321
ANOMALY_DATE = pd.Timestamp("2025-01-11")
COMPLETION_HORIZONS = [30, 21, 14, 7, 3, 1]
V2_FEATURES = [
    "completion_D30", "completion_D21", "completion_D14", "completion_D7",
    "completion_D3", "completion_D1", "weekday_index_mon", "weekday_index_tue",
    "weekday_index_wed", "weekday_index_thu", "weekday_index_fri",
    "weekday_index_sat", "weekday_index_sun", "weekend_index_365",
    "holiday_index_365", "demand_cv_365", "log_recent_growth_90",
]


@lru_cache(maxsize=1)
def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, set[pd.Timestamp]]:
    search = pd.read_csv(
        ROOT / "data" / "raw" / "search_data.csv",
        parse_dates=["log_date", "checkin"], low_memory=False,
    )
    search["lead_time"] = (search["checkin"] - search["log_date"]).dt.days
    cities = pd.read_csv(ROOT / "data" / "raw" / "cities.csv")
    min_year = search["checkin"].dt.year.min() - 1
    max_year = BLIND_CUTOFF.year + 1
    ir_holidays = holidays.country_holidays(
        "IR", years=list(range(min_year, max_year + 1)), language="fa_IR"
    )
    official_dates = {pd.Timestamp(date) for date in ir_holidays.keys()}
    return search, cities, official_dates


def _safe_ratio(numerator: pd.Series, denominator: pd.Series | float) -> np.ndarray:
    return np.where(denominator > 0, numerator / denominator, np.nan)


def build_cluster_v2_snapshot(cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """Build the exact cutoff-local V2 fingerprint needed for clustering."""
    cutoff = pd.Timestamp(cutoff).normalize()
    if cutoff >= BLIND_CUTOFF:
        raise ValueError("Blind cutoff and later cutoffs are forbidden.")
    search, cities, official_dates = _inputs()
    train = search.loc[search["checkin"] <= cutoff].copy()
    assert train["checkin"].max() <= cutoff
    pair = (
        train.groupby(["city_code", "checkin"], as_index=False)["search_count"]
        .sum().rename(columns={"search_count": "final_demand"})
    )

    support = train.groupby("city_code").agg(
        first_checkin=("checkin", "min"), last_checkin=("checkin", "max"),
        positive_checkins=("checkin", "nunique"), total_searches=("search_count", "sum"),
    ).reset_index()
    support["is_one_off_anomaly"] = (
        (support["positive_checkins"] == 1)
        & (support["first_checkin"] == ANOMALY_DATE)
        & (support["last_checkin"] == ANOMALY_DATE)
        & (support["total_searches"] == 2)
    )
    support["fit_eligible"] = (
        (~support["is_one_off_anomaly"])
        & (support["positive_checkins"] >= 180)
        & (support["total_searches"] >= 1000)
    ).astype(int)

    city_lead = train.groupby(["city_code", "lead_time"], as_index=False)["search_count"].sum()
    completion = city_lead.groupby("city_code")["search_count"].sum().rename(
        "completion_total"
    ).reset_index()
    for horizon in COMPLETION_HORIZONS:
        numerator = (
            city_lead.loc[city_lead["lead_time"] >= horizon]
            .groupby("city_code")["search_count"].sum()
            .rename(f"completion_D{horizon}").reset_index()
        )
        completion = completion.merge(numerator, on="city_code", how="left")
        completion[f"completion_D{horizon}"] = (
            completion[f"completion_D{horizon}"].fillna(0) / completion["completion_total"]
        )
    completion = completion.drop(columns="completion_total")

    recent_start = cutoff - pd.Timedelta(days=89)
    previous_start = cutoff - pd.Timedelta(days=179)
    previous_end = cutoff - pd.Timedelta(days=90)
    recent = (
        pair.loc[pair["checkin"].between(recent_start, cutoff)]
        .groupby("city_code")["final_demand"].sum().rename("recent_90_total").reset_index()
    )
    previous = (
        pair.loc[pair["checkin"].between(previous_start, previous_end)]
        .groupby("city_code")["final_demand"].sum().rename("previous_90_total").reset_index()
    )
    recent_level = recent.merge(previous, on="city_code", how="outer").fillna(0)
    recent_level["log_recent_growth_90"] = (
        np.log1p(recent_level["recent_90_total"])
        - np.log1p(recent_level["previous_90_total"])
    )

    profile_start = cutoff - pd.Timedelta(days=364)
    profile_dates = pd.date_range(profile_start, cutoff, freq="D")
    grid = pd.MultiIndex.from_product(
        [np.sort(cities["city_code"].unique()), profile_dates],
        names=["city_code", "checkin"],
    ).to_frame(index=False)
    profile_pair = pair.loc[pair["checkin"].between(profile_start, cutoff)]
    grid = grid.merge(profile_pair, on=["city_code", "checkin"], how="left")
    grid["final_demand"] = grid["final_demand"].fillna(0)
    grid = grid.merge(support[["city_code", "first_checkin"]], on="city_code", how="left")
    grid["valid_city_day"] = grid["first_checkin"].notna() & (
        grid["checkin"] >= grid["first_checkin"]
    )
    grid.loc[~grid["valid_city_day"], "final_demand"] = np.nan
    grid["weekday_num"] = grid["checkin"].dt.dayofweek
    grid["is_weekend_iran"] = grid["weekday_num"].isin([3, 4]).astype(int)
    grid["is_official_holiday"] = grid["checkin"].isin(official_dates).astype(int)
    valid = grid.loc[grid["valid_city_day"]]

    overall = valid.groupby("city_code").agg(
        mean_daily_demand_365=("final_demand", "mean"),
        std_daily_demand_365=("final_demand", "std"),
    ).reset_index()
    overall["demand_cv_365"] = _safe_ratio(
        overall["std_daily_demand_365"], overall["mean_daily_demand_365"]
    )

    weekday = valid.groupby(["city_code", "weekday_num"])["final_demand"].mean().unstack()
    weekday = weekday.reindex(columns=range(7))
    weekday.columns = [
        "weekday_mean_mon", "weekday_mean_tue", "weekday_mean_wed",
        "weekday_mean_thu", "weekday_mean_fri", "weekday_mean_sat", "weekday_mean_sun",
    ]
    weekday = weekday.reset_index().merge(
        overall[["city_code", "mean_daily_demand_365"]], on="city_code", how="left"
    )
    for short_name in ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]:
        weekday[f"weekday_index_{short_name}"] = _safe_ratio(
            weekday[f"weekday_mean_{short_name}"], weekday["mean_daily_demand_365"]
        )
    weekday = weekday[["city_code"] + [f"weekday_index_{x}" for x in [
        "mon", "tue", "wed", "thu", "fri", "sat", "sun"
    ]]]

    weekend = valid.groupby(["city_code", "is_weekend_iran"])["final_demand"].mean().unstack()
    weekend = weekend.reindex(columns=[0, 1])
    weekend.columns = ["workingday_mean", "weekend_mean"]
    weekend = weekend.reset_index().merge(
        overall[["city_code", "mean_daily_demand_365"]], on="city_code", how="left"
    )
    weekend["weekend_index_365"] = _safe_ratio(
        weekend["weekend_mean"], weekend["mean_daily_demand_365"]
    )

    holiday = valid.groupby(["city_code", "is_official_holiday"])["final_demand"].mean().unstack()
    holiday = holiday.reindex(columns=[0, 1])
    holiday.columns = ["nonholiday_mean", "holiday_mean"]
    holiday = holiday.reset_index().merge(
        overall[["city_code", "mean_daily_demand_365"]], on="city_code", how="left"
    )
    holiday["holiday_index_365"] = _safe_ratio(
        holiday["holiday_mean"], holiday["mean_daily_demand_365"]
    )

    snapshot = (
        cities[["city_code"]]
        .merge(support[["city_code", "total_searches", "fit_eligible"]], on="city_code", how="left")
        .merge(completion, on="city_code", how="left")
        .merge(recent_level[["city_code", "log_recent_growth_90"]], on="city_code", how="left")
        .merge(overall[["city_code", "demand_cv_365"]], on="city_code", how="left")
        .merge(weekday, on="city_code", how="left")
        .merge(weekend[["city_code", "weekend_index_365"]], on="city_code", how="left")
        .merge(holiday[["city_code", "holiday_index_365"]], on="city_code", how="left")
    )
    snapshot["total_searches"] = snapshot["total_searches"].fillna(0)
    snapshot["fit_eligible"] = snapshot["fit_eligible"].fillna(0).astype(int)
    snapshot["snapshot_cutoff"] = cutoff
    assert len(snapshot) == EXPECTED_CITIES
    assert snapshot["city_code"].nunique() == EXPECTED_CITIES
    return snapshot


def fit_clustering_v2(snapshot: pd.DataFrame) -> dict[int, int]:
    """Corrected Clustering V2 logic from pol4-clustering-v2-fix."""
    snapshot_cities = {int(city) for city in snapshot["city_code"]}
    assert len(snapshot) == EXPECTED_CITIES
    assert len(snapshot_cities) == EXPECTED_CITIES
    train_cities = snapshot.loc[snapshot["fit_eligible"] == 1].copy()
    all_cities = snapshot.copy()
    q_low, q_high = train_cities["total_searches"].quantile([0.33, 0.66]).values

    def get_tier(value: float) -> str:
        if value <= q_low:
            return "Low"
        if value <= q_high:
            return "Med"
        return "High"

    train_cities["volume_tier"] = train_cities["total_searches"].apply(get_tier)
    all_cities["volume_tier"] = all_cities["total_searches"].apply(get_tier)
    mapping: dict[int, int] = {}
    cluster_id_counter = 0
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    for tier in ["Low", "Med", "High"]:
        tier_fit = train_cities.loc[train_cities["volume_tier"] == tier]
        tier_all = all_cities.loc[all_cities["volume_tier"] == tier]
        if len(tier_fit) < 3:
            for city in tier_all["city_code"]:
                mapping[int(city)] = cluster_id_counter
            cluster_id_counter += 1
            continue
        train_values = imputer.fit_transform(tier_fit[V2_FEATURES])
        all_values = imputer.transform(tier_all[V2_FEATURES])
        train_scaled = scaler.fit_transform(train_values)
        all_scaled = scaler.transform(all_values)
        model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(train_scaled)
        predictions = model.predict(all_scaled)
        for city, prediction in zip(tier_all["city_code"], predictions):
            mapping[int(city)] = int(cluster_id_counter + prediction)
        cluster_id_counter += 3
    assert len(mapping) == EXPECTED_CITIES
    assert set(mapping) == snapshot_cities
    return mapping


def build_cluster_v2_assignments(cutoffs: pd.DatetimeIndex) -> pd.DataFrame:
    parts = []
    for index, cutoff in enumerate(pd.to_datetime(cutoffs), start=1):
        cutoff = pd.Timestamp(cutoff)
        if cutoff >= BLIND_CUTOFF:
            raise ValueError("Blind cutoff entered cluster assignment build.")
        print(f"Clustering V2 snapshot {index:02d}/{len(cutoffs)}: {cutoff.date()}", flush=True)
        snapshot = build_cluster_v2_snapshot(cutoff)
        mapping = fit_clustering_v2(snapshot)
        part = pd.DataFrame(sorted(mapping.items()), columns=["city_code", "cluster_v2"])
        part["cutoff"] = cutoff
        part["cluster_snapshot_cutoff"] = cutoff
        assert len(part) == EXPECTED_CITIES and part["city_code"].nunique() == EXPECTED_CITIES
        assert part["cluster_v2"].notna().all()
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    assert (result["cluster_snapshot_cutoff"] == result["cutoff"]).all()
    return result

