"""Frozen Structural V1 forecast extracted from ``01_data_audit.ipynb``.

The implementation intentionally keeps the notebook's formulas and constants:
raw city completion curves with global fallback, a 12-week same-weekday
platform prior blended with gamma=0.25, calendar residual correction with
20-day shrinkage and [0.70, 2.50] clipping, and the
``cityprior28_b025_NEAR_LONG`` allocation gate.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import holidays
import jdatetime
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEARCH_PATH = ROOT / "data" / "raw" / "search_data.csv"
CITIES_PATH = ROOT / "data" / "raw" / "cities.csv"

FINAL_BLIND_CUTOFF = pd.Timestamp("2025-10-22")
PLATFORM_PRIOR_WEEKS = 12
GAMMA = 0.25
CALENDAR_SHRINK_DAYS = 20
ALLOCATION_BETA = 0.25


@lru_cache(maxsize=1)
def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    search = pd.read_csv(
        SEARCH_PATH,
        parse_dates=["log_date", "checkin"],
        low_memory=False,
    )
    search["lead_time"] = (search["checkin"] - search["log_date"]).dt.days
    cities = pd.read_csv(CITIES_PATH)
    return search, cities


def _jalali_parts(ts: pd.Timestamp) -> pd.Series:
    value = jdatetime.date.fromgregorian(date=ts.date())
    return pd.Series(
        {
            "jalali_year": value.year,
            "jalali_month": value.month,
            "jalali_day": value.day,
        }
    )


@lru_cache(maxsize=1)
def _calendar_master() -> pd.DataFrame:
    search, _ = _load_inputs()
    calendar_start = search["checkin"].min()
    calendar_end = search["checkin"].max()
    start_buffered = calendar_start - pd.Timedelta(days=30)
    end_buffered = calendar_end + pd.Timedelta(days=30)

    cal = pd.DataFrame(
        {"checkin": pd.date_range(start_buffered, end_buffered, freq="D")}
    )
    cal["weekday_num"] = cal["checkin"].dt.weekday
    cal["weekday"] = cal["checkin"].dt.day_name()
    cal["is_thursday"] = (cal["weekday_num"] == 3).astype(int)
    cal["is_friday"] = (cal["weekday_num"] == 4).astype(int)
    cal["is_weekend_iran"] = cal["weekday_num"].isin([3, 4]).astype(int)
    cal = pd.concat([cal, cal["checkin"].apply(_jalali_parts)], axis=1)

    years = list(range(start_buffered.year - 1, end_buffered.year + 2))
    ir_holidays = holidays.country_holidays("IR", years=years, language="fa_IR")
    holiday_map = {pd.Timestamp(date): name for date, name in ir_holidays.items()}
    cal["holiday_name"] = cal["checkin"].map(holiday_map)
    cal["is_official_holiday"] = cal["holiday_name"].notna().astype(int)

    solar_fixed_dates = {
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 12), (1, 13),
        (3, 14), (3, 15), (11, 22), (12, 29),
    }
    cal["is_solar_fixed_holiday"] = cal.apply(
        lambda row: int(
            row["is_official_holiday"] == 1
            and (row["jalali_month"], row["jalali_day"]) in solar_fixed_dates
        ),
        axis=1,
    )
    religious_keywords = [
        "شهادت", "ولادت", "رحلت", "امام", "حضرت", "فاطمه", "علی",
        "حسین", "محمد", "پیامبر", "عاشورا", "تاسوعا", "اربعین", "فطر",
        "قربان", "غدیر", "بعثت",
    ]
    cal["is_lunar_religious_holiday"] = cal["holiday_name"].apply(
        lambda name: 0 if pd.isna(name) else int(any(k in str(name) for k in religious_keywords))
    )

    official_dates = cal.loc[
        cal["is_official_holiday"] == 1, "checkin"
    ].sort_values().to_numpy(dtype="datetime64[ns]")
    all_dates = cal["checkin"].to_numpy(dtype="datetime64[ns]")

    prev_idx = np.searchsorted(official_dates, all_dates, side="right") - 1
    previous = np.full(len(cal), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_previous = prev_idx >= 0
    previous[valid_previous] = official_dates[prev_idx[valid_previous]]

    next_idx = np.searchsorted(official_dates, all_dates, side="left")
    following = np.full(len(cal), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_following = next_idx < len(official_dates)
    following[valid_following] = official_dates[next_idx[valid_following]]

    cal["prev_official_holiday"] = pd.to_datetime(previous)
    cal["next_official_holiday"] = pd.to_datetime(following)
    cal["days_since_prev_holiday"] = (
        cal["checkin"] - cal["prev_official_holiday"]
    ).dt.days
    cal["days_to_next_holiday"] = (
        cal["next_official_holiday"] - cal["checkin"]
    ).dt.days
    cal["is_nonworking"] = (
        (cal["is_weekend_iran"] == 1) | (cal["is_official_holiday"] == 1)
    ).astype(int)
    previous_nonworking = cal["is_nonworking"].shift(1).fillna(0)
    next_nonworking = cal["is_nonworking"].shift(-1).fillna(0)
    cal["is_bridge_day"] = (
        (cal["is_nonworking"] == 0)
        & (previous_nonworking == 1)
        & (next_nonworking == 1)
    ).astype(int)
    nonworking_groups = cal["is_nonworking"].ne(cal["is_nonworking"].shift()).cumsum()
    cal["nonworking_block_length"] = cal.groupby(nonworking_groups)[
        "is_nonworking"
    ].transform("size")
    cal.loc[cal["is_nonworking"] == 0, "nonworking_block_length"] = 0
    cal["is_travel_block_day"] = (
        (cal["is_nonworking"] == 1) | (cal["is_bridge_day"] == 1)
    ).astype(int)
    travel_groups = cal["is_travel_block_day"].ne(
        cal["is_travel_block_day"].shift()
    ).cumsum()
    cal["travel_block_length"] = cal.groupby(travel_groups)[
        "is_travel_block_day"
    ].transform("size")
    cal.loc[cal["is_travel_block_day"] == 0, "travel_block_length"] = 0
    cal["is_nowruz_travel_window"] = (
        (cal["jalali_month"] == 1) & cal["jalali_day"].between(1, 13)
    ).astype(int)

    bucket = pd.Series("normal", index=cal.index, dtype="object")
    bucket.loc[cal["days_to_next_holiday"].between(1, 3)] = "pre_holiday_1_3"
    bucket.loc[cal["days_since_prev_holiday"].between(1, 3)] = "post_holiday_1_3"
    bucket.loc[cal["is_bridge_day"] == 1] = "bridge"
    bucket.loc[
        (cal["is_official_holiday"] == 1)
        & (cal["is_lunar_religious_holiday"] == 1)
    ] = "official_lunar"
    bucket.loc[
        (cal["is_official_holiday"] == 1)
        & (cal["is_solar_fixed_holiday"] == 1)
    ] = "official_solar"
    bucket.loc[
        (cal["is_official_holiday"] == 1)
        & (cal["is_lunar_religious_holiday"] != 1)
        & (cal["is_solar_fixed_holiday"] != 1)
    ] = "official_other"
    bucket.loc[cal["is_nowruz_travel_window"] == 1] = "nowruz"
    cal["calendar_bucket"] = bucket
    return cal.loc[cal["checkin"].between(calendar_start, calendar_end)].reset_index(drop=True)


def _snapshot_features(cutoff: pd.Timestamp) -> pd.DataFrame:
    """Notebook cell 101, excluding its analysis-only future-derived columns."""
    search, cities = _load_inputs()
    test_start = cutoff + pd.Timedelta(days=1)
    test_end = cutoff + pd.Timedelta(days=30)
    dates = pd.date_range(test_start, test_end, freq="D")
    grid = pd.MultiIndex.from_product(
        [np.sort(cities["city_code"].unique()), dates],
        names=["city_code", "checkin"],
    ).to_frame(index=False)
    grid["cutoff"] = cutoff
    grid["days_to_checkin"] = (grid["checkin"] - cutoff).dt.days

    actual = (
        search.loc[search["checkin"].between(test_start, test_end)]
        .groupby(["city_code", "checkin"], as_index=False)["search_count"]
        .sum()
        .rename(columns={"search_count": "actual_final_demand"})
    )
    grid = grid.merge(actual, on=["city_code", "checkin"], how="left")
    grid["actual_final_demand"] = grid["actual_final_demand"].fillna(0)

    visible = search.loc[
        search["checkin"].between(test_start, test_end)
        & (search["log_date"] <= cutoff)
    ].copy()
    assert visible.empty or visible["log_date"].max() <= cutoff
    visible["days_before_cutoff"] = (cutoff - visible["log_date"]).dt.days

    observed = (
        visible.groupby(["city_code", "checkin"], as_index=False)["search_count"]
        .sum()
        .rename(columns={"search_count": "observed_demand"})
    )
    grid = grid.merge(observed, on=["city_code", "checkin"], how="left")
    grid["observed_demand"] = grid["observed_demand"].fillna(0)

    for window in [1, 3, 7, 14]:
        feature = f"search_last_{window}d"
        temp = (
            visible.loc[visible["days_before_cutoff"] < window]
            .groupby(["city_code", "checkin"], as_index=False)["search_count"]
            .sum()
            .rename(columns={"search_count": feature})
        )
        grid = grid.merge(temp, on=["city_code", "checkin"], how="left")
        grid[feature] = grid[feature].fillna(0)

    for feature, low, high in [("search_prev_3d", 3, 5), ("search_prev_7d", 7, 13)]:
        temp = (
            visible.loc[visible["days_before_cutoff"].between(low, high)]
            .groupby(["city_code", "checkin"], as_index=False)["search_count"]
            .sum()
            .rename(columns={"search_count": feature})
        )
        grid = grid.merge(temp, on=["city_code", "checkin"], how="left")
        grid[feature] = grid[feature].fillna(0)

    active = (
        visible.loc[visible["days_before_cutoff"] < 7]
        .groupby(["city_code", "checkin"])["log_date"]
        .nunique()
        .reset_index(name="active_search_days_last_7d")
    )
    last_search = (
        visible.groupby(["city_code", "checkin"])["log_date"]
        .max()
        .reset_index(name="last_search_date")
    )
    grid = grid.merge(active, on=["city_code", "checkin"], how="left")
    grid = grid.merge(last_search, on=["city_code", "checkin"], how="left")
    grid["active_search_days_last_7d"] = grid["active_search_days_last_7d"].fillna(0)
    grid["days_since_last_search"] = (cutoff - grid["last_search_date"]).dt.days.fillna(999)
    grid["share_last_3d"] = np.where(
        grid["observed_demand"] > 0,
        grid["search_last_3d"] / grid["observed_demand"],
        np.nan,
    )
    grid["share_last_7d"] = np.where(
        grid["observed_demand"] > 0,
        grid["search_last_7d"] / grid["observed_demand"],
        np.nan,
    )
    grid["momentum_3d"] = (grid["search_last_3d"] + 1) / (grid["search_prev_3d"] + 1)
    grid["momentum_7d"] = (grid["search_last_7d"] + 1) / (grid["search_prev_7d"] + 1)
    return grid.drop(columns=["last_search_date"])


def _completion_features(cutoff: pd.Timestamp, out: pd.DataFrame) -> pd.DataFrame:
    search, _ = _load_inputs()
    train = search.loc[search["checkin"] <= cutoff].copy()
    assert train.empty or train["checkin"].max() <= cutoff

    global_volume = train.groupby("lead_time")["search_count"].sum().reindex(range(60), fill_value=0)
    global_rates = (
        global_volume.sort_index(ascending=False).cumsum() / global_volume.sum()
    ).rename("global_completion_rate")
    global_lookup = global_rates.reset_index().rename(columns={"lead_time": "days_to_checkin"})

    train_cities = np.sort(train["city_code"].unique())
    city_grid = pd.MultiIndex.from_product(
        [train_cities, range(60)], names=["city_code", "lead_time"]
    ).to_frame(index=False)
    city_volume = train.groupby(["city_code", "lead_time"], as_index=False)["search_count"].sum()
    city_curve = city_grid.merge(city_volume, on=["city_code", "lead_time"], how="left")
    city_curve["search_count"] = city_curve["search_count"].fillna(0)
    city_curve = city_curve.sort_values(["city_code", "lead_time"], ascending=[True, False])
    city_curve["cumulative"] = city_curve.groupby("city_code")["search_count"].cumsum()
    city_total = city_curve.groupby("city_code")["search_count"].transform("sum")
    city_curve["city_completion_rate"] = np.where(
        city_total > 0, city_curve["cumulative"] / city_total, np.nan
    )
    support = train.groupby("city_code").agg(
        historical_checkins=("checkin", "nunique"),
        historical_searches=("search_count", "sum"),
    ).reset_index()
    city_lookup = city_curve.merge(support, on="city_code", how="left").rename(
        columns={"lead_time": "days_to_checkin"}
    )[["city_code", "days_to_checkin", "city_completion_rate", "historical_checkins", "historical_searches"]]

    out = out.merge(global_lookup, on="days_to_checkin", how="left")
    out = out.merge(city_lookup, on=["city_code", "days_to_checkin"], how="left")
    out["city_rate_with_fallback"] = out["city_completion_rate"].fillna(
        out["global_completion_rate"]
    )
    out["city_rate_with_fallback"] = np.where(
        out["city_rate_with_fallback"] > 0,
        out["city_rate_with_fallback"],
        out["global_completion_rate"],
    )
    out["pred_city_raw"] = out["observed_demand"] / out["city_rate_with_fallback"]
    return out


def _city_recent_features(cutoff: pd.Timestamp) -> pd.DataFrame:
    """Notebook cell 104, with every trailing window ending exactly at cutoff."""
    search, cities = _load_inputs()
    pair_history = (
        search.groupby(["city_code", "checkin"], as_index=False)["search_count"]
        .sum()
        .rename(columns={"search_count": "final_demand"})
    )
    first_checkin = pair_history.groupby("city_code")["checkin"].min().rename("first_checkin").reset_index()
    all_cities = pd.DataFrame({"city_code": np.sort(cities["city_code"].unique())})
    out = all_cities.merge(first_checkin, on="city_code", how="left")

    for window in [28, 56, 90]:
        start = cutoff - pd.Timedelta(days=window - 1)
        assert start <= cutoff
        temp = (
            pair_history.loc[pair_history["checkin"].between(start, cutoff)]
            .groupby("city_code")["final_demand"]
            .sum()
            .rename(f"city_total_{window}")
            .reset_index()
        )
        temp = all_cities.merge(temp, on="city_code", how="left")
        temp[f"city_total_{window}"] = temp[f"city_total_{window}"].fillna(0)
        temp = temp.merge(first_checkin, on="city_code", how="left")
        effective_start = pd.concat(
            [temp["first_checkin"], pd.Series(start, index=temp.index)], axis=1
        ).max(axis=1)
        temp[f"active_days_{window}"] = np.where(
            temp["first_checkin"].notna() & (temp["first_checkin"] <= cutoff),
            (cutoff - effective_start).dt.days + 1,
            0,
        )
        temp[f"city_mean_{window}"] = np.where(
            temp[f"active_days_{window}"] > 0,
            temp[f"city_total_{window}"] / temp[f"active_days_{window}"],
            np.nan,
        )
        platform_total = temp[f"city_total_{window}"].sum()
        temp[f"city_share_{window}"] = np.where(
            platform_total > 0, temp[f"city_total_{window}"] / platform_total, np.nan
        )
        out = out.merge(
            temp[["city_code", f"city_total_{window}", f"active_days_{window}",
                  f"city_mean_{window}", f"city_share_{window}"]],
            on="city_code", how="left",
        )

    for window in [28, 56]:
        recent_start = cutoff - pd.Timedelta(days=window - 1)
        previous_end = recent_start - pd.Timedelta(days=1)
        previous_start = previous_end - pd.Timedelta(days=window - 1)
        temp = (
            pair_history.loc[pair_history["checkin"].between(previous_start, previous_end)]
            .groupby("city_code")["final_demand"]
            .sum()
            .rename(f"city_prev{window}_total")
            .reset_index()
        )
        out = out.merge(temp, on="city_code", how="left")
        out[f"city_prev{window}_total"] = out[f"city_prev{window}_total"].fillna(0)
        out[f"log_growth_{window}"] = np.log1p(out[f"city_total_{window}"]) - np.log1p(
            out[f"city_prev{window}_total"]
        )
    return out


def _platform_and_calendar(cutoff: pd.Timestamp, out: pd.DataFrame) -> pd.DataFrame:
    search, _ = _load_inputs()
    cal = _calendar_master()
    platform_history = (
        search.groupby("checkin", as_index=False)["search_count"]
        .sum()
        .rename(columns={"search_count": "platform_final_demand"})
    )
    platform_history["weekday_num"] = platform_history["checkin"].dt.weekday
    platform_history = platform_history.merge(
        cal[["checkin", "calendar_bucket"]], on="checkin", how="left"
    )
    platform_history["calendar_bucket"] = platform_history["calendar_bucket"].fillna("normal")
    prior_col = f"hist_weekday_prior_{PLATFORM_PRIOR_WEEKS}w"
    platform_history[prior_col] = (
        platform_history.sort_values("checkin")
        .groupby("weekday_num")["platform_final_demand"]
        .transform(lambda values: values.shift(1).rolling(PLATFORM_PRIOR_WEEKS, min_periods=4).median())
    )

    hist = platform_history.loc[
        (platform_history["checkin"] <= cutoff)
        & platform_history[prior_col].notna()
        & (platform_history[prior_col] > 0)
    ].copy()
    assert hist.empty or hist["checkin"].max() <= cutoff
    hist["ratio_vs_weekday_prior"] = (
        hist["platform_final_demand"] / hist[prior_col]
    ).clip(lower=0.25, upper=5.0)
    factors = hist.groupby("calendar_bucket", as_index=False).agg(
        actual_sum=("platform_final_demand", "sum"),
        prior_sum=(prior_col, "sum"),
        calendar_factor_n_days=("checkin", "size"),
    )
    factors["calendar_raw_factor"] = factors["actual_sum"] / factors["prior_sum"]
    factor_weight = factors["calendar_factor_n_days"] / (
        factors["calendar_factor_n_days"] + CALENDAR_SHRINK_DAYS
    )
    factors["calendar_factor"] = (
        1.0 + factor_weight * (factors["calendar_raw_factor"] - 1.0)
    ).clip(lower=0.70, upper=2.50)

    platform_dates = out.groupby(["cutoff", "checkin", "days_to_checkin"], as_index=False).agg(
        cityraw_total=("pred_city_raw", "sum")
    )
    platform_dates["weekday_num"] = platform_dates["checkin"].dt.weekday
    same_weekday_priors = []
    completed_history = platform_history.loc[platform_history["checkin"] <= cutoff]
    for row in platform_dates.itertuples(index=False):
        values = completed_history.loc[
            completed_history["weekday_num"] == row.weekday_num,
            "platform_final_demand",
        ].tail(PLATFORM_PRIOR_WEEKS)
        same_weekday_priors.append(values.median() if len(values) else np.nan)
    platform_dates["weekday_prior_12w"] = same_weekday_priors
    platform_dates = platform_dates.merge(
        cal[["checkin", "calendar_bucket"]], on="checkin", how="left"
    ).merge(
        factors[["calendar_bucket", "calendar_factor", "calendar_factor_n_days", "calendar_raw_factor"]],
        on="calendar_bucket", how="left",
    )
    platform_dates["calendar_bucket"] = platform_dates["calendar_bucket"].fillna("normal")
    platform_dates["calendar_factor"] = platform_dates["calendar_factor"].fillna(1.0)
    platform_dates["calendar_adjusted_prior"] = (
        platform_dates["weekday_prior_12w"] * platform_dates["calendar_factor"]
    )
    platform_dates["live_weight"] = platform_dates["global_completion_rate"] = platform_dates[
        "days_to_checkin"
    ].map(out.drop_duplicates("days_to_checkin").set_index("days_to_checkin")["global_completion_rate"])
    platform_dates["live_weight"] = platform_dates["live_weight"].clip(0, 1) ** GAMMA
    platform_dates["pred_total_calendar"] = (
        platform_dates["live_weight"] * platform_dates["cityraw_total"]
        + (1 - platform_dates["live_weight"]) * platform_dates["calendar_adjusted_prior"]
    )
    platform_dates["calendar_scale"] = np.where(
        platform_dates["cityraw_total"] > 0,
        platform_dates["pred_total_calendar"] / platform_dates["cityraw_total"],
        1.0,
    )
    out = out.merge(
        platform_dates.drop(columns=["global_completion_rate"]),
        on=["cutoff", "checkin", "days_to_checkin"], how="left",
    )
    out["pred_calendar_blend"] = out["pred_city_raw"] * out["calendar_scale"]
    return out


def _allocation(cutoff: pd.Timestamp, out: pd.DataFrame) -> pd.DataFrame:
    recent = _city_recent_features(cutoff)
    out = out.merge(recent, on="city_code", how="left")
    for window in [28, 56, 90]:
        out[f"implied_vs_city_mean_{window}"] = np.where(
            out[f"city_mean_{window}"] > 0,
            out["pred_city_raw"] / out[f"city_mean_{window}"],
            np.nan,
        )
        out[f"log_implied_vs_city_mean_{window}"] = np.log1p(
            out[f"implied_vs_city_mean_{window}"]
        )
        out[f"log_city_mean_{window}"] = np.log1p(out[f"city_mean_{window}"])
        out[f"log_city_share_{window}"] = np.log1p(out[f"city_share_{window}"] * 1_000_000)

    date_keys = ["cutoff", "checkin"]
    out["calendar_platform_total"] = out.groupby(date_keys)["pred_calendar_blend"].transform("sum")
    out["cityraw_platform_total"] = out.groupby(date_keys)["pred_city_raw"].transform("sum")
    out["live_city_share"] = np.where(
        out["cityraw_platform_total"] > 0,
        out["pred_city_raw"] / out["cityraw_platform_total"],
        0.0,
    )
    out["active_pair"] = (out["pred_city_raw"] > 0).astype(int)
    out["active_prior_raw_28"] = np.where(
        out["active_pair"] == 1, out["city_share_28"].fillna(0), 0.0
    )
    out["active_prior_total_28"] = out.groupby(date_keys)["active_prior_raw_28"].transform("sum")
    out["active_prior_share_28"] = np.where(
        out["active_prior_total_28"] > 0,
        out["active_prior_raw_28"] / out["active_prior_total_28"],
        out["live_city_share"],
    )
    out["allocation_live_weight"] = out["global_completion_rate"].clip(0, 1) ** ALLOCATION_BETA
    out["share_cityprior28_b025"] = (
        out["allocation_live_weight"] * out["live_city_share"]
        + (1 - out["allocation_live_weight"]) * out["active_prior_share_28"]
    )
    share_total = out.groupby(date_keys)["share_cityprior28_b025"].transform("sum")
    out["share_cityprior28_b025"] = np.where(
        share_total > 0,
        out["share_cityprior28_b025"] / share_total,
        out["live_city_share"],
    )
    out["pred_cityprior28_b025"] = (
        out["calendar_platform_total"] * out["share_cityprior28_b025"]
    )
    use_prior = (out["days_to_checkin"] <= 7) | (out["days_to_checkin"] >= 15)
    out["pred_structural_v1"] = np.where(
        use_prior, out["pred_cityprior28_b025"], out["pred_calendar_blend"]
    )
    return out


def build_structural_v1(cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """Build one leakage-safe 321-city x 30-day Structural V1 snapshot.

    The final blind cutoff is deliberately rejected by this development entry
    point so it cannot be evaluated accidentally.
    """
    cutoff = pd.Timestamp(cutoff).normalize()
    if cutoff >= FINAL_BLIND_CUTOFF:
        raise ValueError("Final blind cutoff is reserved and cannot be built in development mode.")
    out = _snapshot_features(cutoff)
    out = _completion_features(cutoff, out)
    out = _platform_and_calendar(cutoff, out)
    out = _allocation(cutoff, out)
    cal = _calendar_master()
    calendar_columns = [
        "checkin", "weekday_num", "weekday", "is_thursday", "is_friday",
        "is_weekend_iran", "jalali_year", "jalali_month", "jalali_day",
        "holiday_name", "is_official_holiday", "is_solar_fixed_holiday",
        "is_lunar_religious_holiday", "days_since_prev_holiday",
        "days_to_next_holiday", "is_nonworking", "is_bridge_day",
        "nonworking_block_length", "is_travel_block_day", "travel_block_length",
        "is_nowruz_travel_window",
    ]
    out = out.merge(cal[calendar_columns], on="checkin", how="left")
    out["holiday_name"] = out["holiday_name"].fillna("None")
    out["horizon_band"] = pd.cut(
        out["days_to_checkin"], bins=[0, 3, 7, 14, 21, 30],
        labels=["D1-3", "D4-7", "D8-14", "D15-21", "D22-30"],
    ).astype(str)
    # ``first_checkin`` is only an internal denominator guard in notebook cell
    # 104.  The notebook computed it from the full dataset, so exposing it to
    # ML would reveal future city starts for cold cities.  It is not one of the
    # tested recent-level features and is deliberately removed here.
    out = out.drop(columns=["first_checkin"])
    out["feature_max_log_date"] = cutoff
    out["completion_history_max_checkin"] = cutoff
    out["rolling_window_end"] = cutoff
    out["calendar_history_max_checkin"] = cutoff
    return out.sort_values(["city_code", "checkin"]).reset_index(drop=True)
