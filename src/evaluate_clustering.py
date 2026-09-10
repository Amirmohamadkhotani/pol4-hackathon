"""Corrected leakage-safe clustering evaluation and diagnostics.

The existing forecasting formula and parameters are intentionally unchanged.
This module corrects the evaluation population, aggregation, and clustering
audit without changing feature snapshot generation.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

DATA_DIR = "data"
PATH_SEARCH = os.environ.get(
    "SEARCH_DATA_PATH", os.path.join(DATA_DIR, "raw/search_data.csv")
)
PATH_SNAPSHOTS = os.path.join(
    DATA_DIR, "processed/city_features_backtest_snapshots.csv"
)
PATH_FEATURES_TXT = os.path.join(
    DATA_DIR, "processed/city_clustering_v1_features.txt"
)

EXPECTED_CITY_COUNT = 321
FORECAST_DAYS = 30
LEAD_TIMES = np.arange(60, dtype=int)
K_VALUES = tuple(range(3, 11))

# Fixed, pre-existing forecasting parameters. Do not tune in this audit.
DEFAULT_SCORE_PARAMS = dict(
    base_lambda=0.015,
    horizon_scale=0.10,
    horizon_power=1.6,
    momentum_damping=0.6,
    cap_headroom=1.35,
    flat_lambda=0.02,
    use_dynamic_lambda=True,
    use_momentum=True,
    use_caps=True,
)


def _load_inputs():
    with open(PATH_FEATURES_TXT, "r", encoding="utf-8") as handle:
        features = [
            line.strip()
            for line in handle
            if line.strip() and not line.startswith("#")
        ]
    if "city_code" in features:
        raise AssertionError("city_code must never be a clustering feature")

    search = pd.read_csv(PATH_SEARCH)
    search["log_date"] = pd.to_datetime(search["log_date"])
    search["checkin"] = pd.to_datetime(search["checkin"])
    search["lead_time"] = (search["checkin"] - search["log_date"]).dt.days
    search["dow"] = search["checkin"].dt.dayofweek

    snapshots = pd.read_csv(PATH_SNAPSHOTS)
    snapshots["snapshot_cutoff"] = pd.to_datetime(snapshots["snapshot_cutoff"])
    return search, snapshots, features


search_df, snapshots_df, clustering_features = _load_inputs()
maximum_historical_checkin = search_df["checkin"].max()
available_cutoffs = sorted(pd.to_datetime(snapshots_df["snapshot_cutoff"].unique()))
labelled_cutoffs = [
    cutoff
    for cutoff in available_cutoffs
    if cutoff + pd.Timedelta(days=FORECAST_DAYS) <= maximum_historical_checkin
]


def calculate_wape_components(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have identical shapes")
    actual_total = float(actual.sum())
    if actual_total <= 0:
        raise ValueError("WAPE is undefined when total actual demand is zero")
    abs_error = float(np.abs(predicted - actual).sum())
    return abs_error, actual_total, abs_error / actual_total


def calculate_wape(actual, predicted):
    return calculate_wape_components(actual, predicted)[2]


def _completion_rate_dict(volume_df, group_col=None, entities=None):
    """Build complete h=0..59 curves, treating absent increments as zero."""
    if group_col is None:
        volume = (
            volume_df.groupby("lead_time")["search_count"]
            .sum()
            .reindex(LEAD_TIMES, fill_value=0.0)
            .astype(float)
        )
        total = float(volume.sum())
        if total <= 0:
            return {int(h): 1.0 for h in LEAD_TIMES}
        cumulative = volume.sort_index(ascending=False).cumsum() / total
        return {int(h): float(cumulative.loc[h]) for h in LEAD_TIMES}

    entities = list(entities)
    if not entities:
        return {}
    grid = pd.MultiIndex.from_product(
        [entities, LEAD_TIMES], names=[group_col, "lead_time"]
    ).to_frame(index=False)
    grouped = (
        volume_df.groupby([group_col, "lead_time"], as_index=False)["search_count"]
        .sum()
    )
    grid = grid.merge(grouped, on=[group_col, "lead_time"], how="left")
    grid["search_count"] = grid["search_count"].fillna(0.0)
    grid = grid.sort_values([group_col, "lead_time"], ascending=[True, False])
    grid["cumulative"] = grid.groupby(group_col)["search_count"].cumsum()
    totals = grid.groupby(group_col)["search_count"].transform("sum")
    grid["rate"] = np.where(totals > 0, grid["cumulative"] / totals, np.nan)
    return {
        entity: {
            int(h): float(rate)
            for h, rate in zip(part["lead_time"], part["rate"])
            if pd.notna(rate)
        }
        for entity, part in grid.groupby(group_col, sort=False)
    }


def compute_curves(hist_search_df, city_to_cluster):
    df = hist_search_df.copy()
    df["cluster"] = df["city_code"].map(city_to_cluster).fillna(-1).astype(int)
    tot_city = df.groupby("city_code")["search_count"].sum().to_dict()

    global_rates = _completion_rate_dict(df)
    cluster_rates = _completion_rate_dict(
        df, "cluster", sorted(df["cluster"].unique())
    )
    eligible_cities = [city for city, total in tot_city.items() if total >= 1500]
    city_rates = _completion_rate_dict(df, "city_code", eligible_cities)

    dow_city = df.groupby(["city_code", "dow"])["search_count"].mean().reset_index()
    city_mean = df.groupby("city_code")["search_count"].mean().to_dict()
    dow_city["mult"] = (
        dow_city["search_count"]
        / dow_city["city_code"].map(city_mean).replace(0, 1)
    ).clip(0.7, 1.5)
    dow_dict = dow_city.set_index(["city_code", "dow"])["mult"].to_dict()

    unique_checkins = df.groupby("city_code")["checkin"].nunique().to_dict()
    city_priors = {
        city: total / max(1, unique_checkins.get(city, 1))
        for city, total in tot_city.items()
    }
    return city_rates, cluster_rates, global_rates, city_priors, dow_dict, tot_city


MOMENTUM_MIN_PRIOR = 50


def compute_momentum_raw(hist_search_df, cutoff_dt):
    recent_start = cutoff_dt - pd.Timedelta(days=30)
    prior_start = cutoff_dt - pd.Timedelta(days=60)
    recent = hist_search_df[
        (hist_search_df["log_date"] > recent_start)
        & (hist_search_df["log_date"] <= cutoff_dt)
    ]
    prior = hist_search_df[
        (hist_search_df["log_date"] > prior_start)
        & (hist_search_df["log_date"] <= recent_start)
    ]
    recent_by_city = recent.groupby("city_code")["search_count"].sum()
    prior_by_city = prior.groupby("city_code")["search_count"].sum()
    global_recent = recent["search_count"].sum()
    global_prior = prior["search_count"].sum()
    global_momentum = float(global_recent / global_prior) if global_prior > 0 else 1.0
    momentum = {}
    for city in set(recent_by_city.index) | set(prior_by_city.index):
        prior_value = prior_by_city.get(city, 0)
        momentum[city] = (
            float(recent_by_city.get(city, 0) / prior_value)
            if prior_value >= MOMENTUM_MIN_PRIOR
            else global_momentum
        )
    return momentum, global_momentum


def apply_momentum(raw_value, damping, clip_min=0.65, clip_max=1.6):
    return np.clip(raw_value, clip_min, clip_max) ** damping


def compute_volume_tier_multipliers(
    snap_df, tot_city, low_mult=1.9, mid_mult=1.0, high_mult=0.55
):
    if "demand_tier" in snap_df.columns:
        tier_map = snap_df.set_index("city_code")["demand_tier"].to_dict()
        lookup = {
            "low": low_mult,
            "medium": mid_mult,
            "mid": mid_mult,
            "high": high_mult,
        }
        return {
            city: lookup.get(str(tier).lower(), mid_mult)
            for city, tier in tier_map.items()
        }
    volumes = pd.Series(tot_city, dtype=float)
    if volumes.empty:
        return {}
    q_low, q_high = volumes.quantile([0.33, 0.66]).values
    return {
        city: (
            low_mult
            if value <= q_low
            else mid_mult
            if value <= q_high
            else high_mult
        )
        for city, value in tot_city.items()
    }


def compute_demand_caps_raw(
    hist_search_df, city_to_cluster, cap_quantile=0.99, min_cap=5.0
):
    totals = (
        hist_search_df.groupby(["city_code", "checkin"])["search_count"]
        .sum()
        .reset_index()
    )
    totals["cluster"] = (
        totals["city_code"].map(city_to_cluster).fillna(-1).astype(int)
    )
    city_caps = (
        totals.groupby("city_code")["search_count"]
        .quantile(cap_quantile)
        .clip(lower=min_cap)
        .to_dict()
    )
    cluster_caps = (
        totals.groupby("cluster")["search_count"]
        .quantile(cap_quantile)
        .clip(lower=min_cap)
        .to_dict()
    )
    global_cap = max(
        float(totals["search_count"].quantile(cap_quantile)), min_cap
    )
    return city_caps, cluster_caps, global_cap


def compute_dynamic_lambda(h, tier_mult, base_lambda, horizon_scale, horizon_power):
    return (
        base_lambda + horizon_scale * (max(h, 0) / 30.0) ** horizon_power
    ) * tier_mult


def _build_full_evaluation_grid(snap, cutoff_dt):
    city_codes = np.sort(snap["city_code"].unique())
    if len(city_codes) != EXPECTED_CITY_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_CITY_COUNT} cities at {cutoff_dt.date()}, "
            f"got {len(city_codes)}"
        )
    target_dates = pd.date_range(
        cutoff_dt + pd.Timedelta(days=1), periods=FORECAST_DAYS
    )
    grid = pd.MultiIndex.from_product(
        [city_codes, target_dates], names=["city_code", "checkin"]
    ).to_frame(index=False)

    actual = (
        search_df[search_df["checkin"].isin(target_dates)]
        .groupby(["city_code", "checkin"])["search_count"]
        .sum()
        .rename("actual")
        .reset_index()
    )
    observed = (
        search_df[
            search_df["checkin"].isin(target_dates)
            & (search_df["log_date"] <= cutoff_dt)
        ]
        .groupby(["city_code", "checkin"])["search_count"]
        .sum()
        .rename("obs")
        .reset_index()
    )
    evaluation = grid.merge(actual, on=["city_code", "checkin"], how="left")
    evaluation = evaluation.merge(
        observed, on=["city_code", "checkin"], how="left"
    )
    evaluation[["actual", "obs"]] = evaluation[["actual", "obs"]].fillna(0.0)
    evaluation["h"] = (evaluation["checkin"] - cutoff_dt).dt.days
    evaluation["dow"] = evaluation["checkin"].dt.dayofweek
    evaluation = evaluation.merge(
        snap[["city_code", "city_reliability", "fit_eligible"]],
        on="city_code",
        how="left",
        validate="many_to_one",
    )
    expected_rows = EXPECTED_CITY_COUNT * FORECAST_DAYS
    if len(evaluation) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, got {len(evaluation)}")
    if evaluation["actual"].sum() <= 0:
        raise ValueError(f"Zero actual demand for labelled cutoff {cutoff_dt.date()}")
    return evaluation


def prepare_backtest_data(n_clusters=4, cutoffs=None):
    cutoffs = labelled_cutoffs if cutoffs is None else [pd.Timestamp(c) for c in cutoffs]
    invalid = [
        cutoff
        for cutoff in cutoffs
        if cutoff + pd.Timedelta(days=FORECAST_DAYS) > maximum_historical_checkin
    ]
    if invalid:
        raise ValueError(f"Unlabelled/incomplete cutoffs cannot be scored: {invalid}")

    prepared = []
    for cutoff_dt in cutoffs:
        snap = snapshots_df[
            snapshots_df["snapshot_cutoff"] == cutoff_dt
        ].copy()
        if snap.empty or not snap["snapshot_cutoff"].eq(cutoff_dt).all():
            raise AssertionError(f"No exact feature snapshot for {cutoff_dt.date()}")

        train_cities = snap[snap["fit_eligible"] == 1].copy()
        imputer = SimpleImputer(strategy="median")
        train_raw = imputer.fit_transform(train_cities[clustering_features])
        train_imp = np.sign(train_raw) * np.log1p(np.abs(train_raw))
        all_raw = imputer.transform(snap[clustering_features])
        all_imp = np.sign(all_raw) * np.log1p(np.abs(all_raw))
        scaler = RobustScaler()
        train_scaled = scaler.fit_transform(train_imp)
        all_scaled = scaler.transform(all_imp)
        km = KMeans(
            n_clusters=n_clusters, random_state=42, n_init=15
        ).fit(train_scaled)
        assignments = km.predict(all_scaled)
        city_to_cluster = dict(zip(snap["city_code"], assignments))

        hist_data = search_df[search_df["checkin"] <= cutoff_dt].copy()
        if (
            hist_data["checkin"].max() > cutoff_dt
            or hist_data["log_date"].max() > cutoff_dt
        ):
            raise AssertionError(
                f"Post-cutoff curve data detected at {cutoff_dt.date()}"
            )
        evaluation = _build_full_evaluation_grid(snap, cutoff_dt)
        curve_parts = compute_curves(hist_data, city_to_cluster)
        (
            city_rates,
            cluster_rates,
            global_rates,
            city_priors,
            dow_dict,
            tot_city,
        ) = curve_parts
        momentum_raw, global_momentum_raw = compute_momentum_raw(
            hist_data, cutoff_dt
        )
        city_caps_raw, cluster_caps_raw, global_cap_raw = compute_demand_caps_raw(
            hist_data, city_to_cluster
        )
        prepared.append(
            dict(
                cutoff=cutoff_dt,
                eval_df=evaluation,
                city_rates=city_rates,
                cluster_rates=cluster_rates,
                global_rates=global_rates,
                city_priors=city_priors,
                dow_dict=dow_dict,
                tier_mult=compute_volume_tier_multipliers(snap, tot_city),
                momentum_raw=momentum_raw,
                global_momentum_raw=global_momentum_raw,
                city_caps_raw=city_caps_raw,
                cluster_caps_raw=cluster_caps_raw,
                global_cap_raw=global_cap_raw,
                city_to_cluster=city_to_cluster,
                assignments=pd.DataFrame(
                    {
                        "city_code": snap["city_code"].to_numpy(),
                        "cluster": assignments,
                        "city_reliability": snap["city_reliability"].to_numpy(),
                        "fit_eligible": snap["fit_eligible"].to_numpy(),
                    }
                ),
                fit_city_count=len(train_cities),
                imputer_fit_scope_ok=bool(
                    len(train_raw) == len(train_cities)
                    and train_cities["fit_eligible"].eq(1).all()
                ),
                scaler_fit_scope_ok=bool(len(train_scaled) == len(train_cities)),
                kmeans_fit_scope_ok=bool(len(train_scaled) == len(train_cities)),
                snapshot_cutoff_ok=bool(
                    snap["snapshot_cutoff"].eq(cutoff_dt).all()
                ),
                curve_history_ok=bool(
                    hist_data["checkin"].le(cutoff_dt).all()
                    and hist_data["log_date"].le(cutoff_dt).all()
                ),
            )
        )
    return prepared


def _predict_bundle(
    bundle,
    fallback_policy="city_cluster_global",
    base_lambda=0.015,
    horizon_scale=0.10,
    horizon_power=1.6,
    momentum_damping=0.6,
    cap_headroom=1.35,
    flat_lambda=0.02,
    use_dynamic_lambda=True,
    use_momentum=True,
    use_caps=True,
):
    """Score one cutoff with either the cluster or no-cluster hierarchy."""
    if fallback_policy not in {"city_cluster_global", "city_global"}:
        raise ValueError(f"Unknown fallback policy: {fallback_policy}")

    predictions = []
    for row in bundle["eval_df"].itertuples(index=False):
        city = int(row.city_code)
        h = min(int(row.h), 59)
        observed = float(row.obs)
        cluster = bundle["city_to_cluster"].get(city, -1)

        rate = bundle["city_rates"].get(city, {}).get(h)
        if (
            rate is None or rate <= 0.002
        ) and fallback_policy == "city_cluster_global":
            rate = bundle["cluster_rates"].get(cluster, {}).get(h)
        if rate is None or rate <= 0.002:
            rate = bundle["global_rates"].get(h, 1.0)
        rate = max(rate, 0.005)

        prior = bundle["city_priors"].get(city, 0.0)
        dow_mult = bundle["dow_dict"].get((city, int(row.dow)), 1.0)
        if use_momentum:
            raw_momentum = bundle["momentum_raw"].get(
                city, bundle["global_momentum_raw"]
            )
            prior_adj = (
                prior
                * dow_mult
                * apply_momentum(raw_momentum, momentum_damping)
            )
        else:
            prior_adj = prior * dow_mult

        if use_dynamic_lambda:
            lam = compute_dynamic_lambda(
                h,
                bundle["tier_mult"].get(city, 1.0),
                base_lambda,
                horizon_scale,
                horizon_power,
            )
        else:
            lam = flat_lambda
        pred_raw = (observed + lam * prior_adj) / (rate + lam)

        if use_caps:
            if city in bundle["city_caps_raw"]:
                cap = bundle["city_caps_raw"][city]
            elif fallback_policy == "city_cluster_global":
                cap = bundle["cluster_caps_raw"].get(
                    cluster, bundle["global_cap_raw"]
                )
            else:
                cap = bundle["global_cap_raw"]
            upper_bound = max(cap * cap_headroom, observed)
        else:
            upper_bound = np.inf
        predictions.append(min(max(pred_raw, observed), upper_bound))
    return np.asarray(predictions)


def summarize_cutoff_metrics(cutoff_metrics):
    pooled = (
        cutoff_metrics["abs_error"].sum()
        / cutoff_metrics["actual_total"].sum()
    )
    worst_idx = cutoff_metrics["wape"].idxmax()
    return pd.DataFrame(
        [
            {
                "n_labelled_cutoffs": len(cutoff_metrics),
                "mean_wape": cutoff_metrics["wape"].mean(),
                "median_wape": cutoff_metrics["wape"].median(),
                "pooled_wape": pooled,
                "std_wape": cutoff_metrics["wape"].std(ddof=0),
                "worst_cutoff": cutoff_metrics.loc[worst_idx, "cutoff"],
                "worst_cutoff_wape": cutoff_metrics.loc[worst_idx, "wape"],
            }
        ]
    )


def score_config(prepared, fallback_policy="city_cluster_global", **score_kwargs):
    params = DEFAULT_SCORE_PARAMS.copy()
    params.update(score_kwargs)
    rows = []
    for bundle in prepared:
        predicted = _predict_bundle(
            bundle, fallback_policy=fallback_policy, **params
        )
        frame = bundle["eval_df"]
        abs_error, actual_total, wape = calculate_wape_components(
            frame["actual"], predicted
        )
        horizon_wapes = {}
        for name, mask in {
            "h1_7_wape": frame["h"].between(1, 7),
            "h8_15_wape": frame["h"].between(8, 15),
            "h16_30_wape": frame["h"].between(16, 30),
        }.items():
            horizon_wapes[name] = calculate_wape(
                frame.loc[mask, "actual"], predicted[mask]
            )
        rows.append(
            {
                "cutoff": bundle["cutoff"],
                "n_rows": len(frame),
                "actual_total": actual_total,
                "abs_error": abs_error,
                "wape": wape,
                **horizon_wapes,
            }
        )
    cutoff_metrics = pd.DataFrame(rows)
    return cutoff_metrics, summarize_cutoff_metrics(cutoff_metrics)


def evaluate_clustering_configuration(
    n_clusters=4, fallback_policy="city_cluster_global"
):
    prepared = prepare_backtest_data(n_clusters=n_clusters)
    cutoff_metrics, summary = score_config(
        prepared, fallback_policy=fallback_policy
    )
    return prepared, cutoff_metrics, summary


def validation_checks(prepared):
    expected_rows = EXPECTED_CITY_COUNT * FORECAST_DAYS
    rows = [
        {
            "check": "competition cutoff excluded",
            "passed": pd.Timestamp("2025-11-21") not in labelled_cutoffs,
        },
        {
            "check": "exactly nine labelled cutoffs",
            "passed": len(labelled_cutoffs) == 9,
        },
        {
            "check": "city_code absent from features",
            "passed": "city_code" not in clustering_features,
        },
    ]
    for bundle in prepared:
        cutoff = bundle["cutoff"]
        cutoff_label = cutoff.date()
        frame = bundle["eval_df"]
        expected_fit_count = int(
            snapshots_df.loc[
                snapshots_df["snapshot_cutoff"].eq(cutoff), "fit_eligible"
            ].sum()
        )
        rows.extend(
            [
                {
                    "check": f"{cutoff_label}: complete 30-day label window",
                    "passed": cutoff + pd.Timedelta(days=30)
                    <= maximum_historical_checkin,
                },
                {
                    "check": f"{cutoff_label}: 9,630 evaluation rows",
                    "passed": len(frame) == expected_rows,
                },
                {
                    "check": f"{cutoff_label}: positive actual denominator",
                    "passed": frame["actual"].sum() > 0,
                },
                {
                    "check": f"{cutoff_label}: eligible-only fit population count",
                    "passed": bundle["fit_city_count"] == expected_fit_count,
                },
                {
                    "check": f"{cutoff_label}: imputer fit only on eligible rows",
                    "passed": bundle["imputer_fit_scope_ok"],
                },
                {
                    "check": f"{cutoff_label}: scaler fit only on eligible rows",
                    "passed": bundle["scaler_fit_scope_ok"],
                },
                {
                    "check": f"{cutoff_label}: KMeans fit only on eligible rows",
                    "passed": bundle["kmeans_fit_scope_ok"],
                },
                {
                    "check": f"{cutoff_label}: historical curves use no future rows",
                    "passed": bundle["curve_history_ok"],
                },
                {
                    "check": f"{cutoff_label}: exact cutoff snapshot used",
                    "passed": bundle["snapshot_cutoff_ok"],
                },
            ]
        )
    checks = pd.DataFrame(rows)
    if not checks["passed"].all():
        raise AssertionError(checks.loc[~checks["passed"]].to_string(index=False))
    return checks


def k_sensitivity(k_values=K_VALUES):
    """Descriptive K comparison; this function does not select a winner."""
    details = {}
    summaries = []
    for k in k_values:
        prepared, cutoff_metrics, summary = evaluate_clustering_configuration(k)
        assignments = [
            {
                "cutoff": bundle["cutoff"],
                "frame": bundle["assignments"].copy(),
            }
            for bundle in prepared
        ]
        all_sizes = [
            item["frame"]["cluster"].value_counts() for item in assignments
        ]
        row = summary.iloc[0].to_dict()
        row.update(
            {
                "k": k,
                "min_cluster_size": int(
                    min(int(sizes.min()) for sizes in all_sizes)
                ),
                "largest_cluster_share": float(
                    max(int(sizes.max()) for sizes in all_sizes)
                    / EXPECTED_CITY_COUNT
                ),
            }
        )
        summaries.append(row)
        details[k] = {
            "cutoff_metrics": cutoff_metrics,
            "assignments": assignments,
            "prepared": prepared if k == 4 else None,
        }
    columns = [
        "k",
        "mean_wape",
        "median_wape",
        "pooled_wape",
        "std_wape",
        "worst_cutoff",
        "worst_cutoff_wape",
        "min_cluster_size",
        "largest_cluster_share",
    ]
    return pd.DataFrame(summaries)[columns].sort_values("k"), details


def temporal_k_selection(
    k_details, minimum_history=3, practical_threshold_pp=0.01
):
    """Select K using only prior cutoffs, then report the next held-out score."""
    threshold_fraction = practical_threshold_pp / 100.0
    rows = []
    for heldout_index in range(minimum_history, len(labelled_cutoffs)):
        heldout = labelled_cutoffs[heldout_index]
        prior_cutoffs = set(labelled_cutoffs[:heldout_index])
        candidates = []
        for k, detail in k_details.items():
            metrics = detail["cutoff_metrics"]
            prior = metrics[metrics["cutoff"].isin(prior_cutoffs)]
            prior_pooled = (
                prior["abs_error"].sum() / prior["actual_total"].sum()
            )
            heldout_row = metrics[metrics["cutoff"].eq(heldout)].iloc[0]
            candidates.append((k, prior_pooled, heldout_row))
        candidates.sort(key=lambda item: item[1])

        # Select the exact prior-data minimum. We deliberately do not replace it
        # with the smaller K when differences fall below the practical threshold.
        selected_k, selected_prior, heldout_row = candidates[0]
        prior_spread = candidates[-1][1] - candidates[0][1]
        rows.append(
            {
                "heldout_cutoff": heldout,
                "n_prior_cutoffs": heldout_index,
                "selected_k": selected_k,
                "prior_pooled_wape": selected_prior,
                "next_best_gap": candidates[1][1] - selected_prior,
                "prior_k_spread": prior_spread,
                "practically_negligible": prior_spread
                <= threshold_fraction,
                "heldout_actual_total": heldout_row["actual_total"],
                "heldout_abs_error": heldout_row["abs_error"],
                "heldout_wape": heldout_row["wape"],
            }
        )
    result = pd.DataFrame(rows)
    pooled = (
        result["heldout_abs_error"].sum()
        / result["heldout_actual_total"].sum()
    )
    summary = pd.DataFrame(
        [
            {
                "n_heldout_cutoffs": len(result),
                "mean_heldout_wape": result["heldout_wape"].mean(),
                "median_heldout_wape": result["heldout_wape"].median(),
                "pooled_heldout_wape": pooled,
                "std_heldout_wape": result["heldout_wape"].std(ddof=0),
            }
        ]
    )
    return result, summary


def clustering_ablation(prepared):
    """Remove both cluster curve and cluster cap fallback in variant B."""
    cluster_metrics, _ = score_config(
        prepared, fallback_policy="city_cluster_global"
    )
    no_cluster_metrics, _ = score_config(
        prepared, fallback_policy="city_global"
    )
    result = cluster_metrics.merge(
        no_cluster_metrics, on="cutoff", suffixes=("_cluster", "_no_cluster")
    )
    result["delta_wape_no_cluster_minus_cluster"] = (
        result["wape_no_cluster"] - result["wape_cluster"]
    )
    result["delta_abs_error_no_cluster_minus_cluster"] = (
        result["abs_error_no_cluster"] - result["abs_error_cluster"]
    )
    pooled_cluster = (
        result["abs_error_cluster"].sum()
        / result["actual_total_cluster"].sum()
    )
    pooled_no_cluster = (
        result["abs_error_no_cluster"].sum()
        / result["actual_total_no_cluster"].sum()
    )
    summary = pd.DataFrame(
        [
            {
                "cutoffs_cluster_wins": int(
                    (result["delta_abs_error_no_cluster_minus_cluster"] > 0).sum()
                ),
                "mean_delta_wape": result[
                    "delta_wape_no_cluster_minus_cluster"
                ].mean(),
                "pooled_cluster_wape": pooled_cluster,
                "pooled_no_cluster_wape": pooled_no_cluster,
                "pooled_delta": pooled_no_cluster - pooled_cluster,
            }
        ]
    )
    keep = [
        "cutoff",
        "wape_cluster",
        "wape_no_cluster",
        "delta_wape_no_cluster_minus_cluster",
        "delta_abs_error_no_cluster_minus_cluster",
    ]
    return result[keep], summary


def _optimal_label_agreement(previous, current, k):
    """Maximum same-label agreement after exact label permutation."""
    overlap = np.zeros((k, k), dtype=int)
    for old_label, new_label in zip(previous, current):
        overlap[int(old_label), int(new_label)] += 1

    # Exact bitmask assignment is cheap for k <= 10 and avoids a SciPy dependency.
    scores = {0: (0, [])}
    for old_label in range(k):
        next_scores = {}
        for mask, (score, assignment) in scores.items():
            for new_label in range(k):
                if mask & (1 << new_label):
                    continue
                new_mask = mask | (1 << new_label)
                candidate = (
                    score + overlap[old_label, new_label],
                    assignment + [new_label],
                )
                if (
                    new_mask not in next_scores
                    or candidate[0] > next_scores[new_mask][0]
                ):
                    next_scores[new_mask] = candidate
        scores = next_scores
    matched = scores[(1 << k) - 1][0]
    return matched / len(previous)


def stability_diagnostics(k_details):
    rows = []
    size_rows = []
    for k, detail in k_details.items():
        assignments = detail["assignments"]
        for item in assignments:
            counts = item["frame"]["cluster"].value_counts()
            size_rows.append(
                {
                    "k": k,
                    "cutoff": item["cutoff"],
                    "min_cluster_size": int(counts.min()),
                    "max_cluster_size": int(counts.max()),
                    "largest_cluster_share": float(
                        counts.max() / EXPECTED_CITY_COUNT
                    ),
                }
            )
        for before, after in zip(assignments[:-1], assignments[1:]):
            merged = before["frame"].merge(
                after["frame"],
                on="city_code",
                suffixes=("_before", "_after"),
                validate="one_to_one",
            )
            populations = {
                "all_cities": np.ones(len(merged), dtype=bool),
                "reliable_at_both": merged["fit_eligible_before"].eq(1)
                & merged["fit_eligible_after"].eq(1),
            }
            for population, mask in populations.items():
                part = merged.loc[mask]
                old = part["cluster_before"].to_numpy()
                new = part["cluster_after"].to_numpy()
                agreement = _optimal_label_agreement(old, new, k)
                rows.append(
                    {
                        "k": k,
                        "cutoff_from": before["cutoff"],
                        "cutoff_to": after["cutoff"],
                        "population": population,
                        "n_cities": len(part),
                        "adjusted_rand_index": adjusted_rand_score(old, new),
                        "normalized_mutual_info": normalized_mutual_info_score(
                            old, new
                        ),
                        "reassignment_rate": 1.0 - agreement,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(size_rows)


def _print_table(title, frame, percent_columns=()):
    print(f"\n{title}\n" + "=" * len(title))
    formatters = {
        column: (lambda value: f"{value:.6%}") for column in percent_columns
    }
    print(frame.to_string(index=False, formatters=formatters))


def main():
    k_table, k_details = k_sensitivity()
    k4_prepared = k_details[4]["prepared"]
    validation = validation_checks(k4_prepared)
    cutoff_metrics = k_details[4]["cutoff_metrics"]
    k4_summary = summarize_cutoff_metrics(cutoff_metrics)
    temporal, temporal_summary = temporal_k_selection(k_details)
    ablation, ablation_summary = clustering_ablation(k4_prepared)
    stability, cluster_sizes = stability_diagnostics(k_details)
    stability_summary = (
        stability.groupby(["k", "population"])
        .agg(
            adjusted_rand_mean=("adjusted_rand_index", "mean"),
            adjusted_rand_min=("adjusted_rand_index", "min"),
            normalized_mutual_info_mean=("normalized_mutual_info", "mean"),
            reassignment_rate_mean=("reassignment_rate", "mean"),
            reassignment_rate_max=("reassignment_rate", "max"),
        )
        .reset_index()
    )

    _print_table(
        "K=4 corrected per-cutoff metrics",
        cutoff_metrics,
        ("wape", "h1_7_wape", "h8_15_wape", "h16_30_wape"),
    )
    _print_table(
        "K=4 corrected summary",
        k4_summary,
        (
            "mean_wape",
            "median_wape",
            "pooled_wape",
            "std_wape",
            "worst_cutoff_wape",
        ),
    )
    _print_table(
        "K=3..10 sensitivity (descriptive only; no validated winner)",
        k_table,
        (
            "mean_wape",
            "median_wape",
            "pooled_wape",
            "std_wape",
            "worst_cutoff_wape",
            "largest_cluster_share",
        ),
    )
    _print_table(
        "Expanding-window temporal K selection",
        temporal,
        (
            "prior_pooled_wape",
            "next_best_gap",
            "prior_k_spread",
            "heldout_wape",
        ),
    )
    _print_table(
        "Temporal K selection summary",
        temporal_summary,
        (
            "mean_heldout_wape",
            "median_heldout_wape",
            "pooled_heldout_wape",
            "std_heldout_wape",
        ),
    )
    _print_table(
        "True K=4 clustering ablation",
        ablation,
        (
            "wape_cluster",
            "wape_no_cluster",
            "delta_wape_no_cluster_minus_cluster",
        ),
    )
    _print_table(
        "K=4 clustering ablation summary",
        ablation_summary,
        (
            "mean_delta_wape",
            "pooled_cluster_wape",
            "pooled_no_cluster_wape",
            "pooled_delta",
        ),
    )
    _print_table(
        "Adjacent-cutoff stability summary",
        stability_summary,
        ("reassignment_rate_mean", "reassignment_rate_max"),
    )
    _print_table("Cluster-size diagnostics", cluster_sizes)

    print("\nValidation checks\n=================")
    print(validation.to_string(index=False))
    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
