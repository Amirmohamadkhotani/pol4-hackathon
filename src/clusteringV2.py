import os
import itertools
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# 1. فرمول رسمی WAPE
# ============================================================
def calculate_wape(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    denom = np.sum(actual)
    if denom == 0:
        raise ValueError("WAPE denominator is zero")
    return np.sum(np.abs(predicted - actual)) / denom


print("در حال بارگذاری داده‌ها برای Clustering V2...")
ROOT = Path(__file__).resolve().parents[1] if '__file__' in locals() else Path.cwd()
PATH_SEARCH = ROOT / "data" / "raw" / "search_data.csv"
PATH_SNAPSHOTS = ROOT / "data" / "processed" / "city_features_backtest_snapshots.csv"

search_df = pd.read_csv(PATH_SEARCH)
search_df['log_date'] = pd.to_datetime(search_df['log_date'])
search_df['checkin'] = pd.to_datetime(search_df['checkin'])
search_df['lead_time'] = (search_df['checkin'] - search_df['log_date']).dt.days
search_df['dow'] = search_df['checkin'].dt.dayofweek

snapshots_df = pd.read_csv(PATH_SNAPSHOTS)
snapshots_df['snapshot_cutoff'] = pd.to_datetime(snapshots_df['snapshot_cutoff'])
all_cutoffs = sorted(snapshots_df['snapshot_cutoff'].unique())
COMPETITION_CUTOFF = pd.Timestamp('2025-11-21')
EXPECTED_CITIES = 321
EXPECTED_EVAL_ROWS = EXPECTED_CITIES * 30

# 🛑 اصلاح باگ ۱: فیلتر کات‌آف‌هایی که ۳۰ روز کامل لیبل آینده در search_df دارند (حذف کات‌آف مسابقه از بک‌تست)
max_search_checkin = search_df['checkin'].max()
available_cutoffs = [
    c for c in all_cutoffs
    if pd.to_datetime(c) != COMPETITION_CUTOFF
    and pd.to_datetime(c) + pd.Timedelta(days=30) <= max_search_checkin
]

print(f"کل کات‌آف‌های اسنپ‌شات: {len(all_cutoffs)} | کات‌آف‌های معتبر برای بک‌تست (دارای لیبل ۳۰ روزه): {len(available_cutoffs)}")

# تعریف ویژگی‌های Behavioral Fingerprint V2 (بدون تسلط مقیاس خام)
v2_features = [
    'completion_D30', 'completion_D21', 'completion_D14', 'completion_D7', 'completion_D3', 'completion_D1',
    'weekday_index_mon', 'weekday_index_tue', 'weekday_index_wed', 'weekday_index_thu', 'weekday_index_fri', 'weekday_index_sat', 'weekday_index_sun',
    'weekend_index_365', 'holiday_index_365', 'demand_cv_365', 'log_recent_growth_90'
]


# ============================================================
# 2. منطق Clustering V2 (دو مرحله‌ای: Tiering حجمی + Shape Clustering)
# ============================================================
def fit_clustering_v2(snap_df):
    snapshot_cities = {int(c) for c in snap_df['city_code']}
    assert len(snap_df) == EXPECTED_CITIES, f"Expected 321 snapshot rows, got {len(snap_df)}"
    assert len(snapshot_cities) == EXPECTED_CITIES, f"Expected 321 unique cities, got {len(snapshot_cities)}"
    fit_mask = snap_df['fit_eligible'] == 1
    train_cities = snap_df[fit_mask].copy()
    all_cities = snap_df.copy()

    # مرحله ۱: تفکیک حجمی (Two-Stage Tiering)
    q_low, q_high = train_cities['total_searches'].quantile([0.33, 0.66]).values
    
    def get_tier(v):
        if v <= q_low: return 'Low'
        elif v <= q_high: return 'Med'
        else: return 'High'

    train_cities['volume_tier'] = train_cities['total_searches'].apply(get_tier)
    all_cities['volume_tier'] = all_cities['total_searches'].apply(get_tier)

    cluster_map = {}
    cluster_id_counter = 0

    imputer = SimpleImputer(strategy='median')
    scaler = RobustScaler()

    for tier in ['Low', 'Med', 'High']:
        tier_fit = train_cities[train_cities['volume_tier'] == tier]
        tier_all = all_cities[all_cities['volume_tier'] == tier]

        if len(tier_fit) < 3:
            for c in tier_all['city_code']:
                cluster_map[c] = cluster_id_counter
            cluster_id_counter += 1
            continue

        X_train = imputer.fit_transform(tier_fit[v2_features])
        X_all = imputer.transform(tier_all[v2_features])

        X_train_sc = scaler.fit_transform(X_train)
        X_all_sc = scaler.transform(X_all)

        k_sub = 3  # ۳ خوشه برای هر تیتر (مجموعاً ۹ خوشه متوازن)
        km = KMeans(n_clusters=k_sub, random_state=42, n_init=10).fit(X_train_sc)
        preds = km.predict(X_all_sc)

        for c, p in zip(tier_all['city_code'], preds):
            cluster_map[c] = cluster_id_counter + p
        cluster_id_counter += k_sub

    result = {
        int(city_code): int(cluster_id)
        for city_code, cluster_id in cluster_map.items()
    }
    assert len(result) == EXPECTED_CITIES, f"Expected 321 mappings, got {len(result)}"
    assert set(result) == snapshot_cities, "Mapping keys do not match snapshot city_code values"
    missing_valid_cities = snapshot_cities - set(result)
    assert not missing_valid_cities, f"Missing valid city mappings: {sorted(missing_valid_cities)}"
    missing_key_fallbacks = sum(city not in result for city in snapshot_cities)
    assert missing_key_fallbacks == 0, "Valid cities would be assigned through a missing-key fallback"
    return result


# ============================================================
# 3. منحنی‌های انباشت (Rate Curves) + Prior + DOW Multiplier
# ============================================================
def compute_curves(hist_search_df, city_to_cluster):
    df = hist_search_df.copy()
    valid_rows = df['city_code'].isin(city_to_cluster)
    df = df[valid_rows].copy()
    df['cluster'] = df['city_code'].map(city_to_cluster).astype(int)

    tot_global = df['search_count'].sum()
    tot_cluster = df.groupby('cluster')['search_count'].sum().to_dict()
    tot_city = df.groupby('city_code')['search_count'].sum().to_dict()

    dow_city = df.groupby(['city_code', 'dow'])['search_count'].mean().reset_index()
    city_mean = df.groupby('city_code')['search_count'].mean().to_dict()
    dow_city['mult'] = (dow_city['search_count'] / dow_city['city_code'].map(city_mean).replace(0, 1)).clip(0.7, 1.5)
    dow_dict = dow_city.set_index(['city_code', 'dow'])['mult'].to_dict()

    unique_checkins = df.groupby('city_code')['checkin'].nunique().to_dict()
    city_priors = {c: tot_city[c] / max(1, unique_checkins.get(c, 1)) for c in tot_city}

    lt_city = df.groupby(['city_code', 'lead_time'])['search_count'].sum().reset_index()
    lt_cluster = df.groupby(['cluster', 'lead_time'])['search_count'].sum().reset_index()
    lt_global = df.groupby('lead_time')['search_count'].sum().reset_index()

    lead_grid = range(60)

    def full_completion_curve(lead_totals, denominator):
        by_lead = lead_totals.set_index('lead_time')['search_count']
        # Include volume beyond day 59 in every applicable cumulative numerator,
        # while explicitly representing missing h=0..59 volume as zero.
        beyond_59 = float(by_lead[by_lead.index > 59].sum())
        grid_volume = by_lead.reindex(lead_grid, fill_value=0.0).astype(float)
        cumulative = grid_volume.iloc[::-1].cumsum().iloc[::-1] + beyond_59
        if denominator <= 0:
            return {h: 1.0 for h in lead_grid}
        return {int(h): float(cumulative.loc[h] / denominator) for h in lead_grid}

    global_rates = full_completion_curve(lt_global, tot_global)

    cluster_rates = {c: {} for c in tot_cluster}
    for c in tot_cluster:
        sub = lt_cluster[lt_cluster['cluster'] == c]
        cluster_rates[c] = full_completion_curve(sub, tot_cluster[c])

    city_rates = {c: {} for c in tot_city if tot_city[c] >= 1500}
    for c in city_rates:
        sub = lt_city[lt_city['city_code'] == c]
        city_rates[c] = full_completion_curve(sub, tot_city[c])

    return city_rates, cluster_rates, global_rates, city_priors, dow_dict, tot_city


# ============================================================
# 4. Momentum و Caps
# ============================================================
MOMENTUM_MIN_PRIOR = 50

def compute_momentum_raw(hist_search_df, cutoff_dt):
    recent_start = cutoff_dt - pd.Timedelta(days=30)
    prior_start = cutoff_dt - pd.Timedelta(days=60)

    recent = hist_search_df[(hist_search_df['log_date'] > recent_start) & (hist_search_df['log_date'] <= cutoff_dt)]
    prior = hist_search_df[(hist_search_df['log_date'] > prior_start) & (hist_search_df['log_date'] <= recent_start)]

    recent_by_city = recent.groupby('city_code')['search_count'].sum()
    prior_by_city = prior.groupby('city_code')['search_count'].sum()

    global_recent = recent['search_count'].sum()
    global_prior = prior['search_count'].sum()
    global_momentum = float(global_recent / global_prior) if global_prior > 0 else 1.0

    momentum = {}
    for c in set(recent_by_city.index) | set(prior_by_city.index):
        p = prior_by_city.get(c, 0)
        r = recent_by_city.get(c, 0)
        momentum[c] = float(r / p) if p >= MOMENTUM_MIN_PRIOR else global_momentum

    return momentum, global_momentum

def apply_momentum(raw_value, damping, clip_min=0.65, clip_max=1.6):
    clipped = np.clip(raw_value, clip_min, clip_max)
    return clipped ** damping

def compute_volume_tier_multipliers(snap_df, tot_city, low_mult=1.9, mid_mult=1.0, high_mult=0.55):
    if 'demand_tier' in snap_df.columns:
        tier_map = snap_df.set_index('city_code')['demand_tier'].to_dict()
        lookup = {'low': low_mult, 'medium': mid_mult, 'mid': mid_mult, 'high': high_mult}
        return {c: lookup.get(str(t).lower(), mid_mult) for c, t in tier_map.items()}
    return {c: mid_mult for c in tot_city}

def compute_demand_caps_raw(hist_search_df, city_to_cluster, cap_quantile=0.99, min_cap=5.0):
    df = hist_search_df.copy()
    df['cluster'] = df['city_code'].map(city_to_cluster).fillna(-1).astype(int)

    totals = df.groupby(['city_code', 'checkin'])['search_count'].sum().reset_index()
    totals['cluster'] = totals['city_code'].map(city_to_cluster).fillna(-1).astype(int)

    city_caps = totals.groupby('city_code')['search_count'].quantile(cap_quantile).clip(lower=min_cap).to_dict()
    cluster_caps = totals.groupby('cluster')['search_count'].quantile(cap_quantile).clip(lower=min_cap).to_dict()
    global_cap = max(totals['search_count'].quantile(cap_quantile), min_cap)

    return city_caps, cluster_caps, float(global_cap)

def compute_dynamic_lambda(h, tier_mult, base_lambda, horizon_scale, horizon_power):
    return (base_lambda + horizon_scale * (max(h, 0) / 30.0) ** horizon_power) * tier_mult


# ============================================================
# 5. آماده‌سازی بک‌تست با گرید کامل (321 cities × 30 days)
# ============================================================
def prepare_backtest_data_v2():
    prepared = []
    for cutoff in available_cutoffs:
        cutoff_dt = pd.to_datetime(cutoff)
        snap = snapshots_df[snapshots_df['snapshot_cutoff'] == cutoff].copy()

        city_to_cluster = fit_clustering_v2(snap)
        active_cities = snap['city_code'].astype(int).unique()
        assert len(active_cities) == EXPECTED_CITIES
        assert set(city_to_cluster) == set(active_cities)

        hist_data = search_df[search_df['checkin'] <= cutoff_dt].copy()
        target_dates = pd.date_range(cutoff_dt + pd.Timedelta(days=1), periods=30)

        # 🛑 اصلاح باگ ۲: ساخت گرید کامل تمام شهرها در ۳۰ روز آینده (جلوگیری از Sparse Evaluation فقط روی ردیف‌های پازتیو)
        full_grid = pd.DataFrame(list(itertools.product(active_cities, target_dates)), columns=['city_code', 'checkin'])

        actual_test = (search_df[search_df['checkin'].isin(target_dates)]
                       .groupby(['city_code', 'checkin'])['search_count'].sum()
                       .reset_index().rename(columns={'search_count': 'actual'}))

        observed_data = search_df[(search_df['checkin'].isin(target_dates)) & (search_df['log_date'] <= cutoff_dt)]
        observed_demand = (observed_data.groupby(['city_code', 'checkin'])['search_count'].sum()
                            .reset_index().rename(columns={'search_count': 'obs'}))

        eval_df = pd.merge(full_grid, actual_test, on=['city_code', 'checkin'], how='left').fillna({'actual': 0})
        eval_df = pd.merge(eval_df, observed_demand, on=['city_code', 'checkin'], how='left').fillna({'obs': 0})
        assert len(eval_df) == EXPECTED_EVAL_ROWS, f"Expected 9,630 rows, got {len(eval_df)} at {cutoff_dt.date()}"
        assert eval_df[['actual', 'obs']].notna().all().all(), f"Missing demand values at {cutoff_dt.date()}"
        
        eval_df['h'] = (eval_df['checkin'] - cutoff_dt).dt.days
        eval_df['dow'] = eval_df['checkin'].dt.dayofweek

        city_rates, cluster_rates, global_rates, city_priors, dow_dict, tot_city = compute_curves(hist_data, city_to_cluster)
        tier_mult = compute_volume_tier_multipliers(snap, tot_city)
        momentum_raw, global_momentum_raw = compute_momentum_raw(hist_data, cutoff_dt)
        city_caps_raw, cluster_caps_raw, global_cap_raw = compute_demand_caps_raw(hist_data, city_to_cluster)

        prepared.append(dict(
            cutoff=cutoff_dt,
            rows=len(eval_df),
            eval_rows=list(eval_df[['city_code', 'h', 'obs', 'dow']].itertuples(index=False)),
            actual=eval_df['actual'].to_numpy(),
            h_arr=eval_df['h'].to_numpy(),
            city_rates=city_rates, cluster_rates=cluster_rates, global_rates=global_rates,
            city_priors=city_priors, dow_dict=dow_dict, tier_mult=tier_mult,
            momentum_raw=momentum_raw, global_momentum_raw=global_momentum_raw,
            city_caps_raw=city_caps_raw, cluster_caps_raw=cluster_caps_raw, global_cap_raw=global_cap_raw,
            city_to_cluster=city_to_cluster,
        ))

    return prepared


def score_config(prepared, use_cluster=True, base_lambda=0.015, horizon_scale=0.10, horizon_power=1.6,
                 momentum_damping=0.6, cap_headroom=1.35, use_dynamic_lambda=True,
                 use_momentum=True, use_caps=True):
    cutoff_wapes, h1_list, h2_list, h3_list = [], [], [], []

    for bundle in prepared:
        city_rates = bundle['city_rates']
        cluster_rates = bundle['cluster_rates'] if use_cluster else {}  # اگر کلستر خاموش باشد (Ablation)
        global_rates = bundle['global_rates']
        city_priors = bundle['city_priors']
        dow_dict = bundle['dow_dict']
        tier_mult = bundle['tier_mult']
        momentum_raw = bundle['momentum_raw']
        global_momentum_raw = bundle['global_momentum_raw']
        city_caps_raw = bundle['city_caps_raw']
        cluster_caps_raw = bundle['cluster_caps_raw'] if use_cluster else {}
        global_cap_raw = bundle['global_cap_raw']
        city_to_cluster = bundle['city_to_cluster']

        preds = []
        for row in bundle['eval_rows']:
            c_code = int(row.city_code)
            h = min(int(row.h), 59)
            obs = row.obs
            d = int(row.dow)
            c_id = city_to_cluster[c_code]

            prior = city_priors.get(c_code, 0.0)
            dow_mult = dow_dict.get((c_code, d), 1.0)

            rate = None
            if c_code in city_rates:
                rate = city_rates[c_code].get(h, None)
            if use_cluster and (rate is None or rate <= 0.002):
                rate = cluster_rates.get(c_id, {}).get(h, None)
            if rate is None or rate <= 0.002:
                rate = global_rates.get(h, 1.0)
            rate = max(rate, 0.005)

            if use_momentum:
                mom_raw = momentum_raw.get(c_code, global_momentum_raw)
                prior_adj = prior * dow_mult * apply_momentum(mom_raw, momentum_damping)
            else:
                prior_adj = prior * dow_mult

            if use_dynamic_lambda:
                tmult = tier_mult.get(c_code, 1.0)
                lam = compute_dynamic_lambda(h, tmult, base_lambda, horizon_scale, horizon_power)
            else:
                lam = 0.02

            pred_raw = (obs + lam * prior_adj) / (rate + lam)

            if use_caps:
                if c_code in city_caps_raw:
                    raw_cap = city_caps_raw[c_code]
                elif use_cluster and c_id in cluster_caps_raw:
                    raw_cap = cluster_caps_raw[c_id]
                else:
                    raw_cap = global_cap_raw
                cap = raw_cap * cap_headroom
                upper_bound = max(cap, obs)
            else:
                upper_bound = np.inf

            pred = min(max(pred_raw, obs), upper_bound)
            preds.append(pred)

        preds_arr = np.array(preds)
        actual = bundle['actual']
        h_arr = bundle['h_arr']

        cutoff_wapes.append(calculate_wape(actual, preds_arr))
        h1_list.append(calculate_wape(actual[h_arr <= 7], preds_arr[h_arr <= 7]))
        h2_list.append(calculate_wape(actual[(h_arr > 7) & (h_arr <= 15)], preds_arr[(h_arr > 7) & (h_arr <= 15)]))
        h3_list.append(calculate_wape(actual[h_arr > 15], preds_arr[h_arr > 15]))

    return np.mean(cutoff_wapes), np.median(cutoff_wapes), np.mean(h1_list), np.mean(h2_list), np.mean(h3_list)


def predict_bundle(bundle, use_cluster):
    """Run the fixed model, changing only cluster fallback availability."""
    city_rates = bundle['city_rates']
    cluster_rates = bundle['cluster_rates']
    global_rates = bundle['global_rates']
    preds = []
    for row in bundle['eval_rows']:
        city = int(row.city_code)
        h = min(int(row.h), 59)
        cluster = bundle['city_to_cluster'][city]
        rate = city_rates.get(city, {}).get(h)
        if use_cluster and (rate is None or rate <= 0.002):
            rate = cluster_rates.get(cluster, {}).get(h)
        if rate is None or rate <= 0.002:
            rate = global_rates[h]
        rate = max(rate, 0.005)

        prior = bundle['city_priors'].get(city, 0.0)
        dow_mult = bundle['dow_dict'].get((city, int(row.dow)), 1.0)
        mom_raw = bundle['momentum_raw'].get(city, bundle['global_momentum_raw'])
        prior_adj = prior * dow_mult * apply_momentum(mom_raw, 0.6)
        lam = compute_dynamic_lambda(h, bundle['tier_mult'].get(city, 1.0), 0.015, 0.10, 1.6)
        pred_raw = (row.obs + lam * prior_adj) / (rate + lam)

        if city in bundle['city_caps_raw']:
            raw_cap = bundle['city_caps_raw'][city]
        elif use_cluster and cluster in bundle['cluster_caps_raw']:
            raw_cap = bundle['cluster_caps_raw'][cluster]
        else:
            raw_cap = bundle['global_cap_raw']
        upper_bound = max(raw_cap * 1.35, row.obs)
        preds.append(min(max(pred_raw, row.obs), upper_bound))
    return np.asarray(preds)


def evaluate_cluster_ablation(prepared):
    results = []
    for bundle in prepared:
        actual = bundle['actual']
        pred_with = predict_bundle(bundle, True)
        pred_without = predict_bundle(bundle, False)
        with_wape = calculate_wape(actual, pred_with)
        without_wape = calculate_wape(actual, pred_without)
        results.append({
            'cutoff': bundle['cutoff'].date().isoformat(),
            'rows': bundle['rows'],
            'actual_total': float(actual.sum()),
            'WAPE_with_cluster': with_wape,
            'WAPE_without_cluster': without_wape,
            'delta_WAPE': without_wape - with_wape,
            'abs_error_with_cluster': float(np.abs(pred_with - actual).sum()),
            'abs_error_without_cluster': float(np.abs(pred_without - actual).sum()),
        })
    return pd.DataFrame(results)


def cluster_distribution(mapping):
    counts = pd.Series(mapping).value_counts().sort_index()
    return pd.DataFrame({
        'cluster_id': counts.index.astype(int),
        'number_of_cities': counts.values.astype(int),
        'share': counts.values / EXPECTED_CITIES,
    })


def run_end_to_end():
    failures = []
    print("\nA. VALIDATION CHECKS")
    snapshot_mappings = {}
    for cutoff in all_cutoffs:
        snap = snapshots_df[snapshots_df['snapshot_cutoff'] == cutoff].copy()
        mapping = fit_clustering_v2(snap)
        snapshot_mappings[pd.Timestamp(cutoff)] = mapping
        print(f"PASS {pd.Timestamp(cutoff).date()}: 321 cities, 321 exact mappings, 0 missing/fallback mappings")
    assert COMPETITION_CUTOFF not in {pd.Timestamp(c) for c in available_cutoffs}
    print(f"PASS competition cutoff {COMPETITION_CUTOFF.date()} excluded from backtest")

    prepared_v2 = prepare_backtest_data_v2()
    assert all(bundle['rows'] == EXPECTED_EVAL_ROWS for bundle in prepared_v2)
    print(f"PASS {len(prepared_v2)} historical cutoffs each contain exactly 9,630 evaluation rows")
    print("PASS missing actual and observed demand filled with zero")
    try:
        calculate_wape([0, 0], [0, 1])
        failures.append("calculate_wape did not reject a zero denominator")
    except ValueError as exc:
        assert str(exc) == "WAPE denominator is zero"
        print("PASS zero WAPE denominator raises the required ValueError")

    ablation = evaluate_cluster_ablation(prepared_v2)
    print("\nB. PER-CUTOFF WITH vs WITHOUT")
    print(ablation.to_string(index=False, formatters={
        'WAPE_with_cluster': '{:.6%}'.format,
        'WAPE_without_cluster': '{:.6%}'.format,
        'delta_WAPE': '{:+.6%}'.format,
        'actual_total': '{:.0f}'.format,
        'abs_error_with_cluster': '{:.3f}'.format,
        'abs_error_without_cluster': '{:.3f}'.format,
    }))

    pooled_with = ablation['abs_error_with_cluster'].sum() / ablation['actual_total'].sum()
    pooled_without = ablation['abs_error_without_cluster'].sum() / ablation['actual_total'].sum()
    wins = int((ablation['delta_WAPE'] > 0).sum())
    losses = int((ablation['delta_WAPE'] < 0).sum())
    print("\nC. POOLED ABLATION")
    print(f"WITH clustering    mean={ablation['WAPE_with_cluster'].mean():.6%} median={ablation['WAPE_with_cluster'].median():.6%} pooled={pooled_with:.6%}")
    print(f"WITHOUT clustering mean={ablation['WAPE_without_cluster'].mean():.6%} median={ablation['WAPE_without_cluster'].median():.6%} pooled={pooled_without:.6%}")
    print(f"Pooled improvement={(pooled_without - pooled_with) * 100:+.6f} percentage points")
    print(f"Cutoffs won/lost={wins}/{losses}")
    print(f"Worst degradation={ablation['delta_WAPE'].min() * 100:+.6f} percentage points")
    print(f"Best improvement={ablation['delta_WAPE'].max() * 100:+.6f} percentage points")

    print("\nD. FINAL CLUSTER DISTRIBUTION")
    for cutoff in all_cutoffs:
        label = "FINAL" if pd.Timestamp(cutoff) == COMPETITION_CUTOFF else "HISTORICAL"
        print(f"\n{label} {pd.Timestamp(cutoff).date()}")
        print(cluster_distribution(snapshot_mappings[pd.Timestamp(cutoff)]).to_string(index=False, formatters={'share': '{:.6%}'.format}))

    final_mapping = snapshot_mappings[COMPETITION_CUTOFF]
    final_dist = cluster_distribution(final_mapping)
    print(f"\nnumber of clusters={len(final_dist)}")
    print(f"min cluster size={final_dist['number_of_cities'].min()}")
    print(f"max cluster size={final_dist['number_of_cities'].max()}")
    print(f"largest cluster share={final_dist['share'].max():.6%}")
    print(f"singleton clusters={(final_dist['number_of_cities'] == 1).sum()}")

    final_snap = snapshots_df[snapshots_df['snapshot_cutoff'] == COMPETITION_CUTOFF]
    final_df = pd.DataFrame(sorted(final_mapping.items()), columns=['city_code', 'cluster_v2'])
    assert set(final_df['city_code']) == set(final_snap['city_code'].astype(int))
    assert final_df['city_code'].is_monotonic_increasing
    output_path = ROOT / "data" / "processed" / "final_city_clusters_v2.csv"
    final_df.to_csv(output_path, index=False)
    print(f"PASS regenerated {output_path} with actual sorted city codes")
    print("\nE. Any failures")
    print("None" if not failures else "\n".join(failures))


if __name__ == "__main__":
    run_end_to_end()


if __name__ == "__legacy_main__":
    print("\n" + "=" * 60)
    print("ارزیابی معتبر Clustering V2 (با گرید کامل و بدون لیک‌اژ کات‌آف)")
    print("=" * 60)

    prepared_v2 = prepare_backtest_data_v2()

    # ۱. ارزیابی مدل با کلاستر V2
    mean_w, med_w, h1, h2, h3 = score_config(prepared_v2, use_cluster=True)
    print(f"\n✨ Clustering V2 (WITH Cluster)  | Mean WAPE: {mean_w:.2%} | Median: {med_w:.2%} | H(1-7): {h1:.2%} | H(8-15): {h2:.2%} | H(16-30): {h3:.2%}")

    # 🛑 🛑 ۳. تست Ablation (اثبات contribution واقعی کلاستر در برابر بدون کلاستر)
    print("\n" + "-" * 60)
    print("اجرای تست Ablation (مقایسه با و بدون کلاستر برای اثبات اثربخشی)...")
    print("-" * 60)
    mean_w_nocap, _, _, _, _ = score_config(prepared_v2, use_cluster=False)
    print(f"🔹 Baseline (WITHOUT Cluster) | Mean WAPE: {mean_w_nocap:.2%}")
    print(f"🎯 بهبود واقعی ایجاد شده توسط Clustering V2: {mean_w_nocap - mean_w:+.2%}")
    print("=" * 60)

    # ذخیره نهایی خوشه‌های V2 برای کات‌آف مسابقه (2025-11-21)
    print("\nدر حال ذخیره خوشه‌های V2 برای کات‌آف نهایی مسابقه (2025-11-21)...")
    final_snap = snapshots_df[snapshots_df['snapshot_cutoff'] == '2025-11-21'].copy()
    final_city_clusters_v2 = fit_clustering_v2(final_snap)
    
    final_df = pd.DataFrame(
        list(final_city_clusters_v2.items()),
        columns=['city_code', 'cluster_v2']
    )
    
    output_path = ROOT / "data" / "processed" / "final_city_clusters_v2.csv"
    os.makedirs(output_path.parent, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"✅ فایل Clustering V2 با موفقیت در '{output_path}' ذخیره شد!")
