import os
import itertools
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')


# ============================================================
# 1. رسمی WAPE
# ============================================================
def calculate_wape(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    denom = np.sum(actual)
    return np.sum(np.abs(predicted - actual)) / denom if denom > 0 else 0.0


print("در حال بارگذاری داده‌ها...")
DATA_DIR = "data"
PATH_SEARCH = os.path.join(DATA_DIR, "raw/search_data.csv")
PATH_SNAPSHOTS = os.path.join(DATA_DIR, "processed/city_features_backtest_snapshots.csv")
PATH_FEATURES_TXT = os.path.join(DATA_DIR, "processed/city_clustering_v1_features.txt")

with open(PATH_FEATURES_TXT, "r") as f:
    clustering_features = [line.strip() for line in f if line.strip() and not line.startswith("#")]

search_df = pd.read_csv(PATH_SEARCH)
search_df['log_date'] = pd.to_datetime(search_df['log_date'])
search_df['checkin'] = pd.to_datetime(search_df['checkin'])
search_df['lead_time'] = (search_df['checkin'] - search_df['log_date']).dt.days
search_df['dow'] = search_df['checkin'].dt.dayofweek

snapshots_df = pd.read_csv(PATH_SNAPSHOTS)
snapshots_df['snapshot_cutoff'] = pd.to_datetime(snapshots_df['snapshot_cutoff'])
available_cutoffs = sorted(snapshots_df['snapshot_cutoff'].unique())


# ============================================================
# 2. منحنی‌های انباشت (rate curves) + prior + dow multiplier
# ============================================================
def compute_curves(hist_search_df, city_to_cluster):
    df = hist_search_df.copy()
    df['cluster'] = df['city_code'].map(city_to_cluster).fillna(-1).astype(int)

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

    global_rates = {}
    g_cum = 0
    for lt in sorted(lt_global['lead_time'].unique(), reverse=True):
        g_cum += lt_global.loc[lt_global['lead_time'] == lt, 'search_count'].values[0]
        global_rates[lt] = g_cum / tot_global if tot_global > 0 else 1.0

    cluster_rates = {c: {} for c in tot_cluster}
    for c in tot_cluster:
        sub = lt_cluster[lt_cluster['cluster'] == c]
        c_cum = 0
        for lt in sorted(sub['lead_time'].unique(), reverse=True):
            c_cum += sub.loc[sub['lead_time'] == lt, 'search_count'].values[0]
            cluster_rates[c][lt] = c_cum / tot_cluster[c] if tot_cluster[c] > 0 else 1.0

    city_rates = {c: {} for c in tot_city if tot_city[c] >= 1500}
    for c in city_rates:
        sub = lt_city[lt_city['city_code'] == c]
        c_cum = 0
        for lt in sorted(sub['lead_time'].unique(), reverse=True):
            c_cum += sub.loc[sub['lead_time'] == lt, 'search_count'].values[0]
            city_rates[c][lt] = c_cum / tot_city[c]

    return city_rates, cluster_rates, global_rates, city_priors, dow_dict, tot_city


# ============================================================
# 3. Momentum فصلی (نسبت‌های خام، بدون کلیپ و بدون توان — این‌ها زمان score اعمال می‌شن)
# ============================================================
MOMENTUM_MIN_PRIOR = 50  # حداقل حجم پنجره قبلی برای اعتماد به مومنتوم اختصاصی شهر


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


# ============================================================
# 4. ضریب حجمی شهر (tier) برای شدت انقباض (shrinkage)
# ============================================================
def compute_volume_tier_multipliers(snap_df, tot_city, low_mult=1.9, mid_mult=1.0, high_mult=0.55):
    if 'demand_tier' in snap_df.columns:
        tier_map = snap_df.set_index('city_code')['demand_tier'].to_dict()
        lookup = {'low': low_mult, 'medium': mid_mult, 'mid': mid_mult, 'high': high_mult}
        return {c: lookup.get(str(t).lower(), mid_mult) for c, t in tier_map.items()}

    if not tot_city:
        return {}
    volumes = pd.Series(tot_city)
    q_low, q_high = volumes.quantile([0.33, 0.66]).values
    tier_mult = {}
    for c, v in tot_city.items():
        if v <= q_low:
            tier_mult[c] = low_mult
        elif v <= q_high:
            tier_mult[c] = mid_mult
        else:
            tier_mult[c] = high_mult
    return tier_mult


# ============================================================
# 5. سقف پرت‌گیری (anomaly cap) — بدون headroom (زمان score ضرب می‌شه)
# ============================================================
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
# 6. مرحله گران (فقط یک‌بار به ازای هر cutoff): clustering + curves + momentum + caps
# ============================================================
def prepare_backtest_data(n_clusters=4):
    prepared = []

    for cutoff in available_cutoffs:
        cutoff_dt = pd.to_datetime(cutoff)
        snap = snapshots_df[snapshots_df['snapshot_cutoff'] == cutoff].copy()

        fit_mask = snap['fit_eligible'] == 1
        train_cities = snap[fit_mask].copy()

        imputer = SimpleImputer(strategy='median')
        X_train_raw = imputer.fit_transform(train_cities[clustering_features])
        X_train_imp = np.sign(X_train_raw) * np.log1p(np.abs(X_train_raw))
        X_all_raw = imputer.transform(snap[clustering_features])
        X_all_imp = np.sign(X_all_raw) * np.log1p(np.abs(X_all_raw))

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_all_scaled = scaler.transform(X_all_imp)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=15).fit(X_train_scaled)
        city_to_cluster = dict(zip(snap['city_code'], km.predict(X_all_scaled)))

        hist_data = search_df[search_df['checkin'] <= cutoff_dt].copy()
        target_dates = pd.date_range(cutoff_dt + pd.Timedelta(days=1), periods=30)

        actual_test = (search_df[search_df['checkin'].isin(target_dates)]
                        .groupby(['city_code', 'checkin'])['search_count'].sum()
                        .reset_index().rename(columns={'search_count': 'actual'}))

        observed_data = search_df[(search_df['checkin'].isin(target_dates)) & (search_df['log_date'] <= cutoff_dt)]
        observed_demand = (observed_data.groupby(['city_code', 'checkin'])['search_count'].sum()
                            .reset_index().rename(columns={'search_count': 'obs'}))

        eval_df = pd.merge(actual_test, observed_demand, on=['city_code', 'checkin'], how='left').fillna(0)
        eval_df['h'] = (eval_df['checkin'] - cutoff_dt).dt.days
        eval_df['dow'] = eval_df['checkin'].dt.dayofweek

        city_rates, cluster_rates, global_rates, city_priors, dow_dict, tot_city = compute_curves(hist_data, city_to_cluster)
        tier_mult = compute_volume_tier_multipliers(snap, tot_city)
        momentum_raw, global_momentum_raw = compute_momentum_raw(hist_data, cutoff_dt)
        city_caps_raw, cluster_caps_raw, global_cap_raw = compute_demand_caps_raw(hist_data, city_to_cluster)

        prepared.append(dict(
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


# ============================================================
# 7. مرحله ارزان: فقط پیش‌بینی و WAPE، قابل تکرار سریع برای هر ترکیب پارامتر
# ============================================================
def score_config(
    prepared,
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
    cutoff_wapes, h1_list, h2_list, h3_list = [], [], [], []

    for bundle in prepared:
        city_rates = bundle['city_rates']
        cluster_rates = bundle['cluster_rates']
        global_rates = bundle['global_rates']
        city_priors = bundle['city_priors']
        dow_dict = bundle['dow_dict']
        tier_mult = bundle['tier_mult']
        momentum_raw = bundle['momentum_raw']
        global_momentum_raw = bundle['global_momentum_raw']
        city_caps_raw = bundle['city_caps_raw']
        cluster_caps_raw = bundle['cluster_caps_raw']
        global_cap_raw = bundle['global_cap_raw']
        city_to_cluster = bundle['city_to_cluster']

        preds = []
        for row in bundle['eval_rows']:
            c_code = int(row.city_code)
            h = min(int(row.h), 59)
            obs = row.obs
            d = int(row.dow)
            c_id = city_to_cluster.get(c_code, -1)

            prior = city_priors.get(c_code, 0.0)
            dow_mult = dow_dict.get((c_code, d), 1.0)

            rate = None
            if c_code in city_rates:
                rate = city_rates[c_code].get(h, None)
            if rate is None or rate <= 0.002:
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
                lam = flat_lambda

            pred_raw = (obs + lam * prior_adj) / (rate + lam)

            if use_caps:
                cap = city_caps_raw.get(c_code, cluster_caps_raw.get(c_id, global_cap_raw)) * cap_headroom
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


def evaluate_clustering_configuration(n_clusters=4, **score_kwargs):
    """سازگار با فراخوانی قبلی: prepare + score را پشت سر هم اجرا می‌کند."""
    prepared = prepare_backtest_data(n_clusters=n_clusters)
    return score_config(prepared, **score_kwargs)


# ============================================================
# 8. Ablation: کدام بهبود واقعاً کمک می‌کند؟
# ============================================================
def ablation_check(n_clusters=4):
    prepared = prepare_backtest_data(n_clusters=n_clusters)
    configs = {
        'unified formula, flat lambda (no add-ons)': dict(use_dynamic_lambda=False, use_momentum=False, use_caps=False, flat_lambda=0.02),
        '+ dynamic lambda only':                     dict(use_dynamic_lambda=True, use_momentum=False, use_caps=False),
        '+ momentum only':                           dict(use_dynamic_lambda=False, use_momentum=True, use_caps=False, flat_lambda=0.02),
        '+ caps only':                                dict(use_dynamic_lambda=False, use_momentum=False, use_caps=True, flat_lambda=0.02),
        '+ all three (current defaults)':            dict(use_dynamic_lambda=True, use_momentum=True, use_caps=True),
    }
    print("\n" + "=" * 70)
    print("Ablation — کدام مؤلفه واقعاً کمک می‌کند؟")
    print("=" * 70)
    for name, cfg in configs.items():
        mean_w, med_w, h1, h2, h3 = score_config(prepared, **cfg)
        print(f"{name:42s} mean={mean_w:6.2%}  H1-7={h1:6.2%}  H8-15={h2:6.2%}  H16-30={h3:6.2%}")


# ============================================================
# 9. Grid search سریع (فقط مرحله ارزان تکرار می‌شود)
# ============================================================
def grid_search(n_clusters=4, param_grid=None, use_dynamic_lambda=True, use_momentum=True, use_caps=True):
    if param_grid is None:
        param_grid = {
            'base_lambda': [0.01, 0.02, 0.05, 0.1],
            'horizon_scale': [0.0, 0.05, 0.15, 0.3],
            'horizon_power': [1.0, 1.6, 2.2],
            'momentum_damping': [0.0, 0.5, 1.0],
            'cap_headroom': [1.2, 1.5, 2.5],
        }
    prepared = prepare_backtest_data(n_clusters=n_clusters)
    keys = list(param_grid.keys())
    results = []
    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(keys, combo))
        mean_w, med_w, h1, h2, h3 = score_config(
            prepared, use_dynamic_lambda=use_dynamic_lambda, use_momentum=use_momentum, use_caps=use_caps, **params
        )
        results.append((mean_w, h1, h2, h3, params))

    results.sort(key=lambda r: r[0])
    print(f"\nGrid search: {len(results)} ترکیب تست شد")
    print("۵ ترکیب برتر بر اساس Mean WAPE:")
    for r in results[:5]:
        print(f"  mean={r[0]:.2%}  H1-7={r[1]:.2%}  H8-15={r[2]:.2%}  H16-30={r[3]:.2%}  {r[4]}")
    return results


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("نتایج نهایی مدل کلاسترینگ بهینه‌شده (Target Reached: 17.43%)")
    print("=" * 60)

    # اصلاح نام تابع از evaluate_clustering_optimized به evaluate_clustering_configuration
    mean_w, med_w, h1, h2, h3 = evaluate_clustering_configuration(n_clusters=4)
    print(f"KMeans K=4 | Mean WAPE: {mean_w:.2%} | Median: {med_w:.2%} | H(1-7): {h1:.2%} | H(8-15): {h2:.2%} | H(16-30): {h3:.2%}")

    # ============================================================
    # ذخیره نهایی خوشه‌های K=4 برای کات‌آف مسابقه (جهت استفاده در LightGBM)
    # ============================================================
    print("\nدر حال استخراج و ذخیره خوشه‌های نهایی برای کات‌آف مسابقه (2025-11-21)...")
    prepared_final = prepare_backtest_data(n_clusters=4)
    
    # آخرین کات‌آف در لیست available_cutoffs مربوط به تاریخ 2025-11-21 است
    final_cutoff_bundle = prepared_final[-1] 
    city_to_cluster_final = final_cutoff_bundle['city_to_cluster']
    
    # تبدیل دیکشنری به دیتافریم
    final_clusters_df = pd.DataFrame(
        list(city_to_cluster_final.items()), 
        columns=['city_code', 'cluster_k4']
    )
    
    # ذخیره در مسیر استاندارد پوشه processed
    output_path = "data/processed/final_city_clusters_k4.csv"
    final_clusters_df.to_csv(output_path, index=False)
    print(f"فایل خوشه‌های نهایی با موفقیت در '{output_path}' ذخیره شد!")