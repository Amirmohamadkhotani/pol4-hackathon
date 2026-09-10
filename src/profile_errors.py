import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
PATH_SEARCH = os.path.join(DATA_DIR, "raw/search_data.csv")
PATH_SNAPSHOTS = os.path.join(DATA_DIR, "processed/city_features_backtest_snapshots.csv")
PATH_FEATURES_TXT = os.path.join(DATA_DIR, "processed/city_clustering_v1_features.txt")

search_df = pd.read_csv(PATH_SEARCH)
search_df['log_date'] = pd.to_datetime(search_df['log_date'])
search_df['checkin'] = pd.to_datetime(search_df['checkin'])
search_df['lead_time'] = (search_df['checkin'] - search_df['log_date']).dt.days

snapshots_df = pd.read_csv(PATH_SNAPSHOTS)
snapshots_df['snapshot_cutoff'] = pd.to_datetime(snapshots_df['snapshot_cutoff'])
available_cutoffs = sorted(snapshots_df['snapshot_cutoff'].unique())

def calculate_wape(actual, predicted):
    denom = np.sum(actual)
    return np.sum(np.abs(predicted - actual)) / denom if denom > 0 else 0.0

# اجرای بک‌تست با لاگ تفکیک‌شده
records = []
for cutoff in available_cutoffs:
    cutoff_dt = pd.to_datetime(cutoff)
    snap = snapshots_df[snapshots_df['snapshot_cutoff'] == cutoff]
    
    hist_data = search_df[search_df['checkin'] <= cutoff_dt]
    target_dates = pd.date_range(cutoff_dt + pd.Timedelta(days=1), periods=30)
    
    actual_test = search_df[search_df['checkin'].isin(target_dates)].groupby(['city_code', 'checkin'])['search_count'].sum().reset_index()
    actual_test.rename(columns={'search_count': 'actual'}, inplace=True)
    
    obs_data = search_df[(search_df['checkin'].isin(target_dates)) & (search_df['log_date'] <= cutoff_dt)]
    obs_demand = obs_data.groupby(['city_code', 'checkin'])['search_count'].sum().reset_index()
    obs_demand.rename(columns={'search_count': 'observed'}, inplace=True)
    
    df_eval = pd.merge(actual_test, obs_demand, on=['city_code', 'checkin'], how='left').fillna(0)
    df_eval['days_to_checkin'] = (df_eval['checkin'] - cutoff_dt).dt.days
    df_eval['cutoff'] = str(cutoff.date())
    df_eval['dayofweek'] = df_eval['checkin'].dt.day_name()
    
    # تفکیک بازه افق زمانی
    df_eval['horizon_bucket'] = pd.cut(
        df_eval['days_to_checkin'], 
        bins=[0, 7, 15, 30], 
        labels=['Horizon 1-7d', 'Horizon 8-15d', 'Horizon 16-30d']
    )
    
    # تقریب پیش‌بینی با نرخ خام
    tot_hist = hist_data['search_count'].sum()
    g_rates = {}
    g_cum = 0
    lt_g = hist_data.groupby('lead_time')['search_count'].sum().reset_index()
    for lt in sorted(lt_g['lead_time'].unique(), reverse=True):
        g_cum += lt_g.loc[lt_g['lead_time'] == lt, 'search_count'].values[0]
        g_rates[lt] = g_cum / tot_hist
        
    df_eval['pred'] = df_eval.apply(lambda r: max(r['observed'] / max(0.01, g_rates.get(min(r['days_to_checkin'], 59), 1.0)), r['observed']), axis=1)
    records.append(df_eval)

all_evals = pd.concat(records, ignore_index=True)

print("--- تفکیک خطای WAPE بر حسب افق زمانی (Lead-time Buckets) ---")
for h, grp in all_evals.groupby('horizon_bucket'):
    print(f"{h}: WAPE = {calculate_wape(grp['actual'], grp['pred']):.2%}")

print("\n--- تفکیک خطای WAPE بر حسب فصل و تاریخ Cutoff ---")
for c, grp in all_evals.groupby('cutoff'):
    print(f"Cutoff {c}: WAPE = {calculate_wape(grp['actual'], grp['pred']):.2%}")