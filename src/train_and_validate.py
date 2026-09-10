import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# ۱. فرمول رسمی WAPE
def calculate_wape(actual, predicted):
    actual, predicted = np.array(actual, dtype=float), np.array(predicted, dtype=float)
    denom = np.sum(actual)
    return np.sum(np.abs(predicted - actual)) / denom if denom > 0 else 0.0

print("در حال بارگذاری داده‌ها و برچسب خوشه‌ها...")

DATA_PATH = 'data/processed/search_data_prepared.csv'
CLUSTER_PATH = 'data/processed/final_city_clusters_k4.csv'

df = pd.read_csv(DATA_PATH)
clusters_df = pd.read_csv(CLUSTER_PATH)

print(f"داده‌ها با موفقیت بارگذاری شدند. تعداد کل ستون‌ها: {df.shape[1]}")

# ۲. الحاق ستون cluster_k4 به دیتاست اصلی بر اساس کد شهر
if 'city_code' in df.columns and 'city_code' in clusters_df.columns:
    df = pd.merge(df, clusters_df, on='city_code', how='left')
    df['cluster_k4'] = df['cluster_k4'].fillna(-1).astype(int)
    print("برچسب‌های کلاستر با موفقیت به دیتاست اصلی اضافه شدند.")
else:
    print("خطا در تطبیق ستون city_code برای الحاق کلاسترها!")

# ۳. شناسایی هوشمند ستون تارگت (هدف)
possible_targets = ['total_demand', 'final_demand', 'demand', 'target', 'search_count', 'y']
target_col = next((col for col in possible_targets if col in df.columns), None)

if target_col is None:
    for col in df.columns:
        if 'demand' in col.lower() or 'target' in col.lower() or 'count' in col.lower():
            target_col = col
            break

if target_col is None:
    raise ValueError("❌ هیچ ستون مناسبی برای تارگت (هدف پیش‌بینی) پیدا نشد!")

print(f"🎯 ستون هدف شناسایی‌شده برای آموزش: '{target_col}'")

observed_col = 'observed_demand' if 'observed_demand' in df.columns else None

# تبدیل تاریخ‌ها
date_cols = [c for c in df.columns if any(k in c.lower() for k in ['date', 'checkin'])]
for d in date_cols:
    df[d] = pd.to_datetime(df[d])

time_col = 'checkin' if 'checkin' in df.columns else (date_cols[0] if date_cols else None)

# ۴. تقسیم‌بندی زمانی (Temporal Split)
if time_col:
    df['year_month'] = df[time_col].dt.to_period('M')
    available_months = sorted(df['year_month'].unique())
    
    if len(available_months) >= 3:
        split_idx = int(len(available_months) * 0.8)
        train_df = df[df['year_month'].isin(available_months[:split_idx])].copy()
        test_df = df[df['year_month'].isin(available_months[split_idx:])].copy()
    else:
        split_idx = int(len(df) * 0.8)
        train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
else:
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

# ۵. تعیین فیچرها و تبدیل همه ستون‌های متنی/صنفی به دسته‌ای (Category)
exclude_cols = [target_col] + date_cols + (['year_month'] if 'year_month' in df.columns else [])
feature_cols = [c for c in df.columns if c not in exclude_cols]

cat_cols = []
for c in feature_cols:
    # هر ستونی که از نوع متن/رشته (object یا string) باشد یا نامش شامل کلمات کلیدی باشد را کاتگوریکال می‌کنیم
    if train_df[c].dtype == 'object' or pd.api.types.is_string_dtype(train_df[c]) or any(k in c.lower() for k in ['code', 'id', 'cluster', 'tier', 'province', 'region', 'name']):
        train_df[c] = train_df[c].astype('category')
        test_df[c] = test_df[c].astype('category')
        cat_cols.append(c)

# اطمینان از دسته‌ای بودن کلاستر
if 'cluster_k4' in feature_cols and 'cluster_k4' not in cat_cols:
    train_df['cluster_k4'] = train_df['cluster_k4'].astype('category')
    test_df['cluster_k4'] = test_df['cluster_k4'].astype('category')
    cat_cols.append('cluster_k4')

print(f"\nتعداد کل فیچرهای نهایی برای LightGBM: {len(feature_cols)}")
print(f"ستون‌های Categorical شناسایی‌شده: {cat_cols}")

X_train, y_train = train_df[feature_cols], train_df[target_col]
X_test, y_test = test_df[feature_cols], test_df[target_col]

# ۶. آموزش مدل LightGBM
print("\n" + "="*40)
print("در حال آموزش مدل LightGBM با بهره‌گیری از کلاسترهای رفتاری...")
print("="*40)

model = LGBMRegressor(
    objective='regression_l1',
    n_estimators=800,
    learning_rate=0.02,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    categorical_feature=cat_cols
)

lgb_preds = model.predict(X_test)

if observed_col and observed_col in test_df.columns:
    lgb_preds = np.maximum(lgb_preds, test_df[observed_col].fillna(0).values)

lgb_preds = np.clip(lgb_preds, 0, None)
wape_final = calculate_wape(y_test, lgb_preds)

print(f"\n✨ WAPE نهایی مدل LightGBM (با احتساب خوشه‌ها): {wape_final:.2%}")
print("="*40)

# ۷. ذخیره مدل نهایی
joblib.dump({
    'model': model,
    'feature_cols': feature_cols,
    'cat_cols': cat_cols,
    'wape': wape_final
}, 'data/processed/lgb_model_clustered_bundle.pkl')

print("مدل آموزش‌دیده در مسیر 'data/processed/lgb_model_clustered_bundle.pkl' ذخیره شد.")