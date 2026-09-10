import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pathlib import Path

# ۱. تعریف متریک رسمی مسابقه
def calculate_wape(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    denom = np.sum(actual)
    return np.sum(np.abs(predicted - actual)) / denom if denom > 0 else 0.0


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "search_data_prepared.csv"

print("File exists:", DATA_PATH.exists())
print("Loading from:", DATA_PATH)
print(f"در حال بارگذاری داده‌های آموزشی: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

target_col = 'final_demand' if 'final_demand' in df.columns else 'search_count'
observed_col = 'observed_demand' if 'observed_demand' in df.columns else None

# شناسایی و تبدیل ستون‌های تاریخ
date_cols = [c for c in df.columns if any(k in c.lower() for k in ['checkin', 'date'])]
for d in date_cols:
    df[d] = pd.to_datetime(df[d])

time_col = 'checkin' if 'checkin' in df.columns else date_cols[0]

# ۳. استخراج ماه‌ها برای تقسیم زمانی (8 ماه Train و 2 ماه Test)
df['year_month'] = df[time_col].dt.to_period('M')
available_months = sorted(df['year_month'].unique())
print(f"تعداد کل ماه‌های شناسایی‌شده در دیتاست: {len(available_months)}")

if len(available_months) < 10:
    raise ValueError("برای استراتژی ۸ ماه آموزش و ۲ ماه تست، حداقل به ۱۰ ماه داده نیاز است.")

train_months = available_months[:8]
test_months = available_months[8:10]

train_df = df[df['year_month'].isin(train_months)].copy()
test_df = df[df['year_month'].isin(test_months)].copy()

print(f"بازه Train (8 ماه): {train_months[0]} تا {train_months[-1]} | ردیف‌ها: {len(train_df)}")
print(f"بازه Test (2 ماه): {test_months[0]} تا {test_months[-1]} | ردیف‌ها: {len(test_df)}")

# ۴. استخراج خودکار فیچرها بر اساس دیتای ارسالی امیر و ایلیا
exclude_cols = [target_col, 'year_month'] + date_cols
feature_cols = [c for c in df.columns if c not in exclude_cols]

cat_cols = []
for c in feature_cols:
    if df[c].dtype == 'object' or any(k in c.lower() for k in ['code', 'id', 'cluster', 'province']):
        train_df[c] = train_df[c].astype('category')
        test_df[c] = test_df[c].astype('category')
        cat_cols.append(c)

print(f"\nتعداد کل ویژگی‌ها: {len(feature_cols)}")
print(f"ویژگی‌های دسته‌ای شناسایی‌شده: {cat_cols}")

X_train, y_train = train_df[feature_cols], train_df[target_col]
X_test, y_test = test_df[feature_cols], test_df[target_col]

# ۵. آموزش مدل با تابع زیان L1 (کمینه‌ساز مستقیم صورت کسر WAPE)
print("\nدر حال آموزش LightGBM...")
model = LGBMRegressor(
    objective='regression_l1',
    n_estimators=600,
    learning_rate=0.03,
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

# ۶. ارزیابی روی دو ماه تست
preds = model.predict(X_test)
if observed_col and observed_col in test_df.columns:
    preds = np.maximum(preds, test_df[observed_col].fillna(0).values)

preds = np.clip(preds, 0, None)
test_wape = calculate_wape(y_test, preds)
print(f"\n>>> نمره ارزیابی روی ۲ ماه تست (WAPE): {test_wape:.4%}")

# ۷. خروجی گرفتن فایل pkl (شامل مدل و متادیتای ساختار ستون‌ها)
model_bundle = {
    'model': model,
    'feature_cols': feature_cols,
    'cat_cols': cat_cols,
    'observed_col': observed_col,
    'target_col': target_col,
    'time_col': time_col
}

OUTPUT_PKL = 'lgb_model_bundle.pkl'
joblib.dump(model_bundle, OUTPUT_PKL)
print(f"\nمدل و متادیتا با موفقیت در فایل '{OUTPUT_PKL}' ذخیره شد.")