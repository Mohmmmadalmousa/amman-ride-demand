import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
import os

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_path = os.path.join(base_dir, 'data', 'processed', 'rides_amman_clean.csv')

# Load data
df = pd.read_csv(input_path)

print(f"Loaded {len(df)} rows")

# =========================
# TEST 1 — Rush vs Non-Rush Fare (T-Test)
# =========================
rush = df[df['is_rush_hour'] == True]['fare_amount_jod']
non_rush = df[df['is_rush_hour'] == False]['fare_amount_jod']

t_stat, p_value = stats.ttest_ind(rush, non_rush, equal_var=False)

print("\n--- Test 1: Rush vs Non-Rush Fare ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < 0.05:
    print("Result: SIGNIFICANT difference ✅")
else:
    print("Result: NOT significant ❌")

print(f"Rush avg fare: {rush.mean():.2f} JOD")
print(f"Non-rush avg fare: {non_rush.mean():.2f} JOD")
print(f"Rush std fare: {rush.std():.2f} JOD")
print(f"Non-rush std fare: {non_rush.std():.2f} JOD")
# =========================
# TEST 2 — Area Fare Differences (ANOVA)
# =========================
zones = df['pickup_area'].unique()
groups = [df[df['pickup_area'] == z]['fare_amount_jod'] for z in zones]

f_stat, p_value2 = stats.f_oneway(*groups)

print("\n--- Test 2: Fare Differences Between Zones ---")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value2:.6f}")

if p_value2 < 0.05:
    print("Result: SIGNIFICANT difference between zones ✅")
else:
    print("Result: NO significant difference ❌")

# =========================
# TEST 3 — Weekend vs Weekday Surge (Mann-Whitney)
# =========================
weekend = df[df['is_weekend'] == True]['surge_multiplier']
weekday = df[df['is_weekend'] == False]['surge_multiplier']

u_stat, p_value3 = stats.mannwhitneyu(weekend, weekday, alternative='two-sided')

print("\n--- Test 3: Weekend vs Weekday Surge ---")
print(f"U-statistic: {u_stat:.4f}")
print(f"P-value: {p_value3:.6f}")

if p_value3 < 0.05:
    print("Result: SIGNIFICANT difference ✅")
else:
    print("Result: NO significant difference ❌")

print(f"Weekend avg surge: {weekend.mean():.2f}")
print(f"Weekday avg surge: {weekday.mean():.2f}")
print(f"Weekend std surge: {weekend.std():.2f}")
print(f"Weekday std surge: {weekday.std():.2f}")
# =========================
# TEST 4 — Correlation Analysis (REQUIRED)
# =========================
print("\n--- Test 4: Correlation Analysis ---")

corr1, p1 = pearsonr(df["trip_distance_km"], df["fare_amount_jod"])
print(f"Distance vs Fare → Corr: {corr1:.3f}, P-value: {p1:.6f}")

corr2, p2 = pearsonr(df["trip_duration_min"], df["fare_amount_jod"])
print(f"Duration vs Fare → Corr: {corr2:.3f}, P-value: {p2:.6f}")

corr3, p3 = pearsonr(df["surge_multiplier"], df["fare_per_km"])
print(f"Surge vs Fare/km → Corr: {corr3:.3f}, P-value: {p3:.6f}")