import pandas as pd
from pathlib import Path

# Build project paths safely
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "raw" / "rides_amman.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "rides_amman_clean.csv"

# Make sure output folder exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load raw data
df = pd.read_csv(INPUT_PATH, parse_dates=["pickup_datetime", "dropoff_datetime"])

print(f"Loaded {len(df):,} rides")
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isna().sum())

# Remove impossible values
df = df[(df["trip_distance_km"] > 0) & (df["trip_duration_min"] > 0)]

# Remove extreme outliers
df = df[df["trip_distance_km"] <= 50]
df = df[df["trip_duration_min"] <= 120]

# Remove rows with missing coordinates
df = df.dropna(subset=["pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon"])

# Remove duplicates
df = df.drop_duplicates()

print(f"\nAfter cleaning: {len(df):,} rides")

# Time features
df["day_of_week"] = df["pickup_datetime"].dt.day_name()
df["hour_of_day"] = df["pickup_datetime"].dt.hour

# Jordan weekend: Friday and Saturday
df["is_weekend"] = df["day_of_week"].isin(["Friday", "Saturday"])

# Rush hour logic
df["is_rush_hour"] = df["hour_of_day"].apply(lambda x: 7 <= x <= 9 or 16 <= x <= 20)

# Extra useful features
df["trip_speed_kmh"] = df["trip_distance_km"] / (df["trip_duration_min"] / 60)
df["fare_per_km"] = df["fare_amount_jod"] / df["trip_distance_km"]

# Save cleaned data
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved clean data to:\n{OUTPUT_PATH}")
print("\nSample:")
print(df.head())
print("\nFinal shape:", df.shape)


# -----------------------------
# Create Power BI time dataset
# -----------------------------
time_summary = (
    df.groupby(["pickup_area", "day_of_week", "hour_of_day"])
    .agg(
        total_trips=("ride_id", "count"),
        total_revenue=("fare_amount_jod", "sum"),
        avg_fare=("fare_amount_jod", "mean"),
        avg_surge=("surge_multiplier", "mean"),
    )
    .round(2)
    .reset_index()
)

output_time_path = BASE_DIR / "data" / "processed" / "demand_time_summary.csv"
time_summary.to_csv(output_time_path, index=False)

print(f"\nSaved time dataset: {output_time_path}")