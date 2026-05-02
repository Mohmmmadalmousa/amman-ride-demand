import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "rides_amman_clean.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_PATH, parse_dates=["pickup_datetime", "dropoff_datetime"])

print(f"Loaded {len(df):,} cleaned rides")

sns.set_style("whitegrid")

def save_plot(filename):
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")

# 1. Trips per hour
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="hour_of_day", hue="hour_of_day", legend=False)
plt.title("Number of Trips per Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Trips")
save_plot("01_trips_per_hour.png")

# 2. Trips per day
plt.figure(figsize=(10, 6))
order_days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
sns.countplot(data=df, x="day_of_week", order=order_days, hue="day_of_week", legend=False)
plt.title("Trips per Day of Week")
plt.xlabel("Day")
plt.ylabel("Trips")
plt.xticks(rotation=30)
save_plot("02_trips_per_day.png")

# 3. Trips per area
plt.figure(figsize=(12, 6))
area_order = df["pickup_area"].value_counts().index
sns.countplot(data=df, x="pickup_area", order=area_order, hue="pickup_area", legend=False)
plt.title("Trips per Pickup Area")
plt.xlabel("Pickup Area")
plt.ylabel("Trips")
plt.xticks(rotation=45, ha="right")
save_plot("03_trips_per_area.png")

# 4. Distance distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["trip_distance_km"], bins=30, kde=True)
plt.title("Trip Distance Distribution")
plt.xlabel("Distance (km)")
plt.ylabel("Count")
save_plot("04_trip_distance_dist.png")

# 5. Fare distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["fare_amount_jod"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare (JOD)")
plt.ylabel("Count")
save_plot("05_fare_dist.png")

# 6. Duration distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["trip_duration_min"], bins=30, kde=True)
plt.title("Trip Duration Distribution")
plt.xlabel("Duration (minutes)")
plt.ylabel("Count")
save_plot("06_trip_duration_dist.png")

# 7. Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
save_plot("07_correlation.png")

# 8. Fare vs Distance
sample_df = df.sample(min(3000, len(df)), random_state=42)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=sample_df,
    x="trip_distance_km",
    y="fare_amount_jod",
    hue="is_rush_hour",
    alpha=0.6
)
plt.title("Fare vs Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Fare (JOD)")
save_plot("08_fare_vs_distance.png")

# 9. Fare distribution: Rush vs Non-Rush
plt.figure(figsize=(10, 6))

sns.histplot(
    df[df["is_rush_hour"] == True]["fare_amount_jod"],
    bins=30,
    kde=True,
    label="Rush Hour",
    stat="density"
)

sns.histplot(
    df[df["is_rush_hour"] == False]["fare_amount_jod"],
    bins=30,
    kde=True,
    label="Non-Rush",
    stat="density"
)

plt.title("Fare Distribution: Rush vs Non-Rush")
plt.xlabel("Fare (JOD)")
plt.ylabel("Density")
plt.legend()
save_plot("09_rush_vs_nonrush_distribution.png")

# 10. Boxplot: Fare comparison
plt.figure(figsize=(8, 6))
sns.boxplot(x="is_rush_hour", y="fare_amount_jod", data=df)

plt.title("Fare Comparison (Rush vs Non-Rush)")
plt.xlabel("Rush Hour")
plt.ylabel("Fare (JOD)")
save_plot("10_fare_boxplot.png")

print("EDA complete.")