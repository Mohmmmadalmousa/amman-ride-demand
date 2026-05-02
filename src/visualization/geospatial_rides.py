import pandas as pd
from pathlib import Path
import folium
from folium.plugins import HeatMap

# -----------------------------
# 1. Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "rides_amman_clean.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 2. Load data
# -----------------------------
df = pd.read_csv(INPUT_PATH)

print(f"Loaded {len(df):,} rides")
print("Columns:")
print(df.columns.tolist())

# -----------------------------
# 3. Area statistics
# -----------------------------
area_stats = (
    df.groupby("pickup_area")
    .agg(
        total_trips=("ride_id", "count"),
        avg_fare=("fare_amount_jod", "mean"),
        total_revenue=("fare_amount_jod", "sum"),
        avg_surge=("surge_multiplier", "mean"),
        avg_lat=("pickup_lat", "mean"),
        avg_lon=("pickup_lon", "mean"),
    )
    .round(2)
    .reset_index()
)

print("\nArea stats sample:")
print(area_stats.head())

# -----------------------------
# 4. Map center
# -----------------------------
AMMAN_CENTER = [31.9539, 35.9106]

# -----------------------------
# 5. Map 1 — Base Amman map
# -----------------------------
base_map = folium.Map(
    location=AMMAN_CENTER,
    zoom_start=11,
    tiles="OpenStreetMap"
)

base_map_path = OUTPUT_DIR / "01_amman_base_map.html"
base_map.save(base_map_path)
print(f"Saved: {base_map_path.name}")

# -----------------------------
# 6. Map 2 — Pickup heatmap
# -----------------------------
heat_map = folium.Map(
    location=AMMAN_CENTER,
    zoom_start=11,
    tiles="CartoDB positron"
)

heat_data = df[["pickup_lat", "pickup_lon"]].sample(
    min(10000, len(df)),
    random_state=42
).values.tolist()

HeatMap(
    heat_data,
    radius=14,
    blur=12
).add_to(heat_map)

heat_map_path = OUTPUT_DIR / "02_pickup_heatmap.html"
heat_map.save(heat_map_path)
print(f"Saved: {heat_map_path.name}")

# -----------------------------
# 7. Map 3 — Area demand map
# -----------------------------
demand_map = folium.Map(
    location=AMMAN_CENTER,
    zoom_start=11,
    tiles="CartoDB positron"
)

for _, row in area_stats.iterrows():
    popup_text = (
        f"Area: {row['pickup_area']}<br>"
        f"Trips: {int(row['total_trips']):,}<br>"
        f"Avg Fare: {row['avg_fare']:.2f} JOD<br>"
        f"Revenue: {row['total_revenue']:.2f} JOD<br>"
        f"Avg Surge: {row['avg_surge']:.2f}"
    )

    folium.CircleMarker(
        location=[row["avg_lat"], row["avg_lon"]],
        radius=max(6, row["total_trips"] / 250),
        popup=popup_text,
        tooltip=row["pickup_area"],
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.6,
    ).add_to(demand_map)

demand_map_path = OUTPUT_DIR / "03_area_demand_map.html"
demand_map.save(demand_map_path)
print(f"Saved: {demand_map_path.name}")

# -----------------------------
# 8. Map 4 — Revenue map
# -----------------------------
revenue_map = folium.Map(
    location=AMMAN_CENTER,
    zoom_start=11,
    tiles="CartoDB dark_matter"
)

max_revenue = area_stats["total_revenue"].max()

for _, row in area_stats.iterrows():
    revenue_ratio = row["total_revenue"] / max_revenue

    if revenue_ratio > 0.75:
        color = "green"
    elif revenue_ratio > 0.45:
        color = "orange"
    else:
        color = "red"

    popup_text = (
        f"Area: {row['pickup_area']}<br>"
        f"Revenue: {row['total_revenue']:.2f} JOD<br>"
        f"Trips: {int(row['total_trips']):,}<br>"
        f"Avg Fare: {row['avg_fare']:.2f} JOD"
    )

    folium.CircleMarker(
        location=[row["avg_lat"], row["avg_lon"]],
        radius=max(6, row["total_revenue"] / 3000),
        popup=popup_text,
        tooltip=row["pickup_area"],
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
    ).add_to(revenue_map)

revenue_map_path = OUTPUT_DIR / "04_area_revenue_map.html"
revenue_map.save(revenue_map_path)
print(f"Saved: {revenue_map_path.name}")

print("\nGeospatial analysis complete.")
print("Maps saved in:")
print(OUTPUT_DIR)
# -----------------------------
# 9. Save data for Power BI
# -----------------------------
output_csv = BASE_DIR / "data" / "processed" / "area_summary.csv"

area_stats.rename(columns={
    "pickup_area": "area",
    "total_trips": "total_rides",
    "avg_lat": "lat",
    "avg_lon": "lon"
}, inplace=True)

area_stats.to_csv(output_csv, index=False)

print(f"\nSaved Power BI dataset: {output_csv}")