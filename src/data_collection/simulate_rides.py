import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data_design.amman_areas import AMMAN_AREAS, JORDAN_WEEKEND, RUSH_HOURS


# -----------------------------
# 1. Reproducibility
# -----------------------------
random.seed(42)
np.random.seed(42)


# -----------------------------
# 2. Project paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "rides_amman.csv")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# -----------------------------
# 3. Constants
# -----------------------------
NUM_RIDES = 30000
START_DATE = datetime(2026, 3, 1)
END_DATE = datetime(2026, 3, 31, 23, 59, 59)

BASE_FARE = 0.75
DISTANCE_RATE = 0.35
TIME_RATE = 0.08
BOOKING_FEE = 0.25
MIN_FARE = 1.50

AREA_NAMES = list(AMMAN_AREAS.keys())
AREA_WEIGHTS = [info["weight"] for info in AMMAN_AREAS.values()]


# -----------------------------
# 4. Helper functions
# -----------------------------
def random_datetime(start, end):
    """Generate a random datetime between start and end."""
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def is_weekend(day_name):
    """Check if day is weekend in Jordan."""
    return day_name in JORDAN_WEEKEND


def is_rush_hour(hour, day_name):
    """Rush hour is only on non-weekend days."""
    if is_weekend(day_name):
        return False

    morning_start, morning_end = RUSH_HOURS["morning"]
    evening_start, evening_end = RUSH_HOURS["evening"]

    return (morning_start <= hour <= morning_end) or (evening_start <= hour <= evening_end)


def choose_pickup_area():
    """Choose pickup area using demand weights."""
    return random.choices(AREA_NAMES, weights=AREA_WEIGHTS, k=1)[0]


def generate_coordinates(area_name):
    """Generate small random coordinate variation around area center."""
    area = AMMAN_AREAS[area_name]
    lat = area["lat"] + np.random.uniform(-0.005, 0.005)
    lon = area["lon"] + np.random.uniform(-0.005, 0.005)
    return round(lat, 6), round(lon, 6)


def choose_dropoff_area(pickup_area):
    """Choose a realistic dropoff area."""
    possible_areas = [area for area in AREA_NAMES if area != pickup_area]
    return random.choice(possible_areas)


def estimate_distance_km(pickup_area, dropoff_area):
    """Estimate trip distance using area centers."""
    p = AMMAN_AREAS[pickup_area]
    d = AMMAN_AREAS[dropoff_area]

    lat_diff = p["lat"] - d["lat"]
    lon_diff = p["lon"] - d["lon"]

    approx_distance = ((lat_diff * 111) ** 2 + (lon_diff * 85) ** 2) ** 0.5

    noise = np.random.uniform(0.8, 1.2)
    distance = max(1.0, approx_distance * noise)

    return round(distance, 2)


def get_area_demand_multiplier(area_name, hour, day_name):
    """Make demand more realistic based on area type and timing."""
    area_type = AMMAN_AREAS[area_name]["type"]

    multiplier = 1.0

    if area_type == "business":
        if not is_weekend(day_name) and 7 <= hour <= 9:
            multiplier = 1.35
        elif not is_weekend(day_name) and 16 <= hour <= 19:
            multiplier = 1.15
        elif is_weekend(day_name):
            multiplier = 0.75

    elif area_type == "commercial":
        if not is_weekend(day_name) and 17 <= hour <= 21:
            multiplier = 1.25
        elif is_weekend(day_name):
            multiplier = 1.10

    elif area_type == "shopping":
        if 17 <= hour <= 22:
            multiplier = 1.30
        if is_weekend(day_name):
            multiplier = 1.40

    elif area_type == "university":
        if not is_weekend(day_name) and 7 <= hour <= 15:
            multiplier = 1.30
        elif is_weekend(day_name):
            multiplier = 0.60

    elif area_type == "airport":
        multiplier = 0.90

    elif area_type == "industrial":
        if not is_weekend(day_name) and 6 <= hour <= 9:
            multiplier = 1.20
        elif is_weekend(day_name):
            multiplier = 0.70

    return multiplier


def estimate_duration_minutes(distance_km, hour, day_name):
    """Estimate duration using traffic conditions."""
    base_speed = np.random.uniform(28, 42)

    if is_rush_hour(hour, day_name):
        base_speed *= 0.70

    duration = (distance_km / base_speed) * 60
    duration += np.random.uniform(-3, 5)

    return round(max(5, duration), 1)


def calculate_surge_multiplier(hour, day_name, pickup_area):
    """Jordan-aware surge logic."""
    area_type = AMMAN_AREAS[pickup_area]["type"]

    surge = 1.0

    if is_rush_hour(hour, day_name):
        surge += np.random.uniform(0.20, 0.80)

    if area_type in ["shopping", "commercial"] and 18 <= hour <= 22:
        surge += np.random.uniform(0.10, 0.40)

    if pickup_area == "Queen Alia Airport":
        surge += np.random.uniform(0.05, 0.25)

    return round(min(surge, 2.5), 2)


def calculate_fare(distance_km, duration_min, surge):
    """Fare calculation."""
    fare = (
        BASE_FARE
        + (distance_km * DISTANCE_RATE)
        + (duration_min * TIME_RATE)
        + BOOKING_FEE
    ) * surge

    return round(max(fare, MIN_FARE), 2)


def get_hour_12(dt):
    """12-hour display format."""
    return dt.strftime("%I %p").lstrip("0")


# -----------------------------
# 5. Simulation
# -----------------------------
records = []

for ride_id in range(1, NUM_RIDES + 1):
    pickup_dt = random_datetime(START_DATE, END_DATE)
    day_name = pickup_dt.strftime("%A")
    hour = pickup_dt.hour

    pickup_area = choose_pickup_area()

    # Re-pick sometimes to strengthen area-time realism
    multiplier = get_area_demand_multiplier(pickup_area, hour, day_name)
    if random.random() < min(0.5, max(0.0, multiplier - 1.0)):
        pickup_area = choose_pickup_area()

    dropoff_area = choose_dropoff_area(pickup_area)

    pickup_lat, pickup_lon = generate_coordinates(pickup_area)
    dropoff_lat, dropoff_lon = generate_coordinates(dropoff_area)

    distance_km = estimate_distance_km(pickup_area, dropoff_area)
    duration_min = estimate_duration_minutes(distance_km, hour, day_name)

    dropoff_dt = pickup_dt + timedelta(minutes=duration_min)

    weekend_flag = is_weekend(day_name)
    rush_flag = is_rush_hour(hour, day_name)
    surge = calculate_surge_multiplier(hour, day_name, pickup_area)
    fare = calculate_fare(distance_km, duration_min, surge)

    records.append(
        [
            ride_id,
            pickup_dt,
            dropoff_dt,
            pickup_area,
            dropoff_area,
            pickup_lat,
            pickup_lon,
            dropoff_lat,
            dropoff_lon,
            distance_km,
            duration_min,
            surge,
            fare,
            day_name,
            hour,
            get_hour_12(pickup_dt),
            weekend_flag,
            rush_flag,
        ]
    )


# -----------------------------
# 6. Save dataset
# -----------------------------
columns = [
    "ride_id",
    "pickup_datetime",
    "dropoff_datetime",
    "pickup_area",
    "dropoff_area",
    "pickup_lat",
    "pickup_lon",
    "dropoff_lat",
    "dropoff_lon",
    "trip_distance_km",
    "trip_duration_min",
    "surge_multiplier",
    "fare_amount_jod",
    "day_of_week",
    "hour_of_day",
    "hour_12",
    "is_weekend",
    "is_rush_hour",
]

df = pd.DataFrame(records, columns=columns)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df):,} rides to:")
print(OUTPUT_PATH)
print("\nSample:")
print(df.head())