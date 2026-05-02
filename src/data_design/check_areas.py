# src/data_design/check_areas.py

from src.data_design.amman_areas import AMMAN_AREAS, JORDAN_WEEKEND, RUSH_HOURS

print("Total areas:", len(AMMAN_AREAS))
print("Jordan weekend:", JORDAN_WEEKEND)
print("Rush hours:", RUSH_HOURS)

total_weight = sum(area["weight"] for area in AMMAN_AREAS.values())
print("Total weight:", round(total_weight, 2))

print("\nAreas:")
for name, info in AMMAN_AREAS.items():
    print(f"- {name}: {info['type']} | weight={info['weight']}")