import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
# -----------------------------
# 1. Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "demand_time_summary.csv"

# -----------------------------
# 2. Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df)} rows")
print("\nColumns:")
print(df.columns.tolist())
print("\nSample:")
print(df.head())

# -----------------------------
# 3. Features and target
# -----------------------------
X = df[[
    "pickup_area",
    "day_of_week",
    "hour_of_day",
    "avg_surge",
    "avg_fare"
]]
y = df["total_trips"]

# -----------------------------
# 4. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Encode text columns
# -----------------------------
categorical_features = ["pickup_area", "day_of_week"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

# -----------------------------
# 6. Model
# -----------------------------
model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
# -----------------------------
# 7. Full pipeline
# -----------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# -----------------------------
# 8. Train
# -----------------------------
pipeline.fit(X_train, y_train)


# Save model
model_path = BASE_DIR / "models" / "xgboost_model.pkl"
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(pipeline, model_path)

print(f"Model saved to: {model_path}")
# -----------------------------
# 9. Predict
# -----------------------------
y_pred = pipeline.predict(X_test)

# -----------------------------
# 10. Evaluate
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
# -----------------------------
# 11. Example prediction
# -----------------------------
example = pd.DataFrame({
    "pickup_area": ["Abdali"],
    "day_of_week": ["Monday"],
    "hour_of_day": [8],
    "avg_surge": [1.20],
    "avg_fare": [7.50]
})
prediction = pipeline.predict(example)

print("\nExample Prediction:")
print(f"Predicted total_trips: {prediction[0]:.0f}")

# -----------------------------
# 12. Compare actual vs predicted
# -----------------------------
results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred.round(2)
})

#-----------------------------
#accurse
#-----------------------------
tolerance = 5
correct = (abs(y_test - y_pred) <= tolerance).sum()
accuracy = correct / len(y_test)

print(f"Within ±{tolerance} trips accuracy: {accuracy * 100:.2f}%")

print("\nActual vs Predicted sample:")
print(results.head(10))