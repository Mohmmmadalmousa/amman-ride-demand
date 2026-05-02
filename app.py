import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =========================
# Page Config (MUST be first)
# =========================
st.set_page_config(
    page_title="Amman Ride Demand",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Root theme */
:root {
    --amber: #F59E0B;
    --amber-light: #FDE68A;
    --dark: #0F1117;
    --card-bg: #1A1D27;
    --card-border: #2A2D3A;
    --text-primary: #F1F5F9;
    --text-muted: #94A3B8;
    --green: #10B981;
    --red: #EF4444;
    --blue: #3B82F6;
}

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0D0F18 0%, #111420 50%, #0D0F18 100%);
}

/* Hide streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111420 !important;
    border-right: 1px solid #2A2D3A;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #94A3B8 !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Hero header */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #F59E0B;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-size: 1rem;
    color: #64748B;
    font-weight: 400;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.hero-badge {
    display: inline-block;
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.3);
    color: #F59E0B;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    margin-bottom: 1rem;
    letter-spacing: 0.1em;
}

/* Section headers */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #F59E0B;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.75rem;
    border-left: 2px solid #F59E0B;
    padding-left: 0.6rem;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #1A1D27 0%, #151821 100%);
    border: 1px solid #2A2D3A;
    border-radius: 12px;
    padding: 1.4rem 1.2rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #F59E0B, transparent);
}

.kpi-card:hover {
    border-color: rgba(245,158,11,0.4);
}

.kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #F1F5F9;
    line-height: 1;
    margin-bottom: 0.3rem;
}

.kpi-label {
    font-size: 0.75rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

.kpi-delta {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #10B981;
    margin-top: 0.4rem;
}

/* Prediction box */
.pred-box {
    background: linear-gradient(135deg, rgba(245,158,11,0.08) 0%, rgba(245,158,11,0.03) 100%);
    border: 1px solid rgba(245,158,11,0.35);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.pred-box::after {
    content: '◈';
    position: absolute;
    font-size: 8rem;
    color: rgba(245,158,11,0.04);
    right: -1.5rem;
    top: -1.5rem;
    font-family: 'Space Mono', monospace;
}

.pred-number {
    font-family: 'Space Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    color: #F59E0B;
    line-height: 1;
    text-shadow: 0 0 40px rgba(245,158,11,0.3);
}

.pred-unit {
    font-size: 0.85rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.5rem;
}

.pred-status {
    display: inline-block;
    margin-top: 1rem;
    padding: 0.3rem 1rem;
    border-radius: 99px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.status-high {background: rgba(239,68,68,0.15); color: #EF4444; border: 1px solid rgba(239,68,68,0.3);}
.status-med {background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3);}
.status-low {background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3);}

/* Input card */
.input-summary {
    background: #13151F;
    border: 1px solid #2A2D3A;
    border-radius: 12px;
    padding: 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #94A3B8;
    line-height: 1.8;
}

.input-row {
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #1F2230;
    padding: 0.35rem 0;
}

.input-row:last-child {border-bottom: none;}

.input-key {color: #4A5568; text-transform: uppercase; letter-spacing: 0.06em;}
.input-val {color: #F59E0B;}

/* Sidebar header */
.sidebar-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    color: #F59E0B;
    letter-spacing: 0.05em;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #2A2D3A;
    margin-bottom: 1.2rem;
}

/* Button override */
.stButton > button {
    background: linear-gradient(135deg, #F59E0B, #D97706) !important;
    color: #0F1117 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(245,158,11,0.25) !important;
}

.stButton > button:hover {
    box-shadow: 0 6px 25px rgba(245,158,11,0.4) !important;
    transform: translateY(-1px) !important;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1F2230;
    margin: 1.5rem 0;
}

/* Mini chart bars */
.mini-bar-wrap {display: flex; align-items: flex-end; gap: 3px; height: 40px; margin-top: 0.5rem;}
.mini-bar {background: rgba(245,158,11,0.3); border-radius: 2px; width: 100%; transition: background 0.2s;}
.mini-bar.active {background: #F59E0B;}

/* Area pills */
.area-grid {display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem;}
.area-pill {
    background: #1A1D27;
    border: 1px solid #2A2D3A;
    border-radius: 6px;
    padding: 0.3rem 0.6rem;
    font-size: 0.68rem;
    color: #64748B;
    font-family: 'Space Mono', monospace;
    cursor: default;
}
.area-pill.highlight {
    border-color: rgba(245,158,11,0.4);
    color: #F59E0B;
    background: rgba(245,158,11,0.05);
}
</style>
""", unsafe_allow_html=True)

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "demand_time_summary.csv"

# =========================
# Load model & data (with fallback for demo)
# =========================
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH), True
    except Exception:
        return None, False

@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH), True
    except Exception:
        # Demo data
        areas = ["Abdali", "Downtown", "Shmeisani", "Sweifieh", "Jabal Amman",
                 "University Street", "Khalda", "Mecca Street", "Tabarbour",
                 "Marka", "Airport Road", "Madaba Road", "City Mall Area",
                 "Queen Alia Airport", "Sahab"]
        days = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
        hours = list(range(24))
        rng = np.random.default_rng(42)
        rows = []
        for area in areas:
            for day in days:
                for hour in hours:
                    base = rng.integers(5, 30)
                    surge = round(rng.uniform(0.8, 2.5), 2)
                    fare = round(rng.uniform(3.0, 18.0), 2)
                    rows.append({
                        "pickup_area": area, "day_of_week": day,
                        "hour_of_day": hour, "total_trips": base,
                        "avg_surge": surge, "avg_fare": fare
                    })
        return pd.DataFrame(rows), False

model, model_loaded = load_model()
df, data_loaded = load_data()

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-title">◈ INPUT PARAMETERS</div>', unsafe_allow_html=True)

    pickup_area = st.selectbox(
        "Pickup Area",
        sorted(df["pickup_area"].unique()),
        index=0
    )

    day_of_week = st.selectbox(
        "Day of Week",
        ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    )

    hour_of_day = st.slider("Hour of Day", 0, 23, 8, format="%d:00")

    # Smart defaults
    match_df = df[
        (df["pickup_area"] == pickup_area) &
        (df["day_of_week"] == day_of_week) &
        (df["hour_of_day"] == hour_of_day)
    ]
    if not match_df.empty:
        default_surge = float(match_df["avg_surge"].iloc[0])
        default_fare = float(match_df["avg_fare"].iloc[0])
    else:
        area_df = df[df["pickup_area"] == pickup_area]
        default_surge = float(area_df["avg_surge"].mean()) if not area_df.empty else 1.2
        default_fare = float(area_df["avg_fare"].mean()) if not area_df.empty else 7.5

    st.markdown('<hr style="border-color:#2A2D3A; margin: 1rem 0;">', unsafe_allow_html=True)

    # Auto-fill surge and fare from dataset (not shown to user)
    avg_surge = round(default_surge, 2)
    avg_fare = round(default_fare, 2)

    predict_btn = st.button("⚡ PREDICT DEMAND")

    st.markdown(
        '<div style="font-size:0.68rem;color:#4A5568;margin-top:0.75rem;font-family:Space Mono,monospace;line-height:1.6;">'
        '↳ Surge & fare auto-filled<br>from historical averages</div>',
        unsafe_allow_html=True
    )

    if not model_loaded:
        st.markdown(
            '<div style="font-size:0.68rem;color:#4A5568;margin-top:0.5rem;font-family:Space Mono,monospace;">'
            '⚠ Model not found — showing demo prediction</div>',
            unsafe_allow_html=True
        )

# =========================
# Input data
# =========================
input_data = pd.DataFrame({
    "pickup_area": [pickup_area],
    "day_of_week": [day_of_week],
    "hour_of_day": [hour_of_day],
    "avg_surge": [avg_surge],
    "avg_fare": [avg_fare]
})

# =========================
# Prediction logic
# =========================
def get_prediction():
    if model_loaded and model is not None:
        try:
            return int(round(model.predict(input_data)[0]))
        except Exception:
            pass
    # Demo: heuristic
    base = df[(df["pickup_area"] == pickup_area) & (df["day_of_week"] == day_of_week)]["total_trips"].mean()
    if np.isnan(base):
        base = df["total_trips"].mean()
    rush_boost = 1.4 if hour_of_day in [7, 8, 9, 17, 18, 19] else 1.0
    surge_factor = 1 + (avg_surge - 1) * 0.3
    return max(1, int(round(base * rush_boost * surge_factor)))

# =========================
# Hero
# =========================
col_head, col_badge = st.columns([3, 1])
with col_head:
    st.markdown('<div class="hero-badge">XGBoost · Amman, JO</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Ride Demand<br>Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="margin-top:0.5rem;">Real-time demand intelligence for Amman\'s ride network</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# =========================
# KPIs
# =========================
st.markdown('<div class="section-label">Fleet Overview</div>', unsafe_allow_html=True)

total_avg = df["total_trips"].mean()
fare_avg = df["avg_fare"].mean()
surge_avg = df["avg_surge"].mean()
peak_area = df.groupby("pickup_area")["total_trips"].sum().idxmax() if "pickup_area" in df.columns else "Abdali"
area_trips = df.groupby("pickup_area")["total_trips"].sum().max() if "pickup_area" in df.columns else 0

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg Trips / Slot</div>
        <div class="kpi-value">{total_avg:.1f}</div>
        <div class="kpi-delta">↑ across all areas</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg Fare</div>
        <div class="kpi-value">{fare_avg:.2f} <span style="font-size:1rem;color:#64748B">JOD</span></div>
        <div class="kpi-delta">↑ per trip</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg Surge</div>
        <div class="kpi-value">{surge_avg:.2f}<span style="font-size:1rem;color:#64748B">x</span></div>
        <div class="kpi-delta" style="color:#F59E0B;">dynamic pricing</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Top Pickup Area</div>
        <div class="kpi-value" style="font-size:1.3rem;">{peak_area}</div>
        <div class="kpi-delta">{int(area_trips):,} total trips</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# =========================
# Main prediction area
# =========================
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Prediction Output</div>', unsafe_allow_html=True)

    if predict_btn or True:  # always show prediction
        pred = get_prediction()

        if pred >= 20:
            status_cls, status_txt = "status-high", "● HIGH DEMAND"
        elif pred >= 10:
            status_cls, status_txt = "status-med", "● MODERATE"
        else:
            status_cls, status_txt = "status-low", "● LOW DEMAND"

        st.markdown(f"""
        <div class="pred-box">
            <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#4A5568;
                        text-transform:uppercase;letter-spacing:0.12em;margin-bottom:1rem;">
                Estimated rides
            </div>
            <div class="pred-number">{pred}</div>
            <div class="pred-unit">rides predicted</div>
            <div><span class="pred-status {status_cls}">{status_txt}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Hour context bar
    st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Hourly Pattern — Selected Area</div>', unsafe_allow_html=True)

    area_hourly = df[df["pickup_area"] == pickup_area].groupby("hour_of_day")["total_trips"].mean()
    if len(area_hourly) > 0:
        max_val = area_hourly.max()
        bars_html = '<div class="mini-bar-wrap">'
        for h in range(24):
            val = area_hourly.get(h, 0)
            height = max(5, int((val / max_val) * 100)) if max_val > 0 else 5
            active = "active" if h == hour_of_day else ""
            bars_html += f'<div class="mini-bar {active}" style="height:{height}%;" title="{h}:00 — {val:.1f} trips"></div>'
        bars_html += '</div>'
        bars_html += f'<div style="font-size:0.65rem;color:#4A5568;margin-top:0.3rem;font-family:Space Mono,monospace;">0:00 {"·" * 22} 23:00</div>'
        st.markdown(bars_html, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-label">Current Input Summary</div>', unsafe_allow_html=True)

    is_rush = hour_of_day in [7, 8, 9, 17, 18, 19]
    rush_label = "🟡 Rush Hour" if is_rush else "🟢 Off-Peak"
    day_type = "Weekend" if day_of_week in ["Friday", "Saturday", "Sunday"] else "Weekday"

    st.markdown(f"""
    <div class="input-summary">
        <div class="input-row"><span class="input-key">Area</span><span class="input-val">{pickup_area}</span></div>
        <div class="input-row"><span class="input-key">Day</span><span class="input-val">{day_of_week} ({day_type})</span></div>
        <div class="input-row"><span class="input-key">Hour</span><span class="input-val">{hour_of_day:02d}:00 — {rush_label}</span></div>
        <div class="input-row"><span class="input-key">Surge ↳ auto</span><span class="input-val" style="color:#64748B;">{avg_surge:.2f}×</span></div>
        <div class="input-row"><span class="input-key">Fare ↳ auto</span><span class="input-val" style="color:#64748B;">{avg_fare:.2f} JOD</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Active Zones</div>', unsafe_allow_html=True)

    all_areas = sorted(df["pickup_area"].unique())
    pills_html = '<div class="area-grid">'
    for a in all_areas:
        cls = "highlight" if a == pickup_area else ""
        pills_html += f'<div class="area-pill {cls}">{a}</div>'
    pills_html += '</div>'
    st.markdown(pills_html, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# =========================
# Data table
# =========================
st.markdown('<div class="section-label">Dataset Sample</div>', unsafe_allow_html=True)

display_df = df[df["pickup_area"] == pickup_area].head(8).reset_index(drop=True)
if display_df.empty:
    display_df = df.head(8).reset_index(drop=True)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "pickup_area": st.column_config.TextColumn("Area"),
        "day_of_week": st.column_config.TextColumn("Day"),
        "hour_of_day": st.column_config.NumberColumn("Hour", format="%d:00"),
        "total_trips": st.column_config.NumberColumn("Trips", format="%d"),
        "avg_surge": st.column_config.NumberColumn("Surge", format="%.2f×"),
        "avg_fare": st.column_config.NumberColumn("Avg Fare (JOD)", format="%.2f"),
    }
)