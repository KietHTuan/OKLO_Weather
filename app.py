# =========================
# Header + Section 1 (Location & Date)
# =========================
import streamlit as st
import requests
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Optional
from zoneinfo import ZoneInfo
from streamlit_searchbox import st_searchbox
from streamlit_folium import st_folium
import folium
import os


# --- Page Setup ---
st.set_page_config(page_title="OKLO - Weather Prediciton", page_icon="⛅", layout="centered")

# --- Global Layout Styling ---
st.markdown("""
<style>
  /* Reduce top padding but add a bit of breathing room for header */
  .block-container {
      max-width: 950px;
      margin: 0 auto;
      padding-top: 1.5rem !important;
  }

  /* Prevent clipping of custom header */
  header[data-testid="stHeader"] {
      height: 0px;
      visibility: hidden;
  }

  /* Center alignment for header text + icon */
  .nimbus-header {
      display: flex;
      flex-direction: row;
      align-items: center;
      justify-content: center;
      gap: 12px;
      margin-top: 10px;
      margin-bottom: 15px;
  }
  .nimbus-icon {
      font-size: 46px;
      line-height: 1;
  }
  .nimbus-title {
      font-size: 30px;
      font-weight: 800;
      margin-bottom: 4px;
      line-height: 1.1;
  }
  .nimbus-subtitle {
      font-size: 13px;
      opacity: 0.85;
      text-align: center;
  }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="nimbus-header">
  <div class="nimbus-icon">🌤️</div>
  <div>
    <div class="nimbus-title">OKLO — Weather Prediction</div>
    <div class="nimbus-subtitle">
      Pick a place and a date to get weather information.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ---- Session defaults ----
st.session_state.setdefault("selected_place", None)         # {"label": str, "lat": float, "lon": float}
st.session_state.setdefault("chosen_date", datetime.now().date())
st.session_state.setdefault("selection_locked", False)      # gate for proceeding to Section 2

# ---- Config ----
USER_AGENT = "nimbus-weather-app"
REQUEST_TIMEOUT = 12

# ---- Geocoding (Nominatim) ----
@st.cache_data(show_spinner=False, ttl=24*3600)
def nominatim_suggest(query: str, limit: int = 6) -> List[Dict]:
    """Return a list of dicts: [{'label': str, 'lat': float, 'lon': float}, ...]."""
    if not query or len(query.strip()) < 3:
        return []
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query.strip(), "format": "jsonv2", "addressdetails": 1, "limit": limit}
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return [{"label": d["display_name"], "lat": float(d["lat"]), "lon": float(d["lon"])} for d in data]

def _searchbox_adapter(query: str):
    """Adapter for streamlit-searchbox: returns list of labels while caching the full records."""
    if not query or len(query.strip()) < 3:
        return []
    try:
        results = nominatim_suggest(query.strip())
        st.session_state["search_results"] = {r["label"]: r for r in results}
        return [r["label"] for r in results]
    except Exception:
        return []

# ---- Timezone resolver (tzfpy if available) ----
def resolve_timezone(lat: float, lon: float) -> Optional[str]:
    try:
        from tzfpy import get_tz   # tzfpy expects (lon, lat)
        return get_tz(lon, lat)
    except Exception:
        return None

# ---- Compute & store context for Section 2 ----
def compute_and_store_context(place: Dict, day: date) -> bool:
    """Given a place (label/lat/lon) and a local date, compute windows and store in session_state."""
    if not place:
        st.error("Please select a location first.")
        return False

    lat, lon, label = float(place["lat"]), float(place["lon"]), place["label"]
    tz_name = resolve_timezone(lat, lon)
    local_zone = ZoneInfo(tz_name) if tz_name else timezone.utc

    # Local-day window & UTC equivalents
    start_local = datetime(day.year, day.month, day.day, 0, 0, tzinfo=local_zone)
    end_local   = start_local + timedelta(days=1)
    start_utc   = start_local.astimezone(timezone.utc)
    end_utc     = end_local.astimezone(timezone.utc)

    # Mode + sampling preference
    now_local = datetime.now(tz=local_zone)
    if day < now_local.date():
        mode_label = "History mode"
        prefer_fine = False
    elif day == now_local.date():
        mode_label = "Live mode"
        prefer_fine = True      # try PT15M where possible
    else:
        mode_label = "Forecast mode"
        prefer_fine = False

    st.session_state.update({
        "selected_place": place,
        "chosen_date": day,
        "selection_locked": True,

        "lat": lat, "lon": lon, "label": label,
        "tz_name": tz_name, "local_zone": local_zone,
        "start_local": start_local, "end_local": end_local,
        "start_utc": start_utc, "end_utc": end_utc,
        "target_local_mid": datetime(day.year, day.month, day.day, 12, 0, tzinfo=local_zone),
        "prefer_fine": prefer_fine,
        "mode_label": mode_label,
    })
    return True

# =========================
# SECTION 1 — Choose Location & Date
# =========================
st.subheader("Choose Location & Event Date")

# 1) Location input (type-ahead)
picked_label = st_searchbox(
    search_function=_searchbox_adapter,
    placeholder="Type a city, address, or landmark (e.g., Kamloops, BC)",
    key="loc_search",
)
if picked_label and "search_results" in st.session_state:
    st.session_state.selected_place = st.session_state["search_results"].get(picked_label)

# 2) Date input
today = datetime.now().date()
chosen_date = st.date_input(
    "Select Date",
    value=st.session_state.get("chosen_date", today),
    min_value=today - timedelta(days=365 * 2),
    max_value=today + timedelta(days=365 * 2),
    key="date_picker",
)
st.session_state.chosen_date = chosen_date

# 3) Single confirm button
if st.button("Use this location & date", type="primary"):
    if compute_and_store_context(st.session_state.get("selected_place"), st.session_state["chosen_date"]):
        st.rerun()

# Gate: if not confirmed yet, guide and stop so Section 2 doesn’t crash
if not st.session_state.get("selection_locked", False):
    st.info("Pick a location and date above, then click **Use this location & date** to continue.")
    st.stop()

# Small summary once confirmed
place = st.session_state["selected_place"]
st.success(f"Selected: {place['label']} • {st.session_state['mode_label']}")
st.caption(
    f"Local window: {st.session_state['start_local'].strftime('%Y-%m-%d %H:%M %Z')} → "
    f"{st.session_state['end_local'].strftime('%Y-%m-%d %H:%M %Z')}"
)
st.divider()

# --- Location map (Folium) ---
with st.expander("📍 Location preview", expanded=True):
    if "lat" in st.session_state and "lon" in st.session_state:
        lat = st.session_state["lat"]
        lon = st.session_state["lon"]
        label = st.session_state.get("label", "Selected location")
        tz_name = st.session_state.get("tz_name") or "UTC"

        m = folium.Map(
            location=[lat, lon],
            zoom_start=11,
            tiles="CartoDB dark_matter",  # nice dark theme
            control_scale=True,
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color="#4F8DF5",
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(f"<b>{label}</b><br/>Timezone: {tz_name}", max_width=250),
            tooltip=label,
        ).add_to(m)

        # Optional: add a small mini-map in the corner
        try:
            from folium.plugins import MiniMap
            MiniMap(toggle_display=True, position="bottomright").add_to(m)
        except Exception:
            pass

        st_folium(m, height=360, width=None)
    else:
        st.info("Pick a location first to see the map.")





# ML MODEL and Recommendations
import pickle
import numpy as np
import streamlit as st

st.subheader("Weather Forecas for your Event!")

# --- Load the trained model ---
@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model("OKLO_Weather/XGB_Classifier.pkl")

# --- Get month from user or use date ---
default_month = st.session_state["chosen_date"].month
month_input = st.number_input("Enter month (1–12)", min_value=1, max_value=12, value=default_month, step=1)

# --- Prepare lat/lon from session ---
lat = st.session_state["lat"]
lon = st.session_state["lon"]

# --- Generate future months (current +2, +4, +6) ---
future_months = [
    month_input,
    (month_input + 2 - 1) % 12 + 1,
    (month_input + 4 - 1) % 12 + 1
    
]
month_labels = ["Current Month", "Next 2 Months", "Next 4 Months"]

# --- Helper: cyclic encode months ---
def encode_month(m):
    month_sin = np.sin(2 * np.pi * m / 12)
    month_cos = np.cos(2 * np.pi * m / 12)
    return month_sin, month_cos

# --- Label mapping for predictions ---
label_map = {
    1: "Sunny",
    2: "Rainy",
    0: "Snowy"
}

# --- Make predictions ---
predictions = []
probabilities = []

for m in future_months:
    month_sin, month_cos = encode_month(m)
    X = np.array([[month_sin, month_cos, lat, lon]])
    probs = model.predict_proba(X)[0]
    pred_class_num = model.predict(X)[0]

    # Map numeric to text
    pred_class = label_map.get(int(pred_class_num), "Unknown")

    predictions.append(pred_class)
    probabilities.append(probs)

# --- Display results ---
st.markdown("### 🌦️ Predicted Weather Categories")
for i, m_label in enumerate(month_labels):
    st.write(f"**{m_label}** ({future_months[i]}):  **{predictions[i]}**")
    prob_dict = {label_map.get(int(cls), str(cls)): f"{prob:.2%}" for cls, prob in zip(model.classes_, probabilities[i])}
    st.progress(float(max(probabilities[i])))  # visual confidence
    st.caption(", ".join([f"{k}: {v}" for k, v in prob_dict.items()]))

# --- Add model performance summary ---
st.info("📊 Our XGBoost model achieves around **83–86% accuracy** and **high precision**, "
        "especially for Snowy conditions. Some overlap occurs between Sunny and Rainy predictions.")

st.divider()

# =========================
# SECTION 4 — Recommendations
# =========================
st.subheader("Section 4 — Smart Recommendations")



# Use the model prediction for the next 2 months
pred_next = predictions[1] if len(predictions) > 1 else predictions[0]

# --- Generate recommendation based on predicted weather ---
if pred_next.lower() == "rainy":
    recommendation = (
        "☔ **Rainy season ahead!** Expect frequent showers. "
        "Keep an umbrella handy, wear waterproof shoes, and plan indoor activities."
    )
elif pred_next.lower() == "snowy":
    recommendation = (
        "❄️ **Snowy conditions predicted.** Make sure your heating system is working, "
        "and check your vehicle’s winter tires if you're traveling."
    )
elif pred_next.lower() == "sunny":
    recommendation = (
        "🌞 **Sunny and clear weather expected!** Ideal for travel, outdoor events, and gardening. "
        "Remember sunscreen and hydration during hot days."
    )
else:
    recommendation = (
        "🌤️ **Mixed conditions ahead.** Stay flexible with your plans and check forecasts regularly."
    )

# --- Display the recommendation ---
st.markdown(f"### 🔍 Recommendation")
st.write(recommendation)








