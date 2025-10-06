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

# =========================
# SECTION 2 — Fetch & Summarize Weather (smart intervals + accurate "current")
# =========================
import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone

FETCH_TIMEOUT = 20  # seconds

# --- small helpers ---
def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def get_meteomatics_creds() -> (str, str):
    # expects .streamlit/secrets.toml:
    # [meteomatics]
    # username = "..."
    # password = "..."
    if "meteomatics" in st.secrets:
        u = st.secrets["meteomatics"].get("username")
        p = st.secrets["meteomatics"].get("password")
        if u and p:
            return u, p
    st.error("Missing Meteomatics credentials in .streamlit/secrets.toml under [meteomatics].")
    st.stop()

def try_fetch_interval(lat: float, lon: float, start_utc: datetime, end_utc: datetime,
                       interval: str, params: List[str]) -> Optional[pd.DataFrame]:
    """Try a specific interval like PT15M / PT1H; return df or None."""
    API_USERNAME, API_PASSWORD = get_meteomatics_creds()
    url = f"https://api.meteomatics.com/{iso_z(start_utc)}--{iso_z(end_utc)}:{interval}/{','.join(params)}/{lat},{lon}/json"
    r = requests.get(url, auth=(API_USERNAME, API_PASSWORD), timeout=FETCH_TIMEOUT)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return None
    rows = {}
    for p in data:
        param = p["parameter"]
        coords = p["coordinates"][0]
        for d in coords["dates"]:
            ts = d["date"]; val = d.get("value")
            rows.setdefault(ts, {})[param] = val
    if not rows:
        return None
    df = pd.DataFrame([{"datetime": pd.to_datetime(ts), **vals} for ts, vals in rows.items()])
    return df.sort_values("datetime")

def pick_nearest_index(ts_series: pd.Series, target: datetime) -> int:
    diffs = (ts_series - target).abs()
    return int(diffs.idxmin())

def interpolate_at(df: pd.DataFrame, target_utc: datetime, cols: List[str]) -> Tuple[pd.Series, float]:
    """
    Interpolate values at target_utc between surrounding timestamps.
    Returns (row_like_series, age_minutes_from_now; positive means in the past, negative future).
    """
    s = df.set_index("datetime").sort_index()
    now_utc = datetime.now(timezone.utc)
    age_min = (now_utc - target_utc).total_seconds() / 60.0

    # exact timestamp
    if target_utc in s.index:
        return s.loc[target_utc], age_min

    before = s.index[s.index < target_utc]
    after  = s.index[s.index > target_utc]

    if len(before) and len(after):
        t0 = before.max(); t1 = after.min()
        w = (target_utc - t0) / (t1 - t0)
        row0 = s.loc[t0]; row1 = s.loc[t1]
        out = {}
        for c in cols:
            if c in s.columns:
                v0 = row0[c]; v1 = row1[c]
                if pd.notna(v0) and pd.notna(v1):
                    out[c] = float(v0) * (1 - w) + float(v1) * w
                else:
                    out[c] = float(v0) if pd.notna(v0) else (float(v1) if pd.notna(v1) else np.nan)
        return pd.Series(out), age_min

    # fallback to nearest row
    nearest_idx = pick_nearest_index(df["datetime"], target_utc)
    return df.set_index("datetime").iloc[nearest_idx], age_min

# --- main fetch with smart interval & parameter fallbacks ---
@st.cache_data(show_spinner=True, ttl=15 * 60)
def fetch_weather_data(lat: float, lon: float, start_utc: datetime, end_utc: datetime,
                       prefer_fine: bool) -> Optional[pd.DataFrame]:
    """
    prefer_fine=True (today): try PT15M first, else PT1H.
    prefer_fine=False (history/forecast): use PT1H.
    Falls back if lightning/fresh_snow params aren’t in plan.
    """
    base_params: List[str] = [
        "t_2m:C",                    # air temp (°C)
        "precip_1h:mm",              # hourly precip (mm)
        "wind_speed_10m:ms",         # wind (m/s)
        "relative_humidity_2m:p",    # RH (%)
        "snow_depth:cm",             # snow on ground (cm)
        "fresh_snow_1h:cm",          # fresh snow per hour (cm) — may be restricted
    ]
    lightning_param = "lightning_strikes_10km_1h:x"

    intervals = ["PT1H"]
    if prefer_fine:
        intervals = ["PT15M", "PT1H"]

    # 1) with lightning
    for itv in intervals:
        try:
            df = try_fetch_interval(lat, lon, start_utc, end_utc, itv, base_params + [lightning_param])
            if df is not None:
                return df
        except requests.HTTPError as e:
            if not (e.response is not None and e.response.status_code in (400, 404)):
                raise

    # 2) without lightning
    for itv in intervals:
        try:
            df = try_fetch_interval(lat, lon, start_utc, end_utc, itv, base_params)
            if df is not None:
                return df
        except requests.HTTPError as e:
            if not (e.response is not None and e.response.status_code in (400, 404)):
                raise

    # 3) drop fresh_snow if still failing
    safest = [p for p in base_params if p != "fresh_snow_1h:cm"]
    for itv in intervals:
        df = try_fetch_interval(lat, lon, start_utc, end_utc, itv, safest)
        if df is not None:
            return df

    return None

# --- feels-like (wind-chill / heat-index) ---
def compute_feels_like(temp_c: float, wind_kmh: float, rh: float) -> float:
    if temp_c <= 10 and wind_kmh > 4.8:  # Wind chill (Environment Canada)
        v = wind_kmh
        feels = 13.12 + 0.6215 * temp_c - 11.37 * (v ** 0.16) + 0.3965 * temp_c * (v ** 0.16)
    elif temp_c >= 27 and rh >= 40:     # Heat index (NOAA)
        T, R = temp_c, rh
        feels = (-8.784695 + 1.61139411*T + 2.338549*R - 0.14611605*T*R
                 - 0.01230809*(T**2) - 0.01642482*(R**2)
                 + 0.00221173*(T**2)*R + 0.00072546*T*(R**2)
                 - 0.000003582*(T**2)*(R**2))
    else:
        feels = temp_c
    return round(feels, 1)

def summarize_weather(df: pd.DataFrame, target_local_mid: datetime, tz_name: Optional[str]) -> Dict:
    """
    Summarize daily stats and compute 'current':
    - Today: interpolate at 'now' (UTC).
    - Past/Future: interpolate at local noon.
    Also returns 'freshness_min' for today (how close to now the anchor is).
    """
    out: Dict[str, Optional[float]] = {}
    if df is None or df.empty:
        return out

    # daily stats
    if "t_2m:C" in df.columns:
        out["high_temp"] = round(float(df["t_2m:C"].max()), 1)
        out["low_temp"]  = round(float(df["t_2m:C"].min()), 1)
    if "precip_1h:mm" in df.columns:
        out["rain_total"] = round(float(df["precip_1h:mm"].fillna(0).sum()), 2)
    if "wind_speed_10m:ms" in df.columns:
        out["wind_avg"] = round(float(df["wind_speed_10m:ms"].mean() * 3.6), 1)
    if "relative_humidity_2m:p" in df.columns:
        out["humidity_avg"] = round(float(df["relative_humidity_2m:p"].mean()), 1)
    if "fresh_snow_1h:cm" in df.columns:
        out["snowfall_cm"] = round(float(df["fresh_snow_1h:cm"].fillna(0).sum()), 1)
    if "snow_depth:cm" in df.columns:
        out["snow_depth_cm"] = round(float(df["snow_depth:cm"].iloc[-1]), 1)

    # decide anchor time
    now_loc = datetime.now(ZoneInfo(tz_name)) if tz_name else datetime.now(timezone.utc)
    is_today = (target_local_mid.date() == now_loc.date())
    anchor_local = now_loc if is_today else target_local_mid
    anchor_utc = anchor_local.astimezone(timezone.utc)

    # interpolate/nearest for "current"
    cols = ["t_2m:C", "wind_speed_10m:ms", "relative_humidity_2m:p"]
    row_like, age_min = interpolate_at(df, anchor_utc, cols)

    if "t_2m:C" in row_like:
        out["current_temp"] = round(float(row_like["t_2m:C"]), 1)
    if "wind_speed_10m:ms" in row_like:
        out["wind_current"] = round(float(row_like["wind_speed_10m:ms"]) * 3.6, 1)
    if "relative_humidity_2m:p" in row_like:
        out["humidity_current"] = round(float(row_like["relative_humidity_2m:p"]), 1)

    # feels-like
    if set(cols).issubset(row_like.index):
        out["feels_like"] = compute_feels_like(
            float(row_like["t_2m:C"]),
            float(row_like["wind_speed_10m:ms"]) * 3.6,
            float(row_like["relative_humidity_2m:p"]),
        )

    # lightning flag
    out["lightning"] = bool(("lightning_strikes_10km_1h:x" in df.columns) and
                            (df["lightning_strikes_10km_1h:x"].fillna(0) > 0).any())

    # freshness label (only meaningful for today)
    out["freshness_min"] = None if not is_today else int(round(abs(age_min)))
    return out

# ---------- Pull state from Section 1 ----------
lat = st.session_state["lat"]
lon = st.session_state["lon"]
label = st.session_state["label"]
tz_name = st.session_state["tz_name"]
local_zone = st.session_state["local_zone"]
start_utc = st.session_state["start_utc"]
end_utc = st.session_state["end_utc"]
prefer_fine = st.session_state["prefer_fine"]
target_local_mid = st.session_state["target_local_mid"]

# ---------- Fetch & summarize ----------
st.subheader("Section 2 — Weather Data Summary")

df = fetch_weather_data(lat, lon, start_utc, end_utc, prefer_fine=prefer_fine)
if df is None or df.empty:
    st.warning("No weather data available for this location/date (or your account’s access window).")
    st.stop()

summary = summarize_weather(df, target_local_mid, tz_name)

# Freshness caption for today
if summary.get("freshness_min") is not None:
    st.caption(f"🕒 Current values near now (±{summary['freshness_min']} min).")

# --- Metrics UI ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🌡️ Current (°C)", summary.get("current_temp", "—"))
    st.metric("🌡️ Feels Like (°C)", summary.get("feels_like", "—"))
with col2:
    st.metric("❄️ Snowfall (cm)", summary.get("snowfall_cm", "—"))
    st.metric("🧊 Snow Depth (cm)", summary.get("snow_depth_cm", "—"))
with col3:
    st.metric("🌧️ Rain (mm)", summary.get("rain_total", "—"))
    st.metric("⚡ Lightning", "Yes" if summary.get("lightning", False) else "No")

# Secondary row
c4, c5, c6 = st.columns(3)
with c4:
    st.metric("🌡️ High (°C)", summary.get("high_temp", "—"))
with c5:
    st.metric("❄️ Low (°C)", summary.get("low_temp", "—"))
with c6:
    st.metric("🌬️ Avg Wind (km/h)", summary.get("wind_avg", "—"))

# Inform if some params weren’t available
missing_msgs = []
for p, label_p in [
    ("fresh_snow_1h:cm", "fresh snow"),
    ("lightning_strikes_10km_1h:x", "lightning"),
]:
    if p not in df.columns:
        missing_msgs.append(label_p)
if missing_msgs:
    st.info(f"{', '.join(missing_msgs).title()} data not included in this account/endpoint — showing other metrics only.")

# --- Hourly (or 15-min) chart(s) ---

# --- Pretty chart + user controls + CSV download ---
with st.expander("📈 Temperature, rain & snowfall over the day", expanded=True):
    pretty = {
        "t_2m:C": "Temperature (°C)",
        "precip_1h:mm": "Precipitation (mm)",
        "fresh_snow_1h:cm": "Fresh snow (cm)",
    }

    available_cols = [c for c in pretty if c in df.columns]
    if not available_cols:
        st.info("No time-series fields available for this account/endpoint.")
    else:
        # initialize default once
        if "chart_series" not in st.session_state:
            default_pick = ["t_2m:C"] if "t_2m:C" in available_cols else [available_cols[0]]
            st.session_state["chart_series"] = default_pick

        # quick select all / none
        col_sa, col_sn = st.columns([1,1])
        with col_sa:
            if st.button("Select all"):
                st.session_state["chart_series"] = available_cols
        with col_sn:
            if st.button("Clear"):
                st.session_state["chart_series"] = []

        pick = st.multiselect(
            "Choose series to display",
            options=available_cols,
            default=st.session_state["chart_series"],
            key="chart_series",  # persist across reruns
            format_func=lambda k: pretty.get(k, k),
            help="Select one or more metrics to plot.",
        )

        if not pick:
            st.warning("Select at least one series to plot.")
        else:
            # Build plotting dataframe with local time
            df_plot = df[["datetime"] + pick].copy()

            try:
                tz_name = st.session_state.get("tz_name")
                if tz_name:
                    df_plot["datetime_local"] = df_plot["datetime"].dt.tz_convert(ZoneInfo(tz_name))
                else:
                    df_plot["datetime_local"] = df_plot["datetime"]
            except Exception:
                df_plot["datetime_local"] = df_plot["datetime"]

            rename_map = {k: pretty[k] for k in pick}
            df_plot = df_plot.rename(columns=rename_map)

            value_vars = [rename_map[k] for k in pick]
            df_long = df_plot.melt(
                id_vars=["datetime_local"],
                value_vars=value_vars,
                var_name="Series",
                value_name="Value",
            )

            import altair as alt
            chart = (
                alt.Chart(df_long)
                .mark_line()
                .encode(
                    x=alt.X("datetime_local:T", title="Local time"),
                    y=alt.Y("Value:Q", title="Value"),
                    color=alt.Color("Series:N", title="Metric"),
                    tooltip=[
                        alt.Tooltip("datetime_local:T", title="Time"),
                        alt.Tooltip("Series:N", title="Metric"),
                        alt.Tooltip("Value:Q", title="Value", format=".2f"),
                    ],
                )
                .properties(height=360)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

            # CSV download
            st.markdown("**Download data**")
            export_local = st.checkbox("Use local time in CSV", value=True, key="csv_local_time")
            export_df = (
                df_plot[["datetime_local"] + value_vars].copy()
                if export_local
                else df[["datetime"] + pick].rename(columns=rename_map).rename(columns={"datetime": "datetime_utc"})
            )

            label_safe = st.session_state.get("label", "location").split(",")[0].replace(" ", "_")
            day_str = st.session_state.get("target_local_mid").strftime("%Y-%m-%d")
            fname = f"weather_{label_safe}_{day_str}.csv"

            st.download_button(
                "⬇️ Download CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name=fname,
                mime="text/csv",
            )





# ML MODEL and Recommendations
import pickle
import numpy as np
import streamlit as st

st.subheader("Section 3 — ML Weather Forecast")

# --- Load the trained model ---
@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# Use a relative path instead of a Windows one
model_path = os.path.join(os.path.dirname(__file__), "XGB_Classifier.pkl")
model = load_model(model_path)

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
    (month_input + 4 - 1) % 12 + 1,
    (month_input + 6 - 1) % 12 + 1
]
month_labels = ["Current Month", "Next 2 Months", "Next 4 Months", "Next 6 Months"]

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

current_temp = summary.get("current_temp", None)
rain_total = summary.get("rain_total", 0)
pred_next = predictions[1]  # next 2 months prediction

recommendation = ""

if pred_next.lower() == "rainy":
    recommendation = "☔ Expect wetter conditions in the coming months. Prepare rain gear and check local flood advisories."
elif pred_next.lower() == "snowy":
    recommendation = "❄️ Snowy conditions predicted — make sure heating and snow equipment are ready."
elif pred_next.lower() == "sunny":
    recommendation = "🌞 Expect mostly sunny conditions — great for outdoor activities and travel."

# Add hint based on API
if current_temp is not None:
    if current_temp < 5:
        recommendation += " It’s currently quite cold — dress warmly."
    elif current_temp > 25:
        recommendation += " The current temperature is hot — stay hydrated."

if rain_total and rain_total > 5:
    recommendation += " Recent rainfall suggests continued moisture in the area."

st.markdown(f"**Recommendation:** {recommendation}")





