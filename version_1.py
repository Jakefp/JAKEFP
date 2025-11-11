import streamlit as st
import pandas as pd
import gzip
from fitparse import FitFile
from io import BytesIO
from geopy.distance import geodesic
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
import itertools
import os
import xml.etree.ElementTree as ET  # for GPX parsing


# Set Mapbox token
os.environ["MAPBOX_ACCESS_TOKEN"] = "pk.eyJ1IjoiamFrZWZwLXNhaWxpbmciLCJhIjoiY21ha2theHZzMTN2NTJqcHoxeXRwMmFnOCJ9.VLxCKga5g5XjCZNos-kqGw"

st.set_page_config(layout="wide")
st.title("Multi-Sailor GPS Analyzer")

# ---------- Helper Functions ----------
def compute_bearing(p1, p2):
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def add_kinematics(df):
    """
    Expects a DataFrame with columns: 'lat', 'lon', 'time' (tz-aware).
    Returns a new DataFrame with speed, speed_knots, heading, distance_m.
    """
    # Ensure chronological order
    df = df.sort_values("time").reset_index(drop=True)

    lats = df["lat"].values
    lons = df["lon"].values
    times = df["time"].values

    speeds = []
    headings = []
    distances = [0]  # pairwise segment distances

    for i in range(1, len(df)):
        p1 = (lats[i - 1], lons[i - 1])
        p2 = (lats[i], lons[i])

        dist = geodesic(p1, p2).meters
        duration = (times[i] - times[i - 1]) / np.timedelta64(1, "s")

        speeds.append(dist / duration if duration > 0 else 0)
        headings.append(compute_bearing(p1, p2))
        distances.append(dist)

    # Drop first row to match original behaviour
    df = df.iloc[1:].copy()
    df["speed"] = speeds
    df["speed_knots"] = df["speed"] * 1.94384
    df["heading"] = headings
    df["distance_m"] = distances[1:]
    return df

# ----------- Uploading functions -------------

def process_fit_gz(file_obj):
    with gzip.open(file_obj, 'rb') as f:
        fit_data = f.read()
    fitfile = FitFile(BytesIO(fit_data))
    records = []
    for record in fitfile.get_messages("record"):
        data = {field.name: field.value for field in record}
        if "position_lat" in data and "position_long" in data:
            records.append(data)
    df = pd.DataFrame(records)

    df["lat"] = df["position_lat"] * (180 / 2**31)
    df["lon"] = df["position_long"] * (180 / 2**31)

    # Drop invalid rows
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]

    # Fit timestamps are usually UTC
    df["time"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC").dt.tz_convert("Australia/Sydney")

    # Compute speed, heading, and distance
    df = add_kinematics(df)
    return df

def process_gpx(file_obj):
    # Read and parse XML
    content = file_obj.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")

    root = ET.fromstring(content)

    points = []
    for trkpt in root.findall(".//{*}trkpt"):
        lat = trkpt.attrib.get("lat")
        lon = trkpt.attrib.get("lon")
        time_el = trkpt.find("{*}time") or trkpt.find(".//{*}time")

        if lat is None or lon is None or time_el is None:
            continue

        points.append({
            "lat": float(lat),
            "lon": float(lon),
            "timestamp": time_el.text
        })

    df = pd.DataFrame(points)
    if df.empty:
        raise ValueError("No trackpoints found in GPX file")

    # GPX times are typically UTC (with 'Z')
    df["time"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Australia/Sydney")

    # Basic sanity filter on positions
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]

    df = add_kinematics(df)
    return df

def process_csv(file_obj):
    df_raw = pd.read_csv(file_obj)

    if df_raw.empty:
        raise ValueError("CSV file is empty")

    # Try to detect column names (case-insensitive)
    cols_lower = {c.lower(): c for c in df_raw.columns}

    lat_col = next((cols_lower[c] for c in ["lat", "latitude", "position_lat"]), None)
    lon_col = next((cols_lower[c] for c in ["lon", "lng", "longitude", "position_long"]), None)
    time_col = next((cols_lower[c] for c in ["time", "timestamp", "date_time", "datetime"]), None)

    if lat_col is None or lon_col is None or time_col is None:
        raise ValueError(
            "CSV must contain latitude, longitude, and time columns "
            "(e.g. 'lat'/'latitude', 'lon'/'longitude', 'time'/'timestamp')."
        )

    df = df_raw.copy()
    df["lat"] = df[lat_col].astype(float)
    df["lon"] = df[lon_col].astype(float)

    # Parse time; if tz-naive, assume Australia/Sydney; if tz-aware, convert.
    times = pd.to_datetime(df[time_col], errors="coerce")
    if times.dt.tz is None:
        times = times.dt.tz_localize("Australia/Sydney")
    else:
        times = times.dt.tz_convert("Australia/Sydney")
    df["time"] = times

    # Drop bad rows
    df = df.dropna(subset=["lat", "lon", "time"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]

    df = add_kinematics(df)
    return df

# ---------- Upload & Process ----------
uploaded_files = st.file_uploader(
    "Upload one or more GPS files (.fit.gz, .fit, .gpx, .csv)",
    type=["gz", "fit", "gpx", "csv"],
    accept_multiple_files=True
)

sailor_data = []
colors = itertools.cycle(px.colors.qualitative.Set2)

if uploaded_files:
    st.subheader("Sailor Assignments")
    for file in uploaded_files:
        filename = file.name
        sailor_name = st.text_input(
            f"Name for {filename}",
            value=filename.split('.')[0],
            key=filename
        )

        ext = filename.lower()

        try:
            if ext.endswith(".fit.gz") or ext.endswith(".fit"):
                df = process_fit_gz(file)
            elif ext.endswith(".gpx"):
                df = process_gpx(file)
            elif ext.endswith(".csv"):
                df = process_csv(file)
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            df["sailor"] = sailor_name.strip()
            color = next(colors)
            sailor_data.append({"name": sailor_name, "df": df, "color": color})

        except Exception as e:
            st.error(f"Failed to parse {filename}: {e}")

# ---------- If data loaded ----------
if sailor_data:
    all_times = pd.concat([s["df"]["time"] for s in sailor_data])
    all_lats = pd.concat([s["df"]["lat"] for s in sailor_data])
    all_lons = pd.concat([s["df"]["lon"] for s in sailor_data])
    start_time, end_time = all_times.min(), all_times.max()

        # ---------- Time range selection ----------
    st.markdown("### Time ranges")
    ranges_count = st.number_input(
        "Number of time ranges",
        min_value=1,
        max_value=4,
        value=1,
        step=1,
        format="%d",
    )

    time_ranges = []
    for i in range(ranges_count):
        start_i, end_i = st.slider(
            f"Time range {i + 1}",
            min_value=start_time.to_pydatetime(),
            max_value=end_time.to_pydatetime(),
            value=(start_time.to_pydatetime(), end_time.to_pydatetime()),
            format="MM-DD HH:mm:ss",
            step=timedelta(minutes=1),
            key=f"time_range_{i + 1}",
        )
        time_ranges.append((start_i, end_i))

    smoothing = st.slider(
    "Polar smoothing amount", min_value=0.0, max_value=1.0, value=0.3, step=0.05,
    help="0 = no smoothing, 1 = full smoothing with neighbors"
    )

    # 🆕 Minimum boatspeed filter for polar plot
    min_speed_knots = st.number_input(
    "Minimum boatspeed to include in polars (knots)",
    min_value=0.0,
    max_value=50.0,
    value=0.0,
    step=0.01,
    format="%.2f",
    help="Speeds below this value are excluded from the polar averages"
    )

    # ---------- Polar Plot ----------
    polar_fig = go.Figure()
    polar_fig.update_layout(polar=dict(
        angularaxis=dict(
            direction="clockwise",
            rotation=90,
            tickmode="linear",
            tick0=0,
            dtick=30,
            tickvals=[0, 90, 180, 270],
            ticktext=["N", "E", "S", "W"]
        ),
        radialaxis=dict(title="Speed (knots)")
    ))

    # ---------- Map Plot ----------
    track_fig = go.Figure()
    track_fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=all_lats.mean(), lon=all_lons.mean()),
            zoom=15 if all_lats.std() + all_lons.std() < 0.01 else 13 if all_lats.std() + all_lons.std() < 0.05 else 12
        ),
        height=500,
        margin=dict(r=0, l=0, t=0, b=0)
    )

    # ---------- Process and Plot Each Sailor ----------
    summary_rows = []
    for sailor in sailor_data:
        df = sailor["df"]
        name = sailor["name"]
        color = sailor["color"]

        # ---------- Apply one or two time ranges ----------
        time_mask = pd.Series(False, index=df.index)
        for start_t, end_t in time_ranges:
            time_mask |= (df["time"] >= start_t) & (df["time"] <= end_t)

        filtered = df[time_mask].copy()
        if filtered.empty:
            continue

        # 🆕 Boatspeed filter for POLAR ONLY
        polar_filtered = filtered[filtered["speed_knots"] >= min_speed_knots].copy()

        # ---------- Polar (only if we have data after speed filter) ----------
        if not polar_filtered.empty:
            polar_filtered.loc[:, "dir_bin"] = (polar_filtered["heading"] // 5) * 5
            avg = polar_filtered.groupby("dir_bin")["speed_knots"].mean().reset_index()

            # Fill all 5° bins
            all_bins = pd.DataFrame({"dir_bin": np.arange(0, 360, 5)})
            avg = all_bins.merge(avg, on="dir_bin", how="left").fillna(0)

            # Smoothing weights
            center_weight = 1 - smoothing
            neighbor_weight = smoothing / 2

            # Apply weighted smoothing
            smoothed = []
            for i in range(len(avg)):
                prev = avg.iloc[i - 1]["speed_knots"] if i > 0 else avg.iloc[-1]["speed_knots"]
                curr = avg.iloc[i]["speed_knots"]
                next_ = avg.iloc[(i + 1) % len(avg)]["speed_knots"]
                smooth_val = neighbor_weight * prev + center_weight * curr + neighbor_weight * next_
                smoothed.append(smooth_val)

            avg["smoothed_speed"] = smoothed

        polar_fig.add_trace(go.Scatterpolar(
            r=avg["smoothed_speed"],
            theta=avg["dir_bin"],
            name=name,
            mode="lines+markers",
            line=dict(color=color)
        ))

        # Track
        track_fig.add_trace(go.Scattermapbox(
        lat=filtered["lat"],
        lon=filtered["lon"],
        mode="lines",
        name=name,
        line=dict(color=color),
        text=filtered["time"].dt.strftime("%H:%M:%S"),
        hoverinfo="text"
    ))

        # Summary values
        total_distance = filtered["distance_m"].sum() / 1000
        avg_speed = filtered["speed_knots"].mean()
        summary_rows.append({
            "Sailor": name,
            "Distance (km)": round(total_distance, 2),
            "Avg Speed (kn)": round(avg_speed, 2),
            "Color": color
        })

    summary_df = pd.DataFrame(summary_rows)

    # ---------- Layout: Map and Polar ----------
    col1, col2 = st.columns([2, 2])
    with col1:
        st.subheader("Track Map")
        st.plotly_chart(track_fig, use_container_width=True)

    with col2:
        st.subheader("Polar Diagram")
        st.plotly_chart(polar_fig, use_container_width=True)


# ========== Summary Table ============
st.subheader("Overalls")

summary_rows = []

for sailor in sailor_data:
    df = sailor["df"]
    name = sailor["name"]
    color = sailor["color"]

    # Apply the same time ranges as for the plots
    time_mask = pd.Series(False, index=df.index)
    for start_t, end_t in time_ranges:
        time_mask |= (df["time"] >= start_t) & (df["time"] <= end_t)

    filtered = df[time_mask].copy()
    if filtered.empty:
        continue

    total_distance = filtered["distance_m"].sum() / 1000  # km
    avg_speed = filtered["speed_knots"].mean()

    summary_rows.append({
        "Sailor": name,
        "Distance (km)": round(total_distance, 2),
        "Avg Speed (kn)": round(avg_speed, 2)
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True)
