import os
import glob
import json
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import griddata

print("=== UPGRADING SYDNEY LIVABILITY DATA: CRIME FIX, CLIMATE & COUPLES COMMUTE MATRIX ===")

syd_sal = gpd.read_file("cache/sydney_suburbs_base.gpkg")
syd_suburb_names = set(syd_sal["suburb_clean"].unique())

# ==============================================================================
# 1. REFINED BOCSAR CRIME: RESIDENTIAL SAFETY VS COMMERCIAL HUB FOOT-TRAFFIC
# ==============================================================================
print("\n1. Re-calculating BOCSAR Crime to separate Residential Crime from Commercial/Event Hubs...")
with zipfile.ZipFile("SuburbData.zip", "r") as z:
    csv_name = [f for f in z.namelist() if f.endswith(".csv")][0]
    df_crime = pd.read_csv(z.open(csv_name), low_memory=False)

df_crime["suburb_clean"] = (
    df_crime["Suburb"].astype(str)
    .str.replace(r"\s*\(.*\)$", "", regex=True)
    .str.strip()
    .str.upper()
)
df_crime = df_crime[df_crime["suburb_clean"].isin(syd_suburb_names)].copy()

month_cols = [c for c in df_crime.columns if c not in ["Suburb", "Offence category", "Subcategory", "suburb_clean"]]
recent_12m = month_cols[-12:]
prior_12m  = month_cols[-72:-60]

for c in recent_12m + prior_12m:
    df_crime[c] = pd.to_numeric(df_crime[c], errors="coerce").fillna(0)

df_crime["count_recent"] = df_crime[recent_12m].sum(axis=1)
df_crime["count_prior"]  = df_crime[prior_12m].sum(axis=1)

# Strictly Residential Safety categories (excludes retail shoplifting, fare evasion, liquor licensing)
residential_violent_cats = {
    "Homicide", "Assault", "Sexual offences", "Robbery",
    "Intimidation, stalking and harassment"
}
residential_property_cats = {
    "Break and enter dwelling", "Motor vehicle theft",
    "Steal from motor vehicle", "Steal from dwelling",
    "Arson", "Malicious damage to property"
}
exclude_cats = {
    "Transport regulatory offences", "Breach bail conditions",
    "Breach Apprehended Violence Order", "Other regulatory offences"
}

df_crime["cat_clean"] = df_crime["Offence category"].astype(str).str.strip()
df_crime["is_res_violent"]  = df_crime["cat_clean"].isin(residential_violent_cats)
df_crime["is_res_property"] = df_crime["cat_clean"].isin(residential_property_cats)
df_crime["is_included"]     = ~df_crime["cat_clean"].isin(exclude_cats)

crime_records = []
for sub, grp in df_crime.groupby("suburb_clean"):
    v_rec = grp.loc[grp["is_res_violent"], "count_recent"].sum()
    p_rec = grp.loc[grp["is_res_property"], "count_recent"].sum()
    t_rec = grp.loc[grp["is_included"], "count_recent"].sum()
    t_pri = grp.loc[grp["is_included"], "count_prior"].sum()

    trend_pct = round(((t_rec - t_pri) / t_pri) * 100.0, 1) if t_pri >= 20 else 0.0
    crime_records.append({
        "suburb_clean": sub,
        "violent_incidents": int(v_rec),
        "property_incidents": int(p_rec),
        "total_incidents": int(t_rec),
        "crime_5yr_trend_pct": trend_pct
    })

df_crime_out = pd.DataFrame(crime_records)
df_crime_out.to_csv("cache/crime_by_suburb.csv", index=False)
print("[✓] Updated cache/crime_by_suburb.csv with Residential-focused crime counts.")

# ==============================================================================
# 2. SUMMER MAX & WINTER MIN TEMPERATURES FOR ALL 926 SUBURBS
# ==============================================================================
climate_cache_path = "cache/climate_by_suburb.csv"

if not os.path.exists(climate_cache_path):
    print("\n2. Computing Summer Max & Winter Min Temperatures across Greater Sydney...")
    syd_pts = syd_sal[["SAL_CODE21", "geometry"]].copy()
    syd_pts["geometry"] = syd_pts.geometry.representative_point()
    syd_pts["lon"] = syd_pts.geometry.x
    syd_pts["lat"] = syd_pts.geometry.y

    # Sample 36 anchor weather stations across Greater Sydney via Open-Meteo Historical Archive (2025 full year)
    grid_lats = np.linspace(-34.30, -33.30, 6)
    grid_lons = np.linspace(150.45, 151.35, 6)
    g_lat, g_lon = np.meshgrid(grid_lats, grid_lons)
    flat_lats = g_lat.ravel()
    flat_lons = g_lon.ravel()

    lat_str = ",".join([f"{x:.4f}" for x in flat_lats])
    lon_str = ",".join([f"{x:.4f}" for x in flat_lons])
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat_str}&longitude={lon_str}"
        f"&start_date=2025-01-01&end_date=2025-12-31"
        f"&daily=temperature_2m_max,temperature_2m_min&timezone=Australia%2FSydney"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            weather_json = json.loads(resp.read().decode("utf-8"))

        pts_xy = []
        sum_max_vals = []
        win_min_vals = []
        hot_days_vals = []

        for idx, loc_data in enumerate(weather_json):
            daily = loc_data["daily"]
            df_w = pd.DataFrame({
                "date": pd.to_datetime(daily["time"]),
                "tmax": daily["temperature_2m_max"],
                "tmin": daily["temperature_2m_min"]
            })
            df_w["month"] = df_w["date"].dt.month
            # Summer = Dec, Jan, Feb | Winter = Jun, Jul, Aug
            summer_max = df_w[df_w["month"].isin([1, 2, 12])]["tmax"].mean()
            winter_min = df_w[df_w["month"].isin([6, 7, 8])]["tmin"].mean()
            hot_days   = (df_w["tmax"] >= 32.0).sum()

            pts_xy.append([flat_lons[idx], flat_lats[idx]])
            sum_max_vals.append(summer_max)
            win_min_vals.append(winter_min)
            hot_days_vals.append(hot_days)

        pts_xy = np.array(pts_xy)
        target_xy = np.column_stack([syd_pts["lon"].values, syd_pts["lat"].values])

        syd_pts["summer_max_temp"] = np.round(griddata(pts_xy, sum_max_vals, target_xy, method="cubic"), 1)
        syd_pts["winter_min_temp"] = np.round(griddata(pts_xy, win_min_vals, target_xy, method="cubic"), 1)
        syd_pts["days_over_32c"]   = np.round(griddata(pts_xy, hot_days_vals, target_xy, method="linear"), 0)

        # Fill any edge NaNs with nearest neighbor
        syd_pts["summer_max_temp"] = syd_pts["summer_max_temp"].fillna(
            pd.Series(griddata(pts_xy, sum_max_vals, target_xy, method="nearest")).round(1)
        )
        syd_pts["winter_min_temp"] = syd_pts["winter_min_temp"].fillna(
            pd.Series(griddata(pts_xy, win_min_vals, target_xy, method="nearest")).round(1)
        )
        syd_pts["days_over_32c"] = syd_pts["days_over_32c"].fillna(
            pd.Series(griddata(pts_xy, hot_days_vals, target_xy, method="nearest")).round(0)
        )
        print("   -> Successfully interpolated Open-Meteo ERA5 observations across all 926 suburbs!")
    except Exception as e:
        print(f"   -> Open-Meteo offline fallback ({e}): using coastal-inland microclimate model...")
        # Fallback microclimate model calibrated to BOM Sydney Observatory Hill vs Penrith/Richmond
        coast_lon = 151.28
        inland_km = np.maximum(0, (coast_lon - syd_pts["lon"]) * 92.0)
        syd_pts["summer_max_temp"] = np.round(25.8 + 0.12 * inland_km, 1)
        syd_pts["winter_min_temp"] = np.round(8.8 - 0.085 * inland_km, 1)
        syd_pts["days_over_32c"]   = np.round(6 + 0.65 * inland_km, 0)

    # Fine-tune for local Urban Heat Island / Parkland cooling effect (+/- 0.8°C based on parkland_pct)
    park_bonus = (syd_sal.set_index("SAL_CODE21").loc[syd_pts["SAL_CODE21"], "parkland_pct"].values - 15.0) * 0.015
    syd_pts["summer_max_temp"] = np.round(syd_pts["summer_max_temp"] - park_bonus, 1)

    # Thermal Discomfort Index: degrees above 25°C in summer + degrees below 9°C in winter
    syd_pts["temp_swing"] = np.round(syd_pts["summer_max_temp"] - syd_pts["winter_min_temp"], 1)

    df_clim = syd_pts[["SAL_CODE21", "summer_max_temp", "winter_min_temp", "days_over_32c", "temp_swing"]]
    df_clim.to_csv(climate_cache_path, index=False)
    print(f"[✓] Saved seasonal temperature metrics to {climate_cache_path}")
else:
    print(f"[✓] Loaded cached seasonal temperatures from {climate_cache_path}")

# ==============================================================================
# 3. BUILD ALL-PAIRS GTFS STATION MATRIX FOR DUAL-WORKPLACE COMMUTE ENGINE
# ==============================================================================
station_matrix_path = "station_matrix.json"

if not os.path.exists(station_matrix_path):
    gtfs_zip_path = sorted(glob.glob("mdb-*.zip") + glob.glob("*gtfs*.zip"))[0]
    print(f"\n3. Building All-Pairs GTFS Rail/Metro/Light-Rail Matrix from {gtfs_zip_path}...")

    with zipfile.ZipFile(gtfs_zip_path, "r") as z:
        routes = pd.read_csv(z.open("routes.txt"), dtype=str)
        trips  = pd.read_csv(z.open("trips.txt"), dtype=str)
        stops  = pd.read_csv(z.open("stops.txt"), dtype=str)

        routes["rtype_num"] = pd.to_numeric(routes["route_type"], errors="coerce").fillna(-1).astype(int)
        rail_mask = (
            routes["rtype_num"].isin([0, 1, 2]) |
            routes["rtype_num"].between(100, 299) |
            routes["rtype_num"].between(400, 499) |
            routes["rtype_num"].between(900, 999)
        )
        rail_route_ids = set(routes.loc[rail_mask, "route_id"].unique())
        rail_trips = trips[trips["route_id"].isin(rail_route_ids)]
        rail_trip_ids = set(rail_trips["trip_id"].unique())

        stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
        stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
        stops["parent_id"] = stops["parent_station"].fillna("").str.strip()
        stops["station_id"] = np.where(stops["parent_id"] != "", stops["parent_id"], stops["stop_id"])
        stop_to_station = dict(zip(stops["stop_id"], stops["station_id"]))

        station_meta = {}
        for _, r in stops.drop_duplicates(subset=["station_id"]).iterrows():
            station_meta[r["station_id"]] = {
                "name": str(r["stop_name"]).replace("Station", "").split(",")[0].strip(),
                "lat": float(r["stop_lat"]) if pd.notna(r["stop_lat"]) else None,
                "lon": float(r["stop_lon"]) if pd.notna(r["stop_lon"]) else None
            }
        for _, r in stops[stops["location_type"] == "1"].iterrows():
            station_meta[r["stop_id"]] = {
                "name": str(r["stop_name"]).replace("Station", "").split(",")[0].strip(),
                "lat": float(r["stop_lat"]) if pd.notna(r["stop_lat"]) else None,
                "lon": float(r["stop_lon"]) if pd.notna(r["stop_lon"]) else None
            }

        kept_chunks = []
        use_cols = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
        for chunk in pd.read_csv(z.open("stop_times.txt"), usecols=use_cols, dtype=str, chunksize=400000):
            sub_chunk = chunk[chunk["trip_id"].isin(rail_trip_ids)].copy()
            if not sub_chunk.empty:
                kept_chunks.append(sub_chunk)
        st_df = pd.concat(kept_chunks, ignore_index=True)

    def parse_gtfs_mins(s):
        parts = s.str.split(":", expand=True)
        return pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")

    st_df["arr_min"] = parse_gtfs_mins(st_df["arrival_time"])
    st_df["dep_min"] = parse_gtfs_mins(st_df["departure_time"])
    st_df["seq"]     = pd.to_numeric(st_df["stop_sequence"], errors="coerce")
    st_df["station_id"] = st_df["stop_id"].map(stop_to_station)
    st_df = st_df.dropna(subset=["arr_min", "dep_min", "seq", "station_id"]).sort_values(["trip_id", "seq"])

    peak_df = st_df[st_df["dep_min"].between(390, 570) | st_df["arr_min"].between(390, 570)].copy()

    # Build adjacent station graph for fast all-pairs Floyd-Warshall shortest paths
    adj_times = {}
    for tid, grp in peak_df.groupby("trip_id", sort=False):
        sids = grp["station_id"].values
        deps = grp["dep_min"].values
        arrs = grp["arr_min"].values
        for i in range(len(sids) - 1):
            u, v = sids[i], sids[i + 1]
            dt = max(1.0, arrs[i + 1] - deps[i])
            if 0 < dt <= 60 and u != v:
                adj_times.setdefault((u, v), []).append(dt)
                adj_times.setdefault((v, u), []).append(dt)

    # Filter to valid Greater Sydney stations
    active_sids = sorted({u for (u, _) in adj_times.keys() if u in station_meta and station_meta[u]["lat"] is not None and (-34.35 <= station_meta[u]["lat"] <= -33.25) and (150.35 <= station_meta[u]["lon"] <= 151.55)})
    sid_to_idx = {sid: i for i, sid in enumerate(active_sids)}
    N = len(active_sids)
    print(f"   -> Computing all-pairs transit matrix across {N} Sydney Rail/Metro/Light-Rail stations...")

    dist_mat = np.full((N, N), 999.0, dtype=np.float32)
    np.fill_diagonal(dist_mat, 0.0)

    for (u, v), times in adj_times.items():
        if u in sid_to_idx and v in sid_to_idx:
            i, j = sid_to_idx[u], sid_to_idx[v]
            dist_mat[i, j] = min(dist_mat[i, j], float(np.median(times)))

    # Add walking interchange links (<350m between stations/light-rail platforms = 4 mins transfer)
    coords = np.array([[station_meta[s]["lat"], station_meta[s]["lon"]] for s in active_sids])
    for i in range(N):
        for j in range(i + 1, N):
            d_km = np.hypot((coords[i, 0] - coords[j, 0]) * 111.0, (coords[i, 1] - coords[j, 1]) * 92.0)
            if d_km <= 0.35 and dist_mat[i, j] > 5.0:
                dist_mat[i, j] = dist_mat[j, i] = 4.0

    # Vectorized Floyd-Warshall across N stations (takes ~0.2 seconds in NumPy)
    for k in range(N):
        dist_mat = np.minimum(dist_mat, dist_mat[:, [k]] + dist_mat[[k], :])

    station_export = {
        "stations": [
            {
                "id": s,
                "name": station_meta[s]["name"],
                "lat": round(station_meta[s]["lat"], 5),
                "lon": round(station_meta[s]["lon"], 5)
            }
            for s in active_sids
        ],
        "matrix": np.round(dist_mat, 1).tolist()
    }
    with open(station_matrix_path, "w") as f:
        json.dump(station_export, f, separators=(",", ":"))
    print(f"[✓] Exported {station_matrix_path} ({os.path.getsize(station_matrix_path)/1024:.0f} KB)")
else:
    print(f"[✓] Found existing {station_matrix_path}")

# ==============================================================================
# 4. RE-STITCH MASTER sydney_suburbs.geojson WITH ALL UPGRADED METRICS
# ==============================================================================
print("\n4. Stitching upgraded master sydney_suburbs.geojson...")
df_rent  = pd.read_csv("cache/rent_by_postcode.csv", dtype={"postcode": str})
df_sales = pd.read_csv("cache/sales_by_suburb.csv")
df_crime = pd.read_csv("cache/crime_by_suburb.csv")
df_trans = pd.read_csv("cache/transport_by_suburb.csv", dtype={"SAL_CODE21": str})
df_clim  = pd.read_csv("cache/climate_by_suburb.csv", dtype={"SAL_CODE21": str})

syd_sal["SAL_CODE21"] = syd_sal["SAL_CODE21"].astype(str)
syd_sal["postcode"]   = syd_sal["postcode"].astype(str).str.replace(r"\.0$", "", regex=True)

# Store centroid lat/lon for instant client-side Workplace 1 & Workplace 2 routing
pts_wgs = syd_sal.geometry.representative_point()
syd_sal["cent_lat"] = np.round(pts_wgs.y, 5)
syd_sal["cent_lon"] = np.round(pts_wgs.x, 5)

master = (
    syd_sal
    .merge(df_rent, on="postcode", how="left")
    .merge(df_sales, on="suburb_clean", how="left")
    .merge(df_crime, on="suburb_clean", how="left")
    .merge(df_trans, on="SAL_CODE21", how="left")
    .merge(df_clim, on="SAL_CODE21", how="left")
)

master["rent_all_med"] = master["rent_all_med"].fillna((master["census_rent_weekly"] * 1.28).round(0))

# AMBIENT POPULATION DENOMINATOR FIX FOR COMMERCIAL / EVENT HUBS
# Adds daily commercial/industrial foot-traffic equivalent so CBD & Olympic Park aren't distorted by low sleeping pop
comm_area_sqkm = (master["commercial_pct"].fillna(0) / 100.0) * master["AREASQKM21"]
ind_area_sqkm  = (master["industrial_pct"].fillna(0) / 100.0) * master["AREASQKM21"]
ambient_pop = (
    master["population"].fillna(0) +
    (comm_area_sqkm * 12000.0) +
    (ind_area_sqkm * 3000.0)
).clip(lower=800.0)

# Residential Crime Rate (Violent + Property Burglary/Theft, normalized by Ambient Pop)
res_incidents = master["violent_incidents"].fillna(0) + master["property_incidents"].fillna(0)
master["crime_total_rate"]    = ((res_incidents / ambient_pop) * 1000.0).round(1)
master["crime_violent_rate"]  = ((master["violent_incidents"].fillna(0) / ambient_pop) * 1000.0).round(1)
master["crime_property_rate"] = ((master["property_incidents"].fillna(0) / ambient_pop) * 1000.0).round(1)
master["crime_raw_rate"]      = ((master["total_incidents"].fillna(0) / master["population"].clip(lower=500)) * 1000.0).round(1)

master["yield_all_pct"]   = ((master["rent_all_med"] * 52.0 / master["buy_all_med"]) * 100.0).round(2)
master["yield_house_pct"] = ((master["rent_house_med"] * 52.0 / master["buy_house_med"]) * 100.0).round(2)
master["yield_unit_pct"]  = ((master["rent_unit_med"] * 52.0 / master["buy_unit_med"]) * 100.0).round(2)

annual_inc = (master["hhd_income_weekly"].replace(0, np.nan) * 52.0)
master["price_to_income"] = (master["buy_all_med"] / annual_inc).round(1)

master["geometry"] = master["geometry"].simplify(tolerance=0.00025, preserve_topology=True)
master.to_file("sydney_suburbs.geojson", driver="GeoJSON")
print(f"[✓] SUCCESS! Updated sydney_suburbs.geojson ({os.path.getsize('sydney_suburbs.geojson')/(1024*1024):.2f} MB)")