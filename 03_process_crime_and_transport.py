import os
import glob
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

os.makedirs("cache", exist_ok=True)

# Load Base Suburbs from Step 2
syd_sal = gpd.read_file("cache/sydney_suburbs_base.gpkg")
syd_suburb_names = set(syd_sal["suburb_clean"].unique())

# ==============================================================================
# 1. PROCESS BOCSAR CRIME DATA (SuburbData.zip)
# ==============================================================================
crime_cache_path = "cache/crime_by_suburb.csv"

if not os.path.exists(crime_cache_path):
    print("1. Processing BOCSAR Suburb Crime Data (Jul 2025 - Jun 2026)...")
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

    # Identify the last 12 months (Jul 2025 - Jun 2026) and 5 years prior (Jul 2020 - Jun 2021)
    month_cols = [c for c in df_crime.columns if c not in ["Suburb", "Offence category", "Subcategory", "suburb_clean"]]
    recent_12m = month_cols[-12:]
    prior_12m  = month_cols[-72:-60]
    print(f"   -> Current 12m window: {recent_12m[0]} to {recent_12m[-1]}")
    print(f"   -> Prior 5yr window:   {prior_12m[0]} to {prior_12m[-1]}")

    for c in recent_12m + prior_12m:
        df_crime[c] = pd.to_numeric(df_crime[c], errors="coerce").fillna(0)

    df_crime["count_recent"] = df_crime[recent_12m].sum(axis=1)
    df_crime["count_prior"]  = df_crime[prior_12m].sum(axis=1)

    violent_cats = {
        "Homicide", "Assault", "Sexual offences", "Robbery",
        "Blackmail and extortion", "Intimidation, stalking and harassment",
        "Other offences against the person"
    }
    property_cats = {
        "Break and enter dwelling", "Break and enter non-dwelling",
        "Receiving or handling stolen goods", "Motor vehicle theft",
        "Steal from motor vehicle", "Steal from retail store",
        "Steal from dwelling", "Steal from person", "Stock theft",
        "Other theft", "Fraud", "Arson", "Malicious damage to property"
    }
    exclude_cats = {
        "Transport regulatory offences", "Breach bail conditions",
        "Breach Apprehended Violence Order", "Other regulatory offences"
    }

    df_crime["cat_clean"] = df_crime["Offence category"].astype(str).str.strip()
    df_crime["is_violent"]  = df_crime["cat_clean"].isin(violent_cats)
    df_crime["is_property"] = df_crime["cat_clean"].isin(property_cats)
    df_crime["is_included"] = ~df_crime["cat_clean"].isin(exclude_cats)

    crime_records = []
    for sub, grp in df_crime.groupby("suburb_clean"):
        v_rec = grp.loc[grp["is_violent"], "count_recent"].sum()
        p_rec = grp.loc[grp["is_property"], "count_recent"].sum()
        t_rec = grp.loc[grp["is_included"], "count_recent"].sum()
        t_pri = grp.loc[grp["is_included"], "count_prior"].sum()

        # 5-year trend (%) where base count >= 20
        trend_pct = round(((t_rec - t_pri) / t_pri) * 100.0, 1) if t_pri >= 20 else 0.0
        crime_records.append({
            "suburb_clean": sub,
            "violent_incidents": int(v_rec),
            "property_incidents": int(p_rec),
            "total_incidents": int(t_rec),
            "crime_5yr_trend_pct": trend_pct
        })

    df_crime_out = pd.DataFrame(crime_records)
    df_crime_out.to_csv(crime_cache_path, index=False)
    print(f"[✓] Saved BOCSAR crime aggregations for {len(df_crime_out)} suburbs to {crime_cache_path}")
else:
    print(f"[✓] Loaded cached crime metrics from {crime_cache_path}")

# ==============================================================================
# 2. PROCESS TfNSW GTFS (STATION DISTANCE & MORNING PEAK CBD COMMUTE)
# ==============================================================================
transport_cache_path = "cache/transport_by_suburb.csv"

if not os.path.exists(transport_cache_path):
    gtfs_zip_path = sorted(glob.glob("mdb-*.zip") + glob.glob("*gtfs*.zip"))[0]
    print(f"\n2. Processing Public Transport GTFS from {gtfs_zip_path}...")

    with zipfile.ZipFile(gtfs_zip_path, "r") as z:
        routes = pd.read_csv(z.open("routes.txt"), dtype=str)
        trips  = pd.read_csv(z.open("trips.txt"), dtype=str)
        stops  = pd.read_csv(z.open("stops.txt"), dtype=str)

        # 1. Filter to Rail (2/100s), Metro (1/400s), and Light Rail (0/900s)
        routes["rtype_num"] = pd.to_numeric(routes["route_type"], errors="coerce").fillna(-1).astype(int)
        rail_mask = (
            routes["rtype_num"].isin([0, 1, 2]) |
            routes["rtype_num"].between(100, 299) |
            routes["rtype_num"].between(400, 499) |
            routes["rtype_num"].between(900, 999)
        )
        rail_route_ids = set(routes.loc[rail_mask, "route_id"].unique())

        # Filter to weekday services if calendar.txt has weekday flags
        weekday_services = None
        if "calendar.txt" in z.namelist():
            cal = pd.read_csv(z.open("calendar.txt"), dtype=str)
            if "wednesday" in cal.columns:
                wd = cal[cal["wednesday"] == "1"]["service_id"].unique()
                if len(wd) > 0:
                    weekday_services = set(wd)

        rail_trips = trips[trips["route_id"].isin(rail_route_ids)].copy()
        if weekday_services:
            rail_trips_wd = rail_trips[rail_trips["service_id"].isin(weekday_services)]
            if len(rail_trips_wd) > 100:
                rail_trips = rail_trips_wd

        rail_trip_ids = set(rail_trips["trip_id"].unique())
        print(f"   -> Found {len(rail_route_ids)} Rail/Metro/Light-Rail routes and {len(rail_trip_ids):,} trips.")

        # 2. Map platform stop_ids to parent_station so all platforms group into 1 station
        stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
        stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
        stops["parent_id"] = stops["parent_station"].fillna("").str.strip()
        stops["station_id"] = np.where(stops["parent_id"] != "", stops["parent_id"], stops["stop_id"])
        stop_to_station = dict(zip(stops["stop_id"], stops["station_id"]))

        # Build master station lookup table
        station_meta = (
            stops.drop_duplicates(subset=["station_id"])
            .set_index("station_id")[["stop_name", "stop_lat", "stop_lon"]]
            .to_dict(orient="index")
        )
        # Prefer parent station row names/coords where location_type == '1'
        parent_rows = stops[stops["location_type"] == "1"]
        for _, r in parent_rows.iterrows():
            station_meta[r["stop_id"]] = {
                "stop_name": r["stop_name"],
                "stop_lat": r["stop_lat"],
                "stop_lon": r["stop_lon"]
            }

        # 3. Stream stop_times.txt in chunks for those rail_trip_ids
        print("   -> Streaming stop_times.txt for morning peak timetable (06:30 - 09:30)...")
        kept_chunks = []
        use_cols = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
        for chunk in pd.read_csv(z.open("stop_times.txt"), usecols=use_cols, dtype=str, chunksize=400000):
            sub_chunk = chunk[chunk["trip_id"].isin(rail_trip_ids)].copy()
            if not sub_chunk.empty:
                kept_chunks.append(sub_chunk)

        st_df = pd.concat(kept_chunks, ignore_index=True)

    # Convert HH:MM:SS to minutes from midnight
    def parse_gtfs_mins(s):
        parts = s.str.split(":", expand=True)
        return pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")

    st_df["arr_min"] = parse_gtfs_mins(st_df["arrival_time"])
    st_df["dep_min"] = parse_gtfs_mins(st_df["departure_time"])
    st_df["seq"]     = pd.to_numeric(st_df["stop_sequence"], errors="coerce")
    st_df["station_id"] = st_df["stop_id"].map(stop_to_station)
    st_df = st_df.dropna(subset=["arr_min", "dep_min", "seq", "station_id"]).sort_values(["trip_id", "seq"])

    # Filter to stops active in Morning Peak (06:30 to 09:30 -> 390 to 570 mins)
    peak_df = st_df[st_df["dep_min"].between(390, 570) | st_df["arr_min"].between(390, 570)].copy()

    # Identify CBD Core Stations (Central, Town Hall, Wynyard, Martin Place, Gadigal, Barangaroo, Circular Quay)
    cbd_station_ids = set()
    for sid, info in station_meta.items():
        lat, lon = info["stop_lat"], info["stop_lon"]
        name = str(info["stop_name"]).upper()
        if pd.notna(lat) and (-33.892 <= lat <= -33.855) and (151.198 <= lon <= 151.220):
            if any(k in name for k in ["CENTRAL", "TOWN HALL", "WYNYARD", "MARTIN PLACE", "GADIGAL", "BARANGAROO", "CIRCULAR QUAY"]):
                cbd_station_ids.add(sid)

    print(f"   -> Identified {len(cbd_station_ids)} CBD Core station nodes.")

    # Build direct single-seat travel times between station pairs on peak trips
    pair_times = {}
    station_trip_counts = {}

    for tid, grp in peak_df.groupby("trip_id", sort=False):
        sids = grp["station_id"].values
        deps = grp["dep_min"].values
        arrs = grp["arr_min"].values
        n = len(sids)
        for i in range(n):
            u = sids[i]
            station_trip_counts[u] = station_trip_counts.get(u, 0) + 1
            for j in range(i + 1, n):
                v = sids[j]
                dt = arrs[j] - deps[i]
                if 0 < dt <= 180 and u != v:
                    pair_times.setdefault((u, v), []).append(dt)

    # Median single-seat travel time for each connected pair (u -> v)
    edge_median = {pair: float(np.median(times)) for pair, times in pair_times.items()}

    # Pass 1: Direct single-seat time to CBD
    best_cbd_time = {sid: 0.0 for sid in cbd_station_ids}
    all_active_stations = set(station_trip_counts.keys()) | cbd_station_ids

    for (u, v), med_t in edge_median.items():
        if v in cbd_station_ids:
            if u not in best_cbd_time or med_t < best_cbd_time[u]:
                best_cbd_time[u] = med_t

    # Pass 2 & 3: Allow 1 or 2 transfers (4-minute transfer penalty) for Metro/Light-Rail/Branch lines
    TRANSFER_PENALTY = 4.0
    for _ in range(2):
        updated = best_cbd_time.copy()
        for (u, v), med_t in edge_median.items():
            if v in best_cbd_time:
                cand = med_t + TRANSFER_PENALTY + best_cbd_time[v]
                if u not in updated or cand < updated[u]:
                    updated[u] = cand
        best_cbd_time = updated

    # Assemble active Greater Sydney rail/metro/light-rail station GeoDataFrame
    station_rows = []
    for sid in all_active_stations:
        info = station_meta.get(sid)
        if not info or pd.isna(info["stop_lat"]) or pd.isna(info["stop_lon"]):
            continue
        clean_st_name = (
            str(info["stop_name"])
            .replace("Station", "")
            .replace("Light Rail", "")
            .split(",")[0]
            .strip()
        )
        station_rows.append({
            "station_id": sid,
            "station_name": clean_st_name,
            "lat": info["stop_lat"],
            "lon": info["stop_lon"],
            "cbd_train_mins": round(best_cbd_time.get(sid, np.nan), 1)
        })

    df_stations = pd.DataFrame(station_rows)
    # Keep only stations with valid CBD travel times within Greater Sydney bounding box
    df_stations = df_stations[
        df_stations["lat"].between(-34.35, -33.25) &
        df_stations["lon"].between(150.35, 151.55) &
        df_stations["cbd_train_mins"].notna()
    ].copy()

    gdf_stations = gpd.GeoDataFrame(
        df_stations,
        geometry=gpd.points_from_xy(df_stations["lon"], df_stations["lat"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:7856")  # Meter-accurate MGA2020 Zone 56 projection

    # Calculate Suburb Representative Point distances in meters (EPSG:7856)
    syd_pts = syd_sal[["SAL_CODE21", "suburb_clean", "geometry"]].copy().to_crs("EPSG:7856")
    syd_pts["geometry"] = syd_pts.geometry.representative_point()

    # Martin Place (Sydney CBD) & Sydney Coastline reference points in EPSG:7856
    cbd_pt = gpd.GeoSeries([Point(151.2100, -33.8678)], crs="EPSG:4326").to_crs("EPSG:7856").iloc[0]
    beach_coords = [
        (151.3275, -33.5986), (151.3029, -33.7176), (151.2874, -33.7969), # Palm Beach, Narrabeen, Manly
        (151.2767, -33.8915), (151.2573, -33.9208), (151.2547, -33.9472), # Bondi, Coogee, Maroubra
        (151.1549, -34.0554), (150.9231, -34.3056)                         # Cronulla, Thirroul
    ]
    beach_pts = gpd.GeoSeries([Point(xy) for xy in beach_coords], crs="EPSG:4326").to_crs("EPSG:7856")

    st_coords = np.column_stack([gdf_stations.geometry.x.values, gdf_stations.geometry.y.values])
    st_names  = gdf_stations["station_name"].values
    st_mins   = gdf_stations["cbd_train_mins"].values

    transport_rows = []
    for _, row in syd_pts.iterrows():
        pt = row.geometry
        dx = st_coords[:, 0] - pt.x
        dy = st_coords[:, 1] - pt.y
        dists_km = np.hypot(dx, dy) / 1000.0

        # Nearest station
        min_idx = int(np.argmin(dists_km))
        nearest_dist_km = round(float(dists_km[min_idx]), 2)
        nearest_name    = st_names[min_idx]

        # Best total morning commute to CBD (checking stations within 5km or nearest)
        # Access time = 8 mins per km (local bus / bike / park-and-ride), minimum 3 mins
        access_penalties = np.maximum(3.0, dists_km * 8.0)
        total_commutes   = st_mins + access_penalties
        best_commute_min = round(float(np.min(total_commutes)), 0)

        # Straight-line distance to CBD & nearest coastal beach
        cbd_km   = round(pt.distance(cbd_pt) / 1000.0, 1)
        beach_km = round(float(beach_pts.distance(pt).min()) / 1000.0, 1)

        transport_rows.append({
            "SAL_CODE21": row["SAL_CODE21"],
            "nearest_station": nearest_name,
            "station_dist_km": nearest_dist_km,
            "cbd_commute_mins": best_commute_min,
            "cbd_dist_km": cbd_km,
            "beach_dist_km": beach_km
        })

    df_trans_out = pd.DataFrame(transport_rows)
    df_trans_out.to_csv(transport_cache_path, index=False)
    print(f"[✓] Saved transport & distance metrics for {len(df_trans_out)} suburbs to {transport_cache_path}")
else:
    print(f"[✓] Loaded cached transport metrics from {transport_cache_path}")

# ==============================================================================
# 3. STITCH MASTER GEOJSON & COMPUTE DERIVED LIVABILITY METRICS
# ==============================================================================
print("\n3. Stitching all datasets into master sydney_suburbs.geojson...")
rent_cache_path  = "cache/rent_by_postcode.csv"
sales_cache_path = "cache/sales_by_suburb.csv"

df_rent  = pd.read_csv(rent_cache_path, dtype={"postcode": str})
df_sales = pd.read_csv(sales_cache_path)
df_crime = pd.read_csv(crime_cache_path)
df_trans = pd.read_csv(transport_cache_path, dtype={"SAL_CODE21": str})

syd_sal["SAL_CODE21"] = syd_sal["SAL_CODE21"].astype(str)
syd_sal["postcode"]   = syd_sal["postcode"].astype(str).str.replace(r"\.0$", "", regex=True)

master = (
    syd_sal
    .merge(df_rent, on="postcode", how="left")
    .merge(df_sales, on="suburb_clean", how="left")
    .merge(df_crime, on="suburb_clean", how="left")
    .merge(df_trans, on="SAL_CODE21", how="left")
)

# Fall back to ABS Census weekly rent (scaled to 2025 levels) only if postcode bond sample is missing
master["rent_all_med"] = master["rent_all_med"].fillna((master["census_rent_weekly"] * 1.28).round(0))

# Calculate Per-Capita Crime Rates (per 1,000 residents, using floor of 500 pop to avoid industrial-zone division spikes)
eff_pop = master["population"].clip(lower=500)
master["crime_total_rate"]    = ((master["total_incidents"].fillna(0) / eff_pop) * 1000.0).round(1)
master["crime_violent_rate"]  = ((master["violent_incidents"].fillna(0) / eff_pop) * 1000.0).round(1)
master["crime_property_rate"] = ((master["property_incidents"].fillna(0) / eff_pop) * 1000.0).round(1)

# Calculate Gross Rental Yields (%) & Local Affordability Ratios
master["yield_all_pct"]   = ((master["rent_all_med"] * 52.0 / master["buy_all_med"]) * 100.0).round(2)
master["yield_house_pct"] = ((master["rent_house_med"] * 52.0 / master["buy_house_med"]) * 100.0).round(2)
master["yield_unit_pct"]  = ((master["rent_unit_med"] * 52.0 / master["buy_unit_med"]) * 100.0).round(2)

annual_inc = (master["hhd_income_weekly"].replace(0, np.nan) * 52.0)
master["price_to_income"] = (master["buy_all_med"] / annual_inc).round(1)

# Simplify geometry (~25m tolerance) so the web map renders 900+ polygons at 60fps
master["geometry"] = master["geometry"].simplify(tolerance=0.00025, preserve_topology=True)

output_geojson = "sydney_suburbs.geojson"
master.to_file(output_geojson, driver="GeoJSON")
size_mb = os.path.getsize(output_geojson) / (1024 * 1024)
print(f"[✓] SUCCESS! Exported {len(master)} suburbs to {output_geojson} ({size_mb:.2f} MB)")
print("    Sample columns:", list(master.columns[:15]))