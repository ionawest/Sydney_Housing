import os
import io
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd

os.makedirs("cache", exist_ok=True)

# ==============================================================================
# 1. BASE GREATER SYDNEY SUBURBS & MESH BLOCK LAND USE
# ==============================================================================
base_gpkg_path = "cache/sydney_suburbs_base.gpkg"

if not os.path.exists(base_gpkg_path):
    print("1. Building Greater Sydney Suburb base & Land Use from Mesh Blocks...")
    mb = gpd.read_file("ABS_mesh_blocks_sydney.gpkg")
    syd_mb = mb[mb["GCC_NAME21"] == "Greater Sydney"].copy()

    # Load Suburbs (SAL) and Postcodes (POA)
    sal = gpd.read_file("SAL_2021_AUST_GDA2020_SHP.zip")
    sal = sal[sal["STE_NAME21"] == "New South Wales"].copy()
    poa = gpd.read_file("POA_2021_AUST_GDA2020_SHP.zip")

    # Use Mesh Block representative points to link Mesh Blocks -> Suburbs
    syd_mb_pts = syd_mb.copy()
    syd_mb_pts["geometry"] = syd_mb_pts.geometry.representative_point()
    mb_sal = gpd.sjoin(
        syd_mb_pts[["MB_CODE21", "MB_CAT21", "AREASQKM21", "geometry"]],
        sal[["SAL_CODE21", "SAL_NAME21", "geometry"]],
        how="inner",
        predicate="within"
    )

    # Calculate Land Use Breakdown per Suburb (Parkland %, Residential %, Commercial %)
    lu_pivot = mb_sal.pivot_table(
        index="SAL_CODE21",
        columns="MB_CAT21",
        values="AREASQKM21",
        aggfunc="sum",
        fill_value=0.0
    )
    lu_totals = lu_pivot.sum(axis=1)

    # Keep only suburbs that have mesh blocks inside Greater Sydney
    syd_sal = sal[sal["SAL_CODE21"].isin(lu_pivot.index)].copy()

    for cat, col_name in [
        ("Parkland", "parkland_pct"),
        ("Residential", "residential_pct"),
        ("Commercial", "commercial_pct"),
        ("Industrial", "industrial_pct")
    ]:
        if cat in lu_pivot.columns:
            pct_series = (lu_pivot[cat] / lu_totals * 100).round(1)
            syd_sal[col_name] = syd_sal["SAL_CODE21"].map(pct_series).fillna(0.0)
        else:
            syd_sal[col_name] = 0.0

    # Clean Suburb Name (strip " (NSW)" or state disambiguators if present for display/matching)
    syd_sal["suburb_clean"] = (
        syd_sal["SAL_NAME21"]
        .str.replace(r"\s*\(.*\)$", "", regex=True)
        .str.strip()
        .str.upper()
    )

    # Map each Suburb to its Primary Postcode via representative point spatial join
    sal_pts = syd_sal[["SAL_CODE21", "geometry"]].copy()
    sal_pts["geometry"] = sal_pts.geometry.representative_point()
    sal_poa = gpd.sjoin(sal_pts, poa[["POA_CODE21", "geometry"]], how="left", predicate="within")
    sal_poa = sal_poa.drop_duplicates(subset=["SAL_CODE21"])
    syd_sal["postcode"] = syd_sal["SAL_CODE21"].map(sal_poa.set_index("SAL_CODE21")["POA_CODE21"])

    # Merge ABS Census G01 (Population) & G02 (Income, Age, Mortgage, Rent)
    print("   -> Merging ABS 2021 Census Demographics...")
    with zipfile.ZipFile("2021_GCP_SAL_for_NSW_short-header.zip", "r") as z:
        g01_path = [f for f in z.namelist() if "G01" in f and f.endswith(".csv")][0]
        g02_path = [f for f in z.namelist() if "G02" in f and f.endswith(".csv")][0]
        df_g01 = pd.read_csv(z.open(g01_path), usecols=["SAL_CODE_2021", "Tot_P_P"])
        df_g02 = pd.read_csv(
            z.open(g02_path),
            usecols=[
                "SAL_CODE_2021", "Median_age_persons", "Median_mortgage_repay_monthly",
                "Median_rent_weekly", "Median_tot_hhd_inc_weekly", "Average_num_psns_per_bedroom"
            ]
        )

    df_census = pd.merge(df_g01, df_g02, on="SAL_CODE_2021", how="left")
    df_census["SAL_CODE21"] = df_census["SAL_CODE_2021"].astype(str).str.replace("SAL", "", regex=False)
    df_census = df_census.rename(columns={
        "Tot_P_P": "population",
        "Median_age_persons": "median_age",
        "Median_mortgage_repay_monthly": "census_mortgage_monthly",
        "Median_rent_weekly": "census_rent_weekly",
        "Median_tot_hhd_inc_weekly": "hhd_income_weekly",
        "Average_num_psns_per_bedroom": "persons_per_bedroom"
    })

    syd_sal = syd_sal.merge(
        df_census.drop(columns=["SAL_CODE_2021"]),
        on="SAL_CODE21",
        how="left"
    )
    syd_sal["pop_density_sqkm"] = (syd_sal["population"] / syd_sal["AREASQKM21"].replace(0, np.nan)).round(1)

    # Reproject to WGS84 (EPSG:4326) for Leaflet and save to cache
    syd_sal = syd_sal.to_crs("EPSG:4326")
    keep_cols = [
        "SAL_CODE21", "SAL_NAME21", "suburb_clean", "postcode", "AREASQKM21",
        "population", "pop_density_sqkm", "median_age", "hhd_income_weekly",
        "census_rent_weekly", "census_mortgage_monthly", "persons_per_bedroom",
        "parkland_pct", "residential_pct", "commercial_pct", "industrial_pct", "geometry"
    ]
    syd_sal = syd_sal[keep_cols]
    syd_sal.to_file(base_gpkg_path, driver="GPKG")
    print(f"[✓] Saved {len(syd_sal)} Greater Sydney suburbs to {base_gpkg_path}")
else:
    print(f"[✓] Loaded cached base suburbs from {base_gpkg_path}")
    syd_sal = gpd.read_file(base_gpkg_path)

# ==============================================================================
# 2. NSW FAIR TRADING RENTAL BONDS (2025 XLSX)
# ==============================================================================
rent_cache_path = "cache/rent_by_postcode.csv"

if not os.path.exists(rent_cache_path):
    print("\n2. Processing 2025 NSW Fair Trading Rental Bonds...")
    df_rent = pd.read_excel("rentalbond_lodgements_year_2025.xlsx", sheet_name=0, header=2)
    df_rent.columns = [str(c).strip() for c in df_rent.columns]

    # Clean Postcode, Weekly Rent, Dwelling Type, and Bedrooms
    df_rent["Postcode"] = pd.to_numeric(df_rent["Postcode"], errors="coerce")
    df_rent["Weekly Rent"] = pd.to_numeric(df_rent["Weekly Rent"], errors="coerce")
    df_rent["Bedrooms_num"] = pd.to_numeric(df_rent["Bedrooms"], errors="coerce")
    df_rent["Dwelling Type"] = df_rent["Dwelling Type"].astype(str).str.strip().str.upper()

    # Filter to valid rents ($100 to $15,000/wk) and Sydney postcodes
    syd_postcodes = set(syd_sal["postcode"].dropna().astype(int).unique())
    df_rent = df_rent[
        df_rent["Postcode"].isin(syd_postcodes) &
        df_rent["Weekly Rent"].between(100, 15000)
    ].copy()
    df_rent["postcode"] = df_rent["Postcode"].astype(int).astype(str)

    # Map Dwelling Type: F=Unit, H=House, T=Townhouse
    dtype_map = {"F": "unit", "H": "house", "T": "townhouse"}
    df_rent["dtype"] = df_rent["Dwelling Type"].map(dtype_map)

    # Map Bedrooms: 0/1 -> 1bed, 2 -> 2bed, 3 -> 3bed, >=4 -> 4bed
    def bin_beds(b):
        if pd.isna(b): return None
        b = int(b)
        if b <= 1: return "1bed"
        elif b == 2: return "2bed"
        elif b == 3: return "3bed"
        else: return "4bed"

    df_rent["bed_cat"] = df_rent["Bedrooms_num"].apply(bin_beds)

    # Aggregate medians and bond counts
    MIN_BONDS = 5
    rent_records = {}

    for pc, grp in df_rent.groupby("postcode"):
        row = {"postcode": pc, "rent_all_count": len(grp)}
        row["rent_all_med"] = round(grp["Weekly Rent"].median(), 0) if len(grp) >= MIN_BONDS else np.nan

        # By Dwelling Type (All bedrooms)
        for dt in ["house", "unit", "townhouse"]:
            sub = grp[grp["dtype"] == dt]
            row[f"rent_{dt}_count"] = len(sub)
            row[f"rent_{dt}_med"] = round(sub["Weekly Rent"].median(), 0) if len(sub) >= MIN_BONDS else np.nan

        # By Bedroom Count (All dwelling types)
        for bc in ["1bed", "2bed", "3bed", "4bed"]:
            sub = grp[grp["bed_cat"] == bc]
            row[f"rent_all_{bc}_med"] = round(sub["Weekly Rent"].median(), 0) if len(sub) >= MIN_BONDS else np.nan

        # Cross-filter: House & Unit by Bedroom Count
        for dt in ["house", "unit"]:
            for bc in ["1bed", "2bed", "3bed", "4bed"]:
                sub = grp[(grp["dtype"] == dt) & (grp["bed_cat"] == bc)]
                row[f"rent_{dt}_{bc}_med"] = round(sub["Weekly Rent"].median(), 0) if len(sub) >= MIN_BONDS else np.nan

        rent_records[pc] = row

    df_rent_out = pd.DataFrame.from_dict(rent_records, orient="index").reset_index(drop=True)
    df_rent_out.to_csv(rent_cache_path, index=False)
    print(f"[✓] Saved rental bond aggregations for {len(df_rent_out)} Sydney postcodes to {rent_cache_path}")
else:
    print(f"[✓] Loaded cached rental metrics from {rent_cache_path}")

# ==============================================================================
# 3. NSW VALUER GENERAL PROPERTY SALES (2025.zip NESTED STREAMING)
# ==============================================================================
sales_cache_path = "cache/sales_by_suburb.csv"

if not os.path.exists(sales_cache_path):
    print("\n3. Streaming 2025 NSW Valuer General Property Sales (.DAT files in memory)...")
    syd_suburb_names = set(syd_sal["suburb_clean"].unique())
    sales_rows = []

    with zipfile.ZipFile("2025.zip", "r") as outer_zip:
        weekly_zips = [f for f in outer_zip.namelist() if f.endswith(".zip")]
        for idx, wz_name in enumerate(weekly_zips):
            wz_bytes = io.BytesIO(outer_zip.read(wz_name))
            with zipfile.ZipFile(wz_bytes, "r") as inner_zip:
                for dat_name in inner_zip.namelist():
                    if not dat_name.upper().endswith(".DAT"):
                        continue
                    raw_text = inner_zip.read(dat_name).decode("latin-1", errors="ignore")
                    for line in raw_text.splitlines():
                        if not line.startswith("B;"):
                            continue
                        parts = line.split(";")
                        if len(parts) < 20:
                            continue

                        suburb = parts[9].strip().upper()
                        if suburb not in syd_suburb_names:
                            continue

                        nature = parts[17].strip().upper()
                        zoning = parts[16].strip().upper()
                        # Keep Residential sales ('R' nature or Residential 'R*' zoning)
                        if nature != "R" and not zoning.startswith("R"):
                            continue

                        try:
                            price = float(parts[15].strip())
                        except ValueError:
                            continue
                        if price < 100000 or price > 50000000:
                            continue

                        # Determine Property Type: Unit/Strata vs House (Non-Strata)
                        unit_num = parts[6].strip()
                        strata_lot = parts[19].strip()
                        prop_type = "unit" if (unit_num != "" or strata_lot != "") else "house"

                        # Parse Land Area in m2
                        area_val = np.nan
                        try:
                            a = float(parts[11].strip())
                            a_type = parts[12].strip().upper()
                            if a > 0:
                                area_val = a * 10000.0 if a_type == "H" else a
                        except ValueError:
                            pass

                        # Deduplicate key: District + Property_ID + Sale_Counter
                        sale_id = f"{parts[1]}_{parts[2]}_{parts[3]}"
                        sales_rows.append((sale_id, suburb, prop_type, price, area_val))

    df_sales = pd.DataFrame(sales_rows, columns=["sale_id", "suburb_clean", "prop_type", "price", "area_m2"])
    df_sales = df_sales.drop_duplicates(subset=["sale_id"])
    print(f"   -> Extracted {len(df_sales):,} unique residential sales across Greater Sydney!")

    MIN_SALES = 5
    sales_records = {}
    for sub_name, grp in df_sales.groupby("suburb_clean"):
        row = {"suburb_clean": sub_name, "buy_all_count": len(grp)}
        row["buy_all_med"] = round(grp["price"].median(), 0) if len(grp) >= MIN_SALES else np.nan

        for pt in ["house", "unit"]:
            sub = grp[grp["prop_type"] == pt]
            row[f"buy_{pt}_count"] = len(sub)
            row[f"buy_{pt}_med"] = round(sub["price"].median(), 0) if len(sub) >= MIN_SALES else np.nan

        # House Land Size Brackets (<400m2, 400-700m2, 700m2+) & Price per m2
        houses_with_area = grp[(grp["prop_type"] == "house") & (grp["area_m2"].between(80, 10000))].copy()
        if len(houses_with_area) >= MIN_SALES:
            ppm2 = houses_with_area["price"] / houses_with_area["area_m2"]
            row["house_price_per_m2"] = round(ppm2.median(), 0)
            row["house_median_area_m2"] = round(houses_with_area["area_m2"].median(), 0)
        else:
            row["house_price_per_m2"] = np.nan
            row["house_median_area_m2"] = np.nan

        h_small = houses_with_area[houses_with_area["area_m2"] < 400]
        h_med   = houses_with_area[houses_with_area["area_m2"].between(400, 700)]
        h_large = houses_with_area[houses_with_area["area_m2"] > 700]

        row["buy_house_small_med"] = round(h_small["price"].median(), 0) if len(h_small) >= 3 else np.nan
        row["buy_house_med_med"]   = round(h_med["price"].median(), 0) if len(h_med) >= 3 else np.nan
        row["buy_house_large_med"] = round(h_large["price"].median(), 0) if len(h_large) >= 3 else np.nan

        sales_records[sub_name] = row

    df_sales_out = pd.DataFrame.from_dict(sales_records, orient="index").reset_index(drop=True)
    df_sales_out.to_csv(sales_cache_path, index=False)
    print(f"[✓] Saved property sales metrics for {len(df_sales_out)} Sydney suburbs to {sales_cache_path}")
else:
    print(f"[✓] Loaded cached property sales metrics from {sales_cache_path}")

# ==============================================================================
# 4. QUICK CHECK OF TRANSPORT GTFS ZIP
# ==============================================================================
print("\n4. Checking for Transport GTFS archive in folder...")
gtfs_candidates = [f for f in os.listdir(".") if "gtfs" in f.lower() or "mdb" in f.lower() or "timetable" in f.lower()]
print("Found GTFS candidate files:", gtfs_candidates)
if gtfs_candidates:
    with zipfile.ZipFile(gtfs_candidates[0], "r") as z:
        print("Files inside GTFS zip:", z.namelist())
        if "stops.txt" in z.namelist():
            df_stops = pd.read_csv(z.open("stops.txt"), nrows=3)
            print("stops.txt columns:", list(df_stops.columns))
        if "routes.txt" in z.namelist():
            df_routes = pd.read_csv(z.open("routes.txt"), nrows=3)
            print("routes.txt columns:", list(df_routes.columns))