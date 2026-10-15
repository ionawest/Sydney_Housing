import zipfile
import os
import pandas as pd
import geopandas as gpd

print("=== 1. CHECKING MESH BLOCKS (LAND USE & SYDNEY BOUNDARY) ===")
if os.path.exists("ABS_mesh_blocks_sydney.gpkg"):
    mb_gpkg = gpd.read_file("ABS_mesh_blocks_sydney.gpkg", rows=5)
    bounds = gpd.read_file("ABS_mesh_blocks_sydney.gpkg").total_bounds
    print(f"[✓] Found ABS_mesh_blocks_sydney.gpkg | Columns: {list(mb_gpkg.columns)}")
    print(f"    Bounding Box (minx, miny, maxx, maxy): {bounds}")
if os.path.exists("MB_2021_AUST_GDA2020_SHP.zip"):
    mb_zip = gpd.read_file("MB_2021_AUST_GDA2020_SHP.zip", rows=5)
    print(f"[✓] Found MB_2021_AUST_GDA2020_SHP.zip | Columns: {list(mb_zip.columns)}")

print("\n=== 2. CHECKING SUBURB (SAL) & POSTCODE (POA) SHAPEFILES ===")
sal_sample = gpd.read_file("SAL_2021_AUST_GDA2020_SHP.zip", rows=5)
poa_sample = gpd.read_file("POA_2021_AUST_GDA2020_SHP.zip", rows=5)
print("SAL Columns:", list(sal_sample.columns))
print("POA Columns:", list(poa_sample.columns))

print("\n=== 3. CHECKING ABS CENSUS DATAPACK (2021_GCP_SAL) ===")
with zipfile.ZipFile("2021_GCP_SAL_for_NSW_short-header.zip", 'r') as z:
    g01_g02 = [f for f in z.namelist() if ("G01" in f or "G02" in f) and f.endswith(".csv")]
    print("Found Census Tables:", g01_g02)
    for f in g01_g02:
        df_c = pd.read_csv(z.open(f), nrows=2)
        print(f"  -> {os.path.basename(f)} first 8 cols:", list(df_c.columns[:8]))

print("\n=== 4. CHECKING NSW FAIR TRADING RENTAL BONDS (2025 XLSX) ===")
xl = pd.ExcelFile("rentalbond_lodgements_year_2025.xlsx")
print("Excel Sheet Names:", xl.sheet_names)
df_rent = pd.read_excel("rentalbond_lodgements_year_2025.xlsx", sheet_name=0, nrows=10)
print(df_rent.head(6))

print("\n=== 5. CHECKING BOCSAR CRIME (SuburbData.zip) ===")
with zipfile.ZipFile("SuburbData.zip", 'r') as z:
    print("Files inside SuburbData.zip:", z.namelist())
    csv_files = [f for f in z.namelist() if f.endswith(".csv")]
    if csv_files:
        df_crime = pd.read_csv(z.open(csv_files[0]), nrows=5)
        print("Crime Columns (first 10):", list(df_crime.columns[:10]))
        print(df_crime.head(2))

print("\n=== 6. CHECKING VALUER GENERAL PROPERTY SALES (2025.zip) ===")
with zipfile.ZipFile("2025.zip", 'r') as z:
    names = z.namelist()
    print(f"Total items in 2025.zip: {len(names)} | First 5 items: {names[:5]}")