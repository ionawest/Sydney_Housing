# Sydney Suburb Livability, Finance & Couples Commute Engine

![Architecture](https://img.shields.io/badge/Architecture-Serverless_Spatial-38bdf8)
![Build](https://img.shields.io/badge/Build-Python_|_GeoPandas-22c55e)
![Frontend](https://img.shields.io/badge/Frontend-Leaflet.js_|_Chart.js-f59e0b)
![License](https://img.shields.io/badge/Data-Open_Source-c084fc)

A serverless, high-performance geospatial decision-support dashboard evaluating Greater Sydney suburbs across 50+ financial, transit, microclimate, socio-economic, and hazard metrics. 

By applying a "Hybrid Approach," heavy spatial mathematics and data cleansing are executed in a local Python ETL pipeline, exporting lightweight `.geojson` files to a client-side Leaflet web map. 

**[View Live Interactive Dashboard](https://ionawest.github.io/Sydney_Housing/)** 

---

## Key Features & Analytical Capabilities

* **Live Vector Formula Engine:** Users can write custom mathematical formulas (e.g., `(1.5 * parkland) + (1 - rent) - (0.5 * hazard)`) evaluated dynamically in `< 2ms` across all 926 polygons using Winsorized (P2–P98) or Percentile normalizations.
* **Dual-Workplace ("Couples") Commuter Router:** Utilizes an all-pairs Floyd-Warshall transit matrix across 272 TfNSW stations to instantly optimize commutes for two partners simultaneously (supporting multi-modal Public Transit vs. Peak Road profiles).
* **Real-Time "Rent vs. Buy" Cashflow Modeling:** Generates dynamic P&I mortgage layers and visualizes the monthly ownership-to-rent cost ratio based on live user interest rate and deposit inputs.
* **2D Value-Anomaly Scatter Plot:** A built-in Chart.js scatter plot instantly visualizes the "Bargain Frontier" (e.g., fast commute vs. low price) allowing users to click anomalies and fly directly to the suburb on the map.
* **Ambient-Adjusted Hazard & Crime Models:** Standard per-capita crime data is adjusted via ABS commercial foot-traffic density to prevent "denominator traps." Features precise environmental risk overlays including Open-Meteo ERA5 microclimate data, ANEF aircraft noise contours, and bushfire/flood basin risk scoring.

---

## Local Installation & Development

Because this application relies on asynchronous JavaScript `fetch()` requests to load the local GeoJSON data, opening `index.html` directly in a browser will be blocked by strict CORS security policies.

To run the dashboard locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ionawest/Sydney_Housing.git
   cd Sydney_Housing

2. **Start a local Python web server:**

    Run "python -m http.server 8000"

3. Navigate to http://localhost:8000 in Chrome, Firefox, or Safari.

---

Primary Data Sources:

Australian Bureau of Statistics (ABS): Digital Boundary Files (SAL/POA), 2021 Census DataPacks, Mesh Block Land Use.

NSW Government: Valuer General Bulk Property Sales Information, Fair Trading Rental Bonds.

BOCSAR: Recorded Criminal Incidents Open Data.

Transport for NSW (TfNSW): Complete GTFS Timetables and Shapes.

OpenStreetMap (Overpass API): Infrastructure mapping (Hospitals, Shopping Centres, Schools).

Open-Meteo: ERA5 Historical Climatology Archive and Digital Elevation Models.