# Jeddah Traffic Data Dashboard

An interactive machine learning dashboard for daily traffic data in **Jeddah, Saudi Arabia** (2024).  
Built with Streamlit. Includes Decision Tree classification and K-Means clustering.

---

## Dataset

`jeddah_traffic_dataset.csv` — 366 daily records (01/01/2024 – 31/12/2024)

| Column | Description | Range |
|---|---|---|
| `Date` | Date (DD/MM/YYYY) | 2024 |
| `TrafficVolume` | Daily vehicles at monitoring point | 1,500 – 24,000 |
| `AverageSpeed` | Average vehicle speed (km/h) | 18 – 110 |
| `AccidentsInsideCity` | Serious accidents inside city | 0 – 18 |
| `AccidentsOutsideCity` | Serious accidents outside city | 0 – 10 |
| `WeatherSeverity` | Weather impact (0 = clear, 5 = severe) | 0 – 5 |
| `RoadWorkLevel` | Active road works (0 = none, 3 = heavy) | 0 – 3 |
| `CongestionIndex` | Congestion level (10 = free flow, 100 = gridlock) | 10 – 100 |

**Data sources:**
- [Saudi Arabia Road Transport Statistics 2024 — GASTAT](https://www.stats.gov.sa/documents/20117/2435281/Road+Transpot+Statistics+2024_EN.pdf)
- [KAPSARC Traffic Accidents Portal](https://datasource.kapsarc.org/explore/assets/saudi-arabia-traffic-accidents-and-casualties-injured-dead-2008/)

Figures are calibrated to official 2024 national statistics:  
17,231 serious accidents nationally · Makkah Region = 20.1 % of driving licenses · Jeddah ≈ 60 % of region · 60.3 % intracity / 39.7 % outside cities.

---

## Requirements

- Python 3.9+
- pip packages: `streamlit`, `pandas`, `plotly`, `scikit-learn`, `matplotlib`

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/taha-yasin-saad/jeddah-traffic-dataset.git
cd jeddah-traffic-dataset

# 2. (Optional) create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install streamlit pandas plotly scikit-learn matplotlib
```

---

## Run the dashboard

```bash
streamlit run streamlit_jeddah_ML.py
```

Then open your browser at **http://localhost:8501**

---

## Features

### Interactive chart
Select any numeric column from the dropdown to plot it as a time series over the full year 2024.

### Decision Tree Classification
Click **Run Decision Tree Classification** to:
- Label each day as **Severe** or **Not Severe** based on whether the selected metric exceeds its yearly average
- Train a Decision Tree on the remaining columns
- View the full tree diagram and the human-readable IF-THEN rules

### K-Means Clustering
Use the slider to choose 2–10 clusters, then click **Run K-Means Clustering** to:
- Group days by traffic pattern similarity
- View the clustered scatter plot (Traffic Volume vs Average Speed)
- Export the cluster-labelled table

---

## Regenerate the dataset

```bash
python generate_dataset.py
```

This recreates `jeddah_traffic_dataset.csv` with the same seed (`numpy.random.seed(2024)`), so results are fully reproducible.

---

## Project structure

```
jeddah-traffic-dataset/
├── jeddah_traffic_dataset.csv   # Daily traffic data for Jeddah 2024
├── streamlit_jeddah_ML.py       # Streamlit dashboard app
├── generate_dataset.py          # Dataset generation script
└── README.md
```
