# Jeddah Traffic Data Dashboard

An interactive Streamlit dashboard that explores and models a full year of daily traffic indicators for **Jeddah, Saudi Arabia (2024)**, with built-in Decision Tree classification and K-Means clustering.

![dashboard placeholder](docs/dashboard.png) <!-- TODO: Taha to add real Streamlit screenshot/GIF -->

---

## Problem

Urban traffic authorities need to understand what drives day-to-day changes in road conditions — volume, speed, weather, road works, accidents — and to distinguish "normal" days from unusually severe ones. This project provides a small, self-contained analytics dashboard that:

- Visualizes each traffic indicator as a time series across 2024.
- Classifies each day as **Severe** or **Not Severe** relative to a chosen metric, and surfaces the human-readable rules a Decision Tree learns to separate the two.
- Clusters days by overall traffic pattern with K-Means, to reveal recurring "types" of day (e.g. quiet weekends vs. congested Hajj-period days).

It is a compact, reproducible demonstration of an end-to-end data + ML workflow on a domain-calibrated dataset — not a production traffic-prediction system.

---

## Data

> **This dataset is SYNTHETIC.** It is **not** raw sensor or government data. Every row is generated programmatically by [`generate_dataset.py`](generate_dataset.py) using `numpy`'s random generator (`numpy.random.seed(2024)`), with the distributions, seasonality, and event effects **calibrated to official 2024 Saudi statistics**. Use it for demonstration, teaching, and prototyping — not as a factual record of real Jeddah traffic.

`jeddah_traffic_dataset.csv` — 366 daily records (01/01/2024 – 31/12/2024).

### Data dictionary

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

### How the synthetic data is built

`generate_dataset.py` models one record per day and applies:

- **Seasonality** — a per-month multiplier reflecting Jeddah's coastal climate (milder summer dip than inland cities).
- **Weekly pattern** — reduced volume on the Saudi weekend (Friday & Saturday).
- **Calendar events** — Ramadan (11 Mar – 9 Apr) lowers daytime traffic but raises accident rates; Eid Al-Fitr and Eid Al-Adha reduce volume; the Hajj window (14–19 Jun) drives a pilgrim-traffic and accident surge; National Day (23 Sep) lowers volume.
- **Physically-motivated relationships** — `AverageSpeed` falls as `TrafficVolume` rises; `CongestionIndex` is built up from volume, road works, weather, and speed. Accidents are drawn from Poisson distributions whose rates come from the calibrated Jeddah share.
- **Noise** — Gaussian/normal noise on each field, with values clipped to the ranges above.

The script prints a validation block comparing generated accident totals against the expected calibrated figures, then writes the CSV.

### Calibration & sources

Figures are calibrated to official 2024 national statistics: **17,231 serious accidents nationally · Makkah Region = 20.1 % of driving licenses · Jeddah ≈ 60 % of region (≈ 12 % national share) · 60.3 % intracity / 39.7 % outside cities.**

- [Saudi Arabia Road Transport Statistics 2024 — GASTAT (General Authority for Statistics)](https://www.stats.gov.sa/documents/20117/2435281/Road+Transpot+Statistics+2024_EN.pdf)
- [KAPSARC Traffic Accidents Portal](https://datasource.kapsarc.org/explore/assets/saudi-arabia-traffic-accidents-and-casualties-injured-dead-2008/)

---

## ML / Methods

Both models run interactively inside the Streamlit app ([`streamlit_jeddah_ML.py`](streamlit_jeddah_ML.py)).

### Decision Tree classification

- **Target (label):** derived on the fly from the numeric column you pick in the dropdown. The app computes that column's **mean** as a threshold and labels each day `"Severe"` if the value is above the mean, otherwise `"Not Severe"`. The label is therefore a self-defined "above-average" flag for the chosen metric, not an externally provided ground truth.
- **Features:** all *other* numeric columns (every numeric column except the one used to build the label).
- **Model:** `sklearn.tree.DecisionTreeClassifier(random_state=42, max_depth=5)`.
- **Output:** the app fits the tree and predicts on the same data, then renders the full tree diagram (`plot_tree`), the text IF-THEN rules (`export_text`), and a table of actual vs. predicted labels.

> **No accuracy or other performance metric is computed.** The code does not perform a train/test split and does not call any scoring function (e.g. `accuracy_score`, cross-validation) — it fits and predicts on the full dataset for interpretability/visualization. Any accuracy figure would be fabricated, so none is reported here. Adding a held-out split and reporting accuracy would be a natural next improvement.

### K-Means clustering

- **Features:** all numeric columns, standardized with `sklearn.preprocessing.StandardScaler`.
- **Model:** `sklearn.cluster.KMeans(n_clusters=k, random_state=42)`, where `k` is chosen with a slider (2–10, default 3).
- **Output:** each day is assigned a cluster label; results are shown as a table and a 2-D scatter plot of the first two numeric columns (`TrafficVolume` vs `AverageSpeed`) colored by cluster.

> **No silhouette score or other cluster-quality metric is computed** by the code, so none is reported. Adding a silhouette/elbow analysis to guide the choice of `k` would be a natural next improvement.

---

## Tech Stack

Derived directly from the imports in the two scripts:

- **Streamlit** — dashboard UI (`streamlit`)
- **pandas** — data loading and manipulation (`pandas`)
- **scikit-learn** — Decision Tree, K-Means, StandardScaler (`scikit-learn`)
- **Plotly Express** — interactive charts (`plotly`)
- **Matplotlib** — Decision Tree diagram rendering (`matplotlib`)
- **NumPy** — used by the dataset generator (`numpy`; also a transitive dependency of pandas/scikit-learn)

Python 3.9+ recommended.

---

## Running Locally

> **Note:** there is currently **no `requirements.txt`** in the repo. Install the dependencies listed below directly, or create the file yourself from this list.

```bash
# 1. Clone the repository
git clone https://github.com/taha-yasin-saad/jeddah-traffic-dataset.git
cd jeddah-traffic-dataset

# 2. (Optional) create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies (no requirements.txt yet — install these explicitly)
pip install streamlit pandas scikit-learn plotly matplotlib

# 4. Run the dashboard
streamlit run streamlit_jeddah_ML.py
```

Then open **http://localhost:8501** in your browser.

### Regenerate the dataset (optional)

```bash
python generate_dataset.py
```

This recreates `jeddah_traffic_dataset.csv` deterministically (`numpy.random.seed(2024)`), so the data is fully reproducible.

---

## Project structure

```
jeddah-traffic-dataset/
├── jeddah_traffic_dataset.csv   # Synthetic daily traffic data for Jeddah 2024 (366 rows)
├── generate_dataset.py          # Calibrated synthetic-data generator
├── streamlit_jeddah_ML.py       # Streamlit dashboard (Decision Tree + K-Means)
└── README.md
```

---

## Screenshot

![dashboard placeholder](docs/dashboard.png) <!-- TODO: Taha to add real Streamlit screenshot/GIF -->
