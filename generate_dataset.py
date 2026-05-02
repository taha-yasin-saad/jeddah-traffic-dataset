"""
Generates a realistic daily traffic dataset for Jeddah (2024)
based on official Saudi Arabia statistics:

Sources:
- Saudi Arabia Road Transport Statistics 2024 (General Authority for Statistics / GASTAT):
    - 17,231 serious accidents nationwide
    - 4,282 fatalities, 24,077 injuries
    - 60.3 % intracity / 39.7 % outside cities
    - 15.8 million registered vehicles
- KAPSARC portal (regional data 2016-2019):
    - Makkah Region (contains Jeddah) holds 20.1 % of national driving licenses
    - Jeddah ≈ 60 % of Makkah Region's urban population → ~12 % national share

City context:
    Jeddah is Saudi Arabia's 2nd-largest city (pop. ~4.7 million), the main
    commercial and sea-port hub on the Red Sea coast.  Traffic patterns differ
    from inland cities: coastal humidity, pilgrim/Hajj traffic, port trucks,
    and King Abdulaziz International Airport all affect daily flow.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(2024)

# ─── Real 2024 Saudi statistics ──────────────────────────────────────────────
NATIONAL_SERIOUS_ACCIDENTS_2024 = 17_231
NATIONAL_FATALITIES_2024 = 4_282
NATIONAL_INJURIES_2024 = 24_077

MAKKAH_REGION_LICENSE_SHARE = 0.201   # Makkah Region (includes Jeddah)
JEDDAH_CITY_FRACTION = 0.60           # Jeddah ≈ 60 % of Makkah Region pop.
JEDDAH_LICENSE_SHARE = MAKKAH_REGION_LICENSE_SHARE * JEDDAH_CITY_FRACTION  # ~12 %

INTRACITY_RATIO = 0.603               # 60.3 % accidents inside cities (2024 national)
OUTSIDE_RATIO = 0.397

# Jeddah estimates (serious accidents, 2024)
JEDDAH_SERIOUS_PER_YEAR = NATIONAL_SERIOUS_ACCIDENTS_2024 * JEDDAH_LICENSE_SHARE
JEDDAH_INSIDE_PER_DAY = (JEDDAH_SERIOUS_PER_YEAR * INTRACITY_RATIO) / 365
JEDDAH_OUTSIDE_PER_DAY = (JEDDAH_SERIOUS_PER_YEAR * OUTSIDE_RATIO) / 365

# ─── 2024 Jeddah calendar events ─────────────────────────────────────────────
RAMADAN_START = datetime(2024, 3, 11)
RAMADAN_END = datetime(2024, 4, 9)
EID_AL_FITR_START = datetime(2024, 4, 10)
EID_AL_FITR_END = datetime(2024, 4, 15)
HAJJ_START = datetime(2024, 6, 14)   # Pilgrims transit through Jeddah → surge
HAJJ_END = datetime(2024, 6, 19)
EID_AL_ADHA_START = datetime(2024, 6, 15)
EID_AL_ADHA_END = datetime(2024, 6, 20)
NATIONAL_DAY = datetime(2024, 9, 23)

# ─── Date range: full year 2024 ───────────────────────────────────────────────
dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(366)]

records = []

for date in dates:
    dow = date.weekday()            # Mon=0 … Fri=4, Sat=5
    is_weekend = dow in (4, 5)     # Saudi weekend: Friday & Saturday
    is_ramadan = RAMADAN_START <= date <= RAMADAN_END
    is_eid = (EID_AL_FITR_START <= date <= EID_AL_FITR_END or
              EID_AL_ADHA_START <= date <= EID_AL_ADHA_END)
    is_hajj = HAJJ_START <= date <= HAJJ_END   # Pilgrim traffic surge
    is_national_day = date == NATIONAL_DAY
    month = date.month

    # ── Seasonal factor ───────────────────────────────────────────────────────
    # Jeddah is coastal: milder than Riyadh.  Summer is hot & humid (not as
    # extreme as inland), so traffic dips less than in Riyadh.
    season_factor = {
        1: 1.06, 2: 1.08, 3: 1.04, 4: 0.97, 5: 0.92,
        6: 0.85, 7: 0.80, 8: 0.82, 9: 0.90, 10: 1.00,
        11: 1.04, 12: 1.05
    }[month]

    weekend_factor = 0.65 if is_weekend else 1.0
    ramadan_factor = 0.83 if is_ramadan else 1.0   # less daytime traffic
    eid_factor = 0.55 if is_eid else 1.0
    hajj_factor = 1.35 if is_hajj else 1.0         # pilgrim surge in Jeddah
    national_day_factor = 0.60 if is_national_day else 1.0

    combined = (season_factor * weekend_factor * ramadan_factor *
                eid_factor * hajj_factor * national_day_factor)

    # ── TrafficVolume (daily vehicles at monitoring point) ────────────────────
    # Jeddah base is slightly higher than Riyadh due to port/airport activity
    base_volume = 9_800
    volume = int(base_volume * combined * np.random.normal(1.0, 0.12))
    volume = int(np.clip(volume, 1_500, 24_000))

    # ── AverageSpeed (km/h) ───────────────────────────────────────────────────
    speed_base = 82 - (volume / 24_000) * 55
    avg_speed = round(float(np.clip(speed_base + np.random.normal(0, 5), 18, 110)), 1)

    # ── Accidents ─────────────────────────────────────────────────────────────
    ramadan_acc_factor = 1.25 if is_ramadan else 1.0
    hajj_acc_factor = 1.40 if is_hajj else 1.0   # more accidents during Hajj

    acc_inside_lambda = JEDDAH_INSIDE_PER_DAY * ramadan_acc_factor * hajj_acc_factor
    acc_outside_lambda = JEDDAH_OUTSIDE_PER_DAY * ramadan_acc_factor * hajj_acc_factor

    acc_inside = int(np.clip(np.random.poisson(acc_inside_lambda), 0, 18))
    acc_outside = int(np.clip(np.random.poisson(acc_outside_lambda), 0, 10))

    # ── WeatherSeverity (0–5) ─────────────────────────────────────────────────
    # Jeddah: coastal humidity in summer (2-3), winter fog / rare flooding (2),
    # spring sandstorms (1-2), generally mild rest of year.
    weather_mean = {
        1: 0.7, 2: 1.0, 3: 1.2, 4: 1.0, 5: 1.0,
        6: 2.0, 7: 2.4, 8: 2.3, 9: 1.5, 10: 0.9,
        11: 0.8, 12: 1.1    # Dec: occasional heavy rain / flooding risk
    }[month]
    weather = int(np.clip(np.random.normal(weather_mean, 0.85), 0, 5))

    # ── RoadWorkLevel (0–3): Vision 2030 & port expansion projects ────────────
    # More construction in cooler months (Oct–Mar)
    rw_mean = 1.7 if month in (10, 11, 12, 1, 2, 3) else 1.0
    road_work = int(np.clip(np.random.normal(rw_mean, 0.65), 0, 3))

    # ── CongestionIndex (10–100) ──────────────────────────────────────────────
    base_cong = 28 + (volume / 24_000) * 68
    base_cong += road_work * 4.5
    base_cong += weather * 2.5
    base_cong -= (avg_speed - 50) * 0.45
    if is_hajj:
        base_cong += 15   # Hajj congestion spike
    congestion = round(float(np.clip(base_cong + np.random.normal(0, 6), 10, 100)), 1)

    records.append({
        "Date": date.strftime("%d/%m/%Y"),
        "TrafficVolume": volume,
        "AverageSpeed": avg_speed,
        "AccidentsInsideCity": acc_inside,
        "AccidentsOutsideCity": acc_outside,
        "WeatherSeverity": weather,
        "RoadWorkLevel": road_work,
        "CongestionIndex": congestion
    })

df = pd.DataFrame(records)
df.to_csv("jeddah_traffic_dataset.csv", index=False)

print(f"Generated {len(df)} records  (01/01/2024 – 31/12/2024)")
print("\nSummary statistics:")
print(df.describe().to_string())
print("\nFirst 5 rows:")
print(df.head().to_string())

# ── Validation against real stats ────────────────────────────────────────────
total_inside  = df["AccidentsInsideCity"].sum()
total_outside = df["AccidentsOutsideCity"].sum()
expected_inside  = round(JEDDAH_INSIDE_PER_DAY * 366)
expected_outside = round(JEDDAH_OUTSIDE_PER_DAY * 366)

print(f"\nValidation vs real 2024 stats (Jeddah serious accidents):")
print(f"  AccidentsInsideCity   — generated: {total_inside:,}  expected ~{expected_inside:,}")
print(f"  AccidentsOutsideCity  — generated: {total_outside:,}  expected ~{expected_outside:,}")
print(f"  Jeddah licence share used: {JEDDAH_LICENSE_SHARE*100:.1f} %")
print(f"\nSources:")
print("  GASTAT – Road Transport Statistics 2024")
print("  https://www.stats.gov.sa/documents/20117/2435281/")
print("  KAPSARC Traffic Accidents Portal")
print("  https://datasource.kapsarc.org/explore/assets/saudi-arabia-traffic-accidents-and-casualties-injured-dead-2008/")
