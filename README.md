<div align="center">

# ⚡ Household Power Consumption — Regression Lab

### Predicting the next minute of household electricity demand from four years of real-world energy data.

<br>

![Status](https://img.shields.io/badge/Status-In%20Progress-8B5CF6?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)

<br>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=20&duration=2800&pause=700&color=F59E0B&center=true&vCenter=true&width=760&lines=2%2C075%2C259+minute-level+observations;Time-series+feature+engineering;Naive+baseline+%E2%86%92+Linear+Regression+%E2%86%92+Tree+Models;Real-world+ML+pipeline+from+raw+data+to+evaluation" alt="Typing animation" />

</div>

---

## 🧭 Project Snapshot

> **Goal:** build an end-to-end regression system that predicts **Global Active Power (kW)** using historical household electricity consumption and calendar/time features.

This project uses the **UCI Individual Household Electric Power Consumption** dataset and treats the problem as a **time-series regression problem**, not ordinary randomly shuffled tabular regression.

### The core question

```text
Given everything we knew BEFORE minute t...

                ⏱️  t-60   t-15   t-5   t-1
                  │        │      │     │
                  └────────┴──────┴─────┤
                                         ▼
                              🤖 Regression Model
                                         │
                                         ▼
                              ⚡ Power at time t
```

---

## 📊 Dataset

| Property | Value |
|---|---:|
| Original observations | **2,075,259** |
| Time resolution | **1 minute** |
| Original period | **2006–2010** |
| Original columns | **9** |
| Modeling rows | **2,045,110** |
| Final modeling features | **11** |
| Missing values after cleaning | **0** |

### Raw signals

- `Global_active_power`
- `Global_reactive_power`
- `Voltage`
- `Global_intensity`
- `Sub_metering_1`
- `Sub_metering_2`
- `Sub_metering_3`

### Calendar signals

- `hour`
- `day_of_week`
- `month`
- `year`
- `is_weekend`

### Historical signals

- `lag_1`
- `lag_5`
- `lag_15`
- `lag_60`
- `rolling_mean_15`
- `rolling_mean_60`

---

# 🧠 Why This Project Is Different

A random train/test split would allow observations from the **future** to leak into training.

Instead, the project respects chronological order:

```text
PAST                                                   FUTURE
│                                                       │
├──────────────── TRAIN ────────────────┤
                                        ├── VALIDATION ─┤
                                                         ├── TEST ───►
```

### Split

| Set | Rows | Period |
|---|---:|---|
| 🟣 Train | 1,431,577 | Dec 2006 → Sep 2009 |
| 🟠 Validation | 306,766 | Sep 2009 → Apr 2010 |
| 🔵 Test | 306,767 | Apr 2010 → Nov 2010 |

This simulates the real-world situation:

> **Train on the past → tune on newer data → evaluate on unseen future data.**

---

# 🧹 Data Cleaning

The original dataset contains missing measurements represented by `?`.

### Numeric conversion

```python
numeric_cols = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3"
]

data[numeric_cols] = data[numeric_cols].apply(
    pd.to_numeric,
    errors="coerce"
)
```

### Missing-value investigation

The largest consecutive missing block contained **7,226 minutes** of missing data.

Instead of blindly interpolating large outages, the modeling dataset removes rows that cannot provide the complete historical information required by the engineered features.

Result:

```text
2,075,259 raw rows
        ↓
   cleaning
        ↓
2,045,110 modeling rows
        ↓
       0 NA
```

---

# ⚙️ Feature Engineering

The project focuses heavily on **temporal signal extraction**.

## ⏱️ Lag Features

A lag feature asks:

> "What was the power consumption X minutes ago?"

```python
data["lag_1"]  = data["global_active_power"].shift(1)
data["lag_5"]  = data["global_active_power"].shift(5)
data["lag_15"] = data["global_active_power"].shift(15)
data["lag_60"] = data["global_active_power"].shift(60)
```

Conceptually:

```text
t-60 ────────────────┐
t-15 ───────────┐    │
t-5  ───────┐   │    │
t-1  ───┐   │   │    │
        ▼   ▼   ▼    ▼
       [ Historical Context ]
                  │
                  ▼
             Prediction t
```

---

## 📈 Rolling Features

Rolling averages capture the **recent trend** rather than a single previous observation.

```python
data["rolling_mean_15"] = (
    data["global_active_power"]
    .shift(1)
    .rolling(15)
    .mean()
)

data["rolling_mean_60"] = (
    data["global_active_power"]
    .shift(1)
    .rolling(60)
    .mean()
)
```

### Why `.shift(1)`?

Because the model must not see the target value it is trying to predict.

```text
❌ Leakage

rolling window ──────► includes t

        prediction
             ▲
             │
             t


✅ Correct

t-60 ... t-2  t-1 ──► rolling window

                     │
                     ▼
                 predict t
```

---

# 🔎 Exploratory Findings

## Distribution

The target is **right-skewed**:

```text
Frequency
  ▲
  │ ███████████
  │ █████████████
  │ █████████
  │ ████
  │ ██
  │ █
  └────────────────────────────► Power
     high concentration    long tail
```

Statistics:

| Metric | Value |
|---|---:|
| Mean | **1.090 kW** |
| Median | **0.600 kW** |
| Std | **1.056 kW** |
| Minimum | **0.076 kW** |
| Maximum | **11.122 kW** |

---

## 🕐 Daily Pattern

Average power changes significantly throughout the day.

```text
00 ──╮
     ╰──╮
03      ╰───╮
05          ╰──╮
07              ╭────
                │
12          ────╯
16        ╭─────
20       ╭╯  ⚡ PEAK
22     ╭─╯
24 ────╯
```

The strongest average consumption occurs around the evening period.

---

## 🗓️ Weekday vs Weekend

| Period | Average Power |
|---|---:|
| Weekday | **1.034 kW** |
| Weekend | **1.232 kW** |

This supports keeping `is_weekend` as a candidate predictive feature.

---

# 🧪 Modeling Strategy

The project deliberately starts simple.

```text
                         ┌──────────────────────┐
                         │   Historical Data    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Feature Engineering  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                 ┌───────────────────────────────────┐
                 │ Chronological Train/Val/Test Split │
                 └───────────────┬───────────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
       🧱 Baseline        📐 Linear Model      🌲 Tree Model
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 ▼
                         📊 Model Comparison
                                 │
                                 ▼
                        🏆 Final Model
```

### Models

- 🧱 **Naive persistence baseline**
- 📐 **Linear Regression**
- 🌲 **Random Forest Regressor**
- 🔧 Hyperparameter tuning
- 🏁 Final evaluation on untouched test data

---

# 🧱 Baseline Result

The first benchmark predicts:

> **Next power value ≈ previous minute's power value**

Using `lag_1`:

| Metric | Baseline |
|---|---:|
| MAE | **0.0834** |
| RMSE | **0.2526** |
| R² | **0.9426** |

This is intentionally difficult to beat.

It demonstrates how strongly correlated minute-to-minute household consumption can be.

---

# 📐 Linear Regression

Initial validation result:

| Metric | Linear Regression |
|---|---:|
| MAE | **0.0960** |
| RMSE | **0.2500** |
| R² | **0.9438** |

Interesting outcome:

```text
MAE   → Baseline wins
RMSE  → Linear Regression wins
R²    → Linear Regression wins
```

This is why the project evaluates multiple metrics rather than declaring a model successful based on one number.

---

# 📏 Evaluation Metrics

### MAE

> Average absolute prediction error.

```text
MAE ↓ = better
```

### RMSE

> Similar to MAE, but heavily penalizes large errors.

```text
RMSE ↓ = better
```

### R²

> How much variance in the target is explained by the model.

```text
R² ↑ = better
```

---

# 🛠️ Tech Stack

<div align="center">

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Scikit-learn` · `Jupyter`

</div>

---

# 📁 Project Structure

```text
household-power-regression/
│
├── 📂 data/
│   ├── raw/
│   └── processed/
│
├── 📂 notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_final_model.ipynb
│
├── 📂 src/
│   ├── preprocessing.py
│   ├── features.py
│   └── train.py
│
├── 📂 models/
│
├── 📂 reports/
│
├── requirements.txt
└── README.md
```

---

# 🚀 Roadmap

- [x] Load and inspect dataset
- [x] Clean numeric columns
- [x] Investigate missing blocks
- [x] Create datetime index
- [x] Engineer calendar features
- [x] Engineer lag features
- [x] Engineer rolling features
- [x] Build clean modeling dataset
- [x] Exploratory analysis
- [x] Chronological split
- [x] Naive baseline
- [x] Linear Regression
- [ ] Random Forest
- [ ] Model comparison
- [ ] Hyperparameter tuning
- [ ] Feature importance
- [ ] Final test evaluation
- [ ] Save production model
- [ ] Prediction pipeline
- [ ] Final project dashboard/visuals
- [ ] Project documentation

---

# 🧠 Key ML Lessons

This project is designed to demonstrate more than simply fitting a regression model.

### 01 — Time changes how you split data

Random splitting can destroy temporal structure and introduce leakage.

### 02 — Simple baselines matter

A sophisticated model must earn its complexity.

### 03 — Feature engineering can dominate model choice

`lag_1` alone provides an extremely strong signal.

### 04 — Leakage is subtle

Rolling features must only contain information available **before** the prediction timestamp.

### 05 — Metrics tell different stories

MAE, RMSE and R² can disagree—and that disagreement is informative.

---

# ⭐ What Makes This Portfolio-Ready

```text
RAW DATA
   ↓
DATA QUALITY
   ↓
TIME-AWARE EDA
   ↓
TEMPORAL FEATURE ENGINEERING
   ↓
LEAKAGE CONTROL
   ↓
CHRONOLOGICAL VALIDATION
   ↓
BASELINE
   ↓
MULTIPLE ML MODELS
   ↓
ERROR ANALYSIS
   ↓
TUNING
   ↓
FINAL MODEL
   ↓
REPRODUCIBLE PIPELINE
```

This is not just:

> "I trained Random Forest on a Kaggle dataset."

It is an attempt to demonstrate the **full reasoning process behind a real regression system**.

---

<div align="center">

### ⚡ Built as an end-to-end ML learning project

**Raw electricity measurements → temporal features → regression → evaluation**

<br>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:78350F,50:F59E0B,100:451A03&height=120&section=footer" width="100%" />

</div>
