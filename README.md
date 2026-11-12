# Credit Risk Scoring System: Freddie Mac Single-Family Loan-Level Dataset

A production-grade credit risk scoring system built on Freddie Mac's Single-Family Loan-Level Dataset (Release 48). The system implements an unweighted LightGBM default prediction model, an interactive business cost threshold optimizer, a Fairlearn-based demographic fairness audit engine with Wilson score confidence intervals, and a Population Stability Index (PSI) drift monitoring suite with precomputed macroeconomic counterfactual stress scenarios.

---

## 🏛️ Architecture Overview

```
                                      +------------------------+
                                      |   Freddie Mac Raw Data  |
                                      | (DataSet/ or data/raw/) |
                                      +-----------+------------+
                                                  |
                                          [src/pipeline/]
                                                  v
                                      +------------------------+
                                      |   Processed Parquet    |
                                      |  train, valid, oot,    |
                                      |       stress           |
                                      +-----------+------------+
                                                  |
                                          [LightGBM Model]
                                                  v
                                      +------------------------+
                                      |       artifacts/       |
                                      | model.txt, bins, shap, |
                                      | reports, holdout.pq    |
                                      +-----------+------------+
                                                  |
                                            (Read-only)
                                                  v
                                      +------------------------+
                                      |       Flask API        |
                                      |     (Port :8000)       |
                                      +-----------+------------+
                                                  |
                                             (REST /v1)
                                                  v
                                      +------------------------+
                                      |  Streamlit Dashboard   |
                                      |     (Port :8501)       |
                                      +------------------------+
```

1. **Shared Common Package (`src/common/`):** Unified business cost formulations, feature cleaning and origination transforms, PSI calculation algorithms, and demographic group encoding logic.
2. **Data Pipeline (`src/pipeline/`):** Freddie Mac loan ingestion, strict 24-month delinquency/termination default labeling (`default_24m`), temporal splits (`train` 2015–2017, `valid` 2018, `oot` 2019, `stress` 2020–2021), LightGBM booster training, TreeSHAP importance extraction, and drift/fairness evaluations.
3. **Flask REST API (`api/`):** High-performance read-only microservice serving predictions, local TreeSHAP explanations, what-if threshold sweeps, fairness metrics, and drift monitoring. Equipped with a strict startup self-check (`MD-13`).
4. **Streamlit Thin-Client Dashboard (`dashboard/`):** Reactive user interface with shared sidebar controls driving both the Business Optimizer and Fairness tabs, Altair interactive visualizations, plain-language drift alerts, and client-side caching.

---

## 🚀 Key Functional Upgrades (Appendix A)

### 1. Interactive What-If Business Threshold Optimizer (Change A)
- Formulates decision threshold selection as a business financial decision balancing original unpaid balance (UPB) losses:
  $$\text{Net Benefit} = L \cdot B_{\text{TP}} - g \cdot B_{\text{FP}}$$
  where $L$ is the loss severity from false negatives (default approval, mean 25%) and $g$ is the net interest margin foregone from false positives (rejecting creditworthy applicants, default 2%).
- Vectorized sweep evaluates all candidate thresholds in $\mathcal{O}(N \log N)$ time, generating net-benefit curves and cross-split evaluations across Validation, Out-of-Time, and Stress windows.
- Automatically flags tail probability miscalibration when the empirical optimum diverges from the analytic threshold $t^* = \frac{g}{g + L}$.

### 2. Fairness & Demographic Bias Audit Engine (Change B)
- Audits outcomes across single-family loan proxies (`first_time_homebuyer`, `borrower_count`, `occupancy`, `loan_purpose`) using Fairlearn metric frames.
- Reports rejection rates, default capture rates (true positive rates), and false rejection rates (false positive rates) with exact **Wilson score 95% confidence intervals**.
- Flags disparate impact violations ($\text{DI} < 0.80$), equal opportunity disparities ($\Delta \text{TPR} > 0.10$), and calibration ratio divergence.
- Displays regulatory contextual notices (FA-02) reminding reviewers that disparities in proxies are prompts for deeper investigation rather than unilateral proof of disparate treatment.

### 3. Population Stability Index (PSI) Drift Monitor (Change C)
- Monitors distribution drift on output scores and input credit features using validation-fitted reference decile bins.
- Categorizes stability into traffic-light status bands: **Stable** ($\text{PSI} < 0.10$), **Warning** ($0.10 \le \text{PSI} < 0.20$), and **Retrain Recommended** ($\text{PSI} \ge 0.20$).
- Tracks historical quarterly drift across 16 origination quarters (2018Q1 through 2021Q4).
- Evaluates 12 precomputed counterfactual macroeconomic shock scenarios (Mild, Moderate, Severe) across 4 stress types: *Interest Rate Shock*, *Debt Burden Shock*, *Credit Deterioration*, and *Leverage Squeeze*.

### 4. Freddie Mac Enterprise Dataset & Temporal Splits (Change D)
- Parses Freddie Mac Release 48 loan origination and monthly performance data.
- Enforces strict leakage controls: zero monthly performance fields used in feature origination tables.
- Chronological partitioning:
  - **Train:** 2015–2017 vintages (101,710 loans, 2.77% default rate)
  - **Validation:** 2018 vintage (49,956 loans, 2.71% default rate)
  - **Out-of-Time (OOT):** 2019 vintage (49,928 loans, 3.08% default rate)
  - **Macroeconomic Stress:** 2020–2021 vintages (99,881 loans, 1.82% default rate; exhibits $\text{PSI} = 1.23$ on interest rate due to pandemic policy shifts).

---

## 🛠️ Installation & Quickstart

### Prerequisites
- Python 3.11+
- Virtual environment tool (`venv`)
- Docker & Docker Compose (optional, for containerized run)

### Local Environment Setup
```bash
# 1. Clone repository and navigate to root
cd Minor-Project-

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# or: .\.venv\Scripts\Activate.ps1 # On Windows PowerShell

# 3. Install dependencies and editable package
pip install -r requirements/pipeline.txt
pip install -r requirements/api.txt
pip install -r requirements/dashboard.txt
pip install -e .
```

### Running the End-to-End Pipeline
Rebuild all datasets, models, reports, and artifacts from raw data using `make` (or Python directly):
```bash
# Ingest raw Freddie Mac archives from data/raw/
python -m src.pipeline.ingest_data

# Process features and labels into temporal splits
python -m src.pipeline.build_dataset

# Train LightGBM model and generate holdout predictions
python -m src.pipeline.train
python -m src.pipeline.make_holdout

# Compute fairness baseline, drift reports, and shock simulations
python -m src.pipeline.fairness
python -m src.pipeline.drift
python -m src.pipeline.simulate_shift

# Or run all steps in order:
make all
```

---

## 🖥️ Running the Application

### 1. Launch Backend Flask API
```bash
python -m api.app
# Serving on http://127.0.0.1:8000
```

### 2. Launch Streamlit Dashboard
```bash
streamlit run dashboard/app.py
# Web UI accessible at http://127.0.0.1:8501
```

### 3. Containerized Deployment (Docker Compose)
```bash
docker compose up --build
```
The API is exposed on `http://127.0.0.1:8000` and the Streamlit dashboard is exposed on `http://127.0.0.1:8501`.

---

## 📡 API Specification Reference (`/v1`)

All endpoints are read-only HTTP `GET` endpoints returning JSON with model provenance headers (`model_version`, `data_release`):

| Endpoint | Description | Key Parameters |
|---|---|---|
| `GET /v1/health` | Readiness probe & split loan counts | None |
| `GET /v1/meta` | System metadata, slider limits, features | None |
| `GET /v1/loans` | Filterable list of holdout loans | `split`, `limit`, `offset`, `search` |
| `GET /v1/loans/<loan_id>` | Single loan characteristics & UPB | None |
| `GET /v1/predict/<loan_id>` | Default probability & decision | `threshold` (optional) |
| `GET /v1/explain/<loan_id>` | Local TreeSHAP log-odds breakdown | None |
| `GET /v1/explain/global` | Global feature importance rankings | `top` (default 20) |
| `GET /v1/optimize` | Business net benefit threshold sweep | `split`, `fp_cost_pct`, `fn_cost_pct`, `mode`, `threshold` |
| `GET /v1/fairness` | Subgroup fairness audit with Wilson CIs | Same as `/optimize`, plus `attribute` |
| `GET /v1/drift` | Score & feature PSI drift reports | `source` (real/simulated), `window`, `scenario`, `severity` |
| `GET /v1/model-report` | Full discrimination & calibration report | None |

---

## 🧪 Testing & Verification

The test suite covers unit properties (TS-01 through TS-07) and integration contracts (TS-08, TS-09, TS-10):

```bash
pytest -v
```

### Verification Checklist:
- `TS-01`: Optimizer sweep matches exact brute-force search over random tie-dense distributions.
- `TS-02`: Net benefit equals portfolio profit minus approve-all profit across all thresholds.
- `TS-03`: 24-month labeling window verified on synthetic event sequences (terminations, payoffs, defects).
- `TS-04`: Feature leakage check verifies no monthly performance attributes exist in feature sets.
- `TS-05`: PSI handworked 2-bin calculations and zero-drift identities verified.
- `TS-06`: Simulated counterfactual drift exhibits monotonic PSI response with severe shocks crossing 0.20.
- `TS-07`: API fairness metrics reconcile with Fairlearn `MetricFrame` reference baselines within $10^{-9}$.
- `TS-08`: API endpoint schema contracts, query parameter range checks, and 404 enforcement on training loans.
- `TS-09`: API startup self-check halts execution if `model.txt` is missing or tampered.
- `TS-10`: End-to-end model training determinism and probability reproducibility verified under fixed seed.

---

## ⚖️ Governance, Compliance & Disclaimers

1. **Academic & Portfolio Demonstration (SRS 1.4):** This software is an academic engineering portfolio project designed to demonstrate MLOps, model explainability, fairness auditing, and drift monitoring. It is not licensed or calibrated for live consumer lending or automated credit origination.
2. **Fairness Proxies (FA-01, FA-02):** The Freddie Mac single-family loan dataset contains no direct protected demographic characteristics (race, gender, religion, national origin). Audit metrics utilize loan-level proxies (`first_time_homebuyer`, `borrower_count`, `occupancy`, `loan_purpose`). Disparities in proxy groups reflect broader structural credit patterns and must not be treated as proof of algorithmic bias without comprehensive demographic surveys.
3. **Data Licensing Terms (DT-09):** All loan-level data is derived from the Freddie Mac Single-Family Loan-Level Dataset under user registration terms. Raw, interim, and processed loan-level data files (`data/`, `artifacts/`, `reports/`, `DataSet/`) are excluded from version control (`.gitignore`) and cannot be commercially redistributed.
