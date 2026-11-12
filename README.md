# 🏦 Credit Risk Scoring System: Freddie Mac Single-Family Loan-Level Dataset

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-success.svg)](https://lightgbm.readthedocs.io/)
[![Fairlearn](https://img.shields.io/badge/Fairness-Fairlearn-orange.svg)](https://fairlearn.org/)
[![Flask](https://img.shields.io/badge/API-Flask%203.1-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit%201.67-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20Compose-blue.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-23%20Passed-brightgreen.svg)](tests/)

---

## 📌 Repository Description

> **An enterprise-grade mortgage default prediction engine and credit risk decision platform built on Freddie Mac's Single-Family Loan-Level Dataset (Release 48). Features an interactive business cost threshold optimizer, a Fairlearn-based demographic fairness audit engine with Wilson score confidence intervals, Population Stability Index (PSI) drift monitoring under simulated macroeconomic stress shocks, and local TreeSHAP explainability.**

### Executive Summary
Traditional credit scoring models evaluate statistical metrics like AUC-ROC in isolation, ignoring the asymmetric financial trade-offs of lending decisions and the regulatory imperatives of fair lending. This system bridges machine learning and credit risk management:
- **Decision Threshold Optimization:** Converts model probabilities into balance-weighted dollar profit and loss through an unweighted LightGBM classifier and a vectorized what-if optimizer.
- **Fairness & Bias Governance:** Audits approval disparities across demographic proxies with exact 95% Wilson confidence intervals and contextual regulatory disclaimers.
- **Population Drift Monitoring:** Detects score and feature distribution shifts against historical reference deciles and evaluates 12 precomputed counterfactual macroeconomic shock scenarios.
- **Production-Ready Architecture:** Designed with strict separation of concerns—a high-throughput, read-only Flask API (`/v1`) with a startup self-check (`MD-13`) and a thin-client Streamlit interactive dashboard.

---

## 🏛️ System Architecture

```
                                  +---------------------------------------+
                                  |    Freddie Mac Single-Family Data     |
                                  |   (Release 48: DataSet/ & data/raw/)  |
                                  +-------------------+-------------------+
                                                      |
                                             [src/pipeline/]
                                                      v
                                  +---------------------------------------+
                                  |      Temporal Partitioning Parquet    |
                                  |   train (2015-17), valid (2018),      |
                                  |   oot (2019), stress (2020-21)        |
                                  +-------------------+-------------------+
                                                      |
                                          [LightGBM Model Training]
                                                      v
                                  +---------------------------------------+
                                  |             artifacts/                |
                                  |   model.txt, reference_bins.json,     |
                                  |   holdout_predictions.parquet,        |
                                  |   fairness, drift & psi_simulated     |
                                  +-------------------+-------------------+
                                                      |
                                               (Read-Only Volume)
                                                      v
                                  +---------------------------------------+
                                  |            Flask REST API             |
                                  |      (Port :8000 · Sync Workers)      |
                                  +-------------------+-------------------+
                                                      |
                                              (HTTP JSON /v1)
                                                      v
                                  +---------------------------------------+
                                  |     Streamlit Interactive Dashboard   |
                                  |      (Port :8501 · Thin Client)       |
                                  +---------------------------------------+
```

### Core Architecture Principles
1. **Unweighted Probabilities:** Model trained without artificial class weights (`is_unbalance=False`, `scale_pos_weight=1.0`) so predicted probabilities remain well-calibrated empirical default risks ($P(Y=1|X)$).
2. **Strict Leakage Prevention:** Feature extraction utilizes only origination characteristics available at underwriting. Zero monthly performance file fields exist in feature vectors (`TS-04`).
3. **Thin Client Design:** The dashboard performs zero direct file reads or model computations, interacting exclusively via the versioned Flask REST API (`UI-13`).
4. **Startup Self-Check:** The API verifies at startup that the serialized model reproduces stored holdout probabilities within $10^{-4}$ tolerance, halting immediately if artifacts are corrupted or mismatched (`MD-13`, `TS-09`).

---

## 🚀 Key Functional Capabilities

### 1. Interactive What-If Business Threshold Optimizer (Change A)
Underwriting approval is a capital allocation decision. Rather than selecting an arbitrary score cut-off (e.g., 0.50), the system evaluates balance-weighted financial trade-offs:
$$\text{Net Benefit} = L \cdot B_{\text{TP}} - g \cdot B_{\text{FP}}$$
where:
- $L$ = Net loss severity incurred from approving a defaulting borrower (mean: **25.0%** of original UPB).
- $g$ = Net interest margin foregone by rejecting a creditworthy borrower (default: **2.0%** of original UPB).
- $B_{\text{TP}}$ = Total original unpaid balance of correctly rejected defaulters (True Positives).
- $B_{\text{FP}}$ = Total original unpaid balance of erroneously rejected good borrowers (False Positives).

The optimizer sweeps candidate thresholds in $\mathcal{O}(N \log N)$ time and flags tail miscalibration when empirical thresholds diverge from the theoretical optimum $t^* = \frac{g}{g + L}$ (`OP-09`).

### 2. Fairness & Demographic Bias Audit Engine (Change B)
- Audits outcomes across single-family loan proxies: `first_time_homebuyer`, `borrower_count`, `occupancy`, and `loan_purpose`.
- Computes disparate impact ratios ($\text{DI} < 0.80$), equal opportunity differences ($\Delta \text{TPR} > 0.10$), and calibration ratios across groups.
- Calculates exact **Wilson score 95% confidence intervals** for default capture rates and false rejection rates (`FA-08`).
- Displays mandatory regulatory contextual notices (`FA-02`) reinforcing that disparities in proxies are prompts for institutional inquiry, not standalone proof of disparate treatment.

### 3. PSI Drift Monitoring & Macroeconomic Stress Shocks (Change C)
- Monitors distribution drift using Population Stability Index (PSI) against validation-fitted reference deciles:
  - **Stable** ($\text{PSI} < 0.10$): Green / Normal cadence.
  - **Warning** ($0.10 \le \text{PSI} < 0.20$): Amber / Increased monitoring.
  - **Retrain Recommended** ($\text{PSI} \ge 0.20$): Red / Recalibration required (`DR-15`).
- Tracks historical quarterly drift across 16 consecutive quarters (`2018Q1` through `2021Q4`).
- Simulates 12 counterfactual stress shock scenarios (Mild, Moderate, Severe) across 4 macroeconomic regimes: *Interest Rate Spike*, *Debt Burden Shock*, *Credit Deterioration*, and *Leverage Squeeze*.

### 4. Freddie Mac Dataset Pipeline & Temporal Splits (Change D)
- Processes single-family 30-year fixed-rate mortgages:
  - **Training Split (2015–2017):** 101,710 loans, 2.77% default rate.
  - **Validation Split (2018):** 49,956 loans, 2.71% default rate.
  - **Out-of-Time / OOT Split (2019):** 49,928 loans, 3.08% default rate.
  - **Macroeconomic Stress Split (2020–2021):** 99,881 loans, 1.82% default rate (exhibits interest rate drift $\text{PSI} = 1.23$ during COVID-era rate drops).
- Strict 24-month default definition (`default_24m`): reaching 90+ days delinquent (`delinq_status >= '03'`) or zero-balance liquidation/short sale within 24 months of first payment date (`DP-07`).

---

## 💻 Installation & Setup Guide for a New Computer

Follow these step-by-step instructions to set up, build, and run the project from scratch on any new machine.

### Step 1: System Prerequisites
Ensure you have the following installed on your machine:
- **Git:** [Download Git](https://git-scm.com/)
- **Python 3.11, 3.12, or 3.14:** [Download Python](https://www.python.org/downloads/) (ensure **"Add Python to PATH"** is checked during Windows installation).
- **C++ Compiler / OpenMP (Linux/macOS only):**
  - **Ubuntu / Debian:** `sudo apt-get update && sudo apt-get install -y build-essential libgomp1`
  - **macOS:** `brew install libomp`
  - **Windows:** Standard Python wheels include OpenMP automatically.
- **Docker & Docker Compose (Optional):** [Download Docker Desktop](https://www.docker.com/products/docker-desktop/) if you prefer containerized execution.

---

### Step 2: Clone the Repository
Open your terminal (PowerShell on Windows, Bash/Zsh on Linux/macOS):
```bash
git clone https://github.com/<your-username>/freddie-mac-credit-risk-scoring.git
cd freddie-mac-credit-risk-scoring
```

---

### Step 3: Create & Activate Virtual Environment

#### On Windows (PowerShell):
```powershell
# Create virtual environment
python -m venv .venv

# If script execution is restricted, enable it for this process:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

#### On Linux / macOS (Bash / Zsh):
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

*(Your terminal prompt should now show `(.venv)`).*

---

### Step 4: Install Dependencies & Editable Package
Install the pinned requirements and register the shared `crs_common` package in editable mode:

```bash
# Upgrade pip, setuptools, and wheel
python -m pip install --upgrade pip setuptools wheel

# Install component requirements
pip install -r requirements/pipeline.txt
pip install -r requirements/api.txt
pip install -r requirements/dashboard.txt

# Install the repository as an editable package
pip install -e .
```

---

### Step 5: Data Placement Guide
Freddie Mac raw loan-level data files should be placed in either the `DataSet/` folder or `data/raw/` directory:
- **Option A (Pre-packaged):** Place the 50,000-loan sampled archives `sample_2015.zip` through `sample_2021.zip` directly into `data/raw/`.
- **Option B (Raw Freddie Mac files):** If starting from the 129 raw monthly performance and origination files from Freddie Mac Single-Family Dataset (Release 48), place them in the `DataSet/` folder and run:
  ```bash
  python src/pipeline/prepare_raw.py
  ```
  *(This automatically cleans and samples the vintages into `data/raw/sample_YYYY.zip`).*

---

### Step 6: Build Datasets, Train Model & Precompute Reports
Run the end-to-end data processing and model pipeline. You can run all steps via `make` or run each Python module sequentially:

#### Using Make:
```bash
make all
```

#### Or Using Python Directly:
```bash
# 1. Ingest raw files and validate schema column counts
python -m src.pipeline.ingest_data

# 2. Build temporal splits (train, valid, oot, stress) with 24-month labels
python -m src.pipeline.build_dataset

# 3. Train LightGBM model and generate holdout predictions
python -m src.pipeline.train
python -m src.pipeline.make_holdout

# 4. Generate fairness baseline, PSI drift reports, and shock simulations
python -m src.pipeline.fairness
python -m src.pipeline.drift
python -m src.pipeline.simulate_shift
```

*(This creates all verified artifacts in `artifacts/`: `model.txt`, `metadata.json`, `categories.json`, `holdout_predictions.parquet`, `model_report.json`, `global_importance.json`, `fairness_baseline.json`, `psi_report.json`, and `psi_simulated.json`).*

---

### Step 7: Launch the Applications

#### 1. Start the Flask Backend API (Terminal 1)
```bash
python -m api.app
```
- **Serving at:** `http://127.0.0.1:8000`
- **Health check:** Open `http://127.0.0.1:8000/v1/health` in your browser.

#### 2. Start the Streamlit Dashboard (Terminal 2)
Open a second terminal, activate `.venv`, and run:
```bash
streamlit run dashboard/app.py
```
- **Web UI:** Automatically opens in your browser at `http://localhost:8501`.

---

### Step 8: Alternative — Run with Docker Compose
If you have Docker installed, you can launch the entire stack (API + Dashboard) with a single command without configuring local Python environments:

```bash
docker compose up --build
```
- **Flask API:** Accessible on `http://localhost:8000`
- **Streamlit Dashboard:** Accessible on `http://localhost:8501`

---

### Step 9: Run Automated Verification Tests
Verify that all unit tests, mathematical properties, and integration contracts pass:

```bash
pytest -v
```

**Expected output:**
```
tests/integration/test_api.py::test_ts08_health PASSED                   [  4%]
tests/integration/test_api.py::test_ts08_meta PASSED                     [  8%]
tests/integration/test_api.py::test_ts08_loans_and_picker PASSED         [ 13%]
tests/integration/test_api.py::test_ts08_loan_detail_predict_explain PASSED [ 17%]
tests/integration/test_api.py::test_ts08_unknown_and_training_loans_404 PASSED [ 21%]
tests/integration/test_api.py::test_ts08_optimize PASSED                 [ 26%]
tests/integration/test_api.py::test_ts08_fairness PASSED                 [ 30%]
tests/integration/test_api.py::test_ts08_drift_and_reports PASSED        [ 34%]
tests/integration/test_api.py::test_ts09_startup_self_check_tampered_model PASSED [ 39%]
tests/integration/test_pipeline.py::test_ts10_reproducibility PASSED     [ 43%]
tests/unit/test_cost.py::test_sweep_matches_bruteforce PASSED            [ 47%]
tests/unit/test_cost.py::test_cost_identity PASSED                       [ 52%]
tests/unit/test_cost.py::test_analytic_threshold_and_calibration PASSED  [ 56%]
tests/unit/test_fairness.py::test_api_fairness_matches_fairlearn_baseline PASSED [ 60%]
tests/unit/test_fairness.py::test_reject_nothing_fairness PASSED         [ 65%]
tests/unit/test_features.py::test_leakage_control PASSED                 [ 69%]
tests/unit/test_features.py::test_borrower_cap PASSED                    [ 73%]
tests/unit/test_features.py::test_not_available_sentinels PASSED         [ 78%]
tests/unit/test_label.py::test_label_construction_and_exclusions PASSED  [ 82%]
tests/unit/test_psi.py::test_psi_identical_data_is_zero PASSED           [ 86%]
tests/unit/test_psi.py::test_psi_handworked_example PASSED               [ 91%]
tests/unit/test_psi.py::test_psi_empty_bin_stays_finite PASSED           [ 95%]
tests/unit/test_simulation.py::test_simulation_psi_properties PASSED     [100%]

============================= 23 passed in ~5.4s ==============================
```

---

## 📁 Repository Structure

```
.
├── api/                             # Flask API microservice
│   ├── routes/
│   │   ├── __init__.py
│   │   └── v1.py                    # All /v1 endpoint implementations
│   ├── __init__.py
│   ├── app.py                       # App factory, request timing, request IDs & logging
│   ├── Dockerfile                   # Python 3.11-slim + libgomp1 container
│   ├── errors.py                    # Custom ApiError & AP-04 JSON handlers
│   ├── loaders.py                   # Store class & startup self-check (MD-13)
│   └── params.py                    # Strict query parameter validator
│
├── dashboard/                       # Streamlit thin-client dashboard
│   ├── tabs/
│   │   ├── __init__.py
│   │   ├── drift.py                 # Drift Monitor tab (UI-06, UI-08)
│   │   ├── fairness.py              # Fairness & Bias Audit tab (UI-05)
│   │   ├── loan_lookup.py           # Loan Lookup & TreeSHAP tab (UI-03)
│   │   ├── model_info.py            # Model & Data Governance tab (UI-07)
│   │   └── optimizer.py             # Business Threshold Optimizer tab (UI-04)
│   ├── __init__.py
│   ├── app.py                       # Main UI layout, shared sidebar controls (UI-02)
│   ├── client.py                    # Cached API client with retry & stale fallbacks
│   └── Dockerfile                   # Streamlit lightweight container
│
├── requirements/                    # Pinned environment specifications
│   ├── api.txt                      # Flask, Gunicorn, LightGBM, Pandas, PyArrow
│   ├── dashboard.txt                # Streamlit, Altair, Requests
│   └── pipeline.txt                 # Scikit-learn, Fairlearn, Scipy, LightGBM
│
├── schema/                          # Freddie Mac Release 48 column schemas
│   ├── origination_columns.json     # 32 origination attributes
│   └── performance_columns.json     # 28 monthly performance attributes
│
├── src/                             # Core Python packages
│   ├── common/                      # Shared reusable libraries
│   │   ├── __init__.py
│   │   ├── config.py                # Config & path resolution helpers
│   │   ├── cost.py                  # Vectorized balance sweep & net benefit logic
│   │   ├── features.py              # Origination transforms & category encoders
│   │   ├── groups.py                # Wilson CIs & fairness group encoders
│   │   └── psi.py                   # PSI decile binning & drift calculation
│   │
│   └── pipeline/                    # Batch data processing & training scripts
│       ├── build_dataset.py         # 24m labeling, cleaning, and parquet splits
│       ├── drift.py                 # Split and quarterly PSI report generation
│       ├── fairness.py              # Fairlearn MetricFrame baseline computation
│       ├── ingest_data.py           # Schema validation & interim parquet creation
│       ├── make_holdout.py          # Prediction generation & TreeSHAP extraction
│       ├── prepare_raw.py           # WLS/STACR file parser & vintage sampler
│       ├── simulate_shift.py        # 12 counterfactual shock scenarios
│       └── train.py                 # 5-fold CV hyperparameter search & training
│
├── tests/                           # Complete automated test suite
│   ├── integration/
│   │   ├── test_api.py              # TS-08 endpoint contracts & TS-09 self-check
│   │   └── test_pipeline.py         # TS-10 training reproducibility & determinism
│   │
│   └── unit/
│       ├── test_cost.py             # TS-01 sweep vs brute-force, TS-02 identity
│       ├── test_fairness.py         # TS-07 vectorized metrics vs Fairlearn
│       ├── test_features.py         # TS-04 leakage control & sentinels
│       ├── test_label.py            # TS-03 synthetic event sequences
│       ├── test_psi.py              # TS-05 handworked PSI & finite bins
│       └── test_simulation.py       # TS-06 monotonic drift response
│
├── .editorconfig
├── .gitattributes
├── .gitignore                       # Excludes data/, artifacts/, reports/, DataSet/
├── config.yaml                      # Centralized configuration (zero hardcoded values)
├── docker-compose.yml               # Multi-container orchestration (127.0.0.1 bound)
├── Makefile                         # Unified automation targets
├── pyproject.toml                   # Project metadata & editable install configuration
└── README.md                        # Documentation
```

---

## 📡 API Specification Reference (`/v1`)

All endpoints are read-only HTTP `GET` endpoints returning JSON with UTF-8 encoding. Every successful response includes model provenance headers (`model_version`, `data_release`):

| Endpoint | Purpose | Key Parameters | Cache Header |
|---|---|---|---|
| `GET /v1/health` | Readiness probe & split loan counts | None | `no-store` |
| `GET /v1/meta` | System settings, slider ranges, feature list | None | `max-age=60` |
| `GET /v1/loans` | Filterable holdout loan list for picker | `split` (req), `limit`, `offset`, `search` | `no-store` |
| `GET /v1/loans/<loan_id>` | Detailed origination features & UPB | None | `no-store` |
| `GET /v1/predict/<loan_id>` | Probability & decision at default or custom threshold | `threshold` (optional, 0–1) | `no-store` |
| `GET /v1/explain/<loan_id>` | Local TreeSHAP log-odds breakdown + other features | None | `no-store` |
| `GET /v1/explain/global` | Global feature importance rankings | `top` (default: 20) | `max-age=60` |
| `GET /v1/optimize` | Business net benefit threshold sweep | `split`, `fp_cost_pct`, `fn_cost_pct`, `mode`, `threshold` | `no-store` |
| `GET /v1/fairness` | Demographic audit with Wilson 95% CIs | Same as `/optimize`, plus `attribute` | `no-store` |
| `GET /v1/drift` | Score & feature PSI drift reports | `source` (real/simulated), `window`, `scenario`, `severity` | `max-age=60` |
| `GET /v1/model-report` | Full model discrimination & calibration report | None | `max-age=60` |

---

## 🖥️ Streamlit Dashboard Tour

The interactive dashboard is organized into 5 functional tabs controlled by the reactive sidebar:

1. **Loan Lookup (`UI-03`):**
   - Search loans by ID prefix across splits.
   - Inspect model predicted default probability vs actual historical 24-month outcome.
   - Interactive horizontal TreeSHAP bar chart with risk-increasing (red) and risk-decreasing (green) log-odds contributions reconciling exactly with the score.
2. **Business Optimizer (`UI-04`):**
   - Headline metrics: Effective Threshold, Net Financial Benefit ($), Benefit vs Approve-All ($), Recall / Default Capture (%), and Rejection Rate (%).
   - Altair interactive Net Benefit Curve with selected optimum marker.
   - Cross-split comparison table across Validation, Out-of-Time, and Macroeconomic Stress splits.
   - Transparent display of calibration warnings and business assumptions (`OP-03`).
3. **Fairness & Bias Audit (`UI-05`):**
   - Subgroup audit across sensitive loan proxies.
   - Disparate impact flags panel with mandatory `FA-02` context notes.
   - Rejection rate bar charts and error rate charts with exact **Wilson score 95% confidence intervals** (`mark_errorbar`).
   - Baseline-versus-current threshold drift comparison.
4. **Drift Monitor (`UI-06`, `UI-08`):**
   - Traffic-light status banner with plain-language status (Stable, Warning, Retrain Recommended) and `DR-15` recommended actions.
   - Feature PSI bar chart with `0.10` and `0.20` reference threshold lines.
   - Reference decile vs current window score distribution overlay.
   - 16-quarter historical PSI drift trend chart (`2018Q1`–`2021Q4`).
   - Scenario selector for 12 simulated counterfactual stress shocks.
5. **Model & Data Governance (`UI-07`):**
   - Model lineage, training vintage metadata, and discrimination metrics (AUC-ROC, Brier score, KS statistic).
   - OOT decile ordering and empirical calibration tables.
   - Global feature importance rankings.

---

## ⚖️ Governance, Compliance & Data Licensing

1. **Academic Portfolio Project (`SRS 1.4`):** This repository is developed for educational, technical demonstration, and research purposes. It does not constitute credit underwriting software or financial lending advice.
2. **Fairness Proxies (`FA-01`, `FA-02`):** The Freddie Mac Single-Family Dataset does not contain direct protected demographic variables (race, gender, religion, national origin). Audit analyses rely on single-family loan proxies (`first_time_homebuyer`, `borrower_count`, `occupancy`, `loan_purpose`). Disparities in proxy groups reflect structural economic patterns and should be treated as prompts for institutional inquiry, not unilateral proof of discriminatory intent.
3. **Freddie Mac Data Licensing Terms (`DT-09`):** All loan data is sourced from Freddie Mac under the Single-Family Loan-Level Dataset terms of use. Raw data, interim tables, processed parquet files, and holdout tables (`data/`, `artifacts/`, `reports/`, `DataSet/`) contain proprietary loan-level records and are excluded from Git version control (`.gitignore`). Redistribution of raw records is strictly prohibited under Freddie Mac licensing terms.
