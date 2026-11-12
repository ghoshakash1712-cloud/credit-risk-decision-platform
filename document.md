# Credit Risk Decision Platform — Teacher Presentation Guide

## 1. Project in one minute

This project is a **mortgage credit-risk decision-support platform** built with the Freddie Mac Single-Family Loan-Level Dataset. It predicts the probability that an already-originated loan will become 90+ days delinquent or experience a credit event within 24 months of its first payment.

It is more than a classifier. It connects the model to three decision-governance questions:

1. **Business decision:** At what risk probability should a lender reject a loan, considering the cost of approving a future defaulter and the opportunity cost of rejecting a good borrower?
2. **Fairness audit:** Do approval and error outcomes differ across meaningful loan-level proxy groups?
3. **Drift monitoring:** Does the current loan population differ enough from the training population that the model should be reviewed or retrained?

The platform is an academic demonstration only. It is **not** intended to make real lending decisions.

### Suggested opening statement

> “My project converts mortgage-default prediction into a decision-support system. It predicts default risk using only origination-time information, chooses a transparent cost-based approval threshold, audits group-level outcomes, and monitors population drift. The user interacts through a Streamlit dashboard, while a Flask API serves a trained LightGBM model and precomputed reports.”

---

## 2. Problem being solved

A normal machine-learning project may report only accuracy or ROC-AUC. That is incomplete for lending because:

- approving a borrower who later defaults can be costly;
- rejecting a borrower who would repay also has a cost because the lender loses interest income;
- a model can perform well overall while producing different outcomes for different groups;
- a model trained on older loans may become unreliable when interest rates, borrower profiles, or loan characteristics shift.

This project therefore combines **prediction, business thresholding, fairness auditing, and drift detection**.

---

## 3. End-to-end architecture

```text
Freddie Mac raw loan files
        |
        v
Offline data and training pipeline
  ingest -> label/features -> train -> holdout/reports
        |
        v
Artifacts folder (model + predictions + reports)
        |
        +-------------------+
        |                   |
        v                   v
Flask REST API         Streamlit dashboard
  reads artifacts       calls only the API
  serves /v1 endpoints  shows five interactive tabs
```

### Important design principle

The offline pipeline performs expensive work: reading loan files, creating labels, training LightGBM, calculating reports, and simulating shocks. The online API does **not** retrain the model. It reads verified artifacts and answers requests quickly. The Streamlit dashboard is a **thin client**: it does not load the model or read loan files directly.

This separation makes the system easier to test, safer to run, and easier to explain.

---

## 4. Technology stack and each component's role

| Component | Technology | Role |
|---|---|---|
| Data processing | Pandas, PyArrow | Reads, cleans, labels, and stores loan data as Parquet |
| ML model | LightGBM | Predicts the 24-month default probability |
| Fairness baseline | Fairlearn | Independently verifies group metrics at the default threshold |
| Drift detection | NumPy/Pandas PSI implementation | Compares current and reference distributions |
| Shared business logic | `src/common/` | Feature engineering, cost function, fairness groups, PSI, configuration |
| API | Flask | Exposes read-only JSON endpoints under `/v1` |
| UI | Streamlit + Altair | Interactive dashboard and charts |
| Deployment | Docker Compose | Runs API and dashboard as separate containers |
| Environment | Conda | Reproducible local environment defined in `environment.yml` |
| Testing | Pytest | Unit and integration tests for the mathematical and API contracts |

---

## 5. Data pipeline: how raw data becomes a model

### 5.1 Input data

The system uses Freddie Mac sample files for vintages 2015–2021. For every year, it expects:

- one origination file: borrower and loan characteristics available at underwriting;
- one monthly performance file: used only to construct the future default label.

Files are placed in `data/raw/` as `sample_YYYY.zip`. If using the fuller `DataSet/` source files, `prepare_raw.py` creates the sampled archives first.

### 5.2 Data validation and cleaning

`src/pipeline/ingest_data.py` validates that:

- every required vintage exists;
- origination and performance schemas have the expected number of columns;
- loan IDs are unique in each origination file;
- enough performance history exists to observe a 24-month label window.

Known “not available” codes such as FICO `9999` or DTI `999` become null values. LightGBM can handle missing values, so the project does not invent median values that may distort risk.

### 5.3 Cohort and label

Some loans are excluded: HARP/refinance-relief loans, loans with zero-balance defect code `96`, and loans with no performance history.

The target is `default_24m`:

```text
1 = the loan reaches 90+ days delinquent, reaches REO status,
    or has a credit-event zero-balance code within 24 months
    of the first payment date.

0 = otherwise.
```

The project deliberately calculates the window from dates, not the loan-age field, because loan age can reset after a modification.

### 5.4 Features and leakage control

The model uses only information that would be known at loan origination. Examples:

- numeric: FICO, original LTV, CLTV, DTI, UPB, interest rate, term, number of borrowers;
- categorical: occupancy, channel, state, property type, loan purpose, seller, and program indicators;
- derived: `cltv_minus_ltv` and `rate_vs_quarter_median`.

Performance data is used to build the label only. It is never used as a feature. This is crucial because using future payment behaviour during training would be **data leakage**.

### 5.5 Temporal splits

| Split | Vintages | Purpose |
|---|---:|---|
| Train | 2015–2017 | Fits the model and cross-validation tuning |
| Validation | 2018 | Early stopping and default threshold optimisation |
| Out-of-time (OOT) | 2019 | Honest later-period test |
| Stress | 2020–2021 | Tests unusual COVID-era/rate-environment behaviour |

The split is by time, not random. This better resembles the real situation: train on the past and evaluate on newer loans.

---

## 6. Model and explainability

### Model choice

The project uses a **LightGBM binary classifier**. LightGBM is suitable because it handles nonlinear relationships, missing values, and categorical features efficiently.

The model is trained without class weighting or oversampling. That helps preserve the interpretation of the output as an estimated default probability. Instead of changing the model probabilities, the project changes the **business decision threshold**.

### Evaluation

The pipeline writes metrics for validation, OOT, and stress data:

- ROC-AUC and Gini;
- PR-AUC;
- Brier score and log loss;
- KS statistic;
- decile and calibration tables.

The model run also has an OOT sanity gate. A very low OOT AUC causes the training run to fail, helping catch broken artifacts or an incorrect pipeline run.

### Explainability

For a selected stored loan, the API asks LightGBM for `pred_contrib=True`. This provides TreeSHAP-style feature contributions:

- positive contribution: feature raises risk;
- negative contribution: feature lowers risk;
- contributions are in **log-odds**, not percentage points;
- the dashboard shows the ten largest contributions and an “other features” value so they reconcile to the final score.

### Teacher-friendly explanation

> “The system does not only say a loan has, for example, 1.6% predicted default risk. It also shows the main features that pushed that prediction up or down. This makes the decision more interpretable than a black-box probability alone.”

---

## 7. Business threshold optimiser

The model probability alone does not decide approve or reject. The decision depends on the business cost of two mistakes.

| Error | Meaning | Cost parameter |
|---|---|---|
| False positive | Rejecting a loan that would repay | `g`: lost margin, % of original balance |
| False negative | Approving a loan that later defaults | `L`: expected loss severity, % of original balance |

For each candidate probability threshold, the system calculates:

```text
net benefit = L × balance of correctly rejected defaulters
              - g × balance of incorrectly rejected good loans
```

It chooses the threshold with the highest net benefit. The system reports money in USD, approval/rejection rate, default capture rate, and the result on validation, OOT, and stress splits.

### Why this is better than a fixed 0.50 threshold

Mortgage default is uncommon, so 0.50 is usually not meaningful. A low but economically justified threshold can be better when default losses are much larger than lost interest margin.

The dashboard also calculates an analytical reference point:

```text
expected threshold ≈ g / (g + L)
```

If the selected threshold differs dramatically from this reference, the dashboard can warn that model calibration deserves review.

### What to demonstrate

1. Open **Business Optimizer**.
2. Point out the false-positive and false-negative cost sliders in the sidebar.
3. Change one slider and show that threshold, net benefit, rejection rate, and the three-split comparison update.
4. Switch from **Optimal** to **Manual** and enter a threshold to show that policy choices can be examined transparently.

Suggested wording:

> “This screen converts ML probabilities into a business decision. It does not assume every error has the same cost; it uses original loan balance and selected cost assumptions.”

---

## 8. Fairness and bias audit

The dataset has no direct protected attributes such as race or sex. The project therefore audits four **proxy attributes**:

- first-time homebuyer;
- number of borrowers;
- occupancy type;
- loan purpose.

These proxies are not proof of discrimination. They are a way to identify patterns that require investigation.

### Metrics shown per group

- number of loans and defaults;
- observed default rate and mean predicted probability;
- rejection and approval rate;
- default capture rate: fraction of actual defaulters rejected;
- false rejection rate: fraction of good borrowers rejected;
- 95% Wilson confidence intervals for rates.

The overall audit flags:

| Metric | Meaning | Initial flag level |
|---|---|---|
| Disparate impact ratio | Lowest approval rate / highest approval rate | Below 0.80 |
| Equalised odds difference | Largest gap in capture or false-rejection rate | Above 0.10 |
| Calibration ratio | Observed default rate / average predicted probability | Outside 0.80–1.25 |

Small groups are displayed but excluded from flags unless they have at least 500 loans and 30 defaults. Unknown groups are also excluded from gap calculations.

### Important presentation point

> “This is an audit, not an automated mitigation tool. The project does not set different thresholds for groups. A flag tells a lender to investigate; it does not prove unfair treatment.”

### Current checkout note

The API returns the fairness response successfully, but the current Fairness dashboard tab has a frontend schema mismatch: it expects flat fields such as `default_capture_rate_ci_lower`, while the API returns `default_capture_rate: {value, ci_lower, ci_upper}`. This causes the visible `KeyError` after `/v1/fairness` returns HTTP 200. Fix the tab before using the Fairness screen in a live assessment, or explain it honestly as a known UI integration defect. The underlying audit calculation and API endpoint are running.

---

## 9. Drift monitoring and stress simulation

Drift means the loans being scored no longer look like the loans used to train the model. The platform measures drift with **Population Stability Index (PSI)**:

```text
PSI = sum over bins of (current share - reference share)
      × ln(current share / reference share)
```

The project compares feature and score distributions using fixed reference bins. Numeric features use deciles plus a missing-value bin. Categorical features include known categories plus unseen and missing bins.

| PSI | Status | Meaning |
|---:|---|---|
| Below 0.10 | Stable | No action |
| 0.10–0.20 | Warning | Monitor more closely |
| At or above 0.20 | Retrain recommended | Population has materially shifted |

The overall status uses score PSI and important feature PSI values. `property_valuation_method` is treated as structural: it is shown but does not drive the overall alert because its availability changes by design across Freddie Mac vintages.

### Simulated shocks

The project also creates twelve precomputed “what-if” scenarios: four shocks × three severities.

| Scenario | Example severe shock |
|---|---|
| Debt stress | DTI +10 points, capped at 65 |
| Credit deterioration | FICO −50 points, bounded 300–850 |
| Rate shock | Original rate +2.0 points |
| Leverage squeeze | LTV and CLTV +10 points, capped at 100 |

These scenarios change features, rescore the OOT set, and measure PSI. They do not produce actual outcomes, so they do not claim a new AUC or profit figure.

### What to demonstrate

1. Open **Drift Monitor**.
2. Explain the green/amber/red banner and PSI thresholds.
3. Show OOT or stress drift, then select a simulated severe shock.
4. Explain that “retrain recommended” is informational; it does not automatically retrain the model.

---

## 10. API and dashboard

### API responsibilities

The Flask API is read-only and serves `/v1` endpoints. It includes the model version and data-release number in every successful response.

| Endpoint group | Examples | Purpose |
|---|---|---|
| Health and metadata | `/health`, `/meta` | Confirms readiness and supplies dashboard settings |
| Loan exploration | `/loans`, `/loans/<id>` | Lists and retrieves stored holdout loans |
| Scoring/explanation | `/predict/<id>`, `/explain/<id>` | Returns a live score and local TreeSHAP explanation |
| Business | `/optimize` | Calculates cost-based threshold results |
| Governance | `/fairness`, `/drift`, `/model-report` | Returns audits, PSI reports, and model metrics |

Only validation, OOT, and stress loans are served. Training loans are intentionally not exposed through the API.

### Startup safety check

When the API starts, it loads the model and rescoring data, samples 100 loans, and verifies the new probabilities match stored training-time probabilities within `1e-6`. If artifacts do not match, the API refuses to start. This prevents the API from silently serving a mismatched model or dataset.

### Dashboard tabs

| Tab | What it demonstrates |
|---|---|
| Loan Lookup | Probability, approve/reject decision, actual historical outcome, local explanation |
| Business Optimizer | Cost-sensitive threshold and net-benefit trade-off |
| Fairness & Bias Audit | Group-level metrics, confidence intervals, flags, and caveats |
| Drift Monitor | Real and simulated population shift using PSI |
| Model & Data | Performance metrics, calibration, global importance, and data governance |

---

## 11. How to run the project

Run all commands from the repository root.

### A. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate credit-risk-decision-platform
```

For an existing environment after dependency changes:

```bash
conda env update -f environment.yml --prune
```

### B. Full data rebuild (only when raw data is available)

1. Place prepared archives `sample_2015.zip` to `sample_2021.zip` in `data/raw/`.
2. Or place full source files in `DataSet/` and run:

```bash
python -m src.pipeline.prepare_raw
```

3. Rebuild all derived data and artifacts:

```bash
make all
```

`make all` runs ingest, dataset construction, training, holdout generation, fairness baseline generation, drift reporting, and the simulated-shock grid.

### C. Run the interactive application

Terminal 1 — keep this running:

```bash
conda activate credit-risk-decision-platform
python -m api.app
```

Terminal 2:

```bash
conda activate credit-risk-decision-platform
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in a browser. The local API health endpoint is `http://127.0.0.1:8000/v1/health`.

### D. Docker alternative

If the artifacts already exist:

```bash
docker compose up --build
```

This starts the API on port 8000 and dashboard on port 8501.

### E. Tests

After the generated `data/processed/` and `artifacts/` files are present:

```bash
make test
```

or:

```bash
pytest -v
```

---

## 12. Recommended 5–7 minute teacher demonstration

### 0:00–0:45 — introduce the problem

Say:

> “A credit-risk score alone is not enough. A lender must decide how to use the score, check whether decisions differ across groups, and detect whether the borrower population has changed. This platform covers all four parts.”

Show the dashboard and point to the model version and data release in the sidebar.

### 0:45–2:00 — show one loan and explainability

1. In **Loan Lookup**, select a validation loan.
2. Explain probability, actual 24-month outcome, and decision threshold.
3. Scroll to feature contributions.

Say:

> “The model's probability is based only on information available at origination. These contributions explain which features increased or reduced risk for this individual stored loan.”

### 2:00–3:30 — show business impact

1. Open **Business Optimizer**.
2. Explain the false-positive versus false-negative cost sliders.
3. Adjust a cost and show the optimal threshold and net benefit change.
4. Point to the OOT and stress results.

Say:

> “The threshold is selected from business costs, not chosen arbitrarily. I also check its outcome on later data and stress-period data.”

### 3:30–4:45 — show fairness

If the dashboard schema issue is fixed, select a fairness attribute and explain rejection rate, capture rate, false rejection, and the caution note. If it is not fixed, state:

> “The API calculates and returns the fairness audit correctly; the current UI has a field-mapping bug which is documented as a pending frontend fix. The design uses proxy groups and does not make claims about protected classes.”

### 4:45–5:45 — show drift

1. Open **Drift Monitor**.
2. Explain PSI bands.
3. Select a severe simulated scenario.

Say:

> “This answers a separate governance question: even if the model was accurate when trained, are today's loans still similar enough to trust the model?”

### 5:45–7:00 — finish with engineering quality

Mention temporal splitting, leakage prevention, API startup self-check, separate artifacts, tests, and data licensing. End with the limitations below.

---

## 13. Requirement-to-implementation traceability

| SRS area | How this repository implements it |
|---|---|
| DP — Data pipeline | `src/pipeline/ingest_data.py`, `build_dataset.py`, `prepare_raw.py` and `src/common/features.py` |
| MD — Model/scoring | `src/pipeline/train.py`, `make_holdout.py`, model/report artifacts, `api/loaders.py` |
| OP — Optimiser | shared `src/common/cost.py`, exposed by `/v1/optimize`, displayed in `dashboard/tabs/optimizer.py` |
| FA — Fairness | group mapping and Wilson CIs in `src/common/groups.py`, baseline in `pipeline/fairness.py`, API `/v1/fairness` |
| DR — Drift | PSI implementation in `src/common/psi.py`, reports in `pipeline/drift.py`, shocks in `simulate_shift.py` |
| AP — API | Flask app in `api/`, strict parameter parsing, JSON errors, request IDs, cache headers |
| UI — Dashboard | `dashboard/app.py`, one sidebar and five tabs |
| DT — Deployment/testing | Dockerfiles, `docker-compose.yml`, Makefile, `tests/unit/`, `tests/integration/` |

---

## 14. Likely teacher questions and strong answers

### Why LightGBM?

It is a strong tabular-data model. It handles nonlinear relationships, missing values, and categorical fields efficiently, which suits mortgage origination data.

### Why not use accuracy?

Default is relatively rare, so a model can appear accurate by predicting “non-default” for most loans. The project uses discrimination, calibration, cost-based decision metrics, and split-wise evaluation instead.

### Why use a time split rather than random train/test split?

The real deployment scenario predicts future loans using older loans. A random split can make the test set too similar to training data and overstate performance.

### What is data leakage here?

Using a value only known after origination—such as payment status or later delinquency—as an input feature. The project restricts features to underwriting-time fields and uses performance data only to construct the label.

### Why is the threshold not 0.5?

The right decision threshold depends on the relative economic costs of a missed default and a wrongly rejected good loan. The optimiser finds the threshold that maximises net benefit for stated assumptions.

### Is fairness guaranteed if no flag appears?

No. A flag is only a screening signal, and no flag is not proof of fairness. The audit uses proxy attributes rather than direct protected characteristics and is limited by the data.

### What does a high PSI mean?

It means the distribution of loan characteristics or model scores has shifted relative to the reference data. It is a monitoring signal, not direct proof that model performance has declined.

### Why is 2020–21 treated as a stress set?

Its rate environment and pandemic-era servicing conditions differ from earlier vintages, making it useful for checking temporal robustness and drift.

### Why keep artifacts separate from the API?

The pipeline generates reproducible model and report files offline. The API loads those files read-only, which keeps serving fast and avoids retraining or data mutation during user requests.

---

## 15. Limitations to state clearly

- The dataset contains loans that Freddie Mac accepted; rejected applicants are not observed, creating selection bias.
- The model uses proxy groups, not direct protected attributes. It cannot establish legal fairness or discriminatory intent.
- The cost assumptions are illustrative; especially the false-positive cost is configurable rather than estimated from a lender's books.
- The label is a 24-month outcome, so performance monitoring of genuinely new loans must wait for outcomes to mature.
- The stress period is unusual because of pandemic forbearance and sharp rate changes.
- The project has no login, automated retraining, model registry, experiment tracking, or new-loan data-entry scoring in version 1.
- Freddie Mac raw and processed loan data is excluded from Git and should not be redistributed.
- The current dashboard has a known Fairness-tab response-schema issue described in Section 8; the API calculation remains available.

---

## 16. Final closing statement

> “The main contribution of this project is not only predicting mortgage default. It demonstrates how a predictive model can be made useful and accountable: decisions are connected to business cost, explained at loan level, audited across proxy groups, monitored for drift, tested, and served through a clear API/dashboard architecture.”
