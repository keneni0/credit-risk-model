# Credit Risk Model

A production-grade credit risk scoring system built on the [Xente eCommerce dataset](https://zindi.africa/competitions/xente-fraud-detection-challenge). It uses transaction-level behavioral data to predict the probability of credit default via a proxy label (FraudResult), and exposes predictions through a FastAPI service.

---

## Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml       # GitHub Actions CI pipeline
├── data/
│   ├── raw/                       # Original source data (git-ignored)
│   └── processed/                 # Transformed feature sets (git-ignored)
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering & preprocessing pipeline
│   ├── train.py                   # Model training with MLflow tracking
│   ├── predict.py                 # Inference utilities
│   └── api/
│       ├── main.py                # FastAPI app
│       └── pydantic_models.py     # Request/response schemas
├── tests/
│   └── test_data_processing.py    # Unit tests for the processing pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Credit Scoring Business Understanding

### 1. How does Basel II's emphasis on risk measurement influence the need for an interpretable and well-documented model?

Basel II's Internal Ratings-Based (IRB) approach requires banks to estimate their own Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). Regulators and internal audit functions must be able to validate these estimates, which demands that models produce outputs that can be traced back to specific inputs and business logic — not just a black-box score.

Concretely, this creates three pressures:

- **Explainability**: Supervisors (e.g., the Basel Committee, national regulators) require institutions to demonstrate *why* a borrower received a given risk rating. A Logistic Regression on Weight of Evidence (WoE)-encoded features satisfies this directly: each coefficient has a clear directional interpretation and the contribution of each variable can be quantified.
- **Documentation**: Basel II Pillar 2 mandates rigorous model governance — model cards, validation reports, backtesting results, and documented assumptions. Gradient Boosting models can meet this bar with SHAP values and partial dependence plots, but require significantly more documentation overhead.
- **Auditability**: Any model used for regulatory capital calculation must be reproducible. Version-controlled pipelines, experiment tracking (MLflow), and pinned dependencies all directly serve this requirement.

In short, Basel II doesn't prohibit complex models, but it raises the cost of opacity. Interpretability is a compliance asset, not just an engineering preference.

---

### 2. Without a direct default label, why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?

A direct credit default label requires observing whether a borrower has failed to repay a loan — data that only exists after a lending relationship has matured. In a dataset like Xente's eCommerce transactions, no such outcome is recorded. A **proxy variable** (here, `FraudResult`) is used as a stand-in because it correlates with the financial behavior we care about: customers who commit fraud exhibit payment irregularities that overlap with credit risk signals.

However, proxy-based prediction introduces several material business risks:

| Risk | Description |
|------|-------------|
| **Label noise** | Fraud and credit default are related but distinct constructs. A fraudulent transaction does not equal a defaulted loan; the proxy introduces systematic mislabeling. |
| **Selection bias** | Fraud detection systems already filter some bad actors before they appear in the dataset, creating survivorship bias. The model learns from a censored population. |
| **Distribution shift** | Behavioral patterns that predict fraud may evolve independently of credit risk drivers, causing model degradation over time without an obvious signal. |
| **Regulatory challenge** | Using a proxy label must be disclosed and justified to regulators. A model trained on fraud signals but deployed for credit decisions may be challenged as lacking construct validity. |
| **Disparate impact** | If the proxy variable is correlated with protected characteristics (e.g., geography, product category serving specific demographics), the model may embed indirect discrimination. |

Mitigation strategies include: threshold calibration against actual default rates when labelled data becomes available, continuous population stability monitoring, and documenting the proxy assumption explicitly in the model card.

---

### 3. What are the key trade-offs between Logistic Regression with WoE vs Gradient Boosting in a regulated financial context?

| Dimension | Logistic Regression + WoE | Gradient Boosting (XGBoost/LightGBM) |
|---|---|---|
| **Interpretability** | High — linear combination of WoE-transformed features; coefficients directly map to log-odds | Moderate — requires post-hoc tools (SHAP, LIME); no closed-form explanation |
| **Regulatory acceptance** | Widely accepted by Basel II/III validators; aligns with scorecard tradition | Accepted but requires additional explainability evidence and documentation |
| **Feature engineering** | WoE encoding handles non-linearity, missing values, and monotonicity constraints explicitly | Handles non-linearity automatically; less manual feature work |
| **Performance on imbalanced data** | Requires explicit handling (class weights, resampling) | More robust via ensemble aggregation; built-in `scale_pos_weight` |
| **Overfitting risk** | Low — strong inductive bias towards simple decision boundaries | Higher — requires careful tuning of `max_depth`, `min_child_weight`, regularization |
| **Deployment & scoring speed** | Very fast — a dot product | Slower for large forests; still sub-millisecond for typical scorecard sizes |
| **Model governance overhead** | Lower — easier to audit, validate, and explain to non-technical stakeholders | Higher — SHAP computation, stability testing, and documentation burden increase |
| **Suitability for this project** | Preferred for regulatory-facing production scoring | Best for challenger models and performance benchmarking |

**Recommendation**: Deploy Logistic Regression + WoE as the champion model for regulatory compliance. Use Gradient Boosting as the challenger model in A/B testing to measure the performance ceiling. Track both in MLflow and promote the challenger only after full model validation.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place raw data

```bash
cp /path/to/xente_data.csv data/raw/xente.csv
```

### 3. Train a model

```bash
# Logistic Regression (interpretable, Basel II friendly)
python -m src.train --data data/raw/xente.csv --model lr

# XGBoost (challenger model)
python -m src.train --data data/raw/xente.csv --model xgb
```

### 4. Run the API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at `http://localhost:8000/docs`

### 5. Run with Docker

```bash
docker-compose up --build
```

### 6. Run tests

```bash
pytest tests/ -v --cov=src
```

---

## MLflow Tracking

Start the MLflow UI to compare experiments:

```bash
mlflow ui --port 5000
```

Or use the bundled Docker Compose service: `docker-compose up mlflow`

---

## License

MIT
