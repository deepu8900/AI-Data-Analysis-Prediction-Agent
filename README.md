# ML Prediction Service

AI-powered risk prediction with drift detection, SHAP explainability, and natural language querying.

## Project Structure

```
ml-simple/
├── main.py           ← entire backend (FastAPI + XGBoost + SHAP + Drift + LangChain)
├── index.html        ← entire frontend (dashboard UI)
└── requirements.txt  ← all dependencies
```

That's it. No subfolders, no config files, no build tools.

## Stack

| What | Tech |
|------|------|
| Web framework | FastAPI + Uvicorn |
| ML model | XGBoost |
| Explainability | SHAP (TreeExplainer) |
| Drift detection | PSI + KS-test (SciPy) |
| NL querying | LangChain + GPT-4 (optional) |
| Frontend | Plain HTML + Chart.js |

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Add OpenAI key for real NL queries
Create a `.env` file in the same folder:
```
OPENAI_API_KEY=sk-...
```
Without it, the NL Query page still works using built-in rule-based responses.

### 3. Run
```bash
uvicorn main:app --reload
```

Then open **http://localhost:8000** in your browser.

## Features

| Page | What it does |
|------|-------------|
| Overview | 30-day accuracy & F1 trend charts, key model metrics |
| Predict | Input features → XGBoost risk score + confidence |
| SHAP Explorer | Feature contribution waterfall chart + breakdown table |
| Drift Monitor | PSI & KS-test scores per feature, 14-day history chart |
| NL Query | Ask questions in plain English, get answers + generated SQL |

## API Endpoints

All endpoints are also available directly at `http://localhost:8000/docs`.

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Serves the dashboard (index.html) |
| `POST` | `/predict` | Run XGBoost prediction |
| `POST` | `/shap` | Get SHAP values for a feature set |
| `GET` | `/shap/{id}` | Get SHAP values by prediction ID |
| `GET` | `/drift` | Run drift detection report |
| `POST` | `/query` | Natural language query |
| `GET` | `/metrics` | Model performance metrics |
| `GET` | `/health` | Health check |

## How main.py is organized

The entire backend lives in one file, split into clearly labeled sections:

```
# ══════ SCHEMAS        — Pydantic request/response models
# ══════ ML SERVICE     — XGBoost training + SHAP inference
# ══════ DRIFT SERVICE  — PSI + KS-test drift detection
# ══════ NL QUERY       — LangChain or rule-based fallback
# ══════ METRICS        — 30-day performance history
# ══════ FASTAPI APP    — app setup + all routes
```

## Connecting a Real Model

Replace the `_train()` method in the `MLService` class inside `main.py`:

```python
import joblib

def _train(self):
    self.model     = joblib.load("your_model.pkl")
    self.explainer = shap.TreeExplainer(self.model)
```
