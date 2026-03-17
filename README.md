# ML Prediction Service — FastAPI Backend

Replaces the Node.js/Express proxy entirely. The React frontend talks directly to FastAPI.

## Architecture

```
React (port 5173)
     │  /api/* → proxy strips /api prefix
     ▼
FastAPI (port 8000)
     ├── POST /predict        XGBoost inference
     ├── POST /shap           SHAP TreeExplainer values
     ├── GET  /shap/:id       SHAP by prediction ID
     ├── GET  /drift          PSI + KS drift report
     ├── POST /query          LangChain NL query
     ├── GET  /metrics        Model performance metrics
     └── GET  /health         Health check
```

## Stack

| Component | Tech |
|-----------|------|
| Web framework | FastAPI 0.110 + Uvicorn |
| ML model | XGBoost 2.0 |
| Explainability | SHAP (TreeExplainer) |
| Drift detection | PSI + KS-test (SciPy) |
| NL querying | LangChain + GPT-4 (optional) |
| Validation | Pydantic v2 |

## Project Structure

```
ml-fastapi/
├── main.py                  # FastAPI app, CORS, router registration
├── requirements.txt
├── .env.example
├── app/
│   ├── schemas.py           # Pydantic request/response models
│   ├── ml_service.py        # XGBoost training + SHAP inference
│   ├── drift_service.py     # PSI + KS drift detection
│   ├── nl_service.py        # LangChain NL query handler
│   └── metrics_service.py   # Model metrics + 30-day history
└── routers/
    ├── predict.py
    ├── shap.py
    ├── drift.py
    ├── query.py
    └── metrics.py
```

## Setup

### 1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment (optional)
```bash
cp .env.example .env
# Add OPENAI_API_KEY to enable real LangChain NL queries
```

### 4. Start FastAPI
```bash
uvicorn main:app --reload --port 8000
```

The server trains an XGBoost model on synthetic data at startup (~3 seconds).

### 5. Start React frontend (separate terminal)
```bash
# In ml-dashboard/
npm install && npm run dev
```

## API Explorer

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Connecting to a Real Model

Replace the `_train()` method in `app/ml_service.py`:
```python
import joblib
self.model = joblib.load("models/xgb_production.pkl")
self.explainer = shap.TreeExplainer(self.model)
```

## NL Query (LangChain)

Without `OPENAI_API_KEY`, the service uses smart rule-based responses that cover:
- Prediction distribution queries
- SHAP / feature importance questions
- Drift alert queries
- Accuracy / performance metrics
- Latency stats
- Confidence breakdowns

Set `OPENAI_API_KEY` to enable full GPT-4 powered SQL generation.
