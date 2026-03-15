# Healthcare Risk Prediction System

**Integrated Big Data Analytics and AI System for Smart Healthcare Risk Prediction**

Predicts 30-day hospital readmission risk for diabetic patients using XGBoost with SHAP explanations.

---

## Stack

| Layer | Tool |
|---|---|
| API | FastAPI + uvicorn |
| Frontend | Streamlit |
| ML | XGBoost + SHAP |
| Database | SQLite (dev) |
| Auth | JWT (python-jose + passlib) |
| Containers | Docker Compose |

---

## Quick start (local, no Docker)

### 1. Train the model first

```bash
cd ml_pipeline
pip install -r requirements.txt

# Put your dataset CSV in data/train.csv (tab-separated)
python train.py ../data/train.csv
# Outputs: ml_pipeline/models/model.pkl + preprocessor.pkl
```

### 2. Start the backend

```bash
cd backend
pip install -r requirements.txt

cp ../.env.example ../.env
# Edit .env — at minimum change SECRET_KEY

uvicorn main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 3. Start the frontend

```bash
cd frontend
pip install -r requirements.txt

streamlit run app.py
# Opens: http://localhost:8501
```

---

## Quick start (Docker)

```bash
# 1. Train model locally first (Docker doesn't do training)
cd ml_pipeline && pip install -r requirements.txt
python train.py ../data/train.csv
cd ..

# Edit SECRET_KEY in .env

# 2. Build and run
docker-compose up --build
```

- Frontend: http://localhost:8501
- API docs: http://localhost:8000/docs

---

## Project structure

```
healthcare-risk-prediction/
├── backend/
│   ├── main.py               FastAPI app
│   ├── config.py             Settings from .env
│   ├── database.py           SQLAlchemy engine + session
│   ├── routes/
│   │   ├── auth.py           POST /auth/login, /auth/register
│   │   ├── predict.py        POST /predict, /predict/batch
│   │   └── patients.py       GET /patients/history, /stats
|   |   |-- deps.py
│   ├── services/
│   │   ├── prediction_service.py  Load model, preprocess, SHAP
│   │   ├── patient_service.py     DB CRUD
│   │   └── auth_service.py        JWT, bcrypt
│   └── models/
│       ├── orm.py            SQLAlchemy: User, Patient, Prediction
│       └── schemas.py        Pydantic request/response models
│
├── ml_pipeline/
│   ├── preprocess.py         Scaling, SMOTE, train/test split
│   ├── train.py              XGBoost, stratified 5-fold CV
│   ├── evaluate.py           Macro F1, ROC-AUC, confusion matrix
│   ├── explainability.py     SHAP TreeExplainer
│   └── models/               model.pkl + preprocessor.pkl (gitignored)
│
├── frontend/
│   ├── app.py                Streamlit entrypoint + auth
│   ├── api_client.py         httpx wrapper for FastAPI
│   └── pages/
│       ├── 01_predict.py     Single patient prediction + SHAP chart
│       ├── 02_dashboard.py   History, stats, charts
│       └── 03_upload.py      CSV bulk prediction + download
│
├── data/                     Raw CSVs — gitignored
├── docker-compose.yml
├── .env
└── README.md
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/register` | Create account |
| POST | `/auth/login` | Get JWT token |
| POST | `/predict` | Single prediction + SHAP |
| POST | `/predict/batch` | Batch prediction from list |
| GET | `/patients/history` | Prediction history |
| GET | `/patients/stats` | Class distribution stats |
| GET | `/health` | API + model status |

Full interactive docs at `http://localhost:8000/docs`

---

## Prediction classes

| Class | Meaning |
|---|---|
| `NO` | Not readmitted |
| `<30` | Readmitted within 30 days (high risk) |
| `>30` | Readmitted after 30 days (moderate risk) |

---

## Switching to PostgreSQL

Change one line in `.env`:

```
DATABASE_URL=postgresql://user:password@localhost:5432/healthcare
```

No code changes needed — SQLAlchemy handles the rest.