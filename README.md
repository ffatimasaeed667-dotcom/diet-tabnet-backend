# NutriHealth AI - TabNet Backend Recommendation Engine

FastAPI-powered machine learning backend designed for comorbid dietary decision-making. The system serves personalized clinical food recommendations and categorical traffic-light food groupings based on user biometric data (BMI, age, gender) and chronic health conditions.

## Key Features
- **Deep Tabular Learning (TabNet):** Uses PyTorch TabNet architecture for interpretable tabular deep learning on nutritional and clinical profiles.
- **Dynamic Multi-Disease Filtering:** Evaluates constraints dynamically for diabetes (glycemic and sugar limits), hypertension (sodium capping), hyperlipidemia (lipid/fat constraints), and renal conditions.
- **Persistent State Sync:** Serves endpoints structured to sync with Flutter client and Firebase Firestore logs.
- **Fallback Engine:** Implements continuous service fallbacks via clinical heuristic filtering in case model artifacts are missing.

## Core API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service health status and dataset availability check |
| `GET` | `/health` | Cloud uptime ping endpoint |
| `POST` | `/recommend` / `/predict` | Generates personalized ranked meal recommendations |
| `POST` | `/food-categories` | Evaluates Safe, Moderate, and Avoid food classifications |


## Tech Stack
- **Framework:** FastAPI & Uvicorn
- **ML / Deep Learning:** PyTorch, PyTorch-TabNet, Scikit-learn
- **Data Engineering:** Pandas, NumPy
- **Deployment Target:** Render / Docker-compatible Cloud Runtimes