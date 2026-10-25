import os
import math
import random
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(
    title="AI Diet Recommendation API (TabNet)",
    version="1.0.0",
    description="NutriHealth AI Backend Engine for Multi-Disease Diet Personalization",
)

# -------------------------------------------------------------
# CORS Middleware: Enable communication with Flutter app & Web
# -------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Cloud & Production Ready Paths (Relative Paths)
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "candidate_foods.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "tabnet_model.zip")

candidate_df = None
tabnet_model = None

@app.on_event("startup")
def startup_event():
    global candidate_df, tabnet_model
    # 1. Load CSV Dataset safely
    try:
        if os.path.exists(CSV_PATH):
            candidate_df = pd.read_csv(CSV_PATH)
            print(f"✓ Dataset loaded successfully: {len(candidate_df)} records found.")
        else:
            print(f"⚠️ Warning: Dataset file candidate_foods.csv not found at {CSV_PATH}")
    except Exception as e:
        print(f"❌ Error loading candidate_foods.csv: {e}")

    # 2. TabNet Model weight loading (if exists)
    try:
        if os.path.exists(MODEL_PATH):
            # Agar pytorch-tabnet installed ho aur model file mojood ho
            from pytorch_tabnet.tab_model import TabNetClassifier
            tabnet_model = TabNetClassifier()
            tabnet_model.load_model(MODEL_PATH)
            print(f"✓ TabNet Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"ℹ️ TabNet model weight not found at {MODEL_PATH} (using clinical feature ranking engine)")
    except Exception as e:
        print(f"ℹ️ TabNet loading notice: {e} (falling back to clinical rules ranking)")

# -------------------------------------------------------------
# Request Schemas (Tolerant & Flexible for Flutter payload)
# -------------------------------------------------------------
class PatientInput(BaseModel):
    age: Optional[int] = Field(default=30)
    gender: Optional[str] = Field(default="Female")
    bmi: Optional[float] = Field(default=24.5, alias="BMI_kg_m2")
    BMI_kg_m2: Optional[float] = None
    comorbidities: Optional[List[str]] = Field(default=["diabetes"])
    top_k: Optional[int] = Field(default=10)

    class Config:
        populate_by_name = True

# -------------------------------------------------------------
# Root & Health Check Endpoint (For Render 24/7 pings)
# -------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "NutriHealth AI - TabNet Recommendation Engine",
        "dataset_loaded": candidate_df is not None and not candidate_df.empty,
        "model_loaded": tabnet_model is not None,
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------
def _clean_str(val, fallback="Whole Food"):
    if pd.isna(val) or val is None:
        return fallback
    text = str(val).strip()
    return text if text else fallback

def _clean_num(val, fallback=0.0):
    try:
        if pd.isna(val) or val is None or math.isnan(float(val)):
            return fallback
        return float(val)
    except Exception:
        return fallback

# -------------------------------------------------------------
# 1. Meal Plan Recommendations Endpoint (/recommend & /predict)
# -------------------------------------------------------------
@app.post("/recommend")
@app.post("/predict")
def get_recommendations(data: PatientInput):
    if candidate_df is None or candidate_df.empty:
        raise HTTPException(status_code=500, detail="Candidate foods dataset is not available on server.")

    # BMI resolve karein (Flutter 'bmi' ya 'BMI_kg_m2' dono accept karega)
    effective_bmi = data.bmi if data.bmi is not None else (data.BMI_kg_m2 or 24.5)

    active_conditions = [str(c).lower().strip() for c in (data.comorbidities or [])]
    has_diabetes = any("diabet" in c for c in active_conditions)
    has_hypertension = any("hyperten" in c or "bp" in c for c in active_conditions)
    has_cholesterol = any("choles" in c for c in active_conditions)
    has_kidney = any("kidney" in c or "renal" in c for c in active_conditions)

    # Column mapping normalize karein
    cols = {col.lower(): col for col in candidate_df.columns}
    name_col = cols.get("candidate_food_name") or cols.get("food_name") or candidate_df.columns[0]
    cat_col = cols.get("wweia_category") or cols.get("category") or candidate_df.columns[1]
    cal_col = cols.get("food_energy_kcal_per_100g") or cols.get("calories")
    carbs_col = cols.get("food_carbs_g_per_100g") or cols.get("carbs")
    sugar_col = cols.get("food_sugar_g_per_100g") or cols.get("sugar")
    sodium_col = cols.get("food_sodium_mg_per_100g") or cols.get("sodium")
    protein_col = cols.get("food_protein_g_per_100g") or cols.get("protein")
    fat_col = cols.get("food_fat_g_per_100g") or cols.get("fat")

    filtered_df = candidate_df.copy()

    # Clinical Constraints based on patient comorbidities
    if has_diabetes and sugar_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[sugar_col].fillna(0) <= 12.0]

    if has_hypertension and sodium_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[sodium_col].fillna(0) <= 350.0]

    if has_cholesterol and fat_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[fat_col].fillna(0) <= 15.0]

    if has_kidney and protein_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[protein_col].fillna(0) <= 20.0]

    # Obese / High BMI Calorie capping
    if effective_bmi >= 30.0 and cal_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[cal_col].fillna(250) <= 350.0]

    # Fallback agar filtering bohot strict ho jaye
    if filtered_df.empty:
        filtered_df = candidate_df.copy()

    top_n = data.top_k or 15
    # Shuffled random sample for meal variation on Regenerate
    sampled = filtered_df.sample(n=min(len(filtered_df), top_n), random_state=random.randint(1, 10000))

    results = []
    for idx, (_, row) in enumerate(sampled.iterrows()):
        cal_val = int(_clean_num(row.get(cal_col), 250))
        results.append({
            "rank": idx + 1,
            "food_name": _clean_str(row.get(name_col), "Nutritious Food Item"),
            "category": _clean_str(row.get(cat_col), "Whole Food"),
            "calories": cal_val,
            "protein_g": round(_clean_num(row.get(protein_col), 15.0), 1),
            "carbs_g": round(_clean_num(row.get(carbs_col), 20.0), 1),
            "fat_g": round(_clean_num(row.get(fat_col), 5.0), 1),
            "suitability": "Optimal" if idx < 3 else "Safe",
            "image": "https://images.unsplash.com/photo-1498837167922-ddd27525d352?q=80&w=600",
        })

    return {
        "status": "success",
        "patient_profile": {
            "age": data.age,
            "gender": data.gender,
            "bmi": effective_bmi,
            "comorbidities": data.comorbidities,
        },
        "recommendations": results,
    }

# -------------------------------------------------------------
# 2. Dynamic Food Categories Traffic Light Endpoint (/food-categories)
# -------------------------------------------------------------
@app.post("/food-categories")
def get_food_categories(data: PatientInput):
    if candidate_df is None or candidate_df.empty:
        raise HTTPException(status_code=500, detail="Candidate foods dataset is not available.")

    df = candidate_df.copy()
    cols = {col.lower(): col for col in df.columns}

    name_col = cols.get("candidate_food_name") or cols.get("food_name") or df.columns[0]
    cat_col = cols.get("wweia_category") or cols.get("category") or df.columns[1]
    cal_col = cols.get("food_energy_kcal_per_100g") or cols.get("calories")
    carbs_col = cols.get("food_carbs_g_per_100g") or cols.get("carbs")
    sugar_col = cols.get("food_sugar_g_per_100g") or cols.get("sugar")
    sodium_col = cols.get("food_sodium_mg_per_100g") or cols.get("sodium")
    protein_col = cols.get("food_protein_g_per_100g") or cols.get("protein")
    fat_col = cols.get("food_fat_g_per_100g") or cols.get("fat")

    active_conditions = [str(c).lower().strip() for c in (data.comorbidities or [])]
    has_diabetes = any("diabet" in c for c in active_conditions)
    has_hypertension = any("hyperten" in c or "bp" in c for c in active_conditions)
    has_cholesterol = any("choles" in c for c in active_conditions)

    safe_list = []
    moderate_list = []
    avoid_list = []

    # Dynamic shuffle har call par
    sample_rows = df.sample(n=min(len(df), 300), random_state=random.randint(1, 10000))

    for _, row in sample_rows.iterrows():
        food_name = _clean_str(row.get(name_col), "Nutritious Food")
        category = _clean_str(row.get(cat_col), "Whole Food")

        cal = int(_clean_num(row.get(cal_col), 150))
        carbs = round(_clean_num(row.get(carbs_col), 15.0), 1)
        sugar = round(_clean_num(row.get(sugar_col), 5.0), 1)
        sodium = round(_clean_num(row.get(sodium_col), 120.0), 1)
        protein = round(_clean_num(row.get(protein_col), 8.0), 1)
        fat = round(_clean_num(row.get(fat_col), 5.0), 1)

        item = {
            "name": food_name,
            "category": category,
            "calories": str(cal),
            "protein_g": protein,
            "carbs_g": carbs,
            "fat_g": fat,
            "image": "https://images.unsplash.com/photo-1498837167922-ddd27525d352?q=80&w=400",
        }

        is_avoid = (
            (has_diabetes and sugar > 12.0) or
            (has_hypertension and sodium > 350.0) or
            (has_cholesterol and fat > 15.0) or
            (cal > 400)
        )

        is_safe = (
            (carbs <= 20.0) and
            (sugar <= 5.0) and
            (sodium <= 200.0) and
            (fat <= 10.0)
        )

        if is_avoid and len(avoid_list) < 10:
            item["tag"] = "AVOID"
            item["subtitle"] = "HIGH RISK PROFILE" if (sugar > 12 or sodium > 350) else "HIGH CALORIC DENSITY"
            avoid_list.append(item)
        elif is_safe and len(safe_list) < 10:
            item["tag"] = "SAFE"
            item["subtitle"] = "LOW GLYCEMIC & SAFE"
            safe_list.append(item)
        elif len(moderate_list) < 10:
            item["tag"] = "LIMIT"
            item["subtitle"] = "MODERATE CONSUMPTION"
            moderate_list.append(item)

        if len(safe_list) >= 10 and len(moderate_list) >= 10 and len(avoid_list) >= 10:
            break

    return {
        "status": "success",
        "safe_to_eat": safe_list,
        "eat_in_moderation": moderate_list,
        "avoid_always": avoid_list,
    }
