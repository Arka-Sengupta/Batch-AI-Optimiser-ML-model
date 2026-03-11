from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="Batch Optimization API", version="1.0")

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models on startup ────────────────────────────────────
model_quality = joblib.load('models/model_quality_final.pkl')
model_energy  = joblib.load('models/model_energy_final.pkl')
model_carbon  = joblib.load('models/model_carbon_final.pkl')
scaler        = joblib.load('models/scaler.pkl')
FEATURES      = joblib.load('models/features.pkl')
pareto_df     = pd.read_csv('golden_signatures.csv')

# ── Sensor defaults for demo ──────────────────────────────────
SENSOR_DEFAULTS = {
    'Temperature_C_mean'        : 35.2,
    'Temperature_C_max'         : 67.8,
    'Temperature_C_min'         : 21.3,
    'Temperature_C_std'         : 12.8,
    'Pressure_Bar_mean'         : 0.98,
    'Motor_Speed_RPM_max'       : 880.0,
    'Compression_Force_kN_max'  : 14.0,
    'Flow_Rate_LPM_mean'        : 1.65,
    'Power_Consumption_kW_max'  : 60.0,
    'Vibration_mm_s_mean'       : 3.0,
    'Vibration_mm_s_max'        : 9.8,
    'Vibration_mm_s_std'        : 2.4,
}

# ── Request models ────────────────────────────────────────────
class UserInput(BaseModel):
    batch_id        : str
    operator_name   : str
    machine_settings: dict
    material_recipe : dict
    environment     : dict

class SensorInput(BaseModel):
    batch_id            : str
    temperature_sensors : dict
    pressure_sensors    : dict
    motor_sensors       : dict
    compression_sensors : dict
    flow_sensors        : dict
    power_sensors       : dict
    vibration_sensors   : dict

class PredictRequest(BaseModel):
    user_json   : UserInput
    sensor_json : Optional[SensorInput] = None

# ── Helper functions ──────────────────────────────────────────
def get_status(value, target, higher_is_better=True):
    if higher_is_better:
        if value >= target * 0.98:   return "OPTIMAL"
        elif value >= target * 0.90: return "GOOD"
        else:                        return "NEEDS IMPROVEMENT"
    else:
        if value <= target * 1.02:   return "OPTIMAL"
        elif value <= target * 1.10: return "ACCEPTABLE"
        else:                        return "HIGH"

# ── Main prediction logic ─────────────────────────────────────
def predict_batch(user_json, sensor_json=None):
    features = {}
    features.update(user_json.get('machine_settings', {}))
    features.update(user_json.get('material_recipe',  {}))
    features.update(user_json.get('environment',      {}))

    if sensor_json:
        features.update(sensor_json.get('temperature_sensors', {}))
        features.update(sensor_json.get('pressure_sensors',    {}))
        features.update(sensor_json.get('motor_sensors',       {}))
        features.update(sensor_json.get('compression_sensors', {}))
        features.update(sensor_json.get('flow_sensors',        {}))
        features.update(sensor_json.get('power_sensors',       {}))
        features.update(sensor_json.get('vibration_sensors',   {}))
    else:
        features.update(SENSOR_DEFAULTS)

    # Build input array in correct feature order
    input_array  = np.array([features[f] for f in FEATURES]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # Predict
    quality = float(model_quality.predict(input_scaled)[0])
    energy  = float(model_energy.predict(input_scaled)[0])
    carbon  = float(model_carbon.predict(input_scaled)[0])

    # Best Golden Signature
    best_gs = pareto_df.iloc[0]

    # Recommendations with 15% max change cap
    MAX_CHANGE_PCT = 0.15
    top_features   = ['Drying_Temp','Binder_Amount',
                      'Granulation_Time','Compression_Force','Machine_Speed']
    recommendations = []

    for feat in top_features:
        if feat in features:
            current     = features[feat]
            recommended = float(best_gs[feat])
            change      = recommended - current
            max_change  = abs(current) * MAX_CHANGE_PCT
            if abs(change) > max_change:
                change      = (1 if change > 0 else -1) * max_change
                recommended = current + change
            if abs(change) > 0.01:
                recommendations.append({
                    'parameter'   : feat,
                    'current'     : round(current, 2),
                    'recommended' : round(recommended, 2),
                    'change'      : round(change, 2),
                    'direction'   : 'increase' if change > 0 else 'decrease'
                })

    return {
        'batch_id'   : user_json.get('batch_id', 'NEW'),
        'predictions': {
            'Quality_Score': round(quality, 3),
            'Energy_kWh'   : round(energy,  2),
            'Carbon_kg'    : round(carbon,  3),
        },
        'status': {
            'Quality': get_status(quality, best_gs['Predicted_Quality'], True),
            'Energy' : get_status(energy,  best_gs['Predicted_Energy'],  False),
            'Carbon' : get_status(carbon,  best_gs['Predicted_Carbon'],  False),
        },
        'vs_golden_signature': {
            'Quality_diff'     : round(quality - best_gs['Predicted_Quality'], 3),
            'Energy_saved_pct' : round((energy - best_gs['Predicted_Energy']) / energy * 100, 1),
            'Carbon_saved_pct' : round((carbon - best_gs['Predicted_Carbon']) / carbon * 100, 1),
        },
        'recommendations': recommendations
    }

# ── API Endpoints ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message" : "✅ Batch Optimization API is running!",
        "team"    : "Crazy Potatoes",
        "version" : "1.0",
        "endpoints": [
            "POST /api/predict",
            "GET  /api/golden-signatures",
            "GET  /api/health"
        ]
    }

@app.get("/api/health")
def health():
    return {
        "status"          : "healthy",
        "models_loaded"   : True,
        "golden_signatures": len(pareto_df)
    }

@app.post("/api/predict")
def predict(request: PredictRequest):
    result = predict_batch(
        request.user_json.dict(),
        request.sensor_json.dict() if request.sensor_json else None
    )
    return result

@app.get("/api/golden-signatures")
def get_golden_signatures():
    top10 = pareto_df[['Predicted_Quality',
                        'Predicted_Energy',
                        'Predicted_Carbon']].head(10)
    return top10.round(2).to_dict(orient='records')