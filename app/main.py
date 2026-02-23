from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd

app = FastAPI(title="Car Insurance Risk API", version="1.0.0")

# --- Load model dict once at startup ---
bundle = joblib.load("model/calibrated_xgboost_final.pkl")
xgb_model   = bundle["xgb_model"]
calibrator  = bundle["calibrator"]   # IsotonicRegression
THRESHOLD   = bundle["threshold"]    # 0.549

# Column order must exactly match what the model was trained on
with open("model/feature_names.json") as f:
    FEATURE_NAMES = json.load(f)
COL_MAP = {name.replace("_", ""): name for name in FEATURE_NAMES}

# --- Input: all ints/floats matching your label-encoded columns ---
class PolicyInput(BaseModel):
    AGE: int = Field(..., ge=0, le=3, description="0=16-25, 1=26-39, 2=40-64, 3=65+")
    GENDER: int = Field(..., ge=0, le=1)
    RACE: int = Field(..., ge=0, le=1)
    DRIVINGEXPERIENCE: int = Field(..., ge=0, le=3, description="0=0-9y, 1=10-19y, 2=20-29y, 3=30y+")
    EDUCATION: int = Field(..., ge=0, le=2, description="0=none, 1=high school, 2=university")
    INCOME: int = Field(..., ge=0, le=3, description="0=poverty, 1=working, 2=middle, 3=upper")
    CREDITSCORE: float = Field(..., ge=0.0, le=1.0)
    VEHICLEOWNERSHIP: int = Field(..., ge=0, le=1)
    VEHICLEYEAR: int = Field(..., ge=0, le=1, description="0=before 2015, 1=after 2015")
    MARRIED: int = Field(..., ge=0, le=1)
    CHILDREN: int = Field(..., ge=0, le=1)
    POSTALCODE: int
    ANNUALMILEAGE: float = Field(..., ge=0)
    VEHICLETYPE: int = Field(..., ge=0, le=1, description="0=sedan, 1=sports car")
    SPEEDINGVIOLATIONS: int = Field(..., ge=0)
    DUIS: int = Field(..., ge=0)
    PASTACCIDENTS: int = Field(..., ge=0)

class PredictionOutput(BaseModel):
    claim_probability: float
    risk_tier: str
    prediction: int  # 0 or 1

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOutput)
def predict(policy: PolicyInput):
    try:
        input_dict = policy.dict()
        
        # DEBUG: print both so you can see the mismatch
        print("INPUT KEYS:   ", list(input_dict.keys()))
        print("FEATURE_NAMES:", FEATURE_NAMES)
        
        # Translate keys: PolicyInput names → model's expected names
        renamed = {COL_MAP.get(k, k): v for k, v in input_dict.items()}

        df = pd.DataFrame([renamed])[FEATURE_NAMES]

        # Step 1: raw XGBoost probability
        raw_proba = xgb_model.predict_proba(df)[0, 1]

        # Step 2: isotonic calibration
        calibrated_proba = float(calibrator.predict([raw_proba])[0])
        calibrated_proba = max(calibrated_proba, 1e-4)  # floor at 0.0001

        # Step 3: threshold
        prediction = int(calibrated_proba >= THRESHOLD)

        # Step 4: tier
        if calibrated_proba < 0.15:          tier = "Very Low"
        elif calibrated_proba < 0.30:        tier = "Low"
        elif calibrated_proba < THRESHOLD:   tier = "Medium"    # below threshold → no claim predicted
        elif calibrated_proba < 0.75:        tier = "High"      # above threshold → claim predicted
        else:                                tier = "Very High"

        return PredictionOutput(
            claim_probability=round(calibrated_proba, 4),
            risk_tier=tier,
            prediction=prediction
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
