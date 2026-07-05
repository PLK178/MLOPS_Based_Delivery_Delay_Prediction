import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Delivery Latency Predictor API")

# Enable CORS so the browser can reach the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained XGBoost model pipeline
MODEL_PATH = "models/best_xgb_model_pipeline.joblib"
model = None

try:
    if os.path.exists(MODEL_PATH):
        print(f"🔄 Loading model pipeline from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        print("✅ Model pipeline loaded successfully.")
    else:
        print(f"⚠️ Warning: Model pipeline not found at {MODEL_PATH}.")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")

# Request schema matching the feature columns
class InferencePayload(BaseModel):
    price: float
    freight_value: float
    product_category_name: str
    product_weight_g: float
    product_volume_cm3: float
    is_same_state: int
    purchase_month: int
    purchase_day_of_week: int
    purchase_hour: int
    estimated_delivery_time_days: float

@app.post("/predict")
def predict(payload: InferencePayload):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded on the server. Please check server logs."
        )
    
    try:
        # Calculate freight_to_price_ratio (avoiding division by zero)
        freight_ratio = payload.freight_value / payload.price if payload.price > 0 else 0.0

        # Convert request to pandas DataFrame (pipeline expects column names to match exactly)
        input_data = pd.DataFrame([{
            "price": payload.price,
            "freight_value": payload.freight_value,
            "product_category_name": payload.product_category_name,
            "product_weight_g": payload.product_weight_g,
            "product_volume_cm3": payload.product_volume_cm3,
            "is_same_state": payload.is_same_state,
            "purchase_month": payload.purchase_month,
            "purchase_day_of_week": payload.purchase_day_of_week,
            "purchase_hour": payload.purchase_hour,
            "estimated_delivery_time_days": payload.estimated_delivery_time_days,
            "freight_to_price_ratio": freight_ratio
        }])
        
        # Make predictions
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        
        # Get probability score for the delay class (index 1)
        probability = float(probabilities[1])
        
        return {
            "prediction": prediction,
            "probability": probability
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Mount the static templates folder directly at "/"
# This serves index.html at "/", and style.css/app.js relatively.
if os.path.exists("templates"):
    app.mount("/", StaticFiles(directory="templates", html=True), name="templates")
else:
    print("⚠️ Warning: 'templates' directory not found. Frontend will not be served.")

if __name__ == "__main__":
    import uvicorn
    # Run server on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
