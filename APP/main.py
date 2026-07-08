import os
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

import numpy as np
import onnxruntime as rt

# Load the trained ONNX model
MODEL_PATH = "models/best_xgb_model_pipeline.onnx"
session = None

try:
    if os.path.exists(MODEL_PATH):
        print(f"🔄 Loading ONNX model from {MODEL_PATH}...")
        session = rt.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        print("✅ ONNX model loaded successfully.")
    else:
        print(f"⚠️ Warning: ONNX model not found at {MODEL_PATH}.")
except Exception as e:
    print(f"❌ Error loading ONNX model: {str(e)}")

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
    if session is None:
        raise HTTPException(
            status_code=503, 
            detail="ONNX model is not loaded on the server. Please check server logs."
        )
    
    try:
        # Calculate freight_to_price_ratio (avoiding division by zero)
        freight_ratio = payload.freight_value / payload.price if payload.price > 0 else 0.0

        # Prepare inputs matching ONNX expected types & shapes
        input_data = {
            'price': np.array([[payload.price]], dtype=np.float32),
            'freight_value': np.array([[payload.freight_value]], dtype=np.float32),
            'product_category_name': np.array([[payload.product_category_name]], dtype=object),
            'product_weight_g': np.array([[payload.product_weight_g]], dtype=np.float32),
            'product_volume_cm3': np.array([[payload.product_volume_cm3]], dtype=np.float32),
            'is_same_state': np.array([[payload.is_same_state]], dtype=np.int64),
            'purchase_month': np.array([[payload.purchase_month]], dtype=np.int64),
            'purchase_day_of_week': np.array([[payload.purchase_day_of_week]], dtype=np.int64),
            'purchase_hour': np.array([[payload.purchase_hour]], dtype=np.int64),
            'estimated_delivery_time_days': np.array([[payload.estimated_delivery_time_days]], dtype=np.float32),
            'freight_to_price_ratio': np.array([[freight_ratio]], dtype=np.float32),
        }
        
        # Make predictions using ONNX Runtime
        raw_preds = session.run(None, input_data)
        prediction = int(raw_preds[0][0])
        probability = float(raw_preds[1][0][1])
        
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
