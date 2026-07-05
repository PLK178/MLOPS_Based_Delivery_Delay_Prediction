# Use a slim Python 3.11 base image to keep it lightweight
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system compilation dependencies (some packages might need compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies in builder stage
COPY requirements.txt .
# Add fastapi and uvicorn in case they are not in requirements.txt
RUN pip install --no-cache-dir --user fastapi uvicorn pydantic joblib pandas scikit-learn xgboost

# --- Final Production Image ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder stage to keep final image clean
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy backend server entrypoint, source pipeline files, and models
COPY main.py .
COPY src/ ./src/
COPY models/best_xgb_model_pipeline.joblib ./models/

EXPOSE 8000

# Start FastAPI with uvicorn (set workers to 1 to limit RAM overhead)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
