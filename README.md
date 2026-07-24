MLOPS PROJECT: Delivery Delay Prediction

This repository implements an end‑to‑end MLOps pipeline that predicts delivery delays. It combines DAG orchestration, MLflow experiment tracking, DVC data versioning, and Docker containerisation to deliver a reproducible, scalable solution.

Key components:
- **DAGs**: orchestrate data ingestion, feature engineering, model training, and evaluation.
- **MLflow**: logs experiments, registers models, and manages lifecycle.
- **DVC**: version controls datasets and pipeline steps.
- **Docker**: provides consistent environments for development and deployment.

Supported models include XGBoost, CatBoost, LightGBM, and more.
