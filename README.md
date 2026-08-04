# MLOPS-Based Delivery Delay Prediction

This repository contains an end-to-end MLOps project for predicting delivery delays using machine learning pipelines, experiment tracking, and reproducible data/model versioning.

## Project Structure

- `/src` – Core ML pipeline code (ingestion, feature engineering, training, evaluation)
- `/APP` – Application entry point code
- `/configs` – Model and pipeline configuration files
- `/Data` – DVC-tracked raw and processed dataset references
- `/models` – Saved model artifacts
- `/templates` – Frontend/static template assets
- `/Notebooks` – Exploratory analysis notebooks

## Features

- Delivery delay prediction workflow
- Modular training and evaluation pipeline
- Multiple model support (including XGBoost)
- DVC integration for dataset tracking
- Dockerized setup for reproducibility

## Prerequisites

- Python 3.9+
- pip
- Docker (optional)
- DVC (optional, for pulling tracked data)

## Installation

```bash
git clone https://github.com/PLK178/MLOPS_Based_Delivery_Delay_Prediction.git
cd MLOPS_Based_Delivery_Delay_Prediction
pip install -r requirements.txt
```

## Running the Project

### 1. Run the application

```bash
python /home/runner/work/MLOPS_Based_Delivery_Delay_Prediction/MLOPS_Based_Delivery_Delay_Prediction/APP/main.py
```

### 2. Run tests

```bash
python /home/runner/work/MLOPS_Based_Delivery_Delay_Prediction/MLOPS_Based_Delivery_Delay_Prediction/src/test/test.py
```

### 3. Docker workflow (optional)

```bash
docker-compose up --build
```

## Data Versioning (DVC)

This project uses DVC for tracking data files in `/Data/raw` and `/Data/processed`.

If DVC remote is configured, pull data with:

```bash
dvc pull
```

## Configuration

Model settings are defined in:

- `/home/runner/work/MLOPS_Based_Delivery_Delay_Prediction/MLOPS_Based_Delivery_Delay_Prediction/configs/model_config.yaml`

## License

This project is licensed under the terms in the `LICENSE` file.
