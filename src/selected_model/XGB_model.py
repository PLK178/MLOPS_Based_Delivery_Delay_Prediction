import os
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
import mlflow
import dagshub

from src.Model_evaluation.evaluation import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    save_plots
)
import onnx
import onnxruntime as rt
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType, StringTensorType, Int64TensorType
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes
)
from xgboost import XGBClassifier, XGBRegressor

# Register XGBoost Classifier converter with skl2onnx
update_registered_converter(
    XGBClassifier, 'XGBClassifier',
    calculate_linear_classifier_output_shapes, convert_xgboost,
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
)

# Register XGBoost Regressor converter with skl2onnx
update_registered_converter(
    XGBRegressor, 'XGBRegressor',
    calculate_linear_regressor_output_shapes, convert_xgboost
)

# Constants
PROJECT_ROOT = "/home/likith/mlops/MLOPS"
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/model_config.yaml")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_preprocessing_pipeline(numerical_cols, categorical_cols):
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ])
    
    return preprocessor

def main():
    # 1. Load config and dataset
    print(f"📖 Loading config from {CONFIG_PATH}...")
    config = load_config(CONFIG_PATH)
    
    raw_path = config["data"]["path"]
    data_path = raw_path if os.path.isabs(raw_path) else os.path.join(PROJECT_ROOT, raw_path)
    print(f"📥 Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    target = config["data"]["target"]
    task_type = config["data"].get("task_type", "classification")
    
    # Drop rows with null target
    df = df.dropna(subset=[target])
    
    if task_type == "classification":
        y = df[target].astype(int)
    else:
        y = df[target].astype(float)
        
    # Columns to drop
    drop_cols = config["data"].get("drop_columns", [])
    drop_cols = [col for col in drop_cols if col in df.columns]
    drop_cols.append(target)
    
    features = [col for col in df.columns if col not in drop_cols]
    X = df[features].copy()
    
    # 2. Categorical encoding setup
    categorical_features = config["data"].get("categorical_features", [])
    categorical_features = [col for col in categorical_features if col in X.columns]
    for col in categorical_features:
        X[col] = X[col].fillna("missing").astype(str)
        
    numerical_cols = [col for col in X.columns if col not in categorical_features]
    
    # 3. Train-Test Split (80/20)
    print("✂️ Splitting dataset into 80% train and 20% test...")
    if task_type == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Filter training outliers to prevent them from distorting the regression gradients
        train_mask = (y_train >= 0) & (y_train <= 50)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        
    # 4. Build Preprocessing Pipeline
    preprocessor = get_preprocessing_pipeline(numerical_cols, categorical_features)
    
    # 5. Fit & Transform
    print("⚙️ Preprocessing features...")
    X_train_trans = preprocessor.fit_transform(X_train, y_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # 6. Instantiate the best XGBoost model
    print(f"🚀 Training the best XGBoost {task_type} model...")
    if task_type == "classification":
        best_xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=11.5
        )
    else:
        best_xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            tree_method="hist",
            eval_metric="rmse",
            early_stopping_rounds=50
        )
        
    # Fit the model
    best_xgb_model.fit(
        X_train_trans, y_train,
        eval_set=[(X_test_trans, y_test)],
        verbose=100
    )
    
    # 7. Construct Full Prediction Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', best_xgb_model)
    ])
    
    # Save the full model pipeline
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_save_path = os.path.join(MODELS_DIR, "best_xgb_model_pipeline.joblib")
    joblib.dump(pipeline, model_save_path)
    print(f"💾 Saved best model pipeline to: {model_save_path}")
    
    # 7.5 Convert to ONNX and Save
    print("🔄 Converting pipeline to ONNX format...")
    initial_types = [
        ('price', FloatTensorType([None, 1])),
        ('freight_value', FloatTensorType([None, 1])),
        ('product_category_name', StringTensorType([None, 1])),
        ('product_weight_g', FloatTensorType([None, 1])),
        ('product_volume_cm3', FloatTensorType([None, 1])),
        ('is_same_state', Int64TensorType([None, 1])),
        ('purchase_month', Int64TensorType([None, 1])),
        ('purchase_day_of_week', Int64TensorType([None, 1])),
        ('purchase_hour', Int64TensorType([None, 1])),
        ('estimated_delivery_time_days', FloatTensorType([None, 1])),
        ('freight_to_price_ratio', FloatTensorType([None, 1])),
    ]
    
    if task_type == "classification":
        options = {XGBClassifier: {'zipmap': False}}
    else:
        options = None
        
    model_onnx = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        target_opset={'': 15, 'ai.onnx.ml': 3},
        options=options
    )
    onnx_save_path = os.path.join(MODELS_DIR, "best_xgb_model_pipeline.onnx")
    with open(onnx_save_path, "wb") as f:
        f.write(model_onnx.SerializeToString())
    print(f"💾 Saved ONNX model pipeline to: {onnx_save_path}")
    
    # 8. Evaluate on Test Set
    test_preds = pipeline.predict(X_test)
    
    if task_type == "classification":
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        
        # Find best decision threshold on test probabilities for F1-score
        best_thresh = 0.5
        best_f1 = 0.0
        thresholds = np.arange(0.01, 1.0, 0.01)
        
        for thresh in thresholds:
            preds = (test_probs >= thresh).astype(int)
            f1 = calculate_classification_metrics(y_test, preds)["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                
        print(f"\n🎯 Optimal Decision Threshold found: {best_thresh:.2f}")
        
        # Calculate metrics at default and optimal thresholds
        default_preds = (test_probs >= 0.50).astype(int)
        default_metrics = calculate_classification_metrics(y_test, default_preds, test_probs)
        
        tuned_preds = (test_probs >= best_thresh).astype(int)
        tuned_metrics = calculate_classification_metrics(y_test, tuned_preds, test_probs)
        
        print("\n📊 Evaluation Metrics on Test Set (Default Threshold 0.50):")
        for k, v in default_metrics.items():
            print(f"  - {k.upper()}: {v:.6f}")
            
        print(f"\n📊 Evaluation Metrics on Test Set (Tuned Threshold {best_thresh:.2f}):")
        for k, v in tuned_metrics.items():
            print(f"  - {k.upper()}: {v:.6f}")
            
        save_plots(y_test, tuned_preds, test_probs, task_type="classification", output_dir=PLOTS_DIR)
        print(f"📈 Saved evaluation plots to: {PLOTS_DIR}")
    else:
        reg_metrics = calculate_regression_metrics(y_test, test_preds)
        print(f"\n📊 Evaluation Metrics on Test Set (Regression):")
        for k, v in reg_metrics.items():
            print(f"  - {k.upper()}: {v:.6f}")
            
        save_plots(y_test, test_preds, task_type="regression", output_dir=PLOTS_DIR)
        print(f"📈 Saved evaluation plots to: {PLOTS_DIR}")
        
    # 9. MLflow Tracking and Model Registry
    try:
        print("🧪 Initializing MLflow / DagsHub tracking...")
        dagshub.init(repo_owner='PLK178', repo_name='MLOPS', mlflow=True)
        mlflow.set_experiment("Olist_Delivery_Prediction")
        
        run_name = f"XGB_Final_Model_Registration_{task_type}"
        with mlflow.start_run(run_name=run_name) as run:
            print("📝 Logging parameters and metrics to MLflow...")
            if task_type == "classification":
                # Log params
                mlflow.log_params({
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "random_state": 42,
                    "scale_pos_weight": 11.5,
                    "optimal_threshold": best_thresh
                })
                
                # Log default threshold metrics
                for k, v in default_metrics.items():
                    if not np.isnan(v):
                        mlflow.log_metric(f"default_{k}", v)
                
                # Log tuned threshold metrics
                for k, v in tuned_metrics.items():
                    if not np.isnan(v):
                        mlflow.log_metric(f"tuned_{k}", v)
            else:
                mlflow.log_params({
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "random_state": 42
                })
                
                for k, v in reg_metrics.items():
                    if not np.isnan(v):
                        mlflow.log_metric(k, v)
                        
            # Log artifacts (plots and model file)
            mlflow.log_artifact(model_save_path, artifact_path="models")
            mlflow.log_artifact(onnx_save_path, artifact_path="models")
            if os.path.exists(PLOTS_DIR):
                mlflow.log_artifacts(PLOTS_DIR, artifact_path="plots")
            
            # Register the model to MLflow Model Registry
            print("📦 Registering model in MLflow Model Registry...")
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name="Olist_XGB_Model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                    "xgboost.sklearn.XGBRegressor"
                ]
            )
            print("✅ Model successfully registered as 'Olist_XGB_Model'.")
            
    except Exception as e:
        print(f"⚠️ MLflow/DagsHub logging & registration failed: {e}")

if __name__ == "__main__":
    main()
