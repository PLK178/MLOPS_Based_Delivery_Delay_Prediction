import os
import yaml
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

# Import all classification models at the beginning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import mlflow

from src.evaluation.evaluation import (
    calculate_classification_metrics,
    save_plots
)

# Constants
PROJECT_ROOT = "/home/likith/mlops/MLOPS"
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/model_config.yaml")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Mapping model names to classification classes
MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "adaboost": AdaBoostClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "xgboost": xgb.XGBClassifier,
    "catboost": cb.CatBoostClassifier,
    "lightgbm": lgb.LGBMClassifier
}

def load_config(config_path):
    """Load the model training configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_preprocessing_pipeline(numerical_cols, categorical_cols):
    """
    Build a standard preprocessing pipeline.
    Handles numerical imputation + scaling, and categorical imputation + one-hot encoding.
    """
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ])
    
    return preprocessor

def train_model(model_name, params, X, y, cv_params, categorical_features):
    """Train a single classification model using cross-validation and log results."""
    print(f"\n==================================================")
    print(f"🚀 Training {model_name.upper()} (classification)")
    print(f"==================================================")
    
    n_splits = cv_params.get("n_splits", 5)
    shuffle = cv_params.get("shuffle", True)
    random_state = cv_params.get("random_state", 42)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    oof_preds_default = np.zeros(len(X))
    oof_probs = np.zeros(len(X))
        
    models = []
    
    # Get model class
    model_class = MODEL_CLASSES[model_name]
    numerical_cols = [c for c in X.columns if c not in categorical_features]
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}...")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Build preprocessing pipeline
        preprocessor = get_preprocessing_pipeline(numerical_cols, categorical_features)
        
        # Fit preprocessor on training data and transform train and validation sets
        X_train_trans = preprocessor.fit_transform(X_train, y_train)
        X_val_trans = preprocessor.transform(X_val)
        

        
        # Copy model params
        model_params = params.copy()
        early_stopping_rounds = model_params.pop("early_stopping_rounds", None)
        
        # Instantiate base estimator
        if model_name == "adaboost":
            # Instantiate AdaBoost with a class-weighted Decision Tree stump
            base_model = model_class(**model_params, estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'))
        else:
            base_model = model_class(**model_params)
        
        # Fit base model, handling early stopping for gradient boosters and sample weighting for Gradient Boosting
        if model_name in ["xgboost", "catboost", "lightgbm"] and early_stopping_rounds:
            if model_name == "xgboost":
                base_model.set_params(early_stopping_rounds=early_stopping_rounds)
                base_model.fit(X_train_trans, y_train, eval_set=[(X_val_trans, y_val)], verbose=False)
            elif model_name == "catboost":
                base_model.fit(X_train_trans, y_train, eval_set=(X_val_trans, y_val), early_stopping_rounds=early_stopping_rounds, verbose=False)
            elif model_name == "lightgbm":
                callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
                base_model.fit(X_train_trans, y_train, eval_set=[(X_val_trans, y_val)], callbacks=callbacks)
        else:
            if model_name == "gradient_boosting":
                # Compute sample weights to balance classes for Gradient Boosting
                from sklearn.utils.class_weight import compute_sample_weight
                sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
                base_model.fit(X_train_trans, y_train, sample_weight=sample_weight)
            else:
                base_model.fit(X_train_trans, y_train)
            
        # Re-construct the full pipeline with the fitted preprocessor and fitted model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', base_model)
        ])
        
        # Predict on validation set using the pipeline (automatically handles transformation of raw X_val)
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)[:, 1]
        
        oof_preds_default[val_idx] = val_preds
        oof_probs[val_idx] = val_probs
        
        # Save fold model pipeline
        model_path = os.path.join(MODELS_DIR, f"{model_name}_fold_{fold}.joblib")
        joblib.dump(model, model_path)
        models.append(model_path)
        
    # --- DECISION THRESHOLD TUNING ON OUT-OF-FOLD PROBABILITIES ---
    best_thresh = 0.5
    best_f1 = 0.0
    thresholds = np.arange(0.01, 1.0, 0.01)
    
    for thresh in thresholds:
        preds = (oof_probs >= thresh).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"\n🎯 Best Decision Threshold found: {best_thresh:.2f} (OOF F1-Score: {best_f1:.6f})")
    
    # Calculate metrics at default (0.50) and tuned thresholds
    default_metrics = calculate_classification_metrics(y, oof_preds_default, oof_probs)
    
    oof_preds_tuned = (oof_probs >= best_thresh).astype(int)
    tuned_metrics = calculate_classification_metrics(y, oof_preds_tuned, oof_probs)
        
    print(f"✨ Overall OOF Metrics at Default Threshold (0.50):")
    for k, v in default_metrics.items():
        print(f"  - {k.upper()}: {v:.6f}")
        
    print(f"🎯 Overall OOF Metrics at Tuned Threshold ({best_thresh:.2f}):")
    for k, v in tuned_metrics.items():
        print(f"  - {k.upper()}: {v:.6f}")
    
    # Save plots (Use tuned predictions so the confusion matrix is balanced and meaningful)
    save_plots(y, oof_preds_tuned, oof_probs, "classification", PLOTS_DIR)
    
    # Rename output plots to prefix them with model name
    if os.path.exists(os.path.join(PLOTS_DIR, "confusion_matrix.png")):
        os.replace(os.path.join(PLOTS_DIR, "confusion_matrix.png"), os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png"))
    if os.path.exists(os.path.join(PLOTS_DIR, "roc_curve.png")):
        os.replace(os.path.join(PLOTS_DIR, "roc_curve.png"), os.path.join(PLOTS_DIR, f"{model_name}_roc_curve.png"))
            
    # Save OOF predictions, probabilities and overall metrics to results
    np.save(os.path.join(RESULTS_DIR, f"{model_name}_oof_preds.npy"), oof_preds_tuned)
    np.save(os.path.join(RESULTS_DIR, f"{model_name}_oof_probs.npy"), oof_probs)
    
    # Save metrics JSON
    metrics_path = os.path.join(RESULTS_DIR, f"{model_name}_metrics.yaml")
    with open(metrics_path, "w") as f:
        yaml_metrics = {
            "best_threshold": float(best_thresh),
            "default_threshold_0.50": {k: float(v) if isinstance(v, (np.float64, np.float32)) else v for k, v in default_metrics.items()},
            "tuned_threshold": {k: float(v) if isinstance(v, (np.float64, np.float32)) else v for k, v in tuned_metrics.items()}
        }
        yaml.safe_dump(yaml_metrics, f)
        
    # MLflow tracking
    try:
        # Use SQLite backend for MLflow tracking to support standard MLflow 3.0+ usage
        mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
        mlflow.set_experiment("Olist_Delivery_Prediction")
        with mlflow.start_run(run_name=f"{model_name}_classification"):
            # Log hyperparameters config
            mlflow.log_params(params)
            mlflow.log_param("task_type", "classification")
            mlflow.log_param("n_splits", n_splits)
            mlflow.log_param("tuned_threshold", best_thresh)
            
            # Log overall metrics at tuned threshold
            for key, val in tuned_metrics.items():
                if not np.isnan(val):
                    mlflow.log_metric(f"oof_{key}", val)
            
            # Log default F1 as a comparison reference
            mlflow.log_metric("oof_default_f1", default_metrics["f1"])
            
            # Log fold models as artifacts
            for fold, model_path in enumerate(models):
                mlflow.log_artifact(model_path, artifact_path="models")
                
            # Log plots as artifacts
            mlflow.log_artifact(os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png"))
            if os.path.exists(os.path.join(PLOTS_DIR, f"{model_name}_roc_curve.png")):
                mlflow.log_artifact(os.path.join(PLOTS_DIR, f"{model_name}_roc_curve.png"))
                
            print(f"✅ Logged to MLflow successfully under run name: {model_name}_classification")
    except Exception as e:
        print(f"⚠️ MLflow logging failed: {e}")
        
    return tuned_metrics

def main():
    # Make sure output dirs exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load config
    print(f"📖 Loading config from {CONFIG_PATH}...")
    config = load_config(CONFIG_PATH)
    
    # 2. Load dataset
    raw_path = config["data"]["path"]
    data_path = raw_path if os.path.isabs(raw_path) else os.path.join(PROJECT_ROOT, raw_path)
    print(f"📥 Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"📊 Dataset Shape: {df.shape}")
    
    target = config["data"]["target"]
    
    # 3. Clean target and drop rows with null targets
    null_target_count = df[target].isnull().sum()
    if null_target_count > 0:
        print(f"🧹 Dropping {null_target_count} rows with null in target '{target}'...")
        df = df.dropna(subset=[target])
        
    # Split features and target
    y = df[target].astype(int)
    
    # Determine columns to drop
    drop_cols = config["data"].get("drop_columns", [])
    drop_cols = [col for col in drop_cols if col in df.columns]
    drop_cols.append(target)
    
    features = [col for col in df.columns if col not in drop_cols]
    X = df[features].copy()
    
    print(f"🔑 Features to use: {features}")
    print(f"🎯 Target: {target} (classification)")
    
    # 4. Handle categorical features
    categorical_features = config["data"].get("categorical_features", [])
    categorical_features = [col for col in categorical_features if col in X.columns]
    
    for col in categorical_features:
        # Fill missing values and convert to string
        print(f"🏷️ Encoding '{col}' as categorical feature...")
        X[col] = X[col].fillna("missing").astype(str)
        
    # 5. Train each configured model
    cv_params = config["cv"]
    configured_models = config["models"]
    
    summary = {}
    for model_name, model_config in configured_models.items():
        params = model_config["params"]
        metrics = train_model(
            model_name=model_name,
            params=params,
            X=X,
            y=y,
            cv_params=cv_params,
            categorical_features=categorical_features
        )
        summary[model_name] = metrics
        
    print("\n" + "="*50)
    print("🏁 TRAINING COMPLETE SUMMARY (Tuned Thresholds)")
    print("="*50)
    summary_df = pd.DataFrame(summary).T
    print(summary_df)
    
    # Save training summary
    summary_df.to_csv(os.path.join(RESULTS_DIR, "training_summary.csv"))
    print(f"\n💾 Summary metrics saved to: {os.path.join(RESULTS_DIR, 'training_summary.csv')}")

if __name__ == "__main__":
    main()
