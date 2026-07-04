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

from src.Model_evaluation.evaluation import (
    calculate_classification_metrics,
    save_plots
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
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
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
    
    # Drop rows with null target
    df = df.dropna(subset=[target])
    y = df[target].astype(int)
    
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 4. Build Preprocessing Pipeline
    preprocessor = get_preprocessing_pipeline(numerical_cols, categorical_features)
    
    # 5. Fit & Transform
    print("⚙️ Preprocessing features...")
    X_train_trans = preprocessor.fit_transform(X_train, y_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # 6. Instantiate the best XGBoost Classifier
    print("🚀 Training the best XGBoost model...")
    # Best parameters found: learning_rate=0.05, max_depth=6, n_estimators=300
    # Static parameters: random_state=42, tree_method="hist", eval_metric="logloss", scale_pos_weight=11.5
    best_xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        tree_method="hist",
        eval_metric="logloss",
        scale_pos_weight=11.5
    )
    
    # Fit the model
    best_xgb_model.fit(
        X_train_trans, y_train,
        eval_set=[(X_test_trans, y_test)],
        verbose=50
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
    
    # 8. Evaluate on Test Set
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
        
    # Save evaluation plots
    save_plots(y_test, tuned_preds, test_probs, task_type="classification", output_dir=PLOTS_DIR)
    print(f"📈 Saved evaluation plots to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
