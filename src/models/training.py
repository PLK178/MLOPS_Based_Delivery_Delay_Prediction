import os
import yaml
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_squared_error

# Import all classification and regression models
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import mlflow
import dagshub

from src.Model_evaluation.evaluation import (
    calculate_classification_metrics,
    calculate_regression_metrics
)

# Constants
PROJECT_ROOT = "/home/likith/mlops/MLOPS"
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/model_config.yaml")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Mapping model names to classification classes
MODEL_CLASSES_CLASSIFICATION = {
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

# Mapping model names to regression classes
MODEL_CLASSES_REGRESSION = {
    "logistic_regression": Ridge,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "adaboost": AdaBoostRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "xgboost": xgb.XGBRegressor,
    "catboost": cb.CatBoostRegressor,
    "lightgbm": lgb.LGBMRegressor
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

def tune_hyperparameters(model_name, static_params, param_grid, X, y, cv_params, categorical_features, task_type="classification"):
    """Perform a grid search over param_grid using a subset CV split for speed."""
    from sklearn.model_selection import ParameterGrid
    grid = list(ParameterGrid(param_grid))
    
    if len(grid) <= 1:
        return grid[0] if grid else {}
        
    print(f"🔍 Tuning hyperparameters for {model_name} (evaluating {len(grid)} combinations)...")
    
    best_params = grid[0]
    
    # 3-fold split for faster hyperparameter search
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        model_classes = MODEL_CLASSES_CLASSIFICATION
        best_score = -1.0
    else:
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        model_classes = MODEL_CLASSES_REGRESSION
        best_score = 1e9
        
    numerical_cols = [c for c in X.columns if c not in categorical_features]
    model_class = model_classes[model_name]
    
    for params_combo in grid:
        scores = []
        full_params = {**static_params, **params_combo}
        
        # Pop early stopping and other train-only params
        early_stopping_rounds = full_params.pop("early_stopping_rounds", None)
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            preprocessor = get_preprocessing_pipeline(numerical_cols, categorical_features)
            X_train_trans = preprocessor.fit_transform(X_train, y_train)
            X_val_trans = preprocessor.transform(X_val)
            
            if model_name == "adaboost":
                if task_type == "classification":
                    base_model = model_class(**full_params, estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'))
                else:
                    base_model = model_class(**full_params, estimator=DecisionTreeRegressor(max_depth=3))
            else:
                base_model = model_class(**full_params)
                
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
                if model_name == "gradient_boosting" and task_type == "classification":
                    from sklearn.utils.class_weight import compute_sample_weight
                    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
                    base_model.fit(X_train_trans, y_train, sample_weight=sample_weight)
                else:
                    base_model.fit(X_train_trans, y_train)
                    
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', base_model)
            ])
            val_preds = pipeline.predict(X_val)
            if task_type == "classification":
                score = f1_score(y_val, val_preds, zero_division=0)
                scores.append(score)
            else:
                score = mean_squared_error(y_val, val_preds)
                scores.append(score)
            
        mean_score = np.mean(scores)
        if task_type == "classification":
            if mean_score > best_score:
                best_score = mean_score
                best_params = params_combo
        else:
            if mean_score < best_score:
                best_score = mean_score
                best_params = params_combo
                
    if task_type == "classification":
        print(f"🎯 Best combination for {model_name}: {best_params} (Mean CV F1: {best_score:.6f})")
    else:
        print(f"🎯 Best combination for {model_name}: {best_params} (Mean CV MSE: {best_score:.6f})")
    return best_params

def train_model(model_name, static_params, param_grid, X, y, cv_params, categorical_features, task_type="classification"):
    """Tune and then train a single model using cross-validation and log results."""
    print(f"\n==================================================")
    print(f"🚀 Training {model_name.upper()} ({task_type})")
    print(f"==================================================")
    
    # 1. Hyperparameter Tuning
    best_tuned_params = tune_hyperparameters(model_name, static_params, param_grid, X, y, cv_params, categorical_features, task_type=task_type)
    
    # 2. Re-combine static and best tuned params for final cross-validated training
    final_params = {**static_params, **best_tuned_params}
    
    n_splits = cv_params.get("n_splits", 5)
    shuffle = cv_params.get("shuffle", True)
    random_state = cv_params.get("random_state", 42)
    
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        oof_preds_default = np.zeros(len(X))
        oof_probs = np.zeros(len(X))
        model_classes = MODEL_CLASSES_CLASSIFICATION
    else:
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        oof_preds = np.zeros(len(X))
        model_classes = MODEL_CLASSES_REGRESSION
        
    models = []
    
    # Get model class
    model_class = model_classes[model_name]
    numerical_cols = [c for c in X.columns if c not in categorical_features]
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"Fold {fold + 1}/{n_splits}...")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Build preprocessing pipeline
        preprocessor = get_preprocessing_pipeline(numerical_cols, categorical_features)
        
        # Fit preprocessor on training data and transform train and validation sets
        X_train_trans = preprocessor.fit_transform(X_train, y_train)
        X_val_trans = preprocessor.transform(X_val)
        
        # Copy model params
        model_params = final_params.copy()
        early_stopping_rounds = model_params.pop("early_stopping_rounds", None)
        
        # Instantiate base estimator
        if model_name == "adaboost":
            if task_type == "classification":
                base_model = model_class(**model_params, estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'))
            else:
                base_model = model_class(**model_params, estimator=DecisionTreeRegressor(max_depth=3))
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
            if model_name == "gradient_boosting" and task_type == "classification":
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
        
        if task_type == "classification":
            val_probs = model.predict_proba(X_val)[:, 1]
            oof_preds_default[val_idx] = val_preds
            oof_probs[val_idx] = val_probs
        else:
            oof_preds[val_idx] = val_preds
            
        # Save fold model pipeline
        model_path = os.path.join(MODELS_DIR, f"{model_name}_fold_{fold}.joblib")
        joblib.dump(model, model_path)
        models.append(model_path)
        
    if task_type == "classification":
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
            
        metrics_to_log = tuned_metrics
    else:
        reg_metrics = calculate_regression_metrics(y, oof_preds)
        print(f"✨ Overall OOF Regression Metrics:")
        for k, v in reg_metrics.items():
            print(f"  - {k.upper()}: {v:.6f}")
            
        metrics_to_log = reg_metrics
    
    # MLflow tracking
    try:
        if mlflow.active_run() is not None:
            # Log directly to the active parent run, prefixing with model_name to prevent key collisions
            prefixed_params = {f"{model_name}_{k}": v for k, v in final_params.items()}
            mlflow.log_params(prefixed_params)
            
            if task_type == "classification":
                mlflow.log_param(f"{model_name}_tuned_threshold", best_thresh)
                # Log default F1 as a comparison reference
                mlflow.log_metric(f"{model_name}_default_f1", default_metrics["f1"])
            
            # Log overall metrics
            for key, val in metrics_to_log.items():
                if not np.isnan(val):
                    mlflow.log_metric(f"{model_name}_{key}", val)
            
            # Log fold models as artifacts
            for fold, model_path in enumerate(models):
                mlflow.log_artifact(model_path, artifact_path=f"models/{model_name}")
                
            print(f"✅ Logged {model_name} metrics & parameters directly to parent run.")
    except Exception as e:
        print(f"⚠️ MLflow logging failed: {e}")
        
    return metrics_to_log

def main():
    # Make sure output dirs exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
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
    task_type = config["data"].get("task_type", "classification")
    
    # 3. Clean target and drop rows with null targets
    null_target_count = df[target].isnull().sum()
    if null_target_count > 0:
        print(f"🧹 Dropping {null_target_count} rows with null in target '{target}'...")
        df = df.dropna(subset=[target])
        
    # Split features and target
    if task_type == "classification":
        y = df[target].astype(int)
    else:
        y = df[target].astype(float)
        
    # Determine columns to drop
    drop_cols = config["data"].get("drop_columns", [])
    drop_cols = [col for col in drop_cols if col in df.columns]
    drop_cols.append(target)
    
    features = [col for col in df.columns if col not in drop_cols]
    X = df[features].copy()
    
    print(f"🔑 Features to use: {features}")
    print(f"🎯 Target: {target} ({task_type})")
    
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
    
    # Initialize DagsHub MLflow tracking once for the run
    mlflow_enabled = False
    try:
        dagshub.init(repo_owner='PLK178', repo_name='MLOPS', mlflow=True)
        mlflow.set_experiment("Olist_Delivery_Prediction")
        mlflow_enabled = True
    except Exception as e:
        print(f"⚠️ MLflow initialization failed: {e}")
        
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_run_name = f"Training_Run_{timestamp}"
    
    summary = {}
    
    if mlflow_enabled:
        print(f"🧪 Starting MLflow Parent Run: {parent_run_name} under experiment: Olist_Delivery_Prediction")
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            for model_name, model_config in configured_models.items():
                static_params = model_config.get("static_params", {})
                param_grid = model_config.get("param_grid", {})
                metrics = train_model(
                    model_name=model_name,
                    static_params=static_params,
                    param_grid=param_grid,
                    X=X,
                    y=y,
                    cv_params=cv_params,
                    categorical_features=categorical_features,
                    task_type=task_type
                )
                summary[model_name] = metrics
    else:
        for model_name, model_config in configured_models.items():
            static_params = model_config.get("static_params", {})
            param_grid = model_config.get("param_grid", {})
            metrics = train_model(
                model_name=model_name,
                static_params=static_params,
                param_grid=param_grid,
                X=X,
                y=y,
                cv_params=cv_params,
                categorical_features=categorical_features,
                task_type=task_type
            )
            summary[model_name] = metrics
        
    print("\n" + "="*50)
    print(f"🏁 TRAINING COMPLETE SUMMARY ({task_type})")
    print("="*50)
    summary_df = pd.DataFrame(summary).T
    print(summary_df)
    
if __name__ == "__main__":
    main()
