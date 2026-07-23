import os
import unittest
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.Model_evaluation.evaluation import calculate_regression_metrics

PROJECT_ROOT = "/home/likith/mlops/MLOPS"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/best_xgb_model_pipeline.joblib")
DATA_PATH = os.path.join(PROJECT_ROOT, "Data/processed/cleaned_dataset.csv")

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
        cls.model = joblib.load(MODEL_PATH)
        
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")
        cls.df = pd.read_csv(DATA_PATH)
        
        cls.target = "delivery_time_days"
        if cls.target in cls.df.columns:
            cls.df = cls.df.dropna(subset=[cls.target])
            cls.y = cls.df[cls.target].astype(float)
        else:
            raise KeyError(f"Target column '{cls.target}' not found in the dataset.")

        # Prepare features
        drop_cols = ["order_id", "product_id", "seller_id", "is_delayed", cls.target]
        features = [col for col in cls.df.columns if col not in drop_cols]
        cls.X = cls.df[features].copy()
        
        if "product_category_name" in cls.X.columns:
            cls.X["product_category_name"] = cls.X["product_category_name"].fillna("missing").astype(str)

        # 80/20 train-test split (no stratification for regression)
        _, cls.X_test, _, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )

    def test_model_loaded(self):
        """Test that the loaded model is a Scikit-learn Pipeline object."""
        self.assertIsInstance(self.model, Pipeline)
        self.assertTrue(hasattr(self.model, "predict"))

    def test_model_predictions_shape(self):
        """Test that model predictions have the correct shape and type."""
        sample_size = min(100, len(self.X_test))
        X_sample = self.X_test.iloc[:sample_size]
        
        preds = self.model.predict(X_sample)
        self.assertEqual(len(preds), sample_size)
        self.assertTrue(np.issubdtype(preds.dtype, np.floating) or np.issubdtype(preds.dtype, np.integer))

    def test_model_performance_on_test_dataset(self):
        """Evaluate and verify model correctness on the 20% test dataset."""
        test_preds = self.model.predict(self.X_test)
        metrics = calculate_regression_metrics(self.y_test, test_preds)
        
        # Verify that MAE/RMSE is reasonable
        self.assertLess(metrics["mae"], 15.0)
        self.assertGreater(metrics["r2"], -1.0)

    def test_predict_on_5_sample_test_dataset(self):
        """Predict on a specific 5-sample batch from the test set and check predictions."""
        # Grab 5 samples from the test set (e.g., indices 10 to 14 for variability)
        X_5 = self.X_test.iloc[10:15]
        y_5_actual = self.y_test.iloc[10:15].values
        
        print("\n\n=== Running Predictions on 5 Test Samples ===")
        
        preds = self.model.predict(X_5)
        
        # Display the result of each sample
        for i in range(5):
            actual = y_5_actual[i]
            predicted = preds[i]
            diff = abs(actual - predicted)
            print(f"Sample {i+1}: Actual Days={actual:.2f} | Predicted Days={predicted:.2f} (Error={diff:.2f})")
            
        print("=============================================\n")
        
        # Simple sanity check
        self.assertEqual(len(preds), 5)

if __name__ == "__main__":
    print("\n🔍 Loading model and evaluating on 20% test dataset partition...")
    try:
        TestModel.setUpClass()
        model = TestModel.model
        X_test = TestModel.X_test
        y_test = TestModel.y_test
        preds = model.predict(X_test)
        metrics = calculate_regression_metrics(y_test, preds)
        print("\n=============================================")
        print("📊 TEST SET EVALUATION SUMMARY")
        print("=============================================")
        print(f"  - Mean Absolute Error (MAE):     {metrics['mae']:.6f} days")
        print(f"  - Root Mean Squared Error (RMSE): {metrics['rmse']:.6f} days")
        print(f"  - Mean Squared Error (MSE):      {metrics['mse']:.6f}")
        print(f"  - R-squared (R2) Score:          {metrics['r2']:.6f}")
        print("=============================================")
    except Exception as e:
        print(f"Error printing evaluation report: {e}")

    print("\nRunning Unittests:")
    unittest.main()
