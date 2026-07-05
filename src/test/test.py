import os
import unittest
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.Model_evaluation.evaluation import calculate_classification_metrics

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
        
        cls.target = "is_delayed"
        if cls.target in cls.df.columns:
            cls.df = cls.df.dropna(subset=[cls.target])
            cls.y = cls.df[cls.target].astype(int)
        else:
            raise KeyError(f"Target column '{cls.target}' not found in the dataset.")

        # Prepare features
        drop_cols = ["order_id", "product_id", "seller_id", "delivery_time_days", cls.target]
        features = [col for col in cls.df.columns if col not in drop_cols]
        cls.X = cls.df[features].copy()
        
        if "product_category_name" in cls.X.columns:
            cls.X["product_category_name"] = cls.X["product_category_name"].fillna("missing").astype(str)

        # 80/20 train-test split to ensure clean test set
        _, cls.X_test, _, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, stratify=cls.y, random_state=42
        )

    def test_model_loaded(self):
        """Test that the loaded model is a Scikit-learn Pipeline object."""
        self.assertIsInstance(self.model, Pipeline)
        self.assertTrue(hasattr(self.model, "predict"))
        self.assertTrue(hasattr(self.model, "predict_proba"))

    def test_model_predictions_shape(self):
        """Test that model predictions have the correct shape and type."""
        sample_size = min(100, len(self.X_test))
        X_sample = self.X_test.iloc[:sample_size]
        
        preds = self.model.predict(X_sample)
        self.assertEqual(len(preds), sample_size)
        self.assertTrue(np.all((preds == 0) | (preds == 1)))

    def test_model_accuracy_on_test_dataset(self):
        """Evaluate and verify model correctness on the 20% test dataset."""
        test_probs = self.model.predict_proba(self.X_test)[:, 1]
        
        # Best decision threshold (found via training/tuning: 0.88)
        best_thresh = 0.88
        tuned_preds = (test_probs >= best_thresh).astype(int)
        tuned_metrics = calculate_classification_metrics(self.y_test, tuned_preds, test_probs)
        
        # Verify that accuracy is high and above baseline
        self.assertGreater(tuned_metrics["accuracy"], 0.70)
        self.assertGreater(tuned_metrics["f1"], 0.30)

    def test_predict_on_5_sample_test_dataset(self):
        """Predict on a specific 5-sample batch from the test set and check predictions."""
        # Grab 5 samples from the test set (e.g., indices 10 to 14 for variability)
        X_5 = self.X_test.iloc[10:15]
        y_5_actual = self.y_test.iloc[10:15].values
        
        print("\n\n=== Running Predictions on 5 Test Samples ===")
        
        # Get raw probabilities and make prediction using optimal threshold (0.88)
        probs = self.model.predict_proba(X_5)[:, 1]
        preds = (probs >= 0.88).astype(int)
        
        # Display the result of each sample
        correct_count = 0
        for i in range(5):
            actual = y_5_actual[i]
            predicted = preds[i]
            prob = probs[i]
            status = "Correct" if actual == predicted else "Incorrect"
            if actual == predicted:
                correct_count += 1
            print(f"Sample {i+1}: Actual Delayed={actual} | Predicted={predicted} (Delay Prob={prob:.4f}) -> {status}")
            
        print(f"Accuracy on these 5 samples: {correct_count / 5 * 100:.1f}%")
        print("=============================================\n")
        
        # Ensure at least 3 out of 5 are predicted correctly
        self.assertGreaterEqual(correct_count, 3, "Model predicted fewer than 3/5 samples correctly.")

if __name__ == "__main__":
    unittest.main()
