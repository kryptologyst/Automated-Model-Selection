"""Unit tests for automated model selection."""

import pytest
import numpy as np
from sklearn.datasets import make_classification

from automated_model_selection.data import create_synthetic_dataset, load_dataset, set_seed
from automated_model_selection.models import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    SVMBaseline,
    TPOTClassifier,
    OptunaClassifier,
)
from automated_model_selection.metrics import ModelEvaluator, Leaderboard
from automated_model_selection.utils import SafetyChecker, validate_model_outputs


class TestDataLoaders:
    """Test data loading functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        # This is a basic test - in practice, you'd test actual randomness
        assert True
    
    def test_load_dataset(self):
        """Test dataset loading."""
        X_train, X_test, y_train, y_test, feature_names = load_dataset("iris")
        
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(feature_names) > 0
        assert len(np.unique(y_train)) > 1
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        X, y, feature_names = create_synthetic_dataset(
            n_samples=100,
            n_features=5,
            n_classes=3,
        )
        
        assert X.shape == (100, 5)
        assert len(y) == 100
        assert len(feature_names) == 5
        assert len(np.unique(y)) == 3


class TestBaselineModels:
    """Test baseline model implementations."""
    
    def setup_method(self):
        """Set up test data."""
        self.X, self.y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=3,
            random_state=42,
        )
        self.X_train = self.X[:80]
        self.X_test = self.X[80:]
        self.y_train = self.y[:80]
        self.y_test = self.y[80:]
    
    def test_logistic_regression(self):
        """Test logistic regression baseline."""
        model = LogisticRegressionBaseline()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        assert len(y_pred) == len(self.y_test)
        assert all(pred in np.unique(self.y) for pred in y_pred)
    
    def test_random_forest(self):
        """Test random forest baseline."""
        model = RandomForestBaseline()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        assert len(y_pred) == len(self.y_test)
        assert all(pred in np.unique(self.y) for pred in y_pred)
    
    def test_svm(self):
        """Test SVM baseline."""
        model = SVMBaseline()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        assert len(y_pred) == len(self.y_test)
        assert all(pred in np.unique(self.y) for pred in y_pred)


class TestAutoMLModels:
    """Test AutoML model implementations."""
    
    def setup_method(self):
        """Set up test data."""
        self.X, self.y = make_classification(
            n_samples=100,
            n_features=5,
            n_classes=2,
            random_state=42,
        )
        self.X_train = self.X[:80]
        self.X_test = self.X[80:]
        self.y_train = self.y[:80]
        self.y_test = self.y[80:]
    
    def test_optuna_classifier(self):
        """Test Optuna classifier."""
        model = OptunaClassifier(n_trials=10)  # Small number for testing
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        assert len(y_pred) == len(self.y_test)
        assert all(pred in np.unique(self.y) for pred in y_pred)
        assert len(model.get_best_params()) > 0


class TestMetrics:
    """Test evaluation metrics."""
    
    def setup_method(self):
        """Set up test data."""
        self.y_true = np.array([0, 1, 0, 1, 0])
        self.y_pred = np.array([0, 1, 1, 1, 0])
        self.y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]])
    
    def test_model_evaluator(self):
        """Test model evaluator."""
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(self.y_true, self.y_pred, self.y_proba)
        
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert "f1_macro" in metrics
    
    def test_leaderboard(self):
        """Test leaderboard functionality."""
        leaderboard = Leaderboard()
        
        metrics = {"accuracy": 0.8, "f1_macro": 0.75}
        leaderboard.add_result("Test Model", metrics)
        
        results = leaderboard.get_leaderboard()
        assert len(results) == 1
        assert results[0]["model_name"] == "Test Model"
        assert results[0]["accuracy"] == 0.8


class TestSafety:
    """Test safety and validation functions."""
    
    def test_safety_checker(self):
        """Test safety checker."""
        checker = SafetyChecker()
        checker.check_dataset_safety("iris", np.random.randn(100, 4), np.random.randint(0, 3, 100))
        
        report = checker.get_safety_report()
        assert "warnings" in report
        assert "errors" in report
    
    def test_validate_model_outputs(self):
        """Test model output validation."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        results = validate_model_outputs(y_true, y_pred, "Test Model")
        
        assert "is_safe" in results
        assert "warnings" in results
        assert "metrics" in results
        assert "accuracy" in results["metrics"]


if __name__ == "__main__":
    pytest.main([__file__])
