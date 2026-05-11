"""Baseline models for automated model selection."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class BaselineClassifier(ABC):
    """Abstract base class for baseline classifiers."""
    
    def __init__(self, random_state: int = 42, **kwargs: Any) -> None:
        """Initialize baseline classifier.
        
        Args:
            random_state: Random state for reproducibility.
            **kwargs: Additional arguments for the classifier.
        """
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.is_fitted = False
        
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model."""
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineClassifier":
        """Fit the model to training data.
        
        Args:
            X: Training features.
            y: Training targets.
            
        Returns:
            Self for method chaining.
        """
        self.model = self._create_model()
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(f"Fitted {self.__class__.__name__}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on test data.
        
        Args:
            X: Test features.
            
        Returns:
            Predicted targets.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Test features.
            
        Returns:
            Predicted class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.__class__.__name__} does not support probability predictions")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is None:
            return {}
        return self.model.get_params()
    
    def set_params(self, **params: Any) -> "BaselineClassifier":
        """Set model parameters."""
        if self.model is not None:
            self.model.set_params(**params)
        return self


class LogisticRegressionBaseline(BaselineClassifier):
    """Logistic Regression baseline classifier."""
    
    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 1000,
        C: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize Logistic Regression baseline.
        
        Args:
            random_state: Random state for reproducibility.
            max_iter: Maximum number of iterations.
            C: Inverse of regularization strength.
            **kwargs: Additional arguments.
        """
        super().__init__(random_state, **kwargs)
        self.max_iter = max_iter
        self.C = C
    
    def _create_model(self) -> LogisticRegression:
        """Create Logistic Regression model."""
        return LogisticRegression(
            random_state=self.random_state,
            max_iter=self.max_iter,
            C=self.C,
        )


class RandomForestBaseline(BaselineClassifier):
    """Random Forest baseline classifier."""
    
    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Random Forest baseline.
        
        Args:
            random_state: Random state for reproducibility.
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees.
            **kwargs: Additional arguments.
        """
        super().__init__(random_state, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def _create_model(self) -> RandomForestClassifier:
        """Create Random Forest model."""
        return RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
        )


class SVMBaseline(BaselineClassifier):
    """Support Vector Machine baseline classifier."""
    
    def __init__(
        self,
        random_state: int = 42,
        C: float = 1.0,
        kernel: str = "rbf",
        probability: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize SVM baseline.
        
        Args:
            random_state: Random state for reproducibility.
            C: Regularization parameter.
            kernel: Kernel type.
            probability: Whether to enable probability estimates.
            **kwargs: Additional arguments.
        """
        super().__init__(random_state, **kwargs)
        self.C = C
        self.kernel = kernel
        self.probability = probability
    
    def _create_model(self) -> SVC:
        """Create SVM model."""
        return SVC(
            random_state=self.random_state,
            C=self.C,
            kernel=self.kernel,
            probability=self.probability,
        )
