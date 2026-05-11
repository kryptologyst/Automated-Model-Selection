"""AutoML model implementations for automated model selection."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import optuna
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from tpot import TPOTClassifier

logger = logging.getLogger(__name__)


class AutoMLClassifier(ABC):
    """Abstract base class for AutoML classifiers."""
    
    def __init__(self, random_state: int = 42, **kwargs: Any) -> None:
        """Initialize AutoML classifier.
        
        Args:
            random_state: Random state for reproducibility.
            **kwargs: Additional arguments for the classifier.
        """
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.is_fitted = False
        
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying AutoML model."""
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AutoMLClassifier":
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
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found by AutoML."""
        if hasattr(self.model, 'best_params_'):
            return self.model.best_params_
        elif hasattr(self.model, 'get_params'):
            return self.model.get_params()
        else:
            return {}


class TPOTClassifier(AutoMLClassifier):
    """TPOT (Tree-based Pipeline Optimization Tool) classifier."""
    
    def __init__(
        self,
        random_state: int = 42,
        generations: int = 5,
        population_size: int = 20,
        verbosity: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialize TPOT classifier.
        
        Args:
            random_state: Random state for reproducibility.
            generations: Number of generations to run.
            population_size: Number of individuals in each generation.
            verbosity: Verbosity level.
            **kwargs: Additional arguments.
        """
        super().__init__(random_state, **kwargs)
        self.generations = generations
        self.population_size = population_size
        self.verbosity = verbosity
    
    def _create_model(self) -> TPOTClassifier:
        """Create TPOT model."""
        return TPOTClassifier(
            generations=self.generations,
            population_size=self.population_size,
            random_state=self.random_state,
            verbosity=self.verbosity,
        )
    
    def export_pipeline(self, filepath: str) -> None:
        """Export the best pipeline to a Python file.
        
        Args:
            filepath: Path to save the pipeline.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before exporting pipeline")
        self.model.export(filepath)
        logger.info(f"Pipeline exported to {filepath}")


class OptunaClassifier(AutoMLClassifier):
    """Optuna-based hyperparameter optimization classifier."""
    
    def __init__(
        self,
        random_state: int = 42,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Optuna classifier.
        
        Args:
            random_state: Random state for reproducibility.
            n_trials: Number of optimization trials.
            timeout: Timeout in seconds for optimization.
            **kwargs: Additional arguments.
        """
        super().__init__(random_state, **kwargs)
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = 0.0
    
    def _create_model(self) -> BaseEstimator:
        """Create Optuna-optimized model."""
        # This will be set during optimization
        return None
    
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object.
            X: Training features.
            y: Training targets.
            
        Returns:
            Cross-validation score to maximize.
        """
        # Choose model type
        model_type = trial.suggest_categorical("model_type", ["logistic", "random_forest", "svm"])
        
        if model_type == "logistic":
            model = LogisticRegression(
                C=trial.suggest_float("C", 0.01, 100, log=True),
                max_iter=trial.suggest_int("max_iter", 100, 2000),
                random_state=self.random_state,
            )
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 10, 200),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                random_state=self.random_state,
            )
        else:  # svm
            model = SVC(
                C=trial.suggest_float("C", 0.01, 100, log=True),
                kernel=trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
                gamma=trial.suggest_categorical("gamma", ["scale", "auto", 0.001, 0.01, 0.1, 1]),
                random_state=self.random_state,
            )
        
        # Cross-validation score
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        return scores.mean()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "OptunaClassifier":
        """Fit the model with Optuna optimization.
        
        Args:
            X: Training features.
            y: Training targets.
            
        Returns:
            Self for method chaining.
        """
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        
        # Optimize
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            timeout=self.timeout,
        )
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Create best model
        model_type = self.best_params["model_type"]
        if model_type == "logistic":
            self.model = LogisticRegression(
                C=self.best_params["C"],
                max_iter=self.best_params["max_iter"],
                random_state=self.random_state,
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=self.best_params["n_estimators"],
                max_depth=self.best_params["max_depth"],
                min_samples_split=self.best_params["min_samples_split"],
                random_state=self.random_state,
            )
        else:  # svm
            self.model = SVC(
                C=self.best_params["C"],
                kernel=self.best_params["kernel"],
                gamma=self.best_params["gamma"],
                random_state=self.random_state,
            )
        
        # Fit best model
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"Optuna optimization completed. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found by Optuna."""
        return self.best_params
    
    def get_best_score(self) -> float:
        """Get best cross-validation score."""
        return self.best_score
