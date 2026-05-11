"""Evaluation metrics and utilities for automated model selection."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation utility."""
    
    def __init__(self, metrics: Optional[List[str]] = None) -> None:
        """Initialize model evaluator.
        
        Args:
            metrics: List of metrics to compute. If None, uses default metrics.
        """
        self.metrics = metrics or [
            "accuracy",
            "precision_macro",
            "recall_macro", 
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        ]
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate model predictions.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional).
            
        Returns:
            Dictionary of metric scores.
        """
        results = {}
        
        for metric in self.metrics:
            if metric == "accuracy":
                results[metric] = accuracy_score(y_true, y_pred)
            elif metric == "precision_macro":
                results[metric] = precision_score(y_true, y_pred, average="macro", zero_division=0)
            elif metric == "recall_macro":
                results[metric] = recall_score(y_true, y_pred, average="macro", zero_division=0)
            elif metric == "f1_macro":
                results[metric] = f1_score(y_true, y_pred, average="macro", zero_division=0)
            elif metric == "precision_weighted":
                results[metric] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            elif metric == "recall_weighted":
                results[metric] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            elif metric == "f1_weighted":
                results[metric] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            elif metric == "roc_auc" and y_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:
                        results[metric] = roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        results[metric] = roc_auc_score(y_true, y_proba, multi_class="ovr")
                except ValueError:
                    results[metric] = 0.0
            else:
                logger.warning(f"Unknown metric: {metric}")
        
        return results
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """Get detailed classification report.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            target_names: Names of target classes.
            
        Returns:
            Classification report string.
        """
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """Get confusion matrix.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            
        Returns:
            Confusion matrix.
        """
        return confusion_matrix(y_true, y_pred)


class Leaderboard:
    """Model leaderboard for comparing different approaches."""
    
    def __init__(self) -> None:
        """Initialize leaderboard."""
        self.results: List[Dict[str, Any]] = []
    
    def add_result(
        self,
        model_name: str,
        metrics: Dict[str, float],
        params: Optional[Dict[str, Any]] = None,
        training_time: Optional[float] = None,
        inference_time: Optional[float] = None,
    ) -> None:
        """Add a model result to the leaderboard.
        
        Args:
            model_name: Name of the model.
            metrics: Dictionary of metric scores.
            params: Model parameters (optional).
            training_time: Training time in seconds (optional).
            inference_time: Inference time in seconds (optional).
        """
        result = {
            "model_name": model_name,
            **metrics,
        }
        
        if params is not None:
            result["params"] = params
        if training_time is not None:
            result["training_time"] = training_time
        if inference_time is not None:
            result["inference_time"] = inference_time
        
        self.results.append(result)
        logger.info(f"Added result for {model_name}")
    
    def get_leaderboard(
        self,
        sort_by: str = "accuracy",
        ascending: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get sorted leaderboard.
        
        Args:
            sort_by: Metric to sort by.
            ascending: Whether to sort in ascending order.
            
        Returns:
            Sorted list of results.
        """
        if not self.results:
            return []
        
        sorted_results = sorted(
            self.results,
            key=lambda x: x.get(sort_by, 0),
            reverse=not ascending,
        )
        
        return sorted_results
    
    def get_best_model(self, metric: str = "accuracy") -> Optional[Dict[str, Any]]:
        """Get the best model by a specific metric.
        
        Args:
            metric: Metric to optimize.
            
        Returns:
            Best model result or None if no results.
        """
        if not self.results:
            return None
        
        return max(self.results, key=lambda x: x.get(metric, 0))
    
    def to_dataframe(self) -> "pd.DataFrame":
        """Convert leaderboard to pandas DataFrame.
        
        Returns:
            DataFrame representation of leaderboard.
        """
        import pandas as pd
        return pd.DataFrame(self.results)
