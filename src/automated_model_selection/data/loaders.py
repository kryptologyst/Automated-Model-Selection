"""Core data loading and preprocessing utilities for automated model selection."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


def load_dataset(
    dataset_name: str = "iris",
    test_size: float = 0.2,
    random_state: int = 42,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load and preprocess a dataset for model selection.
    
    Args:
        dataset_name: Name of the dataset to load ('iris', 'wine', 'breast_cancer').
        test_size: Proportion of data to use for testing.
        random_state: Random state for reproducibility.
        normalize: Whether to normalize features.
        
    Returns:
        Tuple containing (X_train, X_test, y_train, y_test, feature_names).
    """
    set_seed(random_state)
    
    # Load dataset
    if dataset_name == "iris":
        data = load_iris()
    elif dataset_name == "wine":
        data = load_wine()
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalize features if requested
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info("Features normalized using StandardScaler")
    
    logger.info(f"Loaded {dataset_name} dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test, feature_names


def create_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 3,
    noise: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Create a synthetic dataset for testing AutoML methods.
    
    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features.
        n_classes: Number of classes.
        noise: Amount of noise to add.
        random_state: Random state for reproducibility.
        
    Returns:
        Tuple containing (X, y, feature_names).
    """
    set_seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target with some structure
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if X[i, 0] + X[i, 1] > 0:
            y[i] = 0
        elif X[i, 2] + X[i, 3] > 0:
            y[i] = 1
        else:
            y[i] = 2
    
    # Add noise
    noise_mask = np.random.random(n_samples) < noise
    y[noise_mask] = np.random.randint(0, n_classes, size=np.sum(noise_mask))
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    logger.info(f"Created synthetic dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
    return X, y, feature_names


def save_results(
    results: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = "json",
) -> None:
    """Save evaluation results to file.
    
    Args:
        results: Results dictionary to save.
        filepath: Path to save the results.
        format: Format to save in ('json', 'csv', 'pickle').
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        import json
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
    elif format == "csv":
        if isinstance(results, dict):
            df = pd.DataFrame([results])
            df.to_csv(filepath, index=False)
        else:
            pd.DataFrame(results).to_csv(filepath, index=False)
    elif format == "pickle":
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Results saved to {filepath}")


def load_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load evaluation results from file.
    
    Args:
        filepath: Path to load results from.
        
    Returns:
        Loaded results dictionary.
    """
    filepath = Path(filepath)
    
    if filepath.suffix == ".json":
        import json
        with open(filepath, "r") as f:
            return json.load(f)
    elif filepath.suffix == ".csv":
        return pd.read_csv(filepath).to_dict("records")[0]
    elif filepath.suffix in [".pkl", ".pickle"]:
        import pickle
        with open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
