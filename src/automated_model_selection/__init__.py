"""Package initialization for automated model selection."""

__version__ = "1.0.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"
__description__ = "Automated Model Selection using TPOT and other AutoML frameworks"

from .data import load_dataset, create_synthetic_dataset, set_seed
from .models import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    SVMBaseline,
    TPOTClassifier,
    OptunaClassifier,
)
from .metrics import ModelEvaluator, Leaderboard
from .utils import SafetyChecker, get_ethics_disclaimer

__all__ = [
    "load_dataset",
    "create_synthetic_dataset", 
    "set_seed",
    "LogisticRegressionBaseline",
    "RandomForestBaseline",
    "SVMBaseline",
    "TPOTClassifier",
    "OptunaClassifier",
    "ModelEvaluator",
    "Leaderboard",
    "SafetyChecker",
    "get_ethics_disclaimer",
]
