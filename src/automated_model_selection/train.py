"""Main training and evaluation script for automated model selection."""

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf

from automated_model_selection.data import load_dataset, set_seed
from automated_model_selection.metrics import Leaderboard, ModelEvaluator
from automated_model_selection.models import (
    LogisticRegressionBaseline,
    OptunaClassifier,
    RandomForestBaseline,
    SVMBaseline,
    TPOTClassifier,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_baseline_experiments(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    evaluator: ModelEvaluator,
    leaderboard: Leaderboard,
) -> None:
    """Run baseline model experiments.
    
    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training targets.
        y_test: Test targets.
        evaluator: Model evaluator.
        leaderboard: Results leaderboard.
    """
    logger.info("Running baseline experiments...")
    
    baselines = [
        ("Logistic Regression", LogisticRegressionBaseline()),
        ("Random Forest", RandomForestBaseline()),
        ("SVM", SVMBaseline()),
    ]
    
    for name, model in baselines:
        logger.info(f"Training {name}...")
        
        # Training
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Inference
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Get probabilities if available
        try:
            y_proba = model.predict_proba(X_test)
        except NotImplementedError:
            y_proba = None
        
        # Evaluate
        metrics = evaluator.evaluate(y_test, y_pred, y_proba)
        
        # Add to leaderboard
        leaderboard.add_result(
            model_name=name,
            metrics=metrics,
            params=model.get_params(),
            training_time=training_time,
            inference_time=inference_time,
        )
        
        logger.info(f"{name} completed. Accuracy: {metrics['accuracy']:.4f}")


def run_automl_experiments(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    evaluator: ModelEvaluator,
    leaderboard: Leaderboard,
    config: Dict[str, Any],
) -> None:
    """Run AutoML experiments.
    
    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training targets.
        y_test: Test targets.
        evaluator: Model evaluator.
        leaderboard: Results leaderboard.
        config: Configuration dictionary.
    """
    logger.info("Running AutoML experiments...")
    
    # TPOT experiment
    if config.get("run_tpot", True):
        logger.info("Training TPOT...")
        tpot = TPOTClassifier(
            generations=config.get("tpot_generations", 5),
            population_size=config.get("tpot_population_size", 20),
        )
        
        start_time = time.time()
        tpot.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = tpot.predict(X_test)
        inference_time = time.time() - start_time
        
        try:
            y_proba = tpot.predict_proba(X_test)
        except NotImplementedError:
            y_proba = None
        
        metrics = evaluator.evaluate(y_test, y_pred, y_proba)
        
        leaderboard.add_result(
            model_name="TPOT",
            metrics=metrics,
            params=tpot.get_best_params(),
            training_time=training_time,
            inference_time=inference_time,
        )
        
        logger.info(f"TPOT completed. Accuracy: {metrics['accuracy']:.4f}")
    
    # Optuna experiment
    if config.get("run_optuna", True):
        logger.info("Training Optuna...")
        optuna_model = OptunaClassifier(
            n_trials=config.get("optuna_trials", 100),
            timeout=config.get("optuna_timeout", None),
        )
        
        start_time = time.time()
        optuna_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = optuna_model.predict(X_test)
        inference_time = time.time() - start_time
        
        try:
            y_proba = optuna_model.predict_proba(X_test)
        except NotImplementedError:
            y_proba = None
        
        metrics = evaluator.evaluate(y_test, y_pred, y_proba)
        
        leaderboard.add_result(
            model_name="Optuna",
            metrics=metrics,
            params=optuna_model.get_best_params(),
            training_time=training_time,
            inference_time=inference_time,
        )
        
        logger.info(f"Optuna completed. Accuracy: {metrics['accuracy']:.4f}")


def main() -> None:
    """Main training and evaluation function."""
    parser = argparse.ArgumentParser(description="Automated Model Selection")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--dataset", type=str, default="iris", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = OmegaConf.load(config_path)
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        config = OmegaConf.create({
            "run_tpot": True,
            "run_optuna": True,
            "tpot_generations": 5,
            "tpot_population_size": 20,
            "optuna_trials": 100,
            "optuna_timeout": None,
        })
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    X_train, X_test, y_train, y_test, feature_names = load_dataset(
        dataset_name=args.dataset,
        random_state=args.seed,
    )
    
    # Initialize evaluator and leaderboard
    evaluator = ModelEvaluator()
    leaderboard = Leaderboard()
    
    # Run experiments
    run_baseline_experiments(X_train, X_test, y_train, y_test, evaluator, leaderboard)
    run_automl_experiments(X_train, X_test, y_train, y_test, evaluator, leaderboard, config)
    
    # Save results
    results_df = leaderboard.to_dataframe()
    results_df.to_csv(output_dir / "leaderboard.csv", index=False)
    
    # Print leaderboard
    logger.info("\n" + "="*50)
    logger.info("LEADERBOARD")
    logger.info("="*50)
    print(results_df.to_string(index=False))
    
    # Save best model info
    best_model = leaderboard.get_best_model("accuracy")
    if best_model:
        logger.info(f"\nBest model: {best_model['model_name']} (Accuracy: {best_model['accuracy']:.4f})")
        
        # Save best model details
        import json
        with open(output_dir / "best_model.json", "w") as f:
            json.dump(best_model, f, indent=2, default=str)


if __name__ == "__main__":
    main()
