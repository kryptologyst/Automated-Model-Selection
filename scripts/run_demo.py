#!/usr/bin/env python3
"""Simple script to run automated model selection demo."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from automated_model_selection.data import load_dataset, set_seed
from automated_model_selection.models import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    SVMBaseline,
    OptunaClassifier,
)
from automated_model_selection.metrics import ModelEvaluator, Leaderboard


def main():
    """Run a simple demo of automated model selection."""
    print("🤖 Automated Model Selection Demo")
    print("=" * 40)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load Iris dataset
    print("Loading Iris dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_dataset("iris")
    print(f"Dataset loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Features: {', '.join(feature_names)}")
    
    # Initialize evaluator and leaderboard
    evaluator = ModelEvaluator()
    leaderboard = Leaderboard()
    
    # Test baseline models
    print("\nTraining baseline models...")
    
    baselines = [
        ("Logistic Regression", LogisticRegressionBaseline()),
        ("Random Forest", RandomForestBaseline()),
        ("SVM", SVMBaseline()),
    ]
    
    for name, model in baselines:
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        try:
            y_proba = model.predict_proba(X_test)
        except NotImplementedError:
            y_proba = None
        
        metrics = evaluator.evaluate(y_test, y_pred, y_proba)
        leaderboard.add_result(name, metrics)
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
    
    # Test Optuna (with fewer trials for demo)
    print("\nTraining Optuna (10 trials)...")
    optuna_model = OptunaClassifier(n_trials=10)
    optuna_model.fit(X_train, y_train)
    y_pred = optuna_model.predict(X_test)
    
    try:
        y_proba = optuna_model.predict_proba(X_test)
    except NotImplementedError:
        y_proba = None
    
    metrics = evaluator.evaluate(y_test, y_pred, y_proba)
    leaderboard.add_result("Optuna", metrics)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    # Display results
    print("\n📊 Results")
    print("=" * 40)
    
    results_df = leaderboard.to_dataframe()
    results_df = results_df.sort_values("accuracy", ascending=False)
    
    print(results_df[["model_name", "accuracy", "f1_macro"]].to_string(index=False))
    
    # Best model
    best_model = leaderboard.get_best_model("accuracy")
    if best_model:
        print(f"\n🏆 Best Model: {best_model['model_name']} (Accuracy: {best_model['accuracy']:.4f})")
    
    print("\n✅ Demo completed successfully!")
    print("\nTo run the interactive Streamlit demo:")
    print("streamlit run demo/streamlit_demo.py")


if __name__ == "__main__":
    main()
