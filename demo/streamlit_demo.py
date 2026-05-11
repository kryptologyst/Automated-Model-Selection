"""Interactive Streamlit demo for automated model selection."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

from automated_model_selection.data import create_synthetic_dataset, set_seed
from automated_model_selection.metrics import Leaderboard, ModelEvaluator
from automated_model_selection.models import (
    LogisticRegressionBaseline,
    OptunaClassifier,
    RandomForestBaseline,
    SVMBaseline,
    TPOTClassifier,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Automated Model Selection Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Safety disclaimer
st.warning(
    "⚠️ **RESEARCH DEMO ONLY** - This tool is for educational and research purposes only. "
    "Not intended for production decisions or critical applications. "
    "Always validate results with domain experts and proper testing procedures."
)

# Title and description
st.title("🤖 Automated Model Selection Demo")
st.markdown(
    "This demo showcases automated model selection using TPOT and Optuna for hyperparameter optimization. "
    "Compare baseline models with AutoML approaches on various datasets."
)

# Sidebar configuration
st.sidebar.header("Configuration")

# Dataset selection
dataset_options = {
    "Iris": "iris",
    "Wine": "wine", 
    "Breast Cancer": "breast_cancer",
    "Synthetic": "synthetic",
}

selected_dataset = st.sidebar.selectbox(
    "Select Dataset",
    list(dataset_options.keys()),
    index=0,
)

# Model selection
st.sidebar.subheader("Models to Run")
run_baselines = st.sidebar.checkbox("Baseline Models", value=True)
run_tpot = st.sidebar.checkbox("TPOT", value=True)
run_optuna = st.sidebar.checkbox("Optuna", value=True)

# TPOT configuration
if run_tpot:
    st.sidebar.subheader("TPOT Configuration")
    tpot_generations = st.sidebar.slider("Generations", 1, 10, 5)
    tpot_population = st.sidebar.slider("Population Size", 10, 50, 20)

# Optuna configuration
if run_optuna:
    st.sidebar.subheader("Optuna Configuration")
    optuna_trials = st.sidebar.slider("Number of Trials", 10, 200, 100)

# Random seed
random_seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=9999)

# Main content
if st.button("🚀 Run Model Selection", type="primary"):
    with st.spinner("Running automated model selection..."):
        # Set random seed
        set_seed(random_seed)
        
        # Load dataset
        if selected_dataset == "Synthetic":
            X, y, feature_names = create_synthetic_dataset(
                n_samples=1000,
                n_features=10,
                n_classes=3,
                random_state=random_seed,
            )
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_seed, stratify=y
            )
        else:
            dataset_name = dataset_options[selected_dataset]
            if dataset_name == "iris":
                data = load_iris()
            elif dataset_name == "wine":
                data = load_wine()
            elif dataset_name == "breast_cancer":
                data = load_breast_cancer()
            
            X = data.data
            y = data.target
            feature_names = data.feature_names
            
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_seed, stratify=y
            )
            
            # Normalize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Initialize evaluator and leaderboard
        evaluator = ModelEvaluator()
        leaderboard = Leaderboard()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_models = sum([run_baselines, run_tpot, run_optuna])
        if run_baselines:
            total_models += 2  # 3 baselines
        
        current_model = 0
        
        # Run baseline models
        if run_baselines:
            status_text.text("Training baseline models...")
            
            baselines = [
                ("Logistic Regression", LogisticRegressionBaseline()),
                ("Random Forest", RandomForestBaseline()),
                ("SVM", SVMBaseline()),
            ]
            
            for name, model in baselines:
                current_model += 1
                progress_bar.progress(current_model / total_models)
                
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
        
        # Run TPOT
        if run_tpot:
            current_model += 1
            progress_bar.progress(current_model / total_models)
            status_text.text("Training TPOT...")
            
            tpot = TPOTClassifier(
                generations=tpot_generations,
                population_size=tpot_population,
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
        
        # Run Optuna
        if run_optuna:
            current_model += 1
            progress_bar.progress(current_model / total_models)
            status_text.text("Training Optuna...")
            
            optuna_model = OptunaClassifier(
                n_trials=optuna_trials,
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
        
        progress_bar.progress(1.0)
        status_text.text("✅ Model selection completed!")
        
        # Display results
        st.header("📊 Results")
        
        # Get leaderboard
        results_df = leaderboard.to_dataframe()
        
        # Sort by accuracy
        results_df = results_df.sort_values("accuracy", ascending=False)
        
        # Display leaderboard
        st.subheader("Leaderboard")
        st.dataframe(results_df, use_container_width=True)
        
        # Best model
        best_model = leaderboard.get_best_model("accuracy")
        if best_model:
            st.success(f"🏆 Best Model: **{best_model['model_name']}** (Accuracy: {best_model['accuracy']:.4f})")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accuracy Comparison")
            fig = px.bar(
                results_df,
                x="model_name",
                y="accuracy",
                title="Model Accuracy Comparison",
                color="accuracy",
                color_continuous_scale="viridis",
            )
            fig.update_layout(xaxis_title="Model", yaxis_title="Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Training Time Comparison")
            fig = px.bar(
                results_df,
                x="model_name",
                y="training_time",
                title="Training Time Comparison",
                color="training_time",
                color_continuous_scale="plasma",
            )
            fig.update_layout(xaxis_title="Model", yaxis_title="Training Time (seconds)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Metrics comparison
        st.subheader("Detailed Metrics Comparison")
        metrics_to_plot = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        
        fig = go.Figure()
        for metric in metrics_to_plot:
            if metric in results_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=results_df["model_name"],
                        y=results_df[metric],
                        mode="markers+lines",
                        name=metric.replace("_", " ").title(),
                        marker=dict(size=10),
                    )
                )
        
        fig.update_layout(
            title="Metrics Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Dataset info
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Test Samples", len(X_test))
        with col3:
            st.metric("Features", len(feature_names))
        
        # Feature names
        st.write("**Feature Names:**")
        st.write(", ".join(feature_names))

# Footer
st.markdown("---")
st.markdown(
    "**Author:** [kryptologyst](https://github.com/kryptologyst) | "
    "**GitHub:** https://github.com/kryptologyst"
)
st.markdown(
    "*This demo is for educational and research purposes only. "
    "Not intended for production use without proper validation.*"
)
