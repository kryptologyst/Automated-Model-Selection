"""Safety and ethics considerations for automated model selection."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SafetyChecker:
    """Safety checker for automated model selection."""
    
    def __init__(self) -> None:
        """Initialize safety checker."""
        self.warnings: List[str] = []
        self.errors: List[str] = []
    
    def check_dataset_safety(self, dataset_name: str, X: Any, y: Any) -> None:
        """Check dataset for safety concerns.
        
        Args:
            dataset_name: Name of the dataset.
            X: Features.
            y: Targets.
        """
        import numpy as np
        
        # Check for sensitive datasets
        sensitive_datasets = ["breast_cancer", "medical", "health", "financial"]
        
        if any(sensitive in dataset_name.lower() for sensitive in sensitive_datasets):
            self.warnings.append(
                f"Dataset '{dataset_name}' contains sensitive information. "
                "Ensure proper data handling and privacy protection."
            )
        
        # Check for class imbalance
        if hasattr(y, 'shape'):
            unique, counts = np.unique(y, return_counts=True)
            min_class_ratio = counts.min() / counts.max()
            
            if min_class_ratio < 0.1:
                self.warnings.append(
                    f"Severe class imbalance detected (ratio: {min_class_ratio:.3f}). "
                    "Consider stratified sampling and appropriate metrics."
                )
    
    def check_model_safety(self, model_name: str, params: Dict[str, Any]) -> None:
        """Check model for safety concerns.
        
        Args:
            model_name: Name of the model.
            params: Model parameters.
        """
        # Check for overfitting-prone parameters
        if model_name.lower() in ["random_forest", "gradient_boosting"]:
            n_estimators = params.get("n_estimators", 100)
            if n_estimators > 1000:
                self.warnings.append(
                    f"High number of estimators ({n_estimators}) may lead to overfitting. "
                    "Consider regularization or early stopping."
                )
        
        # Check for SVM parameters
        if model_name.lower() == "svm":
            C = params.get("C", 1.0)
            if C > 100:
                self.warnings.append(
                    f"High C value ({C}) may lead to overfitting. "
                    "Consider cross-validation for parameter selection."
                )
    
    def check_automl_safety(self, automl_type: str, config: Dict[str, Any]) -> None:
        """Check AutoML configuration for safety concerns.
        
        Args:
            automl_type: Type of AutoML (tpot, optuna, etc.).
            config: AutoML configuration.
        """
        if automl_type.lower() == "tpot":
            generations = config.get("generations", 5)
            population_size = config.get("population_size", 20)
            
            if generations > 20:
                self.warnings.append(
                    f"High number of generations ({generations}) may lead to overfitting. "
                    "Consider early stopping or validation-based selection."
                )
            
            if population_size > 100:
                self.warnings.append(
                    f"Large population size ({population_size}) increases computational cost. "
                    "Consider reducing for faster experimentation."
                )
        
        elif automl_type.lower() == "optuna":
            n_trials = config.get("n_trials", 100)
            
            if n_trials > 500:
                self.warnings.append(
                    f"High number of trials ({n_trials}) may lead to overfitting. "
                    "Consider using validation-based pruning."
                )
    
    def get_safety_report(self) -> Dict[str, List[str]]:
        """Get safety report.
        
        Returns:
            Dictionary containing warnings and errors.
        """
        return {
            "warnings": self.warnings,
            "errors": self.errors,
        }
    
    def print_safety_report(self) -> None:
        """Print safety report to logger."""
        if self.warnings:
            logger.warning("SAFETY WARNINGS:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        if self.errors:
            logger.error("SAFETY ERRORS:")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        if not self.warnings and not self.errors:
            logger.info("No safety concerns detected.")


def get_ethics_disclaimer() -> str:
    """Get ethics disclaimer for AutoML.
    
    Returns:
        Ethics disclaimer text.
    """
    return """
    ETHICS AND SAFETY DISCLAIMER:
    
    This automated model selection tool is designed for educational and research purposes only.
    
    IMPORTANT CONSIDERATIONS:
    
    1. DATA PRIVACY AND SECURITY:
       - Ensure all data is properly anonymized and de-identified
       - Follow applicable data protection regulations (GDPR, CCPA, etc.)
       - Implement appropriate access controls and audit trails
       - Never use personal identifiable information (PII) without consent
    
    2. MODEL VALIDATION AND TESTING:
       - Always validate models on independent test sets
       - Perform thorough cross-validation and holdout testing
       - Test for bias, fairness, and robustness
       - Document model limitations and failure modes
    
    3. PRODUCTION DEPLOYMENT:
       - This tool is NOT intended for production use without extensive validation
       - Implement proper monitoring and alerting systems
       - Establish rollback procedures and model versioning
       - Ensure human oversight for critical decisions
    
    4. BIAS AND FAIRNESS:
       - Test models for demographic parity and equalized odds
       - Monitor for disparate impact across different groups
       - Implement bias detection and mitigation strategies
       - Document any known biases or limitations
    
    5. TRANSPARENCY AND EXPLAINABILITY:
       - Provide model explanations and feature importance
       - Document the decision-making process
       - Enable model interpretability for stakeholders
       - Maintain audit trails for regulatory compliance
    
    6. CONTINUOUS MONITORING:
       - Monitor model performance over time
       - Detect concept drift and data drift
       - Implement automated retraining procedures
       - Establish performance degradation thresholds
    
    By using this tool, you acknowledge and agree to:
    - Use it responsibly and ethically
    - Comply with all applicable laws and regulations
    - Not use it for malicious or harmful purposes
    - Validate all results before making important decisions
    - Maintain appropriate human oversight and control
    
    The authors and contributors are not responsible for any misuse or
    consequences arising from the use of this tool.
    """


def validate_model_outputs(
    y_true: Any,
    y_pred: Any,
    model_name: str,
    threshold: float = 0.8,
) -> Dict[str, Any]:
    """Validate model outputs for safety concerns.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        model_name: Name of the model.
        threshold: Minimum acceptable accuracy threshold.
        
    Returns:
        Validation results dictionary.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score
    
    results = {
        "is_safe": True,
        "warnings": [],
        "metrics": {},
    }
    
    # Check accuracy
    accuracy = accuracy_score(y_true, y_pred)
    results["metrics"]["accuracy"] = accuracy
    
    if accuracy < threshold:
        results["warnings"].append(
            f"Model accuracy ({accuracy:.3f}) below threshold ({threshold})"
        )
        results["is_safe"] = False
    
    # Check for perfect accuracy (potential overfitting)
    if accuracy > 0.99:
        results["warnings"].append(
            "Perfect or near-perfect accuracy detected. "
            "This may indicate overfitting or data leakage."
        )
    
    # Check prediction distribution
    unique_preds = len(np.unique(y_pred))
    unique_true = len(np.unique(y_true))
    
    if unique_preds != unique_true:
        results["warnings"].append(
            f"Prediction classes ({unique_preds}) don't match true classes ({unique_true})"
        )
    
    return results
