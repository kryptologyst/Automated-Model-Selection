# Automated Model Selection

A comprehensive framework for automated model selection using TPOT and Optuna, featuring baseline models, advanced AutoML methods, and interactive visualization tools.

**Author:** [kryptologyst](https://github.com/kryptologyst)  
**GitHub:** https://github.com/kryptologyst

## Overview

This project demonstrates automated model selection (AutoML) techniques for classification tasks. It compares traditional baseline models with advanced AutoML approaches including TPOT (Tree-based Pipeline Optimization Tool) and Optuna for hyperparameter optimization.

### Key Features

- **Baseline Models**: Logistic Regression, Random Forest, SVM
- **AutoML Methods**: TPOT and Optuna optimization
- **Comprehensive Evaluation**: Multiple metrics and leaderboard
- **Interactive Demo**: Streamlit-based visualization
- **Safety & Ethics**: Built-in safety checks and disclaimers
- **Reproducible**: Deterministic seeding and configuration management

## Safety Disclaimer

⚠️ **RESEARCH DEMO ONLY** - This tool is for educational and research purposes only. Not intended for production decisions or critical applications. Always validate results with domain experts and proper testing procedures.

### Important Considerations

- **Data Privacy**: Ensure all data is properly anonymized and de-identified
- **Model Validation**: Always validate models on independent test sets
- **Bias Detection**: Test for demographic parity and equalized odds
- **Transparency**: Provide model explanations and feature importance
- **Monitoring**: Implement continuous performance monitoring
- **Human Oversight**: Maintain appropriate human oversight for critical decisions

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Automated-Model-Selection.git
cd Automated-Model-Selection
```

2. Install dependencies:
```bash
pip install -e .
```

3. Install development dependencies (optional):
```bash
pip install -e ".[dev]"
```

## Quick Start

### Command Line Usage

Run automated model selection on the Iris dataset:

```bash
python -m automated_model_selection.train --dataset iris --output_dir results
```

Available datasets:
- `iris`: Iris flower classification
- `wine`: Wine quality classification  
- `breast_cancer`: Breast cancer diagnosis
- `synthetic`: Generated synthetic dataset

### Configuration

Modify `configs/default.yaml` to customize:

```yaml
dataset: iris
test_size: 0.2
random_state: 42

automl:
  run_tpot: true
  run_optuna: true
  
  tpot:
    generations: 5
    population_size: 20
    
  optuna:
    n_trials: 100
```

### Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/streamlit_demo.py
```

The demo provides:
- Dataset selection and configuration
- Real-time model training and evaluation
- Interactive visualizations and leaderboards
- Safety warnings and validation checks

## Project Structure

```
automated-model-selection/
├── src/automated_model_selection/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model implementations
│   ├── metrics/        # Evaluation metrics and leaderboard
│   ├── utils/          # Utility functions and safety checks
│   └── train.py        # Main training script
├── configs/            # Configuration files
├── demo/               # Interactive demos
├── tests/              # Unit tests
├── assets/             # Generated outputs and visualizations
└── data/               # Data storage
    ├── raw/            # Raw datasets
    └── processed/      # Processed datasets
```

## Usage Examples

### Basic Model Selection

```python
from automated_model_selection.data import load_dataset
from automated_model_selection.models import TPOTClassifier, OptunaClassifier
from automated_model_selection.metrics import ModelEvaluator, Leaderboard

# Load data
X_train, X_test, y_train, y_test, feature_names = load_dataset("iris")

# Initialize evaluator and leaderboard
evaluator = ModelEvaluator()
leaderboard = Leaderboard()

# Train TPOT
tpot = TPOTClassifier(generations=5, population_size=20)
tpot.fit(X_train, y_train)
y_pred = tpot.predict(X_test)
metrics = evaluator.evaluate(y_test, y_pred)
leaderboard.add_result("TPOT", metrics)

# Train Optuna
optuna_model = OptunaClassifier(n_trials=100)
optuna_model.fit(X_train, y_train)
y_pred = optuna_model.predict(X_test)
metrics = evaluator.evaluate(y_test, y_pred)
leaderboard.add_result("Optuna", metrics)

# Get results
print(leaderboard.get_leaderboard())
```

### Custom Dataset

```python
from automated_model_selection.data import create_synthetic_dataset

# Create synthetic dataset
X, y, feature_names = create_synthetic_dataset(
    n_samples=1000,
    n_features=10,
    n_classes=3,
    noise=0.1
)

# Use with any model
model = TPOTClassifier()
model.fit(X, y)
```

## Evaluation Metrics

The framework evaluates models using multiple metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Macro and weighted averages
- **Recall**: Macro and weighted averages  
- **F1-Score**: Macro and weighted averages
- **ROC-AUC**: Area under ROC curve (when probabilities available)

### Expected Performance Ranges

| Dataset | Baseline Accuracy | AutoML Accuracy |
|---------|------------------|-----------------|
| Iris | 0.90-0.98 | 0.95-0.99 |
| Wine | 0.85-0.95 | 0.90-0.98 |
| Breast Cancer | 0.90-0.97 | 0.93-0.99 |

*Note: Performance may vary based on random seed and configuration*

## Advanced Configuration

### TPOT Configuration

```python
tpot = TPOTClassifier(
    generations=10,           # Number of generations
    population_size=50,        # Population size per generation
    verbosity=2,              # Verbosity level
    random_state=42           # Random seed
)
```

### Optuna Configuration

```python
optuna_model = OptunaClassifier(
    n_trials=200,             # Number of optimization trials
    timeout=3600,             # Timeout in seconds
    random_state=42           # Random seed
)
```

## Safety and Ethics

### Built-in Safety Checks

The framework includes automatic safety validation:

```python
from automated_model_selection.utils import SafetyChecker, validate_model_outputs

# Check dataset safety
safety_checker = SafetyChecker()
safety_checker.check_dataset_safety("breast_cancer", X, y)

# Validate model outputs
validation_results = validate_model_outputs(y_true, y_pred, "TPOT")
if not validation_results["is_safe"]:
    print("Safety warnings:", validation_results["warnings"])
```

### Ethics Guidelines

1. **Data Handling**: Always anonymize sensitive data
2. **Bias Testing**: Test for demographic parity and equalized odds
3. **Transparency**: Document model decisions and limitations
4. **Validation**: Use independent test sets and cross-validation
5. **Monitoring**: Implement continuous performance monitoring
6. **Human Oversight**: Maintain human control over critical decisions

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- TPOT: Tree-based Pipeline Optimization Tool
- Optuna: Hyperparameter optimization framework
- scikit-learn: Machine learning library
- Streamlit: Interactive web app framework

## Citation

If you use this project in your research, please cite:

```bibtex
@software{automated_model_selection,
  title={Automated Model Selection Framework},
  author={kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Automated-Model-Selection}
}
```

# Automated-Model-Selection
