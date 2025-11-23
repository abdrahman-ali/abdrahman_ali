"""
Hyperparameter tuning utilities using Optuna.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable
import logging

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter tuning using Optuna."""

    def __init__(self, model_class: Any, task_type: str = 'classification',
                 n_trials: int = 50, cv: int = 5):
        """
        Initialize hyperparameter tuner.

        Args:
            model_class: Model class to tune
            task_type: Type of ML task
            n_trials: Number of optimization trials
            cv: Number of cross-validation folds
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter tuning. Install with: pip install optuna")

        self.model_class = model_class
        self.task_type = task_type
        self.n_trials = n_trials
        self.cv = cv
        self.best_params = None
        self.best_score = None
        self.study = None

    def tune(self, X: pd.DataFrame, y: pd.Series,
            param_space: Dict[str, Any],
            metric: Optional[str] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna.

        Args:
            X: Training features
            y: Training target
            param_space: Parameter search space
            metric: Scoring metric

        Returns:
            Best parameters
        """
        logger.info(f"Starting hyperparameter tuning with {self.n_trials} trials...")

        # Set default metric
        if metric is None:
            metric = 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'

        def objective(trial):
            """Objective function for Optuna."""
            # Sample parameters
            params = {}
            for param_name, param_config in param_space.items():
                param_type = param_config['type']

                if param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_type == 'float':
                    if param_config.get('log', False):
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )

            # Create and evaluate model
            model = self.model_class(**params)
            score = cross_val_score(model, X, y, cv=self.cv, scoring=metric).mean()

            return score

        # Create study
        direction = 'maximize' if self.task_type == 'classification' or metric in ['r2', 'accuracy'] else 'minimize'
        self.study = optuna.create_study(direction=direction, sampler=TPESampler())

        # Optimize
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        # Get best parameters
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        logger.info(f"Hyperparameter tuning completed!")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.best_params

    def get_best_model(self):
        """
        Get model with best parameters.

        Returns:
            Model instance with best parameters
        """
        if self.best_params is None:
            raise ValueError("Must run tune() before getting best model")

        return self.model_class(**self.best_params)

    def plot_optimization_history(self):
        """Plot optimization history."""
        if self.study is None:
            raise ValueError("Must run tune() before plotting")

        fig = optuna.visualization.plot_optimization_history(self.study)
        return fig

    def plot_param_importances(self):
        """Plot parameter importances."""
        if self.study is None:
            raise ValueError("Must run tune() before plotting")

        fig = optuna.visualization.plot_param_importances(self.study)
        return fig


def get_default_param_space(model_name: str) -> Dict[str, Any]:
    """
    Get default parameter search space for common models.

    Args:
        model_name: Name of the model

    Returns:
        Parameter search space
    """
    param_spaces = {
        'random_forest': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
        },
        'xgboost': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
        },
        'lightgbm': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'num_leaves': {'type': 'int', 'low': 20, 'high': 150},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        },
        'logistic_regression': {
            'C': {'type': 'float', 'low': 0.001, 'high': 100, 'log': True},
            'penalty': {'type': 'categorical', 'choices': ['l1', 'l2']},
            'solver': {'type': 'categorical', 'choices': ['liblinear', 'saga']},
        },
        'svc': {
            'C': {'type': 'float', 'low': 0.1, 'high': 100, 'log': True},
            'kernel': {'type': 'categorical', 'choices': ['rbf', 'poly', 'sigmoid']},
            'gamma': {'type': 'categorical', 'choices': ['scale', 'auto']},
        }
    }

    return param_spaces.get(model_name, {})


def tune_model(model_class: Any, X: pd.DataFrame, y: pd.Series,
              param_space: Optional[Dict[str, Any]] = None,
              task_type: str = 'classification',
              n_trials: int = 50, cv: int = 5) -> Dict[str, Any]:
    """
    Convenience function to tune a model.

    Args:
        model_class: Model class to tune
        X: Training features
        y: Training target
        param_space: Parameter search space
        task_type: Type of task
        n_trials: Number of trials
        cv: Cross-validation folds

    Returns:
        Best parameters
    """
    tuner = HyperparameterTuner(model_class, task_type, n_trials, cv)
    return tuner.tune(X, y, param_space)
