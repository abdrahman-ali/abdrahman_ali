"""
Machine Learning model training utilities.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from typing import Dict, List, Optional, Any, Tuple
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class MLTrainer:
    """Machine Learning model trainer supporting multiple algorithms."""

    def __init__(self, task_type: str = 'classification', config: Optional[Dict] = None):
        """
        Initialize ML trainer.

        Args:
            task_type: Type of ML task ('classification' or 'regression')
            config: Configuration dictionary
        """
        self.task_type = task_type
        self.config = config or {}
        self.models = {}
        self.trained_models = {}

    def get_model(self, model_name: str, params: Optional[Dict] = None):
        """
        Get model instance by name.

        Args:
            model_name: Name of the model
            params: Model parameters

        Returns:
            Model instance
        """
        params = params or {}

        models_dict = {
            'classification': {
                'random_forest': RandomForestClassifier,
                'logistic_regression': LogisticRegression,
                'svc': SVC,
                'gradient_boosting': GradientBoostingClassifier,
                'decision_tree': DecisionTreeClassifier,
                'knn': KNeighborsClassifier,
                'naive_bayes': GaussianNB,
                'xgboost': XGBClassifier if XGBOOST_AVAILABLE else None,
                'lightgbm': LGBMClassifier if LIGHTGBM_AVAILABLE else None,
                'catboost': CatBoostClassifier if CATBOOST_AVAILABLE else None,
            },
            'regression': {
                'random_forest': RandomForestRegressor,
                'linear_regression': LinearRegression,
                'ridge': Ridge,
                'lasso': Lasso,
                'svr': SVR,
                'gradient_boosting': GradientBoostingRegressor,
                'decision_tree': DecisionTreeRegressor,
                'knn': KNeighborsRegressor,
                'xgboost': XGBRegressor if XGBOOST_AVAILABLE else None,
                'lightgbm': LGBMRegressor if LIGHTGBM_AVAILABLE else None,
                'catboost': CatBoostRegressor if CATBOOST_AVAILABLE else None,
            }
        }

        model_class = models_dict.get(self.task_type, {}).get(model_name)

        if model_class is None:
            raise ValueError(f"Model {model_name} not available for {self.task_type}")

        return model_class(**params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              model_name: str, params: Optional[Dict] = None) -> Any:
        """
        Train a single model.

        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            params: Model parameters

        Returns:
            Trained model
        """
        logger.info(f"Training {model_name} model...")

        model = self.get_model(model_name, params)
        model.fit(X_train, y_train)

        self.trained_models[model_name] = model
        logger.info(f"{model_name} training completed")

        return model

    def train_multiple(self, X_train: pd.DataFrame, y_train: pd.Series,
                      models_config: List[Dict]) -> Dict[str, Any]:
        """
        Train multiple models.

        Args:
            X_train: Training features
            y_train: Training target
            models_config: List of model configurations

        Returns:
            Dictionary of trained models
        """
        logger.info(f"Training {len(models_config)} models...")

        for model_config in models_config:
            if not model_config.get('enabled', True):
                continue

            model_name = model_config['name']
            params = model_config.get('params', {})

            try:
                self.train(X_train, y_train, model_name, params)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue

        logger.info(f"Completed training {len(self.trained_models)} models")
        return self.trained_models

    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with a trained model.

        Args:
            model_name: Name of the model
            X: Features to predict

        Returns:
            Predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")

        return self.trained_models[model_name].predict(X)

    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            model_name: Name of the model
            X: Features to predict

        Returns:
            Prediction probabilities
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.trained_models[model_name]

        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model {model_name} does not support predict_proba")

        return model.predict_proba(X)

    def get_feature_importance(self, model_name: str) -> Optional[np.ndarray]:
        """
        Get feature importances from trained model.

        Args:
            model_name: Name of the model

        Returns:
            Feature importances or None
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.trained_models[model_name]

        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_).flatten()
        else:
            logger.warning(f"Model {model_name} does not have feature importances")
            return None

    def save_model(self, model_name: str, path: str):
        """
        Save trained model to disk.

        Args:
            model_name: Name of the model
            path: Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self.trained_models[model_name], f)

        logger.info(f"Model {model_name} saved to {path}")

    def load_model(self, model_name: str, path: str):
        """
        Load trained model from disk.

        Args:
            model_name: Name to assign to the loaded model
            path: Path to the model file
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)

        self.trained_models[model_name] = model
        logger.info(f"Model loaded from {path} as {model_name}")

    def save_all_models(self, output_dir: str):
        """
        Save all trained models.

        Args:
            output_dir: Directory to save models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for model_name in self.trained_models:
            model_path = output_dir / f"{model_name}.pkl"
            self.save_model(model_name, str(model_path))

        logger.info(f"All models saved to {output_dir}")


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                model_name: str, task_type: str = 'classification',
                params: Optional[Dict] = None) -> Any:
    """
    Convenience function to train a single model.

    Args:
        X_train: Training features
        y_train: Training target
        model_name: Name of the model
        task_type: Type of task
        params: Model parameters

    Returns:
        Trained model
    """
    trainer = MLTrainer(task_type)
    return trainer.train(X_train, y_train, model_name, params)
