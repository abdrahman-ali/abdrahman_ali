"""
Model evaluation utilities.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, log_loss
)
from sklearn.model_selection import cross_val_score, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation toolkit."""

    def __init__(self, task_type: str = 'classification'):
        """
        Initialize model evaluator.

        Args:
            task_type: Type of ML task ('classification' or 'regression')
        """
        self.task_type = task_type
        self.results = {}

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                model_name: str = 'model', y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate model predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            y_pred_proba: Prediction probabilities (for classification)

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")

        if self.task_type == 'classification':
            metrics = self._evaluate_classification(y_true, y_pred, y_pred_proba)
        else:
            metrics = self._evaluate_regression(y_true, y_pred)

        self.results[model_name] = metrics
        self._print_metrics(model_name, metrics)

        return metrics

    def _evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate classification model."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        # Add ROC AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            if y_pred_proba.ndim == 2:
                y_pred_proba = y_pred_proba[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

        # Add confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # Add classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, zero_division=0)

        return metrics

    def _evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Evaluate regression model."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }

        # Add MAPE if no zeros in y_true
        if not np.any(y_true == 0):
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)

        return metrics

    def cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series,
                           cv: int = 5, scoring: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            model: Model to evaluate
            X: Features
            y: Target
            cv: Number of cross-validation folds
            scoring: List of scoring metrics

        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation...")

        if scoring is None:
            if self.task_type == 'classification':
                scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            else:
                scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)

        # Process results
        results = {}
        for metric in scoring:
            results[f'{metric}_mean'] = cv_results[f'test_{metric}'].mean()
            results[f'{metric}_std'] = cv_results[f'test_{metric}'].std()

        logger.info("Cross-validation completed")
        return results

    def compare_models(self, models_results: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            models_results: Dictionary of model results (uses stored results if None)

        Returns:
            Comparison DataFrame
        """
        if models_results is None:
            models_results = self.results

        if not models_results:
            logger.warning("No models to compare")
            return pd.DataFrame()

        # Extract key metrics
        comparison_data = []

        for model_name, metrics in models_results.items():
            row = {'model': model_name}

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = value

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by primary metric
        if self.task_type == 'classification' and 'accuracy' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('accuracy', ascending=False)
        elif self.task_type == 'regression' and 'rmse' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('rmse', ascending=True)

        return comparison_df

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = 'model', save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Model name
            save_path: Path to save the plot

        Returns:
            Figure object
        """
        if self.task_type != 'classification':
            logger.warning("Confusion matrix only available for classification")
            return None

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str = 'model', save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot residuals for regression.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Model name
            save_path: Path to save the plot

        Returns:
            Figure object
        """
        if self.task_type != 'regression':
            logger.warning("Residual plot only available for regression")
            return None

        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Residual plot
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values', fontsize=12)
        axes[0].set_ylabel('Residuals', fontsize=12)
        axes[0].set_title(f'Residual Plot - {model_name}', fontsize=14, fontweight='bold')

        # Residual distribution
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residual plot saved to {save_path}")

        return fig

    def plot_model_comparison(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot model comparison.

        Args:
            save_path: Path to save the plot

        Returns:
            Figure object
        """
        comparison_df = self.compare_models()

        if comparison_df.empty:
            logger.warning("No models to compare")
            return None

        # Select metric columns
        metric_cols = [col for col in comparison_df.columns if col != 'model']

        fig, ax = plt.subplots(figsize=(12, 6))

        comparison_df.plot(x='model', y=metric_cols, kind='bar', ax=ax, rot=45)
        ax.set_title('Model Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")

        return fig

    def _print_metrics(self, model_name: str, metrics: Dict[str, Any]):
        """Print evaluation metrics."""
        logger.info(f"{'=' * 60}")
        logger.info(f"EVALUATION RESULTS - {model_name}")
        logger.info(f"{'=' * 60}")

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value:.4f}")

        logger.info(f"{'=' * 60}")


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                  task_type: str = 'classification',
                  model_name: str = 'model',
                  y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        task_type: Type of task
        model_name: Name of the model
        y_pred_proba: Prediction probabilities

    Returns:
        Evaluation metrics
    """
    evaluator = ModelEvaluator(task_type)
    return evaluator.evaluate(y_true, y_pred, model_name, y_pred_proba)
