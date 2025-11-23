#!/usr/bin/env python3
"""
Example script demonstrating how to use the ML/AI pipeline.

This script shows various ways to use the pipeline components.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from src.utils import get_config
from src.data_engineering import DataPipeline, split_data
from src.ml import MLTrainer, ModelEvaluator
from src.utils.experiment_tracker import get_tracker


def create_sample_data():
    """Create sample dataset for demonstration."""
    print("Creating sample dataset...")

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    data['target'] = y

    return data


def example_1_basic_training():
    """Example 1: Basic model training."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Model Training")
    print("="*60 + "\n")

    # Create sample data
    data = create_sample_data()

    # Prepare data
    X = data.drop('target', axis=1)
    y = data['target']

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.2, val_size=0.1, random_state=42
    )

    print(f"Train size: {X_train.shape}")
    print(f"Validation size: {X_val.shape}")
    print(f"Test size: {X_test.shape}\n")

    # Preprocess
    print("Preprocessing data...")
    pipeline = DataPipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_val_processed = pipeline.transform(X_val)
    X_test_processed = pipeline.transform(X_test)

    # Train model
    print("\nTraining Random Forest model...")
    trainer = MLTrainer(task_type='classification')
    model = trainer.train(
        X_train_processed,
        y_train,
        'random_forest',
        params={'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
    )

    # Evaluate
    print("\nEvaluating model...")
    evaluator = ModelEvaluator(task_type='classification')
    y_pred = trainer.predict('random_forest', X_test_processed)
    y_pred_proba = trainer.predict_proba('random_forest', X_test_processed)

    metrics = evaluator.evaluate(y_test, y_pred, 'random_forest', y_pred_proba)

    print("\nResults:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")


def example_2_multiple_models():
    """Example 2: Training and comparing multiple models."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Training Multiple Models")
    print("="*60 + "\n")

    # Create sample data
    data = create_sample_data()
    X = data.drop('target', axis=1)
    y = data['target']

    # Split and preprocess
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    pipeline = DataPipeline()
    X_train = pipeline.fit_transform(X_train)
    X_val = pipeline.transform(X_val)
    X_test = pipeline.transform(X_test)

    # Train multiple models
    trainer = MLTrainer(task_type='classification')
    evaluator = ModelEvaluator(task_type='classification')

    models_config = [
        {'name': 'random_forest', 'params': {'n_estimators': 100, 'random_state': 42}},
        {'name': 'logistic_regression', 'params': {'max_iter': 1000, 'random_state': 42}},
    ]

    print("Training models...")
    for model_config in models_config:
        model_name = model_config['name']
        params = model_config['params']

        print(f"\nTraining {model_name}...")
        trainer.train(X_train, y_train, model_name, params)

        # Evaluate
        y_pred = trainer.predict(model_name, X_test)
        evaluator.evaluate(y_test, y_pred, model_name)

    # Compare models
    print("\nModel Comparison:")
    comparison = evaluator.compare_models()
    print(comparison)


def example_3_with_tracking():
    """Example 3: Training with experiment tracking."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Training with Experiment Tracking")
    print("="*60 + "\n")

    # Create sample data
    data = create_sample_data()
    X = data.drop('target', axis=1)
    y = data['target']

    # Split and preprocess
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    pipeline = DataPipeline()
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    # Initialize tracker (will use simple tracker if MLflow not available)
    print("Initializing experiment tracker...")
    tracker = get_tracker(experiment_name='example_experiment', use_mlflow=False)

    # Train with tracking
    print("\nTraining with experiment tracking...")
    tracker.start_run(run_name='random_forest_example')

    # Log parameters
    params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
    tracker.log_params(params)

    # Train model
    trainer = MLTrainer(task_type='classification')
    trainer.train(X_train, y_train, 'random_forest', params)

    # Evaluate and log metrics
    evaluator = ModelEvaluator(task_type='classification')
    y_pred = trainer.predict('random_forest', X_test)
    metrics = evaluator.evaluate(y_test, y_pred, 'random_forest')

    tracker.log_metrics(metrics)
    tracker.end_run()

    print("\nExperiment tracked successfully!")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("ML/AI PIPELINE - USAGE EXAMPLES")
    print("="*60)

    try:
        example_1_basic_training()
        example_2_multiple_models()
        example_3_with_tracking()

        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
