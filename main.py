#!/usr/bin/env python3
"""
Main entry point for the Data/ML/AI Pipeline.

This script provides a comprehensive CLI for running various pipeline tasks including:
- Data loading and preprocessing
- Exploratory data analysis
- Model training (ML, DL, RL)
- Model evaluation
- Model deployment

Usage:
    python main.py --help
    python main.py --mode train --config config/config.yaml
    python main.py --mode predict --model-path models/saved_models/model.pkl --data data/test.csv
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import get_config
from src.data_engineering import load_data, validate_data, DataPipeline, split_data
from src.data_analysis import analyze_data, visualize_data
from src.ml import MLTrainer, ModelEvaluator
from src.utils.experiment_tracker import get_tracker

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/pipeline.log')
        ]
    )


def load_and_preprocess_data(config):
    """Load and preprocess data."""
    logger.info("=" * 60)
    logger.info("LOADING AND PREPROCESSING DATA")
    logger.info("=" * 60)

    # Load data
    data_path = config.get('data.raw_data_path', 'data/raw')
    # For demo, we'll assume there's a data file. Users will add their own.
    logger.info(f"Data will be loaded from: {data_path}")
    logger.info("NOTE: Add your data files to this directory")

    # Return None for now - users will add their data
    return None


def run_eda(data, config):
    """Run exploratory data analysis."""
    logger.info("=" * 60)
    logger.info("RUNNING EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    if data is None:
        logger.warning("No data provided. Skipping EDA.")
        return

    # Run EDA
    eda_report = analyze_data(data)

    # Generate visualizations
    target_col = config.get('ml.target_column')
    visualize_data(data, target_col=target_col)

    logger.info("EDA completed. Reports saved to docs/reports/figures/")


def train_ml_models(X_train, y_train, X_val, y_val, config):
    """Train machine learning models."""
    logger.info("=" * 60)
    logger.info("TRAINING MACHINE LEARNING MODELS")
    logger.info("=" * 60)

    # Initialize tracker
    tracker = get_tracker(
        experiment_name=config.get('experiment_tracking.mlflow.experiment_name', 'default')
    )

    # Get task type and models config
    task_type = config.get('ml.task_type', 'classification')
    models_config = config.get('ml.models', [])

    # Initialize trainer
    trainer = MLTrainer(task_type=task_type, config=config.config)

    # Initialize evaluator
    evaluator = ModelEvaluator(task_type=task_type)

    # Train models
    for model_config in models_config:
        if not model_config.get('enabled', True):
            continue

        model_name = model_config['name']
        params = model_config.get('params', {})

        # Start tracking run
        tracker.start_run(run_name=f"{model_name}_training")

        try:
            # Log parameters
            tracker.log_params(params)

            # Train model
            model = trainer.train(X_train, y_train, model_name, params)

            # Evaluate on validation set
            y_pred = trainer.predict(model_name, X_val)
            metrics = evaluator.evaluate(y_val, y_pred, model_name)

            # Log metrics
            tracker.log_metrics(metrics)

            # Save model
            model_path = f"models/saved_models/{model_name}.pkl"
            trainer.save_model(model_name, model_path)

            logger.info(f"Model {model_name} trained and saved successfully")

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")

        finally:
            tracker.end_run()

    # Save all models
    trainer.save_all_models("models/saved_models")

    # Print comparison
    comparison = evaluator.compare_models()
    logger.info("\n" + str(comparison))

    logger.info("Model training completed!")


def run_prediction(model_path: str, data_path: str, output_path: str = None):
    """Run predictions on new data."""
    logger.info("=" * 60)
    logger.info("RUNNING PREDICTIONS")
    logger.info("=" * 60)

    import pickle
    import pandas as pd

    # Load model
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load data
    logger.info(f"Loading data from {data_path}")
    data = load_data(data_path)

    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(data)

    # Save predictions
    if output_path:
        output = pd.DataFrame({'predictions': predictions})
        output.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    else:
        logger.info(f"Predictions:\n{predictions}")

    return predictions


def serve_model(config):
    """Start model serving API."""
    logger.info("=" * 60)
    logger.info("STARTING MODEL SERVING API")
    logger.info("=" * 60)

    from src.deployment import create_model_server

    # Get API config
    api_config = config.get('deployment.api', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    workers = api_config.get('workers', 4)

    # Create and start server
    server = create_model_server(auto_load=True)
    server.run(host=host, port=port, workers=workers)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Data Engineering, ML, DL, RL Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'predict', 'eda', 'serve', 'full'],
        help='Pipeline mode'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--data',
        type=str,
        help='Path to data file (for predict mode)'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Path to model file (for predict mode)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output path for predictions'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = get_config(args.config)

    try:
        if args.mode == 'full':
            # Run full pipeline
            logger.info("Running full pipeline...")

            # Load and preprocess data
            data = load_and_preprocess_data(config)

            if data is not None:
                # Run EDA
                run_eda(data, config)

                # Prepare data for training
                target_col = config.get('ml.target_column')
                X = data.drop(columns=[target_col])
                y = data[target_col]

                # Split data
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                    X, y,
                    test_size=config.get('data.train_test_split.test_size', 0.2),
                    val_size=config.get('data.train_test_split.validation_size', 0.1),
                    random_state=config.get('compute.seed', 42)
                )

                # Preprocess data
                pipeline = DataPipeline(config.config)
                X_train = pipeline.fit_transform(X_train)
                X_val = pipeline.transform(X_val)
                X_test = pipeline.transform(X_test)

                # Train models
                train_ml_models(X_train, y_train, X_val, y_val, config)

                logger.info("Full pipeline completed successfully!")
            else:
                logger.warning("No data loaded. Please add your data files.")

        elif args.mode == 'train':
            logger.info("Training mode selected")
            logger.info("NOTE: This is a template. Add your data loading and training logic.")
            # Users will implement their training logic here

        elif args.mode == 'predict':
            if not args.model or not args.data:
                parser.error("--model and --data are required for predict mode")

            run_prediction(args.model, args.data, args.output)

        elif args.mode == 'eda':
            data = load_and_preprocess_data(config)
            if data is not None:
                run_eda(data, config)
            else:
                logger.warning("No data loaded. Please add your data files.")

        elif args.mode == 'serve':
            serve_model(config)

    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
