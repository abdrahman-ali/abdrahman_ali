"""
Experiment tracking utilities using MLflow.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Experiment tracking using MLflow."""

    def __init__(self, experiment_name: str = "default",
                 tracking_uri: str = "mlruns",
                 artifact_location: Optional[str] = None):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI
            artifact_location: Artifact storage location
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required. Install with: pip install mlflow")

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        except Exception:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        logger.info(f"Experiment: {experiment_name} (ID: {self.experiment_id})")

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new run.

        Args:
            run_name: Name of the run
            tags: Tags to add to the run
        """
        mlflow.start_run(run_name=run_name)

        if tags:
            mlflow.set_tags(tags)

        logger.info(f"Started run: {run_name}")

    def end_run(self):
        """End the current run."""
        mlflow.end_run()
        logger.info("Run ended")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.

        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Step number
        """
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Step number
        """
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, artifact_path: str):
        """
        Log an artifact.

        Args:
            artifact_path: Path to the artifact
        """
        mlflow.log_artifact(artifact_path)

    def log_artifacts(self, artifacts_dir: str):
        """
        Log multiple artifacts.

        Args:
            artifacts_dir: Directory containing artifacts
        """
        mlflow.log_artifacts(artifacts_dir)

    def log_model(self, model: Any, model_name: str = "model",
                 framework: str = "sklearn"):
        """
        Log a model.

        Args:
            model: Model to log
            model_name: Name of the model
            framework: Framework (sklearn, pytorch, tensorflow)
        """
        if framework == "sklearn":
            mlflow.sklearn.log_model(model, model_name)
        elif framework == "pytorch":
            mlflow.pytorch.log_model(model, model_name)
        elif framework == "tensorflow":
            mlflow.tensorflow.log_model(model, model_name)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        logger.info(f"Model logged: {model_name}")

    def log_dict(self, dictionary: Dict, artifact_file: str):
        """
        Log a dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Artifact file name
        """
        mlflow.log_dict(dictionary, artifact_file)

    def set_tag(self, key: str, value: str):
        """
        Set a tag.

        Args:
            key: Tag key
            value: Tag value
        """
        mlflow.set_tag(key, value)

    def set_tags(self, tags: Dict[str, str]):
        """
        Set multiple tags.

        Args:
            tags: Dictionary of tags
        """
        mlflow.set_tags(tags)

    def log_figure(self, figure, artifact_file: str):
        """
        Log a matplotlib figure.

        Args:
            figure: Matplotlib figure
            artifact_file: Artifact file name
        """
        mlflow.log_figure(figure, artifact_file)

    def get_run_id(self) -> str:
        """
        Get current run ID.

        Returns:
            Run ID
        """
        return mlflow.active_run().info.run_id if mlflow.active_run() else None


class SimpleTracker:
    """Simple file-based experiment tracker (fallback when MLflow is not available)."""

    def __init__(self, experiment_name: str = "default",
                 tracking_dir: str = "experiments"):
        """
        Initialize simple tracker.

        Args:
            experiment_name: Name of the experiment
            tracking_dir: Directory to store experiments
        """
        self.experiment_name = experiment_name
        self.tracking_dir = Path(tracking_dir) / experiment_name
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

        self.current_run = None
        self.run_dir = None

        logger.info(f"Simple tracker initialized: {self.tracking_dir}")

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new run.

        Args:
            run_name: Name of the run
            tags: Tags to add to the run
        """
        import time
        run_id = f"run_{int(time.time())}"
        self.current_run = run_name or run_id

        self.run_dir = self.tracking_dir / self.current_run
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save tags
        if tags:
            self._save_json(tags, "tags.json")

        logger.info(f"Started run: {self.current_run}")

    def end_run(self):
        """End the current run."""
        self.current_run = None
        self.run_dir = None
        logger.info("Run ended")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        self._save_json(params, "params.json")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        metrics_file = self.run_dir / "metrics.json"

        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []

        all_metrics.append({"step": step, "metrics": metrics})

        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        self.log_metrics({key: value}, step)

    def _save_json(self, data: Dict, filename: str):
        """Save dictionary as JSON."""
        if self.run_dir is None:
            logger.warning("No active run")
            return

        filepath = self.run_dir / filename

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def get_tracker(experiment_name: str = "default",
               tracking_uri: str = "mlruns",
               use_mlflow: bool = True) -> Any:
    """
    Get experiment tracker.

    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI
        use_mlflow: Whether to use MLflow

    Returns:
        Experiment tracker
    """
    if use_mlflow and MLFLOW_AVAILABLE:
        return ExperimentTracker(experiment_name, tracking_uri)
    else:
        logger.warning("MLflow not available, using simple tracker")
        return SimpleTracker(experiment_name, tracking_uri)
