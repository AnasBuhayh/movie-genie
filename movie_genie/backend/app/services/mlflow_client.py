"""
MLflow Client Service for Movie Genie

This service provides a Python interface to MLflow tracking server,
enabling the backend API to query experiment data, model metrics,
and model registry information for display in the frontend UI.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run, Experiment

logger = logging.getLogger(__name__)


class MLflowService:
    """
    Service class for interacting with MLflow tracking server.

    This class wraps the MLflow client and provides convenient methods
    for querying model metrics, experiments, and runs for the Movie Genie
    recommendation system.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MLflow service with configuration.

        Args:
            config_path: Path to MLflow configuration file
        """
        if config_path is None:
            # Get project root (4 levels up from this file)
            project_root = Path(__file__).parent.parent.parent.parent.parent
            config_path = project_root / "configs" / "mlflow.yaml"
        else:
            project_root = Path(__file__).parent.parent.parent.parent.parent

        self.config = self._load_config(str(config_path))

        # Set up MLflow tracking URI with absolute path
        tracking_uri = self.config['mlflow']['tracking_uri']

        # If tracking_uri is a relative file path, make it absolute
        if tracking_uri.startswith('file:./'):
            relative_path = tracking_uri.replace('file:./', '')
            absolute_path = project_root / relative_path
            tracking_uri = f'file:{absolute_path}'

        mlflow.set_tracking_uri(tracking_uri)

        # Initialize MLflow client
        self.client = MlflowClient()

        # Get experiment name
        self.experiment_name = self.config['mlflow']['default_experiment_name']

        logger.info(f"MLflow service initialized with tracking URI: {tracking_uri}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load MLflow configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """
        Get all MLflow experiments.

        Returns:
            List of experiment dictionaries with metadata
        """
        experiments = self.client.search_experiments()

        return [{
            'experiment_id': exp.experiment_id,
            'name': exp.name,
            'artifact_location': exp.artifact_location,
            'lifecycle_stage': exp.lifecycle_stage,
        } for exp in experiments]

    def get_experiment_runs(
        self,
        experiment_name: Optional[str] = None,
        filter_string: str = "",
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all runs for a specific experiment.

        Args:
            experiment_name: Name of the experiment (defaults to movie-genie-models)
            filter_string: MLflow filter string (e.g., "tags.model_type='retrieval'")
            max_results: Maximum number of runs to return

        Returns:
            List of run dictionaries with metrics and parameters
        """
        if experiment_name is None:
            experiment_name = self.experiment_name

        # Get experiment by name
        experiment = self.client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return []

        # Search runs
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=["start_time DESC"]  # Most recent first
        )

        return [self._format_run(run) for run in runs]

    def get_run_by_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific run by ID.

        Args:
            run_id: MLflow run ID

        Returns:
            Run dictionary with all metrics, parameters, and tags
        """
        try:
            run = self.client.get_run(run_id)
            return self._format_run(run)
        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            return None

    def get_model_metrics(self, run_id: str) -> Dict[str, float]:
        """
        Get all metrics for a specific model run.

        Args:
            run_id: MLflow run ID

        Returns:
            Dictionary of metric names to values
        """
        try:
            run = self.client.get_run(run_id)
            return dict(run.data.metrics)
        except Exception as e:
            logger.error(f"Failed to get metrics for run {run_id}: {e}")
            return {}

    def get_metric_history(
        self,
        run_id: str,
        metric_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get the history of a specific metric (e.g., training loss over epochs).

        Args:
            run_id: MLflow run ID
            metric_name: Name of the metric

        Returns:
            List of dicts with timestamp, step, and value
        """
        try:
            history = self.client.get_metric_history(run_id, metric_name)
            return [{
                'step': metric.step,
                'value': metric.value,
                'timestamp': metric.timestamp,
            } for metric in history]
        except Exception as e:
            logger.error(f"Failed to get metric history: {e}")
            return []

    def compare_runs(self, run_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Compare multiple model runs side-by-side.

        Args:
            run_ids: List of MLflow run IDs to compare

        Returns:
            List of run dictionaries with metrics for comparison
        """
        runs_data = []

        for run_id in run_ids:
            run_data = self.get_run_by_id(run_id)
            if run_data:
                runs_data.append(run_data)

        return runs_data

    def get_models_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        """
        Get all models of a specific type (e.g., 'retrieval', 'ranking').

        Args:
            model_type: Type of model to filter by

        Returns:
            List of run dictionaries for that model type
        """
        filter_string = f"tags.model_type='{model_type}'"
        return self.get_experiment_runs(filter_string=filter_string)

    def get_production_models(self) -> List[Dict[str, Any]]:
        """
        Get all models tagged as production/active.

        Returns:
            List of production model runs
        """
        filter_string = "tags.status='active'"
        return self.get_experiment_runs(filter_string=filter_string)

    def get_latest_run_by_model_name(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent run for a specific model name.

        Args:
            model_name: Name of the model (e.g., 'two-tower', 'bert4rec')

        Returns:
            Latest run dictionary or None
        """
        filter_string = f"tags.model_name='{model_name}'"
        runs = self.get_experiment_runs(filter_string=filter_string, max_results=1)

        return runs[0] if runs else None

    def _format_run(self, run: Run) -> Dict[str, Any]:
        """
        Format an MLflow Run object into a dictionary for API responses.

        Args:
            run: MLflow Run object

        Returns:
            Formatted dictionary with run metadata, metrics, params, and tags
        """
        return {
            'run_id': run.info.run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
            'experiment_id': run.info.experiment_id,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'artifact_uri': run.info.artifact_uri,

            # Model metadata from tags
            'model_type': run.data.tags.get('model_type', 'unknown'),
            'model_name': run.data.tags.get('model_name', 'unknown'),
            'framework': run.data.tags.get('framework', 'unknown'),
            'model_status': run.data.tags.get('status', 'inactive'),

            # All metrics
            'metrics': dict(run.data.metrics),

            # All parameters
            'params': dict(run.data.params),

            # All tags
            'tags': dict(run.data.tags),
        }

    def get_registered_models(self) -> List[Dict[str, Any]]:
        """
        Get all registered models from MLflow Model Registry.

        Returns:
            List of registered model information
        """
        try:
            registered_models = self.client.search_registered_models()

            models_info = []
            for rm in registered_models:
                latest_versions = rm.latest_versions

                model_info = {
                    'name': rm.name,
                    'creation_timestamp': rm.creation_timestamp,
                    'last_updated_timestamp': rm.last_updated_timestamp,
                    'description': rm.description,
                    'latest_versions': [{
                        'version': mv.version,
                        'stage': mv.current_stage,
                        'run_id': mv.run_id,
                        'status': mv.status,
                    } for mv in latest_versions]
                }
                models_info.append(model_info)

            return models_info
        except Exception as e:
            logger.error(f"Failed to get registered models: {e}")
            return []

    def get_model_version_details(
        self,
        model_name: str,
        version: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific model version.

        Args:
            model_name: Name of the registered model
            version: Version number

        Returns:
            Model version details including associated run info
        """
        try:
            mv = self.client.get_model_version(model_name, version)

            # Get associated run details
            run_info = self.get_run_by_id(mv.run_id) if mv.run_id else None

            return {
                'name': mv.name,
                'version': mv.version,
                'stage': mv.current_stage,
                'description': mv.description,
                'run_id': mv.run_id,
                'run_info': run_info,
                'status': mv.status,
                'source': mv.source,
                'creation_timestamp': mv.creation_timestamp,
                'last_updated_timestamp': mv.last_updated_timestamp,
            }
        except Exception as e:
            logger.error(f"Failed to get model version: {e}")
            return None
