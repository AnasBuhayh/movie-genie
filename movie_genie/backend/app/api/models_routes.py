"""
Model Metrics API Routes (Flask)

This module provides REST API endpoints for accessing ML model metrics
and experiment data from MLflow. These endpoints power the frontend
model metrics dashboard.
"""

from flask import Blueprint, jsonify, request
from typing import List, Optional
import logging

from movie_genie.backend.app.services.mlflow_client import MLflowService

logger = logging.getLogger(__name__)

# Create Flask blueprint
models_bp = Blueprint('models', __name__)

# Initialize MLflow service
try:
    mlflow_service = MLflowService()
    logger.info("MLflow service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MLflow service: {e}")
    mlflow_service = None


@models_bp.route('/experiments', methods=['GET'])
def list_experiments():
    """
    Get all MLflow experiments.

    Returns:
        JSON response with list of experiments

    Example:
        GET /api/models/experiments
    """
    if not mlflow_service:
        return jsonify({"success": False, "error": "MLflow service not available"}), 503

    try:
        experiments = mlflow_service.get_all_experiments()
        return jsonify({
            "success": True,
            "data": experiments,
            "count": len(experiments)
        })
    except Exception as e:
        logger.error(f"Failed to get experiments: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@models_bp.route('/runs', methods=['GET'])
def list_runs():
    """
    Get all model training runs with optional filtering.

    Query Parameters:
        - experiment_name: Name of experiment
        - model_type: Filter by model type (retrieval, ranking, etc.)
        - model_name: Filter by model name (two-tower, bert4rec, etc.)
        - status: Filter by status (active, inactive, etc.)
        - limit: Maximum number of runs to return (default: 100)

    Example:
        GET /api/models/runs?model_type=retrieval&limit=10
    """
    if not mlflow_service:
        return jsonify({"success": False, "error": "MLflow service not available"}), 503

    try:
        # Get query parameters
        experiment_name = request.args.get('experiment_name')
        model_type = request.args.get('model_type')
        model_name = request.args.get('model_name')
        status = request.args.get('status')
        limit = int(request.args.get('limit', 100))

        # Build filter string
        filters = []
        if model_type:
            filters.append(f"tags.model_type='{model_type}'")
        if model_name:
            filters.append(f"tags.model_name='{model_name}'")
        if status:
            filters.append(f"tags.status='{status}'")

        filter_string = " and ".join(filters) if filters else ""

        runs = mlflow_service.get_experiment_runs(
            experiment_name=experiment_name,
            filter_string=filter_string,
            max_results=limit
        )

        return jsonify({
            "success": True,
            "data": runs,
            "count": len(runs),
            "filters": {
                "model_type": model_type,
                "model_name": model_name,
                "status": status,
            }
        })
    except Exception as e:
        logger.error(f"Failed to get runs: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@models_bp.route('/runs/<run_id>', methods=['GET'])
def get_run(run_id: str):
    """
    Get details for a specific model run.

    Path Parameters:
        - run_id: MLflow run ID

    Example:
        GET /api/models/runs/abc123def456
    """
    if not mlflow_service:
        return jsonify({"success": False, "error": "MLflow service not available"}), 503

    try:
        run = mlflow_service.get_run_by_id(run_id)

        if not run:
            return jsonify({"success": False, "error": f"Run {run_id} not found"}), 404

        return jsonify({
            "success": True,
            "data": run
        })
    except Exception as e:
        logger.error(f"Failed to get run {run_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@models_bp.route('/runs/<run_id>/metrics/<metric_name>/history', methods=['GET'])
def get_metric_history(run_id: str, metric_name: str):
    """
    Get the history of a specific metric over training steps.

    Path Parameters:
        - run_id: MLflow run ID
        - metric_name: Name of the metric (e.g., 'train_loss', 'val_loss')

    Example:
        GET /api/models/runs/abc123/metrics/train_loss/history
    """
    if not mlflow_service:
        return jsonify({"success": False, "error": "MLflow service not available"}), 503

    try:
        history = mlflow_service.get_metric_history(run_id, metric_name)

        return jsonify({
            "success": True,
            "data": history,
            "metric_name": metric_name
        })
    except Exception as e:
        logger.error(f"Failed to get metric history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@models_bp.route('/compare', methods=['POST'])
def compare_runs():
    """
    Compare multiple model runs side-by-side.

    Request Body:
        {"run_ids": ["run_id_1", "run_id_2", "run_id_3"]}

    Example:
        POST /api/models/compare
        Body: {"run_ids": ["abc123", "def456"]}
    """
    if not mlflow_service:
        return jsonify({"success": False, "error": "MLflow service not available"}), 503

    try:
        data = request.get_json()
        run_ids = data.get('run_ids', [])

        if not run_ids:
            return jsonify({"success": False, "error": "No run IDs provided"}), 400

        if len(run_ids) > 10:
            return jsonify({"success": False, "error": "Maximum 10 runs can be compared"}), 400

        runs = mlflow_service.compare_runs(run_ids)

        return jsonify({
            "success": True,
            "data": runs,
            "count": len(runs)
        })
    except Exception as e:
        logger.error(f"Failed to compare runs: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@models_bp.route('/summary', methods=['GET'])
def get_models_summary():
    """
    Get a summary of all models including counts by type and status.

    Example:
        GET /api/models/summary
    """
    if not mlflow_service:
        return jsonify({"success": False, "error": "MLflow service not available"}), 503

    try:
        # Get all runs
        all_runs = mlflow_service.get_experiment_runs(max_results=1000)

        # Count by type
        type_counts = {}
        status_counts = {}

        for run in all_runs:
            model_type = run.get('model_type', 'unknown')
            model_status = run.get('model_status', 'unknown')

            type_counts[model_type] = type_counts.get(model_type, 0) + 1
            status_counts[model_status] = status_counts.get(model_status, 0) + 1

        # Get latest runs by model name
        latest_models = {}
        for run in all_runs:
            model_name = run.get('model_name', 'unknown')
            if model_name not in latest_models:
                latest_models[model_name] = run

        return jsonify({
            "success": True,
            "data": {
                "total_runs": len(all_runs),
                "counts_by_type": type_counts,
                "counts_by_status": status_counts,
                "latest_models": list(latest_models.values()),
            }
        })
    except Exception as e:
        logger.error(f"Failed to get models summary: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
