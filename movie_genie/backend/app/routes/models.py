"""
Model Metrics API Routes

This module provides REST API endpoints for accessing ML model metrics
and experiment data from MLflow. These endpoints power the frontend
model metrics dashboard.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging

from movie_genie.backend.app.services.mlflow_client import MLflowService

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/models", tags=["models"])

# Initialize MLflow service
try:
    mlflow_service = MLflowService()
    logger.info("MLflow service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MLflow service: {e}")
    mlflow_service = None


@router.get("/experiments")
async def list_experiments():
    """
    Get all MLflow experiments.

    Returns:
        List of experiments with metadata

    Example:
        GET /api/models/experiments
    """
    if not mlflow_service:
        raise HTTPException(status_code=503, detail="MLflow service not available")

    try:
        experiments = mlflow_service.get_all_experiments()
        return {
            "success": True,
            "data": experiments,
            "count": len(experiments)
        }
    except Exception as e:
        logger.error(f"Failed to get experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs")
async def list_runs(
    experiment_name: Optional[str] = None,
    model_type: Optional[str] = Query(None, description="Filter by model type (retrieval, ranking, etc.)"),
    model_name: Optional[str] = Query(None, description="Filter by model name (two-tower, bert4rec, etc.)"),
    status: Optional[str] = Query(None, description="Filter by status (active, inactive, etc.)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of runs to return")
):
    """
    Get all model training runs with optional filtering.

    Query Parameters:
        - experiment_name: Name of experiment (defaults to movie-genie-models)
        - model_type: Filter by model type (retrieval, ranking, embedding, hybrid)
        - model_name: Filter by model name (two-tower, bert4rec)
        - status: Filter by deployment status (active, inactive, experimental)
        - limit: Maximum number of runs to return (default: 100)

    Returns:
        List of runs with metrics, parameters, and tags

    Example:
        GET /api/models/runs?model_type=retrieval&limit=10
    """
    if not mlflow_service:
        raise HTTPException(status_code=503, detail="MLflow service not available")

    try:
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

        return {
            "success": True,
            "data": runs,
            "count": len(runs),
            "filters": {
                "model_type": model_type,
                "model_name": model_name,
                "status": status,
            }
        }
    except Exception as e:
        logger.error(f"Failed to get runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """
    Get details for a specific model run.

    Path Parameters:
        - run_id: MLflow run ID

    Returns:
        Run details with all metrics, parameters, and tags

    Example:
        GET /api/models/runs/abc123def456
    """
    if not mlflow_service:
        raise HTTPException(status_code=503, detail="MLflow service not available")

    try:
        run = mlflow_service.get_run_by_id(run_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        return {
            "success": True,
            "data": run
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}/metrics")
async def get_run_metrics(run_id: str):
    """
    Get all metrics for a specific run.

    Path Parameters:
        - run_id: MLflow run ID

    Returns:
        Dictionary of metric names to values

    Example:
        GET /api/models/runs/abc123/metrics
    """
    if not mlflow_service:
        raise HTTPException(status_code=503, detail="MLflow service not available")

    try:
        metrics = mlflow_service.get_model_metrics(run_id)

        return {
            "success": True,
            "data": metrics
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}/metrics/{metric_name}/history")
async def get_metric_history(run_id: str, metric_name: str):
    """
    Get the history of a specific metric over training steps.

    Path Parameters:
        - run_id: MLflow run ID
        - metric_name: Name of the metric (e.g., 'train_loss', 'val_loss')

    Returns:
        List of metric values over time/steps

    Example:
        GET /api/models/runs/abc123/metrics/train_loss/history
    """
    if not mlflow_service:
        raise HTTPException(status_code=503, detail="MLflow service not available")

    try:
        history = mlflow_service.get_metric_history(run_id, metric_name)

        return {
            "success": True,
            "data": history,
            "metric_name": metric_name
        }
    except Exception as e:
        logger.error(f"Failed to get metric history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_runs(run_ids: List[str]):
    """
    Compare multiple model runs side-by-side.

    Request Body:
        List of run IDs to compare

    Returns:
        List of run details for comparison

    Example:
        POST /api/models/compare
        Body: ["run_id_1", "run_id_2", "run_id_3"]
    """
    if not mlflow_service:
        raise HTTPException(status_code=503, detail="MLflow service not available")

    try:
        if not run_ids:
            raise HTTPException(status_code=400, detail="No run IDs provided")

        if len(run_ids) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 runs can be compared")

        runs = mlflow_service.compare_runs(run_ids)

        return {
            "success": True,
            "data": runs,
            "count": len(runs)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-type/{model_type}")
async def get_models_by_type(model_type: str):
    """
    Get all models of a specific type.

    Path Parameters:
        - model_type: Type of model (retrieval, ranking, embedding, hybrid)

    Returns:
        List of runs for that model type

    Example:
        GET /api/models/by-type/retrieval
    """
    if not mlflow_service:
        raise HTTPException(status_code=503, detail="MLflow service not available")

    try:
        valid_types = ['retrieval', 'ranking', 'embedding', 'hybrid']
        if model_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model type. Must be one of: {valid_types}"
            )

        runs = mlflow_service.get_models_by_type(model_type)

        return {
            "success": True,
            "data": runs,
            "model_type": model_type,
            "count": len(runs)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get models by type: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/production")
async def get_production_models():
    """
    Get all models currently tagged as production/active.

    Returns:
        List of production model runs

    Example:
        GET /api/models/production
    """
    if not mlflow_service:
        raise HTTPException(status_code=503, detail="MLflow service not available")

    try:
        runs = mlflow_service.get_production_models()

        return {
            "success": True,
            "data": runs,
            "count": len(runs)
        }
    except Exception as e:
        logger.error(f"Failed to get production models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest/{model_name}")
async def get_latest_model(model_name: str):
    """
    Get the most recent run for a specific model name.

    Path Parameters:
        - model_name: Name of the model (two-tower, bert4rec, etc.)

    Returns:
        Latest run for that model

    Example:
        GET /api/models/latest/two-tower
    """
    if not mlflow_service:
        raise HTTPException(status_code=503, detail="MLflow service not available")

    try:
        run = mlflow_service.get_latest_run_by_model_name(model_name)

        if not run:
            raise HTTPException(
                status_code=404,
                detail=f"No runs found for model '{model_name}'"
            )

        return {
            "success": True,
            "data": run
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latest model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry")
async def list_registered_models():
    """
    Get all registered models from MLflow Model Registry.

    Returns:
        List of registered models with version information

    Example:
        GET /api/models/registry
    """
    if not mlflow_service:
        raise HTTPException(status_code=503, detail="MLflow service not available")

    try:
        models = mlflow_service.get_registered_models()

        return {
            "success": True,
            "data": models,
            "count": len(models)
        }
    except Exception as e:
        logger.error(f"Failed to get registered models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry/{model_name}/versions/{version}")
async def get_model_version(model_name: str, version: str):
    """
    Get details for a specific model version in the registry.

    Path Parameters:
        - model_name: Name of the registered model
        - version: Version number

    Returns:
        Model version details including associated run info

    Example:
        GET /api/models/registry/two-tower-retrieval/versions/1
    """
    if not mlflow_service:
        raise HTTPException(status_code=503, detail="MLflow service not available")

    try:
        model_version = mlflow_service.get_model_version_details(model_name, version)

        if not model_version:
            raise HTTPException(
                status_code=404,
                detail=f"Model version {model_name}/{version} not found"
            )

        return {
            "success": True,
            "data": model_version
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_models_summary():
    """
    Get a summary of all models including counts by type and status.

    Returns:
        Summary statistics of all models

    Example:
        GET /api/models/summary
    """
    if not mlflow_service:
        raise HTTPException(status_code=503, detail="MLflow service not available")

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

        return {
            "success": True,
            "data": {
                "total_runs": len(all_runs),
                "counts_by_type": type_counts,
                "counts_by_status": status_counts,
                "latest_models": list(latest_models.values()),
            }
        }
    except Exception as e:
        logger.error(f"Failed to get models summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
