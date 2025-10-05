/**
 * React hooks for fetching ML model metrics
 */

import { useQuery, UseQueryResult } from '@tanstack/react-query';
import modelMetricsService from '@/services/modelMetricsService';
import type {
  ModelRun,
  MetricHistory,
  RunFilters,
  ModelsSummary,
} from '@/types/models';

/**
 * Hook to fetch all model runs with optional filters
 */
export function useModelRuns(filters?: RunFilters): UseQueryResult<ModelRun[], Error> {
  return useQuery({
    queryKey: ['model-runs', filters],
    queryFn: () => modelMetricsService.getRuns(filters),
    staleTime: 30000, // 30 seconds
  });
}

/**
 * Hook to fetch details for a specific run
 */
export function useRunDetails(runId: string | undefined): UseQueryResult<ModelRun, Error> {
  return useQuery({
    queryKey: ['run-details', runId],
    queryFn: () => modelMetricsService.getRunDetails(runId!),
    enabled: !!runId,
    staleTime: 60000, // 1 minute
  });
}

/**
 * Hook to fetch metric history for charting
 */
export function useMetricHistory(
  runId: string | undefined,
  metricName: string | undefined
): UseQueryResult<MetricHistory[], Error> {
  return useQuery({
    queryKey: ['metric-history', runId, metricName],
    queryFn: () => modelMetricsService.getMetricHistory(runId!, metricName!),
    enabled: !!runId && !!metricName,
    staleTime: 60000, // 1 minute
  });
}

/**
 * Hook to compare multiple model runs
 */
export function useCompareRuns(runIds: string[]): UseQueryResult<ModelRun[], Error> {
  return useQuery({
    queryKey: ['compare-runs', runIds],
    queryFn: () => modelMetricsService.compareRuns(runIds),
    enabled: runIds.length > 0,
    staleTime: 60000, // 1 minute
  });
}

/**
 * Hook to fetch summary statistics
 */
export function useModelsSummary(): UseQueryResult<ModelsSummary, Error> {
  return useQuery({
    queryKey: ['models-summary'],
    queryFn: () => modelMetricsService.getSummary(),
    staleTime: 30000, // 30 seconds
  });
}

/**
 * Hook to get latest run for a specific model
 */
export function useLatestRun(modelName: string | undefined): UseQueryResult<ModelRun | null, Error> {
  return useQuery({
    queryKey: ['latest-run', modelName],
    queryFn: () => modelMetricsService.getLatestRun(modelName!),
    enabled: !!modelName,
    staleTime: 30000, // 30 seconds
  });
}

/**
 * Hook to get runs by model type
 */
export function useRunsByType(
  modelType: 'retrieval' | 'ranking' | 'embedding' | 'hybrid' | undefined
): UseQueryResult<ModelRun[], Error> {
  return useQuery({
    queryKey: ['runs-by-type', modelType],
    queryFn: () => modelMetricsService.getRunsByType(modelType!),
    enabled: !!modelType,
    staleTime: 30000, // 30 seconds
  });
}

/**
 * Hook to get active/production models
 */
export function useActiveModels(): UseQueryResult<ModelRun[], Error> {
  return useQuery({
    queryKey: ['active-models'],
    queryFn: () => modelMetricsService.getActiveModels(),
    staleTime: 30000, // 30 seconds
  });
}
