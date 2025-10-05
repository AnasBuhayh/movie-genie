/**
 * TypeScript types for ML model metrics and MLflow data
 */

export interface ModelRun {
  run_id: string;
  run_name: string;
  experiment_id: string;
  status: string;
  start_time: number;
  end_time: number;
  artifact_uri: string;

  // Model metadata
  model_type: 'retrieval' | 'ranking' | 'embedding' | 'hybrid' | 'unknown';
  model_name: string;
  framework: string;
  model_status: 'active' | 'inactive' | 'experimental' | 'unknown';

  // Metrics and parameters
  metrics: Record<string, number>;
  params: Record<string, string>;
  tags: Record<string, string>;
}

export interface MetricHistory {
  step: number;
  value: number;
  timestamp: number;
}

export interface RunFilters {
  experiment_name?: string;
  model_type?: 'retrieval' | 'ranking' | 'embedding' | 'hybrid';
  model_name?: string;
  status?: 'active' | 'inactive' | 'experimental';
  limit?: number;
}

export interface ModelsSummary {
  total_runs: number;
  counts_by_type: Record<string, number>;
  counts_by_status: Record<string, number>;
  latest_models: ModelRun[];
}

export interface APIResponse<T> {
  success: boolean;
  data: T;
  count?: number;
  error?: string;
  filters?: RunFilters;
  metric_name?: string;
}

export interface CompareRunsRequest {
  run_ids: string[];
}

// UI-specific types
export interface ModelMetricCard {
  label: string;
  value: string | number;
  change?: number; // percentage change from previous
  trend?: 'up' | 'down' | 'stable';
  format?: 'number' | 'percentage' | 'time' | 'count';
}

export interface ChartDataPoint {
  step: number;
  value: number;
  label?: string;
}
