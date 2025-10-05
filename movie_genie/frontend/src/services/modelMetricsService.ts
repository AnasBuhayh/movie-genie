/**
 * Service for fetching ML model metrics from backend API
 */

import type {
  ModelRun,
  MetricHistory,
  RunFilters,
  ModelsSummary,
  APIResponse,
  CompareRunsRequest,
} from '@/types/models';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001/api';

class ModelMetricsService {
  private baseUrl = `${API_BASE_URL}/models`;

  /**
   * Fetch all model runs with optional filtering
   */
  async getRuns(filters?: RunFilters): Promise<ModelRun[]> {
    const params = new URLSearchParams();

    if (filters?.experiment_name) params.append('experiment_name', filters.experiment_name);
    if (filters?.model_type) params.append('model_type', filters.model_type);
    if (filters?.model_name) params.append('model_name', filters.model_name);
    if (filters?.status) params.append('status', filters.status);
    if (filters?.limit) params.append('limit', filters.limit.toString());

    const url = `${this.baseUrl}/runs${params.toString() ? `?${params.toString()}` : ''}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Failed to fetch runs: ${response.statusText}`);
    }

    const result: APIResponse<ModelRun[]> = await response.json();

    if (!result.success) {
      throw new Error(result.error || 'Failed to fetch runs');
    }

    return result.data;
  }

  /**
   * Fetch details for a specific run
   */
  async getRunDetails(runId: string): Promise<ModelRun> {
    const response = await fetch(`${this.baseUrl}/runs/${runId}`);

    if (!response.ok) {
      throw new Error(`Failed to fetch run details: ${response.statusText}`);
    }

    const result: APIResponse<ModelRun> = await response.json();

    if (!result.success) {
      throw new Error(result.error || 'Failed to fetch run details');
    }

    return result.data;
  }

  /**
   * Fetch metric history for charting (e.g., loss curves)
   */
  async getMetricHistory(runId: string, metricName: string): Promise<MetricHistory[]> {
    const response = await fetch(`${this.baseUrl}/runs/${runId}/metrics/${metricName}/history`);

    if (!response.ok) {
      throw new Error(`Failed to fetch metric history: ${response.statusText}`);
    }

    const result: APIResponse<MetricHistory[]> = await response.json();

    if (!result.success) {
      throw new Error(result.error || 'Failed to fetch metric history');
    }

    return result.data;
  }

  /**
   * Compare multiple model runs
   */
  async compareRuns(runIds: string[]): Promise<ModelRun[]> {
    const response = await fetch(`${this.baseUrl}/compare`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ run_ids: runIds } as CompareRunsRequest),
    });

    if (!response.ok) {
      throw new Error(`Failed to compare runs: ${response.statusText}`);
    }

    const result: APIResponse<ModelRun[]> = await response.json();

    if (!result.success) {
      throw new Error(result.error || 'Failed to compare runs');
    }

    return result.data;
  }

  /**
   * Fetch summary statistics for all models
   */
  async getSummary(): Promise<ModelsSummary> {
    const response = await fetch(`${this.baseUrl}/summary`);

    if (!response.ok) {
      throw new Error(`Failed to fetch summary: ${response.statusText}`);
    }

    const result: APIResponse<ModelsSummary> = await response.json();

    if (!result.success) {
      throw new Error(result.error || 'Failed to fetch summary');
    }

    return result.data;
  }

  /**
   * Get latest run for a specific model
   */
  async getLatestRun(modelName: string): Promise<ModelRun | null> {
    const runs = await this.getRuns({ model_name: modelName, limit: 1 });
    return runs.length > 0 ? runs[0] : null;
  }

  /**
   * Get runs by model type
   */
  async getRunsByType(modelType: 'retrieval' | 'ranking' | 'embedding' | 'hybrid'): Promise<ModelRun[]> {
    return this.getRuns({ model_type: modelType });
  }

  /**
   * Get active/production models
   */
  async getActiveModels(): Promise<ModelRun[]> {
    return this.getRuns({ status: 'active' });
  }
}

// Export singleton instance
export const modelMetricsService = new ModelMetricsService();
export default modelMetricsService;
