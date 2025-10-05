# Frontend Model Metrics Dashboard

Complete React dashboard for visualizing MLflow experiment metrics.

## Features

### Dashboard Overview
- **Summary Cards**: Total runs, model counts by type, active models
- **Filtering**: Filter by model type (retrieval, ranking, embedding, hybrid)
- **Two Tabs**:
  - **All Models**: Grid view of all model runs
  - **Compare Models**: Side-by-side comparison

### Model Metrics Card
Shows key metrics per model:
- **Retrieval Models**: Recall@10/50/100, Coverage@100, Training Time
- **Ranking Models**: Train/Val Loss, Epochs, Total Parameters
- **Status Badges**: Active/Inactive/Experimental
- **Actions**: Compare button, link to MLflow UI

### Model Comparison
- **Metrics Tab**: Bar charts comparing evaluation metrics
- **Training Tab**: Loss curves and training time comparison
- **Parameters Tab**: Side-by-side parameter comparison table
- Select up to 5 models to compare

## Files Created

### TypeScript Types
**`frontend/src/types/models.ts`**

Defines all TypeScript interfaces for MLflow data:

```typescript
export interface ModelRun {
  run_id: string;
  run_name: string;
  model_type: 'retrieval' | 'ranking' | 'embedding' | 'hybrid' | 'unknown';
  metrics: Record<string, number>;
  params: Record<string, string>;
  tags: Record<string, string>;
  start_time?: number;
  end_time?: number;
}

export interface MetricHistory {
  step: number;
  value: number;
  timestamp: number;
}

export interface RunFilters {
  model_type?: string;
  model_name?: string;
  status?: string;
  limit?: number;
}

export interface ModelsSummary {
  total_runs: number;
  counts_by_type: Record<string, number>;
  counts_by_status: Record<string, number>;
  latest_models: ModelRun[];
}
```

### API Service
**`frontend/src/services/modelMetricsService.ts`**

Centralized API client for fetching model metrics:

```typescript
class ModelMetricsService {
  async getRuns(filters?: RunFilters): Promise<ModelRun[]>
  async getRunDetails(runId: string): Promise<ModelRun>
  async getMetricHistory(runId: string, metricName: string): Promise<MetricHistory[]>
  async compareRuns(runIds: string[]): Promise<ModelRun[]>
  async getSummary(): Promise<ModelsSummary>
}
```

### React Hooks
**`frontend/src/hooks/useModelMetrics.ts`**

React Query hooks for data fetching with caching:

```typescript
export function useModelRuns(filters?: RunFilters): UseQueryResult<ModelRun[], Error>
export function useRunDetails(runId: string): UseQueryResult<ModelRun, Error>
export function useMetricHistory(runId: string, metricName: string): UseQueryResult<MetricHistory[], Error>
export function useCompareRuns(runIds: string[]): UseQueryResult<ModelRun[], Error>
export function useModelsSummary(): UseQueryResult<ModelsSummary, Error>
```

Features:
- Automatic caching (30-second stale time)
- Error handling
- Loading states
- Refetch on window focus

### Components

**`frontend/src/components/ModelMetricsCard.tsx`**

Individual model run display:
- Key metrics display (different for retrieval vs ranking)
- Status and type badges
- Timestamp formatting
- Compare button
- Link to MLflow UI

**`frontend/src/components/ModelComparisonChart.tsx`**

Multi-model comparison charts:
- Three tabs: Metrics, Training, Parameters
- Bar charts for metric comparison (using Recharts)
- Line charts for training curves
- Parameter comparison table

**`frontend/src/components/ModelMetricsDashboard.tsx`**

Main container component:
- Summary statistics cards
- Model type filter dropdown
- Tabs for "All Models" and "Compare"
- Grid layout for model cards
- Model selection for comparison

### Pages & Routing

**`frontend/src/pages/ModelMetrics.tsx`**

Simple page wrapper for the dashboard.

**Updated `frontend/src/App.tsx`**

Added route:
```typescript
<Route path="/metrics" element={<ModelMetrics />} />
```

**Updated `frontend/src/pages/Index.tsx`**

Added navigation button:
```typescript
<Link to="/metrics">
  <Button variant="outline" size="sm">
    <BarChart3 className="mr-2 h-4 w-4" />
    Model Metrics
  </Button>
</Link>
```

## Usage

### 1. Start Backend

```bash
FLASK_PORT=5001 python scripts/start_server.py
```

### 2. Start Frontend

```bash
cd movie_genie/frontend
npm run dev
```

### 3. Access Dashboard

**Two ways:**

1. **Via Navigation Button**
   - Go to http://localhost:8080
   - Select a user
   - Click "Model Metrics" button in header

2. **Direct Link**
   - Navigate to http://localhost:8080/metrics

## UI Features

### Responsive Design
Works on mobile, tablet, and desktop devices.

### Loading States
Skeleton loaders while fetching data from API.

### Error Handling
Clear error messages if API fails or returns errors.

### Empty States
Helpful messages when no data is available:
- "No models found"
- "No runs available"
- "Select models to compare"

### Tooltips
Hover over truncated names for full details.

### External Links
Quick access to MLflow UI for detailed exploration.

### Badge Colors
Visual distinction for:
- Model types (retrieval, ranking, etc.)
- Status (active, inactive, experimental)

### Truncated Names
Long run names are shortened for readability.

## Data Flow

```
Training Script (DVC)
       ↓
    MLflow
    (mlruns/)
       ↓
Flask Backend
(/api/models/*)
       ↓
React Frontend
(useModelMetrics hooks)
       ↓
   Dashboard
   Components
```

## Component Hierarchy

```
ModelMetricsDashboard (Container)
├── Summary Cards (4 cards)
├── Tabs
│   ├── All Models Tab
│   │   ├── Filter Dropdown
│   │   └── ModelMetricsCard (Grid)
│   │       ├── Key Metrics
│   │       ├── Status Badges
│   │       └── Actions
│   └── Compare Tab
│       ├── Selected Models List
│       ├── ModelComparisonChart
│       │   ├── Metrics Tab (Bar Charts)
│       │   ├── Training Tab (Loss Curves)
│       │   └── Parameters Tab (Table)
│       └── Individual Cards
```

## Use Cases

### 1. Track Model Performance Over Time

```
1. Train model multiple times with different configs
2. Go to /metrics
3. See all runs sorted by date
4. Compare recall@10 across runs
```

### 2. Compare Different Architectures

```
1. Select "Compare" on 2-3 runs
2. Go to "Compare Models" tab
3. View side-by-side bar charts
4. Check parameter differences
```

### 3. Find Best Model

```
1. Filter by model type (e.g., "retrieval")
2. Sort by key metric (e.g., Recall@50)
3. Identify best performing run
4. Click to view in MLflow UI for full details
```

### 4. Monitor Training Progress

```
1. After running dvc repro
2. Refresh /metrics page
3. See new run appear
4. Check if metrics improved
```

## Testing Checklist

### End-to-End Test

1. **Train a Model**
   ```bash
   dvc repro two_tower_training
   ```
   - ✅ Check MLflow logged metrics
   - ✅ Check `mlruns/` directory created
   - ✅ Check metrics JSON file has `mlflow_run_id`

2. **Backend API**
   ```bash
   curl http://localhost:5001/api/models/summary | jq
   ```
   - ✅ Returns total_runs > 0
   - ✅ Shows counts by type
   - ✅ Shows latest models

3. **Frontend Dashboard**
   - ✅ Navigate to http://localhost:8080/metrics
   - ✅ See summary cards with data
   - ✅ See model cards in grid
   - ✅ Click "Compare" on 2 models
   - ✅ Go to "Compare Models" tab
   - ✅ See comparison charts

4. **Model Metrics Card**
   - ✅ Shows correct metrics
   - ✅ Status badge displays
   - ✅ "Compare" button works
   - ✅ MLflow UI link opens

5. **Comparison View**
   - ✅ Bar charts render
   - ✅ Loss comparison shows
   - ✅ Parameters table displays
   - ✅ Can remove models from comparison

## Metric Name Changes

!!! important "Metric Naming Update"
    All metrics using `@` have been changed to `_at_`:

    - `recall@10` → `recall_at_10`
    - `recall@50` → `recall_at_50`
    - `recall@100` → `recall_at_100`
    - `coverage@100` → `coverage_at_100`

    This is required because MLflow doesn't allow special characters in metric names.

## Next Steps (Optional Enhancements)

### Short Term
- [ ] Add loading states for metric history charts
- [ ] Add export to CSV/JSON functionality
- [ ] Add search/filter by run ID or date range
- [ ] Add pagination for large number of runs

### Medium Term
- [ ] Real-time metric updates during training
- [ ] Model promotion workflow (mark as active)
- [ ] Metric thresholds and alerts
- [ ] Custom metric definitions

### Long Term
- [ ] A/B testing framework
- [ ] Automated model selection
- [ ] Cost tracking per run
- [ ] Team collaboration features

## Related Documentation

- [MLflow Setup](setup.md)
- [Integration Summary](integration-summary.md)
- [How to Integrate MLflow](../how-to-guides/mlflow-integration.md)
- [API Reference](../backend-frontend/api-reference.md)

---

**Status**: Complete ✅
**Last Updated**: 2025-01-05
**Frontend**: React + TypeScript + Recharts
**Backend**: Flask + MLflow Python API
