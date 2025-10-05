/**
 * ModelMetricsDashboard - Main dashboard for viewing and comparing ML model metrics
 */

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle, BarChart3, TrendingUp, Layers, ExternalLink } from "lucide-react";
import { ModelMetricsCard, ModelMetricsCardSkeleton } from "./ModelMetricsCard";
import { ModelComparisonChart } from "./ModelComparisonChart";
import {
  useModelRuns,
  useModelsSummary,
  useRunsByType,
} from "@/hooks/useModelMetrics";
import type { RunFilters } from "@/types/models";

export function ModelMetricsDashboard() {
  const [selectedType, setSelectedType] = useState<string>('all');
  const [selectedRuns, setSelectedRuns] = useState<Set<string>>(new Set());
  const [compareMode, setCompareMode] = useState(false);

  // Fetch data
  const { data: summary, isLoading: summaryLoading, error: summaryError } = useModelsSummary();
  const {
    data: allRuns,
    isLoading: runsLoading,
    error: runsError,
  } = useModelRuns({ limit: 50 });

  // Filter runs by selected type
  const filteredRuns = selectedType === 'all'
    ? allRuns || []
    : (allRuns || []).filter(run => run.model_type === selectedType);

  // Get runs for comparison
  const runsToCompare = (allRuns || []).filter(run => selectedRuns.has(run.run_id));

  // Toggle run selection for comparison
  const toggleRunSelection = (runId: string) => {
    const newSelected = new Set(selectedRuns);
    if (newSelected.has(runId)) {
      newSelected.delete(runId);
    } else {
      if (newSelected.size >= 5) {
        alert('Maximum 5 models can be compared at once');
        return;
      }
      newSelected.add(runId);
    }
    setSelectedRuns(newSelected);
  };

  // Clear selection
  const clearSelection = () => {
    setSelectedRuns(new Set());
    setCompareMode(false);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Model Metrics Dashboard</h1>
          <p className="text-muted-foreground">
            Track and compare ML model performance
          </p>
        </div>
        <Button
          variant="outline"
          onClick={() => window.open('http://localhost:5002', '_blank')}
        >
          <ExternalLink className="mr-2 h-4 w-4" />
          Open MLflow UI
        </Button>
      </div>

      {/* Summary Cards */}
      {summaryLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i}>
              <CardHeader>
                <div className="h-4 w-20 bg-muted animate-pulse rounded" />
              </CardHeader>
              <CardContent>
                <div className="h-8 w-16 bg-muted animate-pulse rounded" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : summaryError ? (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Failed to load summary: {summaryError.message}
          </AlertDescription>
        </Alert>
      ) : summary ? (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Runs</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{summary.total_runs}</div>
              <p className="text-xs text-muted-foreground">All model training runs</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Retrieval Models</CardTitle>
              <Layers className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {summary.counts_by_type['retrieval'] || 0}
              </div>
              <p className="text-xs text-muted-foreground">Two-Tower, ALS, etc.</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Ranking Models</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {summary.counts_by_type['ranking'] || 0}
              </div>
              <p className="text-xs text-muted-foreground">BERT4Rec, rerankers, etc.</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Models</CardTitle>
              <AlertCircle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {summary.counts_by_status['active'] || 0}
              </div>
              <p className="text-xs text-muted-foreground">Currently deployed</p>
            </CardContent>
          </Card>
        </div>
      ) : null}

      {/* Main Content */}
      <Tabs defaultValue="all-models" className="w-full">
        <TabsList>
          <TabsTrigger value="all-models">All Models</TabsTrigger>
          <TabsTrigger value="compare">
            Compare Models
            {selectedRuns.size > 0 && (
              <Badge variant="secondary" className="ml-2">
                {selectedRuns.size}
              </Badge>
            )}
          </TabsTrigger>
        </TabsList>

        {/* All Models View */}
        <TabsContent value="all-models" className="space-y-4">
          {/* Filters */}
          <div className="flex items-center gap-4">
            <Select value={selectedType} onValueChange={setSelectedType}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Filter by type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="retrieval">Retrieval</SelectItem>
                <SelectItem value="ranking">Ranking</SelectItem>
                <SelectItem value="embedding">Embedding</SelectItem>
                <SelectItem value="hybrid">Hybrid</SelectItem>
              </SelectContent>
            </Select>

            {selectedRuns.size > 0 && (
              <div className="flex items-center gap-2">
                <Badge variant="outline">
                  {selectedRuns.size} selected for comparison
                </Badge>
                <Button variant="ghost" size="sm" onClick={clearSelection}>
                  Clear
                </Button>
              </div>
            )}
          </div>

          {/* Models Grid */}
          {runsLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <ModelMetricsCardSkeleton key={i} />
              ))}
            </div>
          ) : runsError ? (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Failed to load runs: {runsError.message}
              </AlertDescription>
            </Alert>
          ) : filteredRuns.length === 0 ? (
            <Card>
              <CardContent className="flex items-center justify-center h-64">
                <div className="text-center space-y-2">
                  <p className="text-muted-foreground">No models found</p>
                  <p className="text-sm text-muted-foreground">
                    Run <code className="bg-muted px-2 py-1 rounded">dvc repro</code> to train models
                  </p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredRuns.map((run) => (
                <ModelMetricsCard
                  key={run.run_id}
                  run={run}
                  onCompare={toggleRunSelection}
                  isComparing={selectedRuns.has(run.run_id)}
                  showActions={true}
                />
              ))}
            </div>
          )}
        </TabsContent>

        {/* Comparison View */}
        <TabsContent value="compare" className="space-y-4">
          {runsToCompare.length === 0 ? (
            <Card>
              <CardContent className="flex items-center justify-center h-64">
                <div className="text-center space-y-2">
                  <p className="text-muted-foreground">No models selected for comparison</p>
                  <p className="text-sm text-muted-foreground">
                    Go to "All Models" tab and click "Compare" on models you want to compare
                  </p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <>
              {/* Selected Models Summary */}
              <div className="flex flex-wrap gap-2">
                {runsToCompare.map((run) => (
                  <Badge key={run.run_id} variant="secondary" className="flex items-center gap-2">
                    {run.run_name.substring(0, 30)}
                    <button
                      onClick={() => toggleRunSelection(run.run_id)}
                      className="ml-1 hover:text-destructive"
                    >
                      ×
                    </button>
                  </Badge>
                ))}
              </div>

              {/* Comparison Charts */}
              <ModelComparisonChart runs={runsToCompare} />

              {/* Individual Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {runsToCompare.map((run) => (
                  <ModelMetricsCard
                    key={run.run_id}
                    run={run}
                    showActions={false}
                  />
                ))}
              </div>
            </>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
