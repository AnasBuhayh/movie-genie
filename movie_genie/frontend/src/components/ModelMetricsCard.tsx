/**
 * ModelMetricsCard - Display metrics for a single ML model
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ExternalLink, TrendingUp, TrendingDown, Minus } from "lucide-react";
import type { ModelRun } from "@/types/models";

interface ModelMetricsCardProps {
  run: ModelRun;
  onCompare?: (runId: string) => void;
  isComparing?: boolean;
  showActions?: boolean;
}

export function ModelMetricsCard({
  run,
  onCompare,
  isComparing = false,
  showActions = true,
}: ModelMetricsCardProps) {
  // Format timestamp
  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  // Format metric value
  const formatMetric = (value: number, decimalPlaces: number = 3): string => {
    if (value === undefined || value === null) return 'N/A';
    return value.toFixed(decimalPlaces);
  };

  // Get status badge color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-500';
      case 'inactive':
        return 'bg-gray-500';
      case 'experimental':
        return 'bg-yellow-500';
      default:
        return 'bg-gray-400';
    }
  };

  // Get model type badge color
  const getModelTypeColor = (type: string) => {
    switch (type) {
      case 'retrieval':
        return 'bg-blue-500';
      case 'ranking':
        return 'bg-purple-500';
      case 'embedding':
        return 'bg-indigo-500';
      case 'hybrid':
        return 'bg-pink-500';
      default:
        return 'bg-gray-500';
    }
  };

  // Extract key metrics based on model type
  const getKeyMetrics = () => {
    const metrics = run.metrics;
    const modelType = run.model_type;

    if (modelType === 'retrieval' || modelType === 'ranking') {
      return [
        { label: 'Recall@10', value: metrics['recall_at_10'], format: 'percentage' },
        { label: 'Recall@50', value: metrics['recall_at_50'], format: 'percentage' },
        { label: 'Coverage@100', value: metrics['coverage_at_100'], format: 'percentage' },
        { label: 'Training Time', value: metrics['training_time_seconds'], format: 'time' },
      ];
    }

    // Default metrics for all models
    return [
      { label: 'Final Train Loss', value: metrics['final_train_loss'], format: 'number' },
      { label: 'Final Val Loss', value: metrics['final_val_loss'], format: 'number' },
      { label: 'Epochs Trained', value: metrics['num_epochs_trained'], format: 'count' },
      { label: 'Total Parameters', value: metrics['total_parameters'], format: 'count' },
    ];
  };

  const formatMetricValue = (value: number | undefined, format: string) => {
    if (value === undefined || value === null) return 'N/A';

    switch (format) {
      case 'percentage':
        return `${(value * 100).toFixed(2)}%`;
      case 'time':
        return `${(value / 60).toFixed(1)}m`;
      case 'count':
        return value.toLocaleString();
      case 'number':
      default:
        return formatMetric(value);
    }
  };

  const keyMetrics = getKeyMetrics();

  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="space-y-1 flex-1">
            <CardTitle className="text-lg flex items-center gap-2">
              {run.run_name}
              <Badge className={getStatusColor(run.model_status)} variant="secondary">
                {run.model_status}
              </Badge>
            </CardTitle>
            <CardDescription className="flex items-center gap-2 flex-wrap">
              <Badge className={getModelTypeColor(run.model_type)} variant="outline">
                {run.model_type}
              </Badge>
              <span className="text-xs text-muted-foreground">
                {formatDate(run.start_time)}
              </span>
            </CardDescription>
          </div>
          {showActions && (
            <div className="flex gap-2">
              {onCompare && (
                <Button
                  size="sm"
                  variant={isComparing ? "default" : "outline"}
                  onClick={() => onCompare(run.run_id)}
                >
                  {isComparing ? 'Selected' : 'Compare'}
                </Button>
              )}
              <Button
                size="sm"
                variant="ghost"
                onClick={() => window.open(`http://localhost:5000/#/experiments/${run.experiment_id}/runs/${run.run_id}`, '_blank')}
              >
                <ExternalLink className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          {keyMetrics.map((metric, index) => (
            <div key={index} className="space-y-1">
              <p className="text-sm font-medium text-muted-foreground">
                {metric.label}
              </p>
              <p className="text-2xl font-bold">
                {formatMetricValue(metric.value, metric.format)}
              </p>
            </div>
          ))}
        </div>

        {/* Additional Info */}
        <div className="mt-4 pt-4 border-t space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Model Name:</span>
            <span className="font-medium">{run.model_name}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Framework:</span>
            <span className="font-medium">{run.framework}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Run ID:</span>
            <span className="font-mono text-xs">{run.run_id.slice(0, 8)}...</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Loading skeleton for ModelMetricsCard
 */
export function ModelMetricsCardSkeleton() {
  return (
    <Card>
      <CardHeader>
        <div className="space-y-2">
          <Skeleton className="h-6 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="space-y-2">
              <Skeleton className="h-4 w-20" />
              <Skeleton className="h-8 w-16" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
