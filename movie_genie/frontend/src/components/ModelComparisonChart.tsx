/**
 * ModelComparisonChart - Compare metrics across multiple models
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { ModelRun, MetricHistory } from "@/types/models";
import { useMetricHistory } from "@/hooks/useModelMetrics";

interface ModelComparisonChartProps {
  runs: ModelRun[];
  metricNames?: string[];
}

export function ModelComparisonChart({
  runs,
  metricNames = ['recall_at_10', 'recall_at_50', 'recall_at_100', 'coverage_at_100'],
}: ModelComparisonChartProps) {
  if (runs.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Model Comparison</CardTitle>
          <CardDescription>Select models to compare</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64 text-muted-foreground">
            No models selected for comparison
          </div>
        </CardContent>
      </Card>
    );
  }

  // Prepare data for bar charts
  const prepareComparisonData = (metricName: string) => {
    return runs.map((run) => ({
      name: run.run_name.substring(0, 20), // Truncate long names
      value: run.metrics[metricName] || 0,
      fullName: run.run_name,
    }));
  };

  // Color palette for different runs
  const colors = [
    '#3b82f6', // blue
    '#8b5cf6', // purple
    '#ec4899', // pink
    '#10b981', // green
    '#f59e0b', // amber
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Model Comparison</CardTitle>
        <CardDescription>
          Comparing {runs.length} model{runs.length > 1 ? 's' : ''}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="metrics" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            <TabsTrigger value="training">Training</TabsTrigger>
            <TabsTrigger value="parameters">Parameters</TabsTrigger>
          </TabsList>

          {/* Metrics Comparison */}
          <TabsContent value="metrics" className="space-y-4">
            {metricNames.map((metricName) => {
              const data = prepareComparisonData(metricName);
              const hasData = data.some((d) => d.value > 0);

              if (!hasData) return null;

              return (
                <div key={metricName} className="space-y-2">
                  <h4 className="text-sm font-medium capitalize">
                    {metricName.replace(/@/g, ' @ ').replace(/_/g, ' ')}
                  </h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="name"
                        fontSize={12}
                        tickFormatter={(value) => value}
                      />
                      <YAxis fontSize={12} />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            return (
                              <div className="bg-background border rounded-lg p-2 shadow-lg">
                                <p className="text-sm font-medium">{data.fullName}</p>
                                <p className="text-sm text-muted-foreground">
                                  {metricName}: {(data.value * 100).toFixed(2)}%
                                </p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Bar dataKey="value" fill={colors[0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              );
            })}
          </TabsContent>

          {/* Training Metrics */}
          <TabsContent value="training" className="space-y-4">
            <TrainingComparisonChart runs={runs} />
          </TabsContent>

          {/* Parameters Comparison */}
          <TabsContent value="parameters" className="space-y-4">
            <ParametersTable runs={runs} />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

/**
 * Training metrics comparison (loss curves)
 */
function TrainingComparisonChart({ runs }: { runs: ModelRun[] }) {
  // For now, show final losses in a bar chart
  // In a full implementation, we'd fetch and display full training curves
  const lossData = runs.map((run) => ({
    name: run.run_name.substring(0, 20),
    trainLoss: run.metrics['final_train_loss'],
    valLoss: run.metrics['final_val_loss'],
    fullName: run.run_name,
  }));

  const colors = ['#3b82f6', '#8b5cf6'];

  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium">Training & Validation Loss</h4>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={lossData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" fontSize={12} />
          <YAxis fontSize={12} />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const data = payload[0].payload;
                return (
                  <div className="bg-background border rounded-lg p-2 shadow-lg">
                    <p className="text-sm font-medium">{data.fullName}</p>
                    <p className="text-sm text-blue-500">
                      Train Loss: {data.trainLoss?.toFixed(4) || 'N/A'}
                    </p>
                    <p className="text-sm text-purple-500">
                      Val Loss: {data.valLoss?.toFixed(4) || 'N/A'}
                    </p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Legend />
          <Bar dataKey="trainLoss" fill={colors[0]} name="Train Loss" />
          <Bar dataKey="valLoss" fill={colors[1]} name="Val Loss" />
        </BarChart>
      </ResponsiveContainer>

      {/* Training Time Comparison */}
      <div className="mt-4">
        <h4 className="text-sm font-medium mb-2">Training Time</h4>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart
            data={runs.map((run) => ({
              name: run.run_name.substring(0, 20),
              time: (run.metrics['training_time_seconds'] || 0) / 60, // Convert to minutes
              fullName: run.run_name,
            }))}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" fontSize={12} />
            <YAxis fontSize={12} label={{ value: 'Minutes', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-background border rounded-lg p-2 shadow-lg">
                      <p className="text-sm font-medium">{data.fullName}</p>
                      <p className="text-sm text-muted-foreground">
                        {data.time.toFixed(1)} minutes
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="time" fill="#10b981" name="Training Time (min)" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/**
 * Parameters comparison table
 */
function ParametersTable({ runs }: { runs: ModelRun[] }) {
  // Get all unique parameters
  const allParams = new Set<string>();
  runs.forEach((run) => {
    Object.keys(run.params).forEach((param) => allParams.add(param));
  });

  const paramArray = Array.from(allParams).sort();

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b">
            <th className="text-left p-2 font-medium">Parameter</th>
            {runs.map((run) => (
              <th key={run.run_id} className="text-left p-2 font-medium">
                {run.run_name.substring(0, 20)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {paramArray.map((param) => (
            <tr key={param} className="border-b">
              <td className="p-2 text-muted-foreground">{param}</td>
              {runs.map((run) => (
                <td key={run.run_id} className="p-2 font-mono text-xs">
                  {run.params[param] || '-'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      {/* Model Architecture Info */}
      <div className="mt-4 space-y-2">
        <h4 className="text-sm font-medium">Model Architecture</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              <th className="text-left p-2 font-medium">Metric</th>
              {runs.map((run) => (
                <th key={run.run_id} className="text-left p-2 font-medium">
                  {run.run_name.substring(0, 20)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="border-b">
              <td className="p-2 text-muted-foreground">Total Parameters</td>
              {runs.map((run) => (
                <td key={run.run_id} className="p-2">
                  {(run.metrics['total_parameters'] || 0).toLocaleString()}
                </td>
              ))}
            </tr>
            <tr className="border-b">
              <td className="p-2 text-muted-foreground">Framework</td>
              {runs.map((run) => (
                <td key={run.run_id} className="p-2">
                  {run.framework}
                </td>
              ))}
            </tr>
            <tr className="border-b">
              <td className="p-2 text-muted-foreground">Model Type</td>
              {runs.map((run) => (
                <td key={run.run_id} className="p-2">
                  {run.model_type}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
