import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export interface BenchmarkInfo {
  name: string;
  version: string;
  total_questions: number;
  categories: string[];
  description: string;
}

export interface ModelSpec {
  provider: 'openai' | 'anthropic' | 'local';
  name: string;
  api_key?: string;
}

export interface EvaluationConfig {
  temperature: number;
  max_tokens: number;
  num_questions?: number;
  parallel: boolean;
  seed?: number;
}

export interface EvaluationRequest {
  model: ModelSpec;
  benchmarks: string[];
  config: EvaluationConfig;
}

export interface JobResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface JobStatus {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  created_at: string;
  completed_at?: string;
  error?: string;
}

export interface LeaderboardEntry {
  rank: number;
  model_name: string;
  model_provider: string;
  benchmark: string;
  average_score: number;
  pass_rate: number;
  total_questions: number;
  timestamp: string;
}

export interface JobListItem {
  job_id: string;
  status: string;
  created_at: string;
  model: string;
  provider: string;
  benchmarks: string[];
}

export interface ApiStats {
  total_jobs: number;
  completed_jobs: number;
  running_jobs: number;
  failed_jobs: number;
  available_benchmarks: number;
  uptime: string;
}

export const apiSlice = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api/v1',
    prepareHeaders: (headers) => {
      headers.set('Content-Type', 'application/json');
      return headers;
    },
  }),
  tagTypes: ['Benchmark', 'Job', 'Leaderboard', 'Stats'],
  endpoints: (builder) => ({
    // Health check
    getHealth: builder.query<{ status: string }, void>({
      query: () => '/health',
    }),

    // Benchmarks
    getBenchmarks: builder.query<BenchmarkInfo[], void>({
      query: () => '/benchmarks',
      providesTags: ['Benchmark'],
    }),

    getBenchmarkDetails: builder.query<any, string>({
      query: (name) => `/benchmarks/${name}`,
      providesTags: (result, error, name) => [{ type: 'Benchmark', id: name }],
    }),

    // Evaluations
    startEvaluation: builder.mutation<JobResponse, EvaluationRequest>({
      query: (request) => ({
        url: '/evaluate',
        method: 'POST',
        body: request,
      }),
      invalidatesTags: ['Job', 'Stats'],
    }),

    // Jobs
    getJobStatus: builder.query<JobStatus, string>({
      query: (jobId) => `/jobs/${jobId}`,
      providesTags: (result, error, jobId) => [{ type: 'Job', id: jobId }],
    }),

    getJobResults: builder.query<any, string>({
      query: (jobId) => `/jobs/${jobId}/results`,
    }),

    getJobs: builder.query<{ jobs: JobListItem[] }, void>({
      query: () => '/jobs',
      providesTags: ['Job'],
    }),

    cancelJob: builder.mutation<{ message: string }, string>({
      query: (jobId) => ({
        url: `/jobs/${jobId}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['Job', 'Stats'],
    }),

    // Leaderboard
    getLeaderboard: builder.query<LeaderboardEntry[], { benchmark?: string; metric?: string; limit?: number }>({
      query: ({ benchmark, metric = 'average_score', limit = 50 }) => ({
        url: '/leaderboard',
        params: { benchmark, metric, limit },
      }),
      providesTags: ['Leaderboard'],
    }),

    // Model comparison
    compareModels: builder.mutation<JobResponse, { models: ModelSpec[]; benchmarks: string[]; config: EvaluationConfig }>({
      query: (request) => ({
        url: '/compare',
        method: 'POST',
        body: request,
      }),
      invalidatesTags: ['Job', 'Stats'],
    }),

    // Stats
    getStats: builder.query<ApiStats, void>({
      query: () => '/stats',
      providesTags: ['Stats'],
    }),
  }),
});

export const {
  useGetHealthQuery,
  useGetBenchmarksQuery,
  useGetBenchmarkDetailsQuery,
  useStartEvaluationMutation,
  useGetJobStatusQuery,
  useGetJobResultsQuery,
  useGetJobsQuery,
  useCancelJobMutation,
  useGetLeaderboardQuery,
  useCompareModelsMutation,
  useGetStatsQuery,
} = apiSlice;