import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Button,
  LinearProgress,
  IconButton,
  Tooltip,
  Alert,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
} from '@mui/material';
import {
  Refresh,
  Delete,
  Stop,
  Visibility,
  GetApp,
  FilterList,
} from '@mui/icons-material';
import { format } from 'date-fns';
import { useDispatch } from 'react-redux';
import {
  useGetJobsQuery,
  useGetJobStatusQuery,
  useGetJobResultsQuery,
  useCancelJobMutation,
  JobListItem,
} from '../store/apiSlice';
import { addNotification } from '../store/uiSlice';

const JobsPage: React.FC = () => {
  const dispatch = useDispatch();
  const [selectedJob, setSelectedJob] = React.useState<string | null>(null);
  const [resultsDialog, setResultsDialog] = React.useState(false);
  const [statusFilter, setStatusFilter] = React.useState<string>('all');
  const [providerFilter, setProviderFilter] = React.useState<string>('all');
  const [searchTerm, setSearchTerm] = React.useState<string>('');
  const [autoRefresh, setAutoRefresh] = React.useState<boolean>(true);
  
  const { data: jobsData, isLoading, error, refetch } = useGetJobsQuery(
    undefined,
    { pollingInterval: autoRefresh ? 5000 : 0 }
  );
  const [cancelJob, { isLoading: cancellingJob }] = useCancelJobMutation();
  
  const { data: jobResults } = useGetJobResultsQuery(
    selectedJob || '',
    { skip: !selectedJob }
  );

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'success';
      case 'running':
        return 'primary';
      case 'pending':
        return 'warning';
      case 'failed':
        return 'error';
      case 'cancelled':
        return 'default';
      default:
        return 'default';
    }
  };

  // Filter jobs based on current filters
  const filteredJobs = React.useMemo(() => {
    let filtered = jobsData?.jobs || [];

    // Filter by status
    if (statusFilter !== 'all') {
      filtered = filtered.filter(job => job.status === statusFilter);
    }

    // Filter by provider
    if (providerFilter !== 'all') {
      filtered = filtered.filter(job => job.provider === providerFilter);
    }

    // Filter by search term
    if (searchTerm.trim()) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(job => 
        job.model.toLowerCase().includes(term) ||
        job.job_id.toLowerCase().includes(term) ||
        job.benchmarks.some(b => b.toLowerCase().includes(term))
      );
    }

    return filtered;
  }, [jobsData?.jobs, statusFilter, providerFilter, searchTerm]);

  // Get unique providers for filter
  const uniqueProviders = React.useMemo(() => {
    const providers = new Set(jobsData?.jobs.map(job => job.provider) || []);
    return Array.from(providers);
  }, [jobsData?.jobs]);

  const handleCancelJob = async (jobId: string, jobStatus: string) => {
    try {
      await cancelJob(jobId).unwrap();
      
      const message = jobStatus === 'running' ? 'Job cancelled successfully' : 'Job deleted successfully';
      dispatch(addNotification({
        message,
        type: 'success',
      }));

      refetch();
    } catch (error: any) {
      const errorMessage = error.data?.detail || error.message || 'Failed to cancel/delete job';
      dispatch(addNotification({
        message: errorMessage,
        type: 'error',
      }));
      console.error('Failed to cancel job:', error);
    }
  };

  const handleViewResults = (jobId: string) => {
    setSelectedJob(jobId);
    setResultsDialog(true);
  };

  const handleDownloadResults = (jobId: string) => {
    // In a real implementation, this would trigger a download
    console.log('Download results for job:', jobId);
  };

  if (isLoading) {
    return (
      <Box className="loading-spinner">
        <CircularProgress size={40} />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" className="error-message">
        Failed to load jobs. Please try again later.
      </Alert>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" className="page-header">
          Evaluation Jobs
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={() => refetch()}
            disabled={isLoading}
          >
            {isLoading ? 'Refreshing...' : 'Refresh'}
          </Button>
          <Button
            variant={autoRefresh ? 'contained' : 'outlined'}
            size="small"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            Auto-refresh
          </Button>
        </Box>
      </Box>

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <FilterList /> Filters & Search
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                fullWidth
                size="small"
                label="Search jobs..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Job ID, model, benchmark..."
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Status</InputLabel>
                <Select
                  value={statusFilter}
                  label="Status"
                  onChange={(e) => setStatusFilter(e.target.value)}
                >
                  <MenuItem value="all">All Statuses</MenuItem>
                  <MenuItem value="pending">Pending</MenuItem>
                  <MenuItem value="running">Running</MenuItem>
                  <MenuItem value="completed">Completed</MenuItem>
                  <MenuItem value="failed">Failed</MenuItem>
                  <MenuItem value="cancelled">Cancelled</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Provider</InputLabel>
                <Select
                  value={providerFilter}
                  label="Provider"
                  onChange={(e) => setProviderFilter(e.target.value)}
                >
                  <MenuItem value="all">All Providers</MenuItem>
                  {uniqueProviders.map(provider => (
                    <MenuItem key={provider} value={provider}>{provider}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ display: 'flex', alignItems: 'center', height: '40px' }}>
                <Typography variant="body2" color="text.secondary">
                  {filteredJobs.length} of {jobsData?.jobs.length || 0} jobs
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {filteredJobs.length > 0 ? (
        <Card>
          <CardContent>
            <TableContainer component={Paper} elevation={0}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Job ID</TableCell>
                    <TableCell>Model</TableCell>
                    <TableCell>Provider</TableCell>
                    <TableCell>Benchmarks</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredJobs.map((job: JobListItem) => (
                    <TableRow key={job.job_id} hover>
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {job.job_id.slice(0, 8)}...
                        </Typography>
                      </TableCell>
                      
                      <TableCell>
                        <Typography variant="body2" sx={{ fontWeight: 500 }}>
                          {job.model}
                        </Typography>
                      </TableCell>

                      <TableCell>
                        <Chip 
                          label={job.provider} 
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>

                      <TableCell>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {job.benchmarks.slice(0, 2).map((benchmark) => (
                            <Chip 
                              key={benchmark}
                              label={benchmark}
                              size="small"
                              variant="outlined"
                              color="secondary"
                            />
                          ))}
                          {job.benchmarks.length > 2 && (
                            <Chip 
                              label={`+${job.benchmarks.length - 2}`}
                              size="small"
                              variant="outlined"
                            />
                          )}
                        </Box>
                      </TableCell>

                      <TableCell>
                        <Chip 
                          label={job.status}
                          size="small"
                          color={getStatusColor(job.status) as any}
                          className="status-chip"
                        />
                      </TableCell>

                      <TableCell>
                        <Tooltip title={format(new Date(job.created_at), 'PPpp')}>
                          <Typography variant="body2" color="text.secondary">
                            {format(new Date(job.created_at), 'MMM dd, HH:mm')}
                          </Typography>
                        </Tooltip>
                      </TableCell>

                      <TableCell align="center">
                        <Box sx={{ display: 'flex', gap: 0.5 }}>
                          {job.status === 'completed' && (
                            <>
                              <Tooltip title="View Results">
                                <IconButton 
                                  size="small"
                                  onClick={() => handleViewResults(job.job_id)}
                                >
                                  <Visibility />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Download Results">
                                <IconButton 
                                  size="small"
                                  onClick={() => handleDownloadResults(job.job_id)}
                                >
                                  <GetApp />
                                </IconButton>
                              </Tooltip>
                            </>
                          )}
                          
                          {job.status === 'running' && (
                            <Tooltip title="Cancel Job">
                              <IconButton 
                                size="small"
                                color="error"
                                onClick={() => handleCancelJob(job.job_id, job.status)}
                                disabled={cancellingJob}
                              >
                                <Stop />
                              </IconButton>
                            </Tooltip>
                          )}

                          {['completed', 'failed', 'cancelled'].includes(job.status) && (
                            <Tooltip title="Delete Job">
                              <IconButton 
                                size="small"
                                color="error"
                                onClick={() => handleCancelJob(job.job_id, job.status)}
                                disabled={cancellingJob}
                              >
                                <Delete />
                              </IconButton>
                            </Tooltip>
                          )}
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      ) : (
        <Alert severity="info">
          {jobsData?.jobs.length === 0 
            ? "No evaluation jobs found. Start an evaluation to see jobs here."
            : "No jobs match the current filters. Try adjusting your search criteria."
          }
        </Alert>
      )}

      {/* Results Dialog */}
      <Dialog 
        open={resultsDialog} 
        onClose={() => setResultsDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Evaluation Results
          {selectedJob && (
            <Typography variant="body2" color="text.secondary">
              Job ID: {selectedJob.slice(0, 8)}...
            </Typography>
          )}
        </DialogTitle>
        <DialogContent>
          {jobResults ? (
            <Box>
              <Grid container spacing={3} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={4}>
                  <Card>
                    <CardContent className="metric-card">
                      <Typography className="metric-value">
                        {(jobResults.results.overall_score * 100).toFixed(1)}%
                      </Typography>
                      <Typography className="metric-label">
                        Overall Score
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Card>
                    <CardContent className="metric-card">
                      <Typography className="metric-value">
                        {jobResults.results.overall_pass_rate.toFixed(1)}%
                      </Typography>
                      <Typography className="metric-label">
                        Pass Rate
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Card>
                    <CardContent className="metric-card">
                      <Typography className="metric-value">
                        {jobResults.results.total_questions}
                      </Typography>
                      <Typography className="metric-label">
                        Questions
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>

              <Typography variant="h6" gutterBottom>
                Benchmark Results
              </Typography>
              {Object.entries(jobResults.results.benchmark_scores).map(([benchmark, scores]: [string, any]) => (
                <Box key={benchmark} sx={{ mb: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
                      {benchmark.toUpperCase()}
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {(scores.average_score * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={scores.average_score * 100} 
                    sx={{ mb: 1 }}
                  />
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2" color="text.secondary">
                      Pass Rate: {scores.pass_rate.toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Questions: {scores.total_questions}
                    </Typography>
                  </Box>
                </Box>
              ))}
            </Box>
          ) : (
            <Box className="loading-spinner">
              <CircularProgress />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResultsDialog(false)}>
            Close
          </Button>
          {jobResults && (
            <Button 
              variant="contained"
              onClick={() => handleDownloadResults(selectedJob || '')}
            >
              Download Results
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default JobsPage;