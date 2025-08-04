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
} from '@mui/material';
import {
  Refresh,
  Delete,
  Stop,
  Visibility,
  GetApp,
} from '@mui/icons-material';
import { format } from 'date-fns';
import {
  useGetJobsQuery,
  useGetJobStatusQuery,
  useGetJobResultsQuery,
  useCancelJobMutation,
  JobListItem,
} from '../store/apiSlice';

const JobsPage: React.FC = () => {
  const [selectedJob, setSelectedJob] = React.useState<string | null>(null);
  const [resultsDialog, setResultsDialog] = React.useState(false);
  
  const { data: jobsData, isLoading, error, refetch } = useGetJobsQuery();
  const [cancelJob] = useCancelJobMutation();
  
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

  const handleCancelJob = async (jobId: string) => {
    try {
      await cancelJob(jobId).unwrap();
      refetch();
    } catch (error) {
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

  const jobs = jobsData?.jobs || [];

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" className="page-header">
          Evaluation Jobs
        </Typography>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={() => refetch()}
        >
          Refresh
        </Button>
      </Box>

      {jobs.length > 0 ? (
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
                  {jobs.map((job: JobListItem) => (
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
                                onClick={() => handleCancelJob(job.job_id)}
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
                                onClick={() => handleCancelJob(job.job_id)}
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
          No evaluation jobs found. Start an evaluation to see jobs here.
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