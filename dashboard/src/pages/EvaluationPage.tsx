import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Grid,
  Alert,
  CircularProgress,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControlLabel,
  Switch,
  OutlinedInput,
  SelectChangeEvent,
} from '@mui/material';
import { ExpandMore, PlayArrow, Stop } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import {
  useGetBenchmarksQuery,
  useStartEvaluationMutation,
  useGetJobStatusQuery,
  useGetJobResultsQuery,
  ModelSpec,
} from '../store/apiSlice';
import { RootState } from '../store/store';
import {
  setCurrentModel,
  setCurrentConfig,
  setSelectedBenchmarks,
  setIsEvaluating,
  setCurrentJobId,
  setResults,
  resetEvaluation,
} from '../store/evaluationSlice';
import { addNotification } from '../store/uiSlice';

const EvaluationPage: React.FC = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  
  const {
    currentModel,
    currentConfig,
    selectedBenchmarks,
    isEvaluating,
    currentJobId,
    results,
  } = useSelector((state: RootState) => state.evaluation);

  const [formModel, setFormModel] = useState<ModelSpec>({
    provider: 'openai',
    name: 'gpt-4',
    api_key: '',
  });

  const { data: benchmarks, isLoading: benchmarksLoading } = useGetBenchmarksQuery();
  const [startEvaluation, { isLoading: startingEvaluation }] = useStartEvaluationMutation();
  
  const { data: jobStatus, refetch: refetchJobStatus } = useGetJobStatusQuery(
    currentJobId || '',
    { skip: !currentJobId, pollingInterval: isEvaluating ? 2000 : 0 }
  );
  
  const { data: jobResults } = useGetJobResultsQuery(
    currentJobId || '',
    { skip: !currentJobId || jobStatus?.status !== 'completed' }
  );

  // Update job status and results
  useEffect(() => {
    if (jobStatus) {
      if (jobStatus.status === 'completed') {
        dispatch(setIsEvaluating(false));
        if (jobResults) {
          dispatch(setResults(jobResults.results));
          dispatch(addNotification({
            message: 'Evaluation completed successfully!',
            type: 'success',
          }));
        }
      } else if (jobStatus.status === 'failed') {
        dispatch(setIsEvaluating(false));
        dispatch(addNotification({
          message: `Evaluation failed: ${jobStatus.error || 'Unknown error'}`,
          type: 'error',
        }));
      }
    }
  }, [jobStatus, jobResults, dispatch]);

  const handleProviderChange = (event: SelectChangeEvent) => {
    const provider = event.target.value as ModelSpec['provider'];
    setFormModel({
      ...formModel,
      provider,
      name: provider === 'openai' ? 'gpt-4' : provider === 'anthropic' ? 'claude-3-opus' : 'local-model',
    });
  };

  const handleBenchmarkChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value as string[];
    dispatch(setSelectedBenchmarks(value));
  };

  const handleStartEvaluation = async () => {
    try {
      // Validate inputs
      if (!formModel.name) {
        dispatch(addNotification({
          message: 'Please enter a model name',
          type: 'error',
        }));
        return;
      }

      if (formModel.provider !== 'local' && !formModel.api_key) {
        dispatch(addNotification({
          message: 'API key is required for this provider',
          type: 'error',
        }));
        return;
      }

      dispatch(setCurrentModel(formModel));
      
      const response = await startEvaluation({
        model: formModel,
        benchmarks: selectedBenchmarks,
        config: currentConfig,
      }).unwrap();

      dispatch(setCurrentJobId(response.job_id));
      dispatch(setIsEvaluating(true));
      dispatch(addNotification({
        message: 'Evaluation started successfully!',
        type: 'success',
      }));

    } catch (error: any) {
      dispatch(addNotification({
        message: `Failed to start evaluation: ${error.data?.detail || error.message}`,
        type: 'error',  
      }));
    }
  };

  const handleStopEvaluation = () => {
    dispatch(resetEvaluation());
    dispatch(addNotification({
      message: 'Evaluation stopped',
      type: 'info',
    }));
  };

  const handleViewResults = () => {
    navigate('/jobs');
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom className="page-header">
        Model Evaluation
      </Typography>

      <Grid container spacing={3}>
        {/* Configuration Form */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Model Configuration
              </Typography>

              <Box sx={{ mb: 3 }}>
                <FormControl fullWidth>
                  <InputLabel>Provider</InputLabel>
                  <Select
                    value={formModel.provider}
                    label="Provider"
                    onChange={handleProviderChange}
                    disabled={isEvaluating}
                  >
                    <MenuItem value="openai">OpenAI</MenuItem>
                    <MenuItem value="anthropic">Anthropic</MenuItem>
                    <MenuItem value="local">Local (Testing)</MenuItem>
                  </Select>
                </FormControl>
              </Box>

              <Box sx={{ mb: 3 }}>
                <TextField
                  fullWidth
                  label="Model Name"
                  value={formModel.name}
                  onChange={(e) => setFormModel({ ...formModel, name: e.target.value })}
                  disabled={isEvaluating}
                  placeholder={
                    formModel.provider === 'openai' ? 'gpt-4, gpt-3.5-turbo, etc.' :
                    formModel.provider === 'anthropic' ? 'claude-3-opus, claude-3-sonnet, etc.' :
                    'local-model'
                  }
                />
              </Box>

              {formModel.provider !== 'local' && (
                <Box sx={{ mb: 3 }}>
                  <TextField
                    fullWidth
                    label="API Key"
                    type="password"
                    value={formModel.api_key}
                    onChange={(e) => setFormModel({ ...formModel, api_key: e.target.value })}
                    disabled={isEvaluating}
                    placeholder="Enter your API key"
                  />
                </Box>
              )}

              <Box sx={{ mb: 3 }}>
                <FormControl fullWidth>
                  <InputLabel>Benchmarks</InputLabel>
                  <Select
                    multiple
                    value={selectedBenchmarks}
                    onChange={handleBenchmarkChange}
                    input={<OutlinedInput label="Benchmarks" />}
                    disabled={isEvaluating || benchmarksLoading}
                    renderValue={(selected) => (
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {selected.map((value) => (
                          <Chip key={value} label={value} size="small" />
                        ))}
                      </Box>
                    )}
                  >
                    <MenuItem value="all">All Benchmarks</MenuItem>
                    {benchmarks?.map((benchmark) => (
                      <MenuItem key={benchmark.name} value={benchmark.name}>
                        {benchmark.name.toUpperCase()} ({benchmark.total_questions} questions)
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Box>

              {/* Advanced Configuration */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography>Advanced Configuration</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <TextField
                      label="Temperature"
                      type="number"
                      value={currentConfig.temperature}
                      onChange={(e) => dispatch(setCurrentConfig({ temperature: parseFloat(e.target.value) }))}
                      disabled={isEvaluating}
                      inputProps={{ min: 0, max: 2, step: 0.1 }}
                      helperText="Controls randomness (0 = deterministic, 2 = very random)"
                    />
                    <TextField
                      label="Max Tokens"
                      type="number"
                      value={currentConfig.max_tokens}
                      onChange={(e) => dispatch(setCurrentConfig({ max_tokens: parseInt(e.target.value) }))}
                      disabled={isEvaluating}
                      inputProps={{ min: 1, max: 8192 }}
                      helperText="Maximum tokens per response"
                    />
                    <TextField
                      label="Number of Questions (optional)"
                      type="number"
                      value={currentConfig.num_questions || ''}
                      onChange={(e) => dispatch(setCurrentConfig({ 
                        num_questions: e.target.value ? parseInt(e.target.value) : undefined 
                      }))}
                      disabled={isEvaluating}
                      inputProps={{ min: 1 }}
                      helperText="Limit questions per benchmark (leave empty for all)"
                    />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={currentConfig.parallel}
                          onChange={(e) => dispatch(setCurrentConfig({ parallel: e.target.checked }))}
                          disabled={isEvaluating}
                        />
                      }
                      label="Parallel Execution"
                    />
                  </Box>
                </AccordionDetails>
              </Accordion>

              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  startIcon={<PlayArrow />}
                  onClick={handleStartEvaluation}
                  disabled={isEvaluating || startingEvaluation}
                  fullWidth
                >
                  {startingEvaluation ? 'Starting...' : 'Start Evaluation'}
                </Button>
                {isEvaluating && (
                  <Button
                    variant="outlined"
                    startIcon={<Stop />}
                    onClick={handleStopEvaluation}
                    color="error"
                  >
                    Stop
                  </Button>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Status and Results */}
        <Grid item xs={12} md={6}>
          {/* Job Status */}
          {currentJobId && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Evaluation Status
                </Typography>
                
                {jobStatus ? (
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="body2">
                        Job ID: {currentJobId.slice(0, 8)}...
                      </Typography>
                      <Chip 
                        label={jobStatus.status} 
                        color={
                          jobStatus.status === 'completed' ? 'success' :
                          jobStatus.status === 'failed' ? 'error' :
                          jobStatus.status === 'running' ? 'primary' :
                          'default'
                        }
                        size="small"
                      />
                    </Box>
                    
                    <LinearProgress 
                      variant="determinate" 
                      value={jobStatus.progress * 100} 
                      sx={{ mb: 2 }}
                    />
                    
                    <Typography variant="body2" color="text.secondary">
                      Progress: {(jobStatus.progress * 100).toFixed(1)}%
                    </Typography>
                    
                    {jobStatus.error && (
                      <Alert severity="error" sx={{ mt: 2 }}>
                        {jobStatus.error}
                      </Alert>
                    )}
                  </Box>
                ) : (
                  <Box className="loading-spinner">
                    <CircularProgress />
                  </Box>
                )}
              </CardContent>
            </Card>
          )}

          {/* Results Summary */}
          {results && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Results Summary
                </Typography>
                
                <Box sx={{ mb: 3 }}>
                  <Typography variant="h4" color="primary" gutterBottom>
                    {(results.overall_score * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    Overall Score
                  </Typography>
                </Box>

                <Grid container spacing={2} sx={{ mb: 3 }}>
                  <Grid item xs={6}>
                    <Typography variant="h6">
                      {results.overall_pass_rate.toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Pass Rate
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="h6">
                      {results.total_questions}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Questions
                    </Typography>
                  </Grid>
                </Grid>

                <Typography variant="subtitle1" gutterBottom>
                  Benchmark Scores:
                </Typography>
                {Object.entries(results.benchmark_scores).map(([benchmark, scores]: [string, any]) => (
                  <Box key={benchmark} sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">
                        {benchmark.toUpperCase()}
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {(scores.average_score * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <Box className="score-bar">
                      <Box 
                        className="score-fill"
                        sx={{ width: `${scores.average_score * 100}%` }}
                      />
                    </Box>
                  </Box>
                ))}

                <Button
                  fullWidth
                  variant="outlined"
                  onClick={handleViewResults}
                  sx={{ mt: 2 }}
                >
                  View Detailed Results
                </Button>
              </CardContent>
            </Card>
          )}

          {/* Benchmarks Info */}
          {!currentJobId && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Available Benchmarks
                </Typography>
                
                {benchmarksLoading ? (
                  <Box className="loading-spinner">
                    <CircularProgress />
                  </Box>
                ) : benchmarks && benchmarks.length > 0 ? (
                  <Box>
                    {benchmarks.map((benchmark) => (
                      <Box key={benchmark.name} sx={{ mb: 2, pb: 2, borderBottom: '1px solid #eee' }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
                          {benchmark.name.toUpperCase()}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                          {benchmark.description}
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Chip label={`${benchmark.total_questions} questions`} size="small" />
                          <Chip label={`v${benchmark.version}`} size="small" variant="outlined" />
                        </Box>
                      </Box>
                    ))}
                  </Box>
                ) : (
                  <Alert severity="info">No benchmarks available</Alert>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default EvaluationPage;