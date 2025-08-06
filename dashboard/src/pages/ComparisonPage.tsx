import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Grid,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  OutlinedInput,
  IconButton,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  SelectChangeEvent,
} from '@mui/material';
import { Add, Delete, Compare, Psychology, ContentCopy } from '@mui/icons-material';
import { useDispatch } from 'react-redux';
import {
  useGetBenchmarksQuery,
  useCompareModelsMutation,
  useGetJobStatusQuery,
  useGetJobResultsQuery,
  ModelSpec,
} from '../store/apiSlice';
import { addNotification } from '../store/uiSlice';

interface ModelConfig extends ModelSpec {
  id: string;
}

const ComparisonPage: React.FC = () => {
  const dispatch = useDispatch();
  
  const [models, setModels] = useState<ModelConfig[]>([
    { id: '1', provider: 'openai', name: 'gpt-4', api_key: '' },
    { id: '2', provider: 'anthropic', name: 'claude-3-opus', api_key: '' },
  ]);
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>(['all']);
  const [temperature, setTemperature] = useState(0.0);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [numQuestions, setNumQuestions] = useState<number | undefined>(undefined);
  const [comparisonJobId, setComparisonJobId] = useState<string | null>(null);
  const [isComparing, setIsComparing] = useState(false);

  const { data: benchmarks } = useGetBenchmarksQuery();
  const [compareModels, { isLoading: startingComparison }] = useCompareModelsMutation();
  
  const { data: jobStatus } = useGetJobStatusQuery(
    comparisonJobId || '',
    { skip: !comparisonJobId, pollingInterval: isComparing ? 2000 : 0 }
  );
  
  const { data: comparisonResults } = useGetJobResultsQuery(
    comparisonJobId || '',
    { skip: !comparisonJobId || jobStatus?.status !== 'completed' }
  );

  // Update comparison status
  React.useEffect(() => {
    if (jobStatus) {
      if (jobStatus.status === 'completed') {
        setIsComparing(false);
        dispatch(addNotification({
          message: 'Model comparison completed!',
          type: 'success',
        }));
      } else if (jobStatus.status === 'failed') {
        setIsComparing(false);
        dispatch(addNotification({
          message: `Comparison failed: ${jobStatus.error || 'Unknown error'}`,
          type: 'error',
        }));
      }
    }
  }, [jobStatus, dispatch]);

  const addModel = () => {
    const newId = Math.max(...models.map(m => parseInt(m.id))) + 1;
    setModels([...models, {
      id: newId.toString(),
      provider: 'openai',
      name: 'gpt-4',
      api_key: '',
    }]);
  };

  const duplicateModel = (modelToDuplicate: ModelConfig) => {
    const newId = Math.max(...models.map(m => parseInt(m.id))) + 1;
    const duplicated = {
      ...modelToDuplicate,
      id: newId.toString(),
      name: `${modelToDuplicate.name} (copy)`,
    };
    setModels([...models, duplicated]);
  };

  const removeModel = (id: string) => {
    if (models.length > 2) {
      setModels(models.filter(m => m.id !== id));
    }
  };

  const updateModel = (id: string, updates: Partial<ModelConfig>) => {
    setModels(models.map(m => m.id === id ? { ...m, ...updates } : m));
  };

  const handleBenchmarkChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value as string[];
    setSelectedBenchmarks(value);
  };

  const validateComparison = (): boolean => {
    // Validate models
    for (const model of models) {
      if (!model.name?.trim()) {
        dispatch(addNotification({
          message: 'Please enter names for all models',
          type: 'error',
        }));
        return false;
      }
      if (model.provider !== 'local' && !model.api_key?.trim()) {
        dispatch(addNotification({
          message: 'API keys are required for non-local models',
          type: 'error',
        }));
        return false;
      }
    }

    // Validate at least 2 models
    if (models.length < 2) {
      dispatch(addNotification({
        message: 'At least 2 models are required for comparison',
        type: 'error',
      }));
      return false;
    }

    // Validate benchmarks selection
    if (selectedBenchmarks.length === 0) {
      dispatch(addNotification({
        message: 'Please select at least one benchmark',
        type: 'error',
      }));
      return false;
    }

    // Validate configuration
    if (temperature < 0 || temperature > 2) {
      dispatch(addNotification({
        message: 'Temperature must be between 0 and 2',
        type: 'error',
      }));
      return false;
    }

    if (maxTokens < 1 || maxTokens > 8192) {
      dispatch(addNotification({
        message: 'Max tokens must be between 1 and 8192',
        type: 'error',
      }));
      return false;
    }

    return true;
  };

  const handleStartComparison = async () => {
    // Validate form before submitting
    if (!validateComparison()) {
      return;
    }

    try {
      const modelsToCompare = models.map(({ id, ...model }) => model);
      
      const response = await compareModels({
        models: modelsToCompare,
        benchmarks: selectedBenchmarks,
        config: {
          temperature,
          max_tokens: maxTokens,
          num_questions: numQuestions,
          parallel: true,
        },
      }).unwrap();

      setComparisonJobId(response.job_id);
      setIsComparing(true);
      dispatch(addNotification({
        message: 'Model comparison started!',
        type: 'success',
      }));

    } catch (error: any) {
      const errorMessage = error.data?.detail || error.message || 'Unknown error occurred';
      dispatch(addNotification({
        message: `Failed to start comparison: ${errorMessage}`,
        type: 'error',
      }));
      console.error('Comparison start error:', error);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom className="page-header">
        Model Comparison
      </Typography>

      <Grid container spacing={3}>
        {/* Configuration */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Models to Compare
              </Typography>

              {models.map((model, index) => (
                <Box key={model.id} sx={{ mb: 3, p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
                      Model {index + 1}
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <IconButton 
                        size="small" 
                        color="primary"
                        onClick={() => duplicateModel(model)}
                        disabled={isComparing || models.length >= 5}
                        title="Duplicate model"
                      >
                        <ContentCopy />
                      </IconButton>
                      {models.length > 2 && (
                        <IconButton 
                          size="small" 
                          color="error"
                          onClick={() => removeModel(model.id)}
                          disabled={isComparing}
                          title="Remove model"
                        >
                          <Delete />
                        </IconButton>
                      )}
                    </Box>
                  </Box>

                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Provider</InputLabel>
                        <Select
                          value={model.provider}
                          label="Provider"
                          onChange={(e) => updateModel(model.id, {
                            provider: e.target.value as ModelSpec['provider'],
                            name: e.target.value === 'openai' ? 'gpt-4' :
                                  e.target.value === 'anthropic' ? 'claude-3-opus' : 'local-model'
                          })}
                          disabled={isComparing}
                        >
                          <MenuItem value="openai">OpenAI</MenuItem>
                          <MenuItem value="anthropic">Anthropic</MenuItem>
                          <MenuItem value="local">Local</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        size="small"
                        label="Model Name"
                        value={model.name}
                        onChange={(e) => updateModel(model.id, { name: e.target.value })}
                        disabled={isComparing}
                      />
                    </Grid>
                    {model.provider !== 'local' && (
                      <Grid item xs={12}>
                        <TextField
                          fullWidth
                          size="small"
                          label="API Key"
                          type="password"
                          value={model.api_key}
                          onChange={(e) => updateModel(model.id, { api_key: e.target.value })}
                          disabled={isComparing}
                        />
                      </Grid>
                    )}
                  </Grid>
                </Box>
              ))}

              <Button
                variant="outlined"
                startIcon={<Add />}
                onClick={addModel}
                disabled={isComparing || models.length >= 5}
                fullWidth
                sx={{ mb: 3 }}
              >
                Add Model
              </Button>

              {/* Benchmarks */}
              <FormControl fullWidth sx={{ mb: 3 }}>
                <InputLabel>Benchmarks</InputLabel>
                <Select
                  multiple
                  value={selectedBenchmarks}
                  onChange={handleBenchmarkChange}
                  input={<OutlinedInput label="Benchmarks" />}
                  disabled={isComparing}
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
                      {benchmark.name.toUpperCase()}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Configuration */}
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Temperature"
                    type="number"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    disabled={isComparing}
                    inputProps={{ min: 0, max: 2, step: 0.1 }}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Max Tokens"
                    type="number"
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                    disabled={isComparing}
                    inputProps={{ min: 1, max: 8192 }}
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Questions per Benchmark (optional)"
                    type="number"
                    value={numQuestions || ''}
                    onChange={(e) => setNumQuestions(e.target.value ? parseInt(e.target.value) : undefined)}
                    disabled={isComparing}
                    inputProps={{ min: 1 }}
                  />
                </Grid>
              </Grid>

              <Button
                variant="contained"
                startIcon={<Compare />}
                onClick={handleStartComparison}
                disabled={isComparing || startingComparison || models.length < 2}
                fullWidth
                size="large"
              >
                {startingComparison ? 'Starting...' : 'Start Comparison'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Results */}
        <Grid item xs={12} md={6}>
          {/* Comparison Status */}
          {comparisonJobId && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Comparison Status
                </Typography>
                
                {jobStatus ? (
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="body2">
                        Job ID: {comparisonJobId.slice(0, 8)}...
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
                  </Box>
                ) : (
                  <Alert severity="info">Loading comparison status...</Alert>
                )}
              </CardContent>
            </Card>
          )}

          {/* Comparison Results */}
          {comparisonResults && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Comparison Results
                </Typography>
                
                <TableContainer component={Paper} elevation={0} className="comparison-table">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Model</TableCell>
                        <TableCell align="center">Overall Score</TableCell>
                        <TableCell align="center">Pass Rate</TableCell>
                        <TableCell align="center">Questions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(comparisonResults.results).map(([modelName, results]: [string, any]) => (
                        <TableRow key={modelName}>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <Psychology sx={{ mr: 1, fontSize: 18 }} />
                              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                {modelName}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell align="center">
                            <Typography variant="body2" sx={{ fontWeight: 600, color: 'primary.main' }}>
                              {(results.overall_score * 100).toFixed(1)}%
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <Typography variant="body2">
                              {results.overall_pass_rate.toFixed(1)}%
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <Typography variant="body2">
                              {results.total_questions}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>

                {/* Detailed Benchmark Results */}
                <Typography variant="subtitle1" sx={{ mt: 3, mb: 2 }}>
                  Benchmark Breakdown
                </Typography>
                
                {Object.keys(Object.values(comparisonResults.results)[0].benchmark_scores).map((benchmarkName) => (
                  <Box key={benchmarkName} sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      {benchmarkName.toUpperCase()}
                    </Typography>
                    <TableContainer component={Paper} elevation={0} size="small">
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Model</TableCell>
                            <TableCell align="center">Score</TableCell>
                            <TableCell align="center">Pass Rate</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {Object.entries(comparisonResults.results)
                            .sort(([, a]: [string, any], [, b]: [string, any]) => 
                              b.benchmark_scores[benchmarkName].average_score - a.benchmark_scores[benchmarkName].average_score
                            )
                            .map(([modelName, results]: [string, any]) => (
                            <TableRow key={modelName}>
                              <TableCell>
                                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                  {modelName}
                                </Typography>
                              </TableCell>
                              <TableCell align="center">
                                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                  {(results.benchmark_scores[benchmarkName].average_score * 100).toFixed(1)}%
                                </Typography>
                              </TableCell>
                              <TableCell align="center">
                                <Typography variant="body2">
                                  {results.benchmark_scores[benchmarkName].pass_rate.toFixed(1)}%
                                </Typography>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Box>
                ))}
              </CardContent>
            </Card>
          )}

          {/* Help Text */}
          {!comparisonJobId && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Model Comparison
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Compare multiple models side-by-side on the same benchmarks. 
                  Add at least 2 models, select your benchmarks, and start the comparison.
                  Results will show detailed performance metrics for each model.
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default ComparisonPage;