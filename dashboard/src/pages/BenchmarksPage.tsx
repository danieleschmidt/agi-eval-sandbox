import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Button,
  CircularProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import { ExpandMore, Assessment } from '@mui/icons-material';
import { useGetBenchmarksQuery, useGetBenchmarkDetailsQuery } from '../store/apiSlice';

const BenchmarksPage: React.FC = () => {
  const { data: benchmarks, isLoading, error } = useGetBenchmarksQuery();
  const [selectedBenchmark, setSelectedBenchmark] = React.useState<string | null>(null);
  
  const { data: benchmarkDetails } = useGetBenchmarkDetailsQuery(
    selectedBenchmark || '',
    { skip: !selectedBenchmark }
  );

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
        Failed to load benchmarks. Please try again later.
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom className="page-header">
        Available Benchmarks
      </Typography>

      <Grid container spacing={3}>
        {benchmarks?.map((benchmark) => (
          <Grid item xs={12} md={6} lg={4} key={benchmark.name}>
            <Card 
              className="benchmark-card"
              sx={{ 
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              <CardContent sx={{ flexGrow: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Assessment color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h6" component="h2">
                    {benchmark.name.toUpperCase()}
                  </Typography>
                </Box>

                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {benchmark.description}
                </Typography>

                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                  <Chip 
                    label={`${benchmark.total_questions} questions`} 
                    size="small" 
                    color="primary"
                  />
                  <Chip 
                    label={`v${benchmark.version}`} 
                    size="small" 
                    variant="outlined"
                  />
                </Box>

                {benchmark.categories.length > 0 && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" sx={{ mb: 1, fontWeight: 500 }}>
                      Categories:
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {benchmark.categories.map((category) => (
                        <Chip 
                          key={category}
                          label={category}
                          size="small"
                          variant="outlined"
                          color="secondary"
                        />
                      ))}
                    </Box>
                  </Box>
                )}

                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => setSelectedBenchmark(
                    selectedBenchmark === benchmark.name ? null : benchmark.name
                  )}
                  fullWidth
                  sx={{ mt: 'auto' }}
                >
                  {selectedBenchmark === benchmark.name ? 'Hide Details' : 'View Details'}
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Benchmark Details */}
      {selectedBenchmark && benchmarkDetails && (
        <Box sx={{ mt: 4 }}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                {selectedBenchmark.toUpperCase()} Details
              </Typography>

              <Grid container spacing={3} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="h6" color="primary">
                    {benchmarkDetails.total_questions}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Questions
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="h6" color="primary">
                    {Object.keys(benchmarkDetails.question_types).length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Question Types
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="h6" color="primary">
                    {Object.keys(benchmarkDetails.categories).length}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Categories
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="h6" color="primary">
                    v{benchmarkDetails.version}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Version
                  </Typography>
                </Grid>
              </Grid>

              {/* Question Types */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="h6">Question Types</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    {Object.entries(benchmarkDetails.question_types).map(([type, count]) => (
                      <Grid item xs={12} sm={6} md={4} key={type}>
                        <Box sx={{ textAlign: 'center', p: 2, border: '1px solid #eee', borderRadius: 1 }}>
                          <Typography variant="h6" color="primary">
                            {count as number}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {type.replace('_', ' ').toUpperCase()}
                          </Typography>
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
                </AccordionDetails>
              </Accordion>

              {/* Categories */}
              {Object.keys(benchmarkDetails.categories).length > 0 && (
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="h6">Categories</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      {Object.entries(benchmarkDetails.categories).map(([category, count]) => (
                        <Grid item xs={12} sm={6} md={4} key={category}>
                          <Box sx={{ textAlign: 'center', p: 2, border: '1px solid #eee', borderRadius: 1 }}>
                            <Typography variant="h6" color="primary">
                              {count as number}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {category.toUpperCase()}
                            </Typography>
                          </Box>
                        </Grid>
                      ))}
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              )}

              {/* Sample Questions */}
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="h6">Sample Questions</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  {benchmarkDetails.sample_questions.map((question: any, index: number) => (
                    <Box key={question.id} sx={{ mb: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 500, mb: 1 }}>
                        Question {index + 1}: {question.id}
                      </Typography>
                      
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          <strong>Prompt:</strong>
                        </Typography>
                        <Typography variant="body2" sx={{ pl: 2, fontStyle: 'italic' }}>
                          "{question.prompt}"
                        </Typography>
                      </Box>

                      {question.choices && (
                        <Box sx={{ mb: 2 }}>
                          <Typography variant="body2" sx={{ mb: 1 }}>
                            <strong>Choices:</strong>
                          </Typography>
                          {question.choices.map((choice: string, choiceIndex: number) => (
                            <Typography key={choiceIndex} variant="body2" sx={{ pl: 2 }}>
                              {String.fromCharCode(65 + choiceIndex)}. {choice}
                            </Typography>
                          ))}
                        </Box>
                      )}

                      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        <Chip label={`Type: ${question.type}`} size="small" />
                        {question.category && (
                          <Chip label={`Category: ${question.category}`} size="small" variant="outlined" />
                        )}
                        {question.difficulty && (
                          <Chip label={`Difficulty: ${question.difficulty}`} size="small" color="secondary" />
                        )}
                      </Box>
                    </Box>
                  ))}
                </AccordionDetails>
              </Accordion>
            </CardContent>
          </Card>
        </Box>
      )}

      {!benchmarks || benchmarks.length === 0 ? (
        <Alert severity="info" sx={{ mt: 3 }}>
          No benchmarks are currently available. Please check back later.
        </Alert>
      ) : null}
    </Box>
  );
};

export default BenchmarksPage;