import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Paper,
  CircularProgress,
  Alert,
  Chip,
} from '@mui/material';
import {
  Psychology,
  Assessment,
  TrendingUp,
  Speed,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useGetStatsQuery, useGetBenchmarksQuery, useGetLeaderboardQuery } from '../store/apiSlice';

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const { data: stats, isLoading: statsLoading, error: statsError } = useGetStatsQuery();
  const { data: benchmarks, isLoading: benchmarksLoading } = useGetBenchmarksQuery();
  const { data: leaderboard, isLoading: leaderboardLoading } = useGetLeaderboardQuery({ limit: 5 });

  const quickActions = [
    {
      title: 'Start Evaluation',
      description: 'Evaluate a model on available benchmarks',
      icon: <Psychology color="primary" sx={{ fontSize: 40 }} />,
      action: () => navigate('/evaluate'),
      disabled: false,
    },
    {
      title: 'View Benchmarks',
      description: 'Explore available evaluation benchmarks',
      icon: <Assessment color="primary" sx={{ fontSize: 40 }} />,
      action: () => navigate('/benchmarks'),
      disabled: false,
    },
    {
      title: 'Compare Models',
      description: 'Compare multiple models side by side',
      icon: <TrendingUp color="primary" sx={{ fontSize: 40 }} />,
      action: () => navigate('/compare'),
      disabled: false,
    },
    {
      title: 'View Leaderboard',
      description: 'See top performing models',
      icon: <Speed color="primary" sx={{ fontSize: 40 }} />,
      action: () => navigate('/leaderboard'),
      disabled: false,
    },
  ];

  return (
    <Box>
      {/* Hero Section */}
      <Paper
        elevation={0}
        sx={{
          p: 6,
          mb: 4,
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          borderRadius: 2,
        }}
      >
        <Typography variant="h2" component="h1" gutterBottom align="center">
          AGI Evaluation Sandbox
        </Typography>
        <Typography variant="h5" align="center" sx={{ opacity: 0.9, mb: 3 }}>
          Comprehensive evaluation platform for large language models
        </Typography>
        <Box display="flex" justifyContent="center" gap={2}>
          <Button
            variant="contained"
            size="large"
            sx={{ bgcolor: 'rgba(255,255,255,0.2)', '&:hover': { bgcolor: 'rgba(255,255,255,0.3)' } }}
            onClick={() => navigate('/evaluate')}
          >
            Start Evaluating
          </Button>
          <Button
            variant="outlined"
            size="large"
            sx={{ color: 'white', borderColor: 'rgba(255,255,255,0.5)' }}
            onClick={() => navigate('/benchmarks')}
          >
            View Benchmarks
          </Button>
        </Box>
      </Paper>

      {/* Stats Section */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent className="metric-card">
              {statsLoading ? (
                <CircularProgress size={24} />
              ) : (
                <>
                  <Typography className="metric-value">
                    {stats?.available_benchmarks || 0}
                  </Typography>
                  <Typography className="metric-label">
                    Available Benchmarks
                  </Typography>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent className="metric-card">
              {statsLoading ? (
                <CircularProgress size={24} />
              ) : (
                <>
                  <Typography className="metric-value">
                    {stats?.total_jobs || 0}
                  </Typography>
                  <Typography className="metric-label">
                    Total Evaluations
                  </Typography>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent className="metric-card">
              {statsLoading ? (
                <CircularProgress size={24} />
              ) : (
                <>
                  <Typography className="metric-value">
                    {stats?.running_jobs || 0}
                  </Typography>
                  <Typography className="metric-label">
                    Running Jobs
                  </Typography>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent className="metric-card">
              {statsLoading ? (
                <CircularProgress size={24} />
              ) : (
                <>
                  <Typography className="metric-value">
                    {stats?.completed_jobs || 0}
                  </Typography>
                  <Typography className="metric-label">
                    Completed Jobs
                  </Typography>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Quick Actions */}
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Quick Actions
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {quickActions.map((action, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card 
              className="benchmark-card"
              sx={{ 
                cursor: action.disabled ? 'not-allowed' : 'pointer',
                opacity: action.disabled ? 0.6 : 1,
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
              }}
              onClick={action.disabled ? undefined : action.action}
            >
              <CardContent sx={{ flexGrow: 1, textAlign: 'center', p: 3 }}>
                <Box sx={{ mb: 2 }}>
                  {action.icon}
                </Box>
                <Typography variant="h6" gutterBottom>
                  {action.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {action.description}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Recent Activity & Leaderboard */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
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
                  {benchmarks.slice(0, 3).map((benchmark) => (
                    <Box key={benchmark.name} sx={{ mb: 2, pb: 2, borderBottom: '1px solid #eee' }}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
                        {benchmark.name.toUpperCase()}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        {benchmark.description}
                      </Typography>
                      <Box display="flex" gap={1} flexWrap="wrap">
                        <Chip label={`${benchmark.total_questions} questions`} size="small" />
                        <Chip label={`v${benchmark.version}`} size="small" variant="outlined" />
                      </Box>
                    </Box>
                  ))}
                  <Button
                    fullWidth
                    variant="outlined"
                    onClick={() => navigate('/benchmarks')}
                    sx={{ mt: 2 }}
                  >
                    View All Benchmarks
                  </Button>
                </Box>
              ) : (
                <Alert severity="info">No benchmarks available</Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Top Performers
              </Typography>
              {leaderboardLoading ? (
                <Box className="loading-spinner">
                  <CircularProgress />
                </Box>
              ) : leaderboard && leaderboard.length > 0 ? (
                <Box>
                  {leaderboard.map((entry, index) => (
                    <Box key={entry.rank} sx={{ mb: 2, pb: 2, borderBottom: '1px solid #eee' }}>
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Box>
                          <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
                            #{entry.rank} {entry.model_name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {entry.benchmark} • {entry.model_provider}
                          </Typography>
                        </Box>
                        <Box textAlign="right">
                          <Typography variant="h6" color="primary">
                            {(entry.average_score * 100).toFixed(1)}%
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {entry.pass_rate.toFixed(1)}% pass
                          </Typography>
                        </Box>
                      </Box>
                    </Box>
                  ))}
                  <Button
                    fullWidth
                    variant="outlined"
                    onClick={() => navigate('/leaderboard')}
                    sx={{ mt: 2 }}
                  >
                    View Full Leaderboard
                  </Button>
                </Box>
              ) : (
                <Alert severity="info">No evaluation results yet</Alert>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Status Section */}
      {statsError && (
        <Alert severity="warning" sx={{ mt: 3 }}>
          Unable to connect to API. Some features may not be available.
        </Alert>
      )}
    </Box>
  );
};

export default HomePage;