import React, { useState } from 'react';
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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  CircularProgress,
  Alert,
  TableSortLabel,
  Avatar,
  Tooltip,
} from '@mui/material';
import { 
  EmojiEvents, 
  TrendingUp, 
  Psychology,
  Assessment 
} from '@mui/icons-material';
import { format } from 'date-fns';
import { useGetLeaderboardQuery, useGetBenchmarksQuery } from '../store/apiSlice';

const LeaderboardPage: React.FC = () => {
  const [selectedBenchmark, setSelectedBenchmark] = useState<string>('');
  const [sortMetric, setSortMetric] = useState<string>('average_score');
  const [limit, setLimit] = useState<number>(50);

  const { data: benchmarks } = useGetBenchmarksQuery();
  const { 
    data: leaderboard, 
    isLoading, 
    error 
  } = useGetLeaderboardQuery({
    benchmark: selectedBenchmark || undefined,
    metric: sortMetric,
    limit: limit,
  });

  const getRankIcon = (rank: number) => {
    switch (rank) {
      case 1:
        return <EmojiEvents sx={{ color: '#FFD700' }} />;
      case 2:
        return <EmojiEvents sx={{ color: '#C0C0C0' }} />;
      case 3:
        return <EmojiEvents sx={{ color: '#CD7F32' }} />;
      default:
        return <Typography variant="body2" className="rank-badge">#{rank}</Typography>;
    }
  };

  const getProviderColor = (provider: string) => {
    switch (provider.toLowerCase()) {
      case 'openai':
        return 'primary';
      case 'anthropic':
        return 'secondary';
      case 'local':
        return 'default';
      default:
        return 'info';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return '#4caf50';
    if (score >= 0.6) return '#ff9800';
    return '#f44336';
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
        Failed to load leaderboard. Please try again later.
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom className="page-header">
        🏆 Model Leaderboard
      </Typography>

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Filters
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={4}>
              <FormControl fullWidth>
                <InputLabel>Benchmark</InputLabel>
                <Select
                  value={selectedBenchmark}
                  label="Benchmark"
                  onChange={(e) => setSelectedBenchmark(e.target.value)}
                >
                  <MenuItem value="">All Benchmarks</MenuItem>
                  {benchmarks?.map((benchmark) => (
                    <MenuItem key={benchmark.name} value={benchmark.name}>
                      {benchmark.name.toUpperCase()}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={4}>
              <FormControl fullWidth>
                <InputLabel>Sort By</InputLabel>
                <Select
                  value={sortMetric}
                  label="Sort By"
                  onChange={(e) => setSortMetric(e.target.value)}
                >
                  <MenuItem value="average_score">Average Score</MenuItem>
                  <MenuItem value="pass_rate">Pass Rate</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={4}>
              <FormControl fullWidth>
                <InputLabel>Limit</InputLabel>
                <Select
                  value={limit}
                  label="Limit"
                  onChange={(e) => setLimit(Number(e.target.value))}
                >
                  <MenuItem value={10}>Top 10</MenuItem>
                  <MenuItem value={25}>Top 25</MenuItem>
                  <MenuItem value={50}>Top 50</MenuItem>
                  <MenuItem value={100}>Top 100</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Leaderboard Table */}
      {leaderboard && leaderboard.length > 0 ? (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Results ({leaderboard.length} entries)
            </Typography>
            
            <TableContainer component={Paper} elevation={0}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Rank</TableCell>
                    <TableCell>Model</TableCell>
                    <TableCell>Provider</TableCell>
                    <TableCell>Benchmark</TableCell>
                    <TableCell align="center">
                      <TableSortLabel active={sortMetric === 'average_score'}>
                        Score
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="center">
                      <TableSortLabel active={sortMetric === 'pass_rate'}>
                        Pass Rate
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="center">Questions</TableCell>
                    <TableCell align="center">Date</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {leaderboard.map((entry) => (
                    <TableRow 
                      key={`${entry.model_name}-${entry.benchmark}-${entry.timestamp}`}
                      hover
                      sx={{ 
                        '&:nth-of-type(odd)': { 
                          backgroundColor: 'rgba(0, 0, 0, 0.02)' 
                        },
                        cursor: 'pointer',
                      }}
                    >
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', minWidth: 60 }}>
                          {getRankIcon(entry.rank)}
                        </Box>
                      </TableCell>
                      
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Avatar sx={{ mr: 2, bgcolor: 'primary.main' }}>
                            <Psychology />
                          </Avatar>
                          <Box>
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              {entry.model_name}
                            </Typography>
                          </Box>
                        </Box>
                      </TableCell>

                      <TableCell>
                        <Chip 
                          label={entry.model_provider} 
                          size="small"
                          color={getProviderColor(entry.model_provider) as any}
                          variant="outlined"
                        />
                      </TableCell>

                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Assessment sx={{ mr: 1, fontSize: 18, color: 'text.secondary' }} />
                          <Typography variant="body2">
                            {entry.benchmark.toUpperCase()}
                          </Typography>
                        </Box>
                      </TableCell>

                      <TableCell align="center">
                        <Box>
                          <Typography 
                            variant="body2" 
                            sx={{ 
                              fontWeight: 600,
                              color: getScoreColor(entry.average_score)
                            }}
                          >
                            {(entry.average_score * 100).toFixed(1)}%
                          </Typography>
                          <Box 
                            className="score-bar" 
                            sx={{ width: 60, mx: 'auto', mt: 0.5 }}
                          >
                            <Box 
                              className="score-fill"
                              sx={{ 
                                width: `${entry.average_score * 100}%`,
                                backgroundColor: getScoreColor(entry.average_score)
                              }}
                            />
                          </Box>
                        </Box>
                      </TableCell>

                      <TableCell align="center">
                        <Typography variant="body2">
                          {entry.pass_rate.toFixed(1)}%
                        </Typography>
                      </TableCell>

                      <TableCell align="center">
                        <Typography variant="body2">
                          {entry.total_questions}
                        </Typography>
                      </TableCell>

                      <TableCell align="center">
                        <Tooltip title={format(new Date(entry.timestamp), 'PPpp')}>
                          <Typography variant="body2" color="text.secondary">
                            {format(new Date(entry.timestamp), 'MMM dd')}
                          </Typography>
                        </Tooltip>
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
          No evaluation results found. Run some evaluations to see the leaderboard!
        </Alert>
      )}

      {/* Summary Stats */}
      {leaderboard && leaderboard.length > 0 && (
        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent className="metric-card">
                <Typography className="metric-value">
                  {new Set(leaderboard.map(e => e.model_name)).size}
                </Typography>
                <Typography className="metric-label">
                  Unique Models
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent className="metric-card">
                <Typography className="metric-value">
                  {new Set(leaderboard.map(e => e.model_provider)).size}
                </Typography>
                <Typography className="metric-label">
                  Providers
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent className="metric-card">
                <Typography className="metric-value">
                  {(leaderboard.reduce((sum, e) => sum + e.average_score, 0) / leaderboard.length * 100).toFixed(1)}%
                </Typography>
                <Typography className="metric-label">
                  Avg Score
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent className="metric-card">
                <Typography className="metric-value">
                  {Math.max(...leaderboard.map(e => e.average_score * 100)).toFixed(1)}%
                </Typography>
                <Typography className="metric-label">
                  Top Score
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default LeaderboardPage;