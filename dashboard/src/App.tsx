import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Container } from '@mui/material';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import EvaluationPage from './pages/EvaluationPage';
import BenchmarksPage from './pages/BenchmarksPage';
import LeaderboardPage from './pages/LeaderboardPage';
import JobsPage from './pages/JobsPage';
import ComparisonPage from './pages/ComparisonPage';
import './App.css';

function App() {
  return (
    <div className="App">
      <Navbar />
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/evaluate" element={<EvaluationPage />} />
          <Route path="/benchmarks" element={<BenchmarksPage />} />
          <Route path="/leaderboard" element={<LeaderboardPage />} />
          <Route path="/jobs" element={<JobsPage />} />
          <Route path="/compare" element={<ComparisonPage />} />
        </Routes>
      </Container>
    </div>
  );
}

export default App;