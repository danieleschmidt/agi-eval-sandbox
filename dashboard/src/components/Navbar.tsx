import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Home,
  Assessment,
  Leaderboard,
  Work,
  Compare,
  Psychology,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const Navbar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const navigationItems = [
    { path: '/', label: 'Home', icon: <Home /> },
    { path: '/evaluate', label: 'Evaluate', icon: <Psychology /> },
    { path: '/benchmarks', label: 'Benchmarks', icon: <Assessment /> },
    { path: '/leaderboard', label: 'Leaderboard', icon: <Leaderboard /> },
    { path: '/jobs', label: 'Jobs', icon: <Work /> },
    { path: '/compare', label: 'Compare', icon: <Compare /> },
  ];

  return (
    <AppBar position="static" elevation={1}>
      <Toolbar>
        <Typography
          variant="h6"
          component="div"
          sx={{ flexGrow: 0, mr: 4, fontWeight: 600 }}
        >
          🧠 AGI Eval Sandbox
        </Typography>

        <Box sx={{ flexGrow: 1, display: 'flex', gap: 1 }}>
          {navigationItems.map(({ path, label, icon }) => (
            <Tooltip key={path} title={label}>
              <Button
                color="inherit"
                startIcon={icon}
                onClick={() => navigate(path)}
                sx={{
                  textTransform: 'none',
                  backgroundColor: location.pathname === path ? 'rgba(255,255,255,0.1)' : 'transparent',
                  '&:hover': {
                    backgroundColor: 'rgba(255,255,255,0.1)',
                  },
                }}
              >
                {label}
              </Button>
            </Tooltip>
          ))}
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="body2" sx={{ opacity: 0.8 }}>
            v0.1.0
          </Typography>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;