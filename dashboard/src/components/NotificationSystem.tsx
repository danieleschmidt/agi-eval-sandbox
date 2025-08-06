import React, { useEffect } from 'react';
import {
  Snackbar,
  Alert,
  AlertColor,
  Box,
  IconButton,
} from '@mui/material';
import { Close } from '@mui/icons-material';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../store/store';
import { removeNotification } from '../store/uiSlice';

const NotificationSystem: React.FC = () => {
  const dispatch = useDispatch();
  const notifications = useSelector((state: RootState) => state.ui.notifications);

  // Auto-remove notifications after 6 seconds
  useEffect(() => {
    const timers: NodeJS.Timeout[] = [];

    notifications.forEach((notification) => {
      if (notification.type !== 'error') {
        const timer = setTimeout(() => {
          dispatch(removeNotification(notification.id));
        }, 6000);
        timers.push(timer);
      }
    });

    return () => {
      timers.forEach(timer => clearTimeout(timer));
    };
  }, [notifications, dispatch]);

  const handleClose = (notificationId: string) => {
    dispatch(removeNotification(notificationId));
  };

  return (
    <Box sx={{ position: 'fixed', bottom: 24, right: 24, zIndex: 9999 }}>
      {notifications.map((notification, index) => (
        <Snackbar
          key={notification.id}
          open={true}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
          sx={{
            position: 'static',
            mb: index > 0 ? 1 : 0,
            '& .MuiSnackbar-root': {
              position: 'static',
              transform: 'none',
            },
          }}
        >
          <Alert
            severity={notification.type as AlertColor}
            variant="filled"
            action={
              <IconButton
                size="small"
                color="inherit"
                onClick={() => handleClose(notification.id)}
              >
                <Close fontSize="small" />
              </IconButton>
            }
            sx={{ minWidth: '300px' }}
          >
            {notification.message}
          </Alert>
        </Snackbar>
      ))}
    </Box>
  );
};

export default NotificationSystem;