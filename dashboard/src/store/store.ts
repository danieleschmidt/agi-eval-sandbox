import { configureStore } from '@reduxjs/toolkit';
import { apiSlice } from './apiSlice';
import evaluationReducer from './evaluationSlice';
import uiReducer from './uiSlice';

export const store = configureStore({
  reducer: {
    api: apiSlice.reducer,
    evaluation: evaluationReducer,
    ui: uiReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(apiSlice.middleware),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;