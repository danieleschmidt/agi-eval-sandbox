import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { ModelSpec, EvaluationConfig } from './apiSlice';

interface EvaluationState {
  currentModel: ModelSpec | null;
  currentConfig: EvaluationConfig;
  selectedBenchmarks: string[];
  isEvaluating: boolean;
  currentJobId: string | null;
  results: any | null;
}

const initialState: EvaluationState = {
  currentModel: null,
  currentConfig: {
    temperature: 0.0,
    max_tokens: 2048,
    parallel: true,
  },
  selectedBenchmarks: ['all'],
  isEvaluating: false,
  currentJobId: null,
  results: null,
};

const evaluationSlice = createSlice({
  name: 'evaluation',
  initialState,
  reducers: {
    setCurrentModel: (state, action: PayloadAction<ModelSpec>) => {
      state.currentModel = action.payload;
    },
    setCurrentConfig: (state, action: PayloadAction<Partial<EvaluationConfig>>) => {
      state.currentConfig = { ...state.currentConfig, ...action.payload };
    },
    setSelectedBenchmarks: (state, action: PayloadAction<string[]>) => {
      state.selectedBenchmarks = action.payload;
    },
    setIsEvaluating: (state, action: PayloadAction<boolean>) => {
      state.isEvaluating = action.payload;
    },
    setCurrentJobId: (state, action: PayloadAction<string | null>) => {
      state.currentJobId = action.payload;
    },
    setResults: (state, action: PayloadAction<any>) => {
      state.results = action.payload;
    },
    resetEvaluation: (state) => {
      state.currentJobId = null;
      state.isEvaluating = false;
      state.results = null;
    },
  },
});

export const {
  setCurrentModel,
  setCurrentConfig,
  setSelectedBenchmarks,
  setIsEvaluating,
  setCurrentJobId,
  setResults,
  resetEvaluation,
} = evaluationSlice.actions;

export default evaluationSlice.reducer;