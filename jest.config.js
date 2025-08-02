/** @type {import('jest').Config} */
module.exports = {
  // Basic configuration
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  roots: ['<rootDir>/dashboard/src', '<rootDir>/tests'],
  
  // Test file patterns
  testMatch: [
    '**/__tests__/**/*.(ts|tsx|js|jsx)',
    '**/*.(test|spec).(ts|tsx|js|jsx)'
  ],
  
  // Module resolution
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/dashboard/src/$1',
    '^@/components/(.*)$': '<rootDir>/dashboard/src/components/$1',
    '^@/utils/(.*)$': '<rootDir>/dashboard/src/utils/$1',
    '^@/types/(.*)$': '<rootDir>/dashboard/src/types/$1',
    '^@/hooks/(.*)$': '<rootDir>/dashboard/src/hooks/$1',
    '^@/services/(.*)$': '<rootDir>/dashboard/src/services/$1',
    '^@/store/(.*)$': '<rootDir>/dashboard/src/store/$1',
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    '\\.(jpg|jpeg|png|gif|eot|otf|webp|svg|ttf|woff|woff2|mp4|webm|wav|mp3|m4a|aac|oga)$': '<rootDir>/tests/__mocks__/fileMock.js'
  },
  
  // File extensions
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  
  // Transform configuration
  transform: {
    '^.+\\.(ts|tsx)$': 'ts-jest',
    '^.+\\.(js|jsx)$': 'babel-jest'
  },
  
  // Setup files
  setupFilesAfterEnv: [
    '<rootDir>/tests/setup/jest.setup.ts'
  ],
  
  // Coverage configuration
  collectCoverage: true,
  collectCoverageFrom: [
    'dashboard/src/**/*.{ts,tsx,js,jsx}',
    '!dashboard/src/**/*.d.ts',
    '!dashboard/src/**/*.stories.{ts,tsx,js,jsx}',
    '!dashboard/src/**/index.{ts,tsx,js,jsx}',
    '!dashboard/src/vite-env.d.ts'
  ],
  coverageDirectory: '<rootDir>/coverage',
  coverageReporters: [
    'text',
    'text-summary',
    'html',
    'lcov',
    'json'
  ],
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 70,
      lines: 70,
      statements: 70
    }
  },
  
  // Test environment options
  testEnvironmentOptions: {
    customExportConditions: ['node', 'node-addons']
  },
  
  // Timeout settings
  testTimeout: 10000,
  
  // Error handling
  errorOnDeprecated: true,
  
  // Watch mode
  watchPathIgnorePatterns: [
    '<rootDir>/node_modules/',
    '<rootDir>/dist/',
    '<rootDir>/build/',
    '<rootDir>/coverage/'
  ],
  
  // Verbose output
  verbose: true,
  
  // Reporters
  reporters: [
    'default',
    ['jest-junit', {
      outputDirectory: 'test-results',
      outputName: 'jest-results.xml'
    }],
    ['jest-html-reporters', {
      publicPath: 'test-results',
      filename: 'jest-report.html',
      expand: true
    }]
  ],
  
  // Global variables
  globals: {
    'ts-jest': {
      useESM: true,
      tsconfig: {
        jsx: 'react-jsx'
      }
    }
  },
  
  // Clear mocks
  clearMocks: true,
  restoreMocks: true,
  
  // Max workers for parallel execution
  maxWorkers: '50%',
  
  // Test categories
  projects: [
    {
      displayName: 'unit',
      testMatch: ['<rootDir>/tests/unit/**/*.(test|spec).(ts|tsx|js|jsx)'],
      testEnvironment: 'jsdom'
    },
    {
      displayName: 'integration', 
      testMatch: ['<rootDir>/tests/integration/**/*.(test|spec).(ts|tsx|js|jsx)'],
      testEnvironment: 'node',
      testTimeout: 30000
    }
  ]
};