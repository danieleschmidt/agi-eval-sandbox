// K6 Load Testing Configuration for AGI Evaluation Sandbox
// Comprehensive performance testing suite
// See: https://k6.io/docs/

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';
import { htmlReport } from 'https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';

// ==================================================
// Custom Metrics
// ==================================================
const errorRate = new Rate('error_rate');
const apiResponseTime = new Trend('api_response_time', true);
const evaluationCounter = new Counter('evaluation_requests');
const failedEvaluations = new Counter('failed_evaluations');
const authFailures = new Counter('auth_failures');

// ==================================================
// Configuration
// ==================================================
const BASE_URL = __ENV.API_URL || 'http://localhost:8000';
const API_VERSION = __ENV.API_VERSION || 'v1';
const API_PREFIX = `${BASE_URL}/api/${API_VERSION}`;

// Authentication
const API_KEY = __ENV.API_KEY || 'test-api-key';
const JWT_TOKEN = __ENV.JWT_TOKEN || '';

// Test data
const TEST_MODELS = [
  'gpt-4',
  'gpt-3.5-turbo',
  'claude-3-opus',
  'claude-3-sonnet',
];

const TEST_BENCHMARKS = [
  'mmlu',
  'humaneval',
  'truthfulqa',
  'hellaswag',
  'math',
];

// ==================================================
// Test Scenarios
// ==================================================
export const options = {
  scenarios: {
    // Smoke test - Minimal load to verify basic functionality
    smoke_test: {
      executor: 'constant-vus',
      vus: 1,
      duration: '30s',
      tags: { test_type: 'smoke' },
      env: { SCENARIO: 'smoke' },
    },
    
    // Load test - Expected normal load
    load_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 10 },   // Ramp up
        { duration: '5m', target: 10 },   // Stay at 10 users
        { duration: '2m', target: 20 },   // Ramp up to 20
        { duration: '5m', target: 20 },   // Stay at 20 users
        { duration: '2m', target: 0 },    // Ramp down
      ],
      tags: { test_type: 'load' },
      env: { SCENARIO: 'load' },
    },
    
    // Stress test - Above normal load
    stress_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 20 },   // Ramp up
        { duration: '5m', target: 50 },   // Ramp up to 50
        { duration: '2m', target: 100 },  // Ramp up to 100
        { duration: '5m', target: 100 },  // Stay at 100
        { duration: '2m', target: 0 },    // Ramp down
      ],
      tags: { test_type: 'stress' },
      env: { SCENARIO: 'stress' },
    },
    
    // Spike test - Sudden load increase
    spike_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 10 },   // Normal load
        { duration: '30s', target: 200 }, // Spike
        { duration: '3m', target: 200 },  // Maintain spike
        { duration: '30s', target: 10 },  // Back to normal
        { duration: '3m', target: 10 },   // Maintain normal
        { duration: '30s', target: 0 },   // Ramp down
      ],
      tags: { test_type: 'spike' },
      env: { SCENARIO: 'spike' },
    },
    
    // Soak test - Extended duration at moderate load
    soak_test: {
      executor: 'constant-vus',
      vus: 20,
      duration: '30m',
      tags: { test_type: 'soak' },
      env: { SCENARIO: 'soak' },
    },
    
    // Evaluation-focused test
    evaluation_intensive: {
      executor: 'ramping-arrival-rate',
      startRate: 0.1,
      timeUnit: '1s',
      preAllocatedVUs: 10,
      maxVUs: 50,
      stages: [
        { duration: '2m', target: 0.5 },  // 0.5 requests per second
        { duration: '5m', target: 1 },    // 1 request per second
        { duration: '5m', target: 2 },    // 2 requests per second
        { duration: '2m', target: 0 },    // Ramp down
      ],
      tags: { test_type: 'evaluation' },
      env: { SCENARIO: 'evaluation' },
    },
  },
  
  // Global thresholds
  thresholds: {
    http_req_failed: ['rate<0.05'],        // Error rate < 5%
    http_req_duration: ['p(95)<2000'],     // 95% of requests < 2s
    http_req_duration: ['p(99)<5000'],     // 99% of requests < 5s
    api_response_time: ['p(95)<1500'],     // API-specific response time
    error_rate: ['rate<0.1'],              // Custom error rate
    checks: ['rate>0.95'],                 // 95% of checks pass
  },
  
  // Test execution settings
  setupTimeout: '60s',
  teardownTimeout: '60s',
  noConnectionReuse: false,
  userAgent: 'k6-load-test/1.0 (AGI-Eval-Sandbox)',
  
  // Reporting
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)', 'p(99)'],
  summaryTimeUnit: 'ms',
};

// ==================================================
// Setup Function
// ================================================
export function setup() {
  console.log(`Starting load test against: ${BASE_URL}`);
  console.log(`Scenario: ${__ENV.SCENARIO || 'default'}`);
  
  // Health check
  const healthResponse = http.get(`${BASE_URL}/health`);
  if (healthResponse.status !== 200) {
    throw new Error(`Service is not healthy: ${healthResponse.status}`);
  }
  
  // Authentication setup if needed
  let authToken = JWT_TOKEN;
  if (!authToken && API_KEY) {
    const authResponse = http.post(`${API_PREFIX}/auth/login`, {
      api_key: API_KEY,
    });
    
    if (authResponse.status === 200) {
      const authData = JSON.parse(authResponse.body);
      authToken = authData.access_token;
    }
  }
  
  return {
    authToken,
    baseUrl: BASE_URL,
    apiPrefix: API_PREFIX,
  };
}

// ==================================================
// Main Test Function
// ==================================================
export default function (data) {
  const scenario = __ENV.SCENARIO || 'default';
  
  // Set up headers
  const headers = {
    'Content-Type': 'application/json',
    'User-Agent': 'k6-load-test/1.0',
  };
  
  if (data.authToken) {
    headers['Authorization'] = `Bearer ${data.authToken}`;
  } else if (API_KEY) {
    headers['X-API-Key'] = API_KEY;
  }
  
  // Test groups based on scenario
  switch (scenario) {
    case 'smoke':
      smokeTestScenario(data, headers);
      break;
    case 'evaluation':
      evaluationIntensiveScenario(data, headers);
      break;
    default:
      standardTestScenario(data, headers);
  }
  
  // Random sleep between requests
  sleep(Math.random() * 2 + 1); // 1-3 seconds
}

// ==================================================
// Test Scenarios
// ==================================================

function smokeTestScenario(data, headers) {
  group('Smoke Test - Basic Functionality', () => {
    // Health check
    let response = http.get(`${data.baseUrl}/health`);
    check(response, {
      'health check status is 200': (r) => r.status === 200,
      'health check response time < 500ms': (r) => r.timings.duration < 500,
    });
    
    // API info
    response = http.get(`${data.apiPrefix}/info`, { headers });
    check(response, {
      'API info status is 200': (r) => r.status === 200,
      'API info has version': (r) => JSON.parse(r.body).version !== undefined,
    });
    
    // List models
    response = http.get(`${data.apiPrefix}/models`, { headers });
    check(response, {
      'models list status is 200': (r) => r.status === 200,
      'models list is not empty': (r) => JSON.parse(r.body).length > 0,
    });
    
    apiResponseTime.add(response.timings.duration);
  });
}

function standardTestScenario(data, headers) {
  group('API Endpoints', () => {
    // Authentication test
    group('Authentication', () => {
      const response = http.get(`${data.apiPrefix}/auth/me`, { headers });
      const success = check(response, {
        'auth status is 200': (r) => r.status === 200,
      });
      
      if (!success) {
        authFailures.add(1);
      }
    });
    
    // Models endpoints
    group('Models', () => {
      let response = http.get(`${data.apiPrefix}/models`, { headers });
      check(response, {
        'models list success': (r) => r.status === 200,
        'models response time OK': (r) => r.timings.duration < 1000,
      });
      
      // Get specific model
      const modelId = TEST_MODELS[Math.floor(Math.random() * TEST_MODELS.length)];
      response = http.get(`${data.apiPrefix}/models/${modelId}`, { headers });
      check(response, {
        'model detail success': (r) => r.status === 200 || r.status === 404,
      });
      
      apiResponseTime.add(response.timings.duration);
    });
    
    // Benchmarks endpoints
    group('Benchmarks', () => {
      let response = http.get(`${data.apiPrefix}/benchmarks`, { headers });
      check(response, {
        'benchmarks list success': (r) => r.status === 200,
        'benchmarks response time OK': (r) => r.timings.duration < 1000,
      });
      
      // Get specific benchmark
      const benchmarkId = TEST_BENCHMARKS[Math.floor(Math.random() * TEST_BENCHMARKS.length)];
      response = http.get(`${data.apiPrefix}/benchmarks/${benchmarkId}`, { headers });
      check(response, {
        'benchmark detail success': (r) => r.status === 200 || r.status === 404,
      });
      
      apiResponseTime.add(response.timings.duration);
    });
    
    // Results endpoints
    group('Results', () => {
      const response = http.get(`${data.apiPrefix}/results?limit=10`, { headers });
      check(response, {
        'results list success': (r) => r.status === 200,
        'results response time OK': (r) => r.timings.duration < 2000,
      });
      
      apiResponseTime.add(response.timings.duration);
    });
  });
}

function evaluationIntensiveScenario(data, headers) {
  group('Evaluation Operations', () => {
    // Create evaluation request
    const evaluationPayload = {
      model: TEST_MODELS[Math.floor(Math.random() * TEST_MODELS.length)],
      benchmark: TEST_BENCHMARKS[Math.floor(Math.random() * TEST_BENCHMARKS.length)],
      config: {
        temperature: 0.7,
        max_tokens: 1000,
        timeout: 30,
      },
    };
    
    const response = http.post(
      `${data.apiPrefix}/evaluations`,
      JSON.stringify(evaluationPayload),
      { headers }
    );
    
    const success = check(response, {
      'evaluation creation success': (r) => r.status === 201 || r.status === 202,
      'evaluation response time OK': (r) => r.timings.duration < 3000,
    });
    
    evaluationCounter.add(1);
    if (!success) {
      failedEvaluations.add(1);
      errorRate.add(1);
    } else {
      errorRate.add(0);
    }
    
    // If evaluation was created, check its status
    if (response.status === 201 || response.status === 202) {
      const evaluationData = JSON.parse(response.body);
      if (evaluationData.id) {
        sleep(1); // Wait before checking status
        
        const statusResponse = http.get(
          `${data.apiPrefix}/evaluations/${evaluationData.id}`,
          { headers }
        );
        
        check(statusResponse, {
          'evaluation status check success': (r) => r.status === 200,
        });
      }
    }
    
    apiResponseTime.add(response.timings.duration);
  });
}

// ==================================================
// Teardown Function
// ==================================================
export function teardown(data) {
  console.log('Load test completed');
  console.log(`Base URL: ${data.baseUrl}`);
}

// ==================================================
// Custom Report Generation
// ==================================================
export function handleSummary(data) {
  const timestamp = new Date().toISOString().replace(/:/g, '-').split('.')[0];
  
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    [`reports/load-test-${timestamp}.html`]: htmlReport(data),
    [`reports/load-test-${timestamp}.json`]: JSON.stringify(data, null, 2),
  };
}

// ==================================================
// Utility Functions
// ==================================================

function randomChoice(array) {
  return array[Math.floor(Math.random() * array.length)];
}

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

// Generate realistic test data
function generateTestPrompt() {
  const prompts = [
    'What is the capital of France?',
    'Explain the theory of relativity in simple terms.',
    'Write a Python function to calculate factorial.',
    'What are the benefits of renewable energy?',
    'Describe the process of photosynthesis.',
  ];
  
  return randomChoice(prompts);
}

function generateTestConfig() {
  return {
    temperature: Math.random() * 0.8 + 0.2, // 0.2 to 1.0
    max_tokens: randomInt(100, 2000),
    top_p: Math.random() * 0.5 + 0.5, // 0.5 to 1.0
    frequency_penalty: Math.random() * 0.5,
    presence_penalty: Math.random() * 0.5,
  };
}