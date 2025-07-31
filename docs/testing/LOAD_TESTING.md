# Load Testing Guide

## Overview

This document provides comprehensive guidance for load testing the AGI Evaluation Sandbox, including test strategies, tools configuration, and performance analysis.

## Load Testing Strategy

### Testing Objectives

1. **Performance Validation**
   - Verify system meets performance requirements
   - Identify performance bottlenecks
   - Validate scaling behavior

2. **Capacity Planning**
   - Determine maximum sustainable throughput
   - Identify resource limits
   - Plan for growth scenarios

3. **Reliability Assessment**
   - Test system stability under load
   - Validate error handling
   - Assess recovery capabilities

### Test Types

#### 1. Smoke Testing
- **Purpose**: Basic functionality verification
- **Load**: Minimal (1-5 users)
- **Duration**: 5-10 minutes
- **Goal**: Ensure system is ready for load testing

#### 2. Load Testing
- **Purpose**: Normal expected load
- **Load**: Expected production traffic
- **Duration**: 15-60 minutes
- **Goal**: Verify performance under normal conditions

#### 3. Stress Testing
- **Purpose**: Beyond normal capacity
- **Load**: Above expected maximum
- **Duration**: 15-30 minutes
- **Goal**: Find breaking point and failure modes

#### 4. Spike Testing
- **Purpose**: Sudden traffic increases
- **Load**: Rapid ramp-up to high levels
- **Duration**: 10-20 minutes
- **Goal**: Test system's ability to handle traffic spikes

#### 5. Volume Testing
- **Purpose**: Large amounts of data
- **Load**: High data volumes
- **Duration**: Extended periods
- **Goal**: Test system with large datasets

#### 6. Endurance Testing
- **Purpose**: Extended periods under load
- **Load**: Normal expected load
- **Duration**: Several hours/days
- **Goal**: Identify memory leaks and degradation

## Test Environment Setup

### Environment Requirements

```yaml
# test-environment.yml
environment:
  name: "load-testing"
  isolation: true
  
infrastructure:
  kubernetes:
    nodes: 5
    node_type: "n1-standard-4"
    
  database:
    instance_type: "db-r5.xlarge"
    read_replicas: 2
    
  cache:
    redis_cluster: true
    memory: "8GB"
    
  monitoring:
    prometheus: true
    grafana: true
    jaeger: true

resources:
  api_service:
    replicas: 3
    cpu_limit: "2000m"
    memory_limit: "4Gi"
    
  database:
    connections: 100
    shared_buffers: "1GB"
    
  cache:
    max_memory: "4GB"
    eviction_policy: "allkeys-lru"
```

### Test Data Preparation

```python
# test-data-generator.py
import asyncio
import json
import random
from typing import List, Dict
from faker import Faker

class TestDataGenerator:
    """Generate realistic test data for load testing"""
    
    def __init__(self):
        self.fake = Faker()
        
    def generate_evaluation_requests(self, count: int) -> List[Dict]:
        """Generate evaluation request payloads"""
        models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"]
        benchmarks = ["mmlu", "humaneval", "truthfulqa", "hellaswag"]
        
        requests = []
        for _ in range(count):
            request = {
                "model": random.choice(models),
                "benchmark": random.choice(benchmarks),
                "prompts": [
                    self.fake.text(max_nb_chars=200) for _ in range(random.randint(1, 10))
                ],
                "config": {
                    "temperature": random.uniform(0.0, 1.0),
                    "max_tokens": random.randint(100, 2000),
                    "timeout": random.randint(30, 300)
                }
            }
            requests.append(request)
            
        return requests
    
    def generate_user_scenarios(self, count: int) -> List[Dict]:
        """Generate user behavior scenarios"""
        scenarios = []
        
        for _ in range(count):
            scenario = {
                "user_id": self.fake.uuid4(),
                "session_duration": random.randint(300, 3600),  # 5-60 minutes
                "actions": self._generate_user_actions(),
                "think_time": random.uniform(1.0, 5.0)  # seconds between actions
            }
            scenarios.append(scenario)
            
        return scenarios
    
    def _generate_user_actions(self) -> List[str]:
        """Generate realistic user action sequences"""
        action_patterns = [
            ["login", "list_models", "start_evaluation", "check_status", "download_results", "logout"],
            ["login", "view_dashboard", "compare_models", "export_data", "logout"],
            ["login", "upload_dataset", "create_benchmark", "run_evaluation", "logout"],
        ]
        return random.choice(action_patterns)
    
    async def save_test_data(self, filename: str, data: List[Dict]):
        """Save test data to file"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

# Generate test data
async def main():
    generator = TestDataGenerator()
    
    # Generate evaluation requests
    eval_requests = generator.generate_evaluation_requests(10000)
    await generator.save_test_data("eval_requests.json", eval_requests)
    
    # Generate user scenarios
    user_scenarios = generator.generate_user_scenarios(1000)
    await generator.save_test_data("user_scenarios.json", user_scenarios)

if __name__ == "__main__":
    asyncio.run(main())
```

## Load Testing Tools

### K6 Configuration

#### Basic Load Test
```javascript
// k6-basic-load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';
import { Rate, Trend, Counter } from 'k6/metrics';

// Load test data
const testData = new SharedArray('test data', function () {
  return JSON.parse(open('./eval_requests.json'));
});

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');
const requestCount = new Counter('requests');

export const options = {
  scenarios: {
    smoke_test: {
      executor: 'constant-vus',
      vus: 5,
      duration: '5m',
      tags: { test_type: 'smoke' },
    },
    load_test: {
      executor: 'ramping-vus',
      startVUs: 10,
      stages: [
        { duration: '5m', target: 50 },
        { duration: '10m', target: 50 },
        { duration: '5m', target: 0 },
      ],
      tags: { test_type: 'load' },
    },
    stress_test: {
      executor: 'ramping-vus',
      startVUs: 10,
      stages: [
        { duration: '2m', target: 100 },
        { duration: '5m', target: 100 },
        { duration: '2m', target: 200 },
        { duration: '5m', target: 200 },
        { duration: '2m', target: 0 },
      ],
      tags: { test_type: 'stress' },
    },
    spike_test: {
      executor: 'ramping-vus',
      startVUs: 10,
      stages: [
        { duration: '1m', target: 10 },
        { duration: '30s', target: 200 },
        { duration: '1m', target: 200 },
        { duration: '30s', target: 10 },
        { duration: '1m', target: 10 },
      ],
      tags: { test_type: 'spike' },
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.1'],
    errors: ['rate<0.05'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'https://api.agi-eval.your-org.com';

export default function () {
  const testRequest = testData[Math.floor(Math.random() * testData.length)];
  
  // Authentication
  const authResponse = http.post(`${BASE_URL}/auth/login`, JSON.stringify({
    username: 'test-user',
    password: 'test-password'
  }), {
    headers: { 'Content-Type': 'application/json' },
  });
  
  if (!check(authResponse, { 'auth successful': (r) => r.status === 200 })) {
    errorRate.add(1);
    return;
  }
  
  const token = authResponse.json('access_token');
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  };
  
  // Health check
  const healthResponse = http.get(`${BASE_URL}/health`);
  check(healthResponse, { 'health check OK': (r) => r.status === 200 });
  
  // List models
  const modelsResponse = http.get(`${BASE_URL}/api/v1/models`, { headers });
  check(modelsResponse, { 'models listed': (r) => r.status === 200 });
  
  // Start evaluation
  const evalResponse = http.post(
    `${BASE_URL}/api/v1/evaluate`,
    JSON.stringify(testRequest),
    { headers }
  );
  
  const evalSuccess = check(evalResponse, {
    'evaluation started': (r) => r.status === 202,
    'job_id returned': (r) => r.json('job_id') !== undefined,
  });
  
  if (evalSuccess) {
    const jobId = evalResponse.json('job_id');
    
    // Poll for completion (with timeout)
    let attempts = 0;
    const maxAttempts = 60; // 5 minutes with 5-second intervals
    
    while (attempts < maxAttempts) {
      sleep(5);
      
      const statusResponse = http.get(
        `${BASE_URL}/api/v1/jobs/${jobId}`,
        { headers }
      );
      
      if (statusResponse.status === 200) {
        const status = statusResponse.json('status');
        if (status === 'completed' || status === 'failed') {
          break;
        }
      }
      
      attempts++;
    }
  } else {
    errorRate.add(1);
  }
  
  requestCount.add(1);
  responseTime.add(evalResponse.timings.duration);
  
  // Think time
  sleep(Math.random() * 3 + 1); // 1-4 seconds
}

export function handleSummary(data) {
  return {
    'load-test-results.json': JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}
```

#### Advanced Scenario Testing
```javascript
// k6-scenario-test.js
import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { SharedArray } from 'k6/data';

const userScenarios = new SharedArray('user scenarios', function () {
  return JSON.parse(open('./user_scenarios.json'));
});

export const options = {
  scenarios: {
    user_journey: {
      executor: 'per-vu-iterations',
      vus: 50,
      iterations: 10,
      maxDuration: '30m',
    },
  },
};

export default function () {
  const scenario = userScenarios[Math.floor(Math.random() * userScenarios.length)];
  const BASE_URL = __ENV.BASE_URL || 'https://api.agi-eval.your-org.com';
  
  let authToken = '';
  
  group('User Session', function () {
    scenario.actions.forEach((action, index) => {
      group(`Action ${index + 1}: ${action}`, function () {
        switch (action) {
          case 'login':
            const loginResponse = http.post(`${BASE_URL}/auth/login`, JSON.stringify({
              username: `user_${scenario.user_id.slice(-8)}`,
              password: 'test-password'
            }), {
              headers: { 'Content-Type': 'application/json' },
            });
            
            if (check(loginResponse, { 'login successful': (r) => r.status === 200 })) {
              authToken = loginResponse.json('access_token');
            }
            break;
            
          case 'list_models':
            http.get(`${BASE_URL}/api/v1/models`, {
              headers: { 'Authorization': `Bearer ${authToken}` },
            });
            break;
            
          case 'start_evaluation':
            http.post(`${BASE_URL}/api/v1/evaluate`, JSON.stringify({
              model: 'gpt-4',
              benchmark: 'mmlu',
              prompts: ['Test prompt']
            }), {
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${authToken}`,
              },
            });
            break;
            
          case 'view_dashboard':
            http.get(`${BASE_URL}/api/v1/dashboard`, {
              headers: { 'Authorization': `Bearer ${authToken}` },
            });
            break;
            
          case 'logout':
            http.post(`${BASE_URL}/auth/logout`, {}, {
              headers: { 'Authorization': `Bearer ${authToken}` },
            });
            break;
        }
        
        // Think time between actions
        sleep(scenario.think_time);
      });
    });
  });
}
```

### Artillery Configuration

#### Comprehensive Load Test
```yaml
# artillery-comprehensive.yml
config:
  target: 'https://api.agi-eval.your-org.com'
  plugins:
    - artillery-plugin-prometheus
    - artillery-plugin-metrics-by-endpoint
    
  prometheus:
    pushgateway: 'http://prometheus-pushgateway:9091'
    
  phases:
    # Smoke test
    - name: "Smoke test"
      duration: 300
      arrivalRate: 2
      
    # Load test
    - name: "Load test - ramp up"
      duration: 600
      arrivalRate: 5
      rampTo: 25
      
    - name: "Load test - sustained"
      duration: 1200
      arrivalRate: 25
      
    # Stress test
    - name: "Stress test"
      duration: 600
      arrivalRate: 25
      rampTo: 100
      
    # Spike test
    - name: "Spike test"
      duration: 60
      arrivalRate: 200
      
    # Cool down
    - name: "Cool down"
      duration: 300
      arrivalRate: 10
      
  payload:
    path: "./test-data/eval_requests.csv"
    fields:
      - "model"
      - "benchmark"
      - "prompt"
      
  defaults:
    headers:
      Content-Type: "application/json"

scenarios:
  - name: "Health and status checks"
    weight: 10
    flow:
      - get:
          url: "/health"
          expect:
            - statusCode: 200
            - hasProperty: "status"
            
      - get:
          url: "/api/v1/status"
          expect:
            - statusCode: 200
            
  - name: "Authentication flow"
    weight: 15
    flow:
      - post:
          url: "/auth/login"
          json:
            username: "{{ $randomString() }}"
            password: "test-password"
          capture:
            - json: "$.access_token"
              as: "authToken"
          expect:
            - statusCode: 200
            
      - get:
          url: "/api/v1/profile"
          headers:
            Authorization: "Bearer {{ authToken }}"
          expect:
            - statusCode: 200
            
  - name: "Model operations"
    weight: 25
    flow:
      - post:
          url: "/auth/login"
          json:
            username: "test-user"
            password: "test-password"
          capture:
            - json: "$.access_token"
              as: "authToken"
              
      - get:
          url: "/api/v1/models"
          headers:
            Authorization: "Bearer {{ authToken }}"
          expect:
            - statusCode: 200
            - contentType: json
            
      - get:
          url: "/api/v1/models/gpt-4"
          headers:
            Authorization: "Bearer {{ authToken }}"
          expect:
            - statusCode: 200
            
  - name: "Evaluation workflow"
    weight: 50
    flow:
      - post:
          url: "/auth/login"
          json:
            username: "test-user"
            password: "test-password"
          capture:
            - json: "$.access_token"
              as: "authToken"
              
      - post:
          url: "/api/v1/evaluate"
          headers:
            Authorization: "Bearer {{ authToken }}"
          json:
            model: "{{ model }}"
            benchmark: "{{ benchmark }}"
            prompts: ["{{ prompt }}"]
            config:
              temperature: 0.7
              max_tokens: 1000
          capture:
            - json: "$.job_id"
              as: "jobId"
          expect:
            - statusCode: 202
            
      - loop:
          - get:
              url: "/api/v1/jobs/{{ jobId }}"
              headers:
                Authorization: "Bearer {{ authToken }}"
              capture:
                - json: "$.status"
                  as: "jobStatus"
          while: "{{ jobStatus !== 'completed' && jobStatus !== 'failed' }}"
          count: 60
          delay: 5
          
      - get:
          url: "/api/v1/jobs/{{ jobId }}/results"
          headers:
            Authorization: "Bearer {{ authToken }}"
          expect:
            - statusCode: 200
            
after:
  flow:
    - log: "Load test completed"
```

### JMeter Configuration

#### JMeter Test Plan (XML)
```xml
<?xml version="1.0" encoding="UTF-8"?>
<jmeterTestPlan version="1.2" properties="5.0" jmeter="5.4.1">
  <hashTree>
    <TestPlan guiclass="TestPlanGui" testclass="TestPlan" testname="AGI Eval Sandbox Load Test" enabled="true">
      <stringProp name="TestPlan.comments">Comprehensive load test for AGI Evaluation Sandbox</stringProp>
      <boolProp name="TestPlan.functional_mode">false</boolProp>
      <boolProp name="TestPlan.tearDown_on_shutdown">true</boolProp>
      <boolProp name="TestPlan.serialize_threadgroups">false</boolProp>
      <elementProp name="TestPlan.arguments" elementType="Arguments" guiclass="ArgumentsPanel" testclass="Arguments" testname="User Defined Variables" enabled="true">
        <collectionProp name="Arguments.arguments">
          <elementProp name="BASE_URL" elementType="Argument">
            <stringProp name="Argument.name">BASE_URL</stringProp>
            <stringProp name="Argument.value">https://api.agi-eval.your-org.com</stringProp>
          </elementProp>
        </collectionProp>
      </elementProp>
      <stringProp name="TestPlan.user_define_classpath"></stringProp>
    </TestPlan>
    
    <hashTree>
      <!-- Thread Groups for different test scenarios -->
      <ThreadGroup guiclass="ThreadGroupGui" testclass="ThreadGroup" testname="Load Test - Normal Users" enabled="true">
        <stringProp name="ThreadGroup.on_sample_error">continue</stringProp>
        <elementProp name="ThreadGroup.main_controller" elementType="LoopController" guiclass="LoopControllerGui" testclass="LoopController" testname="Loop Controller" enabled="true">
          <boolProp name="LoopController.continue_forever">false</boolProp>
          <stringProp name="LoopController.loops">-1</stringProp>
        </elementProp>
        <stringProp name="ThreadGroup.num_threads">50</stringProp>
        <stringProp name="ThreadGroup.ramp_time">300</stringProp>
        <longProp name="ThreadGroup.start_time">1640995200000</longProp>
        <longProp name="ThreadGroup.end_time">1640995200000</longProp>
        <boolProp name="ThreadGroup.scheduler">true</boolProp>
        <stringProp name="ThreadGroup.duration">1800</stringProp>
        <stringProp name="ThreadGroup.delay">0</stringProp>
        <boolProp name="ThreadGroup.same_user_on_next_iteration">true</boolProp>
      </ThreadGroup>
      
      <hashTree>
        <!-- HTTP Request Defaults -->
        <ConfigTestElement guiclass="HttpDefaultsGui" testclass="ConfigTestElement" testname="HTTP Request Defaults" enabled="true">
          <elementProp name="HTTPsampler.Arguments" elementType="Arguments" guiclass="HTTPArgumentsPanel" testclass="Arguments" testname="User Defined Variables" enabled="true">
            <collectionProp name="Arguments.arguments"/>
          </elementProp>
          <stringProp name="HTTPSampler.domain">${BASE_URL}</stringProp>
          <stringProp name="HTTPSampler.port"></stringProp>
          <stringProp name="HTTPSampler.protocol">https</stringProp>
          <stringProp name="HTTPSampler.contentEncoding"></stringProp>
          <stringProp name="HTTPSampler.path"></stringProp>
          <stringProp name="HTTPSampler.concurrentPool">6</stringProp>
          <stringProp name="HTTPSampler.connect_timeout">10000</stringProp>
          <stringProp name="HTTPSampler.response_timeout">30000</stringProp>
        </ConfigTestElement>
        
        <!-- Authentication Sampler -->
        <HTTPSamplerProxy guiclass="HttpTestSampleGui" testclass="HTTPSamplerProxy" testname="Login" enabled="true">
          <elementProp name="HTTPsampler.Arguments" elementType="Arguments" guiclass="HTTPArgumentsPanel" testclass="Arguments" testname="User Defined Variables" enabled="true">
            <collectionProp name="Arguments.arguments"/>
          </elementProp>
          <stringProp name="HTTPSampler.domain"></stringProp>
          <stringProp name="HTTPSampler.port"></stringProp>
          <stringProp name="HTTPSampler.protocol"></stringProp>
          <stringProp name="HTTPSampler.contentEncoding"></stringProp>
          <stringProp name="HTTPSampler.path">/auth/login</stringProp>
          <stringProp name="HTTPSampler.method">POST</stringProp>
          <boolProp name="HTTPSampler.follow_redirects">true</boolProp>
          <boolProp name="HTTPSampler.auto_redirects">false</boolProp>
          <boolProp name="HTTPSampler.use_keepalive">true</boolProp>
          <boolProp name="HTTPSampler.DO_MULTIPART_POST">false</boolProp>
          <stringProp name="HTTPSampler.embedded_url_re"></stringProp>
          <stringProp name="HTTPSampler.connect_timeout"></stringProp>
          <stringProp name="HTTPSampler.response_timeout"></stringProp>
        </HTTPSamplerProxy>
        
        <!-- More samplers... -->
      </hashTree>
    </hashTree>
  </hashTree>
</jmeterTestPlan>
```

## Performance Analysis

### Metrics Collection

#### Key Performance Indicators
```yaml
# performance-kpis.yml
response_time_metrics:
  - name: "p50_response_time"
    description: "50th percentile response time"
    target: "<200ms"
    
  - name: "p95_response_time"
    description: "95th percentile response time"
    target: "<500ms"
    
  - name: "p99_response_time"
    description: "99th percentile response time"
    target: "<1000ms"

throughput_metrics:
  - name: "requests_per_second"
    description: "Requests processed per second"
    target: ">100 RPS"
    
  - name: "concurrent_users"
    description: "Maximum concurrent users"
    target: ">500 users"

reliability_metrics:
  - name: "error_rate"
    description: "Percentage of failed requests"
    target: "<1%"
    
  - name: "availability"
    description: "System uptime percentage"
    target: ">99.9%"

resource_metrics:
  - name: "cpu_utilization"
    description: "Average CPU usage"
    target: "<70%"
    
  - name: "memory_utilization"
    description: "Average memory usage"
    target: "<80%"
    
  - name: "database_connections"
    description: "Active database connections"
    target: "<80% of pool"
```

#### Analysis Scripts
```python
# performance-analysis.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

class PerformanceAnalyzer:
    """Analyze load test results and generate reports"""
    
    def __init__(self, results_file: str):
        with open(results_file, 'r') as f:
            self.results = json.load(f)
    
    def analyze_response_times(self) -> Dict[str, float]:
        """Analyze response time metrics"""
        response_times = self.results.get('metrics', {}).get('http_req_duration', {})
        
        analysis = {
            'avg_response_time': response_times.get('avg', 0),
            'p50_response_time': response_times.get('p(50)', 0),
            'p95_response_time': response_times.get('p(95)', 0),
            'p99_response_time': response_times.get('p(99)', 0),
            'max_response_time': response_times.get('max', 0),
        }
        
        return analysis
    
    def analyze_throughput(self) -> Dict[str, float]:
        """Analyze throughput metrics"""
        iterations = self.results.get('metrics', {}).get('iterations', {})
        http_reqs = self.results.get('metrics', {}).get('http_reqs', {})
        
        duration = self.results.get('state', {}).get('testRunDurationMs', 0) / 1000
        
        analysis = {
            'total_requests': http_reqs.get('count', 0),
            'requests_per_second': http_reqs.get('rate', 0),
            'iterations_per_second': iterations.get('rate', 0),
            'test_duration': duration,
        }
        
        return analysis
    
    def analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns"""
        failed_requests = self.results.get('metrics', {}).get('http_req_failed', {})
        
        analysis = {
            'total_errors': failed_requests.get('fails', 0),
            'error_rate': failed_requests.get('rate', 0),
            'error_percentage': (failed_requests.get('fails', 0) / 
                               self.results.get('metrics', {}).get('http_reqs', {}).get('count', 1)) * 100,
        }
        
        return analysis
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'test_summary': {
                'test_type': self.results.get('options', {}).get('scenarios', {}),
                'test_duration': self.results.get('state', {}).get('testRunDurationMs', 0) / 1000,
                'virtual_users': self.results.get('options', {}).get('vus', 0),
            },
            'response_times': self.analyze_response_times(),
            'throughput': self.analyze_throughput(),
            'errors': self.analyze_errors(),
            'recommendations': self._generate_recommendations(),
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        response_times = self.analyze_response_times()
        errors = self.analyze_errors()
        
        # Response time recommendations
        if response_times['p95_response_time'] > 500:
            recommendations.append("95th percentile response time exceeds 500ms. Consider optimizing slow endpoints.")
        
        if response_times['p99_response_time'] > 1000:
            recommendations.append("99th percentile response time exceeds 1s. Investigate performance bottlenecks.")
        
        # Error rate recommendations
        if errors['error_percentage'] > 1:
            recommendations.append("Error rate exceeds 1%. Check application logs and fix failing requests.")
        
        if errors['error_percentage'] > 5:
            recommendations.append("High error rate detected. System may be overloaded or have critical issues.")
        
        return recommendations
    
    def create_visualizations(self, output_dir: str = './reports'):
        """Create performance visualization charts"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Response time distribution
        self._plot_response_time_distribution(output_dir)
        
        # Throughput over time
        self._plot_throughput_timeline(output_dir)
        
        # Error rate analysis
        self._plot_error_analysis(output_dir)
    
    def _plot_response_time_distribution(self, output_dir: str):
        """Plot response time distribution"""
        # This would require more detailed time-series data
        # For now, create a simple bar chart of percentiles
        
        response_times = self.analyze_response_times()
        
        percentiles = ['p50', 'p95', 'p99', 'max']
        values = [
            response_times['p50_response_time'],
            response_times['p95_response_time'],
            response_times['p99_response_time'],
            response_times['max_response_time']
        ]
        
        plt.figure(figsize=(10, 6))
        plt.bar(percentiles, values)
        plt.title('Response Time Distribution')
        plt.ylabel('Response Time (ms)')
        plt.xlabel('Percentiles')
        plt.savefig(f'{output_dir}/response_time_distribution.png')
        plt.close()
    
    def _plot_throughput_timeline(self, output_dir: str):
        """Plot request throughput over time"""
        # Placeholder for throughput timeline
        # Real implementation would require time-series data
        pass
    
    def _plot_error_analysis(self, output_dir: str):
        """Plot error analysis"""
        errors = self.analyze_errors()
        
        labels = ['Successful', 'Failed']
        sizes = [
            100 - errors['error_percentage'],
            errors['error_percentage']
        ]
        colors = ['green', 'red']
        
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title('Request Success/Failure Rate')
        plt.savefig(f'{output_dir}/error_analysis.png')
        plt.close()

# Usage
if __name__ == "__main__":
    analyzer = PerformanceAnalyzer('load-test-results.json')
    report = analyzer.generate_report()
    
    # Print report
    print(json.dumps(report, indent=2))
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Save report
    with open('performance-report.json', 'w') as f:
        json.dump(report, f, indent=2)
```

## Test Execution Workflow

### Automated Test Pipeline

```yaml
# .github/workflows/load-testing.yml
name: Load Testing Pipeline

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday 2 AM
  workflow_dispatch:
    inputs:
      test_type:
        description: 'Type of load test to run'
        required: true
        default: 'load'
        type: choice
        options:
          - smoke
          - load
          - stress
          - spike
          - endurance

jobs:
  setup-environment:
    runs-on: ubuntu-latest
    steps:
      - name: Setup test environment
        run: |
          # Deploy test environment
          kubectl apply -f k8s/test-environment/
          
          # Wait for environment to be ready
          kubectl wait --for=condition=ready pod -l app=agi-eval-test --timeout=300s
  
  run-load-tests:
    needs: setup-environment
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          
      - name: Install K6
        run: |
          sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
          echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
          sudo apt-get update
          sudo apt-get install k6
          
      - name: Run load tests
        run: |
          k6 run --out prometheus=http://prometheus:9090 \
                 --env BASE_URL=${{ secrets.TEST_BASE_URL }} \
                 --env TEST_TYPE=${{ github.event.inputs.test_type || 'load' }} \
                 load-tests/k6-load-test.js
                 
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: load-test-results
          path: load-test-results.json
  
  analyze-results:
    needs: run-load-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install pandas matplotlib seaborn
          
      - name: Download test results
        uses: actions/download-artifact@v3
        with:
          name: load-test-results
          
      - name: Analyze results
        run: |
          python scripts/performance-analysis.py
          
      - name: Upload analysis
        uses: actions/upload-artifact@v3
        with:
          name: performance-analysis
          path: |
            performance-report.json
            reports/
  
  cleanup:
    needs: [run-load-tests, analyze-results]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - name: Cleanup test environment
        run: |
          kubectl delete -f k8s/test-environment/
```

## Best Practices

### Test Design
1. **Start Small**: Begin with smoke tests before running full load tests
2. **Realistic Scenarios**: Use realistic user behavior patterns
3. **Gradual Increase**: Ramp up load gradually to identify breaking points
4. **Consistent Environment**: Use dedicated test environments
5. **Monitor Resources**: Track system resources during tests

### Data Management
1. **Test Data Isolation**: Use separate test data that doesn't affect production
2. **Data Cleanup**: Clean up test data after each run
3. **Data Variation**: Use varied test data to simulate real usage
4. **Data Volume**: Test with appropriate data volumes

### Result Analysis
1. **Baseline comparison**: Compare results against previous baselines
2. **Trend Analysis**: Track performance trends over time  
3. **Correlation Analysis**: Correlate performance with system changes
4. **Action Items**: Generate actionable recommendations

## Troubleshooting Load Tests

### Common Issues

#### High Response Times
```bash
# Diagnosis steps
kubectl top pods
kubectl describe pod <pod-name>
kubectl logs <pod-name>

# Check database performance
kubectl exec -it <db-pod> -- psql -c "SELECT * FROM pg_stat_activity;"

# Check cache performance  
kubectl exec -it <redis-pod> -- redis-cli info stats
```

#### High Error Rates
```bash
# Check application logs
kubectl logs -l app=agi-eval-sandbox --tail=1000

# Check ingress logs
kubectl logs -n ingress-nginx ingress-nginx-controller-xxx

# Check database connections
kubectl exec -it <db-pod> -- psql -c "SELECT count(*) FROM pg_stat_activity;"
```

#### Resource Exhaustion
```bash
# Check resource utilization
kubectl top nodes
kubectl top pods

# Check horizontal pod autoscaler
kubectl get hpa
kubectl describe hpa agi-eval-hpa

# Check resource quotas
kubectl describe resourcequota
```

## Related Documentation

- [Performance Optimization](../operational/PERFORMANCE_OPTIMIZATION.md)
- [Monitoring and Observability](../operational/MONITORING.md) 
- [Infrastructure Requirements](../operational/INFRASTRUCTURE.md)
- [Troubleshooting Guide](../operational/TROUBLESHOOTING.md)

---

**Note**: Load testing should be performed regularly and after significant changes to ensure optimal system performance and reliability.