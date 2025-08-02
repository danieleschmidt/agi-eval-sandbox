import { chromium, FullConfig } from '@playwright/test';

/**
 * Global setup for Playwright tests
 * Runs once before all tests
 */
async function globalSetup(config: FullConfig) {
  console.log('🚀 Starting global setup...');
  
  // Launch browser for setup
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  try {
    // Wait for API to be ready
    console.log('⏳ Waiting for API server...');
    await page.waitForResponse(
      response => response.url().includes('/health') && response.status() === 200,
      { timeout: 60000 }
    );
    
    // Wait for dashboard to be ready
    console.log('⏳ Waiting for dashboard...');
    await page.goto('http://localhost:8080');
    await page.waitForLoadState('networkidle');
    
    // Setup test data
    console.log('📊 Setting up test data...');
    await setupTestData(page);
    
    console.log('✅ Global setup completed successfully');
    
  } catch (error) {
    console.error('❌ Global setup failed:', error);
    throw error;
  } finally {
    await browser.close();
  }
}

/**
 * Setup test data for e2e tests
 */
async function setupTestData(page: any) {
  // Create test models
  const testModels = [
    {
      name: 'test-gpt-3.5',
      provider: 'openai',
      version: 'gpt-3.5-turbo-0613'
    },
    {
      name: 'test-claude-3',
      provider: 'anthropic',
      version: 'claude-3-sonnet-20240229'
    }
  ];
  
  // Create test benchmarks
  const testBenchmarks = [
    {
      name: 'test-mmlu',
      type: 'multiple_choice',
      description: 'Test MMLU benchmark for e2e testing'
    },
    {
      name: 'test-humaneval',
      type: 'code_generation',
      description: 'Test HumanEval benchmark for e2e testing'
    }
  ];
  
  // API calls to setup test data
  for (const model of testModels) {
    await page.request.post('http://localhost:8000/api/v1/models', {
      data: model,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }
  
  for (const benchmark of testBenchmarks) {
    await page.request.post('http://localhost:8000/api/v1/benchmarks', {
      data: benchmark,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }
  
  console.log('📊 Test data setup completed');
}

export default globalSetup;