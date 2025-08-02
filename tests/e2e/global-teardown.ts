import { chromium, FullConfig } from '@playwright/test';

/**
 * Global teardown for Playwright tests
 * Runs once after all tests are complete
 */
async function globalTeardown(config: FullConfig) {
  console.log('🧹 Starting global teardown...');
  
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  try {
    // Clean up test data
    console.log('🗑️ Cleaning up test data...');
    await cleanupTestData(page);
    
    // Reset database to clean state (if applicable)
    if (process.env.RESET_DB_AFTER_TESTS === 'true') {
      console.log('🔄 Resetting database...');
      await resetDatabase(page);
    }
    
    // Generate test reports
    console.log('📊 Generating test reports...');
    await generateTestReports();
    
    console.log('✅ Global teardown completed successfully');
    
  } catch (error) {
    console.error('❌ Global teardown failed:', error);
    // Don't throw error in teardown to avoid masking test failures
    console.warn('⚠️ Continuing despite teardown errors');
  } finally {
    await browser.close();
  }
}

/**
 * Clean up test data created during e2e tests
 */
async function cleanupTestData(page: any) {
  // Remove test models
  const modelsResponse = await page.request.get('http://localhost:8000/api/v1/models?test=true');
  if (modelsResponse.ok()) {
    const models = await modelsResponse.json();
    for (const model of models) {
      if (model.name.startsWith('test-')) {
        await page.request.delete(`http://localhost:8000/api/v1/models/${model.id}`);
      }
    }
  }
  
  // Remove test benchmarks
  const benchmarksResponse = await page.request.get('http://localhost:8000/api/v1/benchmarks?test=true');
  if (benchmarksResponse.ok()) {
    const benchmarks = await benchmarksResponse.json();
    for (const benchmark of benchmarks) {
      if (benchmark.name.startsWith('test-')) {
        await page.request.delete(`http://localhost:8000/api/v1/benchmarks/${benchmark.id}`);
      }
    }
  }
  
  // Remove test evaluations
  const evaluationsResponse = await page.request.get('http://localhost:8000/api/v1/evaluations?test=true');
  if (evaluationsResponse.ok()) {
    const evaluations = await evaluationsResponse.json();
    for (const evaluation of evaluations) {
      await page.request.delete(`http://localhost:8000/api/v1/evaluations/${evaluation.id}`);
    }
  }
  
  console.log('🗑️ Test data cleanup completed');
}

/**
 * Reset database to clean state
 */
async function resetDatabase(page: any) {
  try {
    await page.request.post('http://localhost:8000/api/v1/admin/reset-database', {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.ADMIN_TOKEN}`
      }
    });
    console.log('🔄 Database reset completed');
  } catch (error) {
    console.warn('⚠️ Database reset failed (this may be expected):', error);
  }
}

/**
 * Generate test reports
 */
async function generateTestReports() {
  // This could include:
  // - Copying test artifacts to specific locations
  // - Uploading test results to external systems
  // - Generating summary reports
  // - Notifying stakeholders of test results
  
  console.log('📊 Test reports generation completed');
}

export default globalTeardown;