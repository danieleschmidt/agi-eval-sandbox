/**
 * End-to-end tests for AGI Evaluation Sandbox
 * 
 * These tests verify the complete user workflows from the frontend
 * through to the backend services.
 */

import { test, expect, Page } from '@playwright/test';

// Test data
const TEST_USER = {
  email: 'test@example.com',
  password: 'test_password_123',
  username: 'testuser'
};

const SAMPLE_EVALUATION = {
  model: 'gpt-4',
  benchmark: 'mmlu',
  config: {
    temperature: 0.0,
    maxTokens: 1000
  }
};

test.describe('Authentication Flow', () => {
  test('should allow user to login and logout', async ({ page }) => {
    // Navigate to login page
    await page.goto('/login');
    
    // Fill login form
    await page.fill('[data-testid="email-input"]', TEST_USER.email);
    await page.fill('[data-testid="password-input"]', TEST_USER.password);
    
    // Submit login
    await page.click('[data-testid="login-button"]');
    
    // Verify successful login
    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
    
    // Logout
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="logout-button"]');
    
    // Verify logout
    await expect(page).toHaveURL('/');
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/login');
    
    // Try with invalid credentials
    await page.fill('[data-testid="email-input"]', 'invalid@example.com');
    await page.fill('[data-testid="password-input"]', 'wrongpassword');
    await page.click('[data-testid="login-button"]');
    
    // Verify error message
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Invalid credentials');
  });
});

test.describe('Dashboard Navigation', () => {
  test.beforeEach(async ({ page }) => {
    // Login before each test
    await loginUser(page, TEST_USER);
  });

  test('should navigate between main sections', async ({ page }) => {
    // Start at dashboard
    await expect(page).toHaveURL('/dashboard');
    
    // Navigate to evaluations
    await page.click('[data-testid="nav-evaluations"]');
    await expect(page).toHaveURL('/evaluations');
    await expect(page.locator('h1')).toContainText('Evaluations');
    
    // Navigate to benchmarks
    await page.click('[data-testid="nav-benchmarks"]');
    await expect(page).toHaveURL('/benchmarks');
    await expect(page.locator('h1')).toContainText('Benchmarks');
    
    // Navigate to models
    await page.click('[data-testid="nav-models"]');
    await expect(page).toHaveURL('/models');
    await expect(page.locator('h1')).toContainText('Models');
  });

  test('should display user information in header', async ({ page }) => {
    await expect(page.locator('[data-testid="user-info"]')).toBeVisible();
    await expect(page.locator('[data-testid="user-email"]')).toContainText(TEST_USER.email);
  });
});

test.describe('Evaluation Workflow', () => {
  test.beforeEach(async ({ page }) => {
    await loginUser(page, TEST_USER);
  });

  test('should create new evaluation', async ({ page }) => {
    // Navigate to evaluations
    await page.goto('/evaluations');
    
    // Click create new evaluation
    await page.click('[data-testid="create-evaluation-button"]');
    await expect(page).toHaveURL('/evaluations/new');
    
    // Fill evaluation form
    await page.selectOption('[data-testid="model-select"]', SAMPLE_EVALUATION.model);
    await page.check(`[data-testid="benchmark-${SAMPLE_EVALUATION.benchmark}"]`);
    
    // Advanced configuration
    await page.click('[data-testid="advanced-config-toggle"]');
    await page.fill('[data-testid="temperature-input"]', SAMPLE_EVALUATION.config.temperature.toString());
    await page.fill('[data-testid="max-tokens-input"]', SAMPLE_EVALUATION.config.maxTokens.toString());
    
    // Submit evaluation
    await page.click('[data-testid="submit-evaluation-button"]');
    
    // Verify evaluation was created
    await expect(page).toHaveURL(/\/evaluations\/[a-f0-9-]+/);
    await expect(page.locator('[data-testid="evaluation-status"]')).toContainText('pending');
  });

  test('should display evaluation progress', async ({ page }) => {
    // Create evaluation first (mock or use existing)
    const evaluationId = 'test-eval-123';
    await page.goto(`/evaluations/${evaluationId}`);
    
    // Check progress indicators
    await expect(page.locator('[data-testid="progress-bar"]')).toBeVisible();
    await expect(page.locator('[data-testid="progress-percentage"]')).toBeVisible();
    await expect(page.locator('[data-testid="estimated-time"]')).toBeVisible();
    
    // Check real-time updates (mock WebSocket connection)
    // In a real test, you'd mock the WebSocket to send progress updates
  });

  test('should show evaluation results', async ({ page }) => {
    // Navigate to completed evaluation (mock data)
    const evaluationId = 'test-eval-completed';
    await page.goto(`/evaluations/${evaluationId}/results`);
    
    // Verify results display
    await expect(page.locator('[data-testid="results-summary"]')).toBeVisible();
    await expect(page.locator('[data-testid="accuracy-metric"]')).toBeVisible();
    await expect(page.locator('[data-testid="results-chart"]')).toBeVisible();
    
    // Test result filtering
    await page.selectOption('[data-testid="subject-filter"]', 'physics');
    await expect(page.locator('[data-testid="filtered-results"]')).toBeVisible();
  });

  test('should export evaluation results', async ({ page }) => {
    const evaluationId = 'test-eval-completed';
    await page.goto(`/evaluations/${evaluationId}/results`);
    
    // Test different export formats
    const [downloadPromise] = await Promise.all([
      page.waitForEvent('download'),
      page.click('[data-testid="export-json-button"]')
    ]);
    
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/results.*\.json$/);
  });
});

test.describe('Benchmark Management', () => {
  test.beforeEach(async ({ page }) => {
    await loginUser(page, TEST_USER);
  });

  test('should list available benchmarks', async ({ page }) => {
    await page.goto('/benchmarks');
    
    // Verify benchmark list
    await expect(page.locator('[data-testid="benchmark-list"]')).toBeVisible();
    await expect(page.locator('[data-testid="benchmark-mmlu"]')).toBeVisible();
    await expect(page.locator('[data-testid="benchmark-truthfulqa"]')).toBeVisible();
    
    // Test benchmark search
    await page.fill('[data-testid="benchmark-search"]', 'mmlu');
    await expect(page.locator('[data-testid="benchmark-mmlu"]')).toBeVisible();
    await expect(page.locator('[data-testid="benchmark-truthfulqa"]')).toBeHidden();
  });

  test('should show benchmark details', async ({ page }) => {
    await page.goto('/benchmarks');
    
    // Click on benchmark
    await page.click('[data-testid="benchmark-mmlu"]');
    await expect(page).toHaveURL('/benchmarks/mmlu');
    
    // Verify benchmark details
    await expect(page.locator('[data-testid="benchmark-name"]')).toContainText('MMLU');
    await expect(page.locator('[data-testid="benchmark-description"]')).toBeVisible();
    await expect(page.locator('[data-testid="question-count"]')).toBeVisible();
    await expect(page.locator('[data-testid="subjects-list"]')).toBeVisible();
  });
});

test.describe('Model Comparison', () => {
  test.beforeEach(async ({ page }) => {
    await loginUser(page, TEST_USER);
  });

  test('should compare multiple models', async ({ page }) => {
    await page.goto('/compare');
    
    // Select models to compare
    await page.selectOption('[data-testid="model-1-select"]', 'gpt-4');
    await page.selectOption('[data-testid="model-2-select"]', 'claude-3');
    
    // Select benchmark
    await page.check('[data-testid="benchmark-mmlu-checkbox"]');
    
    // Start comparison
    await page.click('[data-testid="start-comparison-button"]');
    
    // Verify comparison view
    await expect(page.locator('[data-testid="comparison-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="model-1-results"]')).toBeVisible();
    await expect(page.locator('[data-testid="model-2-results"]')).toBeVisible();
  });

  test('should show statistical significance', async ({ page }) => {
    // Navigate to existing comparison (mock data)
    await page.goto('/compare/test-comparison-123');
    
    // Verify statistical analysis
    await expect(page.locator('[data-testid="significance-test"]')).toBeVisible();
    await expect(page.locator('[data-testid="p-value"]')).toBeVisible();
    await expect(page.locator('[data-testid="confidence-interval"]')).toBeVisible();
  });
});

test.describe('Real-time Features', () => {
  test.beforeEach(async ({ page }) => {
    await loginUser(page, TEST_USER);
  });

  test('should show real-time evaluation updates', async ({ page }) => {
    const evaluationId = 'test-eval-running';
    await page.goto(`/evaluations/${evaluationId}`);
    
    // Mock WebSocket connection for real-time updates
    await page.evaluate(() => {
      // Simulate WebSocket message
      window.dispatchEvent(new CustomEvent('evaluation-update', {
        detail: {
          evaluationId: 'test-eval-running',
          progress: 0.75,
          status: 'running'
        }
      }));
    });
    
    // Verify real-time update
    await expect(page.locator('[data-testid="progress-bar"]')).toHaveAttribute('aria-valuenow', '75');
  });

  test('should show live notifications', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Mock notification
    await page.evaluate(() => {
      window.dispatchEvent(new CustomEvent('notification', {
        detail: {
          type: 'success',
          message: 'Evaluation completed successfully!'
        }
      }));
    });
    
    // Verify notification display
    await expect(page.locator('[data-testid="notification"]')).toBeVisible();
    await expect(page.locator('[data-testid="notification"]')).toContainText('Evaluation completed');
  });
});

test.describe('Responsive Design', () => {
  test('should work on mobile devices', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    await loginUser(page, TEST_USER);
    
    // Test mobile navigation
    await expect(page.locator('[data-testid="mobile-menu-button"]')).toBeVisible();
    await page.click('[data-testid="mobile-menu-button"]');
    await expect(page.locator('[data-testid="mobile-nav-menu"]')).toBeVisible();
    
    // Test responsive cards
    await page.goto('/evaluations');
    await expect(page.locator('[data-testid="evaluation-card"]')).toBeVisible();
  });

  test('should adapt to tablet viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    
    await loginUser(page, TEST_USER);
    
    // Test tablet layout
    await page.goto('/dashboard');
    await expect(page.locator('[data-testid="sidebar"]')).toBeVisible();
    await expect(page.locator('[data-testid="main-content"]')).toBeVisible();
  });
});

test.describe('Accessibility', () => {
  test('should be keyboard navigable', async ({ page }) => {
    await page.goto('/');
    
    // Test keyboard navigation
    await page.keyboard.press('Tab');
    await expect(page.locator(':focus')).toBeVisible();
    
    // Navigate through main elements
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('Tab');
    }
    
    // Test Enter key activation
    await page.keyboard.press('Enter');
  });

  test('should have proper ARIA labels', async ({ page }) => {
    await loginUser(page, TEST_USER);
    await page.goto('/evaluations');
    
    // Check ARIA labels
    await expect(page.locator('[data-testid="create-evaluation-button"]')).toHaveAttribute('aria-label');
    await expect(page.locator('[data-testid="evaluation-list"]')).toHaveAttribute('role', 'list');
    await expect(page.locator('[data-testid="evaluation-item"]').first()).toHaveAttribute('role', 'listitem');
  });
});

test.describe('Performance', () => {
  test('should load pages quickly', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    expect(loadTime).toBeLessThan(3000); // 3 seconds
  });

  test('should handle large data sets', async ({ page }) => {
    await loginUser(page, TEST_USER);
    
    // Navigate to page with large dataset (mock 1000 evaluations)
    await page.goto('/evaluations?limit=1000');
    
    // Verify virtualization or pagination
    await expect(page.locator('[data-testid="evaluation-list"]')).toBeVisible();
    
    // Test scrolling performance
    await page.mouse.wheel(0, 1000);
    await page.waitForTimeout(100);
    await expect(page.locator('[data-testid="evaluation-list"]')).toBeVisible();
  });
});

// Helper functions
async function loginUser(page: Page, user: typeof TEST_USER) {
  await page.goto('/login');
  await page.fill('[data-testid="email-input"]', user.email);
  await page.fill('[data-testid="password-input"]', user.password);
  await page.click('[data-testid="login-button"]');
  await page.waitForURL('/dashboard');
}

// Custom test fixtures for API mocking
test.describe('API Error Handling', () => {
  test('should handle API timeouts gracefully', async ({ page }) => {
    // Mock API timeout
    await page.route('/api/v1/evaluations', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 30000)); // 30 second delay
    });
    
    await loginUser(page, TEST_USER);
    await page.goto('/evaluations');
    
    // Verify timeout handling
    await expect(page.locator('[data-testid="timeout-message"]')).toBeVisible({ timeout: 35000 });
  });

  test('should handle server errors gracefully', async ({ page }) => {
    // Mock server error
    await page.route('/api/v1/evaluations', async (route) => {
      await route.fulfill({ status: 500, body: 'Internal Server Error' });
    });
    
    await loginUser(page, TEST_USER);
    await page.goto('/evaluations');
    
    // Verify error handling
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="retry-button"]')).toBeVisible();
  });
});