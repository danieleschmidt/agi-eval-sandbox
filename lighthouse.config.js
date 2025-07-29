// Lighthouse CI Configuration for AGI Evaluation Sandbox
// Performance, accessibility, and best practices auditing
// See: https://github.com/GoogleChrome/lighthouse-ci

module.exports = {
  ci: {
    // ==================================================
    // Collection Configuration
    // ==================================================
    collect: {
      // URLs to audit
      url: [
        'http://localhost:8080',           // Main dashboard
        'http://localhost:8080/models',    // Models page
        'http://localhost:8080/benchmarks', // Benchmarks page
        'http://localhost:8080/results',   // Results page
        'http://localhost:8080/compare',   // Comparison page
        'http://localhost:8080/settings',  // Settings page
        'http://localhost:8080/docs',      // Documentation
      ],
      
      // Collection settings
      numberOfRuns: 3,           // Run lighthouse 3 times per URL
      startServerCommand: 'npm run dev', // Command to start the app
      startServerReadyPattern: 'Local.*:.*8080', // Pattern to detect server ready
      startServerReadyTimeout: 30000, // 30 second timeout
      
      // Chrome settings
      settings: {
        // Chrome flags for CI environment
        chromeFlags: [
          '--headless',
          '--no-sandbox',
          '--disable-dev-shm-usage',
          '--disable-background-timer-throttling',
          '--disable-backgrounding-occluded-windows',
          '--disable-renderer-backgrounding',
          '--disable-features=TranslateUI',
          '--disable-ipc-flooding-protection',
          '--window-size=1200,800',
        ],
        
        // Lighthouse configuration
        preset: 'desktop',         // Use desktop preset
        onlyCategories: [          // Only run these categories
          'performance',
          'accessibility',
          'best-practices',
          'seo',
          'pwa',
        ],
        
        // Skip certain audits that may not be relevant
        skipAudits: [
          'canonical',           // May not have canonical URLs in dev
          'robots-txt',          // May not have robots.txt in dev
          'tap-targets',         // Mobile-specific
          'themed-omnibox',      // PWA-specific
        ],
        
        // Custom audit settings
        throttling: {
          rttMs: 40,               // Round trip time
          throughputKbps: 10240,   // 10 Mbps
          cpuSlowdownMultiplier: 1, // No CPU throttling for CI
          requestLatencyMs: 0,
          downloadThroughputKbps: 0,
          uploadThroughputKbps: 0,
        },
        
        // Device emulation
        emulatedFormFactor: 'desktop',
        
        // Locale
        locale: 'en-US',
        
        // Output settings
        output: ['html', 'json'],
      },
    },
    
    // ==================================================
    // Assertion Configuration
    // ==================================================
    assert: {
      // Performance budgets
      assertions: {
        // Core Web Vitals
        'categories:performance': ['error', { minScore: 0.8 }],
        'categories:accessibility': ['error', { minScore: 0.9 }],
        'categories:best-practices': ['error', { minScore: 0.9 }],
        'categories:seo': ['error', { minScore: 0.8 }],
        'categories:pwa': ['warn', { minScore: 0.6 }],
        
        // Specific metrics
        'first-contentful-paint': ['warn', { maxNumericValue: 2000 }],
        'largest-contentful-paint': ['error', { maxNumericValue: 2500 }],
        'cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],
        'total-blocking-time': ['warn', { maxNumericValue: 300 }],
        'speed-index': ['warn', { maxNumericValue: 3000 }],
        
        // Resource loading
        'uses-optimized-images': 'off',      // May have placeholder images
        'uses-webp-images': 'warn',
        'uses-responsive-images': 'warn',
        'efficient-animated-content': 'warn',
        'unused-css-rules': 'warn',
        'unused-javascript': 'warn',
        'modern-image-formats': 'warn',
        
        // Network
        'uses-http2': 'warn',
        'uses-long-cache-ttl': 'warn',
        'uses-text-compression': 'error',
        
        // JavaScript
        'no-unload-listeners': 'error',
        'no-document-write': 'error',
        'uses-passive-event-listeners': 'warn',
        
        // Security
        'is-on-https': 'off',               // Dev environment may use HTTP
        'uses-https': 'off',                // Dev environment may use HTTP
        'csp-xss': 'warn',
        
        // Accessibility
        'color-contrast': 'error',
        'image-alt': 'error',
        'label': 'error',
        'link-name': 'error',
        'list': 'error',
        'meta-viewport': 'error',
        'heading-order': 'warn',
        'landmark-one-main': 'warn',
        'page-has-heading-one': 'warn',
        'skip-link': 'warn',
        
        // Best practices
        'errors-in-console': 'warn',
        'image-aspect-ratio': 'warn',
        'js-libraries': 'warn',
        'no-vulnerable-libraries': 'error',
        'notification-on-start': 'error',
        'password-inputs-can-be-pasted-into': 'error',
        
        // SEO
        'document-title': 'error',
        'meta-description': 'error',
        'http-status-code': 'error',
        'crawlable-anchors': 'warn',
        'font-size': 'error',
        'tap-targets': 'off',               // Mobile-specific
        
        // PWA (less strict for dashboard app)
        'installable-manifest': 'warn',
        'splash-screen': 'warn',
        'themed-omnibox': 'warn',
        'content-width': 'warn',
        'viewport': 'error',
        'without-javascript': 'warn',
        'apple-touch-icon': 'warn',
        'maskable-icon': 'warn',
      },
      
      // Budget assertions for resource sizes
      budget: [
        {
          path: '/*',
          resourceSizes: [
            { resourceType: 'total', budget: 500 },      // 500 KB total
            { resourceType: 'script', budget: 200 },     // 200 KB JavaScript
            { resourceType: 'stylesheet', budget: 50 },  // 50 KB CSS
            { resourceType: 'image', budget: 150 },      // 150 KB images
            { resourceType: 'font', budget: 100 },       // 100 KB fonts
          ],
          resourceCounts: [
            { resourceType: 'total', budget: 100 },      // Max 100 resources
            { resourceType: 'script', budget: 20 },      // Max 20 JS files
            { resourceType: 'stylesheet', budget: 10 },  // Max 10 CSS files
            { resourceType: 'image', budget: 30 },       // Max 30 images
          ],
          timings: [
            { metric: 'first-contentful-paint', budget: 2000 },
            { metric: 'largest-contentful-paint', budget: 2500 },
            { metric: 'cumulative-layout-shift', budget: 0.1 },
            { metric: 'total-blocking-time', budget: 300 },
          ],
        },
      ],
    },
    
    // ==================================================
    // Upload Configuration
    // ==================================================
    upload: {
      target: 'temporary-public-storage', // Use Lighthouse CI temporary storage
      // For permanent storage, use:
      // target: 'lhci',
      // serverBaseUrl: 'https://your-lhci-server.example.com',
      // token: 'your-lhci-token',
    },
    
    // ==================================================
    # Server Configuration (if running LHCI server)
    // ==================================================
    server: {
      port: 9001,
      storage: {
        storageMethod: 'sql',
        sqlDialect: 'postgres',
        sqlConnectionUrl: process.env.DATABASE_URL,
      },
    },
    
    // ==================================================
    // Wizard Configuration (for setup)
    // ==================================================
    wizard: {
      // Automatically configure common settings
      preset: 'desktop',
    },
  },
  
  // ==================================================
  // Custom Configuration Extensions
  // ==================================================
  extends: 'lighthouse:default',
  
  settings: {
    // Additional custom settings
    maxWaitForFcp: 30000,        // 30 second timeout for FCP
    maxWaitForLoad: 45000,       // 45 second timeout for load
    
    // Custom gatherers (if needed)
    passes: [{
      passName: 'defaultPass',
      gatherers: [
        'css-usage',
        'js-usage',
        'viewport-dimensions',
        'runtime-exceptions',
        'trace-elements',
        'inspector-issues',
        'source-maps',
        'full-page-screenshot',
      ],
    }],
    
    // Custom audits (if any)
    audits: [
      // Add custom audit paths here if needed
      // 'path/to/custom-audit.js',
    ],
    
    // Categories configuration
    categories: {
      performance: {
        title: 'Performance',
        auditRefs: [
          // Custom weightings for audits
          { id: 'first-contentful-paint', weight: 10 },
          { id: 'largest-contentful-paint', weight: 25 },
          { id: 'cumulative-layout-shift', weight: 25 },
          { id: 'total-blocking-time', weight: 30 },
          { id: 'speed-index', weight: 10 },
          // ... other performance audits with default weights
        ],
      },
      
      // Custom category for dashboard-specific checks
      'dashboard-usability': {
        title: 'Dashboard Usability',
        description: 'Specific checks for dashboard usability',
        auditRefs: [
          { id: 'color-contrast', weight: 20 },
          { id: 'tap-targets', weight: 10 },
          { id: 'interactive', weight: 30 },
          { id: 'uses-responsive-images', weight: 10 },
          { id: 'font-display', weight: 10 },
          { id: 'link-name', weight: 20 },
        ],
      },
    },
  },
  
  // ==================================================
  // Environment-specific Overrides
  // ==================================================
  ...(
    process.env.CI ? {
      // CI-specific overrides
      ci: {
        collect: {
          numberOfRuns: 1,  // Single run in CI for speed
          settings: {
            chromeFlags: [
              '--headless',
              '--no-sandbox',
              '--disable-dev-shm-usage',
              '--disable-background-timer-throttling',
              '--disable-backgrounding-occluded-windows',
              '--disable-renderer-backgrounding',
            ],
          },
        },
        assert: {
          assertions: {
            // Slightly more lenient assertions for CI
            'categories:performance': ['warn', { minScore: 0.7 }],
            'categories:accessibility': ['error', { minScore: 0.85 }],
          },
        },
      },
    } : {}
  ),
};

// ==================================================
// Helper Functions
// ==================================================

// Function to get URLs based on environment
function getUrlsForEnvironment() {
  const baseUrl = process.env.LIGHTHOUSE_BASE_URL || 'http://localhost:8080';
  
  return [
    `${baseUrl}`,
    `${baseUrl}/models`,
    `${baseUrl}/benchmarks`,
    `${baseUrl}/results`,
    `${baseUrl}/compare`,
    `${baseUrl}/settings`,
  ];
}

// Function to get Chrome flags based on environment
function getChromeFlagsForEnvironment() {
  const baseFlags = [
    '--headless',
    '--no-sandbox',
    '--disable-dev-shm-usage',
  ];
  
  if (process.env.CI) {
    return [
      ...baseFlags,
      '--disable-background-timer-throttling',
      '--disable-backgrounding-occluded-windows',
      '--disable-renderer-backgrounding',
      '--disable-features=TranslateUI',
      '--disable-ipc-flooding-protection',
      '--virtual-time-budget=5000',
    ];
  }
  
  return baseFlags;
}