// Commitlint configuration for AGI Evaluation Sandbox
// Extends conventional commits specification for better changelog generation
// See: https://commitlint.js.org/

module.exports = {
  extends: ['@commitlint/config-conventional'],
  
  // Custom rules for this project
  rules: {
    // Header
    'header-max-length': [2, 'always', 100],
    'header-min-length': [2, 'always', 10],
    
    // Type
    'type-enum': [
      2,
      'always',
      [
        // Standard types
        'feat',     // New features
        'fix',      // Bug fixes
        'docs',     // Documentation changes
        'style',    // Code style changes (formatting, etc.)
        'refactor', // Code refactoring
        'perf',     // Performance improvements
        'test',     // Test additions or modifications
        'chore',    // Maintenance tasks
        'ci',       // CI/CD changes
        'build',    // Build system changes
        'revert',   // Reverts previous commits
        
        // Project-specific types
        'eval',     // Evaluation benchmark changes
        'model',    // Model provider integrations
        'security', // Security improvements
        'monitoring', // Monitoring and observability
        'config',   // Configuration changes
        'deps',     // Dependency updates
        'release',  // Release-related changes
      ],
    ],
    'type-case': [2, 'always', 'lower-case'],
    'type-empty': [2, 'never'],
    
    // Scope
    'scope-enum': [
      1,
      'always',
      [
        // Core components
        'api',
        'worker',
        'dashboard',
        'cli',
        'database',
        'cache',
        
        // Evaluation components
        'benchmark',
        'evaluator',
        'metrics',
        'results',
        'models',
        
        // Infrastructure
        'docker',
        'k8s',
        'monitoring',
        'logging',
        'security',
        'backup',
        
        // Development
        'tests',
        'docs',
        'scripts',
        'tools',
        'deps',
        
        // Integration
        'github',
        'slack',
        'wandb',
        'aws',
        'gcp',
        'azure',
        
        // Providers
        'openai',
        'anthropic',
        'google',
        'huggingface',
        
        // Benchmarks
        'mmlu',
        'humaneval',
        'truthfulqa',
        'mtbench',
        'hellaswag',
        'math',
        'gsm8k',
      ],
    ],
    'scope-case': [2, 'always', 'lower-case'],
    
    // Subject
    'subject-case': [2, 'always', 'lower-case'],
    'subject-empty': [2, 'never'],
    'subject-full-stop': [2, 'never', '.'],
    'subject-max-length': [2, 'always', 72],
    'subject-min-length': [2, 'always', 5],
    
    // Body
    'body-leading-blank': [2, 'always'],
    'body-max-line-length': [2, 'always', 100],
    
    // Footer
    'footer-leading-blank': [2, 'always'],
    'footer-max-line-length': [2, 'always', 100],
    
    // References
    'references-empty': [1, 'never'],
  },
  
  // Custom parsing options
  parserPreset: {
    parserOpts: {
      // Support for GitHub issue references
      issuePrefixes: ['#', 'PROJ-'],
      
      // Support for breaking changes
      noteKeywords: ['BREAKING CHANGE', 'BREAKING-CHANGE'],
      
      // Reference keywords for automatic linking
      referenceActions: [
        'close',
        'closes',
        'closed',
        'fix',
        'fixes',
        'fixed',
        'resolve',
        'resolves',
        'resolved',
      ],
    },
  },
  
  // Help message for invalid commits
  helpUrl: 'https://github.com/conventional-changelog/commitlint/#what-is-commitlint',
  
  // Ignore certain commit patterns
  ignores: [
    // Ignore merge commits
    (message) => message.includes('Merge branch'),
    (message) => message.includes('Merge pull request'),
    
    // Ignore revert commits (they have their own format)
    (message) => message.includes('Revert "'),
    
    // Ignore WIP commits in development
    (message) => message.startsWith('WIP:'),
    (message) => message.startsWith('wip:'),
    
    // Ignore fixup commits
    (message) => message.startsWith('fixup!'),
    (message) => message.startsWith('squash!'),
  ],
  
  // Default severity level
  defaultIgnores: true,
  
  // Custom formatter for better error messages
  formatter: '@commitlint/format',
  
  // Prompt configuration for interactive usage
  prompt: {
    questions: {
      type: {
        description: 'Select the type of change that you\'re committing:',
        enum: {
          feat: {
            description: 'A new feature',
            title: 'Features',
            emoji: '✨',
          },
          fix: {
            description: 'A bug fix',
            title: 'Bug Fixes',
            emoji: '🐛',
          },
          docs: {
            description: 'Documentation only changes',
            title: 'Documentation',
            emoji: '📚',
          },
          style: {
            description: 'Changes that do not affect the meaning of the code',
            title: 'Styles',
            emoji: '💎',
          },
          refactor: {
            description: 'A code change that neither fixes a bug nor adds a feature',
            title: 'Code Refactoring',
            emoji: '📦',
          },
          perf: {
            description: 'A code change that improves performance',
            title: 'Performance Improvements',
            emoji: '🚀',
          },
          test: {
            description: 'Adding missing tests or correcting existing tests',
            title: 'Tests',
            emoji: '🚨',
          },
          build: {
            description: 'Changes that affect the build system or external dependencies',
            title: 'Builds',
            emoji: '🛠',
          },
          ci: {
            description: 'Changes to our CI configuration files and scripts',
            title: 'Continuous Integrations',
            emoji: '⚙️',
          },
          chore: {
            description: 'Other changes that don\'t modify src or test files',
            title: 'Chores',
            emoji: '♻️',
          },
          revert: {
            description: 'Reverts a previous commit',
            title: 'Reverts',
            emoji: '🗑',
          },
          eval: {
            description: 'Changes to evaluation benchmarks or metrics',
            title: 'Evaluations',
            emoji: '📊',
          },
          security: {
            description: 'Security improvements or fixes',
            title: 'Security',
            emoji: '🔒',
          },
        },
      },
      scope: {
        description: 'What is the scope of this change (e.g. component or file name)',
      },
      subject: {
        description: 'Write a short, imperative tense description of the change',
      },
      body: {
        description: 'Provide a longer description of the change',
      },
      isBreaking: {
        description: 'Are there any breaking changes?',
      },
      breakingBody: {
        description: 'Describe the breaking changes',
      },
      isIssueAffected: {
        description: 'Does this change affect any open issues?',
      },
      issuesBody: {
        description: 'If issues are closed, the commit requires a body. Please enter a longer description of the commit itself',
      },
      issues: {
        description: 'Add issue references (e.g. "fix #123", "re #123".)',
      },
    },
  },
};

// Example commit messages:
// feat(api): add model comparison endpoint
// fix(dashboard): resolve chart rendering issue on mobile
// docs(readme): update installation instructions
// test(benchmark): add integration tests for MMLU evaluation
// chore(deps): update security dependencies
// eval(mmlu): improve accuracy calculation for multi-choice questions
// security(auth): implement rate limiting for API endpoints
// perf(worker): optimize batch processing for large evaluations