// Semantic Release Configuration for AGI Evaluation Sandbox
// ========================================================

const config = {
  // Branch configuration
  branches: [
    '+([0-9])?(.{+([0-9]),x}).x',
    'main',
    'next',
    'next-major',
    {
      name: 'beta',
      prerelease: true
    },
    {
      name: 'alpha',
      prerelease: true
    }
  ],

  // Plugins configuration
  plugins: [
    // ──────────────────────────────────────────────────────────────────
    // 📝 COMMIT ANALYSIS
    // ──────────────────────────────────────────────────────────────────
    [
      '@semantic-release/commit-analyzer',
      {
        preset: 'conventionalcommits',
        releaseRules: [
          // Breaking changes
          { type: 'feat', scope: 'api', release: 'major' },
          { type: 'feat', scope: 'breaking', release: 'major' },
          { breaking: true, release: 'major' },
          
          // Features
          { type: 'feat', release: 'minor' },
          { type: 'feature', release: 'minor' },
          
          // Bug fixes
          { type: 'fix', release: 'patch' },
          { type: 'bugfix', release: 'patch' },
          { type: 'hotfix', release: 'patch' },
          
          // Security fixes
          { type: 'security', release: 'patch' },
          
          // Performance improvements
          { type: 'perf', release: 'patch' },
          { type: 'performance', release: 'patch' },
          
          // Documentation (no release)
          { type: 'docs', release: false },
          { type: 'doc', release: false },
          
          // Refactoring (patch)
          { type: 'refactor', release: 'patch' },
          
          // Tests (no release)
          { type: 'test', release: false },
          
          // Build/CI changes (no release)
          { type: 'build', release: false },
          { type: 'ci', release: false },
          { type: 'chore', release: false },
          
          // Style changes (no release)
          { type: 'style', release: false },
          
          // Reverts
          { type: 'revert', release: 'patch' }
        ],
        parserOpts: {
          noteKeywords: ['BREAKING CHANGE', 'BREAKING CHANGES', 'BREAKING']
        }
      }
    ],

    // ──────────────────────────────────────────────────────────────────
    // 📋 RELEASE NOTES GENERATION
    // ──────────────────────────────────────────────────────────────────
    [
      '@semantic-release/release-notes-generator',
      {
        preset: 'conventionalcommits',
        presetConfig: {
          types: [
            { type: 'feat', section: '🚀 Features' },
            { type: 'feature', section: '🚀 Features' },
            { type: 'fix', section: '🐛 Bug Fixes' },
            { type: 'bugfix', section: '🐛 Bug Fixes' },
            { type: 'hotfix', section: '🐛 Hotfixes' },
            { type: 'perf', section: '⚡ Performance Improvements' },
            { type: 'performance', section: '⚡ Performance Improvements' },
            { type: 'refactor', section: '♻️ Code Refactoring' },
            { type: 'security', section: '🛡️ Security Fixes' },
            { type: 'revert', section: '⏪ Reverts' },
            { type: 'docs', section: '📚 Documentation', hidden: false },
            { type: 'doc', section: '📚 Documentation', hidden: false },
            { type: 'style', section: '💄 Styles', hidden: true },
            { type: 'build', section: '🏗️ Build System', hidden: true },
            { type: 'ci', section: '👷 CI/CD', hidden: true },
            { type: 'test', section: '🧪 Tests', hidden: true },
            { type: 'chore', section: '🔧 Chores', hidden: true }
          ]
        },
        writerOpts: {
          groupBy: 'type',
          commitGroupsSort: 'title',
          commitsSort: ['scope', 'subject'],
          noteGroupsSort: 'title',
          mainTemplate: `# {{#if isPatch}}{{#if title}}{{title}}{{else}}Bug Fixes{{/if}}{{else}}{{#if title}}{{title}}{{else}}{{#if isMajor}}Breaking Changes{{else}}Features{{/if}}{{/if}}{{/if}}

{{#each commitGroups}}
{{#if title}}
## {{title}}

{{/if}}
{{#each commits}}
{{#if scope}}* **{{scope}}:** {{/if}}{{subject}}{{#if hash}} ([{{hash}}]({{repository.url}}/commit/{{hash}})){{/if}}
{{#if body}}

  {{indent body "  "}}
{{/if}}
{{/each}}
{{/each}}
{{#if noteGroups}}
{{#each noteGroups}}

## {{title}}

{{#each notes}}
* {{text}}
{{/each}}
{{/each}}
{{/if}}
`,
          commitTemplate: `* {{#if scope}}**{{scope}}:** {{/if}}{{subject}}{{#if hash}} ([{{hash}}]({{@root.repository.url}}/commit/{{hash}})){{/if}}
{{~!-- commit link --}}`,
          footerPartial: `{{#if noteGroups}}
{{#each noteGroups}}

### {{title}}
{{#each notes}}
* {{text}}
{{/each}}
{{/each}}
{{/if}}`
        }
      }
    ],

    // ──────────────────────────────────────────────────────────────────
    // 📋 CHANGELOG GENERATION
    // ──────────────────────────────────────────────────────────────────
    [
      '@semantic-release/changelog',
      {
        changelogFile: 'CHANGELOG.md',
        changelogTitle: `# 📋 Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
<!-- Add new changes here -->
`
      }
    ],

    // ──────────────────────────────────────────────────────────────────
    // 📦 PACKAGE.JSON VERSION UPDATE
    // ──────────────────────────────────────────────────────────────────
    [
      '@semantic-release/npm',
      {
        npmPublish: false, // We don't publish to npm registry
        tarballDir: 'dist'
      }
    ],

    // ──────────────────────────────────────────────────────────────────
    // 📝 GIT COMMIT AND TAG
    // ──────────────────────────────────────────────────────────────────
    [
      '@semantic-release/git',
      {
        assets: [
          'CHANGELOG.md',
          'package.json',
          'package-lock.json',
          'api/pyproject.toml',
          'dashboard/package.json',
          'dashboard/package-lock.json'
        ],
        message: 'chore(release): ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}'
      }
    ],

    // ──────────────────────────────────────────────────────────────────
    // 🏷️ GITHUB RELEASE
    // ──────────────────────────────────────────────────────────────────
    [
      '@semantic-release/github',
      {
        assets: [
          {
            path: 'dist/*.tgz',
            label: 'Source Distribution'
          },
          {
            path: 'api/dist/*.whl',
            label: 'Python Wheel'
          },
          {
            path: 'api/dist/*.tar.gz',
            label: 'Python Source Distribution'
          },
          {
            path: 'dashboard/dist/*.zip',
            label: 'Dashboard Build'
          }
        ],
        assignees: process.env.GITHUB_RELEASE_ASSIGNEES?.split(',') || [],
        addReleases: 'top',
        draftRelease: false,
        failComment: false,
        failTitle: 'Release failed: v${nextRelease.version}',
        labels: ['release'],
        releasedLabels: ['released'],
        successComment: `🎉 This ${issue.pull_request ? 'PR is included' : 'issue has been resolved'} in version [v\${nextRelease.version}](\${releases.filter(release => release.name)[0].url}) 🎉

The release is available on:
- [GitHub Releases](\${releases.filter(release => release.name)[0].url})
- [Container Registry](https://ghcr.io/\${context.repo.owner}/\${context.repo.repo}:v\${nextRelease.version})

Your **[semantic-release](https://github.com/semantic-release/semantic-release)** bot 📦🚀`,
        releaseBodyTemplate: `{{#each releases}}{{#if @first}}## What's Changed

{{/if}}{{notes}}
{{#unless @last}}

---

{{/unless}}{{/each}}

## Installation

### Docker (Recommended)
\`\`\`bash
docker pull ghcr.io/{{owner}}/{{repo}}:v{{version}}
\`\`\`

### From Source
\`\`\`bash
git clone https://github.com/{{owner}}/{{repo}}.git
cd {{repo}}
git checkout v{{version}}
make install
\`\`\`

## Documentation

- 📖 [Full Documentation](https://docs.{{owner}}.com/{{repo}})
- 🚀 [Quick Start Guide](https://github.com/{{owner}}/{{repo}}/blob/v{{version}}/README.md#quick-start)
- 📋 [API Reference](https://docs.{{owner}}.com/{{repo}}/api)

## Support

- 🐛 [Report Issues](https://github.com/{{owner}}/{{repo}}/issues)
- 💬 [Join Discord](https://discord.gg/{{owner}})
- 📧 [Email Support](mailto:support@{{owner}}.com)

**Full Changelog**: https://github.com/{{owner}}/{{repo}}/compare/{{previousTag}}...v{{version}}`
      }
    ]
  ],

  // Git configuration
  repositoryUrl: process.env.GITHUB_REPOSITORY 
    ? `https://github.com/${process.env.GITHUB_REPOSITORY}.git`
    : undefined,

  // CI configuration
  ci: true,
  dryRun: false,
  debug: process.env.NODE_ENV === 'development' || process.env.SEMANTIC_RELEASE_DEBUG === 'true',

  // Tag format
  tagFormat: 'v${version}',

  // Global options
  preset: 'conventionalcommits'
};

module.exports = config;