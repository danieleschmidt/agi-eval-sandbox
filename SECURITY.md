# Security Policy

## Supported Versions

We actively support security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| 0.9.x   | :white_check_mark: |
| < 0.9   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do NOT create a public GitHub issue

Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.

### 2. Send a private report

Instead, please send an email to **security@terragon.ai** with the following information:

- **Subject Line**: "Security Vulnerability Report - AGI Eval Sandbox"
- **Description**: A clear description of the vulnerability
- **Impact**: The potential impact if exploited
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Proof of Concept**: Code snippets, screenshots, or videos (if applicable)
- **Suggested Fix**: Any suggestions for remediation (optional)

### 3. Response Timeline

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Initial Assessment**: We will provide an initial assessment within 5 business days
- **Regular Updates**: We will provide updates every 7 days until resolution
- **Resolution**: We aim to resolve critical issues within 30 days

### 4. Responsible Disclosure

We follow responsible disclosure principles:

- We will work with you to understand and validate the issue
- We will not take legal action against researchers who:
  - Make a good faith effort to avoid disruption and data destruction
  - Only interact with accounts they own or have explicit permission to test
  - Report vulnerabilities privately and allow reasonable time for fixes
- We will publicly acknowledge your responsible disclosure (with your permission)

## Security Measures

### Application Security

- **Authentication**: JWT-based authentication with secure token handling
- **Authorization**: Role-based access control (RBAC)
- **Input Validation**: Comprehensive input sanitization and validation
- **SQL Injection Prevention**: Parameterized queries and ORM usage
- **XSS Prevention**: Content Security Policy and output encoding
- **CSRF Protection**: Anti-CSRF tokens for state-changing operations

### Infrastructure Security

- **HTTPS Enforcement**: All traffic encrypted with TLS 1.3
- **Security Headers**: Comprehensive security headers implementation
- **Container Security**: Regular vulnerability scanning and minimal base images
- **Secrets Management**: Secure handling and rotation of API keys and secrets
- **Network Security**: Firewall rules and network segmentation

### Data Protection

- **Encryption at Rest**: Database and file storage encryption
- **Encryption in Transit**: TLS encryption for all communications
- **Data Minimization**: Only collect and store necessary data
- **Data Retention**: Automatic cleanup of old evaluation data
- **Access Logging**: Comprehensive audit trails

### Monitoring & Detection

- **Security Monitoring**: Real-time security event monitoring
- **Anomaly Detection**: Automated detection of unusual patterns
- **Vulnerability Scanning**: Regular automated security scans
- **Penetration Testing**: Annual third-party security assessments
- **Incident Response**: Defined procedures for security incidents

## Security Best Practices for Users

### API Key Security

- **Never commit API keys** to version control
- **Use environment variables** for configuration
- **Rotate keys regularly** (at least every 90 days)
- **Use least privilege** principle for API permissions
- **Monitor usage** for unexpected activity

### Deployment Security

- **Use secure configurations** in production
- **Enable security headers** in reverse proxy
- **Keep dependencies updated** with automated tools
- **Use secrets management** for sensitive data
- **Enable audit logging** for compliance

### Development Security

- **Use pre-commit hooks** for security scanning
- **Perform dependency audits** regularly
- **Follow secure coding practices**
- **Test security controls** in development
- **Use static analysis tools**

## Security Tools Integration

### Automated Security Testing

- **SAST**: Static Application Security Testing with Bandit
- **Dependency Scanning**: Safety and Snyk for vulnerability detection
- **Container Scanning**: Trivy for Docker image vulnerabilities
- **Secret Scanning**: detect-secrets for credential detection
- **License Scanning**: Automated license compliance checking

### Runtime Security

- **Rate Limiting**: API rate limiting and DDoS protection
- **WAF Integration**: Web Application Firewall support
- **Intrusion Detection**: Automated threat detection
- **Security Metrics**: Prometheus-based security monitoring

## Compliance

### Standards Compliance

- **OWASP Top 10**: Protection against common web vulnerabilities
- **SOC 2 Type II**: Security, availability, and confidentiality controls
- **GDPR**: Data protection and privacy compliance
- **ISO 27001**: Information security management system

### Privacy Protection

- **Data Anonymization**: PII removal from evaluation data
- **Consent Management**: User consent tracking and management
- **Right to Deletion**: Support for data deletion requests
- **Data Portability**: Export capabilities for user data

## Security Updates

### Notification Channels

- **Security Advisories**: GitHub Security Advisories
- **Release Notes**: Security fixes highlighted in releases
- **Email Notifications**: Critical security updates via email
- **Status Page**: Real-time security incident updates

### Update Process

1. **Vulnerability Assessment**: Impact and severity analysis
2. **Patch Development**: Security fix implementation
3. **Testing**: Comprehensive security testing
4. **Release**: Coordinated security release
5. **Communication**: User notification and guidance

## Bug Bounty Program

We run a responsible bug bounty program for security researchers:

### Scope

- **In Scope**: Web application, API endpoints, authentication systems
- **Out of Scope**: Social engineering, physical attacks, DoS attacks
- **Rewards**: Based on severity and impact (up to $5,000)

### Rules

- **No disruption** to production services
- **No data exfiltration** or destruction
- **Report only** - do not exploit further
- **Follow responsible disclosure** timeline

## Contact Information

- **Security Team**: security@terragon.ai
- **General Contact**: support@terragon.ai
- **Emergency**: Use security email with "URGENT" in subject

## Acknowledgments

We thank the following security researchers for their responsible disclosure:

*This section will be updated as we receive and address security reports.*

---

**Last Updated**: January 2024  
**Next Review**: July 2024