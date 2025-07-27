#!/usr/bin/env python3
"""
Health check script for AGI Evaluation Sandbox Docker containers.

This script performs comprehensive health checks including:
- HTTP endpoint availability
- Database connectivity
- Redis connectivity
- Basic service functionality
"""

import asyncio
import os
import sys
import time
from typing import Dict, List, Optional

import aiohttp
import asyncpg
import redis


class HealthChecker:
    """Comprehensive health checker for all services."""
    
    def __init__(self):
        self.port = int(os.getenv("PORT", "8000"))
        self.database_url = os.getenv("DATABASE_URL", "")
        self.redis_url = os.getenv("REDIS_URL", "")
        self.timeout = int(os.getenv("HEALTH_CHECK_TIMEOUT", "10"))
        
    async def check_http_endpoint(self) -> Dict[str, any]:
        """Check if HTTP endpoint is responding."""
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"http://localhost:{self.port}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "healthy",
                            "response_time": data.get("response_time", 0),
                            "details": data
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "error": f"HTTP {response.status}",
                            "details": await response.text()
                        }
        except asyncio.TimeoutError:
            return {
                "status": "unhealthy",
                "error": "Timeout",
                "details": f"No response within {self.timeout}s"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "HTTP endpoint check failed"
            }
    
    async def check_database(self) -> Dict[str, any]:
        """Check database connectivity and basic operations."""
        if not self.database_url:
            return {
                "status": "skipped",
                "details": "No DATABASE_URL configured"
            }
        
        try:
            # Parse database URL
            conn = await asyncpg.connect(self.database_url)
            
            # Test basic query
            start_time = time.time()
            result = await conn.fetchval("SELECT 1")
            query_time = time.time() - start_time
            
            # Test table existence (basic schema check)
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            
            await conn.close()
            
            if result == 1:
                return {
                    "status": "healthy",
                    "response_time": query_time,
                    "details": {
                        "tables_count": len(tables),
                        "query_time_ms": round(query_time * 1000, 2)
                    }
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "Unexpected query result",
                    "details": f"Expected 1, got {result}"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Database connectivity check failed"
            }
    
    def check_redis(self) -> Dict[str, any]:
        """Check Redis connectivity and basic operations."""
        if not self.redis_url:
            return {
                "status": "skipped",
                "details": "No REDIS_URL configured"
            }
        
        try:
            # Connect to Redis
            r = redis.from_url(self.redis_url, socket_timeout=self.timeout)
            
            # Test basic operations
            start_time = time.time()
            r.ping()
            ping_time = time.time() - start_time
            
            # Test set/get operations
            test_key = "health_check_test"
            test_value = "test_value"
            r.set(test_key, test_value, ex=10)  # Expire in 10 seconds
            retrieved_value = r.get(test_key)
            r.delete(test_key)
            
            if retrieved_value and retrieved_value.decode() == test_value:
                return {
                    "status": "healthy",
                    "response_time": ping_time,
                    "details": {
                        "ping_time_ms": round(ping_time * 1000, 2),
                        "operations": "set/get/delete successful"
                    }
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "Redis operation failed",
                    "details": f"Expected {test_value}, got {retrieved_value}"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Redis connectivity check failed"
            }
    
    async def check_service_specific(self) -> Dict[str, any]:
        """Check service-specific functionality."""
        try:
            # This would contain service-specific health checks
            # For example, checking if model providers are accessible,
            # if benchmark data is loaded, etc.
            
            # For now, just check basic Python imports
            start_time = time.time()
            
            # Test critical imports
            import fastapi
            import sqlalchemy
            import celery
            
            import_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": import_time,
                "details": {
                    "fastapi_version": fastapi.__version__,
                    "sqlalchemy_version": sqlalchemy.__version__,
                    "celery_version": celery.__version__,
                    "import_time_ms": round(import_time * 1000, 2)
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Service-specific check failed"
            }
    
    async def run_all_checks(self) -> Dict[str, any]:
        """Run all health checks and return comprehensive status."""
        start_time = time.time()
        
        # Run checks
        http_check = await self.check_http_endpoint()
        db_check = await self.check_database()
        redis_check = self.check_redis()
        service_check = await self.check_service_specific()
        
        total_time = time.time() - start_time
        
        # Determine overall status
        checks = {
            "http": http_check,
            "database": db_check,
            "redis": redis_check,
            "service": service_check
        }
        
        # Overall status is healthy if all critical checks pass
        critical_checks = ["http", "service"]
        overall_healthy = all(
            checks[check]["status"] in ["healthy", "skipped"] 
            for check in critical_checks
        )
        
        # Database and Redis are important but not critical for basic health
        important_checks = ["database", "redis"]
        important_healthy = all(
            checks[check]["status"] in ["healthy", "skipped"] 
            for check in important_checks
        )
        
        if overall_healthy and important_healthy:
            overall_status = "healthy"
        elif overall_healthy:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "total_check_time": round(total_time, 3),
            "checks": checks,
            "summary": {
                "critical_healthy": overall_healthy,
                "important_healthy": important_healthy,
                "total_checks": len(checks),
                "healthy_checks": sum(1 for c in checks.values() if c["status"] == "healthy"),
                "unhealthy_checks": sum(1 for c in checks.values() if c["status"] == "unhealthy")
            }
        }


async def main():
    """Main health check function."""
    try:
        checker = HealthChecker()
        result = await checker.run_all_checks()
        
        # Print JSON result for machine readability
        import json
        print(json.dumps(result, indent=2))
        
        # Exit with appropriate code
        if result["status"] == "healthy":
            sys.exit(0)
        elif result["status"] == "degraded":
            # Degraded state might be acceptable for some use cases
            sys.exit(0 if os.getenv("ALLOW_DEGRADED", "false").lower() == "true" else 1)
        else:
            sys.exit(1)
            
    except Exception as e:
        # If health check itself fails, report critical error
        error_result = {
            "status": "critical",
            "timestamp": time.time(),
            "error": str(e),
            "details": "Health check script failed"
        }
        
        import json
        print(json.dumps(error_result, indent=2))
        sys.exit(2)  # Critical error code


def simple_check():
    """Simple, fast health check for basic liveness probe."""
    try:
        # Just check if the process is running and basic imports work
        import sys
        import os
        
        # Check if this is the right process
        if os.getenv("HEALTH_CHECK_SIMPLE", "false").lower() == "true":
            print('{"status": "alive", "timestamp": ' + str(time.time()) + '}')
            sys.exit(0)
        
        # Fall back to full check
        asyncio.run(main())
        
    except Exception as e:
        print('{"status": "dead", "error": "' + str(e) + '", "timestamp": ' + str(time.time()) + '}')
        sys.exit(1)


if __name__ == "__main__":
    # Support both simple and comprehensive health checks
    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        simple_check()
    else:
        asyncio.run(main())