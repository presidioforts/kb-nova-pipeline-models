#!/usr/bin/env python3
"""
Production Entry Point for Hybrid Knowledge Base Service
Uses the modular enterprise architecture from /src

This script provides a robust production-ready launcher with:
- Production-optimized uvicorn settings
- Comprehensive error handling and logging
- Environment-based configuration
- Graceful startup/shutdown procedures
- Health check verification
"""

import os
import sys
import logging
import signal
import asyncio
from pathlib import Path

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging():
    """Configure production-grade logging with Windows Unicode support"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Configure handlers with UTF-8 encoding for Windows
    handlers = []
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')
    handlers.append(console_handler)
    
    # File handler with UTF-8 encoding
    try:
        file_handler = logging.FileHandler('kb-service.log', mode='a', encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level))
        handlers.append(file_handler)
    except Exception:
        # Fallback if file logging fails
        pass
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=handlers,
        force=True
    )
    
    # Set specific loggers
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('uvicorn.access').setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

def validate_environment():
    """Validate required environment and dependencies"""
    logger = logging.getLogger(__name__)
    
    try:
        # Test critical imports
        import uvicorn
        import fastapi
        from src.main import app
        logger.info("All critical dependencies available")
        return True
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install required packages: pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False

def get_server_config():
    """Get production server configuration from environment"""
    return {
        "host": os.getenv('HOST', '0.0.0.0'),
        "port": int(os.getenv('PORT', 8000)),
        "workers": int(os.getenv('WORKERS', 1)),  # Single worker for development, increase for production
        "reload": os.getenv('RELOAD', 'false').lower() == 'true',
        "log_level": os.getenv('LOG_LEVEL', 'info').lower(),
        "access_log": os.getenv('ACCESS_LOG', 'true').lower() == 'true',
        "use_colors": os.getenv('USE_COLORS', 'true').lower() == 'true',
    }

async def health_check_startup():
    """Perform startup health check"""
    logger = logging.getLogger(__name__)
    
    try:
        from src.models.hybrid_knowledge_base import get_hybrid_kb
        
        logger.info("Performing startup health check...")
        hybrid_kb = await get_hybrid_kb()
        health = await hybrid_kb.health_check()
        
        if health.get('status') == 'healthy':
            logger.info("Startup health check passed")
            return True
        else:
            logger.warning(f"Health check warning: {health}")
            return True  # Continue startup but log warning
            
    except Exception as e:
        logger.error(f"Startup health check failed: {e}")
        logger.error("Service may not be fully functional")
        return False

def setup_signal_handlers():
    """Setup graceful shutdown signal handlers"""
    logger = logging.getLogger(__name__)
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point with comprehensive error handling"""
    # Setup logging first
    logger = setup_logging()
    logger.info("Starting Hybrid Knowledge Base Service...")
    
    try:
        # Setup signal handlers
        setup_signal_handlers()
        
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed, exiting...")
            sys.exit(1)
        
        # Get server configuration
        config = get_server_config()
        logger.info(f"Server configuration: {config}")
        
        # Perform startup health check
        try:
            startup_healthy = asyncio.run(health_check_startup())
            if not startup_healthy:
                logger.warning("Startup health check failed, but continuing...")
        except Exception as e:
            logger.warning(f"Could not perform startup health check: {e}")
        
        # Import the app
        from src.main import app
        
        # Start uvicorn server
        import uvicorn
        
        logger.info(f"Starting server on {config['host']}:{config['port']}")
        logger.info(f"API Documentation: http://{config['host']}:{config['port']}/docs")
        logger.info(f"Health Check: http://{config['host']}:{config['port']}/api/v1/health")
        
        uvicorn.run(
            "src.main:app",
            host=config['host'],
            port=config['port'],
            workers=config['workers'],
            reload=config['reload'],
            log_level=config['log_level'],
            access_log=config['access_log'],
            use_colors=config['use_colors'],
        )
        
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error starting service: {e}", exc_info=True)
        logger.error("Check logs above for details")
        sys.exit(1)
    finally:
        logger.info("Hybrid Knowledge Base Service shutdown complete")

if __name__ == "__main__":
    main() 