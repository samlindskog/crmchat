import os
import logging
from pathlib import Path

# Create logger for this module
logger = logging.getLogger(__name__)


class Config:
    """Configuration class that loads environment variables."""
    
    def __init__(self):
        # Required environment variables
        self.REQUIRED_VARS = {
            'AIRTABLE_API_KEY': os.getenv('AIRTABLE_API_KEY'),
            'AIRTABLE_BASE_ID': os.getenv('AIRTABLE_BASE_ID'),
            'AIRTABLE_TABLES': os.getenv('AIRTABLE_TABLES', '').split(',')
        }
        
        # Optional environment variables with defaults
        self.OPTIONAL_VARS = {
            'AIRTABLE_OUTPUT_DIR': Path(os.getenv('AIRTABLE_OUTPUT_DIR', 'csv_output')),
            'AIRTABLE_SYNC_INTERVAL': int(os.getenv('AIRTABLE_SYNC_INTERVAL', '60')),  # minutes
            'AIRTABLE_RATE_LIMIT': float(os.getenv('AIRTABLE_RATE_LIMIT', '5')),  # requests per second
            'AIRTABLE_ERROR_WAIT': int(os.getenv('AIRTABLE_ERROR_WAIT', '35')),  # seconds
            'AIRTABLE_TEST_MODE': os.getenv('AIRTABLE_TEST_MODE', 'false').lower() in ['1', 'true'],
            'AIRTABLE_SYNC_LOG_LEVEL': os.getenv('AIRTABLE_SYNC_LOG_LEVEL', 'INFO').upper(),
            'XAI_API_KEY': os.getenv('XAI_API_KEY'),
        }
        
        # Track if logging has been set up
        self._logging_configured = False
    
    # Convenience properties for backward compatibility
    @property
    def AIRTABLE_API_KEY(self):
        return self.REQUIRED_VARS['AIRTABLE_API_KEY']
    
    @property
    def AIRTABLE_BASE_ID(self):
        return self.REQUIRED_VARS['AIRTABLE_BASE_ID']
    
    @property
    def AIRTABLE_TABLES(self):
        return self.REQUIRED_VARS['AIRTABLE_TABLES']
    
    @property
    def AIRTABLE_OUTPUT_DIR(self):
        return self.OPTIONAL_VARS['AIRTABLE_OUTPUT_DIR']
    
    @property
    def AIRTABLE_SYNC_INTERVAL(self):
        return self.OPTIONAL_VARS['AIRTABLE_SYNC_INTERVAL']
    
    @property
    def AIRTABLE_RATE_LIMIT(self):
        return self.OPTIONAL_VARS['AIRTABLE_RATE_LIMIT']
    
    @property
    def AIRTABLE_ERROR_WAIT(self):
        return self.OPTIONAL_VARS['AIRTABLE_ERROR_WAIT']
    
    @property
    def AIRTABLE_TEST_MODE(self):
        return self.OPTIONAL_VARS['AIRTABLE_TEST_MODE']
    
    @property
    def AIRTABLE_SYNC_LOG_LEVEL(self):
        return self.OPTIONAL_VARS['AIRTABLE_SYNC_LOG_LEVEL']
    
    @property
    def XAI_API_KEY(self):
        return self.OPTIONAL_VARS['XAI_API_KEY']
    
    def setup_logging(self):
        """Set up logging configuration."""
        if not self._logging_configured:
            log_level = self.AIRTABLE_SYNC_LOG_LEVEL
            numeric_level = getattr(logging, log_level, None)
            if not isinstance(numeric_level, int):
                raise ValueError(f'Invalid log level: {log_level}')

            logging.basicConfig(
                level=numeric_level,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self._logging_configured = True
    
    def validate_required(self):
        """Validate that required environment variables are set."""
        missing_vars = []
        for var_name, var_value in self.REQUIRED_VARS.items():
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            raise ValueError(f"Required environment variables must be set: {', '.join(missing_vars)}")
    
    def log_configuration(self):
        """Log the current configuration."""
        logger.info(f"Output directory set to: {self.AIRTABLE_OUTPUT_DIR.absolute()}")
        logger.info(f"Sync interval set to: {self.AIRTABLE_SYNC_INTERVAL} minutes")
        logger.info(f"Tables to sync: {', '.join(self.AIRTABLE_TABLES)}")
        logger.info(f"Rate limit set to: {self.AIRTABLE_RATE_LIMIT} requests per second")
        logger.info(f"Error wait time set to: {self.AIRTABLE_ERROR_WAIT} seconds")
        logger.info(f"Test mode: {'ENABLED' if self.AIRTABLE_TEST_MODE else 'disabled'}")
        if not self.XAI_API_KEY:
            logger.warning('XAI_API_KEY not set. LinkedIn summaries will be skipped.') 