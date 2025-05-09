import logging
import sys
from pathlib import Path
from src.config.config import LOG_FILE

def setup_logger(name: str) -> logging.Logger:
    """Set up and configure logger.
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_error(logger: logging.Logger, error: Exception, context: str = None):
    """Log error with context.
    
    Args:
        logger (logging.Logger): Logger instance
        error (Exception): Error to log
        context (str, optional): Context of the error
    """
    error_msg = f"{context + ': ' if context else ''}{str(error)}"
    logger.error(error_msg)
    logger.exception(error) 