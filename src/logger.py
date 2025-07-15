"""
Logging Configuration Module

This module sets up logging for the housing prices regression project.
It creates a log file with timestamp and configures logging format.
"""

import logging
import os
from datetime import datetime

# Create a log file name with current timestamp (month_day_year_hour_minute_second.log)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the logs directory path in the current working directory
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create the logs directory if it doesn't exist (exist_ok=True prevents errors if directory already exists)
os.makedirs(logs_path, exist_ok=True)

# Full path to the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging system with:
# - filename: where to save the log file
# - format: how each log message should be formatted
#   [timestamp] line_number module_name - log_level - message
# - level: minimum level of messages to log (INFO and above)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# This block only runs if this file is executed directly (not imported)
if __name__ == "__main__":
    # Log a test message to verify logging is working
    logging.info("Logging has started")
