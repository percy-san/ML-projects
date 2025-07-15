"""
Custom Exception Handling Module

This module provides custom exception handling for the project.
It creates detailed error messages that include file name, line number,
and the actual error message to help with debugging.
"""

import sys
import logging
from src.logger import logging


def error_message_details(error, error_detail):
    """
    Extract detailed information about an error for better debugging.
    
    This function takes an error and extracts the file name, line number,
    and error message to create a comprehensive error description.
    
    Args:
        error: The exception that was raised
        error_detail: The exception info object containing traceback details
    
    Returns:
        str: A formatted error message with file name, line number, and error details
    """
    # Get the traceback information (file, line, function, etc.)
    _, _, exc_tb = error_detail.exc_info()
    
    # Extract the file name where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Create a detailed error message with file name, line number, and error message
    error_message = "Error occurred in python script name [{0}] line number[{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)

    )
    return error_message


class CustomException(Exception):
    """
    Custom exception class that provides detailed error information.
    
    This class extends the built-in Exception class and adds
    detailed error tracking including file name and line number.
    """
    
    def __init__(self, error_message, error_detail: sys):
        """
        Initialize the custom exception with detailed error information.
        
        Args:
            error_message: The main error message
            error_detail: System information for detailed error tracking
        """
        # Call the parent Exception class constructor
        super().__init__(error_message)
        
        # Store the error message
        self.error_message = error_message
        
        # Create detailed error information including file name and line number
        self.error_detail = error_message_details(error_message, error_detail=error_detail)

    def __str__(self):
        """
        Return the error message when the exception is converted to string.
        
        Returns:
            str: The error message
        """
        return self.error_message
    

