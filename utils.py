import logging
import os
import json
import pandas as pd
from typing import Dict, Any, Optional
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Create and configure a logger with the specified name.
    
    Args:
        name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    # Create a logger
    logger = logging.getLogger(name)
    
    # Set the logging level
    logger.setLevel(logging.INFO)
    
    # If the logger already has handlers, return it
    if logger.handlers:
        return logger
    
    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add the formatter to the handler
    console_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(console_handler)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create a file handler
    file_handler = logging.FileHandler(f'logs/{name.lower()}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(file_handler)
    
    return logger

def read_data_file(file_path: str) -> pd.DataFrame:
    """
    Read data from a CSV or Excel file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Pandas DataFrame containing the data
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine the file type based on the extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Read the file into a DataFrame
    if file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif file_extension == '.csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def save_data_file(df: pd.DataFrame, output_path: str, file_format: str = None):
    """
    Save a DataFrame to a CSV or Excel file.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the file
        file_format: Optional format override ('csv' or 'excel')
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine the file format based on the extension or the specified format
    if file_format:
        file_format = file_format.lower()
    else:
        file_extension = os.path.splitext(output_path)[1].lower()
        if file_extension in ['.xlsx', '.xls']:
            file_format = 'excel'
        elif file_extension == '.csv':
            file_format = 'csv'
        else:
            file_format = 'csv'  # Default to CSV
    
    # Save the DataFrame
    if file_format == 'excel':
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

def parse_json_safely(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse a JSON string, handling potential errors.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed JSON object or None if parsing fails
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to extract JSON from a string that might contain other text
        try:
            # Look for JSON-like patterns
            start_idx = json_str.find("{")
            end_idx = json_str.rfind("}")
            
            if start_idx >= 0 and end_idx > start_idx:
                json_content = json_str[start_idx:end_idx + 1]
                return json.loads(json_content)
            
            return None
        except Exception:
            return None
