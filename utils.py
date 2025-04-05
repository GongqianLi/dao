import os
import pandas as pd
from typing import Union, Tuple, List, Dict, Any

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_dataframe(filepath: str) -> pd.DataFrame:
    """Read a dataframe from a file based on its extension."""
    ext = filepath.rsplit('.', 1)[1].lower()
    if ext == 'csv':
        return pd.read_csv(filepath)
    elif ext in ['xlsx', 'xls']:
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def dataframe_preview(df: pd.DataFrame, max_rows: int = 5, max_cols: int = None) -> str:
    """Generate a preview of a dataframe suitable for LLM consumption."""
    preview_df = df.head(max_rows)
    
    if max_cols and len(df.columns) > max_cols:
        preview_df = preview_df.iloc[:, :max_cols]
    
    # Get dataframe info
    num_rows, num_cols = df.shape
    dtypes = df.dtypes.to_dict()
    dtypes_str = ', '.join([f"{col}: {dtype}" for col, dtype in dtypes.items()])
    
    # Create preview string
    preview = f"Dataframe Shape: {num_rows} rows × {num_cols} columns\n"
    preview += f"Column Data Types: {dtypes_str}\n\n"
    preview += "Data Preview:\n"
    preview += preview_df.to_string()
    
    return preview

def format_output_based_on_content(result: Union[pd.DataFrame, str, dict]) -> Tuple[Union[pd.DataFrame, str], str]:
    """
    Determine the appropriate output format based on the content.
    Returns a tuple of (formatted_result, format_extension)
    """
    if isinstance(result, pd.DataFrame):
        return result, 'xlsx'
    elif isinstance(result, dict) or isinstance(result, list):
        # Convert to DataFrame if it's a structured data
        try:
            return pd.DataFrame(result), 'xlsx'
        except:
            return str(result), 'txt'
    else:
        # Default to text
        return str(result), 'txt'
