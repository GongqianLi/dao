import os
import pandas as pd
import numpy as np
import requests
import json
from typing import List, Dict, Any, Tuple, Union, Callable
import openai
import importlib
import inspect
from dotenv import load_dotenv
from .utils import dataframe_preview

# Load environment variables from .env file
load_dotenv()

class YangAgent:
    """
    The Yang agent acts as the executor/tool in the dual-LLM architecture.
    Responsible for executing data tasks and calling external APIs based on Yin's instructions.
    """
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize the Yang agent with configuration.
        
        Args:
            model: The OpenAI model to use for the Yang agent
        """
        self.model = model
        
        # Initialize OpenAI client from environment variables loaded from .env
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Register available tools
        self.available_tools = self._register_tools()
    
    def _register_tools(self) -> Dict[str, Callable]:
        """Register all available data processing tools."""
        return {
            # Data transformation tools
            "filter_data": self._filter_data,
            "sort_data": self._sort_data,
            "group_data": self._group_data,
            "aggregate_data": self._aggregate_data,
            "pivot_data": self._pivot_data,
            "merge_data": self._merge_data,
            "clean_data": self._clean_data,
            "transform_column": self._transform_column,
            "add_column": self._add_column,
            "drop_column": self._drop_column,
            "rename_columns": self._rename_columns,
            "value_counts": self._value_counts,
            "sample_data": self._sample_data,
            
            # Analysis tools
            "describe_data": self._describe_data,
            "correlation": self._correlation,
            
            # API tools
            "call_api": self._call_api
        }
    
    def execute_task(
        self, 
        instruction: str, 
        data: pd.DataFrame,
        conversation_history: List[Dict[str, str]]
    ) -> Tuple[Any, str]:
        """
        Execute data processing tasks based on Yin's instructions.
        
        Args:
            instruction: The detailed instruction from Yin
            data: The data to process
            conversation_history: The conversation context so far
            
        Returns:
            Tuple of (result, explanation)
            - result: The result data (DataFrame or other output)
            - explanation: A detailed explanation of what was done
        """
        # Generate data preview
        data_preview = dataframe_preview(data, max_rows=5)
        
        # Tool documentation for context
        tool_docs = "\n".join([
            f"{name}: {inspect.getdoc(func)}" 
            for name, func in self.available_tools.items()
        ])
        
        # Prepare the messages for the LLM
        messages = [
            {"role": "system", "content": f"""You are Yang, the executor agent in a dual-LLM data processing system.
Your job is to execute data tasks according to Yin's instructions using pandas, numpy, and other Python tools.

You have access to these tools:
{tool_docs}

For each task:
1. Analyze what operations are needed based on Yin's instructions
2. Write and execute Python code using pandas to accomplish the task
3. Return the result and a detailed explanation of what you did

You can use Python code to manipulate the data. Be precise and follow Yin's instructions exactly.
Always explain your reasoning and include any code you would execute."""},
        ]
        
        # Add the execution request
        messages.append({"role": "user", "content": 
            f"Execute the following data processing task according to Yin's instructions:\n\n"
            f"Yin's Instructions: {instruction}\n\n"
            f"Current Data Preview:\n{data_preview}\n\n"
            f"Please execute this task and provide:\n"
            f"1. Your plan for executing this task\n"
            f"2. The Python code you would use to implement it\n"
            f"3. A detailed explanation of what your code does\n"
            f"4. The expected output format"
        })
        
        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4,  # Slightly higher temperature for creativity in solutions
            max_tokens=1500
        )
        
        # Extract the explanation from the response
        explanation = response.choices[0].message.content
        
        # Print the Yang execution response
        print("\n===== YANG EXECUTION RESPONSE =====")
        print(explanation)
        print("====================================\n")
        
        try:
            # Now execute the code on the actual data
            # For safety, we'll execute a second call to generate safe executable code
            exec_messages = messages.copy()
            exec_messages.append({"role": "assistant", "content": explanation})
            exec_messages.append({"role": "user", "content": 
                f"Based on your analysis, generate ONLY a self-contained Python function called 'execute_task' that takes a pandas DataFrame as input and returns the processed result. Include ONLY the function definition, no other text or explanation. The function should implement the operations you described above."
            })
            
            # Call the OpenAI API for executable code
            exec_response = self.client.chat.completions.create(
                model=self.model,
                messages=exec_messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            # Extract the executable code
            code = exec_response.choices[0].message.content
            
            # Print the Yang code generation response
            print("\n===== YANG CODE GENERATION RESPONSE =====")
            print(code)
            print("=========================================\n")
            
            # Ensure we only have the function definition
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
                
            # Create a local namespace to execute the code
            local_namespace = {
                "pd": pd,
                "np": np,
                "data": data.copy(),  # Use a copy to avoid modifying the original
                "requests": requests,
                "json": json
            }
            
            # Execute the code in the local namespace
            exec(code, globals(), local_namespace)
            
            # Get the result from the execute_task function
            if "execute_task" in local_namespace:
                result = local_namespace["execute_task"](data.copy())
            else:
                # Fallback if function name is different
                # Find a function in the namespace and call it with data
                for name, obj in local_namespace.items():
                    if callable(obj) and name not in globals() and name != "data":
                        result = obj(data.copy())
                        break
                else:
                    # If no function is found, look for a result variable
                    result = local_namespace.get("result", data.copy())
            
            return result, explanation
            
        except Exception as e:
            # If execution fails, return an error explanation
            error_explanation = f"Error executing code: {str(e)}\n\nOriginal plan: {explanation}"
            return data, error_explanation  # Return original data on error

    # Tool implementations
    def _filter_data(self, df: pd.DataFrame, condition: str) -> pd.DataFrame:
        """Filter dataframe rows based on a condition."""
        return df.query(condition)
    
    def _sort_data(self, df: pd.DataFrame, by: List[str], ascending: Union[bool, List[bool]] = True) -> pd.DataFrame:
        """Sort dataframe by one or more columns."""
        return df.sort_values(by=by, ascending=ascending)
    
    def _group_data(self, df: pd.DataFrame, by: List[str]) -> pd.core.groupby.DataFrameGroupBy:
        """Group dataframe by one or more columns."""
        return df.groupby(by)
    
    def _aggregate_data(self, df: pd.DataFrame, by: List[str], agg_dict: Dict[str, str]) -> pd.DataFrame:
        """Group and aggregate dataframe using specified aggregation functions."""
        return df.groupby(by).agg(agg_dict).reset_index()
    
    def _pivot_data(self, df: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame:
        """Create a pivot table from dataframe."""
        return pd.pivot_table(df, index=index, columns=columns, values=values)
    
    def _merge_data(self, df1: pd.DataFrame, df2: pd.DataFrame, how: str = 'inner', on: Union[str, List[str]] = None) -> pd.DataFrame:
        """Merge two dataframes."""
        return pd.merge(df1, df2, how=how, on=on)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning: remove duplicates, drop NAs, reset index."""
        return df.drop_duplicates().dropna().reset_index(drop=True)
    
    def _transform_column(self, df: pd.DataFrame, column: str, function: Callable) -> pd.DataFrame:
        """Apply a function to transform values in a column."""
        df_copy = df.copy()
        df_copy[column] = df_copy[column].apply(function)
        return df_copy
    
    def _add_column(self, df: pd.DataFrame, column: str, value) -> pd.DataFrame:
        """Add a new column with a constant or calculated value."""
        df_copy = df.copy()
        df_copy[column] = value
        return df_copy
    
    def _drop_column(self, df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
        """Drop one or more columns from dataframe."""
        return df.drop(columns=columns)
    
    def _rename_columns(self, df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
        """Rename columns based on a mapping dictionary."""
        return df.rename(columns=column_map)
    
    def _value_counts(self, df: pd.DataFrame, column: str, normalize: bool = False) -> pd.Series:
        """Get value counts for a column."""
        return df[column].value_counts(normalize=normalize)
    
    def _sample_data(self, df: pd.DataFrame, n: int = None, frac: float = None) -> pd.DataFrame:
        """Sample n rows or fraction of rows from dataframe."""
        return df.sample(n=n, frac=frac)
    
    def _describe_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate descriptive statistics for dataframe."""
        return df.describe()
    
    def _correlation(self, df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix for numeric columns."""
        return df.select_dtypes(include=[np.number]).corr(method=method)
    
    def _call_api(self, url: str, method: str = 'GET', params: Dict = None, data: Dict = None, headers: Dict = None) -> Dict:
        """Call an external REST API endpoint."""
        response = requests.request(
            method=method,
            url=url,
            params=params,
            json=data,
            headers=headers
        )
        return response.json()
