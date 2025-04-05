import os
import pandas as pd
from typing import Tuple, Union, Dict, Any, List
from .yin import YinAgent
from .yang import YangAgent
from .utils import dataframe_preview, format_output_based_on_content

class YinYangProcessor:
    """
    Main processor that implements the Yin-Yang Dual-LLM architecture for data processing.
    Coordinates the conversation between Yin (controller) and Yang (executor) agents.
    """
    
    def __init__(self, max_iterations: int = 10):
        """
        Initialize the Yin-Yang processor.
        
        Args:
            max_iterations: Maximum number of Yin-Yang conversation iterations
        """
        self.max_iterations = max_iterations
        self.yin = YinAgent()
        self.yang = YangAgent()
        self.conversation_history = []
        
    def process(self, data: pd.DataFrame, user_command: str) -> Tuple[Union[pd.DataFrame, str], str]:
        """
        Process user data and command through the Yin-Yang architecture.
        
        Args:
            data: User-provided dataframe
            user_command: Natural language instruction from the user
            
        Returns:
            Tuple of (result, output_format)
            - result: Either a dataframe or string depending on the processing output
            - output_format: File format extension ('csv', 'xlsx', or 'txt')
        """
        # Generate a preview of the data for the agents
        data_preview = dataframe_preview(data, max_rows=5)
        
        # Initialize conversation with user command
        self.conversation_history = []
        
        # Add initial user request to conversation
        self.conversation_history.append({
            "role": "user",
            "content": f"User Command: {user_command}\n\nData Preview:\n{data_preview}"
        })
        
        current_data = data
        final_result = None
        output_format = 'xlsx'  # Default format
        
        # Start the Yin-Yang conversation loop
        for iteration in range(self.max_iterations):
            # Yin's turn: understand and plan
            yin_instruction = self.yin.generate_instruction(
                user_command, 
                current_data,
                self.conversation_history
            )
            
            self.conversation_history.append({
                "role": "yin",
                "content": yin_instruction
            })
            
            # Yang's turn: execute
            yang_result, yang_explanation = self.yang.execute_task(
                yin_instruction,
                current_data,
                self.conversation_history
            )
            
            self.conversation_history.append({
                "role": "yang",
                "content": yang_explanation
            })
            
            # Update current data if Yang returned a dataframe
            if isinstance(yang_result, pd.DataFrame):
                current_data = yang_result
            
            # Yin's validation
            is_complete, yin_feedback = self.yin.validate_result(
                user_command,
                yang_result,
                yang_explanation,
                self.conversation_history
            )
            
            self.conversation_history.append({
                "role": "yin_validation",
                "content": yin_feedback
            })
            
            # If Yin is satisfied, we're done
            if is_complete:
                final_result = yang_result
                break
                
        # If we haven't set a final result (e.g., reached max iterations), use the latest result
        if final_result is None:
            final_result = current_data
        
        # Format the output appropriately
        formatted_result, output_format = format_output_based_on_content(final_result)
        
        return formatted_result, output_format
