import os
import pandas as pd
from typing import List, Dict, Any, Tuple, Union
import openai
from dotenv import load_dotenv
from .utils import dataframe_preview

# Load environment variables from .env file
load_dotenv()

class YinAgent:
    """
    The Yin agent acts as the controller/planner in the dual-LLM architecture.
    Responsible for understanding user intent, planning tasks, and validating Yang's output.
    """
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize the Yin agent with configuration.
        
        Args:
            model: The OpenAI model to use for the Yin agent
        """
        self.model = model
        
        # Initialize OpenAI client from environment variables loaded from .env
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
    
    def generate_instruction(
        self, 
        user_command: str, 
        data: pd.DataFrame,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Generate instructions for Yang based on user command and data.
        
        Args:
            user_command: The natural language command from the user
            data: The current state of the data
            conversation_history: The conversation context so far
            
        Returns:
            Detailed instructions for Yang to execute
        """
        # Generate data preview for context
        data_preview = dataframe_preview(data, max_rows=5)
        
        # Prepare the messages for the LLM
        messages = [
            {"role": "system", "content": """You are Yin, the controller/planner in a dual-LLM data processing system. 
Your job is to understand the user's intent, plan data processing tasks, and generate clear instructions 
for Yang, the executor agent. Focus on:
1. Understanding exactly what the user wants
2. Creating a precise and detailed plan
3. Providing clear, step-by-step instructions for Yang to follow
4. Describing the expected format of the final output

Yang can perform data operations and analysis using Python pandas, numpy, etc. and can call external APIs 
if needed. Be specific about what data transformations are needed and why."""},
        ]
        
        # Add relevant conversation history
        for entry in conversation_history:
            messages.append({"role": entry["role"], "content": entry["content"]})
        
        # Add the current request
        messages.append({"role": "user", "content": 
            f"Based on the user's command and the data, provide detailed instructions for Yang to execute.\n\n"
            f"User Command: {user_command}\n\n"
            f"Current Data Preview:\n{data_preview}\n\n"
            f"Generate clear, step-by-step instructions for Yang to execute this task. Be specific about what "
            f"data operations are needed and what the expected output should look like."
        })
        
        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,  # Lower temperature for more precise planning
            max_tokens=1000
        )
        
        # Extract and return the instruction
        return response.choices[0].message.content
    
    def validate_result(
        self,
        user_command: str,
        result: Any,
        explanation: str,
        conversation_history: List[Dict[str, str]]
    ) -> Tuple[bool, str]:
        """
        Validate if Yang's result satisfies the user's original request.
        
        Args:
            user_command: Original user command
            result: The result data from Yang's execution
            explanation: Yang's explanation of what was done
            conversation_history: The conversation context so far
            
        Returns:
            Tuple of (is_complete, feedback)
            - is_complete: Boolean indicating if the task is complete
            - feedback: Feedback string for Yang if task is not complete
        """
        # Prepare result preview for validation
        result_preview = ""
        if isinstance(result, pd.DataFrame):
            result_preview = dataframe_preview(result, max_rows=5)
        else:
            result_preview = str(result)[:1000]  # Limit length for non-DataFrame results
            
        # Prepare the messages for the LLM
        messages = [
            {"role": "system", "content": """You are Yin, the controller/validator in a dual-LLM data processing system.
Your job is to validate whether Yang's execution meets the user's original requirements. Be critical and thorough.
Only approve the result if it fully satisfies what the user asked for. If it's incomplete or incorrect, 
provide clear feedback on what's missing or wrong, and specific instructions on how to fix it."""},
        ]
        
        # Add the validation request
        messages.append({"role": "user", "content": 
            f"Validate if Yang's result meets the original user requirements.\n\n"
            f"Original User Command: {user_command}\n\n"
            f"Yang's Execution Explanation: {explanation}\n\n"
            f"Result Preview:\n{result_preview}\n\n"
            f"1. Does this result completely satisfy the user's request? (Yes/No)\n"
            f"2. If not, what's missing or incorrect?\n"
            f"3. Provide feedback for Yang, including any additional steps needed to complete the task correctly."
        })
        
        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=1000
        )
        
        feedback = response.choices[0].message.content
        
        # Determine if the task is complete based on the response
        is_complete = "yes" in feedback.lower().split("\n")[0].lower() and "no" not in feedback.lower().split("\n")[0].lower()
        
        return is_complete, feedback
