import json
import os
import requests
from typing import Dict, Tuple, Any, List, Optional
import openai
from dotenv import load_dotenv

from utils import get_logger

# Load environment variables
load_dotenv()

class YinAgent:
    """
    Yin Agent: The planner and validator component of the Yin-Yang architecture.
    Responsible for:
    1. Understanding user commands
    2. Processing input data row by row
    3. Building context for each row
    4. Formulating tasks for Yang
    5. Validating Yang's responses
    6. Managing the enrichment process
    """
    
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the Yin Agent.
        
        Args:
            model_name: The OpenAI model to use
        """
        self.model_name = model_name
        self.logger = get_logger("YinAgent")
        self.user_command = None
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def initialize_with_command(self, user_command: str):
        """
        Initialize the agent with a user command.
        
        Args:
            user_command: Natural language command from the user
        """
        self.user_command = user_command
        self.logger.info(f"Initialized with command: {user_command}")
    
    def build_row_context(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build context for a specific row, possibly integrating external API data.
        
        Args:
            row_data: Dictionary containing the row data
            
        Returns:
            Context dictionary including row data and any API results
        """
        context = {
            "row_data": row_data,
            # "api_results": {}, remove api calling until implmented
        }
        
        # Check if we need to call any APIs based on the column names and values
        # for column, value in row_data.items():
        #     if isinstance(value, str) and column.lower() in ["name", "full_name", "person_name"]:
        #         try:
        #             # Example of calling an external API for name information
        #             api_result = self._lookup_name_info(value)
        #             if api_result:
        #                 context["api_results"]["name_info"] = api_result
        #         except Exception as e:
        #             self.logger.error(f"Error calling API for {column}={value}: {str(e)}")
        
        return context
    
    # def _lookup_name_info(self, name: str) -> Optional[Dict[str, Any]]:
    #     """
    #     Example function to lookup information about a name using an external API.
    #     In a real implementation, this would call an actual API.
    #     
    #     Args:
    #         name: The name to look up
    #         
    #     Returns:
    #         Dictionary with name information or None if the lookup failed
    #     """
    #     # This is a mock implementation. In a real system, you would call an actual API.
    #     try:
    #         # Example API call (mocked)
    #         # In a real implementation, you would use requests.get() to call an actual API
    #         mock_response = {
    #             "name": name,
    #             "gender_probability": 0.95,
    #             "likely_gender": "male" if name.lower() in ["john", "david", "michael"] else "female",
    #             "likely_country": "United States",
    #             "country_probability": 0.85
    #         }
    #         
    #         return mock_response
    #     except Exception as e:
    #         self.logger.error(f"Error looking up name info for {name}: {str(e)}")
    #         return None
    
    def formulate_yang_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formulate a task for Yang based on the context and user command.
        
        Args:
            context: The context dictionary
            
        Returns:
            Task dictionary for Yang
        """
        # Prepare the system message that explains Yang's role
        system_message = {
            "role": "system",
            "content": (
                "You are Yang, an AI assistant specialized in data enrichment. "
                "Your task is to analyze data and provide enrichment attributes. "
                "You should return ONLY a JSON object with your enrichment results. "
                "Each key in the JSON object should be an attribute, and each value should be the enrichment value in string data type."
                "Only include your explanations and evidence weblink in evidence field when requested, to keep other fields clean "
            )
        }
        
        # Prepare the user message that contains the task for Yang
        row_description = json.dumps(context["row_data"], indent=2)
        # api_info = json.dumps(context["api_results"], indent=2) if context["api_results"] else "No API results available" # TODO: add api when available
        
        user_message = {
            "role": "user",
            "content": (
                f"I need you to enrich the following data row:\n\n"
                f"DATA ROW:\n{row_description}\n\n"
                #f"API INFORMATION:\n{api_info}\n\n"
                f"ENRICHMENT TASK:\n{self.user_command}\n\n"
                f"Return ONLY a JSON object with the enriched attributes as key-value pairs. "
                f"Include your explanations and evidence weblink in evidence field when requested"
            )
        }
        
        task = {
            "messages": [system_message, user_message],
            "context": context
        }
        
        return task
    
    def validate_yang_response(self, yang_response: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate Yang's response using the LLM's judgment.
        
        Args:
            yang_response: Response from Yang
            context: The context used to generate the response
            
        Returns:
            Tuple of (is_valid, validation_message)
        """
        try:
            # Check if the response is a valid JSON object
            if not isinstance(yang_response, dict):
                return False, "Response is not a valid JSON object"
            
            # Check if the response is empty
            if not yang_response:
                return False, "Response is empty"
            
            # For more complex validation, we can use the LLM itself to validate the response
            system_message = {
                "role": "system",
                "content": (
                    "You are a validation assistant. Your task is to determine if the enrichment data "
                    "provided matches the requirements and is likely to be accurate. "
                    "Respond with your validation result and reasoning."
                )
            }
            
            row_description = json.dumps(context["row_data"], indent=2)
            enrichment_result = json.dumps(yang_response, indent=2)
            
            user_message = {
                "role": "user",
                "content": (
                    f"Please validate the following enrichment result:\n\n"
                    f"ORIGINAL DATA ROW:\n{row_description}\n\n"
                    f"ENRICHMENT TASK:\n{self.user_command}\n\n"
                    f"ENRICHMENT RESULT:\n{enrichment_result}\n\n"
                    f"Is this enrichment result looks reasonable and likely accurate with mid-low level of confidence? "
                    f"If yes, respond with 'WOOHOO', if not, respond with 'NAYNAY'. followed by your reasoning."
                )
            }
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                # tools=[
                #     {
                #     "type": "web_search_preview",
                #     "user_location": {
                #         "type": "approximate",
                #         "country": "US"
                #     },
                #     "search_context_size": "low"
                #     }
                # ],
                messages=[system_message, user_message],
                max_tokens=1000,
                # temperature=0.3,
            )
            
            validation_response = response.choices[0].message.content
            
            # Check if the validation response indicates the enrichment is valid
            is_valid = "WOOHOO" in validation_response.upper()
            
            return is_valid, validation_response
        
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            return False, f"Validation error: {str(e)}"
