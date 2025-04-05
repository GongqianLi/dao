import json
import os
from typing import Dict, Any, List, Optional
import openai
from dotenv import load_dotenv

from utils import get_logger

# Load environment variables
load_dotenv()

class YangAgent:
    """
    Yang Agent: The executor and retriever component of the Yin-Yang architecture.
    Responsible for:
    1. Receiving tasks from Yin
    2. Performing internet searches and information retrieval
    3. Analyzing data and reasoning about enrichment
    4. Returning structured enrichment results in JSON format
    """
    
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the Yang Agent.
        
        Args:
            model_name: The OpenAI model to use
        """
        self.model_name = model_name
        self.logger = get_logger("YangAgent")
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task received from Yin.
        
        Args:
            task: Dictionary containing the task details
            
        Returns:
            JSON-serializable dictionary with enrichment results
        """
        try:
            # Extract the messages from the task
            messages = task.get("messages", [])
            
            # Create a chat completion request with the messages
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=2000,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            # Extract the content from the response
            content = response.choices[0].message.content
            
            # Parse the JSON response
            try:
                result = json.loads(content)
                self.logger.info(f"Successfully processed task and generated JSON result")
                return result
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON response: {str(e)}")
                # If parsing fails, attempt to extract JSON from the response using string manipulation
                extracted_json = self._extract_json(content)
                if extracted_json:
                    return extracted_json
                
                # If extraction fails, return a simple error object
                return {"error": "Failed to generate valid JSON response"}
        
        except Exception as e:
            self.logger.error(f"Error processing task: {str(e)}")
            return {"error": f"Processing error: {str(e)}"}
    
    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to extract a JSON object from a string that might contain additional text.
        
        Args:
            content: String that might contain a JSON object
            
        Returns:
            Extracted JSON object or None if extraction fails
        """
        try:
            # Look for JSON-like patterns in the content
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx + 1]
                return json.loads(json_str)
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting JSON: {str(e)}")
            return None
    
    def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a web search using the LLM's browsing capabilities.
        This is a placeholder for a real implementation that would use the
        browsing capabilities of models like GPT-4 with browsing.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        # This is a mock implementation. In a real system, you would use
        # a model with browsing capabilities or integrate with a search API.
        system_message = {
            "role": "system",
            "content": (
                "You are a web search assistant. Your task is to simulate the results "
                "of a web search for the user's query. Return relevant information "
                "that would likely be found through such a search."
            )
        }
        
        user_message = {
            "role": "user",
            "content": f"Please perform a web search for: {query}"
        }
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[system_message, user_message],
                max_tokens=1000,
                temperature=0.7,
            )
            
            # In a real implementation, this would parse and structure actual search results
            mock_results = [
                {
                    "title": "Search Result 1",
                    "content": response.choices[0].message.content,
                    "url": "https://example.com/search-result-1"
                }
            ]
            
            return mock_results
        
        except Exception as e:
            self.logger.error(f"Error performing web search: {str(e)}")
            return []
