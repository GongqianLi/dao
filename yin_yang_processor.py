import pandas as pd

import json
from typing import Callable, Dict, List, Any, Optional

from yin import YinAgent
from yang import YangAgent
from utils import get_logger

class YinYangProcessor:
    """
    Main processor for the Yin-Yang row-wise enrichment system.
    Coordinates the Yin and Yang agents to enrich data row by row.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        max_retries: int = 3,
        log_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
        output_file: Optional[str] = None,
    ):
        """
        Initialize the Yin-Yang processor.
        
        Args:
            model_name: Name of the OpenAI model to use
            max_retries: Maximum number of retries per row
            log_callback: Callback function for logging messages
            progress_callback: Callback function for updating progress
            output_file: Path to output file for streaming results (optional)
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.output_file = output_file
        self.logger = get_logger("YinYangProcessor")
        
        # Initialize agents
        self.yin = YinAgent(model_name)
        self.yang = YangAgent(model_name)
    
    def log(self, message_type: str, content: str):
        """Log a message using the callback if available."""
        self.logger.info(f"{message_type}: {content}")
        if self.log_callback:
            self.log_callback(message_type, content)
    
    def update_progress(self, current_row: int):
        """Update progress using the callback if available."""
        if self.progress_callback:
            self.progress_callback(current_row)
    
    def process_data(self, df: pd.DataFrame, user_command: str) -> pd.DataFrame:
        """
        Process the data frame row by row using the Yin-Yang architecture.
        
        Args:
            df: Input data frame
            user_command: Natural language command from the user
            
        Returns:
            Enriched data frame
        """
        # Create a copy of the input data frame to avoid modifying the original
        result_df = df.copy()
        
        # Add an ai_decision column to track validation status of each row
        result_df["ai_decision"] = ""
        
        # Log the start of processing
        total_rows = len(df)
        self.log("system", f"Starting to process {total_rows} rows")
        self.log("system", f"User command: {user_command}")
        
        # Initialize Yin with the user command
        self.log("system", "Initializing Yin agent with user command")
        self.yin.initialize_with_command(user_command)
        
        # Set up streaming output if an output file is specified
        if self.output_file:
            self.log("system", f"Streaming results to {self.output_file}")
            # Create the file with headers
            result_df.head(0).to_csv(self.output_file, index=False, mode='w')
        
        # Process each row
        for index, row in df.iterrows():
            current_row = index + 1
            self.update_progress(current_row)
            
            self.log("system", f"Processing row {current_row}/{total_rows}")
            
            # Convert row to a dictionary
            row_dict = row.to_dict()
            
            # Initialize retry counter and state tracking
            retry_count = 0
            success = False
            had_error = False  # To track if any errors occurred during processing
            
            while not success and retry_count < self.max_retries:
                try:
                    if retry_count > 0:
                        self.log("system", f"Retry {retry_count}/{self.max_retries} for row {current_row}")
                    
                    # Step 1: Yin analyzes the row and creates a context
                    self.log("yin", f"Analyzing row {current_row} and building context")
                    yin_context = self.yin.build_row_context(row_dict)
                    
                    # Step 2: Yin formulates a task for Yang
                    self.log("yin", "Formulating task for Yang")
                    yang_task = self.yin.formulate_yang_task(yin_context)
                    
                    # Step 3: Yang processes the task
                    self.log("yang", "Processing task")
                    yang_result = self.yang.process_task(yang_task)
                    
                    # Log Yang's response (in a simplified form)
                    self.log("yang", f"Generated enrichment data: {json.dumps(yang_result, indent=2)}")
                    
                    # Step 4: Yin validates Yang's result
                    self.log("yin", "Validating Yang's response")
                    validation_result, validation_message = self.yin.validate_yang_response(yang_result, yin_context)
                    
                    # Add new columns to the result dataframe regardless of validation outcome
                    # This ensures enriched fields are added even for invalid results
                    for key, value in yang_result.items():
                        result_df.loc[index, key] = str(value)  # Convert all values to string for consistency
                    
                    if validation_result:
                        # If valid, update ai_decision status
                        self.log("yin", f"Validation successful: {validation_message}")
                        
                        # Mark this row as valid
                        result_df.loc[index, "ai_decision"] = "valid"
                        
                        # Set success flag and immediately break out of the retry loop
                        success = True
                        break  # Skip any remaining retries since we have a valid result
                    else:
                        # If invalid, log the reason and retry
                        self.log("yin", f"Validation failed: {validation_message}")
                        
                        # Mark this attempt as invalid in ai_decision
                        # May be overwritten by future successful attempts
                        result_df.loc[index, "ai_decision"] = "invalid"
                        
                        retry_count += 1
                        
                        # If this is the last retry, log that all attempts failed
                        if retry_count >= self.max_retries:
                            self.log("yin", f"All validation attempts failed for row {current_row}")
                
                except Exception as e:
                    error_message = str(e)
                    self.log("error", f"Error processing row {current_row}: {error_message}")
                    had_error = True
                    retry_count += 1
                    
                    # If this is the last retry, prepare to mark as error
                    if retry_count >= self.max_retries:
                        self.log("error", f"All processing attempts failed with errors for row {current_row}")
            
            if not success:
                # Detailed log message about the final status
                if had_error:
                    # If any error occurred during processing, mark as "error"
                    result_df.loc[index, "ai_decision"] = "error"
                    self.log("system", f"Row {current_row}: Final status 'error' - Processing exceptions prevented completion")
                else:
                    # If we just couldn't get a valid result (validation failures), mark as "invalid"
                    result_df.loc[index, "ai_decision"] = "invalid"
                    self.log("system", f"Row {current_row}: Final status 'invalid' - Validation criteria not met after {self.max_retries} attempts")
            
            # Stream the processed row to the output file if specified
            if self.output_file:
                # Extract just the current row and append it to the output file
                row_df = result_df.iloc[[index]]
                row_df.to_csv(self.output_file, mode='a', header=False, index=False)
                self.log("system", f"Row {current_row} written to output file")
        
        self.log("system", f"Processing complete. Enriched {total_rows} rows.")
        return result_df
