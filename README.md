# Yin-Yang Row-wise LLM Enrichment Web App

A local AI-powered data enrichment tool that processes structured data (CSV/Excel) row by row, using dual LLM agents ("Yin" and "Yang") to enrich each row with additional attributes based on natural language instructions.

## Features

- Upload CSV or Excel files
- Input natural language commands for data enrichment
- Dual LLM architecture:
  - Yin (Planner/Validator): Processes commands, validates results
  - Yang (Executor/Retriever): Performs web searches, generates enrichment data
- Row-by-row processing with fallback mechanisms
- Integration with external APIs for enhanced data quality
- Interactive logs showing the enrichment process
- Download enriched data files

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Upload your data file (CSV or Excel)
3. Enter your natural language enrichment command
4. Click "Start Enrichment" to begin the process
5. Monitor the progress in the log panel
6. Download the enriched file when processing is complete

## Architecture

The application follows a Yin-Yang dual LLM architecture:

- **Yin Agent**: Controller/planner that understands the task, manages row processing, and validates results
- **Yang Agent**: Executor that performs web searches and data enrichment based on Yin's instructions

The system processes each row as an individual task, allowing for precise, focused enrichment with validation and retry mechanisms.
