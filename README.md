# Yin-Yang Dual-LLM Architecture

A locally deployed Python web application that implements a dual-agent LLM collaboration architecture for data analysis.

## Architecture Overview

This system adopts a dual-agent LLM collaboration architecture, with a Flask-based web interface that allows users to upload data files and issue natural language commands.

### Key Components

- **Web Interface (Flask App)**: A simple web UI for uploading CSV/Excel files and inputting instructions.
- **Yin Node (LLM Controller Agent)**: Plans tasks, keeps track of conversation context, and validates results.
- **Yang Node (LLM Execution Agent)**: Executes data analysis tasks assigned by Yin.
- **Data Processing Module**: A collection of backend tools and functions for data manipulation.

## Workflow

1. **User Input**: Upload a CSV/Excel file and enter a natural language command
2. **Yin Understanding**: Yin reads the command and generates instructions for Yang
3. **Yang Execution**: Yang chooses and performs the appropriate data operations
4. **Yin Verification**: Yin checks if Yang's output satisfies the request
5. **Looping**: This conversation can iterate up to 10 times for refinement
6. **Result Delivery**: The final result is saved to file and returned for download

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. Clone this repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root directory and add your OpenAI API key:
   ```
   # .env file
   OPENAI_API_KEY=your-api-key-here
   ```
   
   Note: A `.env.example` file is provided as a template. You can copy this file:
   ```
   # On Windows
   copy .env.example .env
   
   # On Linux/Mac
   cp .env.example .env
   ```
   
   Then edit the `.env` file to add your actual API key.

## Running the Application

Start the Flask server:
```
python app.py
```

The web interface will be available at http://localhost:5000

## Usage

1. Open the web interface in your browser
2. Upload a CSV or Excel file
3. Enter your data processing request in natural language
4. Click "Process Data" and wait for the result
5. Download the result file when processing is complete

## Examples of Commands

- "Find the top 5 products by sales and create a bar chart"
- "Group by region and calculate the average and sum of sales for each region"
- "Clean the data by removing duplicates and filling missing values with the mean"
- "Identify outliers in the 'Price' column and create a new dataset without them"

## Project Structure

```
dao/
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates
│   └── index.html        # Web interface
└── dao/                  # Core package
    ├── __init__.py
    ├── yin_yang.py       # Coordinates the Yin-Yang conversation
    ├── yin.py            # Controller/planner agent
    ├── yang.py           # Executor agent with data tools
    └── utils.py          # Utility functions
```

## License

[MIT License](LICENSE)
