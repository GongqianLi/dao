import streamlit as st
import pandas as pd
import os
import tempfile
import json
from datetime import datetime
import time
from dotenv import load_dotenv

from yin_yang_processor import YinYangProcessor

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Yin-Yang Row-wise Enrichment",
    page_icon="☯️",
    layout="wide",
)

def main():
    # Title and description
    st.title("☯️ Yin-Yang Row-wise LLM Enrichment")
    st.markdown(
        "Upload a CSV or Excel file and provide a natural language command to enrich each row with additional attributes."
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file", 
        type=["csv", "xlsx", "xls"],
    )
    
    # Command input
    user_command = st.text_area(
        "Enter your enrichment command",
        placeholder="Example: For each customer, add their gender, most likely nationality, and famous people with the same name.",
        height=100,
    )
    
    # Advanced options with expander
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            max_retries = st.number_input("Max retries per row", min_value=1, max_value=5, value=3)
        with col2:
            model_name = st.selectbox(
                "LLM Model",
                ["gpt-4o", "gpt-3.5-turbo","gpt-4o-search-preview","gpt-4o-mini-search-preview"],
                index=0,
            )
    
    # Start button
    start_button = st.button("Start Enrichment", type="primary", disabled=not (uploaded_file and user_command))
    
    # Initialize session state
    if "log_messages" not in st.session_state:
        st.session_state.log_messages = []
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "current_row" not in st.session_state:
        st.session_state.current_row = 0
    if "total_rows" not in st.session_state:
        st.session_state.total_rows = 0
    
    # Create columns for log and progress
    log_col, progress_col = st.columns([2, 1])
    
    # Process data when the Start button is clicked
    if start_button:
        with st.spinner("Processing..."):
            st.session_state.log_messages = []
            st.session_state.is_processing = True
            st.session_state.current_row = 0
            
            # Read the uploaded file
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            st.session_state.total_rows = len(df)
            
            # Create a temporary directory for output files
            temp_dir = tempfile.mkdtemp()
            
            # Generate a unique output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if uploaded_file and uploaded_file.name:
                base_filename = os.path.splitext(uploaded_file.name)[0]
                output_filename = f"{base_filename}_enriched_{timestamp}.csv"
            else:
                output_filename = f"enriched_data_{timestamp}.csv"
            
            output_filepath = os.path.join(temp_dir, output_filename)
            st.session_state.output_filepath = output_filepath  # Store for later access
            
            # Initialize the YinYang processor with streaming output enabled
            processor = YinYangProcessor(
                model_name=model_name,
                max_retries=max_retries,
                log_callback=add_log_message,
                progress_callback=update_progress,
                output_file=output_filepath  # Enable streaming output
            )
            
            # Process the data
            result_df = processor.process_data(df, user_command)
            
            # Save the processed data
            st.session_state.processed_data = result_df
            st.session_state.is_processing = False
            
            # Show success message
            st.success(f"✅ Processing complete! Enriched {len(result_df)} rows. Data has been automatically saved to {output_filepath}")
    
    # Display log messages
    with log_col:
        st.subheader("Processing Log")
        log_container = st.container(height=400)
        with log_container:
            for msg in st.session_state.log_messages:
                if msg.get("type") == "yin":
                    st.markdown(f"**🔵 Yin:** {msg.get('content')}")
                elif msg.get("type") == "yang":
                    st.markdown(f"**🟡 Yang:** {msg.get('content')}")
                elif msg.get("type") == "system":
                    st.markdown(f"**⚙️ System:** {msg.get('content')}")
                elif msg.get("type") == "error":
                    st.markdown(f"**❌ Error:** {msg.get('content')}")
    
    # Display progress
    with progress_col:
        st.subheader("Progress")
        if st.session_state.is_processing:
            progress = st.session_state.current_row / max(1, st.session_state.total_rows)
            st.progress(progress)
            st.markdown(f"**Processing row:** {st.session_state.current_row} / {st.session_state.total_rows}")
        
        # Display download button if processing is complete and data is available
        if st.session_state.processed_data is not None and not st.session_state.is_processing:
            # Prepare the file for download
            if uploaded_file and uploaded_file.name:
                filename, ext = os.path.splitext(uploaded_file.name)
                if ext.lower() in ['.xlsx', '.xls']:
                    # For Excel files, offer both Excel and CSV downloads
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        output_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                        with pd.ExcelWriter(output_buffer.name) as writer:
                            st.session_state.processed_data.to_excel(writer, index=False)
                        
                        with open(output_buffer.name, "rb") as f:
                            st.download_button(
                                label="📥 Download as Excel",
                                data=f,
                                file_name=f"{filename}_enriched.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                    
                    with col2:
                        csv_data = st.session_state.processed_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv_data,
                            file_name=f"{filename}_enriched.csv",
                            mime="text/csv",
                        )
                else:
                    # For CSV files, just offer CSV download
                    csv_data = st.session_state.processed_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Enriched File",
                        data=csv_data,
                        file_name=f"{filename}_enriched.csv",
                        mime="text/csv",
                    )
                
                # Add note about auto-saved data
                if hasattr(st.session_state, 'output_filepath') and os.path.exists(st.session_state.output_filepath):
                    st.info(f"💾 Data was also saved automatically to: {st.session_state.output_filepath}")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(st.session_state.processed_data.head(5), use_container_width=True)

def add_log_message(message_type, content):
    """Add a message to the log."""
    st.session_state.log_messages.append({
        "type": message_type,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

def update_progress(current_row):
    """Update the progress state."""
    st.session_state.current_row = current_row

if __name__ == "__main__":
    main()
