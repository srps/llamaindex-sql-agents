# LlamaIndex Query Pipeline Demo

This demo showcases an advanced text-to-SQL pipeline using LlamaIndex, FastAPI, and Streamlit.

## Prerequisites

- Python 3.8+
- Docker (for running Qdrant)
- OpenAI API key

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/llamaindex-query-pipeline-demo.git
   cd llamaindex-query-pipeline-demo
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up Qdrant:
   ```
   docker run -p 6333:6333,6334:6334 qdrant/qdrant
   ```

5. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

   You can also refedine the data directories in `.env`:
   ```
   TTABLEINFO_DIR=data/WikiTableQuestions_TableInfo
   TABLE_INDEX_DIR=data/WikiTableQuestions_TableIndex
   DATA_DIR=data/WikiTableQuestions/csv/200-csv
   ```

## Initializing the Demo

1. Place your CSV/XLSX files in the `data` directory.

2. Run the initialization script:
   ```
   python init_demo.py
   ```

   This script will process your data files, create a SQLite database, generate table summaries, and build the necessary indexes.

## Running the Demo

1. Start the FastAPI backend:
   ```
   python app.py
   ```

2. In a new terminal, start the Streamlit frontend:
   ```
   streamlit run frontend.py
   ```

3. Open your browser and navigate to `http://localhost:8501` to interact with the demo.

## Project Structure

- `init_data.py`: Initialization script for processing data and building indexes
- `app.py`: FastAPI backend
- `frontend.py`: Streamlit frontend
- `query_pipeline.py`: LlamaIndex query pipeline implementation
- `data/`: Directory for input CSV/XLSX files
- `table_index_dir/`: Directory for storing the table index (created during initialization)

## Troubleshooting

If you encounter any issues, please check the following:

1. Ensure all dependencies are installed correctly.
2. Verify that Qdrant is running and accessible.
3. Check that your OpenAI API key is set correctly in the `.env` file.
4. Make sure your input data files are in the correct format and located in the `data` directory.

For further assistance, please open an issue on the GitHub repository.