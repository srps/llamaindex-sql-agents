import os
import json
import re
from typing import Dict
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import Column, Engine, Integer, MetaData, String, Table, create_engine, text
from llama_index.core import VectorStoreIndex, Document, SQLDatabase, StorageContext, Settings, load_index_from_storage
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient
from llama_index.core.program import LLMTextCompletionProgram
import openai

from utils.models import TableInfo

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Qdrant client
qdrant_client = QdrantClient("localhost", prefer_grpc=True)

# Initialize vector store and embedding model
vector_store = QdrantVectorStore(client=qdrant_client, collection_name="demo_index")
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.embed_model = embed_model  

# Initialize LLM
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")

# Define prompt for table summary generation
prompt_str = """\
Give me a summary of the table with the following JSON format.

- The table name must be unique to the table and describe it while being concise. 
- Do NOT output a generic table name (e.g. table, my_table).

Do NOT make the table name one of the following: {exclude_table_name_list}

Ensure that the JSON is valid and can be parsed by Python's json.loads function.

Table:
{table_str}

Summary: """

# Initialize LLMTextCompletionProgram
program = LLMTextCompletionProgram.from_defaults(
    output_cls=TableInfo,
    llm=llm,
    prompt_template_str=prompt_str,
)

def process_files(data_dir: Path) -> list[pd.DataFrame]:
    dfs = []
    files = sorted([f for f in data_dir.glob("*")])
    for file in files:
        if file.suffix.lower() in ['.csv', '.xlsx']:
            print(f"Processing file: {file}")
            try:
                if file.suffix.lower() == '.csv':
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                dfs.append(df)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
    return dfs

def sanitize_column_name(col_name):
    # Remove special characters and replace spaces with underscores
    return re.sub(r"\W+", "_", col_name)


# Function to create a table from a DataFrame using SQLAlchemy
def create_table_from_dataframe(
    df: pd.DataFrame, table_name: str, engine: Engine, metadata_obj: MetaData
):
    # Sanitize column names
    sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=sanitized_columns)

    # Dynamically create columns based on DataFrame columns and data types
    columns = [
        Column(col, String if dtype == "object" else Integer)
        for col, dtype in zip(df.columns, df.dtypes)
    ]

    # Create a table with the defined columns
    table = Table(table_name, metadata_obj, *columns)

    # Create the table in the database
    metadata_obj.drop_all(engine)
    metadata_obj.create_all(engine)

    # Insert data from DataFrame into the table
    with engine.connect() as conn:
        for _, row in df.iterrows():
            insert_stmt = table.insert().values(**row.to_dict())
            conn.execute(insert_stmt)
        conn.commit()

def create_database(dfs: list[pd.DataFrame], table_infos: Dict[int, TableInfo], db_name: str = "demo_database.db"):
    engine = create_engine(f"sqlite:///{db_name}")
    metadata_obj = MetaData()
    for idx, df in enumerate(dfs): 
        table_info = table_infos[idx]
        print(f"Creating table: {table_info.table_name}")
        create_table_from_dataframe(df, table_info.table_name, engine, metadata_obj)
    return engine

def _get_tableinfo_with_index(idx: int, tableinfo_dir: str) -> TableInfo:
    results_gen = Path(tableinfo_dir).glob(f"{idx}_*")
    results_list = list(results_gen)
    if len(results_list) == 0:
        return None
    elif len(results_list) == 1:
        path = results_list[0]
        return TableInfo.model_validate_json(path.read_text())
    else:
        raise ValueError(
            f"More than one file matching index: {list(results_gen)}"
        )

def generate_table_infos(dfs: list[pd.DataFrame], tableinfo_dir: str) -> Dict[int, TableInfo]:
    table_names = set()
    table_infos = dict()
    for idx, df in enumerate(dfs):
        table_info = _get_tableinfo_with_index(idx, tableinfo_dir)
        if table_info is None:
            while True:
                df_str = df.head(10).to_csv()
                table_info = program(
                    table_str=df_str,
                    exclude_table_name_list=str(list(table_names)),
                )
                table_name = table_info.table_name
                print(f"Processed table: {table_name}")
                if table_name not in table_names:
                    table_names.add(table_name)
                    break
                else:
                    # try again
                    print(f"Table name {table_name} already exists, trying again.")
                    pass

            out_file = f"{tableinfo_dir}/{idx}_{table_name}.json"
            with open(out_file, "w") as f:
                json.dump(table_info.dict(), f)
        table_infos[idx] = table_info
    return table_infos

# def create_table_index(dfs: list[pd.DataFrame], table_infos: Dict[int, TableInfo]):
#     documents = []
#     for idx, df in enumerate(dfs):
#         table_info = table_infos[idx]
#         # Create a document for table summary
#         doc_text = f"Table {table_info.table_name}:\n{table_info.table_summary}\n\nColumns: {', '.join(df.columns)}"
#         documents.append(Document(text=doc_text, metadata={"table_name": table_info.table_name, "type": "summary"}))
#         print(f"Indexing table {table_info.table_name} summary")
#         print(df.head())
        
#         # Create documents for each row
#         for _, row in df.iterrows():
#             row_content = ", ".join(str(value) for value in row)
#             row_doc = Document(
#                 text=f"Row from {table_info.table_name}: {row_content}",
#                 metadata={"table_name": table_info.table_name, "type": "row", "row_content": row_content}
#             )
#             documents.append(row_doc)
#             print(f"Indexing table {table_info.table_name} row {row_content}")

#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#     index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
#     index.set_index_id("vector_index")
#     return index

def create_table_index(engine: Engine, table_index_dir: str) -> Dict[str, VectorStoreIndex]:
    
    sql_database = SQLDatabase(engine)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index_dict = dict()
    
    for table_name in sql_database.get_usable_table_names():
        print(f"Indexing rows in table: {table_name}")
        if not os.path.exists(f"{table_index_dir}/{table_name}"):
            # get all rows from table
            with engine.connect() as conn:
                cursor = conn.execute(text(f'SELECT * FROM "{table_name}"'))
                result = cursor.fetchall()
                row_tups = []
                for row in result:
                    row_tups.append(tuple(row))

            # index each row, put into vector store index
            nodes = [Document(
                text=str(t),
                metadata={"table_name": table_name, "type": "row", "row_content": str(t)}
            ) for t in row_tups]

            # put into vector store index (use OpenAIEmbeddings by default)
            index = VectorStoreIndex.from_documents(nodes, storage_context=storage_context)

            # save index
            index.set_index_id("vector_index")
            index.storage_context.persist(persist_dir=table_index_dir)
        else:
            # load index
            index = load_index_from_storage(
                storage_context, index_id="vector_index"
            )
        vector_index_dict[table_name] = index
    
    return vector_index_dict

def check_initialization():
    try:
        qdrant_client.get_collection("demo_index")
        return True
    except Exception:
        return False

def main():
    if check_initialization():
        print("Data already initialized. Skipping initialization process.")
        return

    # Process CSV/XLSX files
    data_dir = Path(os.getenv("DATA_DIR"))
    dfs = process_files(data_dir)

    # Generate table infos
    tableinfo_dir = Path(os.getenv("TABLEINFO_DIR"))
    tableinfo_dir.mkdir(exist_ok=True)
    table_infos = generate_table_infos(dfs, str(tableinfo_dir))

    # Create SQLite database
    engine = create_database(dfs, table_infos)
    

    # Create table index
    table_index_dir = Path(os.getenv("TABLE_INDEX_DIR"))
    table_index_dir.mkdir(exist_ok=True)
    table_index = create_table_index(engine, str(table_index_dir))

    print("Data initialization complete!")

if __name__ == "__main__":
    main()