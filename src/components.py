from dotenv import load_dotenv
from llama_index.core import SQLDatabase, VectorStoreIndex
import openai
from qdrant_client import QdrantClient, AsyncQdrantClient
from sqlalchemy import create_engine
from llama_index.core.llms import ChatResponse
from llama_index.core.prompts.default_prompts import (
    PromptTemplate,
    DEFAULT_TEXT_TO_SQL_PROMPT,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.query_pipeline import FnComponent
from llama_index.core.retrievers import SQLRetriever
from llama_index.llms.openai import OpenAI
import os
from pathlib import Path

from utils.models import TableInfo

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Qdrant client
qdrant_client = QdrantClient("localhost", prefer_grpc=True)
qdrant_aclient = AsyncQdrantClient("localhost", prefer_grpc=True)

# Initialize vector store
vector_store = QdrantVectorStore(
    client=qdrant_client, aclient=qdrant_aclient, collection_name="demo_index"
)

# Initialize SQLite database
engine = create_engine("sqlite:///demo_database.db", future=True, echo=True)
sql_database = SQLDatabase(engine)

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)


def load_table_infos(tableinfo_dir: Path) -> dict[str, TableInfo]:
    table_infos = {}
    for file in tableinfo_dir.glob("*_*.json"):
        with open(file, "r") as f:
            table_info = TableInfo.model_validate_json(f.read())
            table_infos[table_info.table_name] = table_info
    return table_infos


tableinfo_dir = Path(os.getenv("TABLEINFO_DIR"))
table_infos = load_table_infos(tableinfo_dir)

table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
    for t in table_infos.values()
]  # add a SQLTableSchema for each table
vector_index = VectorStoreIndex.from_vector_store(vector_store)

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    vector_index,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)


def get_table_context_and_rows_str(
    query_str: str, table_schema_objs: list[SQLTableSchema]
):
    print(f"Query: {query_str}")
    context_strs = []
    for table_schema_obj in table_schema_objs:
        table_info = sql_database.get_single_table_info(table_schema_obj.table_name)
        if table_schema_obj.context_str:
            table_info += f" The table description is: {table_schema_obj.context_str}"

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="table_name", operator=FilterOperator.EQ, value=table_schema_obj.table_name),
            ]
        )

        # Retrieve relevant rows using vector search
        vector_retriever = vector_index.as_retriever(
            similarity_top_k=5,
            filters=filters,
        )
        relevant_nodes = vector_retriever.retrieve(query_str)
        if relevant_nodes:
            table_row_context = "\nHere are some relevant example rows (values in the same order as columns above)\n"
            for node in relevant_nodes:
                table_row_context += str(node.get_content()) + "\n"
            table_info += table_row_context

        context_strs.append(table_info)
    return "\n\n".join(context_strs)


table_parser_component = FnComponent(fn=get_table_context_and_rows_str)


def parse_response_to_sql(response: ChatResponse) -> str:
    """Parse response to SQL."""
    response = response.message.content
    print(f"Response: {response}")
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        
        response = response[sql_query_start:]
        response = response.removeprefix("SQLQuery:")
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    return response.strip().strip("```").strip()


sql_parser_component = FnComponent(fn=parse_response_to_sql)

text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(dialect=engine.dialect.name)

sql_retriever = SQLRetriever(sql_database)

response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL Response: {context_str}\n"
    "Response: "
)
response_synthesis_prompt = PromptTemplate(
    response_synthesis_prompt_str,
)
