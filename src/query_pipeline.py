from functools import lru_cache
from typing import Any
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.query_pipeline import FnComponent, QueryPipeline, InputComponent
from llama_index.core.prompts.default_prompts import PromptTemplate
from llama_index.core.retrievers import SQLRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
import os

# Import your existing components
from components import (
    table_parser_component,
    text2sql_prompt,
    llm,
    sql_parser_component,
    sql_retriever,
    response_synthesis_prompt,
    vector_index,
    obj_retriever,
)


class OptimizedQueryPipeline:
    def __init__(
        self,
        table_parser_component: FnComponent,
        text2sql_prompt: PromptTemplate,
        llm: Any,
        sql_parser_component: FnComponent,
        sql_retriever: SQLRetriever,
        response_synthesis_prompt: PromptTemplate,
        vector_index: VectorStoreIndex,
        obj_retriever: SQLRetriever,
    ):
        self.table_parser_component = table_parser_component
        self.text2sql_prompt = text2sql_prompt
        self.llm = llm
        self.sql_parser_component = sql_parser_component
        self.sql_retriever = sql_retriever
        self.response_synthesis_prompt = response_synthesis_prompt
        self.vector_index = vector_index
        self.obj_retriever = obj_retriever
        self.obj_index = vector_index
        self.obj_retriever = obj_retriever

        self.qp = self._create_pipeline()

    def _create_pipeline(self):
        qp = QueryPipeline(
            modules={
                "input": InputComponent(),
                "table_retriever": self.obj_retriever,
                "table_output_parser": self.table_parser_component,
                "text2sql_prompt": self.text2sql_prompt,
                "text2sql_llm": self.llm,
                "sql_output_parser": self.sql_parser_component,
                "sql_retriever": self.sql_retriever,
                "response_synthesis_prompt": self.response_synthesis_prompt,
                "response_synthesis_llm": self.llm,
            },
            verbose=True,
        )

        qp.add_link("input", "table_retriever")
        qp.add_link("input", "table_output_parser", dest_key="query_str")
        qp.add_link(
            "table_retriever", "table_output_parser", dest_key="table_schema_objs"
        )
        qp.add_link("input", "text2sql_prompt", dest_key="query_str")
        qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")
        qp.add_chain(
            ["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"]
        )
        qp.add_link(
            "sql_output_parser", "response_synthesis_prompt", dest_key="sql_query"
        )
        qp.add_link(
            "sql_retriever", "response_synthesis_prompt", dest_key="context_str"
        )
        qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
        qp.add_link("response_synthesis_prompt", "response_synthesis_llm")

        return qp

    @lru_cache(maxsize=100)
    def run_query(self, query: str):
        try:
            return self.qp.run(query=query)
        except Exception as e:
            # Log the error and return a user-friendly message
            print(f"Error processing query: {str(e)}")
            return "I'm sorry, but I encountered an error while processing your query. Please try again or rephrase your question."

    def save_state(self):
        self.obj_index.storage_context.persist(persist_dir="obj_index_storage")

    def load_state(self):
        self.obj_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="obj_index_storage")
        )
        self.obj_retriever = self.obj_index.as_retriever(similarity_top_k=3)
        self.qp.update_module("table_retriever", self.obj_retriever)


query_pipeline = OptimizedQueryPipeline(
    table_parser_component,
    text2sql_prompt,
    llm,
    sql_parser_component,
    sql_retriever,
    response_synthesis_prompt,
    vector_index,
    obj_retriever,
)
