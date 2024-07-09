import os
from typing import List
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.llms import (
    ChatMessage as LlamaIndexChatMessage,
    MessageRole as LlamaIndexMessageRole,
)
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
import openai
from qdrant_client import QdrantClient
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

ChatMessage = LlamaIndexChatMessage
MessageRole = LlamaIndexMessageRole


class ConversationMemory:
    def __init__(
        self,
        collection_name: str = "conversation_memory",
        host: str = "localhost",
        embed_model: EmbedType = OpenAIEmbedding(
            model=OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL
        ),
    ):
        self.qdrant_client = QdrantClient(host, prefer_grpc=True)
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name=collection_name
        )
        self.embed_model = embed_model or OpenAIEmbedding(
            model=OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store, embed_model=self.embed_model
        )

    def add_turn(self, conversation_id: str, turn: ChatMessage):
        doc = Document(
            text=turn.content,
            metadata={"conversation_id": conversation_id, "role": turn.role},
        )
        self.index.insert(doc)

    def get_conversation_history(
        self, conversation_id: str, limit: int = 5
    ) -> List[ChatMessage]:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="conversation_id",
                    operator=FilterOperator.EQ,
                    value=conversation_id,
                ),
            ]
        )
        try:
            retriever = self.index.as_retriever(similarity_top_k=limit, filters=filters)
            nodes = retriever.retrieve(conversation_id)
            return [ChatMessage(role=node.metadata["role"], content=node.text) 
                    for node in nodes]
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []

    def clear_conversation(self, conversation_id: str):
        try:
        # Remove all documents with the given conversation_id
            self.qdrant_client.delete(
                collection_name=self.vector_store.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {
                                "key": "metadata.conversation_id",
                                "match": {"value": conversation_id},
                            }
                        ]
                    }
                },
            )
        except Exception as e:
            print(f"Error clearing conversation: {e}")
