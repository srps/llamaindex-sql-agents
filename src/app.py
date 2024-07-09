from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
import openai
from pydantic import BaseModel
from typing import Optional
import uvicorn
import uuid
from observability import Observability
from conversation import ConversationMemory, ChatMessage, MessageRole   
from query_pipeline import query_pipeline

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

observability = Observability()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Enable observability
    observability.enable()
    yield
    # Disable observability
    observability.disable()

app = FastAPI(lifespan=lifespan)

conversation_memory = ConversationMemory()

class Query(BaseModel):
    text: str
    conversation_id: Optional[str] = None

class Response(BaseModel):
    answer: str
    conversation_id: str

@app.post("/query", response_model=Response)
async def process_query(query: Query, background_tasks: BackgroundTasks):
    try:
        # Retrieve conversation history if conversation_id is provided
        conversation_id = query.conversation_id or str(uuid.uuid4())
        
        history = conversation_memory.get_conversation_history(conversation_id)
        context = "\n".join([f"{turn.role}: {turn.content}" for turn in history])
        
        full_query = f"{context}\n\nHuman: {query.text}"
        
        # TODO: Modify the pipeline so it can handle queries with context
        response = query_pipeline.run_query(query.text)
        
        # Store the query and response in memory asynchronously
        background_tasks.add_task(conversation_memory.add_turn, conversation_id, ChatMessage(role=MessageRole.USER, content=query.text))
        background_tasks.add_task(conversation_memory.add_turn, conversation_id, ChatMessage(role=MessageRole.ASSISTANT, content=str(response)))
        
        return Response(answer=str(response), conversation_id=conversation_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    await conversation_memory.clear_conversation(conversation_id)
    return {"message": "Conversation cleared successfully"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
