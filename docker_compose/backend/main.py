from fastapi import FastAPI
from pydantic import BaseModel
from model import load_llm


# Initialize FastAPI app
app = FastAPI(
    title="LLM API", description="FastAPI server for LLM inference", version="1.0"
)

# Loading model
model = load_llm()


# Define request schema
class QueryRequest(BaseModel):
    query: str
    max_length: int = 200


@app.post("/generate")
async def generate_text(request: QueryRequest):
    """Generate text based on user query using the LLM"""

    response = model(request.query, max_new_tokens=512)

    return response


@app.get("/")
async def root():
    return {"message": "Welcome to the LLM API!"}


# Example usage
prompt = "Explain the importance of AI in healthcare."
