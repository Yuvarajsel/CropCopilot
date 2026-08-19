import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import our CrewAI agent & data ingestion
from agri_agent import run_agri_agent, run_agri_agent_async
from data_ingestion import ingest_data

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/agriculture.db") or not os.path.exists("data/rag_documents.json"):
        print("Data files missing on startup. Running data ingestion...")
        ingest_data()
    yield

app = FastAPI(title="Agriculture Intelligence System", lifespan=lifespan)

# Ensure the static directory exists
os.makedirs("static", exist_ok=True)

# Mount the static directory to serve HTML, CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/query")
async def handle_query(request: QueryRequest):
    try:
        user_query = request.query
        print(f"Received query: {user_query}")
        # Run the crewai agent asynchronously
        result = await run_agri_agent_async(user_query)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Agriculture Intelligence System on http://0.0.0.0:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
