import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our CrewAI agent & data ingestion
from agri_agent import run_agri_agent, run_agri_agent_async
from data_ingestion import ingest_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join("/tmp", "data") if os.environ.get("VERCEL") else os.path.join(BASE_DIR, "data"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(DATA_DIR, exist_ok=True)
    db_file = os.path.join(DATA_DIR, "agriculture.db")
    rag_file = os.path.join(DATA_DIR, "rag_documents.json")
    if not os.path.exists(db_file) or not os.path.exists(rag_file):
        print("Data files missing on startup. Running initial data ingestion...")
        try:
            ingest_data()
            print("Data ingestion completed successfully.")
        except Exception as e:
            print(f"Warning: Data ingestion during startup encountered an issue: {e}")
    else:
        print(f"Existing data files found in {DATA_DIR}.")
    yield

app = FastAPI(
    title="CropCopilot — Agricultural Intelligence System",
    description="AI-powered agricultural decision intelligence combining RAG, Text-to-SQL, and CrewAI orchestration.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for production deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the static directory exists
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount the static directory to serve HTML, CSS, JS
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
async def health_check():
    """Health check endpoint for cloud load balancers, Vercel, Render, Fly.io, and Docker probes."""
    db_file = os.path.join(DATA_DIR, "agriculture.db")
    rag_file = os.path.join(DATA_DIR, "rag_documents.json")
    has_db = os.path.exists(db_file)
    has_docs = os.path.exists(rag_file)
    has_key = bool(os.environ.get("NVIDIA_API_KEY"))
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "CropCopilot",
            "environment": {
                "sqlite_database_ready": has_db,
                "rag_documents_ready": has_docs,
                "nvidia_api_key_configured": has_key
            }
        },
        status_code=200
    )

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>CropCopilot API is running</h1>")

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
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting CropCopilot on http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port)

