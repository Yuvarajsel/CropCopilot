import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import our CrewAI agent
from agri_agent import run_agri_agent

app = FastAPI(title="Agriculture Intelligence System")

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
        # Run the crewai agent
        result = run_agri_agent(user_query)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("Starting Agriculture Intelligence System on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
