import sys
import os

# Add parent directory to python path so modules (agri_agent, rag_tool, sql_tool, etc.) can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

# Vercel serverless function entrypoint
