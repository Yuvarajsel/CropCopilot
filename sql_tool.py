import os
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI
from crewai.tools import tool

# Load environment variables
load_dotenv()

DB_PATH = "data/agriculture.db"

def get_nvidia_client():
    if 'NVIDIA_API_KEY' not in os.environ:
        raise ValueError("NVIDIA_API_KEY is not set. Please set it in your .env file.")
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"]
    )

def generate_sql(question: str) -> str:
    client = get_nvidia_client()
    
    # Simple schema description
    schema_info = """
    Table Name: 'agriculture_data'
    Columns: N, P, K, temperature, humidity, ph, rainfall, label
    - 'label' is the crop type (e.g., 'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee').
    - environmental factors are numeric.
    """
    
    system_prompt = f"""
    You are a professional SQL expert for SQLite.
    Your task is to convert the following natural language agriculture query into a single, valid SQLite SQL query that can be executed on the 'agriculture_data' table.
    
    DATABASE SCHEMA:
    {schema_info}
    
    Rules:
    - Return ONLY the raw SQL code. No markdown, no commentary, no triple backticks.
    - Be careful with column names. Use double quotes if they contain spaces (but these don't).
    - If the user asks for 'highest yield' but there is no 'yield' column, assume they want the count or frequency of records for that crop, or the highest individual values of environmental parameters as proxies. Wait, in this dataset label=crop, there's no yield column. If user asks for average usage, use 'avg(N)', 'avg(P)', etc.
    """
    
    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"QUERY: {question}\nSQL:"}
        ],
        temperature=0.2,
        max_tokens=256
    )
    
    sql = response.choices[0].message.content.strip()
    return sql

@tool("Agriculture SQLite Database Query Tool")
def query_agri_database(query: str) -> str:
    """
    Executes a natural language query against the structured agriculture SQLite database.
    Useful for questions that require data aggregation, comparisons, or specific filtering from tabular data
    (e.g., 'Show crops with the highest average nitrogen (N) usage', 'Compare average rainfall across different crops').
    Input: A natural language data question.
    Output: A string containing the executed SQL query and the results.
    """
    try:
        # 1. Generate SQL
        sql_query = generate_sql(query)
        
        # 2. Execute SQL directly using sqlite3
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        conn.close()
        
        if not rows:
            return f"Executed SQL: {sql_query}\nResult: No data found for this query."
            
        return f"Executed SQL: {sql_query}\nResult: {str(rows)}\nColumns: {str(column_names)}"
    except Exception as e:
        return f"Error executing database query: {str(e)}"

if __name__ == "__main__":
    print("Testing Direct SQL Tool...")
    print("Query: Show me the first 3 crops.")
    print(query_agri_database("Show me the first 3 records from the database."))
