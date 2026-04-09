import os
from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM

# Import custom refactored direct tools (non-LangChain)
from rag_tool import retrieve_agri_info
from sql_tool import query_agri_database

def create_agri_crew(user_query: str):
    if 'NVIDIA_API_KEY' not in os.environ:
        raise ValueError("NVIDIA_API_KEY is not set. Please set it in your .env file.")
        
    # Using the 'openai/' prefix in CrewAI's LLM class tells LiteLLM to use 
    # the standard OpenAI protocol against the custom Nvidia NIM base_url.
    # This avoids the '404' routing errors.
    agri_llm = LLM(
        model="openai/meta/llama-3.1-70b-instruct",
        api_key=os.environ["NVIDIA_API_KEY"],
        base_url="https://integrate.api.nvidia.com/v1"
    )

    # Define the Agricultural Intelligence Agent
    agri_agent = Agent(
        role="Agricultural Decision Intelligence Specialist",
        goal="Provide accurate, contextual, and data-driven answers to agriculture queries.",
        backstory="""You are an expert in agriculture and data analysis. You have access to a vast 
        knowledge base of agricultural documents and a structured database containing crop records. 
        Your goal is to decide whether to search the knowledge base or query the database (or both) 
        to provide the most helpful response.""",
        verbose=True,
        allow_delegation=False,
        tools=[retrieve_agri_info, query_agri_database],
        llm=agri_llm
    )

    analysis_task = Task(
        description=f"""
        Analyze the following agriculture-related query: '{user_query}'
        
        1. Determine if the user is asking for general knowledge (use RAG retrieval) or structured data (use SQL query tool).
        2. Execute the appropriate tool(s).
        3. Formulate a final intelligent response.
        """,
        expected_output="A well-formatted, detailed, and data-backed response.",
        agent=agri_agent
    )

    crew = Crew(
        agents=[agri_agent],
        tasks=[analysis_task],
        process=Process.sequential,
        verbose=True
    )

    return crew

def run_agri_agent(query: str):
    crew = create_agri_crew(query)
    result = crew.kickoff()
    return str(result)

if __name__ == "__main__":
    test_query = "Compare average rainfall across different crops."
    res = run_agri_agent(test_query)
    print("\nFINAL RESULT:")
    print(res)
