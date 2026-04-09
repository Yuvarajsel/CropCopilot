import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from crewai.tools import tool

# Load environment variables
load_dotenv()

CHROMA_PERSIST_DIR = "data/chroma_db_native"
DATA_FILE = "data/rag_documents.json"

def get_nvidia_client():
    if 'NVIDIA_API_KEY' not in os.environ:
        raise ValueError("NVIDIA_API_KEY is not set. Please set it in your .env file.")
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"]
    )

def get_embeddings(texts):
    client = get_nvidia_client()
    # Using the nvidia embedding model directly
    response = client.embeddings.create(
        input=texts,
        model="nvidia/nv-embedqa-e5-v5",
        encoding_format="float",
        extra_body={"input_type": "query" if len(texts) == 1 else "passage"}
    )
    return [e.embedding for e in response.data]

class NvidiaEmbeddingFunction:
    def __call__(self, input):
        return get_embeddings(input)

def setup_vector_store():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    # Check if collection already exists with documents
    collection_name = "agri_knowledge"
    
    try:
        # If it exists, we just return it
        collection = client.get_collection(name=collection_name, embedding_function=NvidiaEmbeddingFunction())
        if collection.count() > 0:
            print(f"Collection '{collection_name}' already exists with {collection.count()} documents.")
            return collection
    except:
        pass

    # Otherwise, create and populate
    print(f"Creating new collection '{collection_name}'...")
    collection = client.create_collection(name=collection_name, embedding_function=NvidiaEmbeddingFunction())
    
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Run data_ingestion.py first.")
        
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        docs_data = json.load(f)
        
    # Prepare data for Chroma
    ids = [str(i) for i in range(len(docs_data))]
    documents = [d["text"] for d in docs_data]
    metadatas = [d["metadata"] for d in docs_data]
    
    # Chroma handles batching if we use the collection.add method
    # For large datasets, we should batch manually, but 5000 is manageable.
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end = i + batch_size
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"Added batch {i} to {min(end, len(documents))}")
        
    print("Vector Store successfully populated!")
    return collection

@tool("Agriculture Knowledge Retrieval Tool")
def retrieve_agri_info(query: str) -> str:
    """
    Search the agriculture dataset for contextual information based on natural language queries.
    Useful for answering questions about crop requirements, best seasons, ideal soil conditions, and recommended agriculture practices.
    Input: A specific question or query (e.g., 'Which crops are suitable for sandy soil?').
    Output: Retrieved snippets of text closely matching the query context.
    """
    collection = setup_vector_store()
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    
    docs = results['documents'][0]
    if not docs:
        return "No relevant agriculture information found in the knowledge base."
        
    context_chunks = [f"- {doc}" for doc in docs]
    return "\n".join(context_chunks)

if __name__ == "__main__":
    print("Testing Native Vector Store Initialization...")
    retrieve_agri_info("Which crops are suitable for sandy soil conditions?")
