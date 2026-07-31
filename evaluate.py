import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)
from rag_tool import retrieve_agri_info

# Direct OpenAI client for evaluation LLM calls (Nvidia NIM API)
def get_nvidia_client():
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"]
    )

def create_eval_dataset():
    questions = [
        "Which crops are suitable for sandy soil conditions?",
        "What is the best season for cultivating tomato?",
        "How much rainfall is needed for growing rice?"
    ]
    
    ground_truths = [
        ["Sandy soil is suitable for crops like watermelon, peanuts, and certain root vegetables."],
        ["Tomatoes grow best in warm seasons."],
        ["Rice cultivation requires substantial rainfall (e.g., 115-200 cm)."]
    ]
    
    contexts = []
    answers = []
    
    print("Generating responses for RAG evaluation dataset...")
    client = get_nvidia_client()
    
    for q in questions:
        # 1. Retrieve context
        ctx_text = retrieve_agri_info(q)
        contexts.append([ctx_text]) 
        
        # 2. Generate answer directly using direct LLM call
        response = client.chat.completions.create(
            model="meta/llama3-70b-instruct",
            messages=[
                {"role": "system", "content": "Answer the question based ONLY on the provided context."},
                {"role": "user", "content": f"CONTEXT:\n{ctx_text}\n\nQUESTION: {q}"}
            ]
        )
        answers.append(response.choices[0].message.content)
        
    data = {"question": questions, "contexts": contexts, "answer": answers, "ground_truths": ground_truths}
    return Dataset.from_dict(data)

def run_evaluation():
    if 'NVIDIA_API_KEY' not in os.environ:
        print("Error: NVIDIA_API_KEY not set.")
        return

    print("Building Evaluation Dataset...")
    eval_dataset = create_eval_dataset()
    
    print("\nRunning Ragas Evaluation (Faithfulness, Answer Relevancy, Context Precision)...")
    
    # Ragas usually wraps an LLM. We'll use the LangChain adapter just for the evaluator 
    # to avoid complex manual implementation of the metrics.
    from langchain_openai import ChatOpenAI
    eval_llm = ChatOpenAI(
        model="meta/llama3-70b-instruct",
        openai_api_key=os.environ["NVIDIA_API_KEY"],
        openai_api_base="https://integrate.api.nvidia.com/v1"
    )

    try:
        from ragas.llms import LangchainLLMWrapper
        ragas_llm = LangchainLLMWrapper(eval_llm)
        result = evaluate(eval_dataset, metrics=[faithfulness, answer_relevancy, context_precision], llm=ragas_llm)
    except:
        result = evaluate(eval_dataset, metrics=[faithfulness, answer_relevancy, context_precision], llm=eval_llm)
    
    df = result.to_pandas()
    print("\n--- Evaluation Results ---")
    print(df[['question', 'faithfulness', 'answer_relevancy', 'context_precision']])
    df.to_csv("data/ragas_evaluation_results.csv", index=False)

if __name__ == "__main__":
    run_evaluation()
