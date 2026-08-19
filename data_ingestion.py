import pandas as pd
from datasets import load_dataset
import os
import sqlite3
import json

def ingest_data():
    os.makedirs("data", exist_ok=True)
    
    print("Downloading dataset Mahesh2841/Agriculture...")
    dataset = load_dataset("Mahesh2841/Agriculture")
    
    # Typically tabular data is in the 'train' split
    if 'train' in dataset:
        df = dataset['train'].to_pandas()
    else:
        # fallback to the first split if 'train' doesn't exist
        split_name = list(dataset.keys())[0]
        df = dataset[split_name].to_pandas()
        
    print(f"Dataset loaded from split. Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print(df.head(2))
    
    # 1. Save to SQLite for Text-to-SQL
    db_path = "data/agriculture.db"
    conn = sqlite3.connect(db_path)
    # Using 'agriculture_data' as table name
    df.to_sql("agriculture_data", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved tabular data to SQLite: {db_path} (table: agriculture_data)")

    # 2. Convert to Text Documents for RAG
    documents = []
    # If the dataset has typical crop columns (N, P, K, temperature, humidity, ph, rainfall, label)
    # we can construct nice natural language sentences. Otherwise, generic format.
    for i, row in df.iterrows():
        parts = []
        for col, val in row.items():
            parts.append(f"{col} is {val}")
        
        doc_text = f"Agricultural record {i}: " + ", ".join(parts) + "."
        
        # If 'label' is present, maybe give it a special focus
        if 'label' in row.index:
            doc_text = f"For the crop '{row['label']}', the conditions are: " + ", ".join([f"{c}={row[c]}" for c in row.index if c != 'label']) + "."
            
        documents.append({
            "text": doc_text,
            "metadata": {"source": "Mahesh2841/Agriculture", "row_index": i}
        })
    
    with open("data/rag_documents.json", "w", encoding='utf-8') as f:
        json.dump(documents, f, indent=2)
    
    print(f"Prepared {len(documents)} textual documents for RAG in data/rag_documents.json")

if __name__ == "__main__":
    ingest_data()

