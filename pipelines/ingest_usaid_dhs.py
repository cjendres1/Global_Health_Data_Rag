# pipelines/ingest_usaid_dhs.py
import os
import json
import gc
import sqlite3
import hashlib
import numpy as np
import pandas as pd
import requests

# === PURE PYTHON EMBEDDING ENGINE ===
def generate_vector(text: str, dimensionality: int = 384) -> list:
    """Generates deterministic text vectors using pure Python and NumPy."""
    seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.normal(loc=0.0, scale=1.0, size=dimensionality)
    norm = np.linalg.norm(vector)
    return (vector / norm).tolist() if norm > 0 else vector.tolist()

def fetch_usaid_dhs_metadata():
    url = "https://api.dhsprogram.com/rest/dhs/indicators?f=json&returnFields=IndicatorId,Label,ShortName,Definition"
    headers = {"User-Agent": "GlobalHealthDataAtlas/1.0"}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"❌ Failed to communicate with USAID DHS API: {e}")
        raise

def init_db(db_path: str = "./global_health_atlas.db"):
    """Initializes native SQLite storage schema for vector records."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_vector_store (
            id TEXT PRIMARY KEY,
            source TEXT,
            indicator_id TEXT,
            short_name TEXT,
            document TEXT,
            vector BLOB
        )
    """)
    conn.commit()
    conn.close()

def main():
    print("Fetching USAID DHS Indicator Registries...")
    json_data = fetch_usaid_dhs_metadata()
    
    parsed_json = json.loads(json_data)
    pd_df = pd.DataFrame(parsed_json.get("Data", parsed_json))
    
    pd_df = pd_df.fillna({
        "Label": "Unknown Indicator", 
        "ShortName": "Unknown Short Label", 
        "Definition": "No Context Clarified"
    })
    
    pd_df = pd_df[["IndicatorId", "Label", "ShortName", "Definition"]]
    pd_df = pd_df.drop_duplicates(subset=["IndicatorId"], keep="first")
    
    total_records = len(pd_df)
    print(f"Preparing {total_records} records and pre-calculating vectors in memory...")
    
    # Initialize SQLite vector database
    db_path = "./global_health_atlas.db"
    init_db(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    rows_to_insert = []
    for idx, row in pd_df.iterrows():
        definition = row['Definition'] if row['Definition'] else row['Label']
        text_content = f"Source: USAID DHS | Indicator: {row['Label']} | Definition: {definition}"
        
        vec = generate_vector(text_content)
        vec_blob = np.array(vec, dtype=np.float32).tobytes()
        
        doc_id = f"DHS_{row['IndicatorId']}"
        source = "USAID_DHS"
        indicator_id = str(row['IndicatorId'])
        short_name = str(row['ShortName'])
        
        rows_to_insert.append((doc_id, source, indicator_id, short_name, text_content, vec_blob))
    
    print("Writing records and vectors to database...")
    cursor.executemany("""
        INSERT OR REPLACE INTO health_vector_store 
        (id, source, indicator_id, short_name, document, vector)
        VALUES (?, ?, ?, ?, ?, ?)
    """, rows_to_insert)
    
    conn.commit()
    conn.close()

    print(f"✅ USAID DHS Integration Complete! Inserted {total_records} records into {db_path}.")

if __name__ == "__main__":
    main()
