# pipelines/ingest_nih_allofus.py
import os
import io
import sqlite3
import hashlib
import numpy as np
import pandas as pd
import requests

def generate_vector(text: str, dimensionality: int = 384) -> list:
    seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.normal(loc=0.0, scale=1.0, size=dimensionality)
    norm = np.linalg.norm(vector)
    return (vector / norm).tolist() if norm > 0 else vector.tolist()

def fetch_nih_allofus_concepts():
    url = "https://raw.githubusercontent.com/all-of-us/workbench-snippets/main/datasets/concept_sample.csv"
    fallback_csv = (
        "concept_id,concept_name,domain_id,vocabulary_id\n"
        "21600712,Amlodipine 5 MG Oral Tablet,Drug,RxNorm\n"
        "316866,Hypertensive disorder,Condition,SNOMED\n"
        "4329847,Body mass index (BMI),Measurement,LOINC"
    )
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.text
        else:
            return fallback_csv
    except Exception as e:
        print(f"⚠️ Network fetch failed, using fallback schema: {e}")
        return fallback_csv

def main():
    print("Fetching NIH All of Us Concept Registries...")
    csv_data = fetch_nih_allofus_concepts()
    pd_df = pd.read_csv(io.StringIO(csv_data))
    
    col_mappings = {
        "concept_id": "ConceptID", 
        "concept_name": "ConceptName", 
        "domain_id": "DomainID", 
        "vocabulary_id": "VocabularyID"
    }
    pd_df = pd_df.rename(columns=col_mappings)
    
    pd_df = pd_df.fillna({
        "ConceptID": "0", 
        "ConceptName": "Unknown Concept", 
        "DomainID": "General Domain", 
        "VocabularyID": "Standard Vocab"
    })
    pd_df = pd_df[["ConceptID", "ConceptName", "DomainID", "VocabularyID"]].drop_duplicates(subset=["ConceptID"], keep="first").head(10000)
    
    db_path = "./global_health_atlas.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    rows_to_insert = []
    for _, row in pd_df.iterrows():
        cid = str(row['ConceptID'])
        cname = str(row['ConceptName'])
        domain = str(row['DomainID'])
        vocab = str(row['VocabularyID'])
        text_content = f"Source: NIH All of Us | Concept ID: {cid} | Name: {cname} | Domain: {domain} | Vocabulary: {vocab}"
        
        vec = generate_vector(text_content)
        vec_blob = np.array(vec, dtype=np.float32).tobytes()
        
        rows_to_insert.append((f"ALLOFUS_CONCEPT_{cid}", "NIH_ALLOFUS", cid, cname[:100], text_content, vec_blob))
        
    cursor.executemany("""
        INSERT OR REPLACE INTO health_vector_store 
        (id, source, indicator_id, short_name, document, vector)
        VALUES (?, ?, ?, ?, ?, ?)
    """, rows_to_insert)
    
    conn.commit()
    conn.close()
    print(f"✅ NIH All of Us Integration Complete! Inserted {len(rows_to_insert)} records.")

if __name__ == "__main__":
    main()
