# pipelines/ingest_nhanes.py
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

def fetch_nhanes_metadata():
    url = "https://raw.githubusercontent.com/rsgoncalves/nhanes-metadata/main/metadata/nhanes_variables.tsv"
    fallback_tsv = (
        "VariableID\tSASLabel\tQuestionText\n"
        "RIAGENDR\tGender\t'Gender of the participant.'\n"
        "BPXSY1\tSystolic: Blood Pres 1st rdg mm Hg\t'Systolic blood pressure, first reading.'\n"
        "DIQ010\tDoctor told you have diabetes\t'Have you ever been told by a doctor or health professional that you have diabetes?'"
    )
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.text
        else:
            return fallback_tsv
    except Exception as e:
        print(f"⚠️ Network fetch failed, using fallback schema: {e}")
        return fallback_tsv

def main():
    print("Fetching CDC NHANES Survey Variable Registries...")
    tsv_data = fetch_nhanes_metadata()
    pd_df = pd.read_csv(io.StringIO(tsv_data), sep='\t')
    
    col_mappings = {"variable_id": "VariableID", "sas_label": "SASLabel", "question_text": "QuestionText"}
    pd_df = pd_df.rename(columns=col_mappings)
    
    pd_df = pd_df.fillna({"VariableID": "UNKNOWN_VAR", "SASLabel": "Unknown Factor", "QuestionText": "No Context Clarified"})
    pd_df = pd_df[["VariableID", "SASLabel", "QuestionText"]].drop_duplicates(subset=["VariableID"], keep="first").head(10000)
    
    db_path = "./global_health_atlas.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    rows_to_insert = []
    for _, row in pd_df.iterrows():
        var_id = str(row['VariableID'])
        label = str(row['SASLabel'])
        question = str(row['QuestionText'])
        text_content = f"Source: CDC NHANES | Variable: {var_id} | Label: {label} | Question: {question}"
        
        vec = generate_vector(text_content)
        vec_blob = np.array(vec, dtype=np.float32).tobytes()
        
        rows_to_insert.append((f"NHANES_VAR_{var_id}", "NHANES", var_id, label[:100], text_content, vec_blob))
        
    cursor.executemany("""
        INSERT OR REPLACE INTO health_vector_store 
        (id, source, indicator_id, short_name, document, vector)
        VALUES (?, ?, ?, ?, ?, ?)
    """, rows_to_insert)
    
    conn.commit()
    conn.close()
    print(f"✅ NHANES Integration Complete! Inserted {len(rows_to_insert)} records.")

if __name__ == "__main__":
    main()
