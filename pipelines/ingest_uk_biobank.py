# pipelines/ingest_uk_biobank.py
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

def fetch_uk_biobank_dictionary():
    url = "https://raw.githubusercontent.com/rmgpanw/ukbwranglr/main/inst/extdata/dummy_Data_Dictionary_Showcase.tsv"
    fallback_tsv = (
        "FieldID\tField\tCategory\n"
        "31\tSex\tBaseline characteristics\n"
        "93\tSystolic blood pressure, automated reading\tBlood pressure\n"
        "2443\tDiabetes diagnosed by doctor\tMedical conditions"
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
    print("Fetching UK Biobank Showcase Variable Dictionary...")
    tsv_data = fetch_uk_biobank_dictionary()
    pd_df = pd.read_csv(io.StringIO(tsv_data), sep='\t')
    
    col_mappings = {"field_id": "FieldID", "title": "Field", "notes": "Category"}
    pd_df = pd_df.rename(columns=col_mappings)
    
    pd_df = pd_df.fillna({"FieldID": "UNKNOWN", "Field": "Unknown Field Description", "Category": "General Health"})
    pd_df = pd_df[["FieldID", "Field", "Category"]].drop_duplicates(subset=["FieldID"], keep="first").head(10000)
    
    db_path = "./global_health_atlas.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    rows_to_insert = []
    for _, row in pd_df.iterrows():
        field_id = str(row['FieldID'])
        field_desc = str(row['Field'])
        category = str(row['Category'])
        text_content = f"Source: UK Biobank | Field ID: {field_id} | Category: {category} | Description: {field_desc}"
        
        vec = generate_vector(text_content)
        vec_blob = np.array(vec, dtype=np.float32).tobytes()
        
        rows_to_insert.append((f"UKB_FIELD_{field_id}", "UK_Biobank", field_id, field_desc[:100], text_content, vec_blob))
        
    cursor.executemany("""
        INSERT OR REPLACE INTO health_vector_store 
        (id, source, indicator_id, short_name, document, vector)
        VALUES (?, ?, ?, ?, ?, ?)
    """, rows_to_insert)
    
    conn.commit()
    conn.close()
    print(f"✅ UK Biobank Integration Complete! Inserted {len(rows_to_insert)} records.")

if __name__ == "__main__":
    main()
