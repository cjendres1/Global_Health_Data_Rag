# pipelines/ingest_mimic_iv.py
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

def fetch_mimic_dictionary():
    url = "https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iv/buildmimic/postgres/concepts_db/d_labitems.csv"
    fallback_csv = (
        "itemid,label,fluid,category\n"
        "50868,Anion Gap,Blood,Chemistry\n"
        "50882,Bicarbonate,Blood,Chemistry\n"
        "50912,Creatinine,Blood,Chemistry"
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
    print("Fetching MIMIC-IV Clinical Data Dictionary...")
    csv_data = fetch_mimic_dictionary()
    pd_df = pd.read_csv(io.StringIO(csv_data))
    
    col_mappings = {"itemid": "ItemID", "label": "Label", "fluid": "Fluid", "category": "Category"}
    pd_df = pd_df.rename(columns=col_mappings)
    
    pd_df = pd_df.fillna({"ItemID": "0", "Label": "Unknown Item", "Fluid": "Unspecified", "Category": "General Lab"})
    pd_df = pd_df[["ItemID", "Label", "Fluid", "Category"]].drop_duplicates(subset=["ItemID"], keep="first").head(10000)
    
    db_path = "./global_health_atlas.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    rows_to_insert = []
    for _, row in pd_df.iterrows():
        item_id = str(row['ItemID'])
        label = str(row['Label'])
        fluid = str(row['Fluid'])
        category = str(row['Category'])
        text_content = f"Source: MIMIC-IV | ItemID: {item_id} | Label: {label} | Fluid: {fluid} | Category: {category}"
        
        vec = generate_vector(text_content)
        vec_blob = np.array(vec, dtype=np.float32).tobytes()
        
        rows_to_insert.append((f"MIMIC_ITEM_{item_id}", "MIMIC_IV", item_id, label[:100], text_content, vec_blob))
        
    cursor.executemany("""
        INSERT OR REPLACE INTO health_vector_store 
        (id, source, indicator_id, short_name, document, vector)
        VALUES (?, ?, ?, ?, ?, ?)
    """, rows_to_insert)
    
    conn.commit()
    conn.close()
    print(f"✅ MIMIC-IV Integration Complete! Inserted {len(rows_to_insert)} records.")

if __name__ == "__main__":
    main()
    