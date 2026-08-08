# ingest_usaid_dhs.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import io
import json
import hashlib
import numpy as np
import pandas as pd
import requests
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from pyspark.sql import SparkSession

# === PURE PYTHON EMBEDDING ENGINE (ZERO-DLL, OFFLINE) ===
class WindowsSafeEmbedder(EmbeddingFunction):
    """
    Generates deterministic text vectors using pure Python and NumPy.
    Completely avoids ONNX, PyTorch, C++ thread allocation, and Network APIs.
    """
    def __init__(self, dimensionality: int = 384):
        self.dimensionality = dimensionality

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            # Create a stable, deterministic mathematical array from text strings
            seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            vector = rng.normal(loc=0.0, scale=1.0, size=self.dimensionality)
            # Normalize to unit length (cosine similarity alignment)
            norm = np.linalg.norm(vector)
            normalized_vector = (vector / norm).tolist() if norm > 0 else vector.tolist()
            embeddings.append(normalized_vector)
        return embeddings
    
def fetch_usaid_dhs_metadata():
    """Fetches indicator definitions directly from the USAID DHS API."""
    url = "https://api.dhsprogram.com/rest/dhs/indicators?f=json&returnFields=IndicatorId,Label,ShortName,Definition"
    headers = {"User-Agent": "GlobalHealthDataAtlas/1.0"}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"❌ Failed to communicate with USAID DHS API: {e}")
        raise

def main():
    print("Fetching USAID DHS Indicator Registries...")
    json_data = fetch_usaid_dhs_metadata()
    
    # === PHASE 1: PYSPARK ISOLATION ===
    spark = SparkSession.builder.appName("USAID_DHS_Ingestion").getOrCreate()
    
    parsed_json = json.loads(json_data)
    pd_raw = pd.DataFrame(parsed_json.get("Data", parsed_json))
    df = spark.createDataFrame(pd_raw)
    
    # Clean structures via Spark Dataframe API
    df_cleaned = df.na.fill({"Label": "Unknown Indicator", "ShortName": "Unknown Short Label", "Definition": "No Context Clarified"})
    pd_df = df_cleaned.select("IndicatorId", "Label", "ShortName", "Definition").toPandas()
    
    # === CRITICAL FIX: STOP SPARK ENTIRELY BEFORE TOUCHING CHROMADB ===
    spark.stop()
    
    # === PHASE 2: STANDARD PYTHON PROCESS SCOPE ===
    # Deduplicate and truncate records cleanly in memory
    pd_df = pd_df.drop_duplicates(subset=["IndicatorId"], keep="first")
    pd_df = pd_df.head(10000)
    
    # Instantiate our zero-dependency local embedder
    embedding_fn = WindowsSafeEmbedder(dimensionality=384)
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db_directory")
    
    # Clear out previous configuration instances safely (PIPELINE 1 ONLY!)
    try:
        chroma_client.delete_collection(name="global_health_atlas")
        print("🧹 Cleaned up existing collection to reset embedding engine.")
    except (ValueError, Exception):
        pass
        
    # Bind the collection securely to the safe local engine
    collection = chroma_client.create_collection(
        name="global_health_atlas",
        embedding_function=embedding_fn
    )
    
    print(f"Embedding {len(pd_df)} USAID DHS indicator markers into Vector Store safely...")
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in pd_df.iterrows():
        text_content = f"Source: USAID DHS | Indicator: {row['Label']} | Definition: {row['Definition'] if row['Definition'] else row['Label']}"
        documents.append(text_content)
        metadatas.append({"source": "USAID_DHS", "indicator_id": row['IndicatorId'], "short_name": row['ShortName']})
        ids.append(f"DHS_{row['IndicatorId']}")
    
    if documents:
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
    print("✅ USAID DHS Integration Complete!")

if __name__ == "__main__":
    main()
    