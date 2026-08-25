# ingestors/fetch_usaid_dhs.py
import requests
import pandas as pd

def fetch_dhs_indicators():
    url = "https://api.dhsprogram.com/rest/dhs/indicators?f=json&perpage=2000"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code == 200:
        return response.json()['Data']
    else:
        raise Exception(f"DHS API Error: {response.status_code}")

def run_ingestion(client, model, batch_size=64):
    print("Fetching USAID DHS Metadata...", flush=True)
    raw_data = fetch_dhs_indicators()
    
    pd_df = pd.DataFrame(raw_data)
    pd_df = pd_df[["IndicatorId", "Label", "Definition", "ShortName"]].fillna("")
    
    # Deduplicate by IndicatorId
    pd_df = pd_df.drop_duplicates(subset=["IndicatorId"], keep="first").reset_index(drop=True)
    
    collection = client.get_or_create_collection(
        name="global_health_atlas",
        metadata={"hnsw:space": "cosine"}
    )
    
    ids = [f"usaid_{id_}" for id_ in pd_df["IndicatorId"].astype(str).tolist()]
    documents = [
        f"Source: USAID DHS. Label: {row['Label']}. Definition: {row['Definition']}" 
        for _, row in pd_df.iterrows()
    ]
    metadatas = [
        {"source": "usaid_dhs", "label": str(row["Label"]), "short_name": str(row["ShortName"])}
        for _, row in pd_df.iterrows()
    ]
    
    total_records = len(pd_df)
    print(f"Generating embeddings and persisting {total_records} unique records in batches of {batch_size}...", flush=True)
    
    for i in range(0, total_records, batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        
        # Encoding without num_workers argument
        batch_embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()
        
        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_meta
        )
        print(f"  └ Ingested {min(i + batch_size, total_records)} / {total_records} records...", flush=True)
        
    print("USAID DHS Ingestion complete.", flush=True)
    