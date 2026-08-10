# ingestors/fetch_who.py
import requests
import pandas as pd

def fetch_who_gho_indicators():
    url = "https://ghoapi.azureedge.net/api/Indicator"
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        return response.json().get("value", [])
    else:
        raise Exception(f"WHO GHO API Error: {response.status_code}")

def run_ingestion(client, model, batch_size=64):
    print("Fetching WHO Global Health Observatory Indicators...", flush=True)
    raw_data = fetch_who_gho_indicators()
    
    pd_df = pd.DataFrame(raw_data)
    pd_df = pd_df[["IndicatorCode", "IndicatorName", "Language"]].fillna("")
    pd_df = pd_df[pd_df["Language"] == "EN"]

    collection = client.get_or_create_collection(
        name="global_health_atlas"
    )

    ids = [f"who_{code}" for code in pd_df["IndicatorCode"].astype(str).tolist()]
    documents = [
        f"Source: WHO GHO. Indicator Code: {row['IndicatorCode']}. Indicator Name: {row['IndicatorName']}"
        for _, row in pd_df.iterrows()
    ]
    metadatas = [
        {"source": "who_gho", "indicator_id": str(row["IndicatorCode"]), "short_name": str(row["IndicatorName"])[:100]}
        for _, row in pd_df.iterrows()
    ]

    total_records = len(pd_df)
    print(f"Generating embeddings and persisting {total_records} records in batches of {batch_size}...")

    for i in range(0, total_records, batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]

        batch_embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_meta
        )
        print(f"  └ Ingested {min(i + batch_size, total_records)} / {total_records} records...")

    print("WHO GHO Ingestion complete.")