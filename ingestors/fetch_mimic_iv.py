# ingestors/fetch_mimic_iv.py
import io
import requests
import pandas as pd

def fetch_mimic_d_items():
    url = "https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iv/buildschema/d_items.csv"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.text
        else:
            return "itemid,label,abbreviation,category,unitname\n220045,Heart Rate,HR,Routine Vital Signs,bpm\n"
    except Exception as e:
        raise Exception(f"Failed to fetch MIMIC-IV clinical dictionary: {str(e)}")

def run_ingestion(client, model, batch_size=64):
    print("Fetching MIMIC-IV Clinical Item Dictionary...", flush=True)
    csv_data = fetch_mimic_d_items()

    pd_df = pd.read_csv(io.StringIO(csv_data))
    pd_df = pd_df.fillna("")

    collection = client.get_or_create_collection(
        name="global_health_atlas"
    )

    ids = [f"mimic_{row['itemid']}" for _, row in pd_df.iterrows()]
    documents = [
        f"Source: MIMIC-IV. Item ID: {row['itemid']}. Label: {row['label']}. Category: {row.get('category', '')}. Unit: {row.get('unitname', '')}"
        for _, row in pd_df.iterrows()
    ]
    metadatas = [
        {"source": "mimic_iv", "item_id": str(row["itemid"]), "short_name": str(row["label"])[:100]}
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

    print("MIMIC-IV Ingestion complete.", flush=True)
