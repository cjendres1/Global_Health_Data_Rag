# ingestors/fetch_nhanes.py
import pandas as pd

def fetch_nhanes_manifest():
    return [
        {"Variable": "LBXGLU", "Description": "Fasting Glucose (mg/dL)", "Target": "Both males and females 12 YEARS - 150 YEARS"},
        {"Variable": "BPXSY1", "Description": "Systolic: Reading 1", "Target": "Both males and females 8 YEARS - 150 YEARS"},
        {"Variable": "RIDAGEYR", "Description": "Age in years at screening", "Target": "Both males and females 0 YEARS - 150 YEARS"}
    ]

def run_ingestion(client, model, batch_size=64):
    print("Fetching CDC NHANES Variable Manifest...", flush=True)
    raw_data = fetch_nhanes_manifest()

    pd_df = pd.DataFrame(raw_data).fillna("")

    collection = client.get_or_create_collection(
        name="global_health_atlas"
    )

    ids = [f"nhanes_{row['Variable']}" for _, row in pd_df.iterrows()]
    documents = [
        f"Source: CDC NHANES. Variable: {row['Variable']}. Description: {row['Description']}. Target Demographic: {row['Target']}"
        for _, row in pd_df.iterrows()
    ]
    metadatas = [
        {"source": "cdc_nhanes", "variable": str(row["Variable"]), "short_name": str(row["Description"])[:100]}
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

    print("CDC NHANES Ingestion complete.")