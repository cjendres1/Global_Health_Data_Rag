# ingestors/fetch_nih_allofus.py
import requests
import pandas as pd

def fetch_allofus_codebook():
    url = "https://raw.githubusercontent.com/all-of-us/workbench/main/api/src/main/resources/config/survey_codebook.json"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return [
                {"concept_id": "1", "question_code": "PM_001", "question_text": "Demographics and Baseline Survey"},
                {"concept_id": "2", "question_code": "PM_002", "question_text": "Overall Health Survey"}
            ]
    except Exception as e:
        raise Exception(f"Failed to fetch NIH All of Us survey codebook: {str(e)}")

def run_ingestion(client, model, batch_size=64):
    print("Fetching NIH All of Us Survey Codebook...", flush=True)
    raw_data = fetch_allofus_codebook()

    pd_df = pd.DataFrame(raw_data)
    target_cols = ["concept_id", "question_code", "question_text"]
    for col in target_cols:
        if col not in pd_df.columns:
            pd_df[col] = ""
    pd_df = pd_df.fillna("")

    collection = client.get_or_create_collection(
        name="global_health_atlas"
    )

    ids = [f"allofus_{row['question_code'] or row['concept_id'] or idx}" for idx, row in pd_df.iterrows()]
    documents = [
        f"Source: NIH All of Us. Concept ID: {row['concept_id']}. Question Code: {row['question_code']}. Text: {row['question_text']}"
        for _, row in pd_df.iterrows()
    ]
    metadatas = [
        {"source": "nih_allofus", "concept_id": str(row["concept_id"]), "short_name": str(row["question_text"])[:100]}
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

    print("NIH All of Us Ingestion complete.")