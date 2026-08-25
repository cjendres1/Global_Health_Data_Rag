# ingestors/fetch_uk_biobank.py
import requests
import pandas as pd
import io

def fetch_uk_biobank_fields():
    urls = [
        "https://biobank.ndph.ox.ac.uk/showcase/schema.cgi?id=1",
        "https://biobank.ctsu.ox.ac.uk/showcase/schema.cgi?id=1"
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    for url in urls:
        try:
            print(f"  └ Attempting UK Biobank schema fetch: {url}...", flush=True)
            res = requests.get(url, headers=headers, timeout=15)
            if res.status_code == 200 and len(res.text) > 200:
                # Read as tab-delimited, auto-detecting whitespace if necessary
                df = pd.read_csv(
                    io.StringIO(res.text), 
                    sep=r"\t+|\s{2,}", 
                    engine="python", 
                    on_bad_lines="skip"
                )
                if not df.empty:
                    print(f"  ✔ Successfully retrieved {len(df)} UK Biobank fields.", flush=True)
                    return df
        except Exception:
            continue
            
    raise Exception("UK Biobank schema endpoints unreachable.")

def run_ingestion(client, model, batch_size=64):
    print("Fetching UK Biobank Metadata...", flush=True)
    try:
        raw_df = fetch_uk_biobank_fields()
    except Exception as e:
        print(f"  ⚠ Skipping UK Biobank Ingestion: {e}", flush=True)
        return

    # Normalize column names to lower case
    raw_df.columns = [str(c).strip().lower() for c in raw_df.columns]

    # Look for field ID and title/description columns
    fid_col = next((c for c in raw_df.columns if "field" in c or "id" in c), None)
    title_col = next((c for c in raw_df.columns if "title" in c or "name" in c or "desc" in c), None)

    if fid_col and title_col:
        raw_df = raw_df[[fid_col, title_col]].copy()
        raw_df.columns = ["field_id", "title"]
    elif len(raw_df.columns) >= 2:
        raw_df = raw_df.iloc[:, :2].copy()
        raw_df.columns = ["field_id", "title"]
    else:
        # Single column fallback (e.g. if field ID and title merged into one text line)
        col_name = raw_df.columns[0]
        raw_df["field_id"] = raw_df.index.astype(str)
        raw_df["title"] = raw_df[col_name].astype(str)
        raw_df = raw_df[["field_id", "title"]]

    raw_df = raw_df.fillna("").astype(str)
    raw_df = raw_df.drop_duplicates(subset=["field_id"], keep="first").reset_index(drop=True)
    
    collection = client.get_or_create_collection(
        name="global_health_atlas",
        metadata={"hnsw:space": "cosine"}
    )
    
    ids = [f"ukbb_{fid}" for fid in raw_df["field_id"].tolist()]
    documents = [
        f"Source: UK Biobank. Field ID: {row['field_id']}. Title: {row['title']}"
        for _, row in raw_df.iterrows()
    ]
    metadatas = [
        {"source": "uk_biobank", "field_id": row["field_id"], "title": row["title"]}
        for _, row in raw_df.iterrows()
    ]
    
    total_records = len(raw_df)
    print(f"Generating embeddings and persisting {total_records} records in batches of {batch_size}...", flush=True)
    
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
        print(f"  └ Ingested {min(i + batch_size, total_records)} / {total_records} records...", flush=True)
        
    print("UK Biobank Ingestion complete.", flush=True)
