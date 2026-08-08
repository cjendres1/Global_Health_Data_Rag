# pipelines/ingest_who_gho.py
import requests
import pandas as pd
from pyspark.sql import SparkSession
import chromadb

def fetch_who_indicators():
    """
    Fetches the indicator list registry from the official WHO GHO OData API v2.
    """
    url = "https://ghoapi.azureedge.net/api/Indicator"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.json().get('value', [])
        else:
            raise Exception(f"WHO GHO API returned status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to connect to WHO GHO API: {str(e)}")

def main():
    print("Fetching WHO Global Health Observatory Metadata...")
    json_data = fetch_who_gho_metadata()
    
    # === PHASE 1: PYSPARK ISOLATION ===
    spark = SparkSession.builder.appName("WHO_GHO_Ingestion").getOrCreate()
    
    # Process structured payload safely through Pandas, then move into Spark
    import json
    parsed_json = json.loads(json_data)
    # Target standard OData response arrays if applicable (e.g., 'value')
    pd_raw = pd.DataFrame(parsed_json.get("value", parsed_json))
    df = spark.createDataFrame(pd_raw)
    
    # Clean structures via Spark Dataframe API
    df_cleaned = df.na.fill({"IndicatorName": "Unknown Metric", "Language": "en"})
    pd_df = df_cleaned.select("IndicatorCode", "IndicatorName", "Language").toPandas()
    
    # Safely stop Spark JVM context
    spark.stop()
    
    # Enforce scaled production ceiling (up to 10,000 records)
    pd_df = pd_df.head(10000)
    
    # === PHASE 2: CHROMADB EMBEDDED PERSISTENT VECTORIZATION ===
    # Updated from HttpClient to write directly to your local static directory
    chroma_client = chromadb.PersistentClient(path="./chroma_db_directory")
    collection = chroma_client.get_or_create_collection(name="global_health_atlas")
    
    print(f"Embedding {len(pd_df)} WHO GHO indicator metrics into Vector Store...")
        
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in pd_df.iterrows():
        # Schema harmonization context mapping block
        text_content = f"Source: WHO GHO | Indicator Code: {row['IndicatorCode']} | Indicator Name: {row['IndicatorName']}"
        
        documents.append(text_content)
        metadatas.append({
            "source": "WHO_GHO",
            "indicator_id": row['IndicatorCode'],
            "short_name": row['IndicatorName'][:100]  # Safe bounds constraint for string metadata
        })
        ids.append(f"WHO_{row['IndicatorCode']}")
        
    if documents:
        # Issue idempotent multi-row batch submission across the network socket
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
    print("✅ WHO GHO Integration Complete!")

if __name__ == "__main__":
    main()
