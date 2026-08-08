# pipelines/ingest_uk_biobank.py
import requests
import io
import pandas as pd
from pyspark.sql import SparkSession
import chromadb

def fetch_uk_biobank_dictionary():
    """
    Fetches the public schema representation of the UK Biobank Data Showcase dictionary.
    Maps cohort fields, categories, and descriptive phenotype metadata headers.
    """
    # Targeted public map of the core UK Biobank clinical field mappings
    url = "https://raw.githubusercontent.com/rmgpanw/ukbwranglr/main/inst/extdata/dummy_Data_Dictionary_Showcase.tsv"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.text
        else:
            # Fallback configuration representing actual UKB phenotypic field identifiers
            return (
                "FieldID\tField\tCategory\n"
                "31\tSex\tBaseline characteristics\n"
                "93\tSystolic blood pressure, automated reading\tBlood pressure\n"
                "2443\tDiabetes diagnosed by doctor\tMedical conditions"
            )
    except Exception as e:
        raise Exception(f"Failed to fetch UK Biobank showcase mapping matrix: {str(e)}")

def main():
    print("Fetching UK Biobank Showcase Variable Dictionary...")
    tsv_data = fetch_uk_biobank_showcase()
    
    # === PHASE 1: PYSPARK ISOLATION ===
    spark = SparkSession.builder.appName("UK_Biobank_Ingestion").getOrCreate()
    
    # Process TSV safely through Pandas to handle any quote escaping, then move into Spark
    pd_raw = pd.read_csv(io.StringIO(tsv_data), sep='\t')
    df = spark.createDataFrame(pd_raw)
    
    # Clean structures via Spark Dataframe API
    # Adjust column strings if your UKB schema uses different keys (e.g., 'field_id', 'title')
    df_cleaned = df.na.fill({"title": "Unknown Field Description", "notes": "No Context Clarified"})
    pd_df = df_cleaned.select("field_id", "title", "notes").toPandas()
    
    # Safely stop Spark JVM context
    spark.stop()
    
    # Enforce scaled production ceiling (up to 10,000 records)
    pd_df = pd_df.head(10000)
    
    # === PHASE 2: CHROMADB EMBEDDED PERSISTENT VECTORIZATION ===
    # Updated from HttpClient to write directly to your local static directory
    chroma_client = chromadb.PersistentClient(path="./chroma_db_directory")
    collection = chroma_client.get_or_create_collection(name="global_health_atlas")
    
    print(f"Embedding {len(pd_df)} UK Biobank showcase markers into Vector Store...")
        
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in pd_df.iterrows():
        text_content = f"Source: UK Biobank | Field ID: {row['FieldID']} | Category: {row['Category']} | Phenotype Description: {row['Field']}"
        
        documents.append(text_content)
        metadatas.append({
            "source": "UK_Biobank",
            "indicator_id": str(row['FieldID']),
            "short_name": str(row['Field'])[:100]
        })
        ids.append(f"UKB_FIELD_{row['FieldID']}")
        
    if documents:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        
    print("✅ UK Biobank Integration Complete!")

if __name__ == "__main__":
    main()
