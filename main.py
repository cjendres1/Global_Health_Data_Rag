# main.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import sys
import time
import chromadb
from sentence_transformers import SentenceTransformer

# Import ingestion modules
from ingestors.fetch_usaid_dhs import run_ingestion as run_usaid
from ingestors.fetch_who import run_ingestion as run_who
from ingestors.fetch_nih_allofus import run_ingestion as run_nih_allofus
from ingestors.fetch_mimic_iv import run_ingestion as run_mimic_iv
from ingestors.fetch_nhanes import run_ingestion as run_nhanes
from ingestors.fetch_uk_biobank import run_ingestion as run_uk_biobank

INGESTORS = {
    "usaid": ("USAID DHS Indicators", run_usaid),
    "who": ("WHO GHO Registry", run_who),
    "nih_allofus": ("NIH All of Us Codebook", run_nih_allofus),
    "mimic_iv": ("MIMIC-IV Clinical Dictionary", run_mimic_iv),
    "nhanes": ("CDC NHANES Variables", run_nhanes),
    "uk_biobank": ("UK Biobank Showcase", run_uk_biobank),
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Global Health Atlas: Multi-Source Vector Ingestion Pipeline"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(INGESTORS.keys()) + ["all"],
        default=["all"],
        help="Specify which data sources to ingest."
    )
    parser.add_argument(
        "--db-path",
        default="data/chroma_db",
        help="Local directory path for ChromaDB storage."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for vector upserts."
    )
    parser.add_argument(
        "--use-http",
        action="store_true",
        help="Connect to external ChromaDB HTTP server."
    )
    parser.add_argument(
        "--host", default="localhost", help="ChromaDB server host."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="ChromaDB server port."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    selected_sources = list(INGESTORS.keys()) if "all" in args.sources else args.sources

    print("=" * 60)
    print("  GLOBAL HEALTH ATLAS - VECTOR INGESTION PIPELINE")
    print("=" * 60)
    print(f"Target Sources  : {', '.join(selected_sources)}")
    print(f"Connection Mode : {'HTTP Server (' + args.host + ':' + str(args.port) + ')' if args.use_http else 'Persistent Disk (' + args.db_path + ')'}")
    print(f"Batch Size      : {args.batch_size}")
    print("=" * 60 + "\n")

    print("Initializing embedding model (all-MiniLM-L6-v2)...", flush=True)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model ready.\n", flush=True)

    if args.use_http:
        client = chromadb.HttpClient(host=args.host, port=args.port)
    else:
        client = chromadb.PersistentClient(path=args.db_path)

    start_total_time = time.time()
    successful = []
    failed = []

    for key in selected_sources:
        name, run_func = INGESTORS[key]
        print(f"► Starting ingestion: {name} [{key}]...", flush=True)
        start_time = time.time()
        
        try:
            run_func(
                client=client,
                model=embedding_model,
                batch_size=args.batch_size
            )
            
            elapsed = time.time() - start_time
            print(f"✔ Completed {name} in {elapsed:.2f}s\n", flush=True)
            successful.append(key)
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"✖ ERROR processing {name}: {str(e)} (Failed after {elapsed:.2f}s)\n", flush=True)
            failed.append((key, str(e)))

    total_elapsed = time.time() - start_total_time

    print("=" * 60)
    print("  PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total Time      : {total_elapsed:.2f} seconds")
    print(f"Successful ({len(successful)}): {', '.join(successful) if successful else 'None'}")
    
    if failed:
        print(f"Failed ({len(failed)}):")
        for key, err in failed:
            print(f"  - {key}: {err}")
        sys.exit(1)

if __name__ == "__main__":
    main()