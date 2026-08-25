import os
import sqlite3
import hashlib
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from data_fetcher import fetch_live_data, fetch_all_live_data

# Gracefully handle optional local Ollama dependency
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# -----------------------------------------------------------------------------
# 📦 IMPORT LOCAL MODULES
# -----------------------------------------------------------------------------
from query_parser import ClinicalQueryParser
from neural_reranker import PyTorchNeuralReranker

# Clamp thread pools to prevent CPU throttling on shared cloud hosts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.set_num_threads(1)

# Set page configuration
st.set_page_config(
    page_title="Global Health Data Atlas",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 💡 PIPELINE & CHROMADB INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_resource
def load_vector_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_vector_embedder()

@st.cache_resource
def initialize_core_pipeline():
    p = ClinicalQueryParser()
    r = PyTorchNeuralReranker()
    
    db_path = os.path.abspath("data/chroma_db")
    
    # Auto-run ingestion if database directory is missing or empty on cloud startup
    if not os.path.exists(db_path) or not os.listdir(db_path):
        st.info("⚡ Persistent vector store not found. Building database via main.py...")
        subprocess.run(["python", "main.py"], check=True)
        
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="global_health_atlas")
    
    return p, r, collection, db_path

try:
    parser, reranker, collection, db_path = initialize_core_pipeline()
except Exception as e:
    st.error(f"⚠️ Initialization Failed during startup: {e}")
    st.exception(e)
    st.stop()

# Helper function to fetch DB stats & unique sources dynamically
# Helper function to fetch DB stats & normalize unique sources dynamically
def get_db_metrics():
    total_records = collection.count()

    if total_records == 0:
        return 0, {}, []

    metas = collection.get(include=["metadatas"])["metadatas"]

    # Map a canonical source name to the actual source values stored in Chroma.
    # This handles values such as WHO_GHO and who_gho as one registry.
    source_aliases = {}

    for meta in metas:
        if not meta:
            continue

        raw_source = str(meta.get("source", "unknown")).strip()
        canonical_source = raw_source.upper()

        source_aliases.setdefault(canonical_source, set()).add(raw_source)

    available_sources = sorted(source_aliases.keys())

    return total_records, source_aliases, available_sources


total_records, source_aliases, available_sources = get_db_metrics()
source_count = len(available_sources)

# -----------------------------------------------------------------------------
# 🔍 CHROMADB VECTOR RETRIEVAL LOGIC WITH METADATA FILTERING
# -----------------------------------------------------------------------------
def query_chroma_vector_db(
    query_text: str,
    target_sources: list,
    max_dist: float,
    top_k: int = 15
) -> list:
    """
    Performs semantic vector search over ChromaDB with case-insensitive
    source filtering.

    ChromaDB is configured to use cosine distance, so lower distance
    means greater semantic similarity.
    """
    query_vec = embedder.encode(
        [query_text],
        normalize_embeddings=True
    ).tolist()

    where_clause = None

    # Only apply a source filter if the user has excluded at least one
    # canonical registry.
    if target_sources and len(target_sources) < len(available_sources):

        # Convert canonical source names back to all actual values present
        # in ChromaDB, e.g. WHO_GHO -> ["WHO_GHO", "who_gho"].
        actual_sources = []

        for source in target_sources:
            actual_sources.extend(
                source_aliases.get(source, [])
            )

        actual_sources = sorted(set(actual_sources))

        if len(actual_sources) == 1:
            where_clause = {
                "source": actual_sources[0]
            }
        elif actual_sources:
            where_clause = {
                "$or": [
                    {"source": src}
                    for src in actual_sources
                ]
            }

    results = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )

    if (
        not results
        or not results["documents"]
        or not results["documents"][0]
    ):
        return []

    scores = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if dist <= max_dist:
            scores.append({
                "vector_score": 1.0 - dist,
                "distance": dist,
                "source_dataset": meta.get("source", "unknown"),
                "table_id": meta.get(
                    "field_id",
                    meta.get("label", "N/A")
                ),
                "variable_id": meta.get(
                    "field_id",
                    meta.get(
                        "variable",
                        meta.get(
                            "short_name",
                            meta.get("IndicatorId", "N/A")
                        )
                    )
                ),
                "variable_name": meta.get(
                    "label",
                    meta.get(
                        "title",
                        meta.get(
                            "short_name",
                            "Indicator"
                        )
                    )
                ),
                "description": doc
            })

    scores.sort(
        key=lambda x: x["vector_score"],
        reverse=True
    )

    return scores

# -----------------------------------------------------------------------------
# 🎨 SIDEBAR CONTROL CENTER
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🌐 Atlas Control Center")
    st.markdown("---")
    
    st.subheader("🎯 Search Filtering & Tuning")
    selected_sources = st.multiselect(
        "Scope to Target Registries:",
        options=available_sources,
        default=available_sources,
        help="Uncheck sources to exclude their variables from the vector search space."
    )
    
    max_distance = st.slider(
        "Maximum Chroma Distance:",
        min_value=0.00,
        max_value=1.00,
        value=0.40,
        step=0.01,
        help=(
            "Cosine distance threshold. Lower values require closer semantic "
            "matches; higher values allow broader matches."
        ),
        format="%.2f"
    )
    
    st.markdown("---")
    st.subheader("Ingestion Status")
    for src in available_sources:
        st.success(f"✅ {src} Active")
    
    st.markdown("---")
    st.caption("Decoupled multi-source RAG data architecture.")

# -----------------------------------------------------------------------------
# 🗺️ MAIN INTERFACE & METRICS
# -----------------------------------------------------------------------------
st.title("🗺️ Global Health Data Atlas AI")
st.markdown(
    """
    This platform demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline, harmonizing 
    disparate health metadata registries into a unified vector search layer with neural re-ranking.
    """
)
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Unified Records", value=total_records)
with col2:
    st.metric(label="Database Status", value="ChromaDB Active" if total_records > 0 else "Empty")
with col3:
    st.metric(label="Harmonized Registries", value=f"{source_count} Sources")

st.markdown("### 🔍 Semantic Search & RAG Retrieval")

user_query = st.text_input(
    "Enter a clinical concept, health indicator, or survey question:",
    placeholder="e.g., Query metrics examining body mass index distributions..."
)

n_results = st.slider("Max context matches to retrieve:", min_value=1, max_value=10, value=5)

if user_query:
    if not selected_sources:
        st.warning("⚠️ Please select at least one source in the sidebar to execute a search.")
    else:
        # Step 1: Query Understanding
        with st.spinner("Extracting structural semantics..."):
            parsed_data = parser.parse(user_query)
        
        c_tok, c_dem = st.columns(2)
        with c_tok:
            st.caption(f"**Extracted Keywords:** {parsed_data.get('extracted_keywords', 'None')}")
        with c_dem:
            st.caption(f"**Inferred Demographics:** {parsed_data.get('inferred_demographics', 'General')}")

        # Step 2 & 3: ChromaDB Vector Match & PyTorch Neural Reranking
        with st.spinner("Searching vector store & executing PyTorch reranking..."):
            candidates = query_chroma_vector_db(user_query, selected_sources, max_distance, top_k=n_results * 2)
            
            if candidates:
                final_results = reranker.rerank(user_query, candidates)[:n_results]
            else:
                final_results = []

        # Step 4: Display Retrieved Context Matches
        st.markdown(f"#### 📦 Retrieved Context Blocks ({len(final_results)} Matches)")
        
        if not final_results:
            st.info("No records matched your search criteria within the current distance threshold.")
        else:
            for idx, item in enumerate(final_results, 1):
                with st.container():
                    col_meta, col_rerank, col_dist = st.columns([3, 1, 1])
                    with col_meta:
                        st.markdown(f"**Match #{idx}: `{item.get('variable_id', 'N/A')}`** — {item.get('variable_name', 'Unnamed Variable')}")
                        st.caption(f"Origin Registry: `{item.get('source_dataset', 'Unknown')}` | Identifier: `{item.get('table_id', 'N/A')}`")
                    
                    with col_rerank:
                        st.metric(label="Neural Relevance", value=f"{item.get('rerank_score', 0.0):.3f}")
                        
                    with col_dist:
                        st.metric(label="Chroma Distance", value=f"{item.get('distance', 0.0):.4f}")
                    
                    st.text_area(
                        label="Metadata Context Payload",
                        value=item.get('description', ''),
                        height=70,
                        key=f"doc_{idx}",
                        disabled=True
                    )
                    st.markdown("---")

# -----------------------------------------------------------------
# 🛠️ LIVE DATA EXPORT & AUTOMATED INGESTION
# -----------------------------------------------------------------
# Check that final_results is defined and contains retrieved records
if 'final_results' in locals() and final_results:

    # Extract the records to export from your retrieved/reranked search results
    export_records = [
        {
            "source_dataset": m.get("source_dataset", ""),
            "variable_id": m.get("variable_id", ""),
            "description": m.get("description", "")
        }
        for m in final_results
    ]

    st.markdown("### 🛠️ Live Observations Export & Ingestion")

    with st.spinner("Fetching live health observations across retrieved match endpoints..."):
        # Pull actual health observations for top matches
        df_raw_combined = fetch_all_live_data(export_records, limit_per_var=50)

tab_table, tab_code = st.tabs(["📊 Raw Observation Data", "🐍 Python Import Script"])

with tab_table:
    if not df_raw_combined.empty:
        st.markdown(f"**Retrieved Microdata Records ({len(df_raw_combined)} rows)**")
        st.dataframe(df_raw_combined, use_container_width=True, hide_index=True)

        col_csv, col_json = st.columns(2)
        with col_csv:
            st.download_button(
                label="📥 Download Fetched Observations (CSV)",
                data=df_raw_combined.to_csv(index=False).encode('utf-8'),
                file_name="atlas_live_observations.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_live_csv_btn"
            )
        with col_json:
            st.download_button(
                label="📥 Download Fetched Observations (JSON)",
                data=df_raw_combined.to_json(orient="records", indent=2),
                file_name="atlas_live_observations.json",
                mime="application/json",
                use_container_width=True,
                key="download_live_json_btn"
            )
    else:
        st.info("No live observation records could be automatically fetched for the current matches. Check API connectivity.")

with tab_code:
    st.markdown("Copy and paste this Python script to pull these raw observations directly into your workspace or notebook:")
    
    # Extract distinct registries and variables for the script
    target_ids = [str(r["variable_id"]) for r in export_records if r.get("variable_id")]
    
    python_snippet = f'''import pandas as pd
import requests

# 1. Target Variables retrieved from Global Health Atlas search
target_variables = {target_ids}

def fetch_atlas_health_data(variable_id):
"""Fetches live health observation records from open API endpoints."""
# Example WHO GHO Endpoint fetch
url = f"https://ghoapi.azureedge.net/api/{{variable_id}}"
try:
res = requests.get(url, timeout=10)
if res.status_code == 200:
data = res.json().get("value", [])
df = pd.DataFrame(data)
df["variable_id"] = variable_id
return df
except Exception as e:
print(f"Error fetching {{variable_id}}: {{e}}")
return pd.DataFrame()

# 2. Iterate and combine fetched observation datasets
all_data = []
for var_id in target_variables:
df_var = fetch_atlas_health_data(var_id)
if not df_var.empty:
all_data.append(df_var)

if all_data:
df_combined = pd.concat(all_data, ignore_index=True)
print(f"Successfully loaded {{len(df_combined)}} live observation rows.")
print(df_combined.head())
else:
print("No observations returned.")
'''
    st.code(python_snippet, language="python")

# -----------------------------------------------------------------
# 🤖 SYNTHESIS GENERATION PHASE
# -----------------------------------------------------------------
st.markdown("### 🤖 Generation Phase (Live RAG Synthesis)")
with st.expander("See live synthesis tracking", expanded=True):
    context_str = "\n\n".join([
        f"[Match #{i+1} | Source: {m.get('source_dataset', 'N/A')} | Code: {m.get('variable_id', 'N/A')}]\n{m.get('description', '')}"
        for i, m in enumerate(final_results)
    ])
    
    system_prompt = (
        "You are an expert global health data translation assistant. Synthesize a concise, "
        "integrated answer to the user's question using ONLY the provided metadata context blocks. "
        "If the answer cannot be verified by the text snippets, explicitly state that the "
        "information is not available in the current Atlas registers. Do not invent details."
    )
    
    user_prompt = f"Context Blocks:\n{context_str}\n\nUser Question: {user_query}"
    
    st.caption("**System Prompt:** " + system_prompt)
    
    if OLLAMA_AVAILABLE:
        st.markdown("**Generated Response (Streaming from Llama 3.2 via Ollama):**")
        def ollama_stream_generator():
            try:
                stream = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=True
                )
                for chunk in stream:
                    yield chunk['message']['content']
            except Exception as err:
                yield f"\n⚠️ Local Ollama daemon unreachable: {err}. Ensure `ollama serve` is running."

        st.write_stream(ollama_stream_generator)
    else:
        st.info("💡 **Cloud Mode Active:** Ollama local LLM inference is disabled in cloud hosting environments. The dense vector matches and PyTorch neural reranked contexts above represent your retrieved RAG context window.")