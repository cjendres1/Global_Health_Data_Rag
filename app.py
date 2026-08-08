import os
import sqlite3
import hashlib
import numpy as np
import streamlit as st
import torch

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

# Pure Python deterministic vector generator matching ingestion pipelines
def generate_vector(text: str, dimensionality: int = 384) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.normal(loc=0.0, scale=1.0, size=dimensionality)
    norm = np.linalg.norm(vector)
    return (vector / norm) if norm > 0 else vector

# -----------------------------------------------------------------------------
# 💡 PIPELINE & DATABASE INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_resource
def initialize_core_pipeline():
    parser = ClinicalQueryParser()
    reranker = PyTorchNeuralReranker()
    db_path = os.path.abspath("global_health_atlas.db")
    
    if not os.path.exists(db_path):
        st.error(f"⚠️ Vector database missing at `{db_path}`. Please run ingestion first.")
        st.stop()
        
    return parser, reranker, db_path

try:
    parser, reranker, db_path = initialize_core_pipeline()
except Exception as e:
    st.error(f"Initialization Failed: {e}")
    st.stop()

# Helper function to fetch DB stats & unique sources dynamically
def get_db_metrics():
    if not os.path.exists(db_path):
        return 0, []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM health_vector_store")
    total_records = cursor.fetchone()[0]
    cursor.execute("SELECT DISTINCT source FROM health_vector_store WHERE source IS NOT NULL")
    sources = [row[0] for row in cursor.fetchall()]
    conn.close()
    return total_records, sorted(sources)

total_records, available_sources = get_db_metrics()
source_count = len(available_sources)

# -----------------------------------------------------------------------------
# 🔍 SQLITE VECTOR RETRIEVAL LOGIC WITH METADATA FILTERING
# -----------------------------------------------------------------------------
def query_sqlite_vector_db(query_text: str, target_sources: list, max_dist: float, top_k: int = 15) -> list:
    """Computes cosine similarity/distance over SQLite stored vectors with source filtering."""
    query_vec = generate_vector(query_text)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if target_sources and len(target_sources) < len(available_sources):
        placeholders = ','.join(['?'] * len(target_sources))
        query = f"SELECT id, source, indicator_id, short_name, document, vector FROM health_vector_store WHERE source IN ({placeholders})"
        cursor.execute(query, target_sources)
    else:
        cursor.execute("SELECT id, source, indicator_id, short_name, document, vector FROM health_vector_store")
        
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []

    scores = []
    for row in rows:
        doc_id, source, indicator_id, short_name, document, vec_bytes = row
        db_vec = np.frombuffer(vec_bytes, dtype=np.float32)
        
        # Cosine Similarity & Distance Conversion
        similarity = float(np.dot(query_vec, db_vec))
        distance = 1.0 - similarity
        
        # Filter out records exceeding the user's distance threshold
        if distance <= max_dist:
            scores.append({
                "vector_score": similarity,
                "distance": distance,
                "source_dataset": source,
                "table_id": doc_id,
                "variable_id": indicator_id,
                "variable_name": short_name,
                "description": document
            })

    # Sort descending by vector similarity score
    scores.sort(key=lambda x: x["vector_score"], reverse=True)
    return scores[:top_k]

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
        "Maximum Distance Threshold:",
        min_value=0.0,
        max_value=2.0,
        value=1.5,
        step=0.1,
        help="Lower values enforce strict matching. Higher values allow looser conceptual connections."
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
    st.metric(label="Database Status", value="SQLite Active" if os.path.exists(db_path) else "Missing")
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
        # Step 1: spaCy Query Understanding
        with st.spinner("Extracting structural semantics..."):
            parsed_data = parser.parse(user_query)
        
        c_tok, c_dem = st.columns(2)
        with c_tok:
            st.caption(f"**Extracted Keywords:** {parsed_data.get('extracted_keywords', 'None')}")
        with c_dem:
            st.caption(f"**Inferred Demographics:** {parsed_data.get('inferred_demographics', 'General')}")

        # Step 2 & 3: SQLite Vector Match & PyTorch Neural Reranking
        with st.spinner("Searching vector store & executing PyTorch reranking..."):
            candidates = query_sqlite_vector_db(user_query, selected_sources, max_distance, top_k=n_results * 2)
            
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
                    col_meta, col_dist = st.columns([4, 1])
                    with col_meta:
                        st.markdown(f"**Match #{idx}: `{item['variable_id']}`** — {item['variable_name']}")
                        st.caption(f"Origin Registry: `{item['source_dataset']}` | Table ID: `{item['table_id']}`")
                    with col_dist:
                        st.metric(label="Distance", value=f"{item['distance']:.4f}")
                    
                    st.text_area(
                        label="Metadata Context Payload",
                        value=item['description'],
                        height=70,
                        key=f"doc_{idx}",
                        disabled=True
                    )
                    st.markdown("---")

            # -----------------------------------------------------------------
            # 🤖 SYNTHESIS GENERATION PHASE
            # -----------------------------------------------------------------
            st.markdown("### 🤖 Generation Phase (Live RAG Synthesis)")
            with st.expander("See live synthesis tracking", expanded=True):
                context_str = "\n\n".join([
                    f"[Match #{i+1} | Source: {m['source_dataset']} | Code: {m['variable_id']}]\n{m['description']}"
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
