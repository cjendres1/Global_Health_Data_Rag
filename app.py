import os
import streamlit as st
import torch
import chromadb
import subprocess
from sentence_transformers import SentenceTransformer

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
    db_path = os.path.abspath("data/chroma_db")

    # Trigger ingestion if DB directory is missing or empty
    if not os.path.exists(db_path) or not os.listdir(db_path):
        st.info("⚡ Initializing vector database on fresh cloud container...")
        subprocess.run(["python", "main.py"], check=True)

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="global_health_atlas")
    return parser, reranker, collection, db_path

try:
    parser, reranker, collection, db_path = initialize_core_pipeline()
except Exception as e:
    st.error(f"Initialization Failed: {e}")
    st.stop()

# Helper function to fetch DB stats & unique sources dynamically
def get_db_metrics():
    total_records = collection.count()
    if total_records == 0:
        return 0, []
    
    # Query all metadata to extract distinct sources
    metas = collection.get(include=["metadatas"])["metadatas"]
    sources = list(set(m.get("source", "unknown") for m in metas if m))
    return total_records, sorted(sources)

total_records, available_sources = get_db_metrics()
source_count = len(available_sources)

# -----------------------------------------------------------------------------
# 🔍 CHROMADB VECTOR RETRIEVAL LOGIC WITH METADATA FILTERING
# -----------------------------------------------------------------------------
def query_chroma_vector_db(query_text: str, target_sources: list, max_dist: float, top_k: int = 15) -> list:
    """Performs native vector similarity search over ChromaDB with source filtering."""
    query_vec = embedder.encode([query_text]).tolist()
    
    # Construct metadata 'where' filter for ChromaDB
    where_clause = None
    if target_sources and len(target_sources) < len(available_sources):
        if len(target_sources) == 1:
            where_clause = {"source": target_sources[0]}
        else:
            where_clause = {"$or": [{"source": src} for src in target_sources]}
            
    results = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )
    
    if not results or not results["documents"] or not results["documents"][0]:
        return []

    scores = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        # Filter out records exceeding the user's distance threshold
        if dist <= max_dist:
            similarity = 1.0 - dist
            scores.append({
                "vector_score": similarity,
                "distance": dist,
                "source_dataset": meta.get("source", "unknown"),
                "table_id": meta.get("field_id", meta.get("label", "N/A")),
                "variable_id": meta.get("field_id", meta.get("short_name", "N/A")),
                "variable_name": meta.get("label", meta.get("title", meta.get("short_name", "Indicator"))),
                "description": doc
            })

    scores.sort(key=lambda x: x["vector_score"], reverse=True)
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
                        st.markdown(f"**Match #{idx}: `{item['variable_id']}`** — {item['variable_name']}")
                        st.caption(f"Origin Registry: `{item['source_dataset']}` | Identifier: `{item['table_id']}`")
                    
                    with col_rerank:
                        # Display reranker confidence if available
                        r_score = item.get("rerank_score", 0.0)
                        st.metric(label="Rerank Score", value=f"{r_score:.4f}")
                        
                    with col_dist:
                        st.metric(label="Chroma Distance", value=f"{item['distance']:.4f}")
                    
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