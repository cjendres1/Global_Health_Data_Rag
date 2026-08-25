import os
import time
import subprocess

import numpy as np
import pandas as pd
import streamlit as st
import torch
import chromadb
from sentence_transformers import SentenceTransformer

from data_fetcher import fetch_all_live_data
from query_parser import ClinicalQueryParser
from neural_reranker import PyTorchNeuralReranker


# -----------------------------------------------------------------------------
# OPTIONAL OLLAMA
# -----------------------------------------------------------------------------
try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# -----------------------------------------------------------------------------
# CPU THREAD CONTROL
# -----------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.set_num_threads(1)


# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Global Health Data Atlas",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# MODEL CONFIGURATION
# -----------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# -----------------------------------------------------------------------------
# LOAD EMBEDDING MODEL
# -----------------------------------------------------------------------------
@st.cache_resource
def load_vector_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


embedder = load_vector_embedder()


# -----------------------------------------------------------------------------
# INITIALIZE CORE PIPELINE
# -----------------------------------------------------------------------------
@st.cache_resource
def initialize_core_pipeline():

    parser = ClinicalQueryParser()
    reranker = PyTorchNeuralReranker()

    db_path = os.path.abspath("data/chroma_db")

    # Auto-build database if missing.
    if not os.path.exists(db_path) or not os.listdir(db_path):
        st.info(
            "⚡ Persistent vector store not found. "
            "Building database via main.py..."
        )

        subprocess.run(
            ["python", "main.py"],
            check=True,
        )

    client = chromadb.PersistentClient(path=db_path)

    collection = client.get_or_create_collection(
        name="global_health_atlas",
        metadata={"hnsw:space": "cosine"},
    )

    return parser, reranker, collection, db_path


try:

    parser, reranker, collection, db_path = initialize_core_pipeline()

except Exception as e:

    st.error(
        f"⚠️ Initialization failed during startup: {e}"
    )

    st.exception(e)

    st.stop()


# -----------------------------------------------------------------------------
# DATABASE METRICS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_db_metrics():

    total_records = collection.count()

    if total_records == 0:
        return 0, {}, []

    metas = collection.get(
        include=["metadatas"]
    )["metadatas"]

    source_aliases = {}

    for meta in metas:

        if not meta:
            continue

        raw_source = str(
            meta.get("source", "unknown")
        ).strip()

        canonical_source = raw_source.upper()

        source_aliases.setdefault(
            canonical_source,
            set(),
        ).add(raw_source)

    available_sources = sorted(
        source_aliases.keys()
    )

    return (
        total_records,
        source_aliases,
        available_sources,
    )


(
    total_records,
    source_aliases,
    available_sources,
) = get_db_metrics()

source_count = len(available_sources)


# -----------------------------------------------------------------------------
# CHROMADB VECTOR RETRIEVAL
# -----------------------------------------------------------------------------
def query_chroma_vector_db(
    query_text: str,
    target_sources: list,
    max_dist: float,
    top_k: int = 15,
    return_timing: bool = False,
):

    start_total = time.perf_counter()

    # -------------------------------------------------------------------------
    # Query embedding
    # -------------------------------------------------------------------------
    start_embedding = time.perf_counter()

    query_vec = embedder.encode(
        [query_text],
        normalize_embeddings=True,
    ).tolist()

    embedding_seconds = (
        time.perf_counter() - start_embedding
    )

    # -------------------------------------------------------------------------
    # Construct case-insensitive source filter
    # -------------------------------------------------------------------------
    where_clause = None

    if (
        target_sources
        and len(target_sources) < len(available_sources)
    ):

        actual_sources = []

        for source in target_sources:

            actual_sources.extend(
                source_aliases.get(
                    source,
                    [],
                )
            )

        actual_sources = sorted(
            set(actual_sources)
        )

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

    # -------------------------------------------------------------------------
    # Chroma retrieval
    # -------------------------------------------------------------------------
    start_chroma = time.perf_counter()

# TEMPORARY DEBUG — disable source filtering
    where_clause = None

    results = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        where=where_clause,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    st.write("DEBUG: Raw Chroma distances:", results["distances"][0])
    st.write("DEBUG: Raw Chroma metadata:", results["metadatas"][0])
    st.write("DEBUG: Requested top_k:", top_k)    

    chroma_seconds = (
        time.perf_counter() - start_chroma
    )

    # -------------------------------------------------------------------------
    # Process results
    # -------------------------------------------------------------------------
    if (
        not results
        or not results["documents"]
        or not results["documents"][0]
    ):

        timing = {
            "embedding_seconds": embedding_seconds,
            "chroma_seconds": chroma_seconds,
            "retrieval_total_seconds": (
                time.perf_counter()
                - start_total
            ),
            "candidate_count": 0,
        }

        if return_timing:
            return [], timing

        return []

    scores = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):

# TEMPORARY DEBUG: do not filter on distance
        similarity = 1.0 - dist

        scores.append(
            {
                "vector_score": similarity,
                "distance": dist,
                "source_dataset": meta.get(
                    "source",
                    "unknown",
                ),
                "table_id": meta.get(
                    "field_id",
                    meta.get(
                        "label",
                        "N/A",
                    ),
                ),
                "variable_id": meta.get(
                    "field_id",
                    meta.get(
                        "variable",
                        meta.get(
                            "short_name",
                            meta.get(
                                "IndicatorId",
                                "N/A",
                            ),
                        ),
                    ),
                ),
                "variable_name": meta.get(
                    "label",
                    meta.get(
                        "title",
                        meta.get(
                            "short_name",
                            "Indicator",
                        ),
                    ),
                ),
                "description": doc,
            }
        )

    scores.sort(
        key=lambda x: x["vector_score"],
        reverse=True,
    )

    timing = {
        "embedding_seconds": embedding_seconds,
        "chroma_seconds": chroma_seconds,
        "retrieval_total_seconds": (
            time.perf_counter()
            - start_total
        ),
        "candidate_count": len(scores),
    }

    if return_timing:
        return scores, timing

    return scores


# -----------------------------------------------------------------------------
# STREAMLIT STATE INITIALIZATION
# -----------------------------------------------------------------------------
final_results = []
export_records = []
df_raw_combined = pd.DataFrame()


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:

    st.title("🌐 Atlas Control Center")

    st.markdown("---")

    st.subheader("🎯 Search Filtering & Tuning")

    selected_sources = st.multiselect(
        "Scope to Target Registries:",
        options=available_sources,
        default=available_sources,
        help=(
            "Select the registries to include in vector retrieval."
        ),
    )

    max_distance = st.slider(
        "Maximum Chroma Distance:",
        min_value=0.00,
        max_value=1.00,
        value=0.40,
        step=0.01,
        format="%.2f",
        help=(
            "Cosine distance threshold. "
            "Lower values require closer semantic matches."
        ),
    )

    st.caption(
        f"Cosine similarity floor: "
        f"{1.0 - max_distance:.2f}"
    )

    n_results = st.slider(
        "Maximum context matches to retrieve:",
        min_value=1,
        max_value=10,
        value=5,
    )

    use_reranker = st.checkbox(
        "Use PyTorch neural reranking",
        value=True,
        help=(
            "Apply the cross-encoder after ChromaDB "
            "candidate retrieval."
        ),
    )

    st.markdown("---")

    st.subheader("Ingestion Status")

    for src in available_sources:
        st.success(
            f"✅ {src} Active"
        )

    st.markdown("---")

    st.caption(
        "Decoupled multi-source RAG data architecture."
    )


# -----------------------------------------------------------------------------
# MAIN INTERFACE
# -----------------------------------------------------------------------------
st.title("🗺️ Global Health Data Atlas AI")

st.markdown(
    """
This platform demonstrates a **Retrieval-Augmented Generation (RAG)**
pipeline, harmonizing disparate health metadata registries into a
unified vector search layer with neural re-ranking.
"""
)

st.markdown("---")


# -----------------------------------------------------------------------------
# DATABASE METRICS
# -----------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Unified Records",
        value=total_records,
    )

with col2:
    st.metric(
        label="Database Status",
        value=(
            "ChromaDB Active"
            if total_records > 0
            else "Empty"
        ),
    )

with col3:
    st.metric(
        label="Harmonized Registries",
        value=source_count,
    )


# -----------------------------------------------------------------------------
# QUERY
# -----------------------------------------------------------------------------
st.markdown("### 🔍 Semantic Search & RAG Retrieval")

user_query = st.text_input(
    "Enter a clinical concept, health indicator, or survey question:",
    placeholder=(
        "e.g., Query metrics examining body mass index distributions..."
    ),
)


# -----------------------------------------------------------------------------
# QUERY EXECUTION
# -----------------------------------------------------------------------------
if user_query:

    if not selected_sources:

        st.warning(
            "⚠️ Please select at least one source "
            "in the sidebar to execute a search."
        )

    else:

        # ---------------------------------------------------------------------
        # Stage 1 — spaCy query parsing
        # ---------------------------------------------------------------------
        with st.spinner(
            "Extracting structural semantics..."
        ):

            start_parse = time.perf_counter()

            parsed_data = parser.parse(
                user_query
            )

            parse_seconds = (
                time.perf_counter()
                - start_parse
            )

        c_tok, c_dem = st.columns(2)

        with c_tok:

            st.caption(
                "**Extracted Keywords:** "
                f"{parsed_data.get('extracted_keywords', 'None')}"
            )

        with c_dem:

            st.caption(
                "**Inferred Demographics:** "
                f"{parsed_data.get('inferred_demographics', 'General')}"
            )

        # ---------------------------------------------------------------------
        # Stage 2 — ChromaDB retrieval
        # ---------------------------------------------------------------------
        with st.spinner(
            "Searching vector store..."
        ):

            candidates, retrieval_timing = (
                query_chroma_vector_db(
                    user_query,
                    selected_sources,
                    max_distance,
                    top_k=n_results * 2,
                    return_timing=True,
                )
            )

        # ---------------------------------------------------------------------
        # Stage 3 — PyTorch reranking
        # ---------------------------------------------------------------------
        rerank_timing = {
            "tokenization_seconds": 0.0,
            "inference_seconds": 0.0,
            "total_seconds": 0.0,
            "candidate_count": len(candidates),
        }

        if use_reranker and candidates:

            with st.spinner(
                "Executing PyTorch neural reranking..."
            ):

                (
                    final_results,
                    rerank_timing,
                ) = reranker.rerank(
                    user_query,
                    candidates,
                    return_timing=True,
                )

                final_results = final_results[
                    :n_results
                ]

        else:

            final_results = candidates[
                :n_results
            ]

        # ---------------------------------------------------------------------
        # Timing summary
        # ---------------------------------------------------------------------
        total_retrieval_seconds = (
            parse_seconds
            + retrieval_timing["retrieval_total_seconds"]
            + (
                rerank_timing["total_seconds"]
                if use_reranker
                else 0.0
            )
        )

        st.markdown("#### ⏱️ Pipeline Timing")

        timing_cols = st.columns(5)

        with timing_cols[0]:
            st.metric(
                "spaCy",
                f"{parse_seconds:.3f}s",
            )

        with timing_cols[1]:
            st.metric(
                "Embedding",
                f"{retrieval_timing['embedding_seconds']:.3f}s",
            )

        with timing_cols[2]:
            st.metric(
                "ChromaDB",
                f"{retrieval_timing['chroma_seconds']:.3f}s",
            )

        with timing_cols[3]:
            st.metric(
                "PyTorch",
                (
                    f"{rerank_timing['total_seconds']:.3f}s"
                    if use_reranker
                    else "Off"
                ),
            )

        with timing_cols[4]:
            st.metric(
                "Retrieval Total",
                f"{total_retrieval_seconds:.3f}s",
            )

        # ---------------------------------------------------------------------
        # Results
        # ---------------------------------------------------------------------
        st.markdown(
            f"#### 📦 Retrieved Context Blocks "
            f"({len(final_results)} Matches)"
        )

        if not final_results:

            st.info(
                "No records matched your search criteria "
                "within the current distance threshold."
            )

        else:

            for idx, item in enumerate(
                final_results,
                1,
            ):

                with st.container():

                    col_meta, col_rerank, col_dist = st.columns(
                        [3, 1, 1]
                    )

                    with col_meta:

                        st.markdown(
                            f"**Match #{idx}: "
                            f"`{item.get('variable_id', 'N/A')}`** — "
                            f"{item.get('variable_name', 'Unnamed Variable')}"
                        )

                        st.caption(
                            f"Origin Registry: "
                            f"`{item.get('source_dataset', 'Unknown')}` | "
                            f"Identifier: "
                            f"`{item.get('table_id', 'N/A')}`"
                        )

                    with col_rerank:

                        if use_reranker:

                            st.metric(
                                label="Neural Relevance",
                                value=(
                                    f"{item.get('rerank_score', 0.0):.3f}"
                                ),
                            )

                        else:

                            st.metric(
                                label="Vector Similarity",
                                value=(
                                    f"{item.get('vector_score', 0.0):.3f}"
                                ),
                            )

                    with col_dist:

                        st.metric(
                            label="Chroma Distance",
                            value=(
                                f"{item.get('distance', 0.0):.3f}"
                            ),
                        )

                    st.text_area(
                        label="Metadata Context Payload",
                        value=item.get(
                            "description",
                            "",
                        ),
                        height=70,
                        key=f"doc_{idx}",
                        disabled=True,
                    )

                    st.markdown("---")


# -----------------------------------------------------------------------------
# LIVE DATA EXPORT
# -----------------------------------------------------------------------------
if final_results:

    export_records = [
        {
            "source_dataset": m.get(
                "source_dataset",
                "",
            ),
            "variable_id": m.get(
                "variable_id",
                "",
            ),
            "description": m.get(
                "description",
                "",
            ),
        }
        for m in final_results
    ]

    st.markdown(
        "### 🛠️ Live Observations Export & Ingestion"
    )

    with st.spinner(
        "Fetching live health observations across "
        "retrieved match endpoints..."
    ):

        start_live_data = time.perf_counter()

        df_raw_combined = fetch_all_live_data(
            export_records,
            limit_per_var=50,
        )

        live_data_seconds = (
            time.perf_counter()
            - start_live_data
        )

    st.caption(
        f"Live data retrieval time: "
        f"{live_data_seconds:.3f} seconds"
    )

    tab_table, tab_code = st.tabs(
        [
            "📊 Raw Observation Data",
            "🐍 Python Import Script",
        ]
    )

    with tab_table:

        if not df_raw_combined.empty:

            st.markdown(
                f"**Retrieved Microdata Records "
                f"({len(df_raw_combined)} rows)**"
            )

            st.dataframe(
                df_raw_combined,
                use_container_width=True,
                hide_index=True,
            )

            col_csv, col_json = st.columns(2)

            with col_csv:

                st.download_button(
                    label=(
                        "📥 Download Fetched "
                        "Observations (CSV)"
                    ),
                    data=df_raw_combined.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="atlas_live_observations.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_live_csv_btn",
                )

            with col_json:

                st.download_button(
                    label=(
                        "📥 Download Fetched "
                        "Observations (JSON)"
                    ),
                    data=df_raw_combined.to_json(
                        orient="records",
                        indent=2,
                    ),
                    file_name="atlas_live_observations.json",
                    mime="application/json",
                    use_container_width=True,
                    key="download_live_json_btn",
                )

        else:

            st.info(
                "No live observation records could be automatically "
                "fetched for the current matches. Check API connectivity."
            )

    with tab_code:

        st.markdown(
            "Copy and paste this Python script to pull these "
            "raw observations directly into your workspace or notebook:"
        )

        target_ids = [
            str(r["variable_id"])
            for r in export_records
            if r.get("variable_id")
        ]

        python_snippet = f'''import pandas as pd
import requests

# 1. Target Variables retrieved from Global Health Atlas search
target_variables = {target_ids}

def fetch_atlas_health_data(variable_id):
    """Fetches live health observation records from open API endpoints."""
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

    df_combined = pd.concat(
        all_data,
        ignore_index=True,
    )

    print(
        f"Successfully loaded "
        f"{{len(df_combined)}} live observation rows."
    )

    print(df_combined.head())

else:

    print("No observations returned.")
'''

        st.code(
            python_snippet,
            language="python",
        )


# -----------------------------------------------------------------------------
# LLM SYNTHESIS
# -----------------------------------------------------------------------------
if final_results:

    st.markdown(
        "### 🤖 Generation Phase (Live RAG Synthesis)"
    )

    with st.expander(
        "See live synthesis tracking",
        expanded=True,
    ):

        context_str = "\n\n".join(
            [
                (
                    f"[Match #{i + 1} | "
                    f"Source: {m.get('source_dataset', 'N/A')} | "
                    f"Code: {m.get('variable_id', 'N/A')}]\n"
                    f"{m.get('description', '')}"
                )
                for i, m in enumerate(final_results)
            ]
        )

        system_prompt = (
            "You are an expert global health data translation assistant. "
            "Synthesize a concise, integrated answer to the user's question "
            "using ONLY the provided metadata context blocks. "
            "If the answer cannot be verified by the text snippets, "
            "explicitly state that the information is not available "
            "in the current Atlas registers. "
            "Do not invent details."
        )

        user_prompt = (
            f"Context Blocks:\n{context_str}\n\n"
            f"User Question: {user_query}"
        )

        st.caption(
            "**System Prompt:** "
            + system_prompt
        )

        if OLLAMA_AVAILABLE:

            st.markdown(
                "**Generated Response "
                "(Streaming from Llama 3.2 via Ollama):**"
            )

            def ollama_stream_generator():

                start_llm = time.perf_counter()

                try:

                    stream = ollama.chat(
                        model="llama3.2",
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": user_prompt,
                            },
                        ],
                        stream=True,
                    )

                    for chunk in stream:

                        yield chunk[
                            "message"
                        ][
                            "content"
                        ]

                except Exception as err:

                    yield (
                        "\n⚠️ Local Ollama daemon "
                        f"unreachable: {err}. "
                        "Ensure `ollama serve` is running."
                    )

                finally:

                    llm_seconds = (
                        time.perf_counter()
                        - start_llm
                    )

                    st.session_state[
                        "last_llm_seconds"
                    ] = llm_seconds

            st.write_stream(
                ollama_stream_generator
            )

            if (
                "last_llm_seconds"
                in st.session_state
            ):

                st.caption(
                    "LLM generation time: "
                    f"{st.session_state['last_llm_seconds']:.3f} seconds"
                )

        else:

            st.info(
                "💡 **Cloud Mode Active:** Ollama local LLM inference "
                "is disabled in cloud hosting environments. "
                "The dense vector matches and PyTorch neural reranked "
                "contexts above represent your retrieved RAG context window."
            )
