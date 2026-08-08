# app.py
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingFunction
import json

# Set up page configurations for a professional portfolio layout
st.set_page_config(
    page_title="Global Health Data Atlas",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Embedded Static ChromaDB Persistent Client
@st.cache_resource
def get_chroma_client():
    try:
        # Instead of HttpClient(host="localhost"), we read a relative path directory
        return chromadb.PersistentClient(path="./chroma_db_directory")
    except Exception as e:
        st.error(f"Could not initialize the embedded vector registry: {str(e)}")
        return None

#chroma_client = get_chroma_client()
embedding_fn = WindowsSafeEmbedder(dimensionality=384)
chroma_client = chromadb.PersistentClient(path="./chroma_db_directory")
collection = chroma_client.get_collection(name="global_health_atlas", embedding_function=embedding_fn)

# --- FETCH COLLECTION METRICS & ACTIVE SOURCES ---
total_records = 0
source_count = 0
available_sources = []

if chroma_client:
    try:
        collection = chroma_client.get_collection(name="global_health_atlas")
        total_records = collection.count()
        
        # Pull metadata to identify active unique sources dynamically
        all_data = collection.get(include=["metadatas"])
        if all_data and all_data.get("metadatas"):
            unique_sources = set(meta.get("source") for meta in all_data["metadatas"] if meta)
            available_sources = sorted(list(unique_sources))
            source_count = len(available_sources)
    except Exception as e:
        st.warning("⚠️ Could not verify collection metrics. Ensure your pipeline ingestion has run and the Chroma server is active.")

# --- SIDEBAR: SYSTEM METRICS & ADVANCED FILTERS ---
with st.sidebar:
    st.title("🌐 Atlas Control Center")
    st.markdown("---")
    
    # Dynamic Search Tuning Section
    st.subheader("🎯 Search Filtering & Tuning")
    
    # 1. Filter by specific source registries
    selected_sources = st.multiselect(
        "Scope to Target Registries:",
        options=available_sources,
        default=available_sources,
        help="Uncheck sources to exclude their variables from the vector search space entirely."
    )
    
    # 2. Maximum Distance Threshold Slider (Cosine/L2 Precision Tuning)
    max_distance = st.slider(
        "Maximum Distance Threshold:",
        min_value=0.0,
        max_value=2.0,
        value=1.5,
        step=0.1,
        help="Lower values enforce strict matching. Higher values allow looser conceptual connections."
    )
    
    st.markdown("---")
    st.subheader("Ingestion Pipelines")
    # Dynamically verify pipeline completion indicators
    for src in available_sources:
        st.success(f"✅ {src} Active")
    
    st.markdown("---")
    st.caption("Developed as a highly decoupled multi-source data engineering showcase.")

# --- MAIN PAGE: HEADER & LAYOUT ---
st.title("🗺️ Global Health Data Atlas")
st.markdown(
    """
    This interactive platform demonstrates a robust **Retrieval-Augmented Generation (RAG)** pipeline. 
    It unifies disparate metadata registries and data schemas across international and domestic public health sources into a centralized, searchable vector space.
    """
)

st.markdown("---")

# Display high-level metrics banner
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Unified Records Ingested", value=total_records)
with col2:
    st.metric(label="ChromaDB Server Status", value="Connected (Port 8000)" if chroma_client else "Disconnected")
with col3:
    st.metric(label="Harmonized Schemas", value=f"{source_count} Sources Active")

st.markdown("### 🔍 Semantic Search Retrieval Layer")

# Search and parameter inputs
user_query = st.text_input(
    "Enter a clinical concept, health indicator, or survey question to explore across schemas:",
    placeholder="e.g., How is patient hypertension or high blood pressure categorized?"
)

n_results = st.slider("Max number of context matches to retrieve:", min_value=1, max_value=10, value=3)

# Execution phase
if user_query and chroma_client:
    if not selected_sources:
        st.warning("⚠️ Please select at least one source in the sidebar filters to execute a search.")
    else:
        with st.spinner("Traversing vector space embeddings..."):
            try:
                # Construct metadata filter block for ChromaDB standard client syntax
                # If all sources are selected, we don't need a constraint filter array
                where_filter = None
                if len(selected_sources) < len(available_sources):
                    if len(selected_sources) == 1:
                        where_filter = {"source": selected_sources[0]}
                    else:
                        where_filter = {"$or": [{"source": src} for src in selected_sources]}

                # Query the database
                results = collection.query(
                    query_texts=[user_query],
                    n_results=n_results * 2, # Fetch slightly more to filter distances comfortably
                    where=where_filter
                )
                
                # Unpack payload structures safely
                raw_documents = results['documents'][0]
                raw_metadatas = results['metadatas'][0]
                raw_distances = results['distances'][0]
                raw_ids = results['ids'][0]
                
                # Filter results based on the user's maximum distance threshold selection
                documents, metadatas, distances, ids = [], [], [], []
                for i in range(len(raw_documents)):
                    if raw_distances[i] <= max_distance:
                        documents.append(raw_documents[i])
                        metadatas.append(raw_metadatas[i])
                        distances.append(raw_distances[i])
                        ids.append(raw_ids[i])
                        if len(documents) == n_results:  # Stop once we fulfill user requested count
                            break
                
                st.markdown(f"#### 📦 Retrieved Context Blocks ({len(documents)} Matches)")
                
                if not documents:
                    st.info("No records found matching your query within the current distance threshold or selected sources.")
                
                # Render each match inside a clean custom container
                for i in range(len(documents)):
                    with st.container():
                        col_meta, col_dist = st.columns([4, 1])
                        
                        source_provider = metadatas[i].get("source", "UNKNOWN")
                        indicator_id = metadatas[i].get("indicator_id", "N/A")
                        
                        with col_meta:
                            st.markdown(f"**Match #{i+1}: Source ID: `{ids[i]}`**")
                            st.caption(f"Origin Registry: `{source_provider}` | Structural Code: `{indicator_id}`")
                        with col_dist:
                            # Vector distance confidence mapping metric
                            st.metric(label="Semantic Distance", value=f"{distances[i]:.4f}")
                            
                        st.text_area(
                            label="Unified Text Payload", 
                            value=documents[i], 
                            height=70, 
                            key=f"doc_{i}",
                            disabled=True
                        )
                        st.markdown("---")
                
# --- LIVE LOCAL LLM GENERATION PHASE ---
                st.markdown("### 🤖 Generation Phase (Live RAG Synthesis)")
                
                with st.expander("See live local synthesis tracking", expanded=True):
                    if not documents:
                        st.info("No context blocks available to pass to the inference model.")
                    else:
                        import ollama
                        
                        # Pack the distinct snippets cleanly into a text summary for the context prompt
                        context_str = "\n\n".join([
                            f"[Match #{idx+1} | Source: {meta.get('source')} | Code: {meta.get('indicator_id')}]\n{doc}"
                            for idx, (doc, meta) in enumerate(zip(documents, metadatas))
                        ])
                        
                        # Design a clean prompt grounding the model strictly to your ChromaDB findings
                        system_prompt = (
                            "You are an expert global health data translation assistant. Synthesize a concise, "
                            "integrated answer to the user's question using ONLY the provided metadata context blocks. "
                            "If the answer cannot be verified by the text snippets, explicitly state that the "
                            "information is not available in the current Atlas registers. Do not invent details."
                        )
                        
                        user_prompt = f"Context Blocks:\n{context_str}\n\nUser Question: {user_query}"
                        
                        st.markdown("**System Instructions Given to LLM:**")
                        st.caption(system_prompt)
                        
                        st.markdown("**Generated Response (Streaming from llama3.2):**")
                        
                        # Create a generator function to stream response chunks to st.write_stream
                        def ollama_stream_generator():
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
                        
                        # Stream the text output live onto the page interface
                        st.write_stream(ollama_stream_generator)

            except Exception as e:
                st.error(f"Error querying vector records or generating text: {str(e)}")
