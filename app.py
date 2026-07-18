import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from query_parser import ClinicalQueryParser
from neural_reranker import PyTorchNeuralReranker

# -----------------------------------------------------------------------------
# 🛠️ CACHING OBJECT INITIALIZATIONS FOR PERFORMANCE
# -----------------------------------------------------------------------------
@st.cache_resource
def initialize_core_pipeline():
    """Caches large models in memory so they don't reload on user clicks."""
    parser = ClinicalQueryParser()
    reranker = PyTorchNeuralReranker()
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Connect to our local persistent storage layer built in Step 2
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")
    collection = chroma_client.get_collection(name="global_health_metadata")
    
    return parser, reranker, embedding_model, collection

# Initialize your advanced backend pipeline components
try:
    parser, reranker, embedding_model, collection = initialize_core_pipeline()
except Exception as e:
    st.error(f"Initialization Failed: Ensure you have run your PySpark and embedding scripts. Error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 🎨 STREAMLIT UI CONFIGURATION & TEXT LAYOUT
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Global Health Data Atlas AI", layout="wide")

st.title("🌐 Global Health Data Atlas AI")
st.caption("⚡ Advanced Data Portfolio: Unified Semantic Discovery & Automated Research Code Generation")
st.markdown("---")

# Layout: Split main interactive viewport into Sidebar and Main Content
with st.sidebar:
    st.header("⚙️ Pipeline Technical Specs")
    st.markdown("""
    This interface serves as an **AI-driven RAG architecture** designed to catalog and harmonize global health study metrics.
    
    **Architectural Components Built:**
    *   **Data Lake Layer:** `PySpark`
    *   **Linguistic Parsing Engine:** `spaCy (en_core_web_sm)`
    *   **Dense Vector Retrieval:** `Hugging Face + ChromaDB`
    *   **Neural Deep Sorting:** `PyTorch Cross-Encoder`
    """)
    st.info("💡 Pro-Tip: Try typing terms like 'body mass index metrics', 'gender profiles', or 'physical activity logs'.")

# Main Interface Query Input Entry Window Box
user_query = st.text_input(
    "What health variables or structural study tables are you looking for today?",
    placeholder="e.g., Query metrics examining body mass index distributions across target demographic profiles..."
)

if user_query:
    # -------------------------------------------------------------------------
    # STEP 1: SPACY QUERY UNDERSTANDING
    # -------------------------------------------------------------------------
    with st.spinner("Processing structural query semantics via spaCy..."):
        parsed_data = parser.parse(user_query)
    
    # Display linguistic data parsing metrics in neat expandable boxes
    col_k, col_d = st.columns(2)
    with col_k:
        st.subheader("🎯 NLP Tokens Extracted")
        st.write(parsed_data["extracted_keywords"] if parsed_data["extracted_keywords"] else "None identified.")
    with col_d:
        st.subheader("👥 Demographic Traps Caught")
        st.write(parsed_data["inferred_demographics"] if parsed_data["inferred_demographics"] else "General/No constraints.")

    # -------------------------------------------------------------------------
    # STEP 2 & 3: VECTOR MATCHING & PYTORCH RERANKING
    # -------------------------------------------------------------------------
    with st.spinner("Executing dense vector query matching & running PyTorch Cross-Encoder reranking..."):
        # Convert incoming raw string into dense multi-dimensional vector space representation
        query_vector = embedding_model.encode(user_query).tolist()
        
        # Pull candidate vectors out of ChromaDB collection memory
        raw_db_results = collection.query(query_embeddings=[query_vector], n_results=5)
        
        # Reformat payload dictionary blocks cleanly for the PyTorch forward pass evaluation logic
        candidates = []
        if raw_db_results and raw_db_results["metadatas"]:
            for meta in raw_db_results["metadatas"][0]:
                candidates.append(meta)
        
        # Execute deep learning sorting logic across retrieved item profiles
        final_reranked_results = reranker.rerank(user_query, candidates)

    # -------------------------------------------------------------------------
    # STEP 4: UI GENERATION TABLES & AUTOMATED CODE GENERATION LAYOUT
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🎯 Neural Ranked Cross-Study Variable Matches")
    
    if not final_reranked_results:
        st.warning("No highly aligned health data schema attributes located matching that signature parameters.")
    else:
        # Display results cleanly within native Streamlit tables
        display_list = []
        for rank, item in enumerate(final_reranked_results, 1):
            display_list.append({
                "Rank": rank,
                "Match Score": f"{item['rerank_score']:.4f}",
                "Source System": item["source_dataset"],
                "Table ID": item["table_id"],
                "Variable Code": item["variable_id"],
                "Variable Description Name": item["variable_name"],
                "Detailed Documentation Context": item["description"]
            })
        st.table(display_list)

        # Build dynamic programmatic snippet strings using top matched variables
        top_match = final_reranked_results[0]
        top_var = top_match["variable_id"]
        top_table = top_match["table_id"]
        top_source = top_match["source_dataset"]

        st.markdown("---")
        st.subheader("🤖 Automated Replication Pipeline Code Synthesis")
        st.caption(f"The RAG engine compiled your top asset target matches from **{top_source}** into runnable source structures:")

        # Present clean multi-column layout for side-by-side script evaluation formats
        tab_py, tab_r = st.tabs(["🐍 Executable Python Pipeline", "📊 Statistical R Code Pipeline"])
        
        with tab_py:
            st.markdown("#### Automated Query Extraction Script using Pandas")
            python_script = f"""import pandas as pd

def download_and_extract_features():
    print("Initiating direct programmatic access pipeline download target: {top_source}...")
    # Dynamic orchestration parameters configured for your search targets:
    target_table = "{top_table}"
    target_variable = "{top_var}"
    
    print(f"Isolating data coordinates for Variable Match: {{target_variable}} from Registry: {{target_table}}.")
    
    # Code template simulated generation for researcher integration profiles
    # In a production environment, this triggers a pandas or web scraper query hook
    return pd.DataFrame({{
        "participant_id": [1001, 1002, 1003],
        target_variable: [24.5, 31.2, 18.9],
        "study_origin": ["{top_source}"] * 3
    }})

df = download_and_extract_features()
print(df.head())"""
            st.code(python_script, language="python")

        with tab_r:
            st.markdown("#### Automated Statistical Script using R Survey Engines")
            r_script = f"""# Automated script built for {top_source} Research Data Alignment
library(dplyr)

load_global_study_metric <- function() {{
  message("Connecting to active metadata asset index profiles...")
  
  # Contextually inferred parameters extracted during your RAG interaction loop:
  study_source <- "{top_source}"
  table_id     <- "{top_table}"
  variable_id  <- "{top_var}"
  
  cat(sprintf("Constructing analysis matrix framework for feature: %s inside table %s\\n", variable_id, table_id))
  
  # Return structured frame elements for researcher convenience
  data.frame(
    SubjectID = c(1, 2, 3),
    MetricValue = c(24.5, 31.2, 18.9),
    DatasetSource = study_source
  )
}}

study_data <- load_global_study_metric()
summary(study_data)"""
            st.code(r_script, language="r")
            