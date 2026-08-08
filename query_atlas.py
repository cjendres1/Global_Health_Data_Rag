# query_atlas.py
import sqlite3
import hashlib
import numpy as np

def generate_vector(text: str, dimensionality: int = 384) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vector = rng.normal(loc=0.0, scale=1.0, size=dimensionality)
    norm = np.linalg.norm(vector)
    return (vector / norm) if norm > 0 else vector

def query_vector_store(query_text: str, top_k: int = 5, db_path: str = "./global_health_atlas.db"):
    query_vec = generate_vector(query_text)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, source, indicator_id, short_name, document, vector FROM health_vector_store")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("Database is empty!")
        return

    scores = []
    for row in rows:
        doc_id, source, indicator_id, short_name, document, vec_bytes = row
        db_vec = np.frombuffer(vec_bytes, dtype=np.float32)
        
        # Cosine Similarity
        similarity = np.dot(query_vec, db_vec)
        scores.append((similarity, source, indicator_id, short_name, document))

    scores.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\n🔍 Top {top_k} Matches for Query: '{query_text}'\n" + "="*60)
    for score, source, ind_id, name, doc in scores[:top_k]:
        print(f"[{source}] ID: {ind_id} | Similarity: {score:.4f}")
        print(f"     Name: {name}")
        print(f"     Details: {doc}\n")

if __name__ == "__main__":
    import sys
    search_term = sys.argv[1] if len(sys.argv) > 1 else "Diabetes mellitus diagnosis"
    query_vector_store(search_term)