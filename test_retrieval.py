import pickle
import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# RAG RETRIEVAL TEST
# ============================================================

VECTORSTORE_DIR = "vectorstore"
INDEX_PATH = f"{VECTORSTORE_DIR}/index.faiss"
CHUNKS_PATH = f"{VECTORSTORE_DIR}/chunks.pkl"

print("=" * 60)
print("RAG RETRIEVAL TEST")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load embedding model
# ------------------------------------------------------------

print("\n[1/3] Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# ------------------------------------------------------------
# 2. Load FAISS index and chunks
# ------------------------------------------------------------

print("\n[2/3] Loading vector store...")

index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print(f"FAISS index loaded.")
print(f"Number of chunks: {len(chunks)}")
print(f"Vector dimension: {index.d}")


# ------------------------------------------------------------
# 3. Retrieval function
# ------------------------------------------------------------

def retrieve(query, top_k=3):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for distance, index_number in zip(distances[0], indices[0]):

        if index_number == -1:
            continue

        results.append({
            "chunk": chunks[index_number],
            "distance": float(distance)
        })

    return results


# ------------------------------------------------------------
# Test queries
# ------------------------------------------------------------

test_queries = [
    "What is the minimum attendance requirement?",
    "What are the laboratory rules?",
    "What items are not allowed in the examination hall?",
    "What are the library rules?"
]


print("\n" + "=" * 60)
print("TESTING RETRIEVAL")
print("=" * 60)


for query in test_queries:

    print("\n" + "-" * 60)
    print(f"QUERY: {query}")
    print("-" * 60)

    results = retrieve(query)

    for i, result in enumerate(results, start=1):

        print(f"\nRESULT {i}")
        print(f"Distance: {result['distance']:.4f}")
        print("-" * 40)
        print(result["chunk"])


print("\n" + "=" * 60)
print("RETRIEVAL TEST COMPLETED")
print("=" * 60)