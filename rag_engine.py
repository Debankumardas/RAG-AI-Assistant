import os
import pickle
import faiss

from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_MODEL = "gemini-3.6-flash"

VECTORSTORE_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(VECTORSTORE_DIR, "chunks.pkl")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# LOAD GEMINI
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    local_files_only=True
)

print("Embedding model loaded.")


# ============================================================
# LOAD VECTOR STORE
# ============================================================

print("Loading FAISS vector store...")

index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print(f"Vector store loaded.")
print(f"Number of chunks: {len(chunks)}")
print(f"Embedding dimension: {index.d}")


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_context(query, top_k=3):

    # Convert question into embedding
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    # Search FAISS
    distances, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if idx == -1:
            continue

        retrieved_chunks.append(chunks[idx])

    return retrieved_chunks


# ============================================================
# GENERATE RAG ANSWER
# ============================================================

def ask_rag(query, top_k=3):

    print("\n" + "=" * 60)
    print("RAG QUERY")
    print("=" * 60)

    print(f"Question: {query}")

    # --------------------------------------------------------
    # Retrieve relevant information
    # --------------------------------------------------------

    retrieved_chunks = retrieve_context(
        query,
        top_k=top_k
    )

    if not retrieved_chunks:

        return "I couldn't find relevant information in the uploaded document."

    # Combine retrieved chunks
    context = "\n\n".join(retrieved_chunks)

    print(f"Retrieved chunks: {len(retrieved_chunks)}")

    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = f"""
You are a document-based AI assistant.

Answer the user's question using ONLY the information
provided in the document context below.

If the answer cannot be found in the context, say:

"I couldn't find that information in the uploaded document."

Do not invent information.
Do not use outside knowledge.

DOCUMENT CONTEXT:
-----------------
{context}
-----------------

USER QUESTION:
{query}

Answer clearly and concisely.
"""

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("RAG ENGINE TEST")
    print("=" * 60)

    question = "What is the minimum attendance requirement?"

    answer = ask_rag(question)

    print("\nANSWER:")
    print("-" * 60)
    print(answer)
    print("-" * 60)