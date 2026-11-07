
import os
import pickle
from pathlib import Path

import faiss
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
CHUNKS_PATH = VECTORSTORE_DIR / "chunks.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# LOAD GEMINI
# ============================================================

load_dotenv()

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
# VECTOR STORE STATE
# ============================================================

index = None
chunks = []

_index_signature = None


def get_vectorstore_signature():
    """
    Return a signature describing the current vectorstore files.

    The signature changes when either file is replaced or modified.
    """
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None

    index_stat = INDEX_PATH.stat()
    chunks_stat = CHUNKS_PATH.stat()

    return (
        index_stat.st_mtime_ns,
        index_stat.st_size,
        chunks_stat.st_mtime_ns,
        chunks_stat.st_size,
    )


def load_vectorstore(force_reload=False):
    """
    Load the vectorstore when it is not loaded or its files change.

    Returns True when a valid vectorstore is available.
    """
    global index, chunks, _index_signature

    signature = get_vectorstore_signature()

    if signature is None:
        index = None
        chunks = []
        _index_signature = None

        print("Vectorstore files not found.")

        return False

    if not force_reload and signature == _index_signature:
        return index is not None and len(chunks) > 0

    print("Loading FAISS vector store...")

    try:
        new_index = faiss.read_index(str(INDEX_PATH))

        with open(CHUNKS_PATH, "rb") as file:
            new_chunks = pickle.load(file)

        if new_index.ntotal != len(new_chunks):
            raise ValueError(
                "FAISS index and chunk count do not match: "
                f"{new_index.ntotal} vectors, "
                f"{len(new_chunks)} chunks."
            )

        # Update active state only after both files load successfully.
        index = new_index
        chunks = new_chunks
        _index_signature = signature

        print("Vectorstore loaded.")
        print(f"Number of chunks: {len(chunks)}")
        print(f"Embedding dimension: {index.d}")

        return True

    except Exception as error:
        print(f"Failed to load vectorstore: {error}")

        return False


# Load the initial vectorstore.
load_vectorstore()


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_context(query, top_k=3):

    if not query or not query.strip():
        return []

    if top_k <= 0:
        return []

    if not load_vectorstore():
        return []

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype("float32")

    search_k = min(top_k, index.ntotal)

    if search_k == 0:
        return []

    distances, indices = index.search(
        query_embedding,
        search_k
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if idx < 0 or idx >= len(chunks):
            continue

        retrieved_chunks.append(chunks[idx])

    return retrieved_chunks


# ============================================================
# GENERATE RAG ANSWER
# ============================================================

def ask_rag(query, top_k=3):

    if not query or not query.strip():
        return "Please enter a question."

    print("\n" + "=" * 60)
    print("RAG QUERY")
    print("=" * 60)

    print(f"Question: {query}")

    retrieved_chunks = retrieve_context(
        query,
        top_k=top_k
    )

    if not retrieved_chunks:
        return (
            "I couldn't find relevant information "
            "in the uploaded document."
        )

    context = "\n\n".join(retrieved_chunks)

    print(f"Retrieved chunks: {len(retrieved_chunks)}")

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

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:
        return "The AI model returned an empty response."

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
