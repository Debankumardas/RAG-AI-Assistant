
import json
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VECTORSTORE_DIR = BASE_DIR / "vectorstore"
MANIFEST_PATH = VECTORSTORE_DIR / "manifest.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"

CHUNK_FORMAT_VERSION = 2


# ============================================================
# ENVIRONMENT AND MODEL INITIALIZATION
# ============================================================

load_dotenv()

client = None

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# VECTORSTORE STATE
# ============================================================

index = None
chunks = []

_index_signature = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _clear_vectorstore():
    """Clear the currently loaded vectorstore from memory."""

    global index, chunks, _index_signature

    index = None
    chunks = []
    _index_signature = None


def _get_manifest_data():
    """
    Read and validate manifest.json.

    Only metadata-aware chunk format version 2 is accepted.
    """

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Vectorstore manifest not found: {MANIFEST_PATH}"
        )

    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict):
        raise ValueError(
            "Vectorstore manifest must contain a JSON object."
        )

    required_fields = {
        "generation_id",
        "index_path",
        "chunks_path",
        "chunk_count",
        "embedding_dimension",
        "document_name",
        "page_count",
        "chunk_format_version",
    }

    missing_fields = required_fields - manifest.keys()

    if missing_fields:
        raise ValueError(
            f"Manifest is missing required fields: {missing_fields}"
        )

    if (
        not isinstance(manifest["generation_id"], str)
        or not manifest["generation_id"].strip()
    ):
        raise ValueError("Invalid generation ID in manifest.")

    for field in ("index_path", "chunks_path"):
        if (
            not isinstance(manifest[field], str)
            or not manifest[field].strip()
        ):
            raise ValueError(f"Invalid {field} in manifest.")

    if (
        not isinstance(manifest["chunk_count"], int)
        or isinstance(manifest["chunk_count"], bool)
        or manifest["chunk_count"] <= 0
    ):
        raise ValueError("Invalid chunk count in manifest.")

    if (
        not isinstance(manifest["embedding_dimension"], int)
        or isinstance(manifest["embedding_dimension"], bool)
        or manifest["embedding_dimension"] <= 0
    ):
        raise ValueError("Invalid embedding dimension in manifest.")

    if (
        not isinstance(manifest["document_name"], str)
        or not manifest["document_name"].strip()
    ):
        raise ValueError("Invalid document name in manifest.")

    if (
        not isinstance(manifest["page_count"], int)
        or isinstance(manifest["page_count"], bool)
        or manifest["page_count"] <= 0
    ):
        raise ValueError("Invalid page count in manifest.")

    if manifest["chunk_format_version"] != CHUNK_FORMAT_VERSION:
        raise ValueError(
            "Unsupported chunk format version. "
            f"Expected {CHUNK_FORMAT_VERSION}, "
            f"got {manifest['chunk_format_version']}."
        )

    return manifest


def _resolve_manifest_path(relative_path):
    """
    Resolve a manifest path safely within VECTORSTORE_DIR.

    Prevents paths such as ../../some_other_file from being loaded.
    """

    relative_path = Path(relative_path)

    if relative_path.is_absolute():
        raise ValueError("Manifest paths must be relative.")

    base_path = VECTORSTORE_DIR.resolve()
    resolved_path = (base_path / relative_path).resolve()

    try:
        resolved_path.relative_to(base_path)
    except ValueError:
        raise ValueError(
            "Manifest path points outside the vectorstore directory."
        )

    return resolved_path


def _validate_chunk_records(chunk_records, manifest):
    """Validate the structure and metadata of stored chunk records."""

    if not isinstance(chunk_records, list):
        raise ValueError("Stored chunks must be a list.")

    if len(chunk_records) != manifest["chunk_count"]:
        raise ValueError(
            "Stored chunk count does not match manifest."
        )

    for expected_id, chunk in enumerate(chunk_records):

        if not isinstance(chunk, dict):
            raise ValueError(
                f"Chunk {expected_id} must be a dictionary."
            )

        required_fields = {
            "chunk_id",
            "text",
            "page_number",
            "document_name",
        }

        missing_fields = required_fields - chunk.keys()

        if missing_fields:
            raise ValueError(
                f"Chunk {expected_id} is missing fields: {missing_fields}"
            )

        if (
            not isinstance(chunk["chunk_id"], int)
            or isinstance(chunk["chunk_id"], bool)
            or chunk["chunk_id"] != expected_id
        ):
            raise ValueError(
                f"Invalid chunk ID at position {expected_id}."
            )

        if (
            not isinstance(chunk["text"], str)
            or not chunk["text"].strip()
        ):
            raise ValueError(
                f"Chunk {expected_id} contains invalid text."
            )

        if (
            not isinstance(chunk["page_number"], int)
            or isinstance(chunk["page_number"], bool)
            or chunk["page_number"] < 1
        ):
            raise ValueError(
                f"Chunk {expected_id} has an invalid PDF page number."
            )

        if (
            not isinstance(chunk["document_name"], str)
            or chunk["document_name"] != manifest["document_name"]
        ):
            raise ValueError(
                f"Chunk {expected_id} has an invalid document name."
            )


def get_vectorstore_signature():
    """
    Return a signature representing the current vectorstore generation.

    Returns None if the manifest or referenced files are missing or invalid.
    """

    try:
        manifest = _get_manifest_data()

        index_path = _resolve_manifest_path(manifest["index_path"])
        chunks_path = _resolve_manifest_path(manifest["chunks_path"])

        if not index_path.is_file() or not chunks_path.is_file():
            return None

        manifest_stat = MANIFEST_PATH.stat()
        index_stat = index_path.stat()
        chunks_stat = chunks_path.stat()

        return (
            manifest["generation_id"],
            manifest_stat.st_mtime_ns,
            manifest_stat.st_size,
            str(index_path),
            index_stat.st_mtime_ns,
            index_stat.st_size,
            str(chunks_path),
            chunks_stat.st_mtime_ns,
            chunks_stat.st_size,
        )

    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        return None


# ============================================================
# VECTORSTORE LOADING
# ============================================================

def load_vectorstore(force_reload=False):
    """
    Load and validate the active metadata-aware vectorstore.

    Global state is updated only after all validation succeeds.
    """

    global index, chunks, _index_signature

    current_signature = get_vectorstore_signature()

    if current_signature is None:
        print("[WARNING] No valid vectorstore generation found.")
        _clear_vectorstore()
        return False

    if (
        not force_reload
        and index is not None
        and _index_signature == current_signature
    ):
        return True

    try:
        print("[INFO] Loading vectorstore generation...")

        manifest = _get_manifest_data()

        index_path = _resolve_manifest_path(manifest["index_path"])
        chunks_path = _resolve_manifest_path(manifest["chunks_path"])

        new_index = faiss.read_index(str(index_path))

        with open(chunks_path, "rb") as file:
            new_chunks = pickle.load(file)

        _validate_chunk_records(new_chunks, manifest)

        if new_index.ntotal != len(new_chunks):
            raise ValueError(
                "FAISS index and chunk count do not match. "
                f"Index: {new_index.ntotal}, "
                f"chunks: {len(new_chunks)}"
            )

        if new_index.d != manifest["embedding_dimension"]:
            raise ValueError(
                "FAISS embedding dimension does not match manifest."
            )

        updated_signature = get_vectorstore_signature()

        if updated_signature != current_signature:
            raise RuntimeError(
                "Vectorstore changed while it was being loaded. "
                "Retry the operation."
            )

        # Publish state only after successful validation.
        index = new_index
        chunks = new_chunks
        _index_signature = current_signature

        print("[SUCCESS] Vectorstore loaded successfully.")
        print(f"[INFO] Generation: {manifest['generation_id']}")
        print(f"[INFO] Document: {manifest['document_name']}")
        print(f"[INFO] Pages: {manifest['page_count']}")
        print(f"[INFO] Chunks loaded: {len(chunks)}")
        print(f"[INFO] Embedding dimension: {index.d}")

        return True

    except Exception as error:
        print(f"[ERROR] Failed to load vectorstore: {error}")
        _clear_vectorstore()
        return False


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_context(query, top_k=3):
    """
    Retrieve relevant metadata-aware chunk records.

    Returns:
        list[dict]: Retrieved chunks containing text and source metadata.
    """

    if not isinstance(query, str) or not query.strip():
        return []

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k <= 0
    ):
        return []

    current_signature = get_vectorstore_signature()

    if current_signature is None:
        print("[WARNING] No valid vectorstore available.")
        _clear_vectorstore()
        return []

    if index is None or _index_signature != current_signature:
        if not load_vectorstore():
            return []

    if index is None or not chunks:
        return []

    try:
        query_embedding = embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if (
            query_embedding.ndim != 2
            or query_embedding.shape[0] != 1
            or not np.isfinite(query_embedding).all()
        ):
            raise ValueError("Invalid query embedding.")

        if query_embedding.shape[1] != index.d:
            raise ValueError(
                "Query embedding dimension does not match FAISS index."
            )

        k = min(top_k, index.ntotal)

        if k <= 0:
            return []

        scores, indices = index.search(query_embedding, k)

        retrieved_chunks = []

        for chunk_index in indices[0]:
            if 0 <= chunk_index < len(chunks):
                retrieved_chunks.append(chunks[chunk_index])

        return retrieved_chunks

    except Exception as error:
        print(f"[ERROR] Retrieval failed: {error}")
        return []


# ============================================================
# GEMINI CLIENT
# ============================================================

def _get_gemini_client():
    """Create and cache the Gemini client."""

    global client

    if client is not None:
        return client

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )

    client = genai.Client(api_key=api_key)

    return client


# ============================================================
# RAG RESPONSE GENERATION
# ============================================================

def ask_rag(query):
    """
    Generate an answer using retrieved document context and Gemini.

    Source metadata is included in the prompt so answers can cite
    the original PDF pages.
    """

    if not isinstance(query, str) or not query.strip():
        return "Please enter a valid question."

    print("\n" + "=" * 60)
    print("RAG QUERY")
    print("=" * 60)

    print(f"\nQuestion: {query}")

    retrieved_chunks = retrieve_context(query, top_k=3)

    if not retrieved_chunks:
        return (
            "I couldn't find relevant information in the uploaded "
            "documents. Please upload and process a document first."
        )

    context_parts = []

    for chunk in retrieved_chunks:

        source = (
            f"Document: {chunk['document_name']}\n"
            f"PDF page: {chunk['page_number']}\n"
            f"Chunk ID: {chunk['chunk_id']}"
        )

        context_parts.append(
            f"[{source}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document-based AI assistant.

Answer the user's question using ONLY the information provided
in the document context below.

Rules:
1. Do not invent facts or information.
2. Do not use outside knowledge.
3. If the answer is not available in the context, clearly say so.
4. Give a clear and concise answer.
5. Organize the answer into bullet points when appropriate.
6. When making a factual claim, cite its source using this exact format:
   [Source: document_name, PDF page N]
7. Only cite documents and page numbers explicitly provided in the context.
8. Do not invent citations.
9. If multiple pages support the answer, cite each relevant page.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    try:
        gemini_client = _get_gemini_client()

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        answer = response.text

        if not answer or not answer.strip():
            return "The AI model returned an empty response."

        return answer.strip()

    except Exception as error:
        print(f"[ERROR] Gemini response generation failed: {error}")

        return (
            "An error occurred while generating the answer. "
            "Please check your Gemini API configuration and try again."
        )


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    print("\nTesting vectorstore loading...")

    if load_vectorstore(force_reload=True):

        print("\nVectorstore loaded successfully.")

        test_query = "What is the main topic of the document?"

        print(f"\nTest query: {test_query}")

        results = retrieve_context(test_query, top_k=3)

        print("\nRetrieved context:")

        for i, chunk in enumerate(results, start=1):
            print(f"\n--- Chunk {i} ---")
            print(f"Document: {chunk['document_name']}")
            print(f"PDF page: {chunk['page_number']}")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(chunk["text"])

    else:
        print(
            "\nNo valid metadata-aware vectorstore found. "
            "Process a PDF document first."
        )
