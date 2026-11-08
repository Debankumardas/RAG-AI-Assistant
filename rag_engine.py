
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
GENERATIONS_DIR = VECTORSTORE_DIR / "generations"
MANIFEST_PATH = VECTORSTORE_DIR / "manifest.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"

CHUNK_FORMAT_VERSION = 2

load_dotenv()


# ============================================================
# SHARED MODELS
# ============================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    local_files_only=True,
)

client = None


# ============================================================
# GEMINI CLIENT
# ============================================================

def _get_gemini_client():
    """Create the Gemini client lazily."""

    global client

    if client is not None:
        return client

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)

    return client


# ============================================================
# VECTORSTORE PATHS
# ============================================================

def _get_vectorstore_paths(vectorstore_dir=None):
    """
    Resolve vectorstore paths.

    If no directory is supplied, use the default vectorstore.
    Otherwise, use the supplied session-specific directory.
    """

    if vectorstore_dir is None:
        directory = Path(VECTORSTORE_DIR).resolve()
        manifest_path = Path(MANIFEST_PATH).resolve()
    else:
        directory = Path(vectorstore_dir).resolve()
        manifest_path = directory / "manifest.json"

    generations_dir = directory / "generations"

    return directory, generations_dir, manifest_path


# ============================================================
# LOAD AND VALIDATE MANIFEST
# ============================================================

def _get_manifest_data(vectorstore_dir=None):
    """Read and validate the vectorstore manifest."""

    _, _, manifest_path = _get_vectorstore_paths(
        vectorstore_dir
    )

    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)

    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(manifest, dict):
        return None

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

    if not required_fields.issubset(manifest):
        return None

    if not isinstance(manifest["generation_id"], str):
        return None

    if not manifest["generation_id"].strip():
        return None

    for field in ("index_path", "chunks_path"):
        if not isinstance(manifest[field], str):
            return None

        if not manifest[field].strip():
            return None

    for field in (
        "chunk_count",
        "embedding_dimension",
        "page_count",
        "chunk_format_version",
    ):
        if (
            not isinstance(manifest[field], int)
            or isinstance(manifest[field], bool)
        ):
            return None

    if manifest["chunk_count"] <= 0:
        return None

    if manifest["embedding_dimension"] <= 0:
        return None

    if manifest["page_count"] <= 0:
        return None

    if (
        manifest["chunk_format_version"]
        != CHUNK_FORMAT_VERSION
    ):
        return None

    if not isinstance(manifest["document_name"], str):
        return None

    if not manifest["document_name"].strip():
        return None

    return manifest


# ============================================================
# SAFE MANIFEST PATH RESOLUTION
# ============================================================

def _resolve_manifest_path(
    relative_path,
    vectorstore_dir=None,
):
    """
    Resolve a manifest path safely.

    Reject paths that escape the vectorstore directory.
    """

    if not isinstance(relative_path, str):
        return None

    if not relative_path.strip():
        return None

    directory, _, _ = _get_vectorstore_paths(
        vectorstore_dir
    )

    try:
        base_directory = directory.resolve()
        resolved_path = (
            base_directory / relative_path
        ).resolve()

        resolved_path.relative_to(base_directory)

        return resolved_path

    except (ValueError, OSError):
        return None


# ============================================================
# CHUNK METADATA VALIDATION
# ============================================================

def _validate_chunk_records(chunk_records, manifest):
    """Validate stored chunk records against the manifest."""

    if not isinstance(chunk_records, list):
        return False

    if len(chunk_records) != manifest["chunk_count"]:
        return False

    for expected_id, record in enumerate(chunk_records):

        if not isinstance(record, dict):
            return False

        required_fields = {
            "chunk_id",
            "text",
            "page_number",
            "document_name",
        }

        if not required_fields.issubset(record):
            return False

        chunk_id = record["chunk_id"]
        text = record["text"]
        page_number = record["page_number"]
        document_name = record["document_name"]

        if (
            not isinstance(chunk_id, int)
            or isinstance(chunk_id, bool)
            or chunk_id != expected_id
        ):
            return False

        if not isinstance(text, str) or not text.strip():
            return False

        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number <= 0
        ):
            return False

        if document_name != manifest["document_name"]:
            return False

    return True


# ============================================================
# VECTORSTORE SIGNATURE
# ============================================================

def get_vectorstore_signature(vectorstore_dir=None):
    """
    Return a signature describing the current vectorstore.

    Returns None when the manifest or referenced files are
    missing or invalid.
    """

    manifest = _get_manifest_data(vectorstore_dir)

    if manifest is None:
        return None

    index_path = _resolve_manifest_path(
        manifest["index_path"],
        vectorstore_dir,
    )

    chunks_path = _resolve_manifest_path(
        manifest["chunks_path"],
        vectorstore_dir,
    )

    if index_path is None or chunks_path is None:
        return None

    try:
        index_stat = index_path.stat()
        chunks_stat = chunks_path.stat()

    except OSError:
        return None

    return (
        manifest["generation_id"],
        index_stat.st_size,
        index_stat.st_mtime_ns,
        chunks_stat.st_size,
        chunks_stat.st_mtime_ns,
    )


# ============================================================
# LOAD VECTORSTORE
# ============================================================

def load_vectorstore(
    force_reload=False,
    vectorstore_dir=None,
):
    """
    Load and validate a vectorstore.

    Returns:
        (index, chunks) on success.
        None on failure.

    No shared mutable retrieval state is maintained.
    """

    manifest = _get_manifest_data(vectorstore_dir)

    if manifest is None:
        return None

    index_path = _resolve_manifest_path(
        manifest["index_path"],
        vectorstore_dir,
    )

    chunks_path = _resolve_manifest_path(
        manifest["chunks_path"],
        vectorstore_dir,
    )

    if index_path is None or chunks_path is None:
        return None

    if not index_path.is_file() or not chunks_path.is_file():
        return None

    try:
        loaded_index = faiss.read_index(str(index_path))

        with open(chunks_path, "rb") as file:
            loaded_chunks = pickle.load(file)

    except Exception:
        return None

    if not _validate_chunk_records(
        loaded_chunks,
        manifest,
    ):
        return None

    if loaded_index.ntotal != manifest["chunk_count"]:
        return None

    if loaded_index.d != manifest["embedding_dimension"]:
        return None

    if loaded_index.ntotal != len(loaded_chunks):
        return None

    # Confirm that the manifest still references the same
    # generation after loading the files.
    latest_manifest = _get_manifest_data(vectorstore_dir)

    if latest_manifest is None:
        return None

    if (
        latest_manifest["generation_id"]
        != manifest["generation_id"]
    ):
        return None

    return loaded_index, loaded_chunks


# ============================================================
# RETRIEVE RELEVANT CONTEXT
# ============================================================

def retrieve_context(
    query,
    top_k=3,
    vectorstore_dir=None,
):
    """Retrieve the most relevant chunks for a query."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer.")

    vectorstore = load_vectorstore(
        vectorstore_dir=vectorstore_dir
    )

    if vectorstore is None:
        return []

    index, chunks = vectorstore

    try:
        query_embedding = embedding_model.encode(
            [query.strip()],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

    except Exception:
        return []

    if query_embedding.ndim != 2:
        return []

    if query_embedding.shape[0] != 1:
        return []

    if query_embedding.shape[1] != index.d:
        return []

    if not np.isfinite(query_embedding).all():
        return []

    try:
        scores, indices = index.search(
            query_embedding,
            min(top_k, index.ntotal),
        )

    except Exception:
        return []

    results = []

    for chunk_index in indices[0]:

        if chunk_index < 0:
            continue

        if chunk_index >= len(chunks):
            continue

        results.append(chunks[int(chunk_index)])

    return results


# ============================================================
# RAG RESPONSE
# ============================================================

def ask_rag(query, vectorstore_dir=None):
    """Retrieve context and generate an answer using Gemini."""

    if not isinstance(query, str) or not query.strip():
        return "Please enter a valid question."

    retrieved_chunks = retrieve_context(
        query.strip(),
        top_k=3,
        vectorstore_dir=vectorstore_dir,
    )

    if not retrieved_chunks:
        return (
            "No relevant document context was found. "
            "Please upload a document first or ask a question "
            "related to the uploaded document."
        )

    context_parts = []

    for chunk in retrieved_chunks:

        document_name = chunk["document_name"]
        page_number = chunk["page_number"]
        chunk_id = chunk["chunk_id"]
        chunk_text = chunk["text"]

        context_parts.append(
            f"[Document: {document_name}\n"
            f"PDF page: {page_number}\n"
            f"Chunk ID: {chunk_id}]\n"
            f"{chunk_text}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the supplied context.

Rules:
1. Do not use outside knowledge.
2. If the context does not contain the answer, say so clearly.
3. Do not invent facts, document names, page numbers, or citations.
4. Cite supporting information using this exact format:
   [Source: document_name, PDF page N]
5. Only cite document names and page numbers present in the context.
6. Be clear and concise.

CONTEXT:
{context}

USER QUESTION:
{query.strip()}

ANSWER:
"""

    try:
        gemini_client = _get_gemini_client()

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        answer = getattr(response, "text", None)

        if not answer or not answer.strip():
            return "The language model returned an empty response."

        return answer.strip()

    except Exception as error:
        print(f"Gemini generation error: {error}")

        return (
            "I encountered an error while generating the answer. "
            "Please check your Gemini API configuration and try again."
        )


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    question = input("Ask a question about your document: ")

    answer = ask_rag(question)

    print("\nANSWER:\n")
    print(answer)
