
import json
import os
import pickle
import shutil
import uuid
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from document_processor import extract_pages_from_pdf, create_chunks


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VECTORSTORE_DIR = BASE_DIR / "vectorstore"
GENERATIONS_DIR = VECTORSTORE_DIR / "generations"
MANIFEST_PATH = VECTORSTORE_DIR / "manifest.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


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
# PROCESS DOCUMENT
# ============================================================

def process_document(pdf_path):
    """
    Process a PDF and publish a new versioned vectorstore generation.

    Each stored chunk retains:
    - Chunk ID
    - Text
    - Physical PDF page number
    - Document name

    The previous generation remains untouched until the new
    generation has been written, verified, and published.
    """

    pdf_path = os.path.abspath(pdf_path)

    if not os.path.exists(pdf_path):
        return {
            "success": False,
            "message": f"PDF not found: {pdf_path}"
        }

    document_name = os.path.basename(pdf_path)

    print("\n" + "=" * 60)
    print("DOCUMENT PROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: EXTRACT PDF PAGES
    # --------------------------------------------------------

    print("\n[1/4] Extracting PDF pages...")

    pages = extract_pages_from_pdf(pdf_path)

    if not pages:
        return {
            "success": False,
            "message": "No text could be extracted from the PDF."
        }

    total_characters = sum(
        len(page["text"]) for page in pages
    )

    print(f"Extracted pages: {len(pages)}")
    print(f"Extracted characters: {total_characters}")

    # --------------------------------------------------------
    # STEP 2: CREATE PAGE-AWARE CHUNKS
    # --------------------------------------------------------

    print("\n[2/4] Creating page-aware chunks...")

    chunks = create_chunks(
        pages,
        chunk_size=500,
        overlap=100,
        document_name=document_name
    )

    print(f"Created chunks: {len(chunks)}")

    if not chunks:
        return {
            "success": False,
            "message": "No chunks were created from the document."
        }

    # Extract only text for embedding.
    # Keep the complete dictionaries for storage.

    chunk_texts = [
        chunk["text"] for chunk in chunks
    ]

    # --------------------------------------------------------
    # STEP 3: GENERATE EMBEDDINGS
    # --------------------------------------------------------

    print("\n[3/4] Generating embeddings...")

    embeddings = embedding_model.encode(
        chunk_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a two-dimensional array."
        )

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            "Number of embeddings does not match number of chunks."
        )

    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("Generated embeddings are empty.")

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain invalid numeric values."
        )

    dimension = embeddings.shape[1]

    print(f"Embedding shape: {embeddings.shape}")

    # --------------------------------------------------------
    # STEP 4: CREATE FAISS INDEX
    # --------------------------------------------------------

    print("\n[4/4] Creating FAISS index...")

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # --------------------------------------------------------
    # CREATE A NEW GENERATION
    # --------------------------------------------------------

    generation_id = uuid.uuid4().hex

    generation_dir = GENERATIONS_DIR / generation_id

    index_path = generation_dir / "index.faiss"
    chunks_path = generation_dir / "chunks.pkl"

    manifest_temp_path = VECTORSTORE_DIR / (
        f"manifest.{generation_id}.tmp"
    )

    published = False

    try:
        # ----------------------------------------------------
        # CREATE DIRECTORIES
        # ----------------------------------------------------

        VECTORSTORE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        GENERATIONS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        generation_dir.mkdir(
            parents=True,
            exist_ok=False
        )

        # ----------------------------------------------------
        # SAVE FAISS INDEX
        # ----------------------------------------------------

        faiss.write_index(
            index,
            str(index_path)
        )

        # ----------------------------------------------------
        # SAVE STRUCTURED CHUNKS
        # ----------------------------------------------------

        with open(chunks_path, "wb") as file:

            pickle.dump(chunks, file)

            file.flush()
            os.fsync(file.fileno())

        # ----------------------------------------------------
        # VERIFY SAVED ARTIFACTS
        # ----------------------------------------------------

        verified_index = faiss.read_index(
            str(index_path)
        )

        with open(chunks_path, "rb") as file:
            verified_chunks = pickle.load(file)

        if verified_index.ntotal != len(verified_chunks):
            raise ValueError(
                "Saved FAISS index and chunks do not match."
            )

        if verified_index.d != dimension:
            raise ValueError(
                "Saved FAISS index has an incorrect embedding dimension."
            )

        if verified_chunks != chunks:
            raise ValueError(
                "Saved chunks do not match the original chunks."
            )

        # ----------------------------------------------------
        # PREPARE MANIFEST
        # ----------------------------------------------------

        manifest = {
            "generation_id": generation_id,
            "index_path": (
                f"generations/{generation_id}/index.faiss"
            ),
            "chunks_path": (
                f"generations/{generation_id}/chunks.pkl"
            ),
            "chunk_count": len(chunks),
            "embedding_dimension": dimension,
            "document_name": document_name,
            "page_count": len(pages),
            "chunk_format_version": 2
        }

        with open(
            manifest_temp_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                manifest,
                file,
                indent=4
            )

            file.flush()
            os.fsync(file.fileno())

        # ----------------------------------------------------
        # ATOMICALLY PUBLISH NEW GENERATION
        # ----------------------------------------------------

        # The previous manifest remains active if replacement fails.

        os.replace(
            manifest_temp_path,
            MANIFEST_PATH
        )

        published = True

    finally:

        # Remove unpublished temporary manifest.
        if manifest_temp_path.exists():
            manifest_temp_path.unlink()

        # Remove incomplete generation after failure.
        if not published and generation_dir.exists():
            shutil.rmtree(generation_dir)

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DOCUMENT PROCESSED SUCCESSFULLY")
    print("=" * 60)

    print(f"Document: {document_name}")
    print(f"Pages: {len(pages)}")
    print(f"Characters: {total_characters}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding dimension: {dimension}")
    print(f"Generation: {generation_id}")
    print(f"Manifest: {MANIFEST_PATH}")

    print("=" * 60)

    return {
        "success": True,
        "message": "Document processed successfully.",
        "filename": document_name,
        "pages": len(pages),
        "characters": total_characters,
        "chunks": len(chunks),
        "dimension": dimension,
        "generation_id": generation_id,
    }


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    pdf_path = BASE_DIR / "sample.pdf"

    result = process_document(str(pdf_path))

    print("\nRESULT:")
    print(result)
