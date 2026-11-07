import os
import pickle

import faiss
from sentence_transformers import SentenceTransformer

from document_processor import extract_text_from_pdf, create_chunks


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTORSTORE_DIR = os.path.join(
    BASE_DIR,
    "vectorstore"
)

INDEX_PATH = os.path.join(
    VECTORSTORE_DIR,
    "index.faiss"
)

CHUNKS_PATH = os.path.join(
    VECTORSTORE_DIR,
    "chunks.pkl"
)

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

    # Convert to absolute path
    pdf_path = os.path.abspath(pdf_path)

    if not os.path.exists(pdf_path):
        return {
            "success": False,
            "message": f"PDF not found: {pdf_path}"
        }

    print("\n" + "=" * 60)
    print("DOCUMENT PROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1 — Extract text
    # --------------------------------------------------------

    print("\n[1/4] Extracting PDF text...")

    text = extract_text_from_pdf(pdf_path)

    print(f"Extracted characters: {len(text)}")

    if not text.strip():
        return {
            "success": False,
            "message": "No text could be extracted from the PDF."
        }

    # --------------------------------------------------------
    # STEP 2 — Create chunks
    # --------------------------------------------------------

    print("\n[2/4] Creating chunks...")

    chunks = create_chunks(
        text,
        chunk_size=500,
        overlap=100
    )

    print(f"Created chunks: {len(chunks)}")

    if not chunks:
        return {
            "success": False,
            "message": "No chunks were created from the document."
        }

    # --------------------------------------------------------
    # STEP 3 — Generate embeddings
    # --------------------------------------------------------

    print("\n[3/4] Generating embeddings...")

    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(f"Embedding shape: {embeddings.shape}")

    # --------------------------------------------------------
    # STEP 4 — Create FAISS index
    # --------------------------------------------------------

    print("\n[4/4] Creating FAISS index...")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # --------------------------------------------------------
    # Create vectorstore directory
    # --------------------------------------------------------

    os.makedirs(
        VECTORSTORE_DIR,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Save vectorstore safely
    # --------------------------------------------------------

    os.makedirs(
        VECTORSTORE_DIR,
        exist_ok=True
    )

    temp_index_path = INDEX_PATH + ".tmp"
    temp_chunks_path = CHUNKS_PATH + ".tmp"

    try:

        # Write both temporary files first.
        faiss.write_index(
            index,
            temp_index_path
        )

        with open(temp_chunks_path, "wb") as f:
            pickle.dump(chunks, f)

        # Replace the existing files only after both writes succeed.
        os.replace(
            temp_index_path,
            INDEX_PATH
        )

        os.replace(
            temp_chunks_path,
            CHUNKS_PATH
        )

    finally:

        # Clean up temporary files if anything failed.
        for temp_path in (
            temp_index_path,
            temp_chunks_path
        ):

            if os.path.exists(temp_path):
                os.remove(temp_path)


    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DOCUMENT PROCESSED SUCCESSFULLY")
    print("=" * 60)

    print(f"Document: {os.path.basename(pdf_path)}")
    print(f"Characters: {len(text)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding dimension: {dimension}")

    print(f"\nFAISS index: {INDEX_PATH}")
    print(f"Chunks file: {CHUNKS_PATH}")

    print("=" * 60)

    return {
        "success": True,
        "message": "Document processed successfully.",
        "filename": os.path.basename(pdf_path),
        "characters": len(text),
        "chunks": len(chunks),
        "dimension": dimension
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    pdf_path = os.path.join(
        BASE_DIR,
        "sample.pdf"
    )

    result = process_document(pdf_path)

    print("\nRESULT:")
    print(result)