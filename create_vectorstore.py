import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from document_processor import extract_text_from_pdf, create_chunks


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "data/test.pdf"

VECTORSTORE_DIR = "vectorstore"

INDEX_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(VECTORSTORE_DIR, "chunks.pkl")


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vectorstore():

    print("=" * 60)
    print("CREATING VECTOR STORE")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Check PDF
    # --------------------------------------------------------

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    # --------------------------------------------------------
    # 2. Extract text
    # --------------------------------------------------------

    print("\n[1/4] Extracting PDF text...")

    text = extract_text_from_pdf(PDF_PATH)

    print(f"Extracted characters: {len(text)}")

    # --------------------------------------------------------
    # 3. Create chunks
    # --------------------------------------------------------

    print("\n[2/4] Creating chunks...")

    chunks = create_chunks(
        text,
        chunk_size=500,
        overlap=100
    )

    print(f"Created chunks: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks were created from the PDF.")

    # --------------------------------------------------------
    # 4. Load embedding model
    # --------------------------------------------------------

    print("\n[3/4] Loading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Embedding model loaded.")

    # --------------------------------------------------------
    # 5. Generate embeddings
    # --------------------------------------------------------

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(f"Embedding shape: {embeddings.shape}")

    # --------------------------------------------------------
    # 6. Create FAISS index
    # --------------------------------------------------------

    print("\n[4/4] Creating FAISS index...")

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # --------------------------------------------------------
    # 7. Create vectorstore directory
    # --------------------------------------------------------

    os.makedirs(
        VECTORSTORE_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 8. Save FAISS index
    # --------------------------------------------------------

    faiss.write_index(
        index,
        INDEX_PATH
    )

    # --------------------------------------------------------
    # 9. Save chunks
    # --------------------------------------------------------

    with open(
        CHUNKS_PATH,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    # --------------------------------------------------------
    # 10. Final information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VECTOR STORE CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"Chunks: {len(chunks)}")
    print(f"Embedding dimension: {dimension}")
    print(f"FAISS index: {INDEX_PATH}")
    print(f"Chunks file: {CHUNKS_PATH}")

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    create_vectorstore()