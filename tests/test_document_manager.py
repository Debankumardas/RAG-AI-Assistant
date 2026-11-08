
import json
import pickle
from unittest.mock import patch

import faiss
import numpy as np
import pytest


# Prevent loading the real embedding model during test import.
with patch("sentence_transformers.SentenceTransformer"):
    import document_manager


def create_old_vectorstore(vectorstore_dir):
    """Create an existing generation and manifest."""

    generations_dir = vectorstore_dir / "generations"
    old_generation_dir = generations_dir / "old-generation"

    old_generation_dir.mkdir(parents=True)

    old_index_path = old_generation_dir / "index.faiss"
    old_chunks_path = old_generation_dir / "chunks.pkl"

    original_index = b"existing-index-data"
    original_chunks = b"existing-chunks-data"

    old_index_path.write_bytes(original_index)
    old_chunks_path.write_bytes(original_chunks)

    manifest_path = vectorstore_dir / "manifest.json"

    original_manifest = {
        "generation_id": "old-generation",
        "index_path": "generations/old-generation/index.faiss",
        "chunks_path": "generations/old-generation/chunks.pkl",
        "chunk_count": 1,
        "embedding_dimension": 3,
    }

    manifest_path.write_text(
        json.dumps(original_manifest),
        encoding="utf-8",
    )

    return {
        "generations_dir": generations_dir,
        "generation_dir": old_generation_dir,
        "index_path": old_index_path,
        "chunks_path": old_chunks_path,
        "manifest_path": manifest_path,
        "original_index": original_index,
        "original_chunks": original_chunks,
        "original_manifest": original_manifest,
    }


def configure_document_processing(monkeypatch, fake_embedding_model):
    """Configure page-aware document processing dependencies."""

    pages = [
        {
            "page_number": 1,
            "text": "Valid document text",
        }
    ]

    chunks = [
        {
            "chunk_id": 0,
            "text": "Valid chunk",
            "page_number": 1,
            "document_name": "sample.pdf",
        }
    ]

    monkeypatch.setattr(
        document_manager,
        "extract_pages_from_pdf",
        lambda path: pages,
    )

    monkeypatch.setattr(
        document_manager,
        "create_chunks",
        lambda pages, chunk_size, overlap, document_name: chunks,
    )

    monkeypatch.setattr(
        document_manager,
        "embedding_model",
        fake_embedding_model,
    )


def test_process_document_rejects_missing_pdf(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    result = document_manager.process_document(str(missing_pdf))

    assert result["success"] is False
    assert "PDF not found" in result["message"]


def test_process_document_rejects_empty_pdf_text(
    tmp_path, monkeypatch
):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"fake pdf")

    monkeypatch.setattr(
        document_manager,
        "extract_pages_from_pdf",
        lambda path: [],
    )

    result = document_manager.process_document(str(pdf_path))

    assert result["success"] is False
    assert "No text" in result["message"]


def test_process_document_saves_versioned_vectorstore(
    tmp_path, monkeypatch
):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"fake pdf")

    vectorstore_dir = tmp_path / "vectorstore"
    generations_dir = vectorstore_dir / "generations"
    manifest_path = vectorstore_dir / "manifest.json"

    expected_pages = [
        {
            "page_number": 1,
            "text": "First page content",
        },
        {
            "page_number": 2,
            "text": "Second page content",
        },
    ]

    expected_chunks = [
        {
            "chunk_id": 0,
            "text": "First test chunk",
            "page_number": 1,
            "document_name": "sample.pdf",
        },
        {
            "chunk_id": 1,
            "text": "Second test chunk",
            "page_number": 2,
            "document_name": "sample.pdf",
        },
    ]

    expected_chunk_texts = [
        chunk["text"] for chunk in expected_chunks
    ]

    expected_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    class FakeEmbeddingModel:
        def encode(self, texts, **kwargs):
            assert texts == expected_chunk_texts
            assert kwargs["normalize_embeddings"] is True

            return expected_embeddings

    monkeypatch.setattr(
        document_manager,
        "VECTORSTORE_DIR",
        vectorstore_dir,
    )

    monkeypatch.setattr(
        document_manager,
        "GENERATIONS_DIR",
        generations_dir,
    )

    monkeypatch.setattr(
        document_manager,
        "MANIFEST_PATH",
        manifest_path,
    )

    monkeypatch.setattr(
        document_manager,
        "extract_pages_from_pdf",
        lambda path: expected_pages,
    )

    monkeypatch.setattr(
        document_manager,
        "create_chunks",
        lambda pages, chunk_size, overlap, document_name: expected_chunks,
    )

    monkeypatch.setattr(
        document_manager,
        "embedding_model",
        FakeEmbeddingModel(),
    )

    result = document_manager.process_document(str(pdf_path))

    assert result["success"] is True
    assert result["chunks"] == 2
    assert result["pages"] == 2
    assert result["dimension"] == 3
    assert result["characters"] == sum(
        len(page["text"]) for page in expected_pages
    )
    assert result["generation_id"]

    assert manifest_path.exists()

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["generation_id"] == result["generation_id"]
    assert manifest["chunk_count"] == 2
    assert manifest["embedding_dimension"] == 3
    assert manifest["document_name"] == "sample.pdf"
    assert manifest["page_count"] == 2
    assert manifest["chunk_format_version"] == 2

    index_path = vectorstore_dir / manifest["index_path"]
    chunks_path = vectorstore_dir / manifest["chunks_path"]

    assert index_path.exists()
    assert chunks_path.exists()

    saved_index = faiss.read_index(str(index_path))

    with open(chunks_path, "rb") as file:
        saved_chunks = pickle.load(file)

    assert saved_index.ntotal == 2
    assert saved_index.d == 3
    assert saved_chunks == expected_chunks


def test_embedding_failure_preserves_existing_vectorstore(
    tmp_path, monkeypatch
):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"fake pdf")

    vectorstore_dir = tmp_path / "vectorstore"
    existing = create_old_vectorstore(vectorstore_dir)

    class FailingEmbeddingModel:
        def encode(self, texts, **kwargs):
            raise RuntimeError("Simulated embedding failure")

    monkeypatch.setattr(
        document_manager,
        "VECTORSTORE_DIR",
        vectorstore_dir,
    )

    monkeypatch.setattr(
        document_manager,
        "GENERATIONS_DIR",
        existing["generations_dir"],
    )

    monkeypatch.setattr(
        document_manager,
        "MANIFEST_PATH",
        existing["manifest_path"],
    )

    configure_document_processing(
        monkeypatch,
        FailingEmbeddingModel(),
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated embedding failure",
    ):
        document_manager.process_document(str(pdf_path))

    actual_manifest = json.loads(
        existing["manifest_path"].read_text(encoding="utf-8")
    )

    assert actual_manifest == existing["original_manifest"]

    assert existing["index_path"].read_bytes() == existing["original_index"]
    assert existing["chunks_path"].read_bytes() == existing["original_chunks"]


def test_manifest_replacement_failure_preserves_existing_vectorstore(
    tmp_path, monkeypatch
):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"fake pdf")

    vectorstore_dir = tmp_path / "vectorstore"
    existing = create_old_vectorstore(vectorstore_dir)

    class FakeEmbeddingModel:
        def encode(self, texts, **kwargs):
            return np.array(
                [[0.1, 0.2, 0.3]],
                dtype=np.float32,
            )

    monkeypatch.setattr(
        document_manager,
        "VECTORSTORE_DIR",
        vectorstore_dir,
    )

    monkeypatch.setattr(
        document_manager,
        "GENERATIONS_DIR",
        existing["generations_dir"],
    )

    monkeypatch.setattr(
        document_manager,
        "MANIFEST_PATH",
        existing["manifest_path"],
    )

    configure_document_processing(
        monkeypatch,
        FakeEmbeddingModel(),
    )

    original_replace = document_manager.os.replace

    def failing_replace(src, dst):
        if str(dst) == str(existing["manifest_path"]):
            raise OSError("Simulated manifest replacement failure")

        original_replace(src, dst)

    monkeypatch.setattr(
        document_manager.os,
        "replace",
        failing_replace,
    )

    with pytest.raises(
        OSError,
        match="Simulated manifest replacement failure",
    ):
        document_manager.process_document(str(pdf_path))

    actual_manifest = json.loads(
        existing["manifest_path"].read_text(encoding="utf-8")
    )

    # The old generation must remain active.
    assert actual_manifest == existing["original_manifest"]

    # The old generation's files must remain untouched.
    assert existing["index_path"].read_bytes() == existing["original_index"]
    assert existing["chunks_path"].read_bytes() == existing["original_chunks"]

def test_process_document_isolates_separate_vectorstores(
    tmp_path,
    monkeypatch,
):
    first_pdf = tmp_path / "session_a.pdf"
    second_pdf = tmp_path / "session_b.pdf"

    first_pdf.write_bytes(b"fake pdf A")
    second_pdf.write_bytes(b"fake pdf B")

    first_vectorstore = tmp_path / "session_a" / "vectorstore"
    second_vectorstore = tmp_path / "session_b" / "vectorstore"

    def fake_extract_pages(pdf_path):
        if pdf_path.endswith("session_a.pdf"):
            return [
                {
                    "page_number": 1,
                    "text": "Content belonging to session A",
                }
            ]

        return [
            {
                "page_number": 1,
                "text": "Content belonging to session B",
            }
        ]

    def fake_create_chunks(
        pages,
        chunk_size,
        overlap,
        document_name,
    ):
        return [
            {
                "chunk_id": 0,
                "text": pages[0]["text"],
                "page_number": 1,
                "document_name": document_name,
            }
        ]

    class FakeEmbeddingModel:
        def encode(self, texts, **kwargs):
            return np.array(
                [[1.0, 0.0, 0.0]],
                dtype=np.float32,
            )

    monkeypatch.setattr(
        document_manager,
        "extract_pages_from_pdf",
        fake_extract_pages,
    )

    monkeypatch.setattr(
        document_manager,
        "create_chunks",
        fake_create_chunks,
    )

    monkeypatch.setattr(
        document_manager,
        "embedding_model",
        FakeEmbeddingModel(),
    )

    # Process session A's document.
    first_result = document_manager.process_document(
        str(first_pdf),
        vectorstore_dir=first_vectorstore,
    )

    assert first_result["success"] is True

    first_manifest_path = first_vectorstore / "manifest.json"
    first_manifest_before = json.loads(
        first_manifest_path.read_text(encoding="utf-8")
    )

    # Process session B's document.
    second_result = document_manager.process_document(
        str(second_pdf),
        vectorstore_dir=second_vectorstore,
    )

    assert second_result["success"] is True

    second_manifest_path = second_vectorstore / "manifest.json"

    first_manifest_after = json.loads(
        first_manifest_path.read_text(encoding="utf-8")
    )
    second_manifest = json.loads(
        second_manifest_path.read_text(encoding="utf-8")
    )

    # Each session has its own generation.
    assert (
        first_manifest_before["generation_id"]
        != second_manifest["generation_id"]
    )

    # Processing session B must not change session A's manifest.
    assert first_manifest_after == first_manifest_before

    # Verify the actual saved chunk contents.
    first_chunks_path = (
        first_vectorstore / first_manifest_after["chunks_path"]
    )
    second_chunks_path = (
        second_vectorstore / second_manifest["chunks_path"]
    )

    with open(first_chunks_path, "rb") as file:
        first_chunks = pickle.load(file)

    with open(second_chunks_path, "rb") as file:
        second_chunks = pickle.load(file)

    assert first_chunks[0]["text"] == "Content belonging to session A"
    assert second_chunks[0]["text"] == "Content belonging to session B"

    assert first_chunks[0]["document_name"] == "session_a.pdf"
    assert second_chunks[0]["document_name"] == "session_b.pdf"