
import json
import pickle

import faiss
import numpy as np
import pytest

import rag_engine


@pytest.fixture
def vectorstore_environment(tmp_path, monkeypatch):
    """Create an isolated vectorstore environment for each test."""

    vectorstore_dir = tmp_path / "vectorstore"
    generations_dir = vectorstore_dir / "generations"
    generations_dir.mkdir(parents=True)

    manifest_path = vectorstore_dir / "manifest.json"

    monkeypatch.setattr(rag_engine, "VECTORSTORE_DIR", vectorstore_dir)
    monkeypatch.setattr(rag_engine, "MANIFEST_PATH", manifest_path)

    rag_engine._clear_vectorstore()

    yield vectorstore_dir, generations_dir, manifest_path

    rag_engine._clear_vectorstore()


class FakeEmbeddingModel:
    """Return predictable embeddings for retrieval tests."""

    def encode(self, texts, normalize_embeddings=True):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def create_chunks():
    """Create valid page-aware chunk records."""

    return [
        {
            "chunk_id": 0,
            "text": "First document chunk",
            "page_number": 1,
            "document_name": "sample.pdf",
        },
        {
            "chunk_id": 1,
            "text": "Second document chunk",
            "page_number": 2,
            "document_name": "sample.pdf",
        },
    ]


def create_generation(
    generations_dir,
    generation_id="generation-1",
    chunks=None,
    embeddings=None,
    page_count=2,
):
    """Create a valid FAISS generation and return its manifest."""

    if chunks is None:
        chunks = create_chunks()

    if embeddings is None:
        embeddings = np.array(
            [[1.0, 0.0] for _ in chunks],
            dtype=np.float32,
        )

    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have exactly one embedding.")

    generation_dir = generations_dir / generation_id
    generation_dir.mkdir(parents=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path = generation_dir / "index.faiss"
    chunks_path = generation_dir / "chunks.pkl"

    faiss.write_index(index, str(index_path))

    with open(chunks_path, "wb") as file:
        pickle.dump(chunks, file)

    return {
        "generation_id": generation_id,
        "index_path": f"generations/{generation_id}/index.faiss",
        "chunks_path": f"generations/{generation_id}/chunks.pkl",
        "chunk_count": len(chunks),
        "embedding_dimension": embeddings.shape[1],
        "document_name": "sample.pdf",
        "page_count": page_count,
        "chunk_format_version": 2,
    }


def write_manifest(manifest_path, manifest):
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file)


# ============================================================
# VECTORSTORE LOADING TESTS
# ============================================================

def test_load_vectorstore_success(vectorstore_environment):
    _, generations_dir, manifest_path = vectorstore_environment

    expected_chunks = create_chunks()

    manifest = create_generation(
        generations_dir,
        chunks=expected_chunks,
    )

    write_manifest(manifest_path, manifest)

    assert rag_engine.load_vectorstore(force_reload=True) is True

    assert rag_engine.index is not None
    assert rag_engine.index.ntotal == 2
    assert rag_engine.index.d == 2
    assert rag_engine.chunks == expected_chunks


def test_load_vectorstore_rejects_invalid_manifest(
    vectorstore_environment,
):
    _, _, manifest_path = vectorstore_environment

    manifest_path.write_text(
        '{"generation_id": "generation-1"}',
        encoding="utf-8",
    )

    assert rag_engine.load_vectorstore(force_reload=True) is False
    assert rag_engine.index is None
    assert rag_engine.chunks == []


def test_manifest_path_traversal_is_rejected(
    vectorstore_environment,
):
    _, _, manifest_path = vectorstore_environment

    manifest = {
        "generation_id": "generation-1",
        "index_path": "../../outside/index.faiss",
        "chunks_path": "generations/generation-1/chunks.pkl",
        "chunk_count": 1,
        "embedding_dimension": 2,
        "document_name": "sample.pdf",
        "page_count": 1,
        "chunk_format_version": 2,
    }

    write_manifest(manifest_path, manifest)

    assert rag_engine.get_vectorstore_signature() is None
    assert rag_engine.load_vectorstore(force_reload=True) is False


def test_load_vectorstore_rejects_chunk_count_mismatch(
    vectorstore_environment,
):
    _, generations_dir, manifest_path = vectorstore_environment

    manifest = create_generation(generations_dir)
    manifest["chunk_count"] = 10

    write_manifest(manifest_path, manifest)

    assert rag_engine.load_vectorstore(force_reload=True) is False
    assert rag_engine.index is None
    assert rag_engine.chunks == []


def test_load_vectorstore_rejects_embedding_dimension_mismatch(
    vectorstore_environment,
):
    _, generations_dir, manifest_path = vectorstore_environment

    manifest = create_generation(generations_dir)
    manifest["embedding_dimension"] = 5

    write_manifest(manifest_path, manifest)

    assert rag_engine.load_vectorstore(force_reload=True) is False
    assert rag_engine.index is None


def test_load_vectorstore_rejects_unsupported_chunk_format(
    vectorstore_environment,
):
    _, generations_dir, manifest_path = vectorstore_environment

    manifest = create_generation(generations_dir)
    manifest["chunk_format_version"] = 1

    write_manifest(manifest_path, manifest)

    assert rag_engine.load_vectorstore(force_reload=True) is False
    assert rag_engine.index is None
    assert rag_engine.chunks == []


def test_load_vectorstore_rejects_invalid_chunk_metadata(
    vectorstore_environment,
):
    _, generations_dir, manifest_path = vectorstore_environment

    invalid_chunks = create_chunks()
    invalid_chunks[0]["page_number"] = 0

    manifest = create_generation(
        generations_dir,
        chunks=invalid_chunks,
    )

    write_manifest(manifest_path, manifest)

    assert rag_engine.load_vectorstore(force_reload=True) is False
    assert rag_engine.index is None
    assert rag_engine.chunks == []


def test_load_vectorstore_rejects_document_name_mismatch(
    vectorstore_environment,
):
    _, generations_dir, manifest_path = vectorstore_environment

    invalid_chunks = create_chunks()
    invalid_chunks[0]["document_name"] = "another.pdf"

    manifest = create_generation(
        generations_dir,
        chunks=invalid_chunks,
    )

    write_manifest(manifest_path, manifest)

    assert rag_engine.load_vectorstore(force_reload=True) is False
    assert rag_engine.index is None


def test_load_vectorstore_preserves_physical_page_numbers(
    vectorstore_environment,
):
    _, generations_dir, manifest_path = vectorstore_environment

    # Page 2 is blank and therefore has no extracted chunk.
    # Page 3 must retain its original physical PDF page number.
    page_chunks = [
        {
            "chunk_id": 0,
            "text": "Content from page one",
            "page_number": 1,
            "document_name": "sample.pdf",
        },
        {
            "chunk_id": 1,
            "text": "Content from page three",
            "page_number": 3,
            "document_name": "sample.pdf",
        },
    ]

    manifest = create_generation(
        generations_dir,
        chunks=page_chunks,
        page_count=2,
    )

    write_manifest(manifest_path, manifest)

    assert rag_engine.load_vectorstore(force_reload=True) is True
    assert [chunk["page_number"] for chunk in rag_engine.chunks] == [1, 3]


# ============================================================
# RETRIEVAL TESTS
# ============================================================

def test_retrieve_context_returns_relevant_chunks(
    vectorstore_environment,
    monkeypatch,
):
    _, generations_dir, manifest_path = vectorstore_environment

    expected_chunks = create_chunks()

    manifest = create_generation(
        generations_dir,
        chunks=expected_chunks,
    )

    write_manifest(manifest_path, manifest)

    monkeypatch.setattr(
        rag_engine,
        "embedding_model",
        FakeEmbeddingModel(),
    )

    results = rag_engine.retrieve_context("First document", top_k=1)

    assert results == [expected_chunks[0]]
    assert results[0]["page_number"] == 1
    assert results[0]["document_name"] == "sample.pdf"


def test_retrieve_context_respects_top_k(
    vectorstore_environment,
    monkeypatch,
):
    _, generations_dir, manifest_path = vectorstore_environment

    manifest = create_generation(generations_dir)
    write_manifest(manifest_path, manifest)

    monkeypatch.setattr(
        rag_engine,
        "embedding_model",
        FakeEmbeddingModel(),
    )

    results = rag_engine.retrieve_context("document", top_k=1)

    assert len(results) == 1
    assert isinstance(results[0], dict)


def test_retrieve_context_reloads_new_generation(
    vectorstore_environment,
    monkeypatch,
):
    _, generations_dir, manifest_path = vectorstore_environment

    first_chunks = [
        {
            "chunk_id": 0,
            "text": "Old document chunk",
            "page_number": 1,
            "document_name": "sample.pdf",
        }
    ]

    first_manifest = create_generation(
        generations_dir,
        generation_id="generation-1",
        chunks=first_chunks,
        page_count=1,
    )

    write_manifest(manifest_path, first_manifest)

    monkeypatch.setattr(
        rag_engine,
        "embedding_model",
        FakeEmbeddingModel(),
    )

    assert rag_engine.load_vectorstore(force_reload=True) is True
    assert rag_engine.chunks == first_chunks

    second_chunks = [
        {
            "chunk_id": 0,
            "text": "New document chunk",
            "page_number": 1,
            "document_name": "sample.pdf",
        }
    ]

    second_manifest = create_generation(
        generations_dir,
        generation_id="generation-2",
        chunks=second_chunks,
        page_count=1,
    )

    write_manifest(manifest_path, second_manifest)

    results = rag_engine.retrieve_context("document", top_k=1)

    assert results == second_chunks
    assert rag_engine.chunks == second_chunks


def test_retrieve_context_rejects_invalid_query(
    vectorstore_environment,
):
    assert rag_engine.retrieve_context("") == []
    assert rag_engine.retrieve_context("   ") == []
    assert rag_engine.retrieve_context(None) == []


def test_retrieve_context_rejects_invalid_top_k(
    vectorstore_environment,
):
    assert rag_engine.retrieve_context("question", top_k=0) == []
    assert rag_engine.retrieve_context("question", top_k=-1) == []
    assert rag_engine.retrieve_context("question", top_k=True) == []
