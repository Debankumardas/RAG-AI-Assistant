
import json
import pickle

import faiss
import numpy as np
import pytest

import rag_engine


@pytest.fixture
def vectorstore_environment(tmp_path):
    """Create an isolated vectorstore environment for each test."""

    vectorstore_dir = tmp_path / "vectorstore"
    generations_dir = vectorstore_dir / "generations"
    generations_dir.mkdir(parents=True)

    manifest_path = vectorstore_dir / "manifest.json"

    return vectorstore_dir, generations_dir, manifest_path


class FakeEmbeddingModel:
    """Return predictable embeddings for retrieval tests."""

    def encode(self, texts, normalize_embeddings=True, **kwargs):
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
    vectorstore_dir, generations_dir, manifest_path = vectorstore_environment

    expected_chunks = create_chunks()

    manifest = create_generation(
        generations_dir,
        chunks=expected_chunks,
    )

    write_manifest(manifest_path, manifest)

    result = rag_engine.load_vectorstore(
        vectorstore_dir=vectorstore_dir
    )

    assert result is not None

    loaded_index, loaded_chunks = result

    assert loaded_index is not None
    assert loaded_index.ntotal == 2
    assert loaded_index.d == 2
    assert loaded_chunks == expected_chunks


def test_load_vectorstore_rejects_invalid_manifest(
    vectorstore_environment,
):
    vectorstore_dir, _, manifest_path = vectorstore_environment

    manifest_path.write_text(
        '{"generation_id": "generation-1"}',
        encoding="utf-8",
    )

    result = rag_engine.load_vectorstore(
        vectorstore_dir=vectorstore_dir
    )

    assert result is None


def test_manifest_path_traversal_is_rejected(
    vectorstore_environment,
):
    vectorstore_dir, _, manifest_path = vectorstore_environment

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

    assert (
        rag_engine.get_vectorstore_signature(vectorstore_dir)
        is None
    )

    assert (
        rag_engine.load_vectorstore(
            vectorstore_dir=vectorstore_dir
        )
        is None
    )


def test_load_vectorstore_rejects_chunk_count_mismatch(
    vectorstore_environment,
):
    vectorstore_dir, generations_dir, manifest_path = vectorstore_environment

    manifest = create_generation(generations_dir)
    manifest["chunk_count"] = 10

    write_manifest(manifest_path, manifest)

    result = rag_engine.load_vectorstore(
        vectorstore_dir=vectorstore_dir
    )

    assert result is None


def test_load_vectorstore_rejects_embedding_dimension_mismatch(
    vectorstore_environment,
):
    vectorstore_dir, generations_dir, manifest_path = vectorstore_environment

    manifest = create_generation(generations_dir)
    manifest["embedding_dimension"] = 5

    write_manifest(manifest_path, manifest)

    result = rag_engine.load_vectorstore(
        vectorstore_dir=vectorstore_dir
    )

    assert result is None


def test_load_vectorstore_rejects_unsupported_chunk_format(
    vectorstore_environment,
):
    vectorstore_dir, generations_dir, manifest_path = vectorstore_environment

    manifest = create_generation(generations_dir)
    manifest["chunk_format_version"] = 1

    write_manifest(manifest_path, manifest)

    result = rag_engine.load_vectorstore(
        vectorstore_dir=vectorstore_dir
    )

    assert result is None


def test_load_vectorstore_rejects_invalid_chunk_metadata(
    vectorstore_environment,
):
    vectorstore_dir, generations_dir, manifest_path = vectorstore_environment

    invalid_chunks = create_chunks()
    invalid_chunks[0]["page_number"] = 0

    manifest = create_generation(
        generations_dir,
        chunks=invalid_chunks,
    )

    write_manifest(manifest_path, manifest)

    result = rag_engine.load_vectorstore(
        vectorstore_dir=vectorstore_dir
    )

    assert result is None


def test_load_vectorstore_rejects_document_name_mismatch(
    vectorstore_environment,
):
    vectorstore_dir, generations_dir, manifest_path = vectorstore_environment

    invalid_chunks = create_chunks()
    invalid_chunks[0]["document_name"] = "another.pdf"

    manifest = create_generation(
        generations_dir,
        chunks=invalid_chunks,
    )

    write_manifest(manifest_path, manifest)

    result = rag_engine.load_vectorstore(
        vectorstore_dir=vectorstore_dir
    )

    assert result is None


def test_load_vectorstore_preserves_physical_page_numbers(
    vectorstore_environment,
):
    vectorstore_dir, generations_dir, manifest_path = vectorstore_environment

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

    result = rag_engine.load_vectorstore(
        vectorstore_dir=vectorstore_dir
    )

    assert result is not None

    _, loaded_chunks = result

    assert [chunk["page_number"] for chunk in loaded_chunks] == [1, 3]


# ============================================================
# RETRIEVAL TESTS
# ============================================================

def test_retrieve_context_returns_relevant_chunks(
    vectorstore_environment,
    monkeypatch,
):
    vectorstore_dir, generations_dir, manifest_path = vectorstore_environment

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

    results = rag_engine.retrieve_context(
        "First document",
        top_k=1,
        vectorstore_dir=vectorstore_dir,
    )

    assert results == [expected_chunks[0]]
    assert results[0]["page_number"] == 1
    assert results[0]["document_name"] == "sample.pdf"


def test_retrieve_context_respects_top_k(
    vectorstore_environment,
    monkeypatch,
):
    vectorstore_dir, generations_dir, manifest_path = vectorstore_environment

    manifest = create_generation(generations_dir)
    write_manifest(manifest_path, manifest)

    monkeypatch.setattr(
        rag_engine,
        "embedding_model",
        FakeEmbeddingModel(),
    )

    results = rag_engine.retrieve_context(
        "document",
        top_k=1,
        vectorstore_dir=vectorstore_dir,
    )

    assert len(results) == 1
    assert isinstance(results[0], dict)


def test_retrieve_context_reloads_new_generation(
    vectorstore_environment,
    monkeypatch,
):
    vectorstore_dir, generations_dir, manifest_path = vectorstore_environment

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

    first_results = rag_engine.retrieve_context(
        "document",
        top_k=1,
        vectorstore_dir=vectorstore_dir,
    )

    assert first_results == first_chunks

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

    results = rag_engine.retrieve_context(
        "document",
        top_k=1,
        vectorstore_dir=vectorstore_dir,
    )

    assert results == second_chunks



def test_retrieve_context_rejects_invalid_query(
    vectorstore_environment,
):
    vectorstore_dir, _, _ = vectorstore_environment

    with pytest.raises(ValueError, match="non-empty string"):
        rag_engine.retrieve_context(
            "", vectorstore_dir=vectorstore_dir
        )

    with pytest.raises(ValueError, match="non-empty string"):
        rag_engine.retrieve_context(
            "   ", vectorstore_dir=vectorstore_dir
        )

    with pytest.raises(ValueError, match="non-empty string"):
        rag_engine.retrieve_context(
            None, vectorstore_dir=vectorstore_dir
        )


def test_retrieve_context_rejects_invalid_top_k(
    vectorstore_environment,
):
    vectorstore_dir, _, _ = vectorstore_environment

    for invalid_top_k in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            rag_engine.retrieve_context(
                "question",
                top_k=invalid_top_k,
                vectorstore_dir=vectorstore_dir,
            )


def test_retrieve_context_isolates_separate_vectorstores(
    vectorstore_environment,
    tmp_path,
    monkeypatch,
):
    first_dir, first_generations, first_manifest_path = (
        vectorstore_environment
    )

    second_dir = tmp_path / "second_vectorstore"
    second_generations = second_dir / "generations"
    second_generations.mkdir(parents=True)
    second_manifest_path = second_dir / "manifest.json"

    first_chunks = [
        {
            "chunk_id": 0,
            "text": "Confidential content belonging to session A",
            "page_number": 1,
            "document_name": "session_a.pdf",
        }
    ]

    second_chunks = [
        {
            "chunk_id": 0,
            "text": "Private content belonging to session B",
            "page_number": 1,
            "document_name": "session_b.pdf",
        }
    ]

    first_manifest = create_generation(
        first_generations,
        chunks=first_chunks,
        page_count=1,
    )
    first_manifest["document_name"] = "session_a.pdf"
    write_manifest(first_manifest_path, first_manifest)

    second_manifest = create_generation(
        second_generations,
        chunks=second_chunks,
        page_count=1,
    )
    second_manifest["document_name"] = "session_b.pdf"
    write_manifest(second_manifest_path, second_manifest)

    monkeypatch.setattr(
        rag_engine,
        "embedding_model",
        FakeEmbeddingModel(),
    )

    # Retrieve from session A.
    first_results = rag_engine.retrieve_context(
        "private content",
        top_k=1,
        vectorstore_dir=first_dir,
    )

    # Retrieve from session B.
    second_results = rag_engine.retrieve_context(
        "private content",
        top_k=1,
        vectorstore_dir=second_dir,
    )

    # Retrieve from session A again after accessing session B.
    first_results_again = rag_engine.retrieve_context(
        "private content",
        top_k=1,
        vectorstore_dir=first_dir,
    )

    assert first_results == first_chunks
    assert second_results == second_chunks
    assert first_results_again == first_chunks

    assert first_results[0]["document_name"] != second_results[0]["document_name"]

def test_ask_rag_uses_session_vectorstore(monkeypatch, tmp_path):
    expected_chunks = [
        {
            "chunk_id": 0,
            "text": "Session-specific content",
            "page_number": 1,
            "document_name": "session.pdf",
        }
    ]

    calls = {}

    def fake_retrieve_context(query, top_k, vectorstore_dir):
        calls["query"] = query
        calls["top_k"] = top_k
        calls["vectorstore_dir"] = vectorstore_dir
        return expected_chunks

    class FakeResponse:
        text = "The answer is 42. [Source: session.pdf, PDF page 1]"

    class FakeClient:
        def __init__(self):
            self.models = self

        def generate_content(self, **kwargs):
            return FakeResponse()

    vectorstore_dir = tmp_path / "session_a"

    monkeypatch.setattr(
        rag_engine,
        "retrieve_context",
        fake_retrieve_context,
    )

    monkeypatch.setattr(
        rag_engine,
        "_get_gemini_client",
        lambda: FakeClient(),
    )

    answer = rag_engine.ask_rag(
        "  What is the answer?  ",
        vectorstore_dir=vectorstore_dir,
    )

    assert "42" in answer
    assert calls["query"] == "What is the answer?"
    assert calls["top_k"] == 3
    assert calls["vectorstore_dir"] == vectorstore_dir