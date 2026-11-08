
import pymupdf
import pytest

from document_processor import extract_pages_from_pdf, create_chunks


# ============================================================
# PDF EXTRACTION TESTS
# ============================================================

def test_extract_text_from_multiple_pdf_pages(tmp_path):
    pdf_path = tmp_path / "multiple_pages.pdf"

    with pymupdf.open() as doc:
        page1 = doc.new_page()
        page1.insert_text((72, 72), "First page content")

        page2 = doc.new_page()
        page2.insert_text((72, 72), "Second page content")

        doc.save(pdf_path)

    result = extract_pages_from_pdf(str(pdf_path))

    assert len(result) == 2

    assert result[0]["page_number"] == 1
    assert "First page content" in result[0]["text"]

    assert result[1]["page_number"] == 2
    assert "Second page content" in result[1]["text"]


def test_extract_text_from_empty_pdf(tmp_path):
    pdf_path = tmp_path / "empty.pdf"

    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(pdf_path)

    result = extract_pages_from_pdf(str(pdf_path))

    assert result == []


def test_extract_pages_preserves_physical_page_numbers(tmp_path):
    pdf_path = tmp_path / "blank_middle_page.pdf"

    with pymupdf.open() as doc:
        page1 = doc.new_page()
        page1.insert_text((72, 72), "First page")

        doc.new_page()  # Intentionally blank page.

        page3 = doc.new_page()
        page3.insert_text((72, 72), "Third page")

        doc.save(pdf_path)

    result = extract_pages_from_pdf(str(pdf_path))

    assert [page["page_number"] for page in result] == [1, 3]


def test_pdf_document_closes_when_extraction_fails(monkeypatch):
    class FakePage:
        def get_text(self):
            raise RuntimeError("Simulated extraction failure")

    class FakeDocument:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.closed = True

        def __iter__(self):
            return iter([FakePage()])

    fake_document = FakeDocument()

    monkeypatch.setattr(
        "document_processor.pymupdf.open",
        lambda path: fake_document,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated extraction failure",
    ):
        extract_pages_from_pdf("fake.pdf")

    assert fake_document.closed is True


# ============================================================
# CHUNK CREATION TESTS
# ============================================================

def test_create_chunks_returns_empty_list_for_empty_pages():
    assert create_chunks([]) == []


def test_create_chunks_preserves_complete_words():
    pages = [
        {
            "page_number": 1,
            "text": "one two three four five"
        }
    ]

    chunks = create_chunks(
        pages,
        chunk_size=10,
        overlap=3,
    )

    assert [chunk["text"] for chunk in chunks] == [
        "one two",
        "two three",
        "four five",
    ]

    for chunk in chunks:
        assert len(chunk["text"]) <= 10


def test_create_chunks_normalizes_whitespace():
    pages = [
        {
            "page_number": 1,
            "text": "one   two\nthree\r\nfour"
        }
    ]

    chunks = create_chunks(
        pages,
        chunk_size=100,
        overlap=10,
    )

    assert len(chunks) == 1
    assert chunks[0]["text"] == "one two three four"


def test_create_chunks_never_exceeds_chunk_size_with_overlap():
    pages = [
        {
            "page_number": 1,
            "text": "aaaaa bbbbb ccccc"
        }
    ]

    chunks = create_chunks(
        pages,
        chunk_size=10,
        overlap=5,
    )

    assert len(chunks) > 1

    assert all(
        len(chunk["text"]) <= 10
        for chunk in chunks
    )


def test_create_chunks_preserves_page_metadata():
    pages = [
        {
            "page_number": 1,
            "text": "First page content"
        },
        {
            "page_number": 2,
            "text": "Second page content"
        }
    ]

    chunks = create_chunks(
        pages,
        document_name="sample.pdf",
    )

    assert len(chunks) == 2

    assert chunks[0] == {
        "chunk_id": 0,
        "text": "First page content",
        "page_number": 1,
        "document_name": "sample.pdf",
    }

    assert chunks[1] == {
        "chunk_id": 1,
        "text": "Second page content",
        "page_number": 2,
        "document_name": "sample.pdf",
    }


def test_create_chunks_never_crosses_page_boundaries():
    pages = [
        {
            "page_number": 1,
            "text": "Content from first page"
        },
        {
            "page_number": 2,
            "text": "Content from second page"
        }
    ]

    chunks = create_chunks(pages)

    assert len(chunks) == 2

    assert all(
        chunk["page_number"] == 1
        for chunk in chunks
        if "first page" in chunk["text"]
    )

    assert all(
        chunk["page_number"] == 2
        for chunk in chunks
        if "second page" in chunk["text"]
    )


def test_create_chunks_assigns_sequential_chunk_ids():
    pages = [
        {
            "page_number": 1,
            "text": "First page content"
        },
        {
            "page_number": 2,
            "text": "Second page content"
        }
    ]

    chunks = create_chunks(pages)

    assert [
        chunk["chunk_id"] for chunk in chunks
    ] == list(range(len(chunks)))


def test_create_chunks_rejects_zero_or_negative_chunk_size():
    pages = [{"page_number": 1, "text": "some text"}]

    with pytest.raises(ValueError):
        create_chunks(pages, chunk_size=0)

    with pytest.raises(ValueError):
        create_chunks(pages, chunk_size=-1)


def test_create_chunks_rejects_invalid_overlap():
    pages = [{"page_number": 1, "text": "some text"}]

    with pytest.raises(ValueError):
        create_chunks(pages, chunk_size=10, overlap=-1)

    with pytest.raises(ValueError):
        create_chunks(pages, chunk_size=10, overlap=10)

    with pytest.raises(ValueError):
        create_chunks(pages, chunk_size=10, overlap=12)


def test_create_chunks_rejects_word_larger_than_chunk_size():
    pages = [
        {
            "page_number": 1,
            "text": "extraordinary"
        }
    ]

    with pytest.raises(
        ValueError,
        match="Word exceeds chunk_size",
    ):
        create_chunks(pages, chunk_size=5, overlap=1)
