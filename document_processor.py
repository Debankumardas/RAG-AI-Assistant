
import pymupdf


# ============================================================
# EXTRACT TEXT FROM PDF PAGE BY PAGE
# ============================================================

def extract_pages_from_pdf(pdf_path):
    """
    Extract text from each PDF page separately.

    Returns:
        list[dict]: Each record contains the physical PDF page
        number and extracted text.
    """

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):

            text = page.get_text()

            if text.strip():
                pages.append({
                    "page_number": page_number,
                    "text": text
                })

    return pages


# ============================================================
# CREATE PAGE-AWARE CHUNKS
# ============================================================

def create_chunks(
    pages,
    chunk_size=500,
    overlap=100,
    document_name="unknown.pdf"
):
    """
    Split PDF pages into word-based chunks.

    Guarantees:
    - Words are never split.
    - Chunks never exceed chunk_size.
    - Overlap is kept as close as possible to the requested size.
    - Chunks never cross PDF page boundaries.
    - Every chunk retains its page number and document name.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks = []

    for page in pages:

        page_number = page["page_number"]
        text = page["text"]

        words = (
            text.replace("\r\n", "\n")
            .replace("\r", "\n")
            .split()
        )

        current_words = []

        for word in words:

            if len(word) > chunk_size:
                raise ValueError(
                    f"Word exceeds chunk_size ({chunk_size} characters): {word}"
                )

            candidate = " ".join(current_words + [word])

            if len(candidate) <= chunk_size:
                current_words.append(word)
                continue

            if current_words:
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": " ".join(current_words),
                    "page_number": page_number,
                    "document_name": document_name
                })

            # Build the largest possible overlap from previous words.
            overlap_words = []

            for previous_word in reversed(current_words):

                candidate_overlap = " ".join(
                    [previous_word] + overlap_words
                )

                if len(candidate_overlap) > overlap:
                    break

                overlap_words.insert(0, previous_word)

            # Reduce overlap if it prevents the next word from fitting.
            while overlap_words:

                candidate = " ".join(overlap_words + [word])

                if len(candidate) <= chunk_size:
                    break

                overlap_words.pop(0)

            current_words = overlap_words + [word]

        if current_words:
            chunks.append({
                "chunk_id": len(chunks),
                "text": " ".join(current_words),
                "page_number": page_number,
                "document_name": document_name
            })

    return chunks


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    pdf_path = "sample.pdf"

    print("=" * 60)
    print("PDF DOCUMENT PROCESSOR")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: Extract pages
    # --------------------------------------------------------

    pages = extract_pages_from_pdf(pdf_path)

    print(f"\nExtracted pages: {len(pages)}")

    total_characters = sum(
        len(page["text"]) for page in pages
    )

    print(f"Extracted characters: {total_characters}")

    # --------------------------------------------------------
    # STEP 2: Create page-aware chunks
    # --------------------------------------------------------

    chunks = create_chunks(
        pages,
        chunk_size=500,
        overlap=100,
        document_name=pdf_path
    )

    print(f"Created chunks: {len(chunks)}")

    # --------------------------------------------------------
    # STEP 3: Display chunks
    # --------------------------------------------------------

    for chunk in chunks:

        print("\n" + "=" * 60)
        print(f"CHUNK {chunk['chunk_id']}")
        print(f"PAGE: {chunk['page_number']}")
        print(f"DOCUMENT: {chunk['document_name']}")
        print("=" * 60)

        print(chunk["text"])

    print("\n" + "=" * 60)
    print("PDF PROCESSING COMPLETED")
    print("=" * 60)
