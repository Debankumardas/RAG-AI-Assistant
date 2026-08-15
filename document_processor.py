import pymupdf


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_text_from_pdf(pdf_path):

    document = pymupdf.open(pdf_path)

    pages = []

    for page in document:

        text = page.get_text()

        if text.strip():
            pages.append(text)

    document.close()

    return "\n".join(pages)


# ============================================================
# CREATE BETTER CHUNKS
# ============================================================

def create_chunks(text, chunk_size=500, overlap=100):
    """
    Create chunks without cutting words in the middle.

    Chunks are built from complete words.
    Overlap is also kept at word boundaries.
    """

    # Normalize whitespace
    words = text.replace("\r\n", "\n").replace("\r", "\n").split()

    chunks = []

    current_words = []
    current_length = 0

    for word in words:

        # +1 accounts for the space between words
        new_length = current_length + len(word) + (
            1 if current_words else 0
        )

        if new_length <= chunk_size:

            current_words.append(word)
            current_length = new_length

        else:

            # Save current chunk
            if current_words:
                chunks.append(" ".join(current_words))

            # ------------------------------------------------
            # Create word-based overlap
            # ------------------------------------------------

            overlap_words = []
            overlap_length = 0

            for previous_word in reversed(current_words):

                added_length = len(previous_word) + (
                    1 if overlap_words else 0
                )

                if overlap_length + added_length > overlap:
                    break

                overlap_words.insert(
                    0,
                    previous_word
                )

                overlap_length += added_length

            # Start new chunk with overlap + current word
            current_words = overlap_words + [word]

            current_length = sum(
                len(w) for w in current_words
            ) + len(current_words) - 1

    # Add final chunk
    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    pdf_path = "sample.pdf"

    print("=" * 60)
    print("PDF DOCUMENT PROCESSOR")
    print("=" * 60)

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    text = extract_text_from_pdf(pdf_path)

    print(f"\nExtracted characters: {len(text)}")

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = create_chunks(text)

    print(f"Created chunks: {len(chunks)}")

    # --------------------------------------------------------
    # Display chunks
    # --------------------------------------------------------

    for i, chunk in enumerate(chunks):

        print("\n" + "=" * 60)
        print(f"CHUNK {i + 1}")
        print("=" * 60)

        print(chunk)

    print("\n" + "=" * 60)
    print("PDF PROCESSING COMPLETED")
    print("=" * 60)