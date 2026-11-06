
import re

from doubtdesk.ingestion.schemas import PageRecord, ChunkRecord


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    boundaries = (
        "\n\n",
        ". ",
        "? ",
        "! ",
        "\n",
        "; ",
        " ",
    )

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            lower = start + max(
                1,
                int(chunk_size * 0.65),
            )

            if lower < end:
                for separator in boundaries:
                    position = text.rfind(
                        separator,
                        lower,
                        end,
                    )

                    if position >= lower:
                        end = position + len(separator)
                        break

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        next_start = max(
            start + 1,
            end - overlap,
        )

        if (
            next_start > 0
            and not text[next_start - 1].isspace()
        ):
            limit = min(
                next_start + 32,
                end - 1,
            )

            if limit > next_start:
                match = re.search(
                    r"\s+",
                    text[next_start:limit],
                )

                if match:
                    next_start += match.end()

        start = next_start

    return chunks


def chunk_document(
    pages: list[PageRecord],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[ChunkRecord]:
    all_chunks = []

    for page in pages:
        text_chunks = chunk_text(
            page.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, text in enumerate(text_chunks):
            all_chunks.append(
                ChunkRecord(
                    doc_id=page.doc_id,
                    page_number=page.page_number,
                    chunk_index=index,
                    text=text,
                )
            )

    return all_chunks
