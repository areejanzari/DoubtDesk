from pathlib import Path

def group_chunks_by_page(chunks: list[dict]) -> list[dict]:
    pages = {}

    for chunk in chunks:
        key = (chunk["doc_id"], chunk["page_number"])

        if key not in pages:
            pages[key] = {
                "doc_id": chunk["doc_id"],
                "page_number": chunk["page_number"],
                "distance": chunk["distance"],
                "chunks": [],
            }

        pages[key]["chunks"].append(chunk["text"])

        pages[key]["distance"] = min(
            pages[key]["distance"],
            chunk["distance"],
        )

    return sorted(
        pages.values(),
        key=lambda page: page["distance"],
    )


def expand_neighbor_pages(
    pages: list[dict],
    page_counts: dict[str, int],
    radius: int = 1,
    max_pages: int = 6,
) -> list[dict]:
    if radius < 0:
        raise ValueError("Radius cannot be negative")

    if max_pages <= 0:
        raise ValueError("max_pages must be positive")

    expanded = []
    seen = set()

    for page in pages:
        key = (page["doc_id"], page["page_number"])

        if key not in seen:
            seen.add(key)
            expanded.append({
                "doc_id": page["doc_id"],
                "page_number": page["page_number"],
            })

        if len(expanded) >= max_pages:
            return expanded
        
    for page in pages:
        doc_id = page["doc_id"]
        page_number = page["page_number"]
        total_pages = page_counts[doc_id]

        start = max(1, page_number - radius)
        end = min(total_pages, page_number + radius)

        for number in range(start, end + 1):
            key = (doc_id, number)

            if key in seen:
                continue

            seen.add(key)
            expanded.append({
                "doc_id": doc_id,
                "page_number": number,
            })

            if len(expanded) >= max_pages:
                return expanded

    return expanded


def get_page_image_paths(
    pages: list[dict],
    image_root: Path = Path("data/page_images"),
) -> list[dict]:
    results = []

    for page in pages:
        doc_id = page["doc_id"]
        page_number = page["page_number"]

        image_path = (
            image_root / doc_id
            / f"page_{page_number:03d}.png"
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Page image not found: {image_path}"
            )

        results.append({
            "doc_id": doc_id,
            "page_number": page_number,
            "image_path": image_path,
        })

    return results
