from pathlib import Path
from collections import Counter

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

    selected = []
    seen = set()

    for page in pages:
        key = (page["doc_id"], page["page_number"])

        if key not in seen:
            selected.append(page)
            seen.add(key)

        if len(selected) >= max_pages:
            return selected

    neighbor_counts = Counter()

    for page in pages:
        doc_id = page["doc_id"]
        page_number = page["page_number"]
        total_pages = page_counts.get(doc_id, 0)

        for offset in range(-radius, radius + 1):
            neighbor_number = page_number + offset
            key = (doc_id, neighbor_number)

            if (
                1 <= neighbor_number <= total_pages
                and key not in seen
            ):
                neighbor_counts[key] += 1

    ranked_neighbors = sorted(
        neighbor_counts,
        key=lambda key: -neighbor_counts[key],
    )

    for doc_id, page_number in ranked_neighbors:
        if len(selected) >= max_pages:
            break

        selected.append({
            "doc_id": doc_id,
            "page_number": page_number,
        })

    return selected


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
