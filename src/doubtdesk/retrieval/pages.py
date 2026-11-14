from doubtdesk.config import settings
from pathlib import Path

def get_page_image_paths(
    pages: list[dict],
    image_root: Path = settings.page_image_dir,
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
