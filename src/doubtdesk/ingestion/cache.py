from pathlib import Path
from chromadb.api.models.Collection import Collection

def cached_artifacts_exist(
    doc_id: str,
    page_count: int,
    collection: Collection,
) -> bool:
    if page_count <= 0:
        return False

    image_dir = Path("data/page_images") / doc_id

    for page_number in range(1, page_count + 1):
        image_path = image_dir / f"page_{page_number:03}.png"

        if not image_path.is_file():
            return False

    records = collection.get(
        where={"doc_id": doc_id},
        include=[],
    )

    return len(records["ids"]) > 0
